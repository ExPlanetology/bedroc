# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Synthetic data generation for category difference modeling"""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

from bedroc import RANDOM_SEED
from bedroc.core.data_container import DataContainer
from bedroc.core.type_aliases import NpArray, NpFloat, NpInt
from bedroc.difference import DEFAULT_CATEGORY_NAMES, DEFAULT_INFERENCE_MODEL, InferenceModel
from bedroc.difference.pipelines import run_pipeline as _run_pipeline
from bedroc.difference.utils import log_pipeline_run

logger: logging.Logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generates synthetic multivariate data for two categories with configurable parameters.

    The generator is intended primarily for testing and illustrating the classification workflow
    using data with known category differences. It provides control over the total sample size,
    category proportions, feature-specific mean offsets, and either independent within-feature
    noise or a prescribed feature covariance shared between both categories (e.g. an empirical
    covariance estimated from real data, for closer comparison). The generated data represent a
    simplified version of the generative model and do not currently reproduce all components of
    the hierarchical Bayesian model. In particular, the generator does not explicitly model
    hierarchical variation in feature differences, observation-specific measurement uncertainties,
    or missing observations. Consequently, it is suitable for basic model and workflow testing, but
    should not be considered a complete simulation of the fitted hierarchical model.

    Args:
        n_samples: Total number of samples across both categories. Defaults to ``100``.
        category_0_fraction: Fraction of samples assigned to category 0. The number of category-0
            samples is rounded to the nearest integer, with the remaining samples assigned to
            category 1. Defaults to ``0.5``.
        n_features: Number of features per sample. Defaults to ``5``.
        feature_offsets: Optional shift to apply to the category 1 feature means relative to
            category 0. May be either a scalar (applied to every feature) or an array of shape
            ``(n_features,)`` specifying per-feature offsets. Defaults to ``1.0``.
        feature_sigma: Standard deviation of the noise (stddev) for features, assuming features are
            independent. May be either a scalar (applied to every feature) or an array of shape
            ``(n_features,)`` specifying per-feature noise. Ignored if ``covariance`` is provided.
            Defaults to ``0.5``.
        covariance: Optional feature covariance matrix of shape ``(n_features, n_features)``,
            shared between both categories (matching the shared-covariance assumption of
            :class:`~bedroc.difference.models.unified_covariance.UnifiedCovarianceModel`). If
            provided, features are drawn jointly from a multivariate normal with this covariance
            instead of independently via ``feature_sigma``. Must be symmetric positive-definite.
            Defaults to ``None``.
        random_seed: Optional seed for reproducibility. Defaults to :obj:`RANDOM_SEED`.
        output_directory: Optional path to save generated data. Defaults to ``None`` (no saving).
    """

    def __init__(
        self,
        n_samples: int = 100,
        *,
        n_features: int = 5,
        feature_offsets: ArrayLike = 1.0,
        feature_sigma: ArrayLike = 0.5,
        covariance: NpArray | None = None,
        category_0_fraction: float = 0.5,
        random_seed: int | None = RANDOM_SEED,
        output_directory: Path | None = None,
    ):
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1.")
        if not 0.0 <= category_0_fraction <= 1.0:
            raise ValueError("category_0_fraction must be between 0.0 and 1.0.")

        self.n_samples: int = n_samples
        self.category_0_fraction: float = category_0_fraction
        self.n_features: int = n_features
        self.feature_offsets: NpFloat = np.full(self.n_features, feature_offsets, dtype=float)
        self.feature_sigma: NpFloat = np.full(self.n_features, feature_sigma, dtype=float)
        self.covariance: NpFloat | None = self._validate_covariance(covariance)
        self.random_seed: int | None = random_seed
        self.output_directory: Path | None = output_directory
        self._rng = np.random.default_rng(self.random_seed)

        # For Category 0, each feature gets its own true mean (center of distribution)
        self.mu_0: NpFloat = self._rng.normal(loc=0.0, scale=1.0, size=self.n_features)
        logger.debug("mu_0 = %s", self.mu_0)

        # Shift distribution of Category 1 relative to Category 0 by the specified offsets
        self.mu_1: NpFloat = self.mu_0 + self.feature_offsets
        logger.debug("mu_1 = %s", self.mu_1)

        # Internal storage for generated data
        self._X: NpFloat | None = None
        self._X_category_idx: NpInt | None = None

    def _validate_covariance(self, covariance: NpArray | None) -> NpFloat | None:
        """Validates an optional shared feature covariance matrix.

        Args:
            covariance: Candidate covariance matrix, or ``None``.

        Returns:
            Validated covariance matrix, or ``None`` if ``covariance`` is ``None``.

        Raises:
            ValueError: If ``covariance`` has the wrong shape, is not symmetric, or is not
                positive-definite.
        """
        if covariance is None:
            return None

        covariance = np.asarray(covariance, dtype=float)

        if covariance.shape != (self.n_features, self.n_features):
            raise ValueError(
                f"covariance must have shape ({self.n_features}, {self.n_features}), got "
                f"{covariance.shape}."
            )

        if not np.allclose(covariance, covariance.T):
            raise ValueError("covariance must be symmetric.")

        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as err:
            raise ValueError("covariance must be positive-definite.") from err

        return covariance

    @property
    def X(self) -> NpArray:
        """Observed data (n_samples, n_features)"""
        if self._X is None:
            raise ValueError("Data not yet generated. Call 'generate()' first.")

        return self._X

    @property
    def X_category_idx(self) -> NpInt:
        """Category indices (0 for category 0, 1 for category 1) corresponding to the rows of
        ``X``"""
        if self._X_category_idx is None:
            raise ValueError("Data not yet generated. Call 'generate()' first.")

        return self._X_category_idx

    def generate(self) -> None:
        """Generates multivariate data for 2 categories and stores internally."""

        logger.info("Generating synthetic data with random_seed=%s", self.random_seed)

        n_category_0: int = int(round(self.n_samples * self.category_0_fraction))
        n_category_1: int = self.n_samples - n_category_0

        # Generate samples. If a shared covariance is prescribed, features are drawn jointly
        # (correlated); otherwise each feature is drawn independently via feature_sigma.
        if self.covariance is not None:
            X_0: NpFloat = self._rng.multivariate_normal(
                self.mu_0, self.covariance, size=n_category_0
            )
            X_1: NpFloat = self._rng.multivariate_normal(
                self.mu_1, self.covariance, size=n_category_1
            )
        else:
            X_0 = self._rng.normal(
                self.mu_0, self.feature_sigma, size=(n_category_0, self.n_features)
            )
            X_1 = self._rng.normal(
                self.mu_1, self.feature_sigma, size=(n_category_1, self.n_features)
            )
        logger.debug("X_0 = %s", X_0)
        logger.debug("X_1 = %s", X_1)

        # Store internally
        self._X = np.vstack([X_0, X_1])
        self._X_category_idx = np.hstack(
            [np.zeros(X_0.shape[0], dtype=int), np.ones(X_1.shape[0], dtype=int)]
        )

        logger.info(
            "Synthetic data generation complete. Generated %d samples "
            "(%d category 0, %d category 1) with %d features.",
            self.n_samples,
            n_category_0,
            n_category_1,
            self.n_features,
        )

    def to_data_container(
        self, *, category_names: tuple[str, str] = DEFAULT_CATEGORY_NAMES, **kwargs
    ) -> DataContainer:
        """Converts generated data to a :class:`~bedroc.core.DataContainer` object.

        Args:
            category_names: Display names for category 0 and category 1, respectively. Must be
                given in alphabetical order, since :class:`~bedroc.core.data_container.DataContainer`
                locks its category-to-code mapping by sorting the label strings; this ensures that
                category 0 (mean :attr:`mu_0`) and category 1 (mean :attr:`mu_1`) correspond
                respectively to category codes 0 and 1. Defaults to
                :obj:`~bedroc.difference.DEFAULT_CATEGORY_NAMES`.
            **kwargs: Additional keyword arguments passed to the
                :class:`~bedroc.core.DataContainer` constructor (e.g. ``name``).

        Returns:
            DataContainer with generated data and category labels

        Raises:
            ValueError: If ``category_names`` is not given in alphabetical order
        """
        if list(category_names) != sorted(category_names):
            raise ValueError(
                f"category_names must be given in alphabetical order, got {category_names!r}."
            )

        category_column = "category"

        # Index defaults to sequential integers, so we don't need to specify it explicitly
        values: pd.DataFrame = pd.DataFrame(
            self.X,
            columns=[f"Feature {i}" for i in range(self.n_features)],  # pyright: ignore
        )
        metadata: pd.DataFrame = pd.DataFrame(
            {category_column: np.asarray(category_names)[self.X_category_idx]}
        )

        if self.output_directory is not None:
            self.output_directory.mkdir(parents=True, exist_ok=True)
            name = kwargs.get("name", "synthetic")
            values.join(metadata).to_excel(self.output_directory / f"{name}_data.xlsx")

        return DataContainer(
            values=values, metadata=metadata, category_column=category_column, **kwargs
        )


def run_pipeline(
    generator: SyntheticDataGenerator,
    *,
    inference: InferenceModel = DEFAULT_INFERENCE_MODEL,
    category_names: tuple[str, str] = DEFAULT_CATEGORY_NAMES,
    name: str = "Synthetic",
) -> None:
    """Generates synthetic data and runs the full category-comparison analysis on it.

    Args:
        generator: A configured (but not yet generated) SyntheticDataGenerator. Its
            ``random_seed`` is reused for the downstream train/test split and model inference, and
            its ``output_directory`` (if set) is reused for saving all pipeline outputs — so both
            stay consistent with how the data itself was generated.
        inference: Type of inference to run. Defaults to :obj:`DEFAULT_INFERENCE_MODEL`.
        category_names: Display names for category 0 and category 1. Must be given in alphabetical
            order (see :meth:`SyntheticDataGenerator.to_data_container`). Defaults to
            :obj:`~bedroc.difference.DEFAULT_CATEGORY_NAMES`.
        name: Name for the generated :class:`~bedroc.core.DataContainer`. Defaults to
            ``"Synthetic"``.
    """
    with log_pipeline_run(f"synthetic analysis pipeline with inference: {inference}"):
        generator.generate()
        data = generator.to_data_container(name=name, category_names=category_names)

        _run_pipeline(
            data,
            inference=inference,
            output_directory=generator.output_directory,
            random_seed=generator.random_seed,
        )


def demo_correlation_alignment(
    *,
    covariance: NpArray | None = None,
    n_features: int = 2,
    rho: float = 0.8,
    sigma: float = 1.0,
    shift_magnitude: ArrayLike = 0.6,
    sign_patterns: Mapping[str, Sequence[int]] | None = None,
    n_samples: int = 20_000,
    category_0_fraction: float = 0.5,
    random_seed: int | None = RANDOM_SEED,
) -> None:
    """Demonstrates that joint separability depends on shift *direction* relative to correlation.

    Compares candidate class mean-shift directions (each built from the same per-feature shift
    magnitudes, differing only in sign) against a shared feature covariance, for an arbitrary
    number of features. For each candidate, and for the true theoretical-optimum direction, logs
    the analytical Mahalanobis distance between the class means (``D^2 = delta^T Sigma^-1
    delta``) and the cross-validated LDA classification accuracy on freshly generated synthetic
    data — showing that sign choice alone can swing joint separability enormously despite
    identical per-feature effect sizes.

    Args:
        covariance: Optional shared feature covariance matrix, shape ``(n_features,
            n_features)``. If ``None``, a compound-symmetric matrix is built from ``rho``,
            ``sigma``, and ``n_features`` (every feature pairwise-correlated at the same ``rho``,
            equal variance ``sigma**2``) — convenient for the simple illustrative case; pass an
            arbitrary matrix (e.g. from
            :meth:`~bedroc.core.data_container.DataDiagnostics.covariance_matrix`) to explore this
            for a specific dataset's actual covariance structure. Defaults to ``None``.
        n_features: Number of features. Only used to build the default compound-symmetric
            ``covariance`` when one isn't supplied; ignored (and inferred from ``covariance``'s
            shape) otherwise. Defaults to ``2``.
        rho: Pairwise correlation used to build the default compound-symmetric ``covariance``.
            Ignored if ``covariance`` is given. Defaults to ``0.8``.
        sigma: Per-feature standard deviation used to build the default compound-symmetric
            ``covariance``. Ignored if ``covariance`` is given. Defaults to ``1.0``.
        shift_magnitude: Magnitude(s) of the per-feature mean shift between categories. May be a
            scalar (applied to every feature) or an array of shape ``(n_features,)``. Defaults to
            ``0.6``.
        sign_patterns: Optional mapping from a descriptive label to a sequence of ``+1``/``-1``
            signs (one per feature), each defining a candidate shift direction ``delta =
            shift_magnitude * signs``. If ``None``, defaults to comparing "all positive" (fully
            aligned with the correlation) against the most sign-balanced split ``n_features``
            allows (as close to an equal +/- split as possible) — the two extremes for a
            compound-symmetric covariance. Defaults to ``None``.
        n_samples: Number of samples to generate per scenario. Defaults to ``20_000``.
        category_0_fraction: Fraction of samples assigned to category 0. Defaults to ``0.5``.
        random_seed: Random seed for reproducibility. Defaults to :obj:`RANDOM_SEED`.
    """
    if covariance is None:
        covariance = sigma**2 * ((1 - rho) * np.eye(n_features) + rho * np.ones((n_features,) * 2))
    else:
        covariance = np.asarray(covariance, dtype=float)
        n_features = covariance.shape[0]

    shift_magnitude_arr: NpFloat = np.full(n_features, shift_magnitude, dtype=float)

    if sign_patterns is None:
        n_negative = n_features // 2
        balanced_signs = [1] * (n_features - n_negative) + [-1] * n_negative
        sign_patterns = {
            "all positive (fully aligned)": [1] * n_features,
            "most balanced +/- split (most opposed)": balanced_signs,
        }

    scenario_deltas: dict[str, NpFloat] = {
        label: shift_magnitude_arr * np.asarray(signs, dtype=float)
        for label, signs in sign_patterns.items()
    }

    # Reference: the true-optimal shift direction for this covariance (the eigenvector of its
    # smallest eigenvalue), rescaled to the same total shift "budget" as the sign-pattern
    # candidates above, ignoring the sign-pattern constraint entirely.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    optimal_direction = eigenvectors[:, np.argmin(eigenvalues)]
    total_budget = float(np.sum(shift_magnitude_arr**2))
    scenario_deltas["theoretical optimum (eigenvector-aligned)"] = optimal_direction * np.sqrt(
        total_budget
    )

    logger.info(
        "Correlation-alignment demo: n_features=%d, shift magnitude(s)=%s",
        n_features,
        shift_magnitude_arr,
    )

    for label, delta in scenario_deltas.items():
        effect_sizes = delta / np.sqrt(np.diag(covariance))
        mahalanobis_sq = float(delta @ np.linalg.solve(covariance, delta))

        generator = SyntheticDataGenerator(
            n_samples=n_samples,
            n_features=n_features,
            feature_offsets=delta,
            covariance=covariance,
            category_0_fraction=category_0_fraction,
            random_seed=random_seed,
        )
        generator.generate()

        lda = LinearDiscriminantAnalysis()
        accuracy = cross_val_score(lda, generator.X, generator.X_category_idx, cv=5).mean()

        logger.info(
            "%s: per-feature effect size=%s, Mahalanobis D^2=%.4f, D=%.4f, "
            "LDA cross-validated accuracy=%.4f",
            label,
            effect_sizes,
            mahalanobis_sq,
            np.sqrt(mahalanobis_sq),
            accuracy,
        )
