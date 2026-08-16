# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Synthetic data generation for group difference modeling"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from bedroc.core import RANDOM_SEED, DataContainer
from bedroc.difference.group_classifier import GroupClassifierModel
from bedroc.difference.group_difference import HierarchicalGroupDifferenceModel
from bedroc.difference.group_plotter import GroupPlotter
from bedroc.type_aliases import NpArray, NpFloat, NpInt

logger: logging.Logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generates synthetic multivariate data for two groups with configurable parameters.

    The generator is intended primarily for testing and illustrating the classification workflow
    using data with known group differences. It provides control over the total sample size, group
    proportions, feature-specific mean offsets, and within-feature noise. The generated data
    represent a simplified version of the generative model and do not currently reproduce all
    components of the hierarchical Bayesian model. In particular, the generator does not explicitly
    model hierarchical variation in feature differences, observation-specific measurement
    uncertainties, or missing observations. Consequently, it is suitable for basic model and
    workflow testing, but should not be considered a complete simulation of the fitted hierarchical
    model.

    Args:
        n_samples: Total number of samples across both groups. Defaults to ``100``.
        group_0_fraction: Fraction of samples assigned to group 0. The number of group-0 samples is
            rounded to the nearest integer, with the remaining samples assigned to group 1.
            Defaults to ``0.5``.
        n_features: Number of features per sample. Defaults to ``5``.
        feature_offsets: Optional shift to apply to the group 1 feature means relative to group 0.
            May be either a scalar (applied to every feature) or an array of shape
            ``(n_features,)`` specifying per-feature offsets. Defaults to ``1.0``.
        feature_sigma: Standard deviation of the noise (stddev) for features. May be either a
            scalar (applied to every feature) or an array of shape ``(n_features,)`` specifying
            per-feature noise. Defaults to ``0.5``.
        random_seed: Optional seed for reproducibility. Defaults to :obj:`RANDOM_SEED`.
        output_directory: Optional path to save generated data. Defaults to ``None`` (no saving).
    """

    def __init__(
        self,
        n_samples: int = 100,
        *,
        group_0_fraction: float = 0.5,
        n_features: int = 5,
        feature_offsets: ArrayLike = 1.0,
        feature_sigma: ArrayLike = 0.5,
        random_seed: int | None = RANDOM_SEED,
        output_directory: Path | None = None,
    ):
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1.")
        if not 0.0 <= group_0_fraction <= 1.0:
            raise ValueError("group_0_fraction must be between 0.0 and 1.0.")

        self.n_samples: int = n_samples
        self.group_0_fraction: float = group_0_fraction
        self.n_features: int = n_features
        self.feature_offsets: NpFloat = np.full(self.n_features, feature_offsets, dtype=float)
        self.feature_sigma: NpFloat = np.full(self.n_features, feature_sigma, dtype=float)
        self.random_seed: int | None = random_seed
        self.output_directory: Path | None = output_directory
        self._rng = np.random.default_rng(self.random_seed)

        # For Group 0, each feature gets its own true mean (center of distribution)
        self.mu_0: NpFloat = self._rng.normal(loc=0.0, scale=1.0, size=self.n_features)
        logger.debug("mu_0 = %s", self.mu_0)

        # Shift distribution of Group 1 relative to Group 0 by the specified offsets
        self.mu_1: NpFloat = self.mu_0 + self.feature_offsets
        logger.debug("mu_1 = %s", self.mu_1)

        # Internal storage for generated data
        self._X: NpFloat | None = None
        self._X_group_idx: NpInt | None = None

    @property
    def X(self) -> NpArray:
        """Observed data (n_samples, n_features)"""
        if self._X is None:
            raise ValueError("Data not yet generated. Call 'generate()' first.")

        return self._X

    @property
    def X_group_idx(self) -> NpInt:
        """Group indices (0 for group 0, 1 for group 1) corresponding to the rows of ``X``"""
        if self._X_group_idx is None:
            raise ValueError("Data not yet generated. Call 'generate()' first.")

        return self._X_group_idx

    def generate(self) -> None:
        """Generates multivariate data for 2 groups and stores internally."""

        logger.info("Generating synthetic data with random_seed=%s", self.random_seed)

        n_group_0: int = int(round(self.n_samples * self.group_0_fraction))
        n_group_1: int = self.n_samples - n_group_0

        # Generate samples
        X_0: NpFloat = self._rng.normal(
            self.mu_0, self.feature_sigma, size=(n_group_0, self.n_features)
        )
        logger.debug("X_0 = %s", X_0)
        X_1: NpFloat = self._rng.normal(
            self.mu_1, self.feature_sigma, size=(n_group_1, self.n_features)
        )
        logger.debug("X_1 = %s", X_1)

        # Store internally
        self._X = np.vstack([X_0, X_1])
        self._X_group_idx = np.hstack(
            [np.zeros(X_0.shape[0], dtype=int), np.ones(X_1.shape[0], dtype=int)]
        )

        logger.info(
            "Synthetic data generation complete. Generated %d samples "
            "(%d group 0, %d group 1) with %d features.",
            self.n_samples,
            n_group_0,
            n_group_1,
            self.n_features,
        )

    def to_data_container(self, **kwargs) -> DataContainer:
        """Converts generated data to a :class:`~bedroc.core.DataContainer` object.

        Args:
            **kwargs: Arbitrary keyword arguments passed to the :class:`~bedroc.core.DataContainer`
                constructor.

        Returns:
            DataContainer with generated data and group indices
        """
        # Index defaults to sequential integers, so we don't need to specify it explicitly
        values: pd.DataFrame = pd.DataFrame(
            self._X, columns=[f"Feature {i}" for i in range(self.n_features)]
        )
        metadata: pd.DataFrame = pd.DataFrame({"group_idx": self._X_group_idx})

        return DataContainer(values=values, metadata=metadata, **kwargs)


if __name__ == "__main__":
    # Example usage
    generator = SyntheticDataGenerator(
        n_samples=500,
        n_features=4,
        feature_offsets=0.3,
        feature_sigma=0.2,
        group_0_fraction=0.275,
    )
    generator.generate()

    output_directory = Path("synthetic_data_output")

    data = generator.to_data_container(name="synthetic")

    train, test = data.train_test_split(
        random_state=RANDOM_SEED, stratify=data.metadata["group_idx"]
    )

    # Train a hierarchical group model
    fitted_model = HierarchicalGroupDifferenceModel(
        train.name,
        train.values_std.to_numpy(),
        train.metadata["group_idx"].to_numpy(),
        feature_names=train.feature_names,
        X_sigma=train.uncertainties_std.to_numpy(),
        output_directory=output_directory,
    )
    fitted_model.run_and_plot()

    classifier: GroupClassifierModel = GroupClassifierModel(
        fitted_model,
        test.values_std.to_numpy(),
        X_sigma=test.uncertainties_std.to_numpy(),
        output_directory=output_directory,
    )

    plotter: GroupPlotter = GroupPlotter(
        classifier,
        group_idx=test.metadata["group_idx"].to_numpy(),
        output_directory=output_directory,
    )
    plotter.confusion_matrix()
    plotter.plot_group_fraction_posterior(prior_alpha=1, prior_beta=1)

    # logger.info("Generated synthetic data:\n%s", data_container)

    # print(data_container.get_dataframe())
