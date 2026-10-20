# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Kernel density estimate (KDE) mixture estimate of a population fraction.

A non-parametric comparison for the category difference models. Each labeled category's density is
a multivariate Gaussian kernel density estimate of its samples over all features together, with no
parametric (e.g. Gaussian) form assumed and correlations between features included. The densities
are held fixed, and the category-0 fraction of the target population is the single mixture weight
that best explains it,

.. math::

    \\hat\\pi_0 = \\arg\\max_{\\pi} \\sum_i \\log\\left[\\pi \\, \\hat p_0(x_i)
    + (1 - \\pi) \\, \\hat p_1(x_i)\\right],

found by expectation-maximization (see :func:`~bedroc.difference.utils.kde_mixture_weight`).
Because the densities come from the labeled data alone, the target population cannot pull them
towards itself (unlike the joint semi-supervised Bayesian models).

Kernel density estimates need rapidly more samples as the number of features grows, so this is
best suited to a few features (about four) with a few hundred samples in each category.
Measurement uncertainties are not used.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from bedroc import RANDOM_SEED
from bedroc.core.data_container import DataContainer
from bedroc.core.plotting import get_figure, save_figure
from bedroc.core.type_aliases import NpBool, NpFloat, NpInt
from bedroc.difference.partitioning import Labeled, Unlabeled, resolve_labeled_unlabeled
from bedroc.difference.plotting import plot_group_fraction_posterior
from bedroc.difference.utils import kde_mixture_weight, save_fraction_summary

logger: logging.Logger = logging.getLogger(__name__)

MAX_RECOMMENDED_FEATURES: int = 4
"""Above this many features a warning is logged: kernel density estimates need rapidly more
samples per category as dimensions are added (e.g. Silverman 1986, Table 4.2)."""


class KDEModel:
    """Kernel density estimate mixture estimate of a population's category-0 fraction.

    Args:
        name: Name of the dataset
        labeled: Labeled data, whose categories' densities are estimated
        unlabeled: Target population whose category fraction is estimated
        n_bootstrap: Number of bootstrap replicates for :meth:`pi_0_samples`. Defaults to
            ``500``.
        bw_method: Kernel bandwidth rule passed to :class:`scipy.stats.gaussian_kde`. Defaults to
            ``None`` (Scott's rule).
        random_seed: Random seed for the bootstrap. Defaults to :data:`~bedroc.RANDOM_SEED`.
    """

    def __init__(
        self,
        name: str,
        labeled: Labeled,
        unlabeled: Unlabeled,
        *,
        n_bootstrap: int = 500,
        bw_method: str | float | None = None,
        random_seed: int | None = RANDOM_SEED,
    ):
        self.name: str = name
        self.labeled: Labeled = labeled
        self.unlabeled: Unlabeled = unlabeled
        self.n_bootstrap: int = n_bootstrap
        self.bw_method: str | float | None = bw_method
        self.random_seed: int | None = random_seed

        category_names = labeled.data.category_names
        assert category_names is not None  # Guaranteed by Labeled
        self.category_names: list[str] = [str(category) for category in category_names]

        # Kernel density estimates cannot use incomplete rows, so they are excluded
        labeled_values: pd.DataFrame = labeled.data.values_std
        codes = labeled.data.category_codes
        assert codes is not None  # Guaranteed by Labeled
        complete: NpBool = labeled_values.notna().all(axis=1).to_numpy()
        self.X_0: NpFloat = labeled_values.to_numpy()[complete & (codes.to_numpy() == 0)]
        self.X_1: NpFloat = labeled_values.to_numpy()[complete & (codes.to_numpy() == 1)]

        target_values: pd.DataFrame = unlabeled.data.values_std
        self._target_idx: NpInt = np.where(target_values.notna().all(axis=1).to_numpy())[0]
        self.X_target: NpFloat = target_values.to_numpy()[self._target_idx]
        n_dropped: int = len(target_values) - self._target_idx.size
        if n_dropped:
            logger.warning(
                "Excluding %d of %d target samples with missing feature values",
                n_dropped,
                len(target_values),
            )

        n_features: int = self.X_target.shape[1]
        if n_features > MAX_RECOMMENDED_FEATURES:
            logger.warning(
                "KDE mixture with %d features: kernel density estimates need rapidly more samples "
                "as features are added, so the densities (and the fraction) may be unreliable "
                "with %d and %d samples in the two categories",
                n_features,
                len(self.X_0),
                len(self.X_1),
            )

        self._pi_0: float | None = None
        self._pi_0_samples: NpFloat | None = None

    def fit(self) -> None:
        """Estimates the category-0 fraction from the full labeled and target data."""
        self._pi_0 = kde_mixture_weight(
            self.X_0, self.X_1, self.X_target, bw_method=self.bw_method
        )
        self._pi_0_samples = None

    @property
    def pi_0(self) -> float:
        """Point estimate of the category-0 fraction"""
        if self._pi_0 is None:
            raise ValueError("Model not yet fitted. Call 'fit()' first.")
        return self._pi_0

    def pi_0_samples(self) -> NpFloat:
        """Bootstrap replicates of the category-0 fraction.

        Each replicate resamples, with replacement and at their original sizes, each labeled
        category's samples and the target samples, refits both kernel density estimates, and
        re-estimates the fraction. The spread therefore includes the uncertainty in the estimated
        densities as well as in the finite target sample.

        Returns:
            Bootstrap replicates, shape ``(n_bootstrap,)``. Computed once per fit and then reused.
        """
        if self._pi_0_samples is not None:
            return self._pi_0_samples

        rng: np.random.Generator = np.random.default_rng(self.random_seed)
        samples: NpFloat = np.empty(self.n_bootstrap)
        for b in range(self.n_bootstrap):
            samples[b] = kde_mixture_weight(
                self.X_0[rng.integers(0, len(self.X_0), len(self.X_0))],
                self.X_1[rng.integers(0, len(self.X_1), len(self.X_1))],
                self.X_target[rng.integers(0, len(self.X_target), len(self.X_target))],
                bw_method=self.bw_method,
            )

        self._pi_0_samples = samples

        return samples

    def true_counts(self) -> pd.Series | None:
        """True category counts over the target samples used, or ``None`` if unknown."""
        categories = self.unlabeled.data.categories
        if categories is None:
            return None
        return categories.iloc[self._target_idx].value_counts(sort=False)

    def summary(self, output_directory: Path | None = None) -> pd.DataFrame:
        """Summary statistics of the fraction estimate.

        Args:
            output_directory: Directory to write ``<name>_summary_statistics.xlsx`` to (see
                :func:`~bedroc.difference.utils.save_fraction_summary`). ``None`` for no output.
                Defaults to ``None``.

        Returns:
            One-row dataframe of summary statistics for :meth:`pi_0_samples` (with the true
            fraction, if known), plus the ``point_estimate`` column
        """
        return save_fraction_summary(
            self.pi_0_samples(),
            category_counts=self.true_counts(),
            name=self.name,
            output_directory=output_directory,
            extra_columns={"point_estimate": self.pi_0},
        )


def pipeline(
    data: DataContainer,
    *,
    unlabeled: Unlabeled | None = None,
    output_directory: Path | None = None,
    random_seed: int | None = RANDOM_SEED,
    build_model_kwargs: dict[str, Any] | None = None,
) -> KDEModel:
    """Pipeline for the kernel density estimate mixture estimate of a population fraction.

    Uses the same labeled/unlabeled split as every other model for a given seed (see
    :func:`~bedroc.difference.partitioning.resolve_labeled_unlabeled`), fits a :class:`KDEModel`,
    and saves the fraction plot and the summary statistics.

    Args:
        data: The container holding the labeled input data
        unlabeled: Optional real unlabeled target population. If ``None``, a held-out split of
            ``data`` stands in for it instead, for self-validation. Defaults to ``None``.
        output_directory: Directory to save outputs. ``None`` for no output.
        random_seed: Random seed for reproducibility. Defaults to :data:`~bedroc.RANDOM_SEED`.
        build_model_kwargs: Optional keyword arguments for :class:`KDEModel` (e.g.
            ``{"n_bootstrap": 1000, "bw_method": "silverman"}``). Defaults to ``None``.

    Returns:
        The fitted :class:`KDEModel`
    """
    logger.info("Running KDE mixture pipeline for %s", data.name)

    if output_directory is not None:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

    labeled, unlabeled = resolve_labeled_unlabeled(data, unlabeled, random_seed=random_seed)
    model: KDEModel = KDEModel(
        data.name, labeled, unlabeled, random_seed=random_seed, **(build_model_kwargs or {})
    )
    model.fit()

    summary: pd.DataFrame = model.summary(output_directory=output_directory)
    logger.info(
        "KDE mixture %s fraction for %s: %.3f (95%% interval [%.3f, %.3f]), true fraction=%s",
        model.category_names[0],
        data.name,
        model.pi_0,
        float(summary["lower_95"].iloc[0]),
        float(summary["upper_95"].iloc[0]),
        "unknown" if "truth" not in summary else f"{float(summary['truth'].iloc[0]):.3f}",
    )

    ax: Axes = plot_group_fraction_posterior(
        model.pi_0_samples(),
        category_names=model.category_names,
        category_counts=model.true_counts(),
        show_prior=False,
        title="Bootstrap distribution of category fractions (KDE mixture)",
        interval_label="95% CI",
    )
    save_figure(get_figure(ax), Path(f"{data.name}_group_fraction_posterior"), output_directory)

    logger.info("KDE mixture pipeline completed for %s", data.name)

    return model
