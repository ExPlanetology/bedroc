# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for bedroc.difference.models.kde (kernel density estimate mixture estimate of a
population fraction). No MCMC."""

import numpy as np
import pandas as pd

from bedroc.core.data_container import DataContainer
from bedroc.difference.models.kde import KDEModel
from bedroc.difference.partitioning import Labeled, Unlabeled
from bedroc.difference.pipelines import MODEL_PIPELINES


def _labeled_unlabeled(
    *, offset: float, n_labeled_per_category: int, n_target: tuple[int, int], random_seed: int = 0
) -> tuple[Labeled, Unlabeled]:
    """Balanced labeled data and a target population with the given per-category counts, drawn
    from the same two category distributions (the target keeps its true categories)."""
    rng = np.random.default_rng(random_seed)
    means = (np.zeros(3), np.full(3, offset))

    def draw(n_0: int, n_1: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        values = np.vstack(
            [rng.normal(means[0], 1.0, (n_0, 3)), rng.normal(means[1], 1.0, (n_1, 3))]
        )
        return (
            pd.DataFrame(values, columns=["f0", "f1", "f2"]),
            pd.DataFrame({"Type": ["a"] * n_0 + ["b"] * n_1}),
        )

    values, metadata = draw(n_labeled_per_category, n_labeled_per_category)
    labeled = DataContainer(values, metadata=metadata, category_column="Type", name="test")
    target_values, target_metadata = draw(*n_target)
    target = DataContainer(
        target_values,
        metadata=target_metadata,
        category_column="Type",
        scaling_params=labeled.scaling,
    )
    return Labeled(labeled), Unlabeled(target)


def test_kde_mixture_recovers_shifted_fraction() -> None:
    # Overlapping categories and a target fraction (0.8) far from the balanced labeled data
    labeled, unlabeled = _labeled_unlabeled(
        offset=1.0, n_labeled_per_category=500, n_target=(800, 200)
    )
    model = KDEModel("shifted", labeled, unlabeled, n_bootstrap=50)
    model.fit()

    assert abs(model.pi_0 - 0.8) < 0.06

    # The bootstrap varies, is centred near the point estimate, and has a plausible width (a
    # single draw is not required to cover the truth: about 1 in 20 should not)
    samples = model.pi_0_samples()
    assert samples.shape == (50,)
    assert samples.std() > 0
    assert abs(np.median(samples) - model.pi_0) < 0.03
    lower, upper = np.quantile(samples, [0.025, 0.975])
    assert 0.02 < upper - lower < 0.25


def test_kde_mixture_true_counts_exclude_missing_rows() -> None:
    labeled, unlabeled = _labeled_unlabeled(
        offset=2.0, n_labeled_per_category=100, n_target=(30, 30)
    )
    values = unlabeled.data.values.copy()
    values.iloc[[0, 1, 40], 0] = np.nan
    target = Unlabeled(
        DataContainer(
            values,
            metadata=unlabeled.data.metadata,
            category_column="Type",
            scaling_params=labeled.data.scaling,
        )
    )

    model = KDEModel("nan", labeled, target, n_bootstrap=5)
    counts = model.true_counts()

    assert counts is not None
    assert counts.tolist() == [28, 29]
    assert model.X_target.shape[0] == 57


def test_kde_is_registered() -> None:
    assert "kde" in MODEL_PIPELINES
