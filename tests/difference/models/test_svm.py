# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for bedroc.difference.models.svm (support vector machine population-fraction estimation).
Plain sklearn fits on small synthetic data, no MCMC."""

import numpy as np
import pandas as pd
from sklearn.svm import SVC

from bedroc.core.data_container import DataContainer
from bedroc.difference.group_synthetic import SyntheticDataGenerator
from bedroc.difference.models.svm import SVMModel
from bedroc.difference.partitioning import Labeled, Unlabeled
from bedroc.difference.pipelines import MODEL_PIPELINES


def _labeled_unlabeled(
    *,
    offset: float,
    n_labeled_per_category: int,
    n_unlabeled: tuple[int, int],
    random_seed: int = 0,
) -> tuple[Labeled, Unlabeled]:
    """Balanced labeled data and an unlabeled population with the given per-category counts,
    drawn from the same two synthetic category distributions."""
    n_0: int = n_labeled_per_category + n_unlabeled[0]
    n_1: int = n_labeled_per_category + n_unlabeled[1]
    generator = SyntheticDataGenerator(
        n_samples=n_0 + n_1,
        n_features=3,
        feature_offsets=np.full(3, offset),
        feature_sigma=np.ones(3),
        category_0_fraction=n_0 / (n_0 + n_1),
        random_seed=random_seed,
    )
    generator.generate()
    values = pd.DataFrame(generator.X, columns=["f0", "f1", "f2"])
    metadata = pd.DataFrame({"Type": np.where(generator.X_category_idx == 0, "a", "b")})

    idx_0 = np.where(generator.X_category_idx == 0)[0]
    idx_1 = np.where(generator.X_category_idx == 1)[0]
    labeled_idx = np.concatenate([idx_0[:n_labeled_per_category], idx_1[:n_labeled_per_category]])
    unlabeled_idx = np.concatenate([idx_0[n_labeled_per_category:], idx_1[n_labeled_per_category:]])

    labeled = DataContainer(
        values.iloc[labeled_idx], metadata=metadata.iloc[labeled_idx], category_column="Type"
    )
    unlabeled = DataContainer(
        values.iloc[unlabeled_idx],
        metadata=metadata.iloc[unlabeled_idx],
        category_column="Type",
        scaling_params=labeled.scaling,
    )
    return Labeled(labeled), Unlabeled(unlabeled)


def test_svc_arguments_pass_through() -> None:
    labeled, unlabeled = _labeled_unlabeled(
        offset=1.0, n_labeled_per_category=20, n_unlabeled=(10, 10)
    )

    default = SVMModel("default", labeled, unlabeled)
    assert default.svc.get_params() == SVC().get_params()

    custom = SVMModel("custom", labeled, unlabeled, kernel="linear", C=0.5)
    assert custom.svc.kernel == "linear"
    assert custom.svc.C == 0.5


def test_adjusted_count_recovers_fraction_when_well_separated() -> None:
    labeled, unlabeled = _labeled_unlabeled(
        offset=5.0, n_labeled_per_category=100, n_unlabeled=(140, 60)
    )
    model = SVMModel("separated", labeled, unlabeled)
    model.fit()

    tpr, fpr = model.rates
    assert tpr > 0.95
    assert fpr < 0.05
    assert abs(model.adjusted_count() - 0.7) < 0.05


def test_adjusted_count_corrects_classify_and_count_bias_under_overlap() -> None:
    # Overlapping categories and a population fraction (0.8) far from the balanced training data:
    # classify-and-count is pulled towards 0.5 by misclassification; the adjusted count is not
    labeled, unlabeled = _labeled_unlabeled(
        offset=0.6, n_labeled_per_category=600, n_unlabeled=(1600, 400)
    )
    model = SVMModel("overlap", labeled, unlabeled)
    model.fit()

    assert model.classify_and_count() < 0.75
    assert abs(model.adjusted_count() - 0.8) < abs(model.classify_and_count() - 0.8)

    samples = model.pi_0_samples()
    assert samples.shape == (model.n_bootstrap,)
    assert np.quantile(samples, 0.025) < 0.8 < np.quantile(samples, 0.975)


def test_missing_unlabeled_rows_are_excluded_from_counts() -> None:
    labeled, unlabeled = _labeled_unlabeled(
        offset=2.0, n_labeled_per_category=40, n_unlabeled=(20, 20)
    )
    values = unlabeled.data.values.copy()
    values.iloc[[0, 1, 25], 0] = np.nan
    unlabeled_with_nan = Unlabeled(
        DataContainer(
            values,
            metadata=unlabeled.data.metadata,
            category_column="Type",
            scaling_params=labeled.data.scaling,
        )
    )

    model = SVMModel("nan", labeled, unlabeled_with_nan)
    model.fit()

    counts = model.true_counts()
    assert counts is not None
    assert counts.sum() == 37
    assert model.unlabeled_predictions.size == 37


def test_svm_is_registered() -> None:
    assert "svm" in MODEL_PIPELINES
