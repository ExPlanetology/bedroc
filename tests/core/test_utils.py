# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for bedroc.core.utils"""

import numpy as np
import pandas as pd
import pytest

from bedroc import HIGH_CI_PERCENTILE, LOW_CI_PERCENTILE
from bedroc.core.utils import (
    SummaryStatistics,
    eigen_summary,
    pooled_within_category_covariance,
    trim_samples,
)


def test_pooled_within_category_covariance_matches_manual_formula() -> None:
    """Matches the hand-computed sample-size-weighted pooled covariance formula."""
    rng = np.random.default_rng(0)
    group_0 = rng.normal(size=(12, 3))
    group_1 = rng.normal(size=(20, 3))

    result = pooled_within_category_covariance(group_0, group_1)

    n_0, n_1 = len(group_0), len(group_1)
    cov_0 = np.cov(group_0, rowvar=False, ddof=1)
    cov_1 = np.cov(group_1, rowvar=False, ddof=1)
    expected = ((n_0 - 1) * cov_0 + (n_1 - 1) * cov_1) / (n_0 + n_1 - 2)

    np.testing.assert_allclose(result.to_numpy(), expected)


def test_pooled_within_category_covariance_nan_safety() -> None:
    """A NaN confined to one feature doesn't corrupt covariance entries for other feature
    pairs, and doesn't produce NaN in the result at all."""
    rng = np.random.default_rng(1)
    group_0 = pd.DataFrame(rng.normal(size=(15, 3)), columns=["a", "b", "c"])
    group_1 = pd.DataFrame(rng.normal(size=(18, 3)), columns=["a", "b", "c"])

    # Feature "b" and "c" are untouched by the NaN, so their pooled covariance entry should
    # exactly match a version computed with no missing data at all.
    clean_result = pooled_within_category_covariance(group_0, group_1)

    group_0_with_nan = group_0.copy()
    group_0_with_nan.loc[0, "a"] = np.nan

    result = pooled_within_category_covariance(group_0_with_nan, group_1)

    assert not result.isna().any().any()
    np.testing.assert_allclose(result.loc["b", "c"], clean_result.loc["b", "c"])


def test_trim_samples_removes_outliers() -> None:
    """Trims samples outside the given percentile range, keeping the rest."""
    samples = np.concatenate([np.full(98, 5.0), [-1000.0, 1000.0]])

    trimmed = trim_samples(samples, low_percentile=5.0, high_percentile=95.0)

    assert -1000.0 not in trimmed
    assert 1000.0 not in trimmed
    assert np.all(trimmed == 5.0)


def test_eigen_summary_diagonal_matrix() -> None:
    """For a diagonal covariance matrix, eigen_summary should recover the diagonal entries as
    eigenvalues (sorted descending) with standard-basis eigenvectors."""
    matrix = pd.DataFrame(
        np.diag([4.0, 1.0, 0.25]), index=["a", "b", "c"], columns=["a", "b", "c"]
    )

    summary = eigen_summary(matrix)

    np.testing.assert_allclose(summary.loc["eigenvalue"].to_numpy(), [4.0, 1.0, 0.25])
    np.testing.assert_allclose(summary.loc["explained variance ratio"].sum(), 1.0)
    # The largest-eigenvalue eigenvector (PC1) should point purely along "a" (up to sign).
    np.testing.assert_allclose(abs(summary.loc["a", "PC1"]), 1.0)


def test_summary_statistics_matches_manual_computation() -> None:
    """SummaryStatistics' properties match a direct numpy computation on the same samples."""
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=2.0, scale=1.0, size=2000)
    truth = 2.0

    stats = SummaryStatistics(samples, truth=truth)

    np.testing.assert_allclose(stats.mean, np.mean(samples))
    np.testing.assert_allclose(stats.median, np.median(samples))
    np.testing.assert_allclose(stats.lower_95, np.percentile(samples, LOW_CI_PERCENTILE))
    np.testing.assert_allclose(stats.upper_95, np.percentile(samples, HIGH_CI_PERCENTILE))
    np.testing.assert_allclose(stats.ci_width, stats.upper_95 - stats.lower_95)

    # truth was provided above, so these are never actually None at runtime; narrow the type
    # explicitly since the properties are typed NpArray | None in general (when truth is absent).
    assert stats.error_mean is not None
    assert stats.abs_error_mean is not None
    assert stats.rmse is not None
    assert stats.mae is not None

    np.testing.assert_allclose(stats.error_mean, stats.mean - truth)
    np.testing.assert_allclose(stats.abs_error_mean, abs(stats.mean - truth))
    np.testing.assert_allclose(stats.rmse, np.sqrt(np.mean((samples - truth) ** 2)))
    np.testing.assert_allclose(stats.mae, np.mean(np.abs(samples - truth)))
    assert bool(stats.within_ci) == (stats.lower_95 <= truth <= stats.upper_95)

    as_dict = stats.to_dict()
    assert as_dict["mean"] == pytest.approx(stats.mean.item())

    as_df = stats.to_dataframe()
    assert "mean" in as_df.columns
    assert len(as_df) == 1
