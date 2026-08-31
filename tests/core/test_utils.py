# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for bedroc.core.utils"""

import numpy as np
import pandas as pd

from bedroc.core.utils import pooled_within_category_covariance


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
