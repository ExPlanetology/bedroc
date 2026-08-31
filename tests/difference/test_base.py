# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for bedroc.difference.base.CategoryComparisonBase's constructor validation and
pre-build/pre-inference property errors -- pure object construction, no PyMC sampling needed."""

import numpy as np
import pytest

from bedroc.difference.models.unified_naive import UnifiedNaiveModel

X = np.zeros((4, 2))
X_category_idx = np.array([0, 0, 1, 1])
X_unlabeled = np.zeros((3, 2))


def test_difference_string_uses_category_names() -> None:
    model = UnifiedNaiveModel(
        "t", X, X_category_idx, X_unlabeled, category_names=["Plutonic", "Volcanic"]
    )
    assert model.difference_string == "(Volcanic - Plutonic)"


def test_idata_raises_before_run_inference() -> None:
    model = UnifiedNaiveModel("t", X, X_category_idx, X_unlabeled)
    with pytest.raises(ValueError, match="Inference has not been run"):
        _ = model.idata


def test_model_raises_before_build_model() -> None:
    model = UnifiedNaiveModel("t", X, X_category_idx, X_unlabeled)
    with pytest.raises(ValueError, match="Model has not been built"):
        _ = model.model


def test_rejects_mismatched_feature_names_length() -> None:
    with pytest.raises(ValueError, match="feature_names"):
        UnifiedNaiveModel("t", X, X_category_idx, X_unlabeled, feature_names=["only_one_name"])


def test_rejects_category_names_not_length_two() -> None:
    with pytest.raises(ValueError, match="exactly two names"):
        UnifiedNaiveModel("t", X, X_category_idx, X_unlabeled, category_names=["only_one"])
