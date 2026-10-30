# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for bedroc.difference.partitioning's Labeled/Unlabeled wrapper types."""

import pandas as pd
import pytest

from bedroc.core.data_container import DataContainer
from bedroc.difference.partitioning import Labeled, LabeledUnlabeledSplit, Unlabeled


def _make_data_container(categories: list[str], *, category_column: str | None) -> DataContainer:
    n = len(categories)
    values = pd.DataFrame({"feature_0": range(n)}, dtype=float)
    metadata = (
        pd.DataFrame({category_column: categories}) if category_column is not None else None
    )
    return DataContainer(values, metadata=metadata, category_column=category_column)


def test_labeled_accepts_two_categories() -> None:
    data = _make_data_container(["a", "b"], category_column="Type")
    labeled = Labeled(data)
    assert labeled.data is data


def test_labeled_rejects_missing_category_column() -> None:
    data = _make_data_container(["a", "b"], category_column=None)
    with pytest.raises(ValueError, match="category_column set"):
        Labeled(data)


def test_labeled_rejects_more_than_two_categories() -> None:
    data = _make_data_container(["a", "b", "c"], category_column="Type")
    with pytest.raises(ValueError, match="exactly two categories"):
        Labeled(data)


def test_unlabeled_accepts_data_with_no_category_column() -> None:
    data = _make_data_container(["a", "b"], category_column=None)
    unlabeled = Unlabeled(data)
    assert unlabeled.data.category_counts is None


def test_unlabeled_accepts_data_with_known_labels() -> None:
    data = _make_data_container(["a", "b"], category_column="Type")
    unlabeled = Unlabeled(data)
    assert unlabeled.data.category_counts is not None


def _three_type_container() -> DataContainer:
    types = ["a"] * 4 + ["b"] * 3 + ["Detrital"] * 5 + ["Unknown"] * 2
    values = pd.DataFrame({"feature_0": range(len(types))}, dtype=float)
    return DataContainer(values, metadata=pd.DataFrame({"Type": types}), category_column="Type")


def test_split_pools_every_other_category_by_default() -> None:
    split = LabeledUnlabeledSplit.from_data_container(
        _three_type_container(), categories=("a", "b")
    )
    assert split.labeled.data.n_data == 7
    assert split.unlabeled.data.n_data == 7


def test_split_restricts_unlabeled_to_given_categories() -> None:
    split = LabeledUnlabeledSplit.from_data_container(
        _three_type_container(), categories=("a", "b"), unlabeled_categories=("Detrital",)
    )
    assert split.labeled.data.n_data == 7
    assert split.unlabeled.data.n_data == 5
    assert set(split.unlabeled.data.metadata["Type"]) == {"Detrital"}


def test_split_rejects_unlabeled_categories_overlapping_labeled() -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        LabeledUnlabeledSplit.from_data_container(
            _three_type_container(), categories=("a", "b"), unlabeled_categories=("a",)
        )
