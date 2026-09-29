# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Ways of partitioning a DataContainer for machine-learning use: train/test splitting and
labeled/unlabeled category-pair splitting."""

import logging
from dataclasses import dataclass
from typing import Self

import pandas as pd
from numpy.typing import ArrayLike
from sklearn.model_selection import train_test_split as sklearn_train_test_split

from bedroc.core.data_container import DataContainer

logger: logging.Logger = logging.getLogger(__name__)


def train_test_split(
    data: DataContainer,
    *,
    test_size: float | None = 0.2,
    random_state: int | None = None,
    shuffle: bool = True,
    stratify: ArrayLike | None = None,
) -> tuple[DataContainer, DataContainer]:
    """Splits a data container into training and test sets.

    Scaling parameters are calculated from the training data and then applied unchanged to both
    the training and test sets.

    Args:
        data: Data container to split.
        test_size: Proportion of the dataset to include in the test split. Defaults to ``0.2``.
        random_state: Controls the shuffling applied to the data before applying the split.
            Defaults to ``None``.
        shuffle: Whether to shuffle the data before splitting. Defaults to ``True``.
        stratify: Target variable used to stratify the split. Defaults to ``None``, which
            stratifies on ``data.categories`` whenever ``data.category_column`` is set.

    Returns:
        Tuple containing the training and test data containers.
    """
    # Default to stratifying on categories if stratify isn't explicitly passed
    if stratify is None and data.category_column is not None:
        stratify = data.categories

    train_idx, test_idx = sklearn_train_test_split(
        data.values.index,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify,
    )

    # Train container calculates its own scaling parameters
    train = DataContainer(
        values=data.values.loc[train_idx],
        uncertainties=data.uncertainties.loc[train_idx],
        metadata=data.metadata.loc[train_idx],
        name=f"{data.name}_train",
        uncertainty_scale=1.0,  # Crucial: Already converted to 1-sigma
        select_data_column=data.select_data_column,
        category_column=data.category_column,
    )

    # Test container uses the parameters learned from the training data
    test = DataContainer(
        values=data.values.loc[test_idx],
        uncertainties=data.uncertainties.loc[test_idx],
        metadata=data.metadata.loc[test_idx],
        name=f"{data.name}_test",
        uncertainty_scale=1.0,  # Crucial: Already converted to 1-sigma
        scaling_params=train.scaling,  # Pass learned scaling directly
        select_data_column=data.select_data_column,
        category_column=data.category_column,
    )

    return train, test


@dataclass(frozen=True)
class Labeled:
    """Wraps a :class:`DataContainer` guaranteed to have exactly two known category labels — safe
    to fit a supervised likelihood on.

    Making this a distinct type, rather than passing a bare :class:`DataContainer`, means "this
    data has known labels" is a property the type system tracks, not a convention every caller
    must remember.
    """

    data: DataContainer

    def __post_init__(self) -> None:
        if self.data.category_column is None or self.data.category_names is None:
            raise ValueError("Labeled requires a DataContainer with category_column set.")
        if len(self.data.category_names) != 2:
            raise ValueError(
                f"Labeled requires exactly two categories, got {len(self.data.category_names)}."
            )


@dataclass(frozen=True)
class Unlabeled:
    """Wraps a :class:`DataContainer` presented to a semi-supervised model as the unlabeled target
    population.

    ``data`` may or may not have an active ``category_column``:

    - Unset (e.g. :attr:`LabeledUnlabeledSplit.unlabeled` on genuinely unknown-provenance data,
      such as Michigan's Detrital zircons): no ground truth exists; ``data.category_counts`` is
      ``None``.
    - Set (e.g. the held-out half of a :func:`train_test_split`, standing in for real unlabeled
      data during self-validation): ``data.category_counts``/``data.category_codes`` give ground
      truth for post-hoc evaluation/plotting only — never fed to a model's likelihood.
    """

    data: DataContainer


@dataclass(frozen=True)
class LabeledUnlabeledSplit:
    """Splits a multi-category :class:`DataContainer` into a binary labeled comparison pair and a
    pooled unlabeled remainder.

    :class:`DataContainer` itself is category-count-agnostic pure storage, but the difference/
    classification models in this package assume exactly two categories (a labeled comparison
    pair) plus, for the semi-supervised joint models, a separate unlabeled population to classify
    against that pair. This class holds that classification-specific policy — picking exactly two
    categories to compare and treating every other row as one pooled unlabeled population — without
    adding any of that vocabulary to :class:`DataContainer` itself.
    """

    labeled: Labeled
    """Rows from the two chosen categories only, with ``category_column`` active. Category codes
    follow the caller-specified order (``categories[0]`` is code 0, ``categories[1]`` is code 1),
    not alphabetical order, so callers can control which category is "0" vs "1" (this matters for
    e.g. :attr:`~bedroc.difference.base.CategoryComparisonBase.difference_string` and
    :meth:`~bedroc.core.data_container.DataDiagnostics.category_mean_difference`, both of which are
    directional)."""
    unlabeled: Unlabeled
    """Every other row, standardized using :attr:`labeled`'s scaling parameters (not its own) so
    both containers are on the same standardized scale — required for the semi-supervised joint
    models and for a classifier fitted on ``labeled`` to meaningfully score ``unlabeled``. Has no
    active ``category_column``; the original category value (if any) is retained as an ordinary
    metadata column for later inspection, not used for modeling."""

    @classmethod
    def from_data_container(
        cls, data: DataContainer, *, categories: tuple[str, str], name: str | None = None
    ) -> Self:
        """Creates a labeled/unlabeled split from a multi-category data container.

        Args:
            data: Source data container, with ``category_column`` set to a column that may have
                any number of distinct category values.
            categories: The two category values to treat as the labeled comparison pair, in order:
                ``categories[0]`` becomes code 0, ``categories[1]`` becomes code 1. Every row whose
                category is not one of these two is pooled into :attr:`unlabeled`.
            name: Base name for the resulting containers, used as
                ``f"{name}_labeled"``/``f"{name}_unlabeled"``. Defaults to ``data.name``.

        Raises:
            ValueError: If ``data`` has no ``category_column`` set, if ``categories`` doesn't
                contain exactly two distinct values, or if either value is not present in
                ``data``'s categories.

        Returns:
            A :class:`LabeledUnlabeledSplit` with the resulting :attr:`labeled`/:attr:`unlabeled`
            containers.
        """
        if data.category_column is None or data.categories is None:
            raise ValueError(
                "LabeledUnlabeledSplit.from_data_container requires a DataContainer with "
                "category_column set."
            )

        if len(set(categories)) != 2:
            raise ValueError(
                f"categories must contain exactly two distinct values, got {categories!r}."
            )

        available: set[str] = set(data.categories.dropna().unique())
        missing: list[str] = [category for category in categories if category not in available]
        if missing:
            raise ValueError(
                f"categories {missing!r} not present in data (available: {sorted(available)})."
            )

        base_name: str = name if name is not None else data.name
        category_column: str = data.category_column
        is_labeled_mask: pd.Series = data.categories.isin(categories)

        # Re-cast to an explicit, caller-ordered CategoricalDtype before construction:
        # DataContainer.__init__ preserves an already-categorical column's exact category universe
        # (rather than re-deriving one via alphabetical sorting), so this is what makes the
        # caller's chosen 0/1 order authoritative.
        labeled_metadata: pd.DataFrame = data.metadata.loc[is_labeled_mask].copy()
        labeled_metadata[category_column] = labeled_metadata[category_column].astype(
            pd.CategoricalDtype(categories=list(categories), ordered=True)
        )

        labeled: DataContainer = DataContainer(
            values=data.values.loc[is_labeled_mask],
            uncertainties=data.uncertainties.loc[is_labeled_mask],
            metadata=labeled_metadata,
            name=f"{base_name}_labeled",
            uncertainty_scale=1.0,  # data.uncertainties is already true 1-sigma
            select_data_column=data.select_data_column,
            category_column=category_column,
        )

        unlabeled_mask: pd.Series = ~is_labeled_mask
        unlabeled: DataContainer = DataContainer(
            values=data.values.loc[unlabeled_mask],
            uncertainties=data.uncertainties.loc[unlabeled_mask],
            metadata=data.metadata.loc[unlabeled_mask],
            name=f"{base_name}_unlabeled",
            uncertainty_scale=1.0,  # data.uncertainties is already true 1-sigma
            scaling_params=labeled.scaling,  # Same standardized scale as `labeled`
            select_data_column=data.select_data_column,
            category_column=None,
        )

        logger.info(
            "Split '%s' into labeled (%d rows, categories=%s) and unlabeled (%d rows)",
            base_name,
            labeled.n_data,
            categories,
            unlabeled.n_data,
        )

        return cls(labeled=Labeled(labeled), unlabeled=Unlabeled(unlabeled))


def resolve_labeled_unlabeled(
    data: DataContainer, unlabeled: Unlabeled | None, *, random_seed: int | None
) -> tuple[Labeled, Unlabeled]:
    """Resolves the labeled data to fit on and the unlabeled population to infer over.

    Shared by every model pipeline, so that for a given seed every model uses exactly the same
    held-out split in self-validation.

    Args:
        data: The container holding the labeled input data
        unlabeled: Real unlabeled target population, or ``None`` for self-validation
        random_seed: Random seed for the self-validation split

    Returns:
        If ``unlabeled`` is ``None``, the training part of :func:`train_test_split` as the labeled
        data and its held-out test part as the unlabeled population. Otherwise all of ``data`` as
        the labeled data and ``unlabeled`` unchanged.
    """
    if unlabeled is None:
        train, test = train_test_split(data, random_state=random_seed)
        return Labeled(train), Unlabeled(test)

    return Labeled(data), unlabeled
