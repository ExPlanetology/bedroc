# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Filtering criteria and transforms for zircon datasets"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureFilter:
    """Filtering criteria and transform for a single feature column.

    Rows with a missing (NaN) value are always kept, so missingness is handled downstream rather
    than here.

    Args:
        minimum: Exclusive lower bound. ``None`` for no lower bound.
        maximum: Exclusive upper bound. ``None`` for no upper bound.
        log_transform: Whether to log-transform the feature (after filtering) to mitigate right
            skewness. Its uncertainty is converted to a relative uncertainty to match.
    """

    minimum: float | None = None
    maximum: float | None = None
    log_transform: bool = False

    def mask(self, values: pd.Series) -> pd.Series:
        """Boolean mask of the rows to keep.

        Args:
            values: Feature values

        Returns:
            ``True`` for rows within the bounds or with a missing value
        """
        keep: pd.Series = pd.Series(True, index=values.index)
        if self.minimum is not None:
            keep &= values > self.minimum
        if self.maximum is not None:
            keep &= values < self.maximum

        return values.isna() | keep


@dataclass(frozen=True)
class ZirconFilter:
    """Filtering criteria and transforms for a zircon dataset.

    Args:
        features: Filter for each feature column, keyed by column name. Columns absent from the
            dataframe are skipped.
    """

    features: Mapping[str, FeatureFilter]

    def renamed(self, renames: Mapping[str, str]) -> "ZirconFilter":
        """A copy with the feature columns renamed.

        Args:
            renames: New column name for each column to rename. Other columns keep their name.

        Returns:
            The renamed filter
        """
        return ZirconFilter(
            {renames.get(column, column): rule for column, rule in self.features.items()}
        )

    def apply(self, df: pd.DataFrame, uncertainty_columns: Mapping[str, str]) -> pd.DataFrame:
        """Applies the filtering criteria and then the log transforms.

        Args:
            df: Dataframe to filter
            uncertainty_columns: Uncertainty column for each feature column

        Returns:
            The filtered and transformed dataframe
        """
        logger.info("Applying filtering criteria to the data")

        for column, feature_filter in self.features.items():
            if column not in df.columns:
                continue
            if feature_filter.minimum is not None:
                logger.info("Removing %s values less than %g", column, feature_filter.minimum)
            if feature_filter.maximum is not None:
                logger.info("Removing %s values greater than %g", column, feature_filter.maximum)
            df = df.loc[feature_filter.mask(df[column])]

        df = df.copy()
        for column, feature_filter in self.features.items():
            if column not in df.columns or not feature_filter.log_transform:
                continue
            df[uncertainty_columns[column]] = df[uncertainty_columns[column]] / df[column]
            df[column] = np.log(df[column])

        return df


ZIRCON_FILTER: ZirconFilter = ZirconFilter(
    {
        "Ti": FeatureFilter(minimum=0, maximum=200, log_transform=True),  # or max 300
        "Hf": FeatureFilter(minimum=5000, log_transform=True),
        "Th": FeatureFilter(maximum=2000, log_transform=True),
        "U": FeatureFilter(maximum=2000, log_transform=True),
        "Y": FeatureFilter(log_transform=True),
        "Nb": FeatureFilter(log_transform=True),
        "Eu/Eu*": FeatureFilter(log_transform=True),
        "Ce/Ce*": FeatureFilter(log_transform=True),
    }
)
"""Default zircon filtering criteria from Olivier and Tobias (7/8/2026). Bounds are in ppm. Keyed
by the standard feature names, so datasets with different raw column names should apply
:meth:`ZirconFilter.renamed` first."""
