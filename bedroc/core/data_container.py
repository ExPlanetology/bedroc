# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Core classes and functions"""

import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, Self

import arviz as az
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from numpy.typing import ArrayLike
from sklearn.model_selection import train_test_split as sklearn_train_test_split

from bedroc.core.type_aliases import NpFloat

logger: logging.Logger = logging.getLogger(__name__)

LOW_CI_PERCENTILE: float = 2.5
"""Low percentile for 95% equal-tailed credible intervals"""
HIGH_CI_PERCENTILE: float = 97.5
"""High percentile for 95% equal-tailed credible intervals"""
CI_PROB: float = (HIGH_CI_PERCENTILE - LOW_CI_PERCENTILE) / 100
"""Probability contained within credible intervals"""
CI_KIND: str = "eti"
"""Type of credible interval used for ArviZ plots"""

RANDOM_SEED: int | None = 321  # 123
"""Random seed for reproducibility. Set to ``None`` for random behavior."""
SAVEFIG_KWARGS: dict[str, Any] = {"dpi": 300, "bbox_inches": "tight", "format": "pdf"}
"""Default savefig options"""

# Update ArviZ rcParams for credible intervals to be consistent. Older versions of Arviz used HDI
# and 0.95, but now the default is ETI and 0.94. We want to use ETI and 0.95.
az.rcParams["stats.ci_prob"] = CI_PROB
az.rcParams["stats.ci_kind"] = CI_KIND
az.rcParams["stats.point_estimate"] = "median"


class DataContainer:
    """Container for feature values, measurement uncertainties, and metadata.

    Feature values and measurement uncertainties are stored separately from metadata. Feature
    values can be standardized using scaling parameters calculated from the supplied data or using
    externally provided scaling parameters.

    Args:
        values: DataFrame containing feature values. Columns represent features and rows represent
            samples.
        uncertainties: Optional DataFrame containing the measurement uncertainties corresponding to
            ``values``. Must have the same index and columns as ``values``. Uncertainties are
            assumed to be reported as ``uncertainty_scale`` standard deviations.
        metadata: Optional DataFrame containing metadata associated with each sample. Must have the
            same index as ``values``.
        name: Data container name. Defaults to ``"data"``.
        uncertainty_scale: Number of standard deviations represented by the supplied uncertainties.
            For example, use ``2.0`` if the input uncertainties are reported as 2-sigma
            uncertainties. Defaults to ``1.0``.
        scaling_means: Optional feature means to use for standardization. If provided,
            ``scaling_stds`` must also be provided. The indices must correspond to the feature
            columns in ``values``. If omitted, the means are calculated from ``values``.
        scaling_stds: Optional feature standard deviations to use for standardization. If provided,
            ``scaling_means`` must also be provided. If omitted, the standard deviations are
            calculated from ``values``.
        select_features: Optional iterable of feature names to retain. Defaults to ``None``, which
            retains all features.
        select_data: Optional iterable of values used to select samples based on ``data_column``.
            Defaults to ``None``, which retains all samples.
        data_column: Name of the metadata column used by ``select_data``. Defaults to ``"ID"``.
    """

    def __init__(
        self,
        values: pd.DataFrame,
        uncertainties: pd.DataFrame | None = None,
        metadata: pd.DataFrame | None = None,
        *,
        name: str = "data",
        uncertainty_scale: float = 1.0,
        scaling_means: pd.Series | None = None,
        scaling_stds: pd.Series | None = None,
        select_features: Iterable[str] | None = None,
        select_data: Iterable[Any] | None = None,
        data_column: str = "ID",
    ):
        self.name: str = name
        self.data_column: str = data_column

        # Validate inputs
        if not values.columns.is_unique:
            raise ValueError("Values must have unique feature names")

        if uncertainties is not None:
            if not uncertainties.columns.is_unique:
                raise ValueError("Uncertainties must have unique feature names")

            if not values.index.equals(uncertainties.index):
                raise ValueError("Values and uncertainties must have the same index")

            if not values.columns.equals(uncertainties.columns):
                raise ValueError("Values and uncertainties must have the same columns")

        if metadata is not None:
            if not metadata.columns.is_unique:
                raise ValueError("Metadata must have unique column names")

            if not values.index.equals(metadata.index):
                raise ValueError("Values and metadata must have the same index")

        if uncertainty_scale <= 0:
            raise ValueError("uncertainty_scale must be greater than zero")

        if (scaling_means is None) != (scaling_stds is None):
            raise ValueError(
                "scaling_means and scaling_stds must either both be provided or both be None"
            )

        # Independent copies
        self.values: pd.DataFrame = values.copy()

        self.uncertainties: pd.DataFrame = (
            uncertainties.copy()
            if uncertainties is not None
            else pd.DataFrame(np.nan, index=self.values.index, columns=self.values.columns)
        )

        self.metadata: pd.DataFrame = (
            metadata.copy() if metadata is not None else pd.DataFrame(index=self.values.index)
        )

        # Select features
        if select_features is not None:
            features = list(select_features)
            self.values = self.values.loc[:, features]
            self.uncertainties = self.uncertainties.loc[:, features]

        # Select samples
        if select_data is not None:
            if data_column not in self.metadata.columns:
                raise ValueError(f"Data column {data_column!r} not found in metadata")

            mask = self.metadata[data_column].isin(list(select_data))  # list for typing

            self.values = self.values.loc[mask]
            self.uncertainties = self.uncertainties.loc[mask]
            self.metadata = self.metadata.loc[mask]

        if self.n_data == 0:
            raise ValueError("No data remain after selection")

        # Standardization parameters
        if scaling_means is None:
            self.scaling_means = self.values.mean(axis=0)
            self.scaling_stds = self.values.std(axis=0, ddof=0)
        else:
            assert scaling_stds is not None

            self.scaling_means = scaling_means.reindex(self.values.columns)
            self.scaling_stds = scaling_stds.reindex(self.values.columns)

            if self.scaling_means.isna().any() or self.scaling_stds.isna().any():
                raise ValueError("Scaling parameters must be provided for every feature")

        if not np.isfinite(self.scaling_means).all():
            raise ValueError("Scaling means must be finite")

        if not np.isfinite(self.scaling_stds).all() or (self.scaling_stds <= 0).any():
            invalid = self.scaling_stds.loc[
                ~np.isfinite(self.scaling_stds) | (self.scaling_stds <= 0)
            ].index.tolist()

            raise ValueError(f"Scaling standard deviations must be finite and positive: {invalid}")

        # Standardized values
        self.values_std: pd.DataFrame = (self.values - self.scaling_means) / self.scaling_stds

        # Convert supplied uncertainties to 1-sigma uncertainties
        self.uncertainties = self.uncertainties / uncertainty_scale

        # Standardized measurement uncertainties
        self.uncertainties_std = self.uncertainties / self.scaling_stds

        logger.info("Data container '%s' initialized", self.name)
        logger.info("Number of samples = %d", self.n_data)
        logger.info("Number of features = %d", self.n_features)
        logger.info("Feature names: %s", self.feature_names.values)

    @property
    def n_data(self) -> int:
        return len(self.values)

    @property
    def n_features(self) -> int:
        return self.values.shape[1]

    @property
    def feature_names(self) -> pd.Index:
        return self.values.columns

    @property
    def data_names(self) -> list[Any]:
        if self.data_column not in self.metadata.columns:
            raise ValueError(f"Data column {self.data_column!r} not found in metadata")
        return self.metadata[self.data_column].to_list()

    @classmethod
    def from_dataframe(
        cls,
        dataframe: pd.DataFrame,
        *,
        feature_suffix: str = "_feature",
        uncertainty_suffix: str = "_uncertainty",
        feature_renames: Mapping[str, str] | None = None,
        **kwargs,
    ) -> Self:
        """Creates a data container from a combined dataframe.

        Feature and uncertainty columns are identified by their suffixes. All remaining columns are
        treated as metadata.

        Args:
            dataframe: A dataframe with columns of feature values and their uncertainties
            feature_suffix: Suffix of feature value columns. Defaults to ``_feature``.
            uncertainty_suffix: Suffix of feature uncertainty columns. Defaults to
                ``_uncertainty``.
            feature_renames: Mapping of feature names to their renamed versions. Defaults to
                ``None``.
            **kwargs: Arbitrary keyword arguments for constructor
        """
        feature_columns: list[str] = [c for c in dataframe.columns if c.endswith(feature_suffix)]
        uncertainty_columns: list[str] = [
            c for c in dataframe.columns if c.endswith(uncertainty_suffix)
        ]

        values: pd.DataFrame = dataframe.loc[:, feature_columns].copy()
        uncertainties: pd.DataFrame | None = (
            dataframe.loc[:, uncertainty_columns].copy() if uncertainty_columns else None
        )

        if feature_renames is not None:
            values = cls._rename_feature_prefixes(values, feature_renames)
            if uncertainties is not None:
                uncertainties = cls._rename_feature_prefixes(uncertainties, feature_renames)

        # Converts both to the same bare feature names.
        values.columns = [c.removesuffix(feature_suffix) for c in values.columns]
        if uncertainties is not None:
            uncertainties.columns = [
                c.removesuffix(uncertainty_suffix) for c in uncertainties.columns
            ]

        # Everything else is metadata
        metadata_columns: list[str] = [
            c
            for c in dataframe.columns
            if c not in feature_columns and c not in uncertainty_columns
        ]
        metadata: pd.DataFrame = dataframe.loc[:, metadata_columns].copy()

        return cls(values=values, uncertainties=uncertainties, metadata=metadata, **kwargs)

    @classmethod
    def from_csv(cls, filename_path: Path | str, **kwargs) -> Self:
        """Creates an instance from a CSV file.

        Args:
            filename_path: Path to the CSV file
            **kwargs: Arbitrary keyword arguments for constructor

        Returns:
            An instance
        """
        data: pd.DataFrame = pd.read_csv(filename_path)

        return cls.from_dataframe(data, **kwargs)

    @classmethod
    def from_excel(cls, filename_path: Path | str, sheet_name: Any, **kwargs) -> Self:
        """Creates an instance from an Excel file.

        Args:
            filename_path: Path to the Excel file
            sheet_name: Sheet name
            **kwargs: Arbitrary keyword arguments for constructor

        Returns:
            An instance
        """
        data: pd.DataFrame = pd.read_excel(filename_path, sheet_name=sheet_name)

        return cls.from_dataframe(data, **kwargs)

    def get_destandardized_values(self, standardized_values: NpFloat) -> NpFloat:
        """Converts standardized values back to the original feature scale.

        Args:
            standardized_values: Standardized values. Must have a shape of
                ``(n_data, n_features)`` or ``(n_data, n_features, n_samples)``.

        Returns:
            Destandardized values with matching shape.
        """
        if standardized_values.ndim not in (2, 3):
            raise ValueError(
                "standardized_values must have 2 or 3 dimensions: "
                "(n_data, n_features) or (n_data, n_features, n_samples)"
            )

        # Shape: (1, n_features, 1)
        means = self.scaling_means.to_numpy()[np.newaxis, :, np.newaxis]
        stds = self.scaling_stds.to_numpy()[np.newaxis, :, np.newaxis]

        if standardized_values.ndim == 2:
            # (n_data, n_features) -> (n_data, n_features, 1)
            standardized_values = standardized_values[..., np.newaxis]
            result = standardized_values * stds + means

            # Return to (n_data, n_features)
            return result[..., 0]

        # (n_data, n_features, n_samples)
        return standardized_values * stds + means

    def plot_correlation_coefficient(
        self,
        *,
        method: Literal["pearson", "kendall", "spearman"] = "pearson",
        min_periods=1,
        numeric_only=False,
    ) -> Axes:
        """Plots a heatmap of the correlation coefficient.

        Args:
            method: Method for calculating correlation. Defaults to ``"pearson"``.
            min_periods: Minimum number of observations required per pair of columns to have a
                valid result. Defaults to ``1``.
            numeric_only: Whether to include only numeric columns. Defaults to ``False``.

        Returns:
            Figure axes
        """
        # Compute pairwise correlation of columns, excluding NA/null values
        # equivalent to np.correcoef(self.values, rowvar=False)
        corr_matrix: pd.DataFrame = self.values.corr(method, min_periods, numeric_only)
        ax: Axes = sns.heatmap(corr_matrix, cmap="magma", annot=True, fmt=".2f", vmin=-1, vmax=1)
        ax.set_title(f"{method.capitalize()} correlation coefficient")

        return ax

    @staticmethod
    def _rename_feature_prefixes(
        dataframe: pd.DataFrame, renames: Mapping[str, str]
    ) -> pd.DataFrame:
        """Replaces feature-name prefixes in dataframe columns.

        Args:
            dataframe: Dataframe with feature columns to rename
            renames: Dictionary mapping old prefixes to new prefixes

        Returns:
            Dataframe with renamed feature columns
        """
        rename_map: dict[str, str] = {
            column: next(
                (
                    column.replace(old, new, 1)
                    for old, new in renames.items()
                    if column.startswith(old)
                ),
                column,
            )
            for column in dataframe.columns
        }

        return dataframe.rename(columns=rename_map)

    def train_test_split(
        self,
        test_size: float | None = 0.2,
        random_state: int | None = None,
        shuffle: bool = True,
        stratify: ArrayLike | None = None,
    ) -> tuple[Self, Self]:
        """Splits the data into training and test sets.

        Scaling parameters are calculated from the training data and then applied unchanged to both
        the training and test sets.

        Args:
            test_size: Proportion of the dataset to include in the test split. Defaults to ``0.2``.
            random_state: Controls the shuffling applied to the data before applying the split.
                Defaults to ``None``.
            shuffle: Whether to shuffle the data before splitting. Defaults to ``True``.
            stratify: Target variable used to stratify the split. Defaults to ``None``.

        Returns:
            Tuple containing the training and test data containers
        """
        train_indices, test_indices = sklearn_train_test_split(
            self.values.index,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )

        train_values: pd.DataFrame = self.values.loc[train_indices]
        test_values: pd.DataFrame = self.values.loc[test_indices]

        train_uncertainties: pd.DataFrame = self.uncertainties.loc[train_indices]
        test_uncertainties: pd.DataFrame = self.uncertainties.loc[test_indices]

        train_metadata: pd.DataFrame = self.metadata.loc[train_indices]
        test_metadata: pd.DataFrame = self.metadata.loc[test_indices]

        # Train container calculates its own scaling parameters.
        train: Self = type(self)(
            values=train_values,
            uncertainties=train_uncertainties,
            metadata=train_metadata,
            name=f"{self.name}_train",
            uncertainty_scale=1.0,
            data_column=self.data_column,
        )

        # Test container uses the parameters learned from the training data.
        test: Self = type(self)(
            values=test_values,
            uncertainties=test_uncertainties,
            metadata=test_metadata,
            name=f"{self.name}_test",
            uncertainty_scale=1.0,
            scaling_means=train.scaling_means,
            scaling_stds=train.scaling_stds,
            data_column=self.data_column,
        )

        return train, test

    def get_dataframe(self) -> pd.DataFrame:
        """Returns the data as a combined dataframe.

        Metadata columns are followed by feature values and uncertainties. Feature values and
        uncertainties use a two-level column index with ``"Values"`` and ``"Uncertainties"`` as the
        top-level labels.

        Returns:
            Combined dataframe containing metadata, values, and uncertainties.
        """
        metadata: pd.DataFrame = self.metadata.copy()
        values: pd.DataFrame = self.values.copy()
        uncertainties: pd.DataFrame = self.uncertainties.copy()

        if len(metadata.columns) > 0:
            metadata.columns = pd.MultiIndex.from_product([["Metadata"], metadata.columns])

        values.columns = pd.MultiIndex.from_product([["Values"], values.columns])
        uncertainties.columns = pd.MultiIndex.from_product(
            [["Uncertainties"], uncertainties.columns]
        )

        return pd.concat([metadata, values, uncertainties], axis=1)

    def to_excel(self, filename_path: Path | str, *, sheet_name: str = "data") -> None:
        """Exports the data container to an Excel file.

        Metadata columns are written first, followed by feature values and uncertainties. Feature
        values and uncertainties use a two-level column index with ``"Values"`` and
        ``"Uncertainties"`` as the top-level labels.

        Args:
            filename_path: Path to the output Excel file.
            sheet_name: Name of the Excel worksheet. Defaults to ``"data"``.
        """
        dataframe: pd.DataFrame = self.get_dataframe()

        dataframe.to_excel(filename_path, sheet_name=sheet_name)
