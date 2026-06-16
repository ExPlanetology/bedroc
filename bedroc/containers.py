# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Containers"""

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from bedroc.type_aliases import NpArray, NpFloat

logger: logging.Logger = logging.getLogger(__name__)


class DataContainer:
    """A generic data container

    Args:
        dataframe: A dataframe with columns of feature values and their standard deviations
        name: Data container name. Defaults to ``data``.
        feature_suffix: Suffix of feature value columns. Defaults to ``_feature``.
        feature_std_suffix: Suffix of feature standard deviation columns. Defaults to
            ``_uncertainty``.
        std_scale: Number of standard deviations represented by the uncertainty columns.
            For example, use ``2.0`` if the input uncertainties are reported as 2SE. Defaults to
            ``1.0``.
        select_features: An optional iterable (tuple or list) of bare feature names (without
            ``feature_suffix``) to select. Defaults to ``None`` to select all features.
        select_data: An optional iterable (tuple or list) of data to select. Defaults to ``None``
            to select all data.
        data_column: Name of the data column used by ``select_data``. Defaults to ``ID``.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        name: str = "data",
        feature_suffix: str = "_feature",
        feature_std_suffix: str = "_uncertainty",
        std_scale: float = 1.0,
        select_features: Optional[Iterable[str]] = None,
        select_data: Optional[Iterable[Any]] = None,
        data_column: str = "ID",
    ):
        if select_features is not None:
            feature: tuple[str, ...] = tuple(select_features)

            cols = dataframe.columns

            # Rule 1: keep columns that are not features or feature standard deviations
            keep_non_suffix = ~cols.str.endswith((feature_suffix, feature_std_suffix))

            # Rule 2: keep columns that match a feature name
            keep_feature_and_suffix = cols.str.startswith(feature) & cols.str.endswith(
                (feature_suffix, feature_std_suffix)
            )

            # Select features based on the column suffix
            mask = keep_non_suffix | keep_feature_and_suffix

            dataframe = dataframe.loc[:, mask].copy()  # Avoid aliasing

        if select_data is not None:
            data_tuple: tuple[str, ...] = tuple(select_data)
            dataframe = cast(
                pd.DataFrame, dataframe[dataframe[data_column].isin(data_tuple)].copy()
            )  # Avoid aliasing

        # Always store an independent copy of the raw data internally
        self.df_raw: pd.DataFrame = dataframe.copy()

        # Rename uncertainty columns to a standard "_uncertainty" suffix
        # Uncertainty columns are treated as 1 sigma-type uncertainties
        unc_rename_map = {
            c: c.replace(feature_std_suffix, "_uncertainty")
            for c in self.df_raw.columns
            if c.endswith(feature_std_suffix)
        }
        self.df_raw = self.df_raw.rename(columns=unc_rename_map)

        self.name: str = name
        self.feature_suffix: str = feature_suffix
        # Set uncertainty suffix to the new standard
        self.feature_std_suffix: str = "_uncertainty"
        self.data_column: str = data_column

        # Apply uncertainty scaling once
        self.df_raw.loc[:, self.feature_std_columns] = (
            self.df_raw.loc[:, self.feature_std_columns] / std_scale
        )

        # Scaling parameters computed from raw data
        self.scaling_means: pd.Series = self._compute_scaling_means()
        self.scaling_stds: pd.Series = self._compute_scaling_stds()

        # Precompute standardized data for speed
        self.df_std: pd.DataFrame = self._compute_standardized_data()

        logger.info("Data container '%s' initialized", self.name)
        logger.info("Number of samples = %d", self.n_data)
        logger.info("Number of features = %d", self.n_features)
        logger.info("Feature names: %s", self.feature_names.values)

    @classmethod
    def from_csv(cls, filename_path: str | Path, **kwargs) -> "DataContainer":
        """Creates an instance from a CSV file.

        Args:
            filename_path: Path to the CSV file
            **kwargs: Arbitrary keyword arguments for constructor

        Returns:
            An instance
        """
        data: pd.DataFrame = pd.read_csv(filename_path)

        return cls(data, **kwargs)

    @classmethod
    def from_excel(cls, filename_path: str | Path, sheet_name: Any, **kwargs) -> "DataContainer":
        """Creates an instance from an Excel file.

        Args:
            filename_path: Path to the Excel file
            sheet_name: Sheet name
            **kwargs: Arbitrary keyword arguments for constructor

        Returns:
            An instance
        """
        data: pd.DataFrame = pd.read_excel(filename_path, sheet_name=sheet_name)

        return cls(data, **kwargs)

    @property
    def data_names(self) -> list[str]:
        """Data names"""
        return self.df_raw[self.data_column].to_list()

    @property
    def feature_columns(self) -> pd.Index:
        """Index of feature columns"""
        return self.df_raw.columns[self.df_raw.columns.str.endswith(self.feature_suffix)]

    @property
    def feature_std_columns(self) -> pd.Index:
        """Index of feature uncertainty columns"""
        return self.df_raw.columns[self.df_raw.columns.str.endswith(self.feature_std_suffix)]

    @property
    def feature_names(self) -> pd.Index:
        """Index of feature names with the suffix removed"""
        return self.df_raw[self.feature_columns].columns.str.removesuffix(self.feature_suffix)

    @property
    def n_data(self) -> int:
        """Number of data"""
        return len(self.df_raw)

    @property
    def n_features(self) -> int:
        """Number of features"""
        return len(self.feature_columns)

    def _compute_scaling_means(self) -> Any:
        """Computes the feature means for scaling"""
        return self.df_raw[self.feature_columns].mean(axis=0)

    def _compute_scaling_stds(self) -> Any:
        """Computes the feature standard deviations for scaling"""
        return self.df_raw[self.feature_columns].std(axis=0, ddof=0)

    def _compute_standardized_data(self) -> pd.DataFrame:
        """Computes standardized data"""
        df: pd.DataFrame = self.df_raw.copy()

        # Standardize feature values
        df[self.feature_columns] = (
            df[self.feature_columns] - self.scaling_means
        ) / self.scaling_stds

        # Standardize feature uncertainties
        # Rename the indices for correct broadcasting
        scaling_stds_unc: pd.Series = self.scaling_stds.copy()
        scaling_stds_unc.index = self.scaling_stds.index.str.replace(
            self.feature_suffix, self.feature_std_suffix
        )
        # Standardize feature standard deviations
        df[self.feature_std_columns] = df[self.feature_std_columns] / scaling_stds_unc

        return df

    def get_dataframe(self, *, standardized: bool = True) -> pd.DataFrame:
        """Returns standardized (default) or raw dataframe"""
        return self.df_std.copy() if standardized else self.df_raw.copy()

    def get_destandardized_values(self, standardized_values: NpFloat) -> NpFloat:
        """Gets destandardized values.

        Args:
            standardized_values: Standardized values. Must have a shape of:
                (n_data, n_features) or (n_data, n_features, n_samples)

        Returns:
            Destandardized values with matching shape
        """
        # Broadcast to (n_data, n_features, n_samples)
        stds: NpFloat = self.scaling_stds.to_numpy()[np.newaxis, :, np.newaxis]
        means: NpFloat = self.scaling_means.to_numpy()[np.newaxis, :, np.newaxis]

        # Broadcast input for the calculation
        dest: NpFloat = (
            standardized_values[..., np.newaxis]
            if standardized_values.ndim == 2
            else standardized_values
        )
        result: NpFloat = dest * stds + means

        # Squeeze the output to match the input dimensions
        return result.squeeze(-1) if standardized_values.ndim == 2 else result

    def get_feature_values(self, *, standardized: bool = True) -> NpFloat:
        """Returns standardized (default) or raw feature values

        Args:
            standardized: Whether to return standardized feature values. Defaults to ``True``.

        Returns:
            Feature values
        """
        return self.get_dataframe(standardized=standardized)[self.feature_columns].to_numpy()

    def get_feature_stds(self, *, standardized: bool = True) -> NpFloat:
        """Returns standardized (default) or raw feature standard deviations

        Args:
            standardized: Whether to return standardized standard deviations. Defaults to ``True``.

        Returns:
            Feature standard deviations
        """
        return self.get_dataframe(standardized=standardized)[self.feature_std_columns].to_numpy()

    def get_covariance_matrix(self, *, standardized: bool = True) -> NpFloat:
        """Gets the covariance matrix.

        Args:
            standardized: Whether to return standardized standard deviations. Defaults to ``True``.

        Returns:
            Covariance matrix
        """
        covariance_matrix: NpArray = np.cov(
            self.get_feature_values(standardized=standardized), rowvar=False, ddof=0
        )
        logger.debug("covariance_matrix = %s", covariance_matrix)

        return covariance_matrix

    def plot_pearson_correlation_coefficient(self, *, standardized: bool = True) -> Axes:
        """Plots a heatmap of the Pearson correlation coefficient.

        Args:
            standardized: Whether to process standardized standard deviations. Defaults to
                ``True``.

        Returns:
            Figure axes
        """
        # Covariance matrix
        corr_matrix = np.corrcoef(self.get_feature_values(standardized=standardized).T)

        ax: Axes = sns.heatmap(
            corr_matrix,
            cmap="magma",
            annot=True,
            fmt=".2f",
            xticklabels=self.feature_names.values,  # pyright: ignore - is a sequence
            yticklabels=self.feature_names.values,  # pyright: ignore - is a sequence
            vmin=-1,
            vmax=1,
        )
        ax.set_title("Pearson correlation coefficient")

        return ax
