#
# Copyright 2025 Dan J. Bower
#
# This file is part of Bedroc.
#
# Bedroc is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Bedroc is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Bedroc. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Containers"""

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure, SubFigure

logger: logging.Logger = logging.getLogger(__name__)


class DataContainer:
    """A generic data container

    Args:
        dataframe: A dataframe with columns of feature values and their standard deviations
        feature_prefix: Prefix for feature value columns. Defaults to ``Feat``.
        feature_std_prefix: Prefix of feature standard deviation columns. Defaults to ``Unc``.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        feature_prefix: str = "Feat",
        feature_std_prefix: str = "Unc",
    ):
        # Always store raw data internally
        self.df_raw: pd.DataFrame = dataframe.copy()

        self.feature_prefix: str = feature_prefix
        self.feature_std_prefix: str = feature_std_prefix

        # Scaling parameters computed from raw data
        self.scaling_means: pd.Series = self._compute_scaling_means()
        self.scaling_stds: pd.Series = self._compute_scaling_stds()

        # Precompute standardized data for speed
        self.df_std: pd.DataFrame = self._compute_standardized_data()

        logger.info("Number of data = %d", self.number_of_data)
        logger.info("Number of features = %d", self.number_of_features)

    @classmethod
    def from_csv(cls, filename_path: str | Path, **kwargs) -> "DataContainer":
        """Creates an instance from a CSV file

        Args:
            filename_path: Path to the CSV data
            **kwargs: Arbitrary keyword arguments for constructor

        Returns:
            An instance
        """
        data: pd.DataFrame = pd.read_csv(filename_path)

        return cls(data, **kwargs)

    @property
    def feature_columns(self) -> pd.Index:
        """Index of feature columns"""
        return self.df_raw.columns[self.df_raw.columns.str.startswith(self.feature_prefix)]

    @property
    def feature_std_labels(self) -> pd.Index:
        """Index of feature uncertainty columns"""
        return self.df_raw.columns[self.df_raw.columns.str.startswith(self.feature_std_prefix)]

    @property
    def feature_names(self) -> list[str]:
        """Feature names with the prefix removed"""
        feature_names: list[str] = [
            label.removeprefix(self.feature_prefix) for label in self.feature_columns.to_list()
        ]
        return feature_names

    @property
    def number_of_data(self) -> int:
        """Number of data"""
        return len(self.df_raw)

    @property
    def number_of_features(self) -> int:
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
            self.feature_prefix, self.feature_std_prefix
        )
        # Standardize feature standard deviations
        df[self.feature_std_labels] = df[self.feature_std_labels] / scaling_stds_unc

        return df

    def get_dataframe(self, *, standardized: bool = True) -> pd.DataFrame:
        """Returns standardized (default) or raw dataframe"""
        return self.df_std.copy() if standardized else self.df_raw.copy()

    def get_feature_values(
        self, *, standardized: bool = True, select: Iterable[str] | None = None
    ) -> Any:
        """Returns standardized (default) or raw feature values

        Args:
            standardized: Whether to return standardized feature values. Defaults to ``True``.
            select: An optional iterable of bare feature names (without prefix) to select. If
                ``None``, all features are returned. Defaults to ``None``.

        Returns:
            Feature values
        """
        df: pd.DataFrame = self.df_std if standardized else self.df_raw

        if select is None:
            cols = self.feature_columns
        else:
            cols = [f"{self.feature_prefix}{feat}" for feat in select]

        return df[cols].values

    def get_feature_stds(
        self, *, standardized: bool = True, select: Iterable[str] | None = None
    ) -> Any:
        """Returns standardized (default) or raw feature standard deviations

        Args:
            standardized: Whether to return standardized standard deviations. Defaults to ``True``.
            select: An optional iterable of bare standard deviation names (without prefix) to
                select. If ``None``, all features are returned. Defaults to ``None``.

        Returns:
            Feature standard deviations
        """
        df: pd.DataFrame = self.df_std if standardized else self.df_raw

        if select is None:
            cols = self.feature_columns
        else:
            cols = [f"{self.feature_std_prefix}{feat}" for feat in select]

        return df[cols].values

    def get_covariance_matrix(
        self, *, standardized: bool = True, select: Iterable[str] | None = None
    ) -> npt.NDArray:
        """Gets the covariance matrix.

        Args:
            standardized: Whether to return standardized standard deviations. Defaults to ``True``.
            select: An optional iterable of bare feature names (without prefix) to select. If
                ``None``, all features are returned. Defaults to ``None``.

        Returns:
            Covariance matrix
        """
        covariance_matrix: npt.NDArray = np.cov(
            self.get_feature_values(standardized=standardized, select=select), rowvar=False, ddof=0
        )
        logger.debug("covariance_matrix = %s", covariance_matrix)

        return covariance_matrix

    def plot_pearson_correlation_coefficient(
        self, *, standardized: bool = True, select: Iterable[str] | None = None
    ) -> Figure | SubFigure:
        """Plots a heatmap of the Pearson correlation coefficient.

        Args:
            standardized: Whether to return standardized standard deviations. Defaults to ``True``.
            select: An optional iterable of bare feature names (without prefix) to select. If
                ``None``, all features are returned. Defaults to ``None``.

        Returns:
            The figure or subfigure containing the heatmap
        """
        # Covariance matrix
        corr_matrix: npt.NDArray = np.corrcoef(
            self.get_feature_values(standardized=standardized, select=select).T
        )

        if select is None:
            feature_names: list[str] = self.feature_names
        else:
            feature_names = [col for col in select]

        ax = sns.heatmap(
            corr_matrix,
            cmap="magma",
            annot=True,
            fmt=".2f",
            xticklabels=feature_names,
            yticklabels=feature_names,
            vmin=-1,
            vmax=1,
        )
        ax.set_title("Pearson correlation coefficient")

        return ax.figure
