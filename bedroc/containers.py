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
from pathlib import Path

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

    def _compute_scaling_means(self) -> pd.Series:
        """Computes the feature means for scaling"""
        return self.df_raw[self.feature_columns].mean(axis=0)

    def _compute_scaling_stds(self) -> pd.Series:
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

    def get_feature_values(self, *, standardized: bool = True) -> pd.DataFrame:
        """Returns standardized (default) or raw feature values"""
        df: pd.DataFrame = self.df_std if standardized else self.df_raw
        return df[self.feature_columns].copy()

    def get_feature_stds(self, *, standardized: bool = True) -> pd.DataFrame:
        """Returns standardized (default) or raw feature standard deviations"""
        df: pd.DataFrame = self.df_std if standardized else self.df_raw
        return df[self.feature_std_labels].copy()

    def get_covariance_matrix(self) -> npt.NDArray:
        """Gets the covariance matrix.

        Returns:
            Covariance matrix
        """
        covariance_matrix: npt.NDArray = np.cov(
            self.get_feature_values().values, rowvar=False, ddof=0
        )
        logger.debug("covariance_matrix = %s", covariance_matrix)

        return covariance_matrix

    def plot_pearson_correlation_coefficient(self) -> Figure | SubFigure:
        """Plots a heatmap of the Pearson correlation coefficient.

        Returns:
            The figure or subfigure containing the heatmap
        """
        # Covariance matrix
        corr_matrix: npt.NDArray = np.corrcoef(self.get_feature_values().values.T)

        ax = sns.heatmap(
            corr_matrix,
            cmap="magma",
            annot=True,
            fmt=".2f",
            xticklabels=self.feature_names,
            yticklabels=self.feature_names,
            vmin=-1,
            vmax=1,
        )
        ax.set_title("Pearson correlation coefficient")

        return ax.figure
