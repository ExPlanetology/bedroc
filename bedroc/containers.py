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
from typing import Any, Optional, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure, SubFigure

logger: logging.Logger = logging.getLogger(__name__)

SUPTITLE_FONTSIZE: str = "xx-large"
"""Font size for the super title"""
savefig_opts: dict[str, Any] = {"dpi": 300, "bbox_inches": "tight", "format": "pdf"}
"""Figure options for savefig"""


class DataContainer:
    """A generic data container

    Args:
        dataframe: A dataframe with columns of feature values and their standard deviations
        name: Data container name. Defaults to ``data``.
        feature_suffix: Suffix of feature value columns. Defaults to ``_feature``.
        feature_std_suffix: Suffix of feature standard deviation columns. Defaults to
            ``_uncertainty``.
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

        self.feature_suffix: str = feature_suffix
        self.feature_std_suffix: str = feature_std_suffix

        # Scaling parameters computed from raw data
        self.scaling_means: pd.Series = self._compute_scaling_means()
        self.scaling_stds: pd.Series = self._compute_scaling_stds()

        # Precompute standardized data for speed
        self.df_std: pd.DataFrame = self._compute_standardized_data()

        logger.info("Number of data = %d", self.n_data)
        logger.info("Number of features = %d", self.n_features)
        logger.info("Feature names: %s", self.feature_names.values)

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
        return self.df_raw.columns[self.df_raw.columns.str.endswith(self.feature_suffix)]

    @property
    def feature_std_columns(self) -> pd.Index:
        """Index of feature uncertainty columns"""
        return self.df_raw.columns[self.df_raw.columns.str.endswith(self.feature_std_suffix)]

    @property
    def feature_names(self) -> pd.Index:
        """Index of feature names with the prefix removed"""
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

    def get_feature_values(self, *, standardized: bool = True) -> Any:
        """Returns standardized (default) or raw feature values

        Args:
            standardized: Whether to return standardized feature values. Defaults to ``True``.

        Returns:
            Feature values
        """
        return self.get_dataframe(standardized=standardized)[self.feature_columns].values

    def get_feature_stds(self, *, standardized: bool = True) -> Any:
        """Returns standardized (default) or raw feature standard deviations

        Args:
            standardized: Whether to return standardized standard deviations. Defaults to ``True``.

        Returns:
            Feature standard deviations
        """
        return self.get_dataframe(standardized=standardized)[self.feature_std_columns].values

    def get_covariance_matrix(self, *, standardized: bool = True) -> npt.NDArray:
        """Gets the covariance matrix.

        Args:
            standardized: Whether to return standardized standard deviations. Defaults to ``True``.

        Returns:
            Covariance matrix
        """
        covariance_matrix: npt.NDArray = np.cov(
            self.get_feature_values(standardized=standardized), rowvar=False, ddof=0
        )
        logger.debug("covariance_matrix = %s", covariance_matrix)

        return covariance_matrix

    def plot_pearson_correlation_coefficient(
        self,
        *,
        standardized: bool = True,
        savefig: bool = False,
        filename_prefix: Path | str = "pearson_correlation_coefficient",
    ) -> Figure | SubFigure:
        """Plots a heatmap of the Pearson correlation coefficient.

        Args:
            standardized: Whether to return standardized standard deviations. Defaults to ``True``.
            savefig: Saves the figure to a file. Defaults to ``False``.
            filename_prefix: Prefix for the saved figure filename. Defaults to
                "pearson_correlation_coefficient".

        Returns:
            The figure or subfigure containing the heatmap
        """
        # Covariance matrix
        corr_matrix: npt.NDArray = np.corrcoef(
            self.get_feature_values(standardized=standardized).T
        )

        ax = sns.heatmap(
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

        if savefig:
            ax.figure.savefig(  # pyright: ignore - not available for a SubFigure
                f"{filename_prefix}.{savefig_opts['format']}", **savefig_opts
            )  # pragma: no cover

        return ax.figure
