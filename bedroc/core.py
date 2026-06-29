# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Core classes and functions"""

import logging
from collections.abc import Iterable
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, Self

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import xarray as xr
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
        select_features: Iterable[str] | None = None,
        select_data: Iterable[Any] | None = None,
        data_column: str = "ID",
    ):
        if select_features is not None:
            feature: tuple[str, ...] = tuple(select_features)

            cols = dataframe.columns

            # Rule 1: keep columns that are not features or feature standard deviations
            keep_non_suffix = ~cols.str.endswith((feature_suffix, feature_std_suffix))

            # Rule 2: keep columns that exactly match a feature name + suffix
            exact_columns = {f"{f}{feature_suffix}" for f in feature} | {
                f"{f}{feature_std_suffix}" for f in feature
            }
            keep_feature_and_suffix = cols.isin(exact_columns)

            # Select features based on the column suffix
            mask = keep_non_suffix | keep_feature_and_suffix

            dataframe = dataframe.loc[:, mask]

        if select_data is not None:
            data_tuple: tuple[Any, ...] = tuple(select_data)
            dataframe = dataframe[dataframe[data_column].isin(data_tuple)]

        # Always store an independent copy of the raw data internally
        self.df_raw: pd.DataFrame = dataframe.copy()

        # Rename uncertainty columns to a standard "_uncertainty" suffix
        # Uncertainty columns are treated as 1 sigma-type uncertainties
        unc_rename_map = {
            c: c.removesuffix(feature_std_suffix) + "_uncertainty"
            for c in self.df_raw.columns
            if c.endswith(feature_std_suffix)
        }
        self.df_raw = self.df_raw.rename(columns=unc_rename_map)

        self.name: str = name
        self.feature_suffix: str = feature_suffix
        # Set uncertainty suffix to the new standard
        self.feature_std_suffix: str = "_uncertainty"
        self.data_column: str = data_column

        # Cache column indices (df_raw columns are fixed after construction)
        self._feature_columns: pd.Index = self.df_raw.columns[
            self.df_raw.columns.str.endswith(self.feature_suffix)
        ]
        self._feature_std_columns: pd.Index = self.df_raw.columns[
            self.df_raw.columns.str.endswith(self.feature_std_suffix)
        ]

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
    def from_csv(cls, filename_path: str | Path, **kwargs) -> Self:
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
    def from_excel(cls, filename_path: str | Path, sheet_name: Any, **kwargs) -> Self:
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
        return self._feature_columns

    @property
    def feature_std_columns(self) -> pd.Index:
        """Index of feature uncertainty columns"""
        return self._feature_std_columns

    @property
    def feature_names(self) -> pd.Index:
        """Index of feature names with the suffix removed"""
        return self.feature_columns.str.removesuffix(self.feature_suffix)

    @property
    def n_data(self) -> int:
        """Number of data"""
        return len(self.df_raw)

    @property
    def n_features(self) -> int:
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
        scaling_stds_unc: pd.Series = self.scaling_stds.set_axis(
            self.scaling_stds.index.str.removesuffix(self.feature_suffix) + self.feature_std_suffix
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
        df = self.df_std if standardized else self.df_raw
        return df[self.feature_columns].to_numpy()

    def get_feature_stds(self, *, standardized: bool = True) -> NpFloat:
        """Returns standardized (default) or raw feature standard deviations

        Args:
            standardized: Whether to return standardized standard deviations. Defaults to ``True``.

        Returns:
            Feature standard deviations
        """
        df = self.df_std if standardized else self.df_raw
        return df[self.feature_std_columns].to_numpy()

    def get_covariance_matrix(self, *, standardized: bool = True) -> NpFloat:
        """Gets the covariance matrix.

        Args:
            standardized: Whether to use standardized feature values. Defaults to ``True``.

        Returns:
            Covariance matrix
        """
        covariance_matrix = np.cov(
            self.get_feature_values(standardized=standardized), rowvar=False, ddof=0
        )
        logger.debug("covariance_matrix = %s", covariance_matrix)

        return covariance_matrix  # type: ignore[return-value]

    def plot_pearson_correlation_coefficient(self, *, standardized: bool = True) -> Axes:
        """Plots a heatmap of the Pearson correlation coefficient.

        Args:
            standardized: Whether to use standardized feature values. Defaults to ``True``.

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


def plot_posterior_predictive(
    model: pm.Model, idata: xr.DataTree, *, thinning_factor: int = 5, **kwargs
) -> az.PlotCollection:
    """Plots posterior predictive check (in-sample predictions).

    This performs in-sample predictions to assess how well the model fits the observed data,
    i.e., test how well the model can reproduce the data it was trained on.

    Args:
        model: PyMC model object
        idata: Trace data from sampling
        thinning_factor: Thinning factor for posterior samples to reduce overplotting.
            Defaults to ``5``.
        **kwargs: Keyword arguments for :func:`pymc.sample_posterior_predictive`

    Returns:
        Plot collection
    """
    thinned_idata: xr.DataTree = idata.sel(draw=slice(None, None, thinning_factor))
    posterior_predictive: xr.DataTree = pm.sample_posterior_predictive(
        thinned_idata, model=model, **kwargs
    )
    collection: az.PlotCollection = az.plot_ppc_dist(
        posterior_predictive, group="posterior_predictive", kind="kde", observed=True
    )

    return collection


def plot_prior_predictive(model: pm.Model, **kwargs) -> az.PlotCollection:
    """Plots prior predictive check.

    This plot is used to determine if the model can generate data plausibly shaped like the
    observed distributions.

    Args:
        model: PyMC model object
        **kwargs: Keyword arguments for :func:`pymc.sample_prior_predictive`

    Returns:
        Plot collection
    """
    prior_predictive: xr.DataTree = pm.sample_prior_predictive(model=model, **kwargs)

    collection: az.PlotCollection = az.plot_ppc_dist(
        prior_predictive, group="prior", observed=True
    )

    return collection


def trim_samples(samples: NpArray) -> NpFloat:
    """Trims samples.

    Args:
        samples: Samples to trim

    Returns:
        Trimmed samples
    """
    # Define the percentage of extreme values to exclude from the hist plot
    # (e.g., 0.5% from each end)
    lower_percentile: float = 0.5
    upper_percentile: float = 99.5

    lower_limit: np.floating = np.percentile(samples, lower_percentile)
    upper_limit: np.floating = np.percentile(samples, upper_percentile)

    # Filter out the extreme values
    trimmed_samples: NpFloat = samples[(samples >= lower_limit) & (samples <= upper_limit)]

    return trimmed_samples


def resolve_path(p: Traversable | Path) -> Path:
    """Resolve a ``Traversable`` or ``Path`` to a concrete filesystem path.

    This function ensures that resources packaged using ``importlib.resources`` (e.g., files inside
    wheels or zipped packages) are converted into a real ``Path`` object. If ``p`` is already a
    ``Path``, it is returned unchanged. Otherwise, the underlying resource is extracted to a
    temporary location and its path is returned.

    Note:
        The temporary file extracted for ``Traversable`` objects is valid only for the duration of
        the context in which it is created. Since this function returns the resolved ``Path``
        inside the context manager, the file is guaranteed to exist when the function returns.

    Args:
        p: A filesystem ``Path`` or an ``importlib.resources.Traversable`` object.

    Returns:
        Path: A concrete filesystem path pointing to the resolved resource
    """
    if isinstance(p, Path):
        return p
    with resources.as_file(p) as temp:
        return Path(temp)
