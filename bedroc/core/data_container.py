# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Core classes and functions"""

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from numpy.typing import ArrayLike
from sklearn.model_selection import train_test_split as sklearn_train_test_split

from bedroc.core.type_aliases import NpArray
from bedroc.core.utils import eigen_summary

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScalingParams:
    """Holds feature scaling statistics."""

    means: pd.Series
    stds: pd.Series

    def transform(self, values: pd.DataFrame) -> pd.DataFrame:
        """Standardizes the feature values using the stored scaling parameters."""
        return (values - self.means) / self.stds

    def inverse_transform(self, std_values: NpArray) -> NpArray:
        """Destandardizes the feature values using the stored scaling parameters."""
        if std_values.ndim not in (2, 3):
            raise ValueError("std_values must have 2 or 3 dimensions.")
        means = self.means.to_numpy()[np.newaxis, :, np.newaxis]
        stds = self.stds.to_numpy()[np.newaxis, :, np.newaxis]

        if std_values.ndim == 2:
            return (std_values[..., np.newaxis] * stds + means)[..., 0]
        return std_values * stds + means

    def align_to(self, columns: pd.Index) -> "ScalingParams":
        """Aligns means and stds to the target feature columns."""
        aligned_means = self.means.reindex(columns)
        aligned_stds = self.stds.reindex(columns)

        if aligned_means.isna().any() or aligned_stds.isna().any():
            missing = columns[aligned_means.isna()].tolist()
            raise ValueError(f"Missing scaling parameters for features: {missing}")

        return ScalingParams(means=aligned_means, stds=aligned_stds)


@dataclass(frozen=True)
class DataDiagnostics:
    """Diagnostic analyses of a DataContainer's feature values."""

    data: "DataContainer"

    def covariance_matrix(self) -> pd.DataFrame:
        """Computes the covariance matrix of the standardized features (:attr:`DataContainer.values_std`),
        pooling all samples regardless of category.

        Since standardizing removes each feature's scale, this is numerically identical to the
        *correlation* matrix of the raw features (:meth:`plot_correlation_coefficient`'s Pearson
        case) — but is computed and framed here as a covariance.

        Note:
            If the container has a real category mean difference (see
            :meth:`category_mean_difference`), this pooled-across-categories covariance conflates
            within-category scatter with the between-category mean spread, and so is *not* a
            reliable stand-in for the shared covariance assumed by
            :class:`~bedroc.difference.models.unified_covariance.UnifiedCovarianceModel`
            (``cov_shared``) or for the ``covariance`` argument of
            :class:`~bedroc.difference.group_synthetic.SyntheticDataGenerator` in that case — use
            :meth:`within_category_covariance_matrix` instead.

        Returns:
            Feature covariance matrix of the standardized features, indexed and labeled by
            feature name
        """
        # ddof=0 matches the population standard deviation used by _fit_scaling() to standardize
        # values_std in the first place; this is what makes the result exactly equal to
        # values.corr() (which is itself ddof-invariant), rather than off by a factor of
        # n / (n - 1).
        return self.data.values_std.cov(ddof=0)

    def within_category_covariance_matrix(self) -> pd.DataFrame:
        """Computes the pooled within-category covariance matrix of the standardized features.

        Unlike :meth:`covariance_matrix` (which pools *all* samples regardless of category, and so
        conflates within-category scatter with the between-category mean difference whenever the
        two categories differ in mean), this computes the standard pooled-within-group estimator
        ``((n0-1)*Cov0 + (n1-1)*Cov1) / (n0+n1-2)`` — the quantity that actually matches the
        shared-covariance assumption of
        :class:`~bedroc.difference.models.unified_covariance.UnifiedCovarianceModel`
        (``cov_shared``) when the two categories have a real mean difference, and so is the
        correct choice for the ``covariance`` argument of
        :class:`~bedroc.difference.group_synthetic.SyntheticDataGenerator` in that case.

        Raises:
            ValueError: If the container has no ``category_column`` set.

        Returns:
            Pooled within-category covariance matrix, indexed and labeled by feature name.
        """
        if self.data.category_codes is None:
            raise ValueError(
                "within_category_covariance_matrix requires a DataContainer with category_column "
                "set."
            )

        values_std = self.data.values_std
        codes = self.data.category_codes
        group_0 = values_std[codes == 0]
        group_1 = values_std[codes == 1]
        n_0, n_1 = len(group_0), len(group_1)

        return ((n_0 - 1) * group_0.cov(ddof=1) + (n_1 - 1) * group_1.cov(ddof=1)) / (
            n_0 + n_1 - 2
        )

    def category_mean_difference(self) -> pd.Series:
        """Computes the standardized per-feature mean difference between the two categories.

        Returns category 1's mean minus category 0's mean, in the same standardized units as
        :attr:`DataContainer.values_std` — directly usable as the ``feature_offsets`` argument of
        :class:`~bedroc.difference.group_synthetic.SyntheticDataGenerator`, which uses the
        identical convention (category 1's mean offset from category 0's).

        Raises:
            ValueError: If the container has no ``category_column`` set.

        Returns:
            Per-feature standardized mean difference, indexed by feature name.
        """
        if self.data.category_codes is None:
            raise ValueError(
                "category_mean_difference requires a DataContainer with category_column set."
            )

        grouped: pd.DataFrame = self.data.values_std.groupby(self.data.category_codes).mean()
        mean_0: pd.Series = grouped.loc[0]  # pyright: ignore[reportAssignmentType]
        mean_1: pd.Series = grouped.loc[1]  # pyright: ignore[reportAssignmentType]

        return mean_1 - mean_0

    def covariance_eigenanalysis(self) -> pd.DataFrame:
        """Eigendecomposes :meth:`within_category_covariance_matrix` to characterize directional
        separability.

        For a fixed per-feature mean-shift budget, the Mahalanobis distance between two category
        means (``D^2 = delta^T Sigma^-1 delta``, as used by
        :func:`~bedroc.difference.group_synthetic.demo_correlation_alignment`) is maximized when
        the shift direction ``delta`` aligns with the smallest-eigenvalue eigenvector of the
        covariance matrix, and minimized when it aligns with the largest. Eigenvectors are
        therefore ordered from largest to smallest eigenvalue: the leading columns are the
        hardest directions to separate along, and the trailing columns are the easiest.

        This decomposes the *within-category* covariance rather than :meth:`covariance_matrix`,
        since the Mahalanobis-alignment interpretation above only holds for the shared covariance
        the category means are actually offset against — the pooled-across-categories covariance
        conflates that with the between-category mean spread itself (see
        :meth:`within_category_covariance_matrix`).

        Raises:
            ValueError: If the container has no ``category_column`` set (see
                :meth:`within_category_covariance_matrix`).

        Returns:
            Summary dataframe with one column per eigenvector (labeled ``PC1``, ``PC2``, ...,
            ordered from largest to smallest eigenvalue), one row per feature giving that
            feature's loading, and two trailing rows giving each eigenvector's eigenvalue and
            explained-variance ratio.
        """
        summary: pd.DataFrame = eigen_summary(self.within_category_covariance_matrix())
        eigenvalues: NpArray = summary.loc["eigenvalue"].to_numpy()

        logger.info(
            "Covariance eigenanalysis for '%s': eigenvalues=%s, condition number=%.4g",
            self.data.name,
            np.round(eigenvalues, 4),
            eigenvalues[0] / eigenvalues[-1],
        )

        return summary

    def mahalanobis_alignment(self) -> pd.DataFrame:
        """Decomposes the Mahalanobis distance between category means by principal direction.

        Since ``D^2 = delta^T Sigma^-1 delta`` and ``Sigma = V @ diag(eigenvalues) @ V.T`` (with
        ``V`` the orthonormal eigenvectors from :meth:`covariance_eigenanalysis`), projecting the
        real observed shift ``delta`` (:meth:`category_mean_difference`) onto each eigenvector
        ``v_k`` gives an exact, additive decomposition: ``D^2 = sum_k (v_k . delta)^2 /
        eigenvalue_k``. This answers whether the real category separation rides mostly on an easy
        direction (small eigenvalue, i.e. low within-category variance) or a hard one (large
        eigenvalue) — the same question
        :func:`~bedroc.difference.group_synthetic.demo_correlation_alignment` explores for
        candidate synthetic shift directions, but computed here for the real, observed shift.

        Raises:
            ValueError: If the container has no ``category_column`` set (see
                :meth:`category_mean_difference`).

        Returns:
            Summary dataframe with one column per eigenvector (labeled ``PC1``, ``PC2``, ...,
            ordered from largest to smallest eigenvalue, matching :meth:`covariance_eigenanalysis`)
            and rows ``"shift projection"`` (signed projection of the real shift onto that
            eigenvector), ``"eigenvalue"``, ``"mahalanobis_sq contribution"`` (that direction's
            additive contribution to ``D^2``), and ``"fraction of mahalanobis_sq"`` (the same,
            normalized to sum to 1).
        """
        eigen: pd.DataFrame = self.covariance_eigenanalysis()
        delta: pd.Series = self.category_mean_difference()

        loadings: pd.DataFrame = eigen.loc[delta.index]
        eigenvalues: pd.Series = eigen.loc["eigenvalue"]  # pyright: ignore[reportAssignmentType]

        projection: pd.Series = loadings.T.dot(delta)
        contribution: pd.Series = projection**2 / eigenvalues
        mahalanobis_sq_total: float = float(contribution.sum())

        logger.info(
            "Mahalanobis alignment for '%s': D^2=%.4f, D=%.4f, fraction of D^2 by direction=%s",
            self.data.name,
            mahalanobis_sq_total,
            np.sqrt(mahalanobis_sq_total),
            np.round((contribution / mahalanobis_sq_total).to_numpy(), 4),
        )

        return pd.DataFrame(
            {
                "shift projection": projection,
                "eigenvalue": eigenvalues,
                "mahalanobis_sq contribution": contribution,
                "fraction of mahalanobis_sq": contribution / mahalanobis_sq_total,
            }
        ).T

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
        # equivalent to np.correcoef(self.data.values, rowvar=False)
        corr_matrix: pd.DataFrame = self.data.values.corr(method, min_periods, numeric_only)
        ax: Axes = sns.heatmap(
            corr_matrix, cmap="coolwarm", annot=True, fmt=".2f", vmin=-1, vmax=1
        )
        ax.set_title(f"{method.capitalize()} correlation coefficient")

        return ax


class DataContainer:
    """Container for feature values, measurement uncertainties, and metadata.

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
        scaling_params: Optional scaling parameters to use for standardizing the feature values.
            If provided, these parameters will be used instead of calculating new ones from the
            data. Defaults to ``None``.
        select_features: Optional iterable of feature names to retain. Defaults to ``None``, which
            retains all features.
        select_data: Optional iterable of values used to select samples (rows) based on
            ``select_data_column``. Defaults to ``None``, which retains all samples.
        select_data_column: Name of the metadata column used by ``select_data``. Defaults to
            ``"ID"``.
        category_column: Optional name of the metadata column that identifies the category of each
            sample. Defaults to ``None``.
    """

    def __init__(
        self,
        values: pd.DataFrame,
        uncertainties: pd.DataFrame | None = None,
        metadata: pd.DataFrame | None = None,
        *,
        name: str = "data",
        uncertainty_scale: float = 1.0,
        scaling_params: ScalingParams | None = None,
        select_features: Iterable[str] | None = None,
        select_data: Iterable[Any] | None = None,
        select_data_column: str = "ID",
        category_column: str | None = None,
    ):
        self.name: str = name
        self.select_data_column: str = select_data_column
        self.category_column: str | None = category_column
        self.uncertainty_scale: float = uncertainty_scale

        self._validate_raw_inputs(values, uncertainties, metadata, category_column)

        # 1. Store base copies
        self.values: pd.DataFrame = values.copy()

        self.uncertainties: pd.DataFrame = (
            uncertainties.copy() / uncertainty_scale
            if uncertainties is not None
            else pd.DataFrame(np.nan, index=self.values.index, columns=self.values.columns)
        )

        self.metadata: pd.DataFrame = (
            metadata.copy() if metadata is not None else pd.DataFrame(index=self.values.index)
        )

        # 2. Lock categorical universe BEFORE row selections are applied
        if self.category_column:
            col: pd.Series = self.metadata[self.category_column]
            # If already categorical, keep its exact category universe (even if counts are 0). This
            # ensures that train and test splits have the same category universe.
            if not isinstance(col.dtype, pd.CategoricalDtype):
                cat_names = sorted(col.dropna().unique())
                cat_type = pd.CategoricalDtype(categories=cat_names, ordered=True)
                self.metadata[self.category_column] = col.astype(cat_type)

        # 3. Filter rows and columns
        self._apply_selections(select_features, select_data)

        # 4. Fit or apply scaling parameters
        if scaling_params is not None:
            self.scaling = scaling_params.align_to(self.values.columns)
        else:
            self.scaling = self._fit_scaling()
        self._validate_scaling()

        # 5. Derive standardized views
        self.values_std: pd.DataFrame = self.scaling.transform(self.values)
        self.uncertainties_std = self.uncertainties / self.scaling.stds

        # 6. Diagnostics namespace (built last, after values/values_std are finalized)
        self.diagnostics: DataDiagnostics = DataDiagnostics(self)

        logger.info(
            "Data container '%s' initialized (%d samples, %d features)",
            self.name,
            self.n_data,
            self.n_features,
        )

    @property
    def scaling_means(self) -> pd.Series:
        return self.scaling.means

    @property
    def scaling_stds(self) -> pd.Series:
        return self.scaling.stds

    @property
    def categories(self) -> pd.Series | None:
        """Returns the series of category labels for each sample."""
        if not self.category_column:
            return None
        return self.metadata[self.category_column]

    @property
    def category_codes(self) -> pd.Series | None:
        """0-indexed integer encoding for statistical models (e.g., PyMC)."""
        if self.categories is None:
            return None
        return self.categories.cat.codes

    @property
    def category_names(self) -> pd.Index | None:
        """List of distinct category names corresponding to codes."""
        if self.categories is None:
            return None
        return self.categories.cat.categories

    @property
    def category_counts(self) -> pd.Series | None:
        """Returns the sample counts for each category if category_column is set."""
        if self.category_column is None or self.categories is None:
            return None

        # sort=False ensures order matches category_names, not count frequency
        counts = self.categories.value_counts(sort=False)

        if len(counts) > 2:
            logger.warning(
                "Expected binary categories in '%s', but found %d categories: %s",
                self.category_column,
                len(counts),
                counts.index.tolist(),
            )

        logger.info(
            "Category counts for '%s': %s", self.category_column, pformat(counts.to_dict())
        )

        return counts

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
        return self.metadata[self.select_data_column].to_list()

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
            feature_suffix: Suffix of feature value columns. Defaults to ``"_feature"``.
            uncertainty_suffix: Suffix of feature uncertainty columns. Defaults to
                ``"_uncertainty"``.
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

    def get_destandardized_values(self, standardized_values: NpArray) -> NpArray:
        return self.scaling.inverse_transform(standardized_values)

    def _fit_scaling(self) -> ScalingParams:
        means = self.values.mean(axis=0)
        stds = self.values.std(axis=0, ddof=0)
        return ScalingParams(means=means, stds=stds)

    def _apply_selections(
        self, select_features: Iterable[str] | None, select_data: Iterable[Any] | None
    ) -> None:
        if select_features is not None:
            features = list(select_features)
            self.values = self.values.loc[:, features]
            self.uncertainties = self.uncertainties.loc[:, features]

        if select_data is not None:
            mask = self.metadata[self.select_data_column].isin(list(select_data))
            self.values = self.values.loc[mask]
            self.uncertainties = self.uncertainties.loc[mask]
            self.metadata = self.metadata.loc[mask]

        if self.n_data == 0:
            raise ValueError("No data remain after selection")

    def _validate_raw_inputs(
        self,
        values: pd.DataFrame,
        uncertainties: pd.DataFrame | None,
        metadata: pd.DataFrame | None,
        category_column: str | None,
    ) -> None:
        if not values.columns.is_unique:
            raise ValueError("`values` must have unique feature names")

        if uncertainties is not None:
            if not values.index.equals(uncertainties.index) or not values.columns.equals(
                uncertainties.columns
            ):
                raise ValueError("`values` and `uncertainties` index/columns must match")

        if metadata is not None:
            if not values.index.equals(metadata.index):
                raise ValueError("`values` and `metadata` indices must match")
            if category_column and category_column not in metadata.columns:
                raise ValueError(f"`category_column` {category_column!r} not found in metadata")

    def _validate_scaling(self) -> None:
        if not np.isfinite(self.scaling.means).all():
            raise ValueError("Scaling means must be finite")
        if not np.isfinite(self.scaling.stds).all() or (self.scaling.stds <= 0).any():
            raise ValueError("Scaling standard deviations must be finite and positive")

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
        # Default to stratifying on categories if stratify isn't explicitly passed
        if stratify is None and self.category_column is not None:
            stratify = self.categories

        train_idx, test_idx = sklearn_train_test_split(
            self.values.index,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )

        # Train container calculates its own scaling parameters
        train: Self = type(self)(
            values=self.values.loc[train_idx],
            uncertainties=self.uncertainties.loc[train_idx],
            metadata=self.metadata.loc[train_idx],
            name=f"{self.name}_train",
            uncertainty_scale=1.0,  # Crucial: Already converted to 1-sigma
            select_data_column=self.select_data_column,
            category_column=self.category_column,
        )

        # Test container uses the parameters learned from the training data
        test: Self = type(self)(
            values=self.values.loc[test_idx],
            uncertainties=self.uncertainties.loc[test_idx],
            metadata=self.metadata.loc[test_idx],
            name=f"{self.name}_test",
            uncertainty_scale=1.0,  # Crucial: Already converted to 1-sigma
            scaling_params=train.scaling,  # Pass learned scaling directly
            select_data_column=self.select_data_column,
            category_column=self.category_column,
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
