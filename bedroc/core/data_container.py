# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Core classes and functions"""

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from bedroc.core.type_aliases import NpArray
from bedroc.core.utils import eigen_summary, pooled_within_category_covariance

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScalingParams:
    """Feature scaling statistics"""

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
    """Diagnostic analyses of a DataContainer's feature values"""

    data: "DataContainer"

    def covariance_matrix(self) -> pd.DataFrame:
        """Computes the covariance matrix of the standardized features, pooling all samples
        regardless of category.

        Numerically identical to the raw features' correlation matrix, since standardizing
        removes scale. Conflates within-category scatter with between-category mean spread when
        categories differ in mean — use :meth:`within_category_covariance_matrix` instead for
        :class:`~bedroc.difference.models.unified_covariance.UnifiedCovarianceModel`'s
        ``cov_shared`` or :class:`~bedroc.difference.group_synthetic.SyntheticDataGenerator`'s
        ``covariance`` in that case.

        Returns:
            Feature covariance matrix, indexed and labeled by feature name.
        """
        # ddof=0 matches values_std's own standardization, making this exactly equal to
        # values.corr() (unlike within_category_covariance_matrix's ddof=1, which unbiasedly
        # estimates an unknown population covariance rather than re-expressing already-
        # standardized data).
        return self.data.values_std.cov(ddof=0)

    def within_category_covariance_matrix(self) -> pd.DataFrame:
        """Computes the pooled within-category covariance matrix, across every distinct category
        present (rows with no category are excluded, same as :meth:`category_counts`).

        Unlike :meth:`covariance_matrix` (pools all samples, conflating within-category scatter
        with between-category mean spread), this is the standard pooled-within-group estimator
        (see :func:`~bedroc.core.utils.pooled_within_category_covariance`) — the correct choice
        for :class:`~bedroc.difference.models.unified_covariance.UnifiedCovarianceModel`'s
        ``cov_shared`` or :class:`~bedroc.difference.group_synthetic.SyntheticDataGenerator`'s
        ``covariance`` when categories have a real mean difference. Uses ``ddof=1`` (unbiased),
        unlike :meth:`covariance_matrix`'s ``ddof=0``.

        Raises:
            ValueError: If the container has no ``category_column`` set, or has fewer than two
                distinct categories present.

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
        groups: list[pd.DataFrame] = [
            values_std[codes == code] for code in sorted(codes.unique()) if code != -1
        ]

        return pooled_within_category_covariance(*groups)

    def category_mean_difference(self) -> pd.DataFrame:
        """Computes each category's standardized per-feature mean difference relative to
        category 0.

        Category 0's own row is included and is identically zero, making explicit which category
        is the reference. For a two-category container, the second row is directly usable as
        :class:`~bedroc.difference.group_synthetic.SyntheticDataGenerator`'s ``feature_offsets``.

        Raises:
            ValueError: If the container has no ``category_column`` set.

        Returns:
            One row per category (indexed by category name, in category order; category 0's row
            is all zeros), one column per feature.
        """
        if self.data.category_codes is None:
            raise ValueError(
                "category_mean_difference requires a DataContainer with category_column set."
            )

        category_names = self.data.category_names
        assert category_names is not None  # Guaranteed by the category_codes check above

        grouped_result = self.data.values_std.groupby(self.data.category_codes).mean()
        grouped: pd.DataFrame = grouped_result  # pyright: ignore[reportAssignmentType]
        mean_0: pd.Series = grouped.loc[0]  # pyright: ignore[reportAssignmentType]

        return pd.DataFrame(
            {category_names[code]: grouped.loc[code] - mean_0 for code in grouped.index}
        ).T

    def covariance_eigenanalysis(self) -> pd.DataFrame:
        """Eigendecomposes :meth:`within_category_covariance_matrix` to characterize directional
        separability.

        Eigenvectors are ordered from largest to smallest eigenvalue: for a fixed mean-shift
        budget, Mahalanobis distance (see :meth:`mahalanobis_alignment`) is hardest to achieve
        along the largest-eigenvalue direction and easiest along the smallest.

        Raises:
            ValueError: If the container has no ``category_column`` set (see
                :meth:`within_category_covariance_matrix`).

        Returns:
            One column per eigenvector (``PC1``, ``PC2``, ..., largest to smallest eigenvalue),
            one row per feature giving that feature's loading, plus trailing ``eigenvalue`` and
            explained-variance-ratio rows.
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

    def category_covariance_eigenanalysis(self) -> pd.DataFrame:
        """Eigendecomposes each category's own covariance matrix separately.

        Unlike :meth:`covariance_eigenanalysis` (which eigendecomposes the pooled, shared-across-
        categories matrix), this computes and eigendecomposes each category's own covariance
        independently — letting you check whether the shared-covariance assumption
        (:meth:`within_category_covariance_matrix`) actually holds, by comparing principal
        directions and eigenvalues across categories.

        Raises:
            ValueError: If the container has no ``category_column`` set.

        Returns:
            MultiIndex-columned dataframe: top-level columns are category names, second-level
            columns are eigenvectors (``PC1``, ``PC2``, ..., per category, largest to smallest
            eigenvalue) — slicing a single category (``result[category_name]``) reproduces
            :meth:`covariance_eigenanalysis`'s shape exactly. Rows are feature names plus trailing
            ``eigenvalue``/explained-variance-ratio rows.
        """
        if self.data.category_codes is None:
            raise ValueError(
                "category_covariance_eigenanalysis requires a DataContainer with category_column "
                "set."
            )

        values_std = self.data.values_std
        codes = self.data.category_codes
        category_names = self.data.category_names
        assert category_names is not None  # Guaranteed by the category_codes check above

        summaries: dict[str, pd.DataFrame] = {
            str(category_names[code]): eigen_summary(values_std[codes == code].cov(ddof=1))
            for code in sorted(codes.unique())
            if code != -1
        }

        return pd.concat(summaries, axis=1)

    def category_mahalanobis_alignment(self) -> pd.DataFrame:
        """Decomposes the Mahalanobis distance of every category from category 0, by principal
        direction.

        For each non-reference category, projecting its observed shift ``delta``
        (:meth:`category_mean_difference`) onto each eigenvector of
        :meth:`covariance_eigenanalysis` gives an exact additive decomposition of
        ``D^2 = delta^T Sigma^-1 delta``: ``D^2 = sum_k (v_k . delta)^2 / eigenvalue_k`` — showing
        whether the real separation rides on an easy (large-eigenvalue) or hard (small-eigenvalue)
        direction.

        Raises:
            ValueError: If the container has no ``category_column`` set, or has fewer than two
                distinct categories present (see :meth:`covariance_eigenanalysis`).

        Returns:
            MultiIndex-columned dataframe: top-level columns are category names, in category
            order (category 0's column is included, all zeros, except
            ``"fraction of mahalanobis_sq"`` which is ``NaN`` there — undefined, since there is no
            distance to decompose into fractions of), second-level columns are eigenvectors
            (matching :meth:`covariance_eigenanalysis`) — slicing a single category
            (``result[category_name]``) reproduces :meth:`mahalanobis_alignment`'s shape exactly
            (rows ``"shift projection"``, ``"eigenvalue"``, ``"mahalanobis_sq contribution"``,
            ``"fraction of mahalanobis_sq"``; ``"eigenvalue"`` is necessarily identical across
            categories, since every category's shift is decomposed against the same shared
            covariance eigenbasis).
        """
        eigen: pd.DataFrame = self.covariance_eigenanalysis()
        delta_df: pd.DataFrame = self.category_mean_difference()
        eigenvalues: pd.Series = eigen.loc["eigenvalue"]  # pyright: ignore[reportAssignmentType]

        alignments: dict[str, pd.DataFrame] = {}
        for category_name, delta in delta_df.iterrows():
            loadings: pd.DataFrame = eigen.loc[delta.index]
            projection: pd.Series = loadings.T.dot(delta)
            contribution: pd.Series = projection**2 / eigenvalues
            mahalanobis_sq_total: float = float(contribution.sum())

            logger.info(
                "Mahalanobis alignment for '%s' (%s vs category 0): D^2=%.4f, D=%.4f, "
                "fraction of D^2 by direction=%s",
                self.data.name,
                category_name,
                mahalanobis_sq_total,
                np.sqrt(mahalanobis_sq_total),
                np.round((contribution / mahalanobis_sq_total).to_numpy(), 4),
            )

            alignments[str(category_name)] = pd.DataFrame(
                {
                    "shift projection": projection,
                    "eigenvalue": eigenvalues,
                    "mahalanobis_sq contribution": contribution,
                    "fraction of mahalanobis_sq": contribution / mahalanobis_sq_total,
                }
            ).T

        return pd.concat(alignments, axis=1)

    def mahalanobis_alignment(self) -> pd.DataFrame:
        """Decomposes the Mahalanobis distance between category means by principal direction.

        A two-categories-only convenience wrapper around
        :meth:`category_mahalanobis_alignment` — see that method for the underlying decomposition
        and for containers with more than two categories.

        Raises:
            ValueError: If the container has no ``category_column`` set, or has other than
                exactly two categories.

        Returns:
            One column per eigenvector (matching :meth:`covariance_eigenanalysis`), and rows
            ``"shift projection"``, ``"eigenvalue"``, ``"mahalanobis_sq contribution"``, and
            ``"fraction of mahalanobis_sq"``.
        """
        alignments: pd.DataFrame = self.category_mahalanobis_alignment()
        category_names = alignments.columns.get_level_values(0).unique()
        if len(category_names) != 2:
            raise ValueError(
                "mahalanobis_alignment requires a DataContainer with exactly two categories, "
                f"got {len(category_names)}."
            )
        return alignments[category_names[1]]

    def run(self, *, output_directory: Path | str | None = None) -> dict[str, pd.DataFrame]:
        """Runs every diagnostic applicable to this container, optionally saving each to Excel.

        Tries each diagnostic in turn and skips (logging why) any that raise ``ValueError``.
        :meth:`covariance_matrix` has no category requirement and is always included; every other
        diagnostic requires ``category_column`` set with at least two distinct categories present,
        except :meth:`mahalanobis_alignment`, which additionally requires *exactly* two categories
        (see :meth:`category_mahalanobis_alignment` for the version that works for any number).

        Args:
            output_directory: Optional directory to save each included result to, as
                ``f"{self.data.name}_{key}.xlsx"`` (``key`` being the diagnostic's method name).
                If ``None``, results are only returned, not saved.

        Returns:
            Dict mapping each applicable diagnostic's method name to its result dataframe.
        """
        providers: dict[str, Callable[[], pd.DataFrame]] = {
            "covariance_matrix": self.covariance_matrix,
            "within_category_covariance_matrix": self.within_category_covariance_matrix,
            "category_mean_difference": self.category_mean_difference,
            "covariance_eigenanalysis": self.covariance_eigenanalysis,
            "category_covariance_eigenanalysis": self.category_covariance_eigenanalysis,
            "mahalanobis_alignment": self.mahalanobis_alignment,
            "category_mahalanobis_alignment": self.category_mahalanobis_alignment,
        }

        results: dict[str, pd.DataFrame] = {}
        for key, method in providers.items():
            try:
                results[key] = method()
            except ValueError as error:
                logger.info("Skipping '%s' diagnostic for '%s': %s", key, self.data.name, error)

        if output_directory is not None:
            output_directory = Path(output_directory)
            output_directory.mkdir(parents=True, exist_ok=True)
            for key, result in results.items():
                result.to_excel(output_directory / f"{self.data.name}_{key}.xlsx")

        return results

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
            col: pd.Series = self.metadata[  # pyright: ignore[reportAssignmentType]
                self.category_column
            ]
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
        return self.metadata[self.category_column]  # pyright: ignore[reportReturnType]

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
    def concat(
        cls,
        containers: Iterable[Self],
        *,
        name: str = "combined",
        source_index_column: str = "_source_index",
        source_name_column: str = "_source_name",
        **kwargs,
    ) -> Self:
        """Concatenates multiple data containers into a single one, row-wise.

        Every input must share the same feature columns (``values.columns``). Since each input
        container may independently use overlapping row labels (e.g. each started from its own
        0-based Excel row index), the combined container is given a fresh row index; each input's
        original index and :attr:`name` are preserved beforehand as new metadata columns (named by
        ``source_index_column``/``source_name_column``) so the original source of any row can still
        be recovered.

        Each input's :attr:`uncertainties` is already expressed in true 1-sigma units (it was
        divided by that container's own ``uncertainty_scale`` in its constructor), so the
        containers can be concatenated directly and the combined container defaults to
        ``uncertainty_scale=1.0``.

        A ``category_column`` metadata column, if present, is decategorized before concatenation
        so that per-container category universes don't clash; passing ``category_column`` in
        ``kwargs`` makes the combined container re-derive a single categorical universe from the
        union of all inputs.

        Args:
            containers: Data containers to concatenate, in order.
            name: Name for the combined container. Defaults to ``"combined"``.
            source_index_column: Metadata column recording each row's original index within its
                source container. Defaults to ``"_source_index"``.
            source_name_column: Metadata column recording each row's source container name.
                Defaults to ``"_source_name"``.
            **kwargs: Additional keyword arguments forwarded to the constructor (e.g.
                ``category_column``, ``select_data_column``). ``uncertainty_scale`` defaults to
                ``1.0`` unless overridden here.

        Raises:
            ValueError: If ``containers`` is empty, or the inputs don't share identical feature
                columns.

        Returns:
            A new data container holding all inputs' rows.
        """
        containers = list(containers)
        if not containers:
            raise ValueError("concat requires at least one DataContainer.")

        reference_columns: pd.Index = containers[0].values.columns
        for container in containers[1:]:
            if not container.values.columns.equals(reference_columns):
                raise ValueError(
                    "All containers must share the same feature columns to be concatenated "
                    f"(got {container.values.columns.tolist()!r} for '{container.name}', "
                    f"expected {reference_columns.tolist()!r})."
                )

        values_parts: list[pd.DataFrame] = []
        uncertainties_parts: list[pd.DataFrame] = []
        metadata_parts: list[pd.DataFrame] = []

        for container in containers:
            metadata: pd.DataFrame = container.metadata.copy()
            for col in metadata.columns:
                if isinstance(metadata[col].dtype, pd.CategoricalDtype):
                    metadata[col] = metadata[col].astype(object)

            metadata[source_index_column] = container.values.index
            metadata[source_name_column] = container.name

            values_parts.append(container.values)
            uncertainties_parts.append(container.uncertainties)
            metadata_parts.append(metadata)

        values: pd.DataFrame = pd.concat(values_parts, axis=0, ignore_index=True)
        uncertainties: pd.DataFrame = pd.concat(uncertainties_parts, axis=0, ignore_index=True)
        metadata_combined: pd.DataFrame = pd.concat(metadata_parts, axis=0, ignore_index=True)

        kwargs.setdefault("uncertainty_scale", 1.0)

        return cls(values, uncertainties, metadata_combined, name=name, **kwargs)

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
