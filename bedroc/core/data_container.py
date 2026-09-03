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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse

from bedroc.core.plotting import get_figure, save_figure
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
        means: NpArray = self.means.to_numpy()[np.newaxis, :, np.newaxis]
        stds: NpArray = self.stds.to_numpy()[np.newaxis, :, np.newaxis]

        if std_values.ndim == 2:
            return (std_values[..., np.newaxis] * stds + means)[..., 0]
        return std_values * stds + means

    def align_to(self, columns: pd.Index) -> "ScalingParams":
        """Aligns means and stds to the target feature columns."""
        aligned_means: pd.Series = self.means.reindex(columns)
        aligned_stds: pd.Series = self.stds.reindex(columns)

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
        # values.corr() -- within_category_covariance_matrix uses ddof=1 instead, since that one
        # estimates an unknown population covariance rather than just re-expressing this data.
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

        Projecting each category's shift ``delta`` (:meth:`category_mean_difference`) onto each
        eigenvector of :meth:`covariance_eigenanalysis` gives an exact additive decomposition:
        ``D^2 = delta^T Sigma^-1 delta = sum_k (v_k . delta)^2 / eigenvalue_k`` — showing whether
        the separation rides on an easy (large-eigenvalue) or hard (small-eigenvalue) direction.

        Raises:
            ValueError: If the container has no ``category_column`` set, or has fewer than two
                distinct categories present (see :meth:`covariance_eigenanalysis`).

        Returns:
            MultiIndex-columned dataframe: top-level columns are category names in category
            order, second-level columns are eigenvectors (matching
            :meth:`covariance_eigenanalysis`) — slicing a single category
            (``result[category_name]``) reproduces :meth:`mahalanobis_alignment`'s shape exactly
            (rows ``"shift projection"``, ``"eigenvalue"``, ``"mahalanobis_sq contribution"``,
            ``"fraction of mahalanobis_sq"``; ``"eigenvalue"`` is identical across categories,
            since every shift is decomposed against the same shared eigenbasis). Category 0's
            column is all zeros, with ``"fraction of mahalanobis_sq"`` as ``NaN`` there (``0/0``,
            undefined — there's no distance to decompose into fractions of).
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
        category_names: pd.Index = alignments.columns.get_level_values(0).unique()
        if len(category_names) != 2:
            raise ValueError(
                "mahalanobis_alignment requires a DataContainer with exactly two categories, "
                f"got {len(category_names)}."
            )
        return alignments[category_names[1]]

    def pca_biplot(
        self,
        *,
        pc_x: int = 1,
        pc_y: int = 2,
        figsize: tuple[float, float] = (8, 8),
        loading_scale: float | None = None,
    ) -> Axes:
        """Plots a PCA biplot: sample scores on two chosen PCs, feature loadings as arrows, and
        each category's mean shift as an arrow.

        Combines :meth:`covariance_eigenanalysis` (the noise: within-category spread along each
        axis) with :meth:`category_mahalanobis_alignment` (the signal: each category's mean shift,
        projected onto the same axes) in one plot, visualizing the signal-to-noise trade-off for
        the two directions shown. Each shift arrow is labeled with the fraction of that category's
        total Mahalanobis distance squared captured by these two components alone.

        Args:
            pc_x: 1-indexed component number for the horizontal axis. Defaults to ``1``.
            pc_y: 1-indexed component number for the vertical axis. Defaults to ``2``.
            figsize: Figure size. Defaults to ``(8, 8)``.
            loading_scale: Extra multiplier stretching the loading arrows (and the reference
                ellipse alongside them) for legibility, since their natural, absolute scale (see
                below) can otherwise draw too small next to the score scatter. Defaults to
                ``None``, which auto-scales so the reference ellipse fills most of the plot's
                data-derived extent (the sample scores and shift arrows, which are never
                rescaled themselves and so fix that extent); pass an explicit float to override
                it, or ``1.0`` for the true, unscaled reading. Shown on the plot itself whenever
                the resolved value isn't ``1.0``, since it's the only thing not on the axis's
                true scale.

        Raises:
            ValueError: If the container has no ``category_column`` set, has fewer than two
                distinct categories present, has fewer than two features, or ``pc_x``/``pc_y``
                are equal or out of range (``1`` to the number of features).

        Returns:
            Figure axes.
        """
        if self.data.n_features < 2:
            raise ValueError("pca_biplot requires a DataContainer with at least two features.")
        if pc_x == pc_y or not (
            1 <= pc_x <= self.data.n_features and 1 <= pc_y <= self.data.n_features
        ):
            raise ValueError(
                f"pc_x and pc_y must be distinct, between 1 and {self.data.n_features} "
                f"(the number of features), got pc_x={pc_x}, pc_y={pc_y}."
            )
        x_label, y_label = f"PC{pc_x}", f"PC{pc_y}"

        eigen: pd.DataFrame = self.covariance_eigenanalysis()
        alignment: pd.DataFrame = self.category_mahalanobis_alignment()

        feature_names: pd.Index = self.data.feature_names
        loadings: pd.DataFrame = eigen.loc[feature_names, [x_label, y_label]]
        eigenvalues: pd.Series = eigen.loc[  # pyright: ignore[reportAssignmentType]
            "eigenvalue", [x_label, y_label]
        ]
        explained: pd.Series = eigen.loc[  # pyright: ignore[reportAssignmentType]
            "explained variance ratio", [x_label, y_label]
        ]

        scores: NpArray = self.data.values_std[feature_names].to_numpy() @ loadings.to_numpy()

        _, ax = plt.subplots(figsize=figsize)

        codes = self.data.category_codes
        category_names = self.data.category_names
        assert codes is not None and category_names is not None  # Guaranteed by the calls above
        for code, category in enumerate(category_names):
            mask: NpArray = (codes == code).to_numpy()
            ax.scatter(scores[mask, 0], scores[mask, 1], label=str(category), alpha=0.5, s=20)
        ax.legend(title=self.data.category_column, loc="best", fontsize=9)

        # Each category's shift arrow (below) is plotted at its true, unscaled "shift projection"
        # -- it's a real data quantity like the scores, not a bounded correlation like a loading,
        # so it's never touched by loading_scale and helps fix the plot's extent below instead.
        non_reference_categories = alignment.columns.get_level_values(0).unique()[1:]
        shift_projections: dict[str, pd.Series] = {
            str(category): alignment[category].loc[  # pyright: ignore[reportAssignmentType]
                "shift projection", [x_label, y_label]
            ]
            for category in non_reference_categories
        }

        # Fix the plot's axis limit from the data alone (scores and shifts, with a fixed margin)
        # before touching the loadings at all, and never revisit it based on where the loadings
        # land -- if the limit were instead re-derived as max(data, loadings) afterwards, inflating
        # the loadings would just inflate the limit right along with them, capping any target
        # fraction of the frame at 1/margin (e.g. ~87% for a 1.15 margin) no matter how large a
        # fraction was actually requested below.
        shift_extents = (
            float(np.abs(shift.to_numpy(dtype=float)).max())
            for shift in shift_projections.values()
        )
        data_extent: float = max(float(np.abs(scores).max()), *shift_extents)
        axis_limit: float = data_extent * 1.15
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)

        # Scale each PC's loading by sqrt(eigenvalue) (the standard correlation-biplot convention).
        # A loading's own length is then a correlation (bounded by 1 once every PC is included),
        # and its magnitude is naturally on the order of one score standard deviation along that
        # axis -- so, unlike a raw eigenvector component, it can share one literal coordinate
        # system with the scores with no extra cosmetic rescale, and still carry an absolute
        # reading. If loading_scale isn't given explicitly, it's instead solved for here: the
        # reference ellipse below (semi-axes sqrt(eigenvalue) per component before this multiplier)
        # is, by definition, the biggest a loading can ever get -- so picking loading_scale to put
        # the ellipse's larger semi-axis at exactly 90% of the fixed axis_limit above means the
        # loading picture reliably fills most of the frame, without guessing a fixed number.
        if loading_scale is None:
            max_eigenvalue: float = max(float(eigenvalues[x_label]), float(eigenvalues[y_label]))
            loading_scale = 0.9 * axis_limit / np.sqrt(max_eigenvalue)
        display_loadings: pd.DataFrame = loadings * np.sqrt(eigenvalues) * loading_scale

        # loading_scale is solved above to keep the loadings within axis_limit already; if it was
        # instead given explicitly and pushes them further out, expand here so nothing gets
        # clipped -- a defensive fallback, not a rescale that would affect the case above.
        loadings_extent: float = float(np.abs(display_loadings.to_numpy()).max())
        if loadings_extent > axis_limit:
            extent: float = loadings_extent * 1.05
            ax.set_xlim(-extent, extent)
            ax.set_ylim(-extent, extent)
        ax.set_aspect("equal")

        # A feature fully captured by these two components alone (zero loading on every other
        # component) lands exactly on this ellipse -- semi-axes sqrt(eigenvalue) per component,
        # since a unit vector confined to this plane maps, under that same scaling, to an ellipse
        # rather than a circle whenever the two eigenvalues differ. How far short of it an arrow
        # falls is directly readable as how much of that feature these two axes miss.
        ax.add_patch(
            Ellipse(
                (0, 0),
                2 * np.sqrt(eigenvalues[x_label]) * loading_scale,
                2 * np.sqrt(eigenvalues[y_label]) * loading_scale,
                fill=False,
                linestyle="--",
                edgecolor="0.75",
                linewidth=1,
            )
        )
        ax.text(
            0,
            np.sqrt(eigenvalues[y_label]) * loading_scale,
            f"{x_label}+{y_label} fully explain feature  ",
            color="0.6",
            fontsize=8,
            ha="right",
            va="bottom",
        )

        for feature in feature_names:
            x, y = display_loadings.loc[feature]
            ax.annotate(
                "",
                xy=(x, y),
                xytext=(0, 0),
                arrowprops={"arrowstyle": "->", "color": "0.3", "lw": 1.2},
            )
            ax.text(x * 1.08, y * 1.08, feature, color="0.2", ha="center", va="center", fontsize=9)

        for category in non_reference_categories:
            shift: pd.Series = shift_projections[str(category)]
            fraction: float = float(
                alignment[category].loc["fraction of mahalanobis_sq", [x_label, y_label]].sum()
            )
            x, y = float(shift[x_label]), float(shift[y_label])
            ax.annotate(
                "",
                xy=(x, y),
                xytext=(0, 0),
                arrowprops={"arrowstyle": "-|>", "color": "crimson", "lw": 2.5},
            )
            ax.text(
                x,
                y,
                f"{category}\n({fraction:.0%} of D²)",
                color="crimson",
                fontsize=9,
                fontweight="bold",
                ha="left",
                va="bottom",
            )

        ax.axhline(0, color="0.85", lw=0.6, zorder=0)
        ax.axvline(0, color="0.85", lw=0.6, zorder=0)
        ax.set_xlabel(f"{x_label} ({explained[x_label]:.1%} variance)")
        ax.set_ylabel(f"{y_label} ({explained[y_label]:.1%} variance)")
        ax.set_title(f"{self.data.name}: PCA biplot")

        # Flag it whenever the loadings/ellipse have been stretched off the axis's true scale --
        # scores and shift arrows never are, so this is the only thing a reader needs to discount.
        if loading_scale != 1:
            ax.text(
                0.02,
                0.02,
                f"loading arrows ×{loading_scale:.1f} for legibility",
                transform=ax.transAxes,
                color="0.5",
                fontsize=8,
                ha="left",
                va="bottom",
            )

        return ax

    def run(self, *, output_directory: Path | str | None = None) -> dict[str, pd.DataFrame | Axes]:
        """Runs every diagnostic applicable to this container, optionally saving each result.

        Tries each diagnostic in turn and skips (logging why) any that raise ``ValueError``.
        :meth:`covariance_matrix`/:meth:`correlation_coefficient` need no category and are always
        included; every other diagnostic needs ``category_column`` set with at least two
        categories present, except :meth:`mahalanobis_alignment`, which needs *exactly* two (see
        :meth:`category_mahalanobis_alignment` for the version that works for any number).

        Args:
            output_directory: Optional directory to save each included result to, as
                ``f"{self.data.name}_{key}"`` (``key`` being the diagnostic's method name) —
                ``.xlsx`` for a dataframe result, or a figure file for a plot. If ``None``,
                results are only returned, not saved.

        Returns:
            Dict mapping each applicable diagnostic's method name to its result (a dataframe, or
            an ``Axes`` for a plot).
        """
        providers: dict[str, Callable[[], pd.DataFrame | Axes]] = {
            "covariance_matrix": self.covariance_matrix,
            "within_category_covariance_matrix": self.within_category_covariance_matrix,
            "category_mean_difference": self.category_mean_difference,
            "covariance_eigenanalysis": self.covariance_eigenanalysis,
            "category_covariance_eigenanalysis": self.category_covariance_eigenanalysis,
            "mahalanobis_alignment": self.mahalanobis_alignment,
            "category_mahalanobis_alignment": self.category_mahalanobis_alignment,
            "pca_biplot": self.pca_biplot,
            "correlation_coefficient": self.correlation_coefficient,
        }

        results: dict[str, pd.DataFrame | Axes] = {}
        for key, method in providers.items():
            try:
                results[key] = method()
            except ValueError as error:
                logger.info("Skipping '%s' diagnostic for '%s': %s", key, self.data.name, error)

        if output_directory is not None:
            output_directory = Path(output_directory)
            output_directory.mkdir(parents=True, exist_ok=True)
            for key, result in results.items():
                if isinstance(result, pd.DataFrame):
                    result.to_excel(output_directory / f"{self.data.name}_{key}.xlsx")
                else:
                    save_figure(
                        get_figure(result), Path(f"{self.data.name}_{key}"), output_directory
                    )

        return results

    def correlation_coefficient(
        self,
        *,
        method: Literal["pearson", "kendall", "spearman"] = "pearson",
        min_periods: int = 1,
        numeric_only: bool = False,
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
        corr_matrix: pd.DataFrame = self.data.values.corr(method, min_periods, numeric_only)
        _, ax = plt.subplots()
        sns.heatmap(corr_matrix, cmap="coolwarm", annot=True, fmt=".2f", vmin=-1, vmax=1, ax=ax)
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
        select_data_column: Name of the metadata column that identifies each sample, read by
            :attr:`data_names`. Defaults to ``"ID"``.
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

        # 2. Lock categorical universe
        if self.category_column:
            col: pd.Series = self.metadata[  # pyright: ignore[reportAssignmentType]
                self.category_column
            ]
            # Keep an already-categorical column's exact universe (even zero-count categories)
            # unchanged, so e.g. train/test splits taken from the same source share one universe.
            if not isinstance(col.dtype, pd.CategoricalDtype):
                cat_names = sorted(col.dropna().unique())
                cat_type = pd.CategoricalDtype(categories=cat_names, ordered=True)
                self.metadata[self.category_column] = col.astype(cat_type)

        # 3. Fit or apply scaling parameters
        if scaling_params is not None:
            self.scaling = scaling_params.align_to(self.values.columns)
        else:
            self.scaling = self._fit_scaling()
        self._validate_scaling()

        # 4. Derive standardized views
        self.values_std: pd.DataFrame = self.scaling.transform(self.values)
        self.uncertainties_std = self.uncertainties / self.scaling.stds

        # 5. Diagnostics namespace, built last since it needs values_std
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
        feature_columns: Mapping[str, str] | Iterable[str],
        uncertainty_columns: Mapping[str, str] | None = None,
        **kwargs,
    ) -> Self:
        """Creates a data container from a combined dataframe.

        Args:
            dataframe: A dataframe with columns of feature values, optionally their
                uncertainties, and metadata.
            feature_columns: Which columns are feature values, and what to call them. A mapping
                gives ``{raw_column_name: clean_feature_name}`` (for datasets whose raw column
                names need renaming, e.g. mass-spec channel names); a plain iterable of names
                means those columns are already clean (used as-is, no rename).
            uncertainty_columns: Optional ``{raw_column_name: clean_feature_name}`` mapping for
                uncertainty columns — each value must be one of ``feature_columns``'s clean
                names. Columns not claimed by ``feature_columns``/``uncertainty_columns`` become
                metadata. Defaults to ``None`` (no uncertainties).
            **kwargs: Arbitrary keyword arguments for constructor

        Raises:
            ValueError: If ``feature_columns`` maps two different raw columns to the same clean
                name, or if ``uncertainty_columns`` names a clean feature not present in
                ``feature_columns``.
            KeyError: If a raw column named in ``feature_columns``/``uncertainty_columns`` isn't
                actually present in ``dataframe``.

        Returns:
            A new data container.
        """
        feature_map: dict[str, str] = (
            dict(feature_columns)
            if isinstance(feature_columns, Mapping)
            else {c: c for c in feature_columns}
        )
        if len(set(feature_map.values())) != len(feature_map):
            raise ValueError(
                f"feature_columns maps multiple raw columns to the same name: {feature_map}"
            )

        values: pd.DataFrame = dataframe[list(feature_map.keys())].rename(columns=feature_map)

        uncertainties: pd.DataFrame | None = None
        if uncertainty_columns is not None:
            uncertainty_map: dict[str, str] = dict(uncertainty_columns)
            unknown: set[str] = set(uncertainty_map.values()) - set(feature_map.values())
            if unknown:
                raise ValueError(
                    f"uncertainty_columns names features not in feature_columns: {unknown}"
                )
            uncertainties = dataframe[list(uncertainty_map.keys())].rename(columns=uncertainty_map)
            uncertainties = uncertainties[values.columns]  # align to values' column order

        claimed: set[str] = set(feature_map) | set(uncertainty_columns or {})
        metadata: pd.DataFrame = dataframe.drop(columns=list(claimed))

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

        Every input must share the same feature columns (``values.columns``). Inputs may have
        overlapping row labels (e.g. each started from its own 0-based Excel row index), so the
        combined container gets a fresh row index; each input's original index and :attr:`name`
        are preserved as new metadata columns (``source_index_column``/``source_name_column``) to
        recover the original source of any row.

        Each input's :attr:`uncertainties` is already true 1-sigma (divided by its own
        ``uncertainty_scale`` at construction), so concatenation is direct and the combined
        container defaults to ``uncertainty_scale=1.0``.

        A ``category_column`` metadata column, if present, is decategorized before concatenation
        so per-container category universes don't clash; pass ``category_column`` in ``kwargs``
        to re-derive one combined categorical universe from the union of all inputs.

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
            An instance.
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
            An instance.
        """
        data: pd.DataFrame = pd.read_excel(filename_path, sheet_name=sheet_name)

        return cls.from_dataframe(data, **kwargs)

    def get_destandardized_values(self, standardized_values: NpArray) -> NpArray:
        return self.scaling.inverse_transform(standardized_values)

    def _fit_scaling(self) -> ScalingParams:
        means = self.values.mean(axis=0)
        stds = self.values.std(axis=0, ddof=0)
        return ScalingParams(means=means, stds=stds)

    def _validate_raw_inputs(
        self,
        values: pd.DataFrame,
        uncertainties: pd.DataFrame | None,
        metadata: pd.DataFrame | None,
        category_column: str | None,
    ) -> None:
        if len(values) == 0:
            raise ValueError("`values` must have at least one sample")

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
