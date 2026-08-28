# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Plotting utilities for group difference models"""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.stats import beta, gaussian_kde

from bedroc.core.data_container import DataContainer
from bedroc.core.plotting import save_figure
from bedroc.core.type_aliases import NpArray, NpFloat
from bedroc.core.utils import SummaryStatistics
from bedroc.difference import DEFAULT_CATEGORY_COLORS, DEFAULT_CATEGORY_NAMES
from bedroc.difference.utils import distribution_overlap

logger: logging.Logger = logging.getLogger(__name__)


def plot_group_fraction_posterior(
    pi_0_samples: NpFloat,
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    bins: int = 50,
    n_grid: int = 101,
    category_names: Sequence | NpArray,
    category_colors: Sequence = DEFAULT_CATEGORY_COLORS,
    category_counts: pd.Series | None = None,
    ax: Axes | None = None,
    figsize: tuple = (8, 5),
) -> Axes:
    """Plots the posterior distribution of group fractions.

    The posterior is shown together with the beta prior and, where available, the observed
    group fraction.

    Args:
        pi_0_samples: Samples from the posterior distribution of the group-0 fraction.
        prior_alpha: Alpha parameter of the beta prior. Defaults to ``1.0``.
        prior_beta: Beta parameter of the beta prior. Defaults to ``1.0``.
        bins: Number of bins for the histogram. Defaults to ``50``.
        n_grid: Number of grid points for the prior and perfect-classification limit. Defaults to
            ``101``.
        category_names: Names for the two categories.
        category_colors: Colors for the two categories. Defaults to :obj:`DEFAULT_CATEGORY_COLORS`.
        category_counts: Known counts for the two categories. If ``None``, the observed fractions
            are not plotted. Defaults to ``None``.
        ax: Matplotlib axes on which to plot. If ``None``, a new figure and axes are created.
        figsize: Size of the figure if ``ax`` is ``None``. Defaults to ``(8, 5)``.

    Returns:
        Matplotlib axes containing the posterior group-fraction plot
    """
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("prior_alpha and prior_beta must be > 0.")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    category_0, category_1 = category_names

    grid: NpArray = np.linspace(0, 1, n_grid)

    def plot_posterior(label: str, samples: NpFloat, color: str, ci_y_loc: float) -> None:
        stats = SummaryStatistics(samples)
        lower, upper = stats.lower_95, stats.upper_95

        counts, bin_edges = np.histogram(samples, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = bin_edges[1] - bin_edges[0]

        is_central = (bin_centers >= lower) & (bin_centers <= upper)

        # Draw tails (reduced opacity)
        ax.bar(bin_centers[~is_central], counts[~is_central], width=width, color=color, alpha=0.2)
        # Draw central 95% interval
        ax.bar(
            bin_centers[is_central],
            counts[is_central],
            width=width,
            color=color,
            alpha=0.5,
            label=label,
        )

        kde_pdf = gaussian_kde(samples)(grid)
        ax.plot(grid, kde_pdf, color=color, linewidth=2)

        ax.errorbar(
            stats.median,
            ci_y_loc,
            xerr=stats.xerr_95,
            fmt="o",
            color=color,
            capsize=4,
            capthick=2,
            elinewidth=2,
        )

    plot_posterior(str(category_0), pi_0_samples, category_colors[0], ci_y_loc=0.4)
    plot_posterior(str(category_1), 1.0 - pi_0_samples, category_colors[1], ci_y_loc=0.6)

    # Dummy line for legend entry for credible interval
    ax.plot([], [], color="black", linewidth=2, marker="o", label="95% CrI")

    # Beta prior
    prior_pdf: NpArray = beta.pdf(grid, prior_alpha, prior_beta)

    ax.plot(
        grid,
        prior_pdf,
        color="black",
        linestyle="--",
        linewidth=2,
        label=rf"{category_0} prior",  #: beta($\alpha={prior_alpha:g},\ \beta={prior_beta:g}$)",
    )

    # Observed fractions, if available
    if category_counts is not None:
        # Perfect-classification limit for category 0
        limiting_posterior_0: NpFloat = beta.pdf(
            grid, prior_alpha + category_counts.iloc[0], prior_beta + category_counts.iloc[1]
        )

        ax.plot(
            grid,
            limiting_posterior_0,
            color=category_colors[0],
            linestyle="--",
            linewidth=2,
            label="Perfect-classification limit",
        )

        observed_fraction_0 = category_counts.iloc[0] / sum(category_counts)
        logger.info(
            "Observed fraction for %s: %.2f (count = %d)",
            category_0,
            observed_fraction_0,
            category_counts.iloc[0],
        )
        observed_fraction_1 = category_counts.iloc[1] / sum(category_counts)
        logger.info(
            "Observed fraction for %s: %.2f (count = %d)",
            category_1,
            observed_fraction_1,
            category_counts.iloc[1],
        )

        ax.annotate(
            f"Obs\n{observed_fraction_0:.2f}",
            xy=(observed_fraction_0, 0.6),
            xytext=(observed_fraction_0, 1.8),
            ha="center",
            va="bottom",
            color=category_colors[0],
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.9),
            arrowprops=dict(arrowstyle="-|>", color=category_colors[0], lw=1.5),
        )

        ax.annotate(
            f"Obs\n{observed_fraction_1:.2f}",
            xy=(observed_fraction_1, 0.8),
            xytext=(observed_fraction_1, 2.2),
            ha="center",
            va="bottom",
            color=category_colors[1],
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.9),
            arrowprops=dict(arrowstyle="-|>", color=category_colors[1], lw=1.5),
        )

        stats_0 = SummaryStatistics(pi_0_samples, truth=observed_fraction_0)

        logger.info(
            "Is observed fraction (%.2f) within 95%% CrI [%.2f, %.2f] for %s? %s",
            observed_fraction_0,
            stats_0.lower_95.item(),
            stats_0.upper_95.item(),
            category_0,
            stats_0.within_ci.item(),  # pyright: ignore[reportOptionalMemberAccess]
        )

    ax.set(xlabel="Category fraction", ylabel="Density", xlim=(0, 1))
    ax.set_title("Posterior distribution of category fractions")

    ax.legend()

    return ax


def plot_distribution_overlap(
    values_0: NpArray,
    values_1: NpArray,
    *,
    ax: Axes | None = None,
    n_grid: int = 2000,
    group_names: Sequence = DEFAULT_CATEGORY_NAMES,
    group_colors: Sequence[str] = DEFAULT_CATEGORY_COLORS,
) -> tuple[Figure, Axes, float]:
    """Plots two distributions and their overlap.

    The samples, KDEs, and overlapping probability density are shown.

    Args:
        values_0: Samples from the first distribution
        values_1: Samples from the second distribution
        ax: Matplotlib axes on which to plot. If ``None``, a new figure and axes are created.
        n_grid: Number of points to use for the grid over which to evaluate the PDFs. Defaults to
            ``2000``.
        group_names: Names for the two groups. Defaults to :obj:`DEFAULT_CATEGORY_NAMES`.
        group_colors: Colors for the two groups. Defaults to :obj:`DEFAULT_CATEGORY_COLORS`.

    Returns:
        Matplotlib figure and axes
    """
    x, pdf_0, pdf_1, overlap_density, overlap = distribution_overlap(
        values_0, values_1, n_grid=n_grid
    )

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Plot KDEs
    ax.plot(x, pdf_0, color=group_colors[0], linewidth=2, label=group_names[0])
    ax.plot(x, pdf_1, color=group_colors[1], linewidth=2, label=group_names[1])

    # Shade the overlap
    ax.fill_between(x, overlap_density, alpha=0.3, label=f"Overlap (OVL = {overlap:.2f})")

    ax.set_xlabel("Standardized units")
    ax.set_ylabel("Density")
    ax.legend()

    return fig, ax, overlap  # pyright: ignore[reportReturnType]


def plot_corner(
    data: DataContainer,
    *,
    feature_labels: Mapping[str, str] | None = None,
    hue_column: str | None = None,
    tick_overrides: Mapping[str, tuple[Sequence[float], Sequence[str]]] | None = None,
    output_directory: Path | None = None,
    savefig_kwargs: dict[str, Any] | None = None,
) -> None:
    """Plots corner plots (pairwise feature relationships) of a data container's features.

    Always generates one category-vs-category corner plot, comparing the two categories defined
    by ``data.category_column``. If ``hue_column`` names a metadata column present on ``data``,
    an additional corner plot is generated per category, colored by that column, with reference
    KDE overlays showing the overall within- and across-category distributions.

    Args:
        data: Data container providing the features and category information to plot. Must have
            ``category_column`` set.
        feature_labels: Optional mapping from feature name to a display label (e.g. with units).
            Features not present in the mapping are labeled with their raw name. Defaults to
            ``None``.
        hue_column: Optional metadata column name used for a secondary, per-category corner plot
            (e.g. sample locality). Skipped if ``None`` or not present in ``data``'s metadata.
            Defaults to ``None``.
        tick_overrides: Optional mapping from a feature's display label to a
            ``(tick_positions, tick_labels)`` pair, for features whose axis should show custom
            tick marks (e.g. to un-transform a log-scaled feature back to its original units).
            Only applied to the category-vs-category plot. Defaults to ``None``.
        output_directory: Directory to save the plot. ``None`` for no output.
        savefig_kwargs: Override keyword arguments for :func:`matplotlib.pyplot.savefig`.
            Defaults to ``None``.

    Raises:
        ValueError: If ``data.category_column`` is not set
    """
    if data.category_column is None:
        raise ValueError("plot_corner requires a DataContainer with a category_column set.")

    df: pd.DataFrame = data.get_dataframe()

    display_names: dict[str, str] = {
        feature: (feature_labels or {}).get(feature, feature) for feature in data.feature_names
    }
    plot_vars: list[str] = list(display_names.values())

    plot_df: pd.DataFrame = cast(pd.DataFrame, df["Values"].copy())
    plot_df["Population"] = df["Metadata"][data.category_column]

    if hue_column is not None and hue_column in df["Metadata"]:
        plot_df[hue_column] = df["Metadata"][hue_column]
    else:
        hue_column = None

    def filter_category(category: str, *, exclude: bool = False) -> pd.DataFrame:
        """Filters the plotting DataFrame by category, optionally inverted."""
        mask = plot_df["Population"] == category

        if exclude:
            mask = ~mask

        return plot_df.loc[mask]

    def apply_tick_overrides(axes) -> None:
        """Replaces axis tick positions/labels for features listed in ``tick_overrides``."""
        if not tick_overrides:
            return

        for ax in axes:
            if ax.get_xlabel() in tick_overrides:
                positions, labels = tick_overrides[ax.get_xlabel()]
                ax.set_xticks(positions)
                ax.set_xticklabels(labels)

            if ax.get_ylabel() in tick_overrides:
                positions, labels = tick_overrides[ax.get_ylabel()]
                ax.set_yticks(positions)
                ax.set_yticklabels(labels)

    # Add display labels for the features to the column names for clarity in the plots
    plot_df.rename(columns=display_names, inplace=True)

    category_0, category_1 = data.category_names  # pyright: ignore[reportGeneralTypeIssues]

    # Category-vs-category pairplot
    g: sns.PairGrid = sns.PairGrid(
        plot_df,
        hue="Population",
        hue_order=data.category_names,
        vars=plot_vars,
        corner=False,
        diag_sharey=False,
    )
    # Histogram to reveal any truncation effects in the data, with KDE overlay to show the smoothed
    # distribution shape
    g.map_diag(sns.histplot, fill=True, alpha=0.6, common_norm=True, stat="density")
    g.map_diag(sns.kdeplot, linewidth=2, linestyle="-", common_norm=True)
    g.map_upper(sns.scatterplot, alpha=0.4, s=20)
    g.map_lower(sns.kdeplot, levels=4)

    apply_tick_overrides(g.figure.axes)

    g.add_legend()
    sns.move_legend(g, "upper right", bbox_to_anchor=(0.4, 0.98), frameon=True)
    g.figure.tight_layout()

    save_figure(
        g.figure,
        Path(f"{data.name}_{category_1}_vs_{category_0}"),
        output_directory,
        savefig_kwargs,
    )

    if hue_column is None:
        return

    # Pair plots for each category, colored by hue_column, with KDE overlays
    for category in data.category_names:  # pyright: ignore[reportOptionalIterable]
        g = sns.PairGrid(
            filter_category(category),
            hue=hue_column,
            vars=plot_vars,
            corner=False,
            diag_sharey=False,
        )
        g.map_diag(sns.kdeplot, fill=True, alpha=0.6, common_norm=True)
        g.map_lower(sns.scatterplot, alpha=0.4, s=20)
        g.map_upper(sns.kdeplot, levels=4, common_norm=True)

        for ax, var in zip(g.diag_axes, plot_vars):  # pyright: ignore
            sns.kdeplot(
                data=filter_category(category, exclude=True),
                x=var,
                ax=ax,
                color="black",
                linewidth=2,
                fill=False,
                linestyle="--",
                common_norm=True,
            )
            sns.kdeplot(
                data=filter_category(category),
                x=var,
                ax=ax,
                color="black",
                linewidth=2,
                fill=False,
                common_norm=True,
            )

        for row, yvar in enumerate(plot_vars):
            for col, xvar in enumerate(plot_vars):
                if row <= col:
                    continue

                ax = g.axes[row, col]

                sns.kdeplot(
                    data=filter_category(category),
                    x=xvar,
                    y=yvar,
                    ax=ax,
                    color="black",
                    levels=4,
                    linewidths=1,
                    fill=False,
                    common_norm=True,
                )

        g.add_legend()
        sns.move_legend(g, "upper right", bbox_to_anchor=(0.4, 0.98), frameon=True)

        other_category: str = [name_ for name_ in data.category_names if name_ != category][0]

        line_legend = [
            Line2D([0], [0], color="black", lw=2, label=f"{category} overall"),
            Line2D([0], [0], color="black", lw=2, ls="--", label=f"{other_category} overall"),
        ]

        g.figure.legend(
            handles=line_legend,
            loc="upper left",
            title="Reference",
            frameon=True,
            bbox_to_anchor=(0.16, 0.7),
        )

        g.figure.suptitle(f"{data.name}: {str(category).capitalize()} by {hue_column}")
        g.figure.tight_layout()

        save_figure(
            g.figure,
            Path(f"{data.name}_{str(category).lower()}_by_{hue_column.lower()}"),
            output_directory,
            savefig_kwargs,
        )
