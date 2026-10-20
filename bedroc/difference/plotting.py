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
from bedroc.core.plotting import get_figure, save_figure
from bedroc.core.type_aliases import NpArray, NpFloat
from bedroc.core.utils import SummaryStatistics
from bedroc.difference import DEFAULT_CATEGORY_COLORS, DEFAULT_CATEGORY_NAMES
from bedroc.difference.utils import distribution_overlap, kde_mixture_weight, mixture_weight

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
    known_parameter_pdf: tuple[NpFloat, NpFloat] | None = None,
    show_prior: bool = True,
    title: str = "Posterior distribution of category fractions",
    interval_label: str = "95% CrI",
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
        category_counts: Known, true counts for the two categories. If ``None``, the observed
            fractions and perfect-classification limit are not plotted. Defaults to ``None``.
        known_parameter_pdf: ``(grid, density)`` pair giving the posterior density of the group-0
            fraction under a fitted model's own (point-estimate) parameters (e.g. from
            :func:`~bedroc.difference.utils.known_parameter_pi0_posterior`). Unlike
            ``category_counts``, this needs no ground truth. If ``None``, the known-parameter limit
            is not plotted. Defaults to ``None``.
        show_prior: Whether to plot the Beta prior. Set ``False`` for non-Bayesian estimates (e.g.
            bootstrap samples), which have no prior; ``prior_alpha``/``prior_beta`` are then only
            used for the perfect-classification limit. Defaults to ``True``.
        title: Plot title. Defaults to ``"Posterior distribution of category fractions"``.
        interval_label: Legend label for the 95% interval markers. Defaults to ``"95% CrI"``
            (credible interval); use e.g. ``"95% CI"`` for a bootstrap confidence interval.
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
    ax.plot([], [], color="black", linewidth=2, marker="o", label=interval_label)

    # Beta prior
    if show_prior:
        prior_pdf: NpArray = beta.pdf(grid, prior_alpha, prior_beta)

        ax.plot(
            grid,
            prior_pdf,
            color="black",
            linestyle="--",
            linewidth=2,
            # Could also show the parameters, e.g. rf"...: beta($\alpha={prior_alpha:g}$, ...)"
            label=rf"{category_0} prior",
        )

    # Observed fractions, if available
    if category_counts is not None:
        # Perfect-classification limit for category 0: the posterior of pi_0 if every unlabeled
        # sample's category were known, i.e. Beta-Binomial counting with no classifier involved.
        # Its width is the finite-sample noise of the unlabeled data alone.
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

    # Known-parameter limit, if available. Unlike the perfect-classification limit above, this needs
    # no ground truth (known_parameter_pdf comes from inferring pi_0 via the same mixture likelihood
    # the model itself uses, but holding its other fitted parameters fixed at a point estimate — a
    # plug-in benchmark, not literal knowledge of the truth), so it can be shown even without
    # category_counts. Its width includes the same finite-sample noise as the perfect-classification
    # limit plus the ambiguity from class overlap (the gap between the two curves), but not
    # parameter uncertainty (the gap between this curve and the full pi_0_samples posterior above,
    # typically but not necessarily wider). For tempered models the tempering also widens it, so it
    # cannot reach the perfect-classification limit even for perfectly separated categories. Not
    # directly comparable across models with different covariance structures, since each model's own
    # fitted parameters are used (see known_parameter_pi0_posterior's docstring).
    if known_parameter_pdf is not None:
        known_grid, known_density = known_parameter_pdf

        ax.plot(
            known_grid,
            known_density,
            color=category_colors[0],
            linestyle=":",
            linewidth=2,
            label="Known-parameter limit",
        )

    ax.set(xlabel="Category fraction", ylabel="Density", xlim=(0, 1))
    ax.set_title(title)

    ax.legend()

    return ax


def plot_mahalanobis_distance(
    samples: NpFloat,
    *,
    reference_value: float | None = None,
    bins: int = 50,
    n_grid: int = 2001,
    color: str = "0.3",
    ax: Axes | None = None,
    figsize: tuple = (8, 5),
) -> Axes:
    """Plots the posterior distribution of the Mahalanobis distance between two category means.

    Mahalanobis distance is the multivariate generalization of a standardized mean difference
    (Cohen's d): it accounts for correlations between features via the shared covariance matrix,
    and reduces to the univariate case in one dimension.

    Args:
        samples: Posterior samples of the Mahalanobis distance.
        reference_value: Optional known/ground-truth Mahalanobis distance (e.g. from a synthetic
            data-generating process) to overlay for validation. Defaults to ``None``.
        bins: Number of bins for the histogram. Defaults to ``50``.
        n_grid: Number of grid points for the KDE curve. Defaults to ``2001``.
        color: Color for the histogram, KDE, and credible interval. Defaults to ``"0.3"``.
        ax: Matplotlib axes on which to plot. If ``None``, a new figure and axes are created.
        figsize: Size of the figure if ``ax`` is ``None``. Defaults to ``(8, 5)``.

    Returns:
        Matplotlib axes containing the posterior Mahalanobis distance plot
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    x_max = max(float(np.max(samples)), 2.0) * 1.05

    stats = SummaryStatistics(samples)
    lower, upper = stats.lower_95, stats.upper_95

    counts, bin_edges = np.histogram(samples, bins=bins, density=True, range=(0.0, x_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = bin_edges[1] - bin_edges[0]

    is_central = (bin_centers >= lower) & (bin_centers <= upper)

    ax.bar(bin_centers[~is_central], counts[~is_central], width=width, color=color, alpha=0.2)
    ax.bar(
        bin_centers[is_central],
        counts[is_central],
        width=width,
        color=color,
        alpha=0.5,
        label="Posterior",
    )

    grid: NpArray = np.linspace(0.0, x_max, n_grid)
    kde_pdf = gaussian_kde(samples)(grid)
    ax.plot(grid, kde_pdf, color=color, linewidth=2)

    ylim = ax.get_ylim()
    ci_y_loc = ylim[1] * 0.1

    ax.errorbar(
        stats.median,
        ci_y_loc,
        xerr=stats.xerr_95,
        fmt="o",
        color=color,
        capsize=4,
        capthick=2,
        elinewidth=2,
        label="95% CrI",
    )

    if reference_value is not None:
        ax.axvline(
            reference_value,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Reference (D = {reference_value:.2f})",
        )

    ax.set(xlabel="Mahalanobis distance (D)", ylabel="Density", xlim=(0.0, x_max), ylim=ylim)
    ax.set_title("Posterior distribution of Mahalanobis distance")
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
        fig = get_figure(ax)

    # Plot KDEs
    ax.plot(x, pdf_0, color=group_colors[0], linewidth=2, label=group_names[0])
    ax.plot(x, pdf_1, color=group_colors[1], linewidth=2, label=group_names[1])

    # Shade the overlap
    ax.fill_between(x, overlap_density, alpha=0.3, label=f"Overlap (OVL = {overlap:.2f})")

    ax.set_xlabel("Standardized units")
    ax.set_ylabel("Density")
    ax.legend()

    return fig, ax, overlap


def _prepare_corner_plot_df(
    data: DataContainer,
    feature_labels: Mapping[str, str] | None,
    *,
    extra_metadata_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, list[str], str, str]:
    """Builds a display-labeled feature DataFrame plus the category column, ready for seaborn
    plotting.

    The category column is included under its own name (``data.category_column``), rather than
    being copied to a new synthetic name, since it is already a real column in ``data``'s
    metadata.

    Args:
        data: Data container providing the features and category information to plot. Must have
            ``category_column`` set.
        feature_labels: Optional mapping from feature name to a display label (e.g. with units).
            Features not present in the mapping are labeled with their raw name.
        extra_metadata_columns: Additional metadata column names to include (under their raw
            names, unaffected by ``feature_labels``). Defaults to no extra columns.

    Returns:
        Tuple of the prepared DataFrame, the list of (display) feature column names, and the two
        category names in ``data.category_names`` order.

    Raises:
        ValueError: If ``data.category_column`` is not set, if it does not have exactly two
            categories, or if any of ``extra_metadata_columns`` is not present in ``data``'s
            metadata
    """
    if data.category_column is None:
        raise ValueError("This plot requires a DataContainer with a category_column set.")

    category_names = data.category_names
    assert category_names is not None  # Guaranteed by the category_column check above
    if len(category_names) != 2:
        raise ValueError(
            f"This plot requires exactly two categories, found {len(category_names)}."
        )
    category_0, category_1 = (str(name) for name in category_names)

    df: pd.DataFrame = data.get_dataframe()

    display_names: dict[str, str] = {
        str(feature): (feature_labels or {}).get(str(feature), str(feature))
        for feature in data.feature_names
    }
    plot_vars: list[str] = list(display_names.values())

    plot_df: pd.DataFrame = cast(pd.DataFrame, df["Values"].copy())
    plot_df[data.category_column] = df["Metadata"][data.category_column]

    for column in extra_metadata_columns:
        if column not in df["Metadata"]:
            raise ValueError(f"{column!r} not found in the data container's metadata.")
        plot_df[column] = df["Metadata"][column]

    # Add display labels for the features to the column names for clarity in the plots
    plot_df.rename(columns=display_names, inplace=True)

    return plot_df, plot_vars, category_0, category_1


def plot_corner(
    data: DataContainer,
    *,
    feature_labels: Mapping[str, str] | None = None,
    tick_overrides: Mapping[str, tuple[NpArray, Sequence[str]]] | None = None,
    output_directory: Path | None = None,
    savefig_kwargs: dict[str, Any] | None = None,
) -> None:
    """Plots a single corner plot of all of a data container's samples, colored by category.

    Args:
        data: Data container providing the features and category information to plot. Must have
            ``category_column`` set.
        feature_labels: Optional mapping from feature name to a display label (e.g. with units).
            Features not present in the mapping are labeled with their raw name. Defaults to
            ``None``.
        tick_overrides: Optional mapping from a feature's display label to a
            ``(tick_positions, tick_labels)`` pair, for features whose axis should show custom
            tick marks (e.g. to un-transform a log-scaled feature back to its original units).
            Defaults to ``None``.
        output_directory: Directory to save the plot. ``None`` for no output.
        savefig_kwargs: Override keyword arguments for :func:`matplotlib.pyplot.savefig`.
            Defaults to ``None``.

    Raises:
        ValueError: If ``data.category_column`` is not set
    """
    plot_df, plot_vars, category_0, category_1 = _prepare_corner_plot_df(data, feature_labels)

    g: sns.PairGrid = sns.PairGrid(
        plot_df,
        hue=data.category_column,
        hue_order=(category_0, category_1),
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

    _apply_tick_overrides(g.figure.axes, tick_overrides)

    g.add_legend()
    sns.move_legend(g, "upper right", bbox_to_anchor=(0.4, 0.98), frameon=True)
    g.figure.tight_layout()

    save_figure(
        g.figure,
        Path(f"{data.name}_{category_1}_vs_{category_0}"),
        output_directory,
        savefig_kwargs,
    )


def _apply_tick_overrides(
    axes: Sequence[Axes], tick_overrides: Mapping[str, tuple[NpArray, Sequence[str]]] | None
) -> None:
    """Sets custom ticks on every axis whose x or y label is in ``tick_overrides``.

    Args:
        axes: Axes to update (e.g. a seaborn ``PairGrid``'s ``figure.axes``)
        tick_overrides: Mapping from a feature's display label to a
            ``(tick_positions, tick_labels)`` pair. ``None`` or empty for no change.
    """
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


def _joint_mixture_weight(
    labeled: DataContainer, unlabeled: DataContainer, *, max_iterations: int = 1000
) -> float:
    """Maximum-likelihood category-0 weight of the unlabeled samples, jointly over all features,
    under the labeled groups' fixed densities.

    Each labeled category's density is a multivariate kernel density estimate over all features
    together (on the standardized values, so every feature gets a comparable bandwidth), with no
    parametric (e.g. Gaussian) assumption; correlations between features are included. Only the
    mixture weight is fitted, by expectation-maximization. This is the single category fraction
    that best explains the unlabeled samples as a mixture of the two labeled groups.

    Args:
        labeled: Labeled data, with exactly two categories
        unlabeled: Unlabeled data, standardized with the labeled data's scaling
        max_iterations: Maximum number of expectation-maximization iterations. Defaults to
            ``1000``.

    Returns:
        The fitted category-0 weight
    """
    features: list[str] = list(labeled.feature_names)
    codes = labeled.category_codes
    assert codes is not None  # Guaranteed by the caller's category checks
    values: pd.DataFrame = labeled.values_std[features]
    complete: pd.Series = values.notna().all(axis=1)

    return kde_mixture_weight(
        values[complete & (codes == 0)].to_numpy(),
        values[complete & (codes == 1)].to_numpy(),
        unlabeled.values_std[features].dropna().to_numpy(),
        max_iterations=max_iterations,
    )


def plot_corner_with_unlabeled(
    labeled: DataContainer,
    unlabeled: DataContainer,
    *,
    unlabeled_label: str = "Unlabeled",
    feature_labels: Mapping[str, str] | None = None,
    tick_overrides: Mapping[str, tuple[NpArray, Sequence[str]]] | None = None,
    category_colors: Sequence = DEFAULT_CATEGORY_COLORS,
    output_directory: Path | None = None,
    savefig_kwargs: dict[str, Any] | None = None,
) -> float:
    """Plots unlabeled samples against the two labeled groups' distributions, to show whether the
    unlabeled samples can be explained as a mixture of the groups.

    - Lower triangle: the unlabeled samples (grey points) with each group's KDE contours.
    - Upper triangle: the unlabeled samples' own KDE contours (black) with each group's contours.
    - Diagonal: each group's density, the unlabeled density (black, filled), and the best single
      mixture of the two group densities (dashed), using the one category-0 weight that best
      explains the unlabeled samples across all features jointly (see
      :func:`_joint_mixture_weight`). If the unlabeled samples were such a mixture, the
      dashed curve would match the unlabeled density for every feature. Each diagonal panel also
      shows (dotted, and annotated) that feature's own best mixture, with the weight fitted to its
      marginal densities alone; for a genuine mixture these weights would all agree with each
      other and with the joint weight.

    Every density, and both kinds of weight, comes from kernel density estimates of the data; no
    parametric (e.g. Gaussian) form is assumed, and only the mixture weights are fitted. If the
    unlabeled samples' true categories are known (e.g. a self-validation held-out split, which is
    a genuine mixture by construction), the true fraction is shown in the title.

    Args:
        labeled: Labeled data, with exactly two categories
        unlabeled: Unlabeled data, standardized with the labeled data's scaling (as in
            :class:`~bedroc.difference.partitioning.LabeledUnlabeledSplit`)
        unlabeled_label: Display name for the unlabeled samples (e.g. ``"Detrital"``). Defaults
            to ``"Unlabeled"``.
        feature_labels: Optional mapping from feature name to a display label (e.g. with units).
            Defaults to ``None``.
        tick_overrides: Optional mapping from a feature's display label to a
            ``(tick_positions, tick_labels)`` pair (e.g. to show log-scaled features in their
            original units). Defaults to ``None``.
        category_colors: Colors for the two categories. Defaults to
            :obj:`DEFAULT_CATEGORY_COLORS`.
        output_directory: Directory to save the plot. ``None`` for no output.
        savefig_kwargs: Override keyword arguments for :func:`matplotlib.pyplot.savefig`.
            Defaults to ``None``.

    Raises:
        ValueError: If ``labeled`` does not have exactly two categories

    Returns:
        The best single category-0 weight shown on the diagonal
    """
    plot_df, plot_vars, category_0, category_1 = _prepare_corner_plot_df(labeled, feature_labels)
    plot_df = plot_df.dropna(subset=plot_vars)
    groups: dict[str, pd.DataFrame] = {
        category: plot_df[plot_df[labeled.category_column] == category]
        for category in (category_0, category_1)
    }

    display_names: dict[str, str] = {
        str(feature): (feature_labels or {}).get(str(feature), str(feature))
        for feature in labeled.feature_names
    }
    unlabeled_df: pd.DataFrame = unlabeled.values.rename(columns=display_names)[plot_vars].dropna()

    weight: float = _joint_mixture_weight(labeled, unlabeled)
    logger.info(
        "Best single %s fraction explaining the %s samples as a mixture of the labeled groups: "
        "%.3f",
        category_0,
        unlabeled_label,
        weight,
    )

    # Shared axis limits per feature, covering the labeled and unlabeled samples
    limits: dict[str, tuple[float, float]] = {}
    for var in plot_vars:
        low: float = float(min(plot_df[var].min(), unlabeled_df[var].min()))
        high: float = float(max(plot_df[var].max(), unlabeled_df[var].max()))
        pad: float = 0.05 * (high - low)
        limits[var] = (low - pad, high + pad)

    n: int = len(plot_vars)
    fig, axes = plt.subplots(n, n, figsize=(2.6 * n, 2.6 * n), squeeze=False)
    feature_weights: dict[str, float] = {}
    colors: dict[str, str] = dict(zip((category_0, category_1), category_colors))

    for row, y_var in enumerate(plot_vars):
        for col, x_var in enumerate(plot_vars):
            ax: Axes = axes[row, col]
            if row == col:
                grid: NpFloat = np.linspace(*limits[x_var], 300)
                densities: dict[str, NpFloat] = {}
                groups_kde: dict[str, gaussian_kde] = {}
                for category, group in groups.items():
                    groups_kde[category] = gaussian_kde(group[x_var])
                    densities[category] = groups_kde[category](grid)
                    ax.plot(grid, densities[category], color=colors[category], lw=2, label=category)
                unlabeled_density: NpFloat = gaussian_kde(unlabeled_df[x_var])(grid)
                ax.fill_between(grid, unlabeled_density, color="black", alpha=0.2)
                ax.plot(grid, unlabeled_density, color="black", lw=1.5, label=unlabeled_label)
                ax.plot(
                    grid,
                    weight * densities[category_0] + (1 - weight) * densities[category_1],
                    color="0.35",
                    ls="--",
                    lw=2,
                    label=f"Best joint mixture ({category_0} fraction {weight:.2f})",
                )
                # This feature's own best weight, from the same (KDE) group densities as plotted
                feature_weight: float = mixture_weight(
                    groups_kde[category_0].logpdf(unlabeled_df[x_var].to_numpy()),
                    groups_kde[category_1].logpdf(unlabeled_df[x_var].to_numpy()),
                )
                feature_weights[x_var] = feature_weight
                ax.plot(
                    grid,
                    feature_weight * densities[category_0]
                    + (1 - feature_weight) * densities[category_1],
                    color="tab:purple",
                    ls=":",
                    lw=2,
                    label="Best mixture per feature",
                )
                ax.text(
                    0.97,
                    0.95,
                    f"this feature:\n{category_0} fraction {feature_weight:.2f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8},
                )
                # Headroom above the tallest curve keeps the annotation clear of the densities
                ax.set_ylim(0, 1.35 * max(ax.get_ylim()[1], 1e-12))
                ax.set_yticks([])
            else:
                if row > col:
                    ax.scatter(
                        unlabeled_df[x_var], unlabeled_df[y_var], s=6, color="0.5", alpha=0.35
                    )
                for category, group in groups.items():
                    sns.kdeplot(
                        x=group[x_var],
                        y=group[y_var],
                        ax=ax,
                        levels=4,
                        color=colors[category],
                        linewidths=1.2,
                    )
                if row < col:
                    sns.kdeplot(
                        x=unlabeled_df[x_var],
                        y=unlabeled_df[y_var],
                        ax=ax,
                        levels=4,
                        color="black",
                        linewidths=1.5,
                    )
                ax.set_ylim(limits[y_var])
            ax.set_xlim(limits[x_var])
            ax.set_xlabel(x_var)
            ax.set_ylabel(y_var)

    # Ticks are matched to axes by label, so apply them before hiding the inner labels; the
    # diagonal's y axis is a density, so its y label is cleared first
    for i in range(n):
        axes[i, i].set_ylabel("")
    _apply_tick_overrides(fig.axes, tick_overrides)
    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            if row < n - 1:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)
            if col > 0 or row == 0:
                ax.set_ylabel("")
                if col > 0:
                    ax.tick_params(labelleft=False)
    axes[0, 0].set_ylabel(plot_vars[0])

    logger.info(
        "Best %s fraction per feature for the %s samples: %s",
        category_0,
        unlabeled_label,
        {feature: round(value, 3) for feature, value in feature_weights.items()},
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    # In self-validation the target population's true fraction is known: show it for comparison
    title: str = (
        f"{unlabeled_label} samples vs the {category_0} and {category_1} groups ({labeled.name})"
    )
    true_counts = unlabeled.category_counts
    if true_counts is not None and true_counts.sum() > 0:
        title += f"; true {category_0} fraction {true_counts.iloc[0] / true_counts.sum():.2f}"
    fig.suptitle(title, y=0.995)
    # One legend row between the title and the panels, clear of both
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=5, frameon=False
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    save_figure(
        fig,
        Path(f"{labeled.name}_{unlabeled_label}_vs_groups"),
        output_directory,
        savefig_kwargs,
    )

    return weight


def plot_corner_by_category(
    data: DataContainer,
    *,
    hue_column: str,
    feature_labels: Mapping[str, str] | None = None,
    tick_overrides: Mapping[str, tuple[NpArray, Sequence[str]]] | None = None,
    output_directory: Path | None = None,
    savefig_kwargs: dict[str, Any] | None = None,
) -> None:
    """Plots one corner plot per category, colored by ``hue_column``.

    Each per-category plot is overlaid with reference KDE curves showing the overall (pooled
    across ``hue_column``) within-category and across-category feature distributions, for
    comparison.

    Args:
        data: Data container providing the features and category information to plot. Must have
            ``category_column`` set.
        hue_column: Metadata column name used to color samples within each category's plot (e.g.
            sample locality). Must be present in ``data``'s metadata.
        feature_labels: Optional mapping from feature name to a display label (e.g. with units).
            Features not present in the mapping are labeled with their raw name. Defaults to
            ``None``.
        tick_overrides: Optional mapping from a feature's display label to a
            ``(tick_positions, tick_labels)`` pair, for features whose axis should show custom
            tick marks (e.g. to un-transform a log-scaled feature back to its original units).
            Defaults to ``None``.
        output_directory: Directory to save the plot. ``None`` for no output.
        savefig_kwargs: Override keyword arguments for :func:`matplotlib.pyplot.savefig`.
            Defaults to ``None``.

    Raises:
        ValueError: If ``data.category_column`` is not set, or if ``hue_column`` is not present
            in ``data``'s metadata
    """
    plot_df, plot_vars, category_0, category_1 = _prepare_corner_plot_df(
        data, feature_labels, extra_metadata_columns=[hue_column]
    )

    def filter_category(category: str, *, exclude: bool = False) -> pd.DataFrame:
        """Filters the plotting DataFrame by category, optionally inverted."""
        mask = plot_df[data.category_column] == category

        if exclude:
            mask = ~mask

        return plot_df.loc[mask]

    for category, other_category in ((category_0, category_1), (category_1, category_0)):
        g: sns.PairGrid = sns.PairGrid(
            filter_category(category),
            hue=hue_column,
            vars=plot_vars,
            corner=False,
            diag_sharey=False,
        )
        g.map_diag(sns.kdeplot, fill=True, alpha=0.6, common_norm=True)
        g.map_lower(sns.scatterplot, alpha=0.4, s=20)
        g.map_upper(sns.kdeplot, levels=4, common_norm=True)

        # map_diag() above always populates diag_axes; it's only None before any diag mapping.
        assert g.diag_axes is not None
        for ax, var in zip(g.diag_axes, plot_vars):
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

        _apply_tick_overrides(g.figure.axes, tick_overrides)

        g.add_legend()
        sns.move_legend(g, "upper right", bbox_to_anchor=(0.4, 0.98), frameon=True)

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

        g.figure.suptitle(f"{data.name}: {category.capitalize()} by {hue_column}")
        g.figure.tight_layout()

        save_figure(
            g.figure,
            Path(f"{data.name}_{category.lower()}_by_{hue_column.lower()}"),
            output_directory,
            savefig_kwargs,
        )
