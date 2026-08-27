# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Plotting utilities for group difference models"""

import logging
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy.stats import beta, gaussian_kde

from bedroc.core.type_aliases import NpArray, NpFloat
from bedroc.core.utils import SummaryStatistics
from bedroc.difference import DEFAULT_CATEGORY_COLORS

logger: logging.Logger = logging.getLogger(__name__)


def plot_group_fraction_posterior(
    pi_0_samples: NpFloat,
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    bins: int = 50,
    n_grid: int = 101,
    category_names: Sequence,
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
