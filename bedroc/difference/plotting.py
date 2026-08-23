# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Plotting utilities for group difference models"""

import logging
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.stats import beta, gaussian_kde

from bedroc.core.type_aliases import NpArray, NpFloat
from bedroc.core.utils import get_sample_summary_statistics

logger: logging.Logger = logging.getLogger(__name__)


def plot_group_fraction_posterior(
    pi_0_samples: NpFloat,
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    bins: int = 50,
    n_grid: int = 2001,
    group_names: Iterable[str] = ("Group 0", "Group 1"),
    group_colors: tuple[str, str] = ("tab:blue", "tab:orange"),
    group_counts: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot the posterior distribution of group fractions.

    The posterior is shown together with the beta prior and, where available, the observed
    group fraction.

    Args:
        pi_0_samples: Samples from the posterior distribution of the group-0 fraction.
        prior_alpha: Alpha parameter of the beta prior. Defaults to ``1.0``.
        prior_beta: Beta parameter of the beta prior. Defaults to ``1.0``.
        bins: Number of bins for the histogram. Defaults to ``50``.
        n_grid: Number of grid points for the prior and perfect-classification limit. Defaults to
            ``2001``.
        group_names: Names for the two groups. Defaults to ``("Group 0", "Group 1")``.
        group_colors: Colors for the two groups. Defaults to ``("tab:blue", "tab:orange")``.
        group_counts: Known counts for the two groups. If ``None``, the observed fractions are not
            plotted. Defaults to ``None``.
        ax: Matplotlib axes on which to plot. If ``None``, a new figure and axes are created.

    Returns:
        Matplotlib axes containing the posterior group-fraction plot
    """
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("prior_alpha and prior_beta must be > 0.")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    group_0, group_1 = group_names

    grid: NpArray = np.linspace(0, 1, n_grid)

    def plot_posterior(label: str, samples: NpFloat, color: str, ci_y_loc: float) -> None:
        stats = get_sample_summary_statistics(samples)
        lower, upper = stats["lower_95"], stats["upper_95"]

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
            stats["median"],
            ci_y_loc,
            xerr=[
                [stats["median"] - stats["lower_95"]],
                [stats["upper_95"] - stats["median"]],
            ],
            fmt="o",
            color=color,
            capsize=4,
            capthick=2,
            elinewidth=2,
        )

    plot_posterior(str(group_0), pi_0_samples, group_colors[0], ci_y_loc=0.4)
    plot_posterior(str(group_1), 1.0 - pi_0_samples, group_colors[1], ci_y_loc=0.6)

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
        label=rf"{group_0} prior",  #: beta($\alpha={prior_alpha:g},\ \beta={prior_beta:g}$)",
    )

    # Observed fractions, if available
    if group_counts is not None:
        # Perfect-classification limit for group 0
        limiting_posterior_0: NpFloat = beta.pdf(
            grid, prior_alpha + group_counts[0], prior_beta + group_counts[1]
        )

        ax.plot(
            grid,
            limiting_posterior_0,
            color="tab:blue",
            linestyle="--",
            linewidth=2,
            label="Perfect-classification limit",
        )

        observed_fraction_0 = group_counts[0] / sum(group_counts)
        observed_fraction_1 = group_counts[1] / sum(group_counts)

        ax.annotate(
            f"Obs\n{observed_fraction_0:.2f}",
            xy=(observed_fraction_0, 0.6),
            xytext=(observed_fraction_0, 1.8),
            ha="center",
            va="bottom",
            color=group_colors[0],
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.9),
            arrowprops=dict(arrowstyle="-|>", color=group_colors[0], lw=1.5),
        )

        ax.annotate(
            f"Obs\n{observed_fraction_1:.2f}",
            xy=(observed_fraction_1, 0.8),
            xytext=(observed_fraction_1, 2.2),
            ha="center",
            va="bottom",
            color=group_colors[1],
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.9),
            arrowprops=dict(arrowstyle="-|>", color=group_colors[1], lw=1.5),
        )

        stats_0 = get_sample_summary_statistics(pi_0_samples)
        is_within_cri = stats_0["lower_95"] <= observed_fraction_0 <= stats_0["upper_95"]

        logger.info(
            "Is observed fraction (%.2f) within 95%% CrI [%.2f, %.2f] for %s? %s",
            observed_fraction_0,
            stats_0["lower_95"],
            stats_0["upper_95"],
            group_0,
            is_within_cri,
        )

    ax.set(xlabel="Population fraction", ylabel="Density", xlim=(0, 1))
    ax.set_title("Posterior distribution of group fractions")

    ax.legend()

    return ax
