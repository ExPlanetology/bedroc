# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Plotting of results for Bayesian hierarchical model for group-centric comparison of two groups"""

import logging

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bedroc.core.type_aliases import NpArray
from bedroc.difference.utils import distribution_overlap

logger: logging.Logger = logging.getLogger(__name__)


def plot_distribution_overlap(
    values_0: NpArray,
    values_1: NpArray,
    *,
    ax: Axes | None = None,
    n_grid: int = 2000,
    labels: tuple[str, str] = ("Population 0", "Population 1"),
) -> tuple[Figure, Axes, float]:
    """Plots two distributions and their overlap.

    The samples, KDEs, and overlapping probability density are shown.

    Args:
        values_0: Samples from the first distribution.
        values_1: Samples from the second distribution.
        ax: Matplotlib axes on which to plot. If ``None``, a new figure and axes are created.
        n_grid: Number of points to use for the grid over which to evaluate the PDFs. Defaults to
            ``2000``.
        labels: Labels for the two populations.

    Returns:
        Matplotlib figure and axes.
    """
    x, pdf_0, pdf_1, overlap_density, overlap = distribution_overlap(
        values_0, values_1, n_grid=n_grid
    )

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Plot samples as rug marks
    # ax.plot(values_0, np.zeros_like(values_0), "|", alpha=0.3, markersize=8)
    # ax.plot(values_1, np.zeros_like(values_1), "|", alpha=0.3, markersize=8)

    # Plot KDEs
    ax.plot(x, pdf_0, color="blue", linewidth=2, label=labels[0])
    ax.plot(x, pdf_1, color="orange", linewidth=2, label=labels[1])

    # Shade the overlap
    ax.fill_between(x, overlap_density, alpha=0.3, label=f"Overlap (OVL = {overlap:.2f})")

    ax.set_xlabel("Standardized units")
    ax.set_ylabel("Density")
    ax.legend()

    return fig, ax, overlap  # pyright: ignore[reportReturnType]
