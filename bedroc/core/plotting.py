# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Core plotting"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from bedroc.core.type_aliases import NpArray, NpFloat, NpInt
from bedroc.difference import DEFAULT_CATEGORY_COLORS

logger: logging.Logger = logging.getLogger(__name__)


SAVEFIG_KWARGS: dict[str, Any] = {"dpi": 300, "bbox_inches": "tight", "format": "pdf"}
"""Default savefig options"""


def add_xaxis_labels_to_bottom_row(figure: Figure | az.PlotCollection, label: str) -> None:
    """Add x-axis labels only to the bottom row of subplots in a figure.

    Args:
        figure: Matplotlib Figure or ArviZ PlotCollection
        label: Label to be added to the x-axis of the bottom row subplots.
    """
    if isinstance(figure, az.PlotCollection):
        fig = figure.get_viz("figure")
    else:
        fig = figure

    bottom_y = min(ax.get_position().y0 for ax in fig.axes)

    for ax in fig.axes:
        if np.isclose(ax.get_position().y0, bottom_y):
            ax.set_xlabel(label)


def get_figure(ax: Axes) -> Figure:
    """Returns an Axes' parent Figure, narrowing away the ``SubFigure``/``None`` cases
    :meth:`~matplotlib.axes.Axes.get_figure` (and the :attr:`~matplotlib.axes.Axes.figure`
    property) are typed to allow but which never occur for axes created the way this codebase
    creates them (``plt.subplots()`` or a caller-supplied top-level ``Axes``, never a subfigure).

    Args:
        ax: Matplotlib axes whose parent figure to return.

    Returns:
        The axes' parent :class:`~matplotlib.figure.Figure`.

    Raises:
        TypeError: If the axes' parent is not a genuine ``Figure`` (e.g. a ``SubFigure``), which
            would indicate the axes came from a subfigure-based layout not used anywhere here.
    """
    figure = ax.get_figure()
    if not isinstance(figure, Figure):
        raise TypeError(f"Expected ax.get_figure() to return a Figure, got {type(figure)!r}.")
    return figure


def save_figure(
    figure: Figure | az.PlotCollection,
    stem: Path | str,
    output_directory: Path | None = None,
    savefig_kwargs: dict[str, Any] | None = None,
    *,
    close_figure: bool = True,
) -> Path | None:
    """Helper function to save a figure with consistent formatting and naming

    Args:
        figure: Matplotlib Figure or ArviZ PlotCollection to save
        stem: Stem of the filename (i.e. without extension)
        output_directory: Directory to save the figure. If ``None``, the figure will not be saved.
        savefig_kwargs: Keyword arguments for :func:`matplotlib.pyplot.savefig`. Defaults to
            :obj:`SAVEFIG_KWARGS`.
        close_figure: Whether to close the underlying figure after saving. Defaults to ``True``.

    Returns:
        Path to the saved figure file, or ``None`` if the figure was not saved
    """
    if isinstance(figure, az.PlotCollection):
        figure_to_save = figure.get_viz("figure")
    else:
        figure_to_save = figure

    if output_directory is None:
        logger.warning("Output directory is None. Figure will not be saved.")
        return

    kwargs: dict[str, Any] = SAVEFIG_KWARGS.copy()
    if savefig_kwargs:
        kwargs.update(savefig_kwargs)

    fmt: str = kwargs.get("format", "pdf")
    out_path: Path = output_directory / Path(f"{stem}.{fmt}")

    figure_to_save.savefig(out_path, **kwargs)
    logger.info("Figure saved to %s", out_path)

    if close_figure:
        plt.close(figure_to_save)  # pyright: ignore[reportArgumentType]

    return out_path


def plot_group_corner(
    X: NpFloat,
    X_group_idx: NpInt,
    feature_names: ArrayLike,
    group_names: ArrayLike,
    *,
    group_colors: Sequence[str] = DEFAULT_CATEGORY_COLORS,
    title_prefix: str | None = None,
    truth_overlay: dict[str, NpArray] | None = None,
) -> sns.PairGrid:
    """Plots a corner plot for comparing the two groups with an optional overlay of truth.

    Args:
        X: Data array of shape (n_samples, n_features)
        X_group_idx: Group indices for each sample in ``X``. Array of shape (n_samples,)
        feature_names: Names of the features. Array of shape (n_features,)
        group_names: Names of the two groups. Array of shape (2,)
        group_colors: Colors for the two groups. Defaults to :obj:`DEFAULT_CATEGORY_COLORS`.
        title_prefix: Optional prefix for the plot title. If ``None``, no prefix is added.
        truth_overlay: Optional dictionary containing true ``mu_0``, ``mu_1``, and optionally
            ``sigma`` values for overlaying on the plot. Defaults to ``None``.

    Returns:
        Pairgrid
    """
    feature_names = np.asarray(feature_names)
    group_names = np.asarray(group_names)

    # Build DataFrame for seaborn
    df: pd.DataFrame = pd.DataFrame(X, columns=feature_names)
    df["Group"] = group_names[X_group_idx]

    # Create corner plot
    pairgrid: sns.PairGrid = sns.pairplot(
        df,
        hue="Group",
        hue_order=group_names,
        corner=True,
        # diag_kind="hist",
        plot_kws=dict(alpha=0.4, s=20),
        diag_kws=dict(alpha=0.6, common_norm=False),
    )

    if truth_overlay is not None:
        # Overlay true means and 1 sigma bands on diagonal
        mu_0: NpFloat | None = truth_overlay.get("mu_0")
        mu_1: NpFloat | None = truth_overlay.get("mu_1")
        sigma: NpFloat | None = truth_overlay.get("sigma")

        def plot_helper(mu: NpFloat | None, color: str) -> None:
            if mu is not None:
                # diag_axes is only None before any diagonal mapping; already populated above.
                assert pairgrid.diag_axes is not None
                for i, ax in enumerate(pairgrid.diag_axes):
                    ax.axvline(
                        mu[i],
                        color=color,
                        linestyle="--",
                        linewidth=2,
                        label="_nolegend_",
                        zorder=1,
                    )
                    if sigma is not None:
                        ax.axvspan(
                            mu[i] - sigma[i],
                            mu[i] + sigma[i],
                            color=color,
                            alpha=0.1,
                            zorder=0,
                        )

        plot_helper(mu_0, group_colors[0])
        plot_helper(mu_1, group_colors[1])

        # Off-diagonal: true multivariate centers
        for row in range(len(feature_names)):  # row index in axes
            for col in range(row):  # col index in axes
                ax: Axes = pairgrid.axes[row, col]
                if mu_0 is not None:
                    ax.plot(
                        mu_0[col],
                        mu_0[row],
                        "o",
                        color=group_colors[0],
                        markersize=8,
                        markeredgecolor="k",
                        label="_nolegend_",
                    )
                if mu_1 is not None:
                    ax.plot(
                        mu_1[col],
                        mu_1[row],
                        "o",
                        color=group_colors[1],
                        markersize=8,
                        markeredgecolor="k",
                        label="_nolegend_",
                    )

    sns.move_legend(pairgrid, "upper left", bbox_to_anchor=(0.18, 0.8), frameon=True)

    if title_prefix is not None:
        pairgrid.figure.suptitle(f"{title_prefix}: {group_names[1]} vs {group_names[0]}")
    else:
        pairgrid.figure.suptitle(f"{group_names[1]} vs {group_names[0]}")

    return pairgrid
