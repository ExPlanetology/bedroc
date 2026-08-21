# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Core plotting"""

import logging
from pathlib import Path
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

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
        close_figure: Whether to close the underlying figure after saving. Defaults to
            ``True``.

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
