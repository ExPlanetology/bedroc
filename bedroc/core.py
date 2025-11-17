#
# Copyright 2025 Dan J. Bower
#
# This file is part of Bedroc.
#
# Bedroc is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Bedroc is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Bedroc. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Core"""

import logging
from pathlib import Path
from typing import Any, cast

import arviz as az
import numpy as np
import pymc as pm
from arviz import InferenceData
from matplotlib.figure import Figure

logger: logging.Logger = logging.getLogger(__name__)

savefig_opts: dict[str, Any] = {"dpi": 300, "bbox_inches": "tight", "format": "pdf"}
"""Figure options for savefig"""


def plot_posterior_predictive(
    model: pm.Model,
    idata: InferenceData,
    *,
    savefig: bool = False,
    filename_prefix: Path | str = "posterior_predictive_check",
    thinning_factor: int = 5,
    suptitle_fontsize: Any = "xx-large",
    **kwargs,
) -> Figure:
    """Plots posterior predictive check (in-sample predictions).

    This performs in-sample predictions to assess how well the model fits the observed data,
    i.e., test how well the model can reproduce the data it was trained on.

    Args:
        model: PyMC model object
        idata: Trace data from sampling
        savefig: Saves the figure to a file. Defaults to ``False``.
        filename_prefix: Prefix for the saved figure filename. Defaults to
            "posterior_predictive_check".
        thinning_factor: Thinning factor for posterior samples to reduce overplotting.
            Defaults to ``5``.
        suptitle_fontsize: Fontsize for the super title. Defaults to ``xx-large``.
        **kwargs: Keyword arguments for :func:`pymc.sample_posterior_predictive`

    Returns:
        Figure
    """
    thinned_idata: InferenceData = cast(
        InferenceData, idata.sel(draw=slice(None, None, thinning_factor))
    )
    posterior_predictive: InferenceData = pm.sample_posterior_predictive(
        thinned_idata, model=model, **kwargs
    )

    axes = az.plot_ppc(posterior_predictive, group="posterior", observed=True)

    # Get the Figure safely
    if isinstance(axes, np.ndarray):
        figure: Figure = axes.flatten()[0].figure
    else:
        figure = axes.figure

    figure.suptitle("Posterior Predictive Check", fontsize=suptitle_fontsize)

    if savefig:  # pragma: no cover
        figure.savefig(f"{filename_prefix}.{savefig_opts['format']}", **savefig_opts)

    return figure


def plot_prior_predictive(
    model: pm.Model,
    *,
    savefig: bool = False,
    filename_prefix: Path | str = "prior_predictive_check",
    suptitle_fontsize: Any = "xx-large",
    **kwargs,
) -> Figure:
    """Plots prior predictive check.

    This plot is used to determine if the model can generate data plausibly shaped like the
    observed distributions.

    Args:
        model: PyMC model object
        savefig: Saves the figure to a file. Defaults to ``False``.
        filename_prefix: Prefix for the saved figure filename. Defaults to
            "prior_predictive_check".
        suptitle_fontsize: Fontsize for the super title. Defaults to ``xx-large``.
        **kwargs: Keyword arguments for :func:`pymc.sample_prior_predictive`

    Returns:
        Figure
    """
    prior_predictive: InferenceData = pm.sample_prior_predictive(model=model, **kwargs)

    axes = az.plot_ppc(prior_predictive, group="prior", observed=True)

    # Get the Figure safely
    if isinstance(axes, np.ndarray):
        figure: Figure = axes.flatten()[0].figure
    else:
        figure = axes.figure

    figure.suptitle("Prior Predictive Check", fontsize=suptitle_fontsize)

    if savefig:  # pragma: no cover
        figure.savefig(f"{filename_prefix}.{savefig_opts['format']}", **savefig_opts)

    return figure
