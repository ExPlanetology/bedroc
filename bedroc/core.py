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
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import cast

import arviz as az
import numpy as np
import pymc as pm
from arviz import InferenceData
from matplotlib.axes import Axes

from bedroc.type_aliases import NpArray, NpFloat

logger: logging.Logger = logging.getLogger(__name__)


def plot_posterior_predictive(
    model: pm.Model, idata: InferenceData, *, thinning_factor: int = 5, **kwargs
) -> Axes:
    """Plots posterior predictive check (in-sample predictions).

    This performs in-sample predictions to assess how well the model fits the observed data,
    i.e., test how well the model can reproduce the data it was trained on.

    Args:
        model: PyMC model object
        idata: Trace data from sampling
        thinning_factor: Thinning factor for posterior samples to reduce overplotting.
            Defaults to ``5``.
        **kwargs: Keyword arguments for :func:`pymc.sample_posterior_predictive`

    Returns:
        Axes
    """
    thinned_idata: InferenceData = cast(
        InferenceData, idata.sel(draw=slice(None, None, thinning_factor))
    )
    posterior_predictive: InferenceData = pm.sample_posterior_predictive(
        thinned_idata, model=model, **kwargs
    )

    axes: Axes = az.plot_ppc(posterior_predictive, group="posterior", observed=True)

    return axes


def plot_prior_predictive(model: pm.Model, **kwargs) -> Axes:
    """Plots prior predictive check.

    This plot is used to determine if the model can generate data plausibly shaped like the
    observed distributions.

    Args:
        model: PyMC model object
        **kwargs: Keyword arguments for :func:`pymc.sample_prior_predictive`

    Returns:
        Figure
    """
    prior_predictive: InferenceData = pm.sample_prior_predictive(model=model, **kwargs)

    axes: Axes = az.plot_ppc(prior_predictive, group="prior", observed=True)

    return axes


def trim_samples(samples: NpArray) -> NpFloat:
    """Trims samples.

    Args:
        samples: Samples to trim

    Returns:
        Trimmed samples
    """
    # Define the percentage of extreme values to exclude from the hist plot
    # (e.g., 0.5% from each end)
    lower_percentile: float = 0.5
    upper_percentile: float = 99.5

    lower_limit: np.floating = np.percentile(samples, lower_percentile)
    upper_limit: np.floating = np.percentile(samples, upper_percentile)

    # Filter out the extreme values
    trimmed_samples: NpFloat = samples[(samples >= lower_limit) & (samples <= upper_limit)]

    return trimmed_samples


def resolve_path(p: Traversable | Path) -> Path:
    """Resolve a ``Traversable`` or ``Path`` to a concrete filesystem path.

    This function ensures that resources packaged using ``importlib.resources`` (e.g., files inside
    wheels or zipped packages) are converted into a real ``Path`` object. If ``p`` is already a
    ``Path``, it is returned unchanged. Otherwise, the underlying resource is extracted to a
    temporary location and its path is returned.

    Note:
        The temporary file extracted for ``Traversable`` objects is valid only for the duration of
        the context in which it is created. Since this function returns the resolved ``Path``
        inside the context manager, the file is guaranteed to exist when the function returns.

    Args:
        p: A filesystem ``Path`` or an ``importlib.resources.Traversable`` object.

    Returns:
        Path: A concrete filesystem path pointing to the resolved resource
    """
    if isinstance(p, Path):
        return p
    with resources.as_file(p) as temp:
        return Path(temp)
