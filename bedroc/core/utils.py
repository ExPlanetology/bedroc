# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Core utils"""

import logging
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

import numpy as np

from bedroc.core.data_container import HIGH_CI_PERCENTILE, LOW_CI_PERCENTILE
from bedroc.core.type_aliases import NpArray, NpFloat

logger: logging.Logger = logging.getLogger(__name__)


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


def trim_samples(
    samples: NpArray, low_percentile: float = 0.5, high_percentile: float = 99.5
) -> NpFloat:
    """Trims samples.

    Args:
        samples: Samples to trim
        low_percentile: Low percentile for trimming. Defaults to ``0.5````.
        high_percentile: High percentile for trimming. Defaults to ``99.5``.

    Returns:
        Trimmed samples
    """
    lower_limit: np.floating = np.percentile(samples, low_percentile)
    upper_limit: np.floating = np.percentile(samples, high_percentile)

    # Filter out the extreme values
    trimmed_samples: NpFloat = samples[(samples >= lower_limit) & (samples <= upper_limit)]

    return trimmed_samples


def get_sample_summary_statistics(samples: NpFloat) -> dict[str, float]:
    """Calculates summary statistics for samples.

    Args:
        samples: Samples, usually from a posterior distribution

    Returns:
        Dictionary containing the mean, median, and 95% credible interval
    """
    mean: float = float(np.mean(samples))
    median: float = float(np.median(samples))
    upper_95: float = float(np.percentile(samples, HIGH_CI_PERCENTILE))
    lower_95: float = float(np.percentile(samples, LOW_CI_PERCENTILE))
    ci_width: float = upper_95 - lower_95

    return {
        "mean": mean,
        "median": median,
        "lower_95": lower_95,
        "upper_95": upper_95,
        "ci_width": ci_width,
    }
