# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Core utils"""

import logging
from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

import numpy as np
import pandas as pd

from bedroc import HIGH_CI_PERCENTILE, LOW_CI_PERCENTILE
from bedroc.core.type_aliases import NpArray, NpBool, NpFloat

logger: logging.Logger = logging.getLogger(__name__)


def resolve_path(p: Traversable | Path) -> Path:
    """Resolves a ``Traversable`` or ``Path`` to a concrete filesystem path.

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


@dataclass
class SummaryStatistics:
    """Summary statistics for 2D sample distributions of shape (n, n_samples)."""

    samples: NpArray
    """Samples, usually from a posterior distribution, (n, n_samples) or (n_samples,)."""
    truth: NpArray | float | None = None
    """Ground truth values. Array of shape (n,), single float, or ``None``. Defaults to
        ``None``."""

    def __post_init__(self):
        # Ensure samples is a 2D array of shape (n, n_samples)
        self.samples = np.asarray(self.samples)
        if self.samples.ndim == 1:
            self.samples = self.samples[np.newaxis, :]

        # Standardize truth shape to (n,) if provided
        if self.truth is not None:
            self.truth = np.asarray(self.truth)
            if self.truth.ndim == 0:
                self.truth = np.full(self.samples.shape[0], float(self.truth))

    @property
    def mean(self) -> NpArray:
        return np.mean(self.samples, axis=1)

    @property
    def median(self) -> NpArray:
        return np.median(self.samples, axis=1)

    @property
    def lower_95(self) -> NpArray:
        return np.percentile(self.samples, LOW_CI_PERCENTILE, axis=1)

    @property
    def upper_95(self) -> NpArray:
        return np.percentile(self.samples, HIGH_CI_PERCENTILE, axis=1)

    @property
    def ci_width(self) -> NpArray:
        return self.upper_95 - self.lower_95

    @property
    def within_ci(self) -> NpBool | None:
        if self.truth is not None:
            return self.lower_95 <= self.truth <= self.upper_95

    @property
    def error_mean(self) -> NpArray | None:
        if self.truth is not None:
            return self.mean - self.truth

    @property
    def abs_error_mean(self) -> NpArray | None:
        if self.truth is not None:
            return np.abs(self.mean - self.truth)

    @property
    def abs_error_median(self) -> NpArray | None:
        if self.truth is not None:
            return np.abs(self.median - self.truth)

    @property
    def rmse(self) -> NpArray | None:
        if self.truth is not None:
            # Subtract truth along axis 0 via explicit column broadcast
            diff = self.samples - self.truth[:, np.newaxis]  # pyright: ignore[reportIndexIssue]
            return np.sqrt(np.mean(diff**2, axis=1))

    @property
    def mae(self) -> NpArray | None:
        if self.truth is not None:
            diff = self.samples - self.truth[:, np.newaxis]  # pyright: ignore[reportIndexIssue]
            return np.mean(np.abs(diff), axis=1)

    @property
    def xerr_95(self) -> np.ndarray:
        """Returns error bounds formatted as (2, n) for matplotlib xerr."""
        err_lower = self.median - self.lower_95
        err_upper = self.upper_95 - self.median
        return np.vstack([err_lower, err_upper])

    def to_dict(self) -> dict[str, NpArray | None]:
        """Returns a dictionary mapping each metric name to its 1D array of values."""
        d: dict = {
            "mean": self.mean,
            "median": self.median,
            "lower_95": self.lower_95,
            "upper_95": self.upper_95,
            "ci_width": self.ci_width,
            "truth": self.truth,  # pyright: ignore[reportReturnType]
            "within_ci": self.within_ci,
            "error_mean": self.error_mean,
            "abs_error_mean": self.abs_error_mean,
            "abs_error_median": self.abs_error_median,
            "rmse": self.rmse,
            "mae": self.mae,
        }
        # If single-row (shape (1,)), squeeze arrays to scalar values for logging/clean dicts
        return {
            k: (v.item() if isinstance(v, np.ndarray) and v.size == 1 else v) for k, v in d.items()
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Returns a pandas DataFrame representation of the summary statistics."""
        data = {k: v for k, v in self.to_dict().items() if v is not None}

        # If the dictionary contains single scalar values, wrap in a list
        if all(np.ndim(v) == 0 for v in data.values()):
            return pd.DataFrame([data])

        return pd.DataFrame(data)
