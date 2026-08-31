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


def pooled_within_category_covariance(
    group_0: NpArray | pd.DataFrame, group_1: NpArray | pd.DataFrame
) -> pd.DataFrame:
    """Computes the sample-size-weighted pooled covariance of two groups.

    ``((n0-1)*Cov0 + (n1-1)*Cov1) / (n0+n1-2)`` — the standard pooled-within-group covariance
    estimator (the same two-sample pooled-variance formula used in a Student's t-test, generalized
    to matrices): an unbiased estimate of a covariance assumed shared between the two groups,
    correctly accounting for each group's own sample size and spread rather than averaging the two
    groups' own covariances (or correlations) as if they carried equal weight.

    Uses pandas' pairwise-complete-observations handling of missing values (via
    :meth:`pandas.DataFrame.cov`), so a missing value in one feature doesn't propagate into an
    entire row/column of the result the way ``numpy.cov`` would.

    Args:
        group_0: Observations for group 0, shape ``(n0, n_features)``. A plain array gets default
            integer column labels; a DataFrame's own column labels are preserved.
        group_1: Observations for group 1, shape ``(n1, n_features)``, matching ``group_0``'s
            columns.

    Returns:
        Pooled covariance matrix, shape ``(n_features, n_features)``, labeled by ``group_0``'s
        columns.
    """
    df_0 = pd.DataFrame(group_0)
    df_1 = pd.DataFrame(group_1)

    n_0, n_1 = len(df_0), len(df_1)

    return ((n_0 - 1) * df_0.cov(ddof=1) + (n_1 - 1) * df_1.cov(ddof=1)) / (n_0 + n_1 - 2)


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


def eigen_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    """Eigendecomposes a symmetric matrix (e.g. a covariance or correlation matrix).

    Args:
        matrix: Symmetric matrix, indexed and labeled by feature name along both axes (e.g. from
            :meth:`~bedroc.core.data_container.DataDiagnostics.covariance_matrix`).

    Returns:
        Summary dataframe with one column per eigenvector (labeled ``PC1``, ``PC2``, ...,
        ordered from largest to smallest eigenvalue), one row per feature giving that feature's
        loading, and two trailing rows giving each eigenvector's eigenvalue and
        explained-variance ratio.
    """
    eigenvalues: NpFloat
    eigenvectors: NpFloat
    eigenvalues, eigenvectors = np.linalg.eigh(matrix.to_numpy())

    order: NpArray = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    columns: list[str] = [f"PC{i + 1}" for i in range(len(eigenvalues))]
    summary: pd.DataFrame = pd.DataFrame(eigenvectors, index=matrix.index, columns=columns)
    summary.loc["eigenvalue"] = eigenvalues
    summary.loc["explained variance ratio"] = eigenvalues / eigenvalues.sum()

    return summary


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
            return (self.lower_95 <= self.truth) & (self.truth <= self.upper_95)

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
            # __post_init__ always normalizes a non-None truth to an array, never a bare float.
            assert isinstance(self.truth, np.ndarray)
            # Subtract truth along axis 0 via explicit column broadcast
            diff = self.samples - self.truth[:, np.newaxis]
            return np.sqrt(np.mean(diff**2, axis=1))

    @property
    def mae(self) -> NpArray | None:
        if self.truth is not None:
            assert isinstance(self.truth, np.ndarray)
            diff = self.samples - self.truth[:, np.newaxis]
            return np.mean(np.abs(diff), axis=1)

    @property
    def xerr_95(self) -> np.ndarray:
        """Returns error bounds formatted as (2, n) for matplotlib xerr."""
        err_lower = self.median - self.lower_95
        err_upper = self.upper_95 - self.median
        return np.vstack([err_lower, err_upper])

    def to_dict(self) -> dict[str, NpArray | NpBool | float | bool | None]:
        """Returns a dictionary mapping each metric name to its value.

        Single-element arrays are squeezed to plain Python scalars (see below), so values are not
        always arrays despite each metric being array-valued in general.
        """
        d: dict = {
            "mean": self.mean,
            "median": self.median,
            "lower_95": self.lower_95,
            "upper_95": self.upper_95,
            "ci_width": self.ci_width,
            "truth": self.truth,
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

    def log_summary(
        self, message: str = "Summary statistics", *, level: int = logging.INFO
    ) -> None:
        """Logs the mean, median, and 95% CI for each row.

        Args:
            message: Prefix for the log message. Defaults to ``"Summary statistics"``.
            level: Logging level. Defaults to ``logging.INFO``.
        """
        for i in range(self.samples.shape[0]):
            logger.log(
                level,
                "%s: mean=%.4f, median=%.4f, 95%% CI=[%.4f, %.4f]",
                message,
                self.mean[i],
                self.median[i],
                self.lower_95[i],
                self.upper_95[i],
            )

    def to_dataframe(self) -> pd.DataFrame:
        """Returns a pandas DataFrame representation of the summary statistics."""
        data = {k: v for k, v in self.to_dict().items() if v is not None}

        # If the dictionary contains single scalar values, wrap in a list
        if all(np.ndim(v) == 0 for v in data.values()):
            return pd.DataFrame([data])

        return pd.DataFrame(data)
