# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Validation functions for observation data used in category difference modeling."""

import numpy as np

from bedroc.core.type_aliases import NpFloat, NpInt


def validate_observation_data(
    X: NpFloat, *, X_sigma: NpFloat | None = None
) -> tuple[NpFloat, NpFloat]:
    """Validates observation data.

    Args:
        X: Observation matrix with shape ``(n_samples, n_features)``. Missing values should be
            represented by ``NaN``.
        X_sigma: Optional 1-sigma uncertainties with the same shape as ``X``. ``NaN`` values are
            treated as zero uncertainty. If ``None``, uncertainties are assumed to be zero.

    Returns:
        Tuple containing validated ``X`` and ``X_sigma`` arrays

    Raises:
        ValueError: If the input arrays have invalid dimensions, shapes, or uncertainties
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array.")

    if np.any(np.isinf(X)):
        raise ValueError("X must not contain infinite values; use NaN for missing values.")

    if X_sigma is None:
        X_sigma = np.zeros_like(X, dtype=float)
    else:
        X_sigma = np.asarray(X_sigma, dtype=float)

        if X_sigma.shape != X.shape:
            raise ValueError(
                f"X_sigma must have the same shape as X ({X.shape}), got {X_sigma.shape}."
            )

        if np.any(np.isinf(X_sigma)):
            raise ValueError(
                "X_sigma must not contain infinite values; use NaN for missing values."
            )

        if np.any(X_sigma < 0):
            raise ValueError("X_sigma must contain only non-negative values.")

        X_sigma = np.nan_to_num(X_sigma, nan=0.0)

    return X, X_sigma


def validate_category_idx(category_idx: NpInt, n_samples: int) -> NpInt:
    """Validates a sample-level binary category index.

    Args:
        category_idx: Array of shape ``(n_samples,)`` containing binary category indices (0 or 1).
        n_samples: Number of samples in the observation data

    Returns:
        Validated ``category_idx`` array

    Raises:
        ValueError: If ``category_idx`` has an invalid shape or contains values other than 0 or 1
    """
    category_idx = np.asarray(
        category_idx
    )  # avoid silently coercing to int, which can change values

    if category_idx.shape != (n_samples,):
        raise ValueError(f"category_idx must have shape ({n_samples},), got {category_idx.shape}.")

    if not np.all(np.isin(category_idx, [0, 1])):
        raise ValueError("category_idx must contain only 0 or 1.")

    return category_idx
