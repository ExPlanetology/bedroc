# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Synthetic data generation for group difference modeling"""

import logging
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from bedroc.core import RANDOM_SEED
from bedroc.type_aliases import NpArray, NpFloat, NpInt

logger: logging.Logger = logging.getLogger(__name__)


# TODO: Needs refreshing to work with new group-difference modeling framework.
class SyntheticDataGenerator:
    """Generates synthetic multivariate data for two types (A & B) with configurable parameters.

    Args:
        n_samples: Number of samples per type. Defaults to ``100``.
        n_features: Number of features per sample. Defaults to ``5``.
        feature_offsets: Optional shift to apply to the Type B feature means relative to Type A.
            May be either a scalar (applied to every feature) or an array of shape
            ``(n_features,)`` specifying per-feature offsets. Defaults to ``1.0``.
        feature_sigma: Standard deviation of the noise (stddev) for features. May be either a
            scalar (applied to every feature) or an array of shape ``(n_features,)`` specifying
            per-feature noise. Defaults to ``0.5``.
        random_seed: Optional seed for reproducibility. Defaults to :obj:`RANDOM_SEED`.
        output_directory: Optional path to save generated data. Defaults to ``None`` (no saving).
    """

    def __init__(
        self,
        n_samples: int = 100,
        *,
        n_features: int = 5,
        feature_offsets: ArrayLike = 1.0,
        feature_sigma: ArrayLike = 0.5,
        random_seed: int | None = RANDOM_SEED,
        output_directory: Path | None = None,
    ):
        self.n_samples: int = n_samples
        self.n_features: int = n_features
        self.feature_offsets: NpFloat = np.full(self.n_features, feature_offsets, dtype=float)
        self.feature_sigma: NpFloat = np.full(self.n_features, feature_sigma, dtype=float)
        self.random_seed: int | None = random_seed
        self.output_directory: Path | None = output_directory
        self._rng = np.random.default_rng(self.random_seed)

        # For Type A, each feature gets its own true mean (center of distribution)
        self.mu_A: NpFloat = self._rng.normal(loc=0.0, scale=1.0, size=self.n_features)
        logger.debug("mu_A = %s", self.mu_A)

        # Shift distribution of Type B relative to Type A by the specified offsets
        self.mu_B: NpFloat = self.mu_A + self.feature_offsets
        logger.debug("mu_B = %s", self.mu_B)

        # Internal storage for generated data
        self._X: NpFloat | None = None
        self._X_group_idx: NpInt | None = None

    @property
    def X(self) -> NpArray:
        """Type A data (n_samples, n_features)"""
        if self._X is None:
            raise ValueError("Data not yet generated. Call 'generate()' first.")

        return self._X

    @property
    def X_group_idx(self) -> NpInt:
        """Group idx"""
        if self._X_group_idx is None:
            raise ValueError("Data not yet generated. Call 'generate()' first.")

        return self._X_group_idx

    def generate(self) -> None:
        """Generates multivariate data for 2 types (A & B) and stores internally."""

        logger.info("Generating synthetic data with random_seed=%s", self.random_seed)

        # Generate samples
        X_A: NpFloat = self._rng.normal(
            self.mu_A, self.feature_sigma, size=(self.n_samples, self.n_features)
        )
        logger.debug("X_A = %s", X_A)
        X_B: NpFloat = self._rng.normal(
            self.mu_B, self.feature_sigma, size=(self.n_samples, self.n_features)
        )
        logger.debug("X_B = %s", X_B)

        # Store internally
        self._X = np.vstack([X_A, X_B])
        self._X_group_idx = np.hstack(
            [np.zeros(X_A.shape[0], dtype=int), np.ones(X_B.shape[0], dtype=int)]
        )

        logger.info(
            "Synthetic data generation complete. Generated %d samples per type with %d features.",
            self.n_samples,
            self.n_features,
        )

    def generate_out_of_sample_data(self, n_samples: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Generates out-of-sample synthetic data using previously-sampled true parameters.

        Args:
            n_samples: Number of out-of-sample points per type. Defaults to ``100``.

        Returns:
            tuple:
                - Type A data (n_samples, n_features)
                - Type B data (n_samples, n_features)
        """
        # Draw new samples from the same ground-truth distribution
        X_A_test: NpFloat = self._rng.normal(
            self.mu_A, self.feature_sigma, size=(n_samples, self.n_features)
        )
        X_B_test: NpFloat = self._rng.normal(
            self.mu_B, self.feature_sigma, size=(n_samples, self.n_features)
        )

        X_test = np.vstack([X_A_test, X_B_test])
        X_test_group_idx = np.hstack(
            [np.zeros(X_A_test.shape[0], dtype=int), np.ones(X_B_test.shape[0], dtype=int)]
        )

        return X_test, X_test_group_idx
