# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared test setup for bedroc.difference tests.

Sets the BLAS thread-count environment variables before numpy/pymc are imported anywhere in the
test session: on macOS, numpy's Accelerate BLAS backend spins up its internal thread pool at
import time, and letting each PyMC multiprocessing sampling worker do so independently causes
workers to crash silently (surfacing as an unhelpful EOFError from pm.sample()). Mirrors the same
mitigation in bedroc/applications/zircons/run.py.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from collections.abc import Callable

import numpy as np
import pytest

from bedroc.core.type_aliases import NpFloat, NpInt

SyntheticTwoCategoryFactory = Callable[..., tuple[NpFloat, NpInt, NpFloat]]


@pytest.fixture
def make_synthetic_two_category() -> SyntheticTwoCategoryFactory:
    """Returns a factory for building small, self-contained synthetic two-category datasets.

    A fixture that hands back a plain factory function (rather than a single dataset), so every
    test using it builds its own independent dataset with its own parameters and random seed —
    no state is shared between tests, and pytest's conftest.py discovery makes this usable from
    any test under ``tests/difference/``, including subdirectories (e.g. ``tests/difference/
    models/``), without a fragile cross-directory ``import``.

    The returned factory's signature:
        ``factory(n_train_per_category, n_unlabeled, n_features, effect_size, *, random_seed,
        correlated=False) -> (X_train, X_category_idx_train, X_unlabeled)``

        n_train_per_category: Number of labeled training samples per category.
        n_unlabeled: Number of unlabeled samples (drawn 50/50 across categories).
        n_features: Number of features.
        effect_size: Standardized mean separation between the two categories.
        random_seed: Random seed for reproducibility.
        correlated: If True, features share a single latent factor (inducing real correlation,
            needed to get a tempering factor meaningfully below 1). If False, features are drawn
            independently. Defaults to False.
    """

    def factory(
        n_train_per_category: int,
        n_unlabeled: int,
        n_features: int,
        effect_size: float,
        *,
        random_seed: int,
        correlated: bool = False,
    ) -> tuple[NpFloat, NpInt, NpFloat]:
        rng = np.random.default_rng(random_seed)

        def draw(n: int, offset: float) -> NpFloat:
            if correlated:
                latent = rng.normal(size=(n, 1))
                loadings = rng.normal(size=(1, n_features))
                noise = rng.normal(size=(n, n_features)) * 0.3
                values = latent @ loadings + noise
            else:
                values = rng.normal(size=(n, n_features))
            values[:, 0] += offset
            return values

        X_0 = draw(n_train_per_category, -effect_size / 2)
        X_1 = draw(n_train_per_category, effect_size / 2)
        X_train = np.vstack([X_0, X_1])
        X_category_idx_train = np.array([0] * n_train_per_category + [1] * n_train_per_category)

        n_0_unlabeled = n_unlabeled // 2
        n_1_unlabeled = n_unlabeled - n_0_unlabeled
        X_unlabeled = np.vstack(
            [draw(n_0_unlabeled, -effect_size / 2), draw(n_1_unlabeled, effect_size / 2)]
        )

        return X_train, X_category_idx_train, X_unlabeled

    return factory
