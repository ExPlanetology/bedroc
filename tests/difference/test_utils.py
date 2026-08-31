# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for bedroc.difference.utils"""

import numpy as np
from scipy.integrate import quad
from scipy.stats import beta as beta_dist

from bedroc.difference.utils import (
    compute_tempering_scale,
    distribution_overlap,
    effect_size_from_overlap,
    participation_ratio,
)


def test_participation_ratio_bounds() -> None:
    """Identity correlation -> every feature is independent (n_eff == n_features); a fully
    correlated (all-ones) correlation matrix -> effectively one feature (n_eff == 1)."""
    n_features = 5

    identity = np.eye(n_features)
    np.testing.assert_allclose(participation_ratio(identity), n_features)

    fully_correlated = np.ones((n_features, n_features))
    np.testing.assert_allclose(participation_ratio(fully_correlated), 1.0)


def test_compute_tempering_scale_bounds() -> None:
    """alpha lands in [1/F, 1] for random correlated data."""
    rng = np.random.default_rng(0)
    n_features = 4

    # Deliberately correlated features (shared latent factor), so alpha is meaningfully < 1.
    latent = rng.normal(size=(200, 1))
    X = latent @ rng.normal(size=(1, n_features)) + rng.normal(size=(200, n_features)) * 0.3
    category_idx = rng.integers(0, 2, size=200)

    alpha = compute_tempering_scale(X, category_idx)

    assert 1.0 / n_features <= alpha <= 1.0


def test_effect_size_from_overlap_roundtrip() -> None:
    """Inverting a target overlap via effect_size_from_overlap, generating two Gaussian samples
    with that effect size, and recomputing the overlap via distribution_overlap should
    approximately recover the original target."""
    rng = np.random.default_rng(0)
    target_overlap = 0.3

    delta = effect_size_from_overlap(target_overlap)

    values_0 = rng.normal(loc=0.0, scale=1.0, size=5000)
    values_1 = rng.normal(loc=delta, scale=1.0, size=5000)

    _, _, _, _, recovered_overlap = distribution_overlap(values_0, values_1)

    assert abs(recovered_overlap - target_overlap) < 0.03


def test_beta_prior_tempering_formula() -> None:
    """Raising a Beta(a, b) density to the power alpha and renormalizing gives exactly
    Beta(alpha*(a-1)+1, alpha*(b-1)+1) -- the correct tempering transform (a real bug in
    tempered_full.py once used Beta(alpha*a, alpha*b) instead, verified wrong this session).
    """
    x_test = np.linspace(0.05, 0.95, 19)

    def raised_pdf(x, a: float, b: float, alpha: float):
        return beta_dist.pdf(x, a, b) ** alpha

    for a, b, alpha in [(1.0, 1.0, 0.3), (1.0, 1.0, 0.8), (3.0, 2.0, 0.6), (2.0, 5.0, 0.4)]:
        normalizing_constant, _ = quad(raised_pdf, 0.0, 1.0, args=(a, b, alpha))

        actual = raised_pdf(x_test, a, b, alpha) / normalizing_constant
        expected = beta_dist.pdf(x_test, alpha * (a - 1.0) + 1.0, alpha * (b - 1.0) + 1.0)

        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    # The uniform prior (a=b=1) should stay exactly uniform under any tempering factor.
    for alpha in (0.1, 0.5, 1.0, 2.0):
        uniform_a = alpha * (1.0 - 1.0) + 1.0
        uniform_b = alpha * (1.0 - 1.0) + 1.0
        expected_uniform = beta_dist.pdf(x_test, uniform_a, uniform_b)
        np.testing.assert_allclose(expected_uniform, np.ones_like(x_test))
