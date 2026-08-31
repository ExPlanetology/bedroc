# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the oracle ceiling (bedroc.difference.utils.oracle_pi0_posterior and each joint
model's oracle_ceiling_pdf())."""

import numpy as np
import pytest

from bedroc.difference.models.tempered_full import TemperedFullModel
from bedroc.difference.models.tempered_likelihood import TemperedLikelihoodModel
from bedroc.difference.utils import compute_tempering_scale, oracle_pi0_posterior


def _ci_width(grid: np.ndarray, density: np.ndarray) -> tuple[float, float]:
    """Returns (mean, 95% CI width) for a (grid, density) posterior."""
    cdf = np.concatenate([[0.0], np.cumsum((density[1:] + density[:-1]) / 2 * np.diff(grid))])
    cdf /= cdf[-1]
    lower = np.interp(0.025, cdf, grid)
    upper = np.interp(0.975, cdf, grid)
    mean = np.trapezoid(grid * density, grid)
    return float(mean), float(upper - lower)


@pytest.mark.pymc
def test_oracle_bracketing_and_center(make_synthetic_two_category) -> None:
    """The oracle ceiling's width never exceeds the model's actual posterior width, and its
    center tracks the actual posterior's center reasonably closely -- the theoretical guarantee
    and empirical property both verified numerically this session. One fit covers both
    assertions."""
    X_train, X_category_idx_train, X_unlabeled = make_synthetic_two_category(
        n_train_per_category=150, n_unlabeled=80, n_features=3, effect_size=2.0, random_seed=0
    )

    model = TemperedLikelihoodModel("t", X_train, X_category_idx_train, X_unlabeled)
    model.build_model()
    model.run_inference(draws=300, tune=300, chains=1, cores=1, progressbar=False)

    grid, density = model.oracle_ceiling_pdf()
    oracle_mean, oracle_width = _ci_width(grid, density)

    pi_0_samples = model.pi_0_samples()
    actual_mean = float(pi_0_samples.mean())
    actual_width = float(
        np.percentile(pi_0_samples, 97.5) - np.percentile(pi_0_samples, 2.5)
    )

    assert oracle_width <= actual_width + 1e-6
    assert abs(oracle_mean - actual_mean) < 0.05


@pytest.mark.pymc
def test_tempered_full_oracle_uses_tempered_prior(make_synthetic_two_category) -> None:
    """TemperedFullModel.oracle_ceiling_pdf() must use the *tempered* pi_0 prior actually fitted
    by build_model(), not the raw untempered prior_alpha/prior_beta arguments -- a direct
    regression test for a real bug found and fixed this session."""
    X_train, X_category_idx_train, X_unlabeled = make_synthetic_two_category(
        n_train_per_category=100,
        n_unlabeled=60,
        n_features=4,
        effect_size=1.5,
        random_seed=1,
        correlated=True,
    )
    prior_alpha, prior_beta = 2.0, 5.0

    model = TemperedFullModel("t", X_train, X_category_idx_train, X_unlabeled)
    model.build_model(prior_alpha=prior_alpha, prior_beta=prior_beta)
    model.run_inference(draws=300, tune=300, chains=1, cores=1, progressbar=False)

    alpha_val = compute_tempering_scale(model.X, model.X_category_idx)
    # Tempering must actually matter for this test to be meaningful.
    assert alpha_val < 0.95

    actual_grid, actual_density = model.oracle_ceiling_pdf()

    tempered_grid, tempered_density = oracle_pi0_posterior(
        model.model,
        model.idata,
        prior_alpha=alpha_val * (prior_alpha - 1.0) + 1.0,
        prior_beta=alpha_val * (prior_beta - 1.0) + 1.0,
    )
    untempered_grid, untempered_density = oracle_pi0_posterior(
        model.model, model.idata, prior_alpha=prior_alpha, prior_beta=prior_beta
    )

    np.testing.assert_allclose(actual_density, tempered_density)
    assert not np.allclose(actual_density, untempered_density)
