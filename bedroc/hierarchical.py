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
"""Utilities for building and working with Bayesian hierarchical models.

This module provides reusable components for specifying and fitting hierarchical models.
Hierarchical (multi-level) models allow parameters to vary across groups, while sharing information
through structured priors. This partial pooling leads to more stable estimates and reduces
overfitting, especially when data are sparse or imbalanced across groups.

Quick Reference Glossary:
    - Partial Pooling: Parameters vary by group but share information through a common prior,
      stabilizing estimates.
    - Shrinkage: Pulling parameter estimates toward a central value (e.g., zero) when data are weak
      or noisy.
    - Hyperparameter: A parameter of a prior controlling variability or central tendency of
      lower-level parameters
    - Hierarchical / Multi-level Model: Parameters structured at multiple levels (e.g., group and
      observation levels) to share information.
    - Feature-wise noise: Standard deviation of observations per feature; shared across groups
    - Standardized Effect Size (SMD): Dimensionless measure of group difference normalized by
      variability.
    - Random Seed: Fixes sampler randomness to enable reproducible posterior draws.
"""

import numpy.typing as npt
import pymc as pm
from arviz import InferenceData


def hierarchical_difference_model(
    X_A: npt.NDArray,
    X_B: npt.NDArray,
    draws: int = 2000,
    tune: int = 1000,
    target_accept: float = 0.95,
    random_seed: int | None = None,
) -> tuple[pm.Model, InferenceData]:
    """Bayesian hierarchical model to estimate feature-wise mean differences between two groups
    with partial pooling.

    The difference parameters (``delta``) for each feature are drawn from a shared prior with
    global scale ``tau``, which induces shrinkage towards zero for features with weak evidence.
    Each feature has its own noise level (``sigma``), but noise is assumed equivalent across
    groups. Observations are modelled as independent given their feature means and noise.

    Args:
        X_A: Observations from group A (n_samples, n_features)
        X_B: Observations from group B (n_samples, n_features)
        draws: Number of posterior draws
        tune: Number of tuning (warm-up) steps
        target_accept: Target acceptance probability for the sampler.
        random_seed: Seed for random number generation to enable reproducibility. Defaults to
            ``None``

    Returns:
        tuple:
            - model: PyMC model object
            - idata: InferenceData containing posterior samples
    """
    _, n_features = X_A.shape

    with pm.Model() as model:
        # Group A feature means (no pooling across features)
        mu_A = pm.Normal("mu_A", mu=0, sigma=10, shape=n_features)

        # Global scale controlling how much deltas vary across features
        tau = pm.HalfNormal("tau", sigma=5)

        # Feature-wise mean differences (hierarchical / partial pooling)
        delta = pm.Normal("delta", mu=0, sigma=tau, shape=n_features)

        # Group B feature means derive from A + delta
        mu_B = pm.Deterministic("mu_B", mu_A + delta)

        # Feature-specific observation noise, shared across groups
        sigma = pm.HalfNormal("sigma", sigma=5, shape=n_features)

        # Standardised effect size (SMD = Cohen's d-like)
        pm.Deterministic("effect", delta / sigma)

        # Observed data (mutable for predictive use)
        X_A_data = pm.Data("X_A_data", X_A)
        X_B_data = pm.Data("X_B_data", X_B)

        # Likelihoods
        pm.Normal("X_A_obs", mu=mu_A, sigma=sigma, observed=X_A_data)
        pm.Normal("X_B_obs", mu=mu_B, sigma=sigma, observed=X_B_data)

        # Sampling
        idata: InferenceData = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
        )

    return model, idata
