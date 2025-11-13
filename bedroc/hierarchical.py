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

import logging
from pprint import pformat
from typing import Optional

import numpy as np
import numpy.typing as npt
import pymc as pm
from arviz import InferenceData

from bedroc import debug_logger

logger: logging.Logger = debug_logger()
logger.setLevel(logging.DEBUG)


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
        draws: Number of posterior draws. Defaults to ``2000``.
        tune: Number of tuning (warm-up) steps. Defaults to ``1000``.
        target_accept: Target acceptance probability for the sampler. Defaults to ``0.95``.
        random_seed: Seed for random number generation to enable reproducibility. Defaults to
            ``None``.

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


def generate_synthetic_data(
    n_samples: int = 50,
    n_features: int = 5,
    difference_scale: float = 0.0,
    type_a_std_of_mean: float = 2.0,
    type_b_std_of_mean: float = 1.0,
    sigma_min: float = 0.5,
    sigma_max: float = 2.0,
    random_seed: Optional[int] = None,
    heteroscedastic: bool = False,
) -> tuple[npt.NDArray, npt.NDArray, dict[str, npt.NDArray]]:
    """Generates multivariate data for 2 types (A & B), each with with optional per-type noise.

    Args:
        n_samples: Number of samples per type. Defaults to ``50``.
        n_features: Number of features per sample. Defaults to ``5``.
        difference_scale: Controls how different Type B is from Type A. Defaults to ``0``.
        type_a_std_of_mean: Standard deviation for Type A feature means. Defaults to ``2.0``.
        type_b_std_of_mean: Standard deviation for Type B feature means. Defaults to ``1.0
        sigma_min: Minimum noise (stddev) for features. Defaults to ``0.5``.
        sigma_max: Maximum noise (stddev) for features. Defaults to ``2.0``.
        random_seed: Optional seed for reproducibility. Defaults to ``None``.
        heteroscedastic: If ``True``, generate independent sigma per type. Defaults to ``False``.

    Returns:
        X_A: Type A data (n_samples, n_features)
        X_B: Type B data (n_samples, n_features)
        true_params: Ground-truth parameters
    """
    rng = np.random.default_rng(random_seed)

    # For Type A, each feature gets its own true mean (center of distribution)
    mu_A: npt.NDArray = rng.normal(loc=0.0, scale=type_a_std_of_mean, size=n_features)
    logger.debug("mu_A = %s", mu_A)

    # For Type B, each feature mean gets a random shift relative to Type A.
    # Scaling by difference_scale controls overall separation between types.
    raw_shift: npt.NDArray = rng.normal(loc=0.0, scale=type_b_std_of_mean, size=n_features)
    mu_B: npt.NDArray = mu_A + difference_scale * raw_shift
    logger.debug("mu_B = %s", mu_B)

    # Noise (standard deviation) per feature
    if heteroscedastic:
        # Noise varies across types as well as features
        sigma_A: npt.NDArray = rng.uniform(sigma_min, sigma_max, size=n_features)
        sigma_B: npt.NDArray = rng.uniform(sigma_min, sigma_max, size=n_features)
        logger.debug("sigma_A = %s", sigma_A)
        logger.debug("sigma_B = %s", sigma_B)
    else:
        # Noise only varies across features, not types
        sigma: npt.NDArray = rng.uniform(sigma_min, sigma_max, size=n_features)
        sigma_A = sigma_B = sigma
        logger.debug("sigma (shared) = %s", sigma)

    # Generate samples
    X_A: npt.NDArray = rng.normal(mu_A, sigma_A, size=(n_samples, n_features))
    X_B: npt.NDArray = rng.normal(mu_B, sigma_B, size=(n_samples, n_features))

    true_params: dict[str, npt.NDArray] = {
        "mu_A": mu_A,
        "mu_B": mu_B,
        "difference_vector": mu_B - mu_A,
        "sigma_A": sigma_A,
        "sigma_B": sigma_B,
    }
    logger.debug("true_params = \n%s", pformat(true_params))

    return X_A, X_B, true_params
