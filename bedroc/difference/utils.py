# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Difference utils"""

import logging

import numpy as np
from scipy.integrate import simpson
from scipy.stats import gaussian_kde

from bedroc.core.data_container import RANDOM_SEED
from bedroc.core.type_aliases import NpArray

logger: logging.Logger = logging.getLogger(__name__)


def distribution_overlap_data(
    values_0: NpArray, values_1: NpArray, *, n_grid: int = 2000
) -> tuple[NpArray, NpArray, NpArray, NpArray, float]:
    """Calculates KDEs and overlap data for two 1D distributions.

    The probability density functions are estimated using Gaussian kernel density estimation (KDE).
    The overlap coefficient is then calculated as the integral of the pointwise minimum of the two
    estimated probability density functions.

    For multimodal or strongly irregular distributions, the estimated overlap may be sensitive to
    the KDE bandwidth.

    Args:
        values_0: Samples from the first distribution.
        values_1: Samples from the second distribution.
        n_grid: Number of points to use for the grid over which to evaluate the PDFs. Defaults to
            ``2000``.

    Returns:
        Tuple containing the evaluation grid, first PDF, second PDF, overlap density, and overlap
        coefficient.
    """
    values_0 = np.asarray(values_0, dtype=float)
    values_1 = np.asarray(values_1, dtype=float)

    values_0 = values_0[np.isfinite(values_0)]
    values_1 = values_1[np.isfinite(values_1)]

    if len(values_0) < 2 or len(values_1) < 2:
        raise ValueError("Both populations require at least two finite observations.")

    lower = min(values_0.min(), values_1.min())
    upper = max(values_0.max(), values_1.max())

    x: NpArray = np.linspace(lower, upper, n_grid)

    pdf_0 = gaussian_kde(values_0)(x)
    pdf_1 = gaussian_kde(values_1)(x)

    overlap_density = np.minimum(pdf_0, pdf_1)

    overlap = simpson(overlap_density, x=x)

    logger.info("Distribution overlap coefficient: %.4f", overlap)

    return x, pdf_0, pdf_1, overlap_density, float(overlap)


def joint_overlap(
    values_0: NpArray,
    values_1: NpArray,
    *,
    n_samples: int = 500_000,
    random_state: int | None = RANDOM_SEED,
) -> float:
    """Calculates the joint overlap coefficient of two empirical distributions.

    A multivariate Gaussian KDE is fitted directly to the observations of each population,
    preserving correlations between features. The overlap coefficient is estimated using Monte
    Carlo integration.

    Args:
        values_0: Array of shape (n_observations_0, n_features) for population 0.
        values_1: Array of shape (n_observations_1, n_features) for population 1.
        n_samples: Number of Monte Carlo samples used for integration. Defaults to ``500_000``.
        random_state: Seed for random number generation to enable reproducibility. Defaults to
            :obj:`RANDOM_SEED`.

    Returns:
        Joint overlap coefficient, between 0 and 1.
    """
    values_0 = np.asarray(values_0, dtype=float)
    values_1 = np.asarray(values_1, dtype=float)

    if values_0.ndim != 2 or values_1.ndim != 2:
        raise ValueError("Input arrays must be two-dimensional.")

    if values_0.shape[1] != values_1.shape[1]:
        raise ValueError("Both populations must have the same number of features.")

    if values_0.shape[0] < 2 or values_1.shape[0] < 2:
        raise ValueError("Both populations require at least two observations.")

    # Fit multivariate KDEs directly to the observations
    kde_0 = gaussian_kde(values_0.T)
    kde_1 = gaussian_kde(values_1.T)

    # Generate samples from population 0's empirical KDE
    rng = np.random.default_rng(random_state)
    samples = kde_0.resample(n_samples, seed=rng)

    # Evaluate both joint PDFs at the sampled points.
    pdf_0 = kde_0(samples)
    pdf_1 = kde_1(samples)

    # OVL = E_{x~p0}[min(1, p1/p0)]
    overlap = np.mean(np.minimum(1.0, pdf_1 / pdf_0))

    logger.info("Joint overlap coefficient: %.4f", overlap)

    return float(overlap)


def joint_naive_bayes_overlap(
    values_0: NpArray,
    values_1: NpArray,
    *,
    n_samples: int = 500_000,
    random_seed: int | None = RANDOM_SEED,
) -> float:
    r"""Calculate the joint overlap coefficient for a Naive Bayes model.

    For example, with two populations, 0 and 1, and three features,

        x = (f1, f2, f3)

    each population has a joint probability distribution,

        p0(f1, f2, f3)
        p1(f1, f2, f3)

    The overlap coefficient is defined as

    .. math::

        \mathrm{OVL} = \int \min[p_0(\mathbf{x}), p_1(\mathbf{x})] \, d\mathbf{x}

    The difficulty is that this is a 3D (or, in general, ND) integral, which becomes increasingly
    difficult to evaluate as the number of features increases. Here, the joint distributions are
    constructed as the product of the marginal KDEs for each feature by applying the Naive Bayes
    assumption of conditional independence.

    The overlap is estimated using Monte Carlo integration by sampling from the Naive Bayes joint
    distribution of population 0.

    Args:
        values_0: Array of shape (n_samples_0, n_features) for population 0
        values_1: Array of shape (n_samples_1, n_features) for population 1
        n_samples: Number of Monte Carlo samples used for integration. Defaults to ``500_000``.
        random_state: Seed for random number generation to enable reproducibility. Defaults to
            :obj:`RANDOM_SEED`.

    Returns:
        Joint overlap coefficient, between 0 and 1.
    """
    values_0 = np.asarray(values_0, dtype=float)
    values_1 = np.asarray(values_1, dtype=float)

    if values_0.ndim != 2 or values_1.ndim != 2:
        raise ValueError("Input arrays must be two-dimensional.")

    if values_0.shape[1] != values_1.shape[1]:
        raise ValueError("Both populations must have the same number of features.")

    # Fit marginal KDEs
    kde_0 = [gaussian_kde(values_0[:, i]) for i in range(values_0.shape[1])]
    kde_1 = [gaussian_kde(values_1[:, i]) for i in range(values_1.shape[1])]

    rng = np.random.default_rng(random_seed)

    # Sample from population 0's Naive Bayes joint distribution
    # Each feature is sampled independently from its marginal KDE
    samples = np.column_stack([kde.resample(n_samples, seed=rng).ravel() for kde in kde_0])

    # Evaluate the Naive Bayes joint PDFs
    pdf_0 = np.ones(n_samples)
    pdf_1 = np.ones(n_samples)

    for i in range(values_0.shape[1]):
        pdf_0 *= kde_0[i](samples[:, i])
        pdf_1 *= kde_1[i](samples[:, i])

    # OVL = E_{x~p0}[min(1, p1/p0)]
    overlap = np.mean(np.minimum(1.0, pdf_1 / pdf_0))

    logger.info("Joint Naive Bayes overlap coefficient: %.4f", overlap)

    return float(overlap)
