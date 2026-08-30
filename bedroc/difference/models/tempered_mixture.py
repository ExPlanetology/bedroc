# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Tempered two-component mixture likelihood, shared by the tempered category-difference models.

Used as the ``logp``/``random`` implementation of a :class:`~pymc.CustomDist` in
:mod:`~bedroc.difference.models.tempered_likelihood` and
:mod:`~bedroc.difference.models.tempered_full`. Kept independent of both those models (rather than
attached to a single distribution class, e.g. via a diagonal-covariance
:class:`~pymc.MvNormal`-based :class:`~pymc.Mixture`) specifically so the two category components
can use different likelihood families if ever needed — a genuine multivariate Student-T, for
example, is not equivalent to a product of independent univariate Student-Ts the way a
diagonal-covariance multivariate Normal is to independent Normals, so that shortcut isn't
available in general.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from bedroc.core.type_aliases import NpArray


def sample_mixture_logp(value, pi_0, comp_0, comp_1, alpha):
    r"""Calculates the sample-level tempered mixture log-likelihood.

    Computes the log-probability density of observed multi-feature samples under a two-component
    mixture model. The full mixture likelihood is calculated and then scaled globally by a
    tempering factor :math:`\alpha`.

    .. math::

        \log p(X_s \mid \pi_0, \alpha) = \alpha \cdot \text{logaddexp}\left(
            \log(\pi_0) + \sum_{f=1}^{F} \log p(X_{s,f} \mid \text{Comp}_0), \;
            \log(1 - \pi_0) + \sum_{f=1}^{F} \log p(X_{s,f} \mid \text{Comp}_1)
        \right)
    """
    # 1. Compute element-wise log-likelihoods
    logp_0 = pm.logp(comp_0, value)
    logp_1 = pm.logp(comp_1, value)

    # 2. Sum across features for each sample (untempered component likelihoods)
    logp_sample_0 = pt.sum(logp_0, axis=1)
    logp_sample_1 = pt.sum(logp_1, axis=1)

    # 3. Combine components using the mixture weights
    log_w0 = pt.log(pi_0) + logp_sample_0  # pyright: ignore[reportOperatorIssue]
    log_w1 = pt.log(1.0 - pi_0) + logp_sample_1  # pyright: ignore[reportOperatorIssue]
    full_mixture_logp = pt.logaddexp(log_w0, log_w1)

    # 4. Apply tempering to the full mixture likelihood
    return alpha * full_mixture_logp


def sample_mixture_random(
    pi_0: float | NpArray,
    comp_0: NpArray,
    comp_1: NpArray,
    alpha: float,
    rng: np.random.Generator | None = None,
    size: tuple[int, ...] | None = None,
) -> NpArray:
    r"""Generates random samples from the two-component mixture distribution.

    Args:
        pi_0: Mixture prior weight for Component 0 as a scalar probability in `[0, 1]`.
        comp_0: Samples from Component 0 distribution, shape `(n_samples, n_features)`.
        comp_1: Samples from Component 1 distribution, shape `(n_samples, n_features)`.
        alpha: Likelihood tempering scaling factor :math:`\alpha \in (0, 1]`.
        rng: Optional random number generator. Defaults to ``None``.
        size: Optional shape of the output samples. Defaults to ``None``, in which case the shape
            of ``comp_0`` is used.

    Returns:
        Random samples from the mixture distribution, shape `(n_samples, n_features)`.
    """
    del alpha

    if rng is None:
        rng = np.random.default_rng()

    target_shape = comp_0.shape if size is None else size

    # Determine number of samples along the first axis
    n_samples = target_shape[0]

    # Draw category assignment PER SAMPLE: shape (n_samples, 1)
    # 1 = Category 0, 0 = Category 1
    is_comp_0 = rng.binomial(n=1, p=pi_0, size=(n_samples, 1))

    # Broadcast sample-level decision across all n_features
    return np.where(is_comp_0 == 1, comp_0, comp_1)
