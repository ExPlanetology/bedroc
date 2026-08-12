# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Likelihood models for Bayesian hierarchical group-difference models.

Provides alternative probability distributions for modeling observations around group-specific
feature means.
"""

import logging
from abc import ABC, abstractmethod

import pymc as pm

from bedroc import override

logger: logging.Logger = logging.getLogger(__name__)


class LikelihoodModel(ABC):
    """Base class for observation likelihood models"""

    def add_parameters(self) -> None:
        """Adds the parameters of the likelihood model to the model."""
        return None

    def get_distribution_scale(self, *, sigma):
        """Converts intrinsic standard deviation to distribution scale.

        Args:
            sigma: Intrinsic within-feature standard deviation

        Returns:
            Scale parameter required by the likelihood distribution
        """
        return sigma

    @abstractmethod
    def add_likelihood(
        self, *, name: str, mu, sigma, observed, shape, dims: str | tuple[str, ...] | None = None
    ) -> None:
        """Adds the observation likelihood to the model.

        Args:
            name: Name of the likelihood distribution
            mu: Mean of the distribution
            sigma: Intrinsic within-feature standard deviation
            observed: Observed data
            shape: Shape of the observed data
            dims: Optional dimension names for the observed data
        """


class NormalLikelihood(LikelihoodModel):
    """Normal likelihood

    This model assumes that the observations are drawn from a Normal distribution centered on the
    group-specific feature means.
    """

    @override
    def add_likelihood(
        self, *, name: str, mu, sigma, observed, shape, dims: str | tuple[str, ...] | None = None
    ) -> None:
        scale = self.get_distribution_scale(sigma=sigma)
        pm.Normal(name, mu=mu, sigma=scale, observed=observed, shape=shape, dims=dims)


class LaplaceLikelihood(LikelihoodModel):
    """Laplace likelihood

    This model assumes that the observations are drawn from a Laplace distribution centered on
    the group-specific feature means.
    """

    @override
    def get_distribution_scale(self, *, sigma):
        """Converts intrinsic standard deviation to Laplace scale.

        Args:
            sigma: Intrinsic within-feature standard deviation

        Returns:
            Scale parameter of the Laplace distribution
        """
        return sigma / pm.math.sqrt(2.0)

    @override
    def add_likelihood(
        self, *, name: str, mu, sigma, observed, shape, dims: str | tuple[str, ...] | None = None
    ) -> None:
        b = self.get_distribution_scale(sigma=sigma)
        pm.Laplace(name, mu=mu, b=b, observed=observed, shape=shape, dims=dims)


class StudentTLikelihood(LikelihoodModel):
    """Student's t likelihood

    This model assumes that the observations are drawn from a Student's t distribution centered on
    the group-specific feature means.
    """

    @property
    def nu(self):
        """Degrees of freedom of the Student's t distribution"""
        return self.nu_minus_2 + 2

    @override
    def get_distribution_scale(self, *, sigma):
        """Converts intrinsic standard deviation to Student-t scale.

        Args:
            sigma: Intrinsic within-feature standard deviation

        Returns:
            Scale parameter of the Student-t distribution
        """
        return sigma * pm.math.sqrt((self.nu - 2) / self.nu)

    @override
    def add_parameters(self) -> None:
        """Adds the shared degrees-of-freedom parameter.

        A single degrees-of-freedom parameter is used across all features and groups. This could be
        extended to allow feature- or group-specific degrees of freedom, but doing so would
        increase model complexity and the number of parameters to estimate.

        The parameter is expressed as ``nu_minus_2 + 2`` to ensure that the degrees of freedom
        remain greater than 2, which is required for the Student's t distribution to have finite
        variance.
        """
        self.nu_minus_2 = pm.Exponential("nu_minus_2", 1 / 29.0)

    @override
    def add_likelihood(
        self, *, name: str, mu, sigma, observed, shape, dims: str | tuple[str, ...] | None = None
    ) -> None:
        scale = self.get_distribution_scale(sigma=sigma)
        pm.StudentT(
            name, mu=mu, sigma=scale, nu=self.nu, observed=observed, shape=shape, dims=dims
        )
