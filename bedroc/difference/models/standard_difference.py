# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Hierarchical Bayesian models for feature-wise group comparisons.

This module provides Bayesian hierarchical models and observation likelihoods for quantifying
differences between two discrete groups across multiple features.

Group-specific feature means are parameterized relative to a reference group ($G_0$). The reference
group mean $\mu_0$ and the group difference $\delta$ are estimated jointly such that:

    $\mu_{G_0} = \mu_0$
    $\mu_{G_1} = \mu_0 + \delta$

The feature-specific differences $\delta$ are hierarchically modeled using a shared, zero-centered
Normal prior with scale parameter ``delta_scale``. This structure induces partial pooling across
features—shrinking weakly supported differences toward zero while allowing features with strong
evidence to deviate.

Models in this module inherit from :class:`~bedroc.difference.GroupComparisonBase` and support
modular observation likelihoods (e.g., :class:`NormalLikelihood`, :class:`LaplaceLikelihood`,
:class:`StudentTLikelihood`).

Once fitted, models populate posterior inference data as an :class:`xarray.DataTree` and can
evaluate class-conditional log-likelihoods on unobserved data via ``compute_log_likelihood()``,
serving as the generative likelihood component of a two-stage Bayesian classifier.
"""

import logging
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import pymc as pm
import xarray as xr

from bedroc import override
from bedroc.core.type_aliases import NpArray, NpFloat, NpInt
from bedroc.difference import DEFAULT_GROUP_NAMES
from bedroc.difference.group_base import GroupComparisonBase, PipelineProtocol, build_pipeline
from bedroc.difference.validation import validate_group_idx, validate_observation_data

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
            dims: Optional dimension names for the observed data. Defaults to ``None``.
        """


class NormalLikelihood(LikelihoodModel):
    """Normal likelihood.

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
    """Laplace likelihood.

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
    """Student's t likelihood.

    This model assumes that the observations are drawn from a Student's t distribution centered on
    the group-specific feature means.
    """

    @property
    def nu(self):
        """Degrees of freedom of the Student's t distribution."""
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


class StandardDifferenceModel(GroupComparisonBase):
    r"""Hierarchical Bayesian model for comparing two groups across multiple features.

    Group 0 is treated as the reference group. Each feature has a reference-group mean ``mu_0`` and
    a group difference ``delta``, such that the corresponding group means are

    ``mu[0] = mu_0`` and ``mu[1] = mu_0 + delta``.

    The feature-specific differences are hierarchically modeled using a shared, zero-centered
    Normal distribution with scale ``delta_scale``. This induces partial pooling: feature
    differences with weak evidence are shrunk toward zero, while features with stronger evidence
    can deviate further.

    The model assumes that ``X`` has been standardized such that each feature has approximately
    unit variance. Consequently, ``mu``, ``delta``, and ``sigma`` are expressed in standardized
    feature units.

    After fitting, the same PyMC model can be reused to evaluate arbitrary new observations without
    refitting. The mutable observation data are replaced using ``pm.set_data()``, allowing the
    model to be used as the likelihood component of a generative classifier.

    Missing values in ``X`` are omitted from the likelihood.

    Args:
        name: Name of the dataset or analysis.
        X: Training observations with shape ``(n_samples, n_features)``.
        X_group_idx: Group index for each training sample, with values 0 or 1.
        X_sigma: Optional measurement uncertainties with the same shape as ``X``. Defaults to
            ``None``, in which case the model assumes that the observations are exact.
        feature_names: Optional names for each feature. Defaults to ``None``, which generates
            ``["Feature 0", "Feature 1", ..., "Feature N"]``.
        group_names: Optional names for each group. Defaults to
            :data:`~bedroc.difference.DEFAULT_GROUP_NAMES`.
        likelihood_model: Likelihood model implementation used for the observations. Defaults to
            :class:`StudentTLikelihood`.
    """

    def __init__(
        self,
        name: str,
        X: NpFloat,
        X_group_idx: NpInt,
        *,
        X_sigma: NpFloat | None = None,
        feature_names: Sequence | None = None,
        group_names: Sequence = DEFAULT_GROUP_NAMES,
        likelihood_model: type[LikelihoodModel] = StudentTLikelihood,
    ):
        super().__init__(
            name,
            X,
            X_group_idx,
            X_sigma=X_sigma,
            feature_names=feature_names,
            group_names=group_names,
        )
        self._likelihood_model: LikelihoodModel = likelihood_model()

    @override
    def build_model(self) -> None:

        with pm.Model(coords=self.coords.to_dict()) as model:
            # Group 0 feature means (standardized space)
            mu_0 = pm.Normal("mu_0", mu=0, sigma=0.5, dims="feature")

            # Hierarchical effect scale
            delta_scale = pm.HalfNormal("delta_scale", sigma=0.5)

            # Feature-wise group differences
            delta = pm.Normal("delta", mu=0, sigma=delta_scale, dims="feature")

            # All group feature means
            mu = pm.Deterministic(
                "mu", pm.math.stack([mu_0, mu_0 + delta], axis=0), dims=("group", "feature")
            )

            # Intrinsic feature variability. ``sigma`` is expressed in standardized feature units.
            sigma = pm.HalfNormal("sigma", sigma=0.5, dims="feature")

            # Intrinsic effect size: separation of the underlying groups in units of their
            # intrinsic within-feature standard deviation. Convenient for downstream plotting to
            # not have underscores in the name since this will be used as the label
            pm.Deterministic("effect_size", delta / sigma, dims="feature")

            # Data
            X_data = pm.Data("X_data", self.X, dims=("observation", "feature"))
            X_sigma_data = pm.Data("X_sigma", self.X_sigma, dims=("observation", "feature"))
            group_idx_data = pm.Data("group_idx", self.X_group_idx, dims="observation")

            # Broadcast group means to shape (observation, feature):
            mu_observed = mu[group_idx_data, :]  # pyright: ignore

            # Broadcast intrinsic sigma across samples and combine with measurement uncertainty:
            # sigma[None, :] has shape (1, feature) and broadcasts against X_sigma_data (sample, feature)
            sigma_observed = pm.math.sqrt(X_sigma_data**2 + sigma[None, :] ** 2)  # pyright: ignore

            self._likelihood_model.add_parameters()

            self._likelihood_model.add_likelihood(
                name="observations",
                mu=mu_observed,
                sigma=sigma_observed,
                observed=X_data,
                # Allows the observation dimension to change via pm.set_data()
                # https://www.pymc.io/projects/docs/en/latest/api/model/generated/pymc.model.core.set_data.html
                shape=X_data.shape,  # pyright: ignore[reportAttributeAccessIssue]
                dims=("observation", "feature"),
            )

        self._model = model

    def compute_log_likelihood(
        self, X: NpFloat, *, X_sigma: NpFloat | None = None, group_idx: NpInt
    ) -> xr.Dataset:
        """Computes posterior log likelihoods for new observations under a group assignment.

        The fitted model parameters are held fixed at each posterior draw while the likelihood
        of each observation is evaluated under the supplied group assignment.

        Args:
            X: Data to evaluate, with shape ``(n_samples, n_features)``. Missing values should
                be represented by ``NaN``.
            X_sigma: Optional 1-sigma uncertainties for ``X``, with shape
                ``(n_samples, n_features)``. If ``None``, observations are treated as exact.
            group_idx: Group index for each sample, with shape ``(n_samples,)``. Values must
                correspond to the groups defined by the fitted model.

        Returns:
            Dataset containing the posterior log likelihood for each finite observation

        Raises:
            ValueError: If ``X``, ``X_sigma``, or ``group_idx`` has an invalid shape or
                contains invalid values
        """
        logger.info("Computing log likelihood for %d new observations", X.shape[0])

        X, X_sigma = validate_observation_data(X, X_sigma=X_sigma)
        group_idx = validate_group_idx(group_idx, n_samples=X.shape[0])

        n_samples, _ = X.shape

        coords: dict[str, NpArray] = {"observation": np.arange(n_samples)}
        data: dict[str, NpArray] = {"X_data": X, "X_sigma": X_sigma, "group_idx": group_idx}

        with self.model:
            pm.set_data(data, coords=coords)

            log_likelihood: xr.Dataset = pm.compute_log_likelihood(
                self.idata,
                var_names=["observations"],
                extend_inferencedata=False,
            )  # pyright: ignore[reportAssignmentType]

        log_likelihood = log_likelihood.rename({"observations": "log_likelihood"})

        logger.debug("Log likelihood: %s", log_likelihood)

        return log_likelihood


# Build the pipeline
pipeline: PipelineProtocol = build_pipeline(StandardDifferenceModel)
