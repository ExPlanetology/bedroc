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
"""Bayesian PCA/latent factor models"""

import logging
from typing import Any, Optional

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from sklearn.decomposition import PCA

from bedroc.type_aliases import NpFloat

logger: logging.Logger = logging.getLogger(__name__)

savefig_opts: dict[str, Any] = {"dpi": 300, "bbox_inches": "tight", "format": "pdf"}
"""Figure options for savefig"""


def bayesian_pca(
    feature_values: NpFloat,
    feature_stds: NpFloat,
    data_labels: Optional[list[str]] = None,
    feature_labels: Optional[list[str]] = None,
    n_components: int = 2,
    draws: int = 2000,
    tune: int = 1000,
    target_accept: float = 0.95,
    random_seed: Optional[int] = None,
) -> tuple[pm.Model, az.InferenceData]:
    """Bayesian PCA model

    Args:
        feature_values: Feature values
        feature_stds: Feature standard deviations
        data_labels: Labels for the data points. Defaults to ``None``.
        feature_labels: Labels for the features. Defaults to ``None``.
        n_components: Number of latent factors. Defaults to ``2``.
        draws: Number of posterior draws. Defaults to ``2000``.
        tune: Number of tuning (warm-up) steps. Defaults to ``1000``.
        target_accept: Target acceptance probability for the sampler. Defaults to ``0.95``.
        random_seed: Seed for random number generation to enable reproducibility. Defaults to
            ``None``.

    Returns:
        tuple:
            - PyMC model object
            - InferenceData containing posterior samples
    """
    logger.debug("feature_values = %s", feature_values)
    logger.debug("feature_stds = %s", feature_stds)

    # The deterministic PCA solution is used for the prior of the latent variables
    pca: PCA = PCA(n_components=n_components)
    latent_variables: NpFloat = pca.fit_transform(feature_values)
    loading_matrix: NpFloat = pca.components_
    logger.debug("latent_variables = %s", latent_variables)
    logger.debug("loading_matrix = %s", loading_matrix)

    number_of_data: int = feature_values.shape[0]
    number_of_features: int = feature_values.shape[1]

    coords: dict[str, Any] = {
        "data_points": data_labels or np.arange(number_of_data),
        "components": np.arange(n_components),
        "features": feature_labels or np.arange(number_of_features),
    }

    # Standard Bayesian approach is to use a diffuse Gamma prior, which has a very large
    # variance and a mean that is not overly influential. Hyperpriors for the Gamma prior on the
    # precision of the loading matrix.
    a_alpha: float = 1e-3
    b_alpha: float = 1e-3

    with pm.Model(coords=coords) as model:
        # Prior for the latent factors (scores)
        Z = pm.Normal(
            "Z",
            mu=latent_variables,
            # Latent variables are independent of each other and each has unit variance
            # cov=np.eye(number_of_components), # Only for MvNormal
            shape=(number_of_data, n_components),
            sigma=1,
            dims=("data_points", "components"),
        )

        # mu = pm.MvNormal(
        #     "mu",
        #     mu=np.zeros(number_of_features),
        #     cov=np.eye(number_of_features),  # / 0.01,
        #     shape=number_of_features,
        # )

        # Hyperprior for the covariance of the columns in alpha. Standard is to use a diffuse
        # Gamma distribution for modelling precision.
        alpha_precision = pm.Gamma(
            "alpha_precision", alpha=a_alpha, beta=b_alpha, shape=n_components, dims="components"
        )

        # alpha represents the loading matrix in MPA. This matrix holds the weights that map
        # the transformed variables (latent factors or scores) to the original data space.
        alpha = pm.MatrixNormal(
            "alpha",
            # This specifies the prior mean of the loading matrix. loadings.T is the transpose
            # of the initial loadings matrix obtained from a preliminary PCA analysis, serving
            # as the prior mean for the Bayesian model.
            mu=loading_matrix,  # np.zeros((number_of_features, number_of_components)),
            # Row covariance, identity matrix implies no correlation between different rows
            # (features) in this case, which is suitable for a PCA as we assume features are
            # independent.
            colcov=np.eye(number_of_features),
            # The column covariance allows for varying degrees of uncertainty across the
            # principal components.
            rowcov=pt.diag(1 / alpha_precision),
            shape=(n_components, number_of_features),
            dims=("component", "features"),
        )

        # Additional parameter compared to the normal distribution
        nu_minus_1 = pm.Exponential("nu-1", 1.0 / 29)

        # Likelihood
        pm.StudentT(
            "Y_obs",
            mu=Z @ alpha,  # + mu.T,
            observed=feature_values,
            sigma=feature_stds,
            nu=nu_minus_1 + 1,
            shape=(number_of_data, number_of_features),
            dims=("data_points", "features"),
        )

        # Sampling
        idata = pm.sample(
            draws=draws, tune=tune, target_accept=target_accept, random_seed=random_seed
        )

    return model, idata
