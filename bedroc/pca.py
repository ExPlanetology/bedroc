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

    n_data: int = feature_values.shape[0]
    n_features: int = feature_values.shape[1]

    coords: dict[str, Any] = {
        "data_points": data_labels or np.arange(n_data),
        "components": np.arange(n_components),
        "features": feature_labels or np.arange(n_features),
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
            # cov=np.eye(n_components), # Only for MvNormal
            shape=(n_data, n_components),
            sigma=1,
            dims=("data_points", "components"),
        )

        # mu = pm.MvNormal(
        #     "mu",
        #     mu=np.zeros(n_features),
        #     cov=np.eye(n_features),  # / 0.01,
        #     shape=n_features,
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
            mu=loading_matrix,  # np.zeros((n_features, n_components)),
            # Row covariance, identity matrix implies no correlation between different rows
            # (features) in this case, which is suitable for a PCA as we assume features are
            # independent.
            colcov=np.eye(n_features),
            # The column covariance allows for varying degrees of uncertainty across the
            # principal components.
            rowcov=pt.diag(1 / alpha_precision),
            shape=(n_components, n_features),
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
            shape=(n_data, n_features),
            dims=("data_points", "features"),
        )

        # Sampling
        idata = pm.sample(
            draws=draws, tune=tune, target_accept=target_accept, random_seed=random_seed
        )

    return model, idata


class PCAFactorAnalyzer:
    """PCA factor analyzer

    Helper class to compute outputs associated with a PCA/factor analysis. This is useful to
    compute output quantities for a Bayesian PCA to compare to a deterministic PCA.

    Args:
        latent_variables: Latent variables, which represent the projections or scores onto the
            latent space. Should be of shape (n_data, n_components, n_samples).
        loading_matrix: Loading matrix, which contains the latent factors. Should be of shape
            (n_components, n_features, n_samples).
    """

    def __init__(self, latent_variables: NpFloat, loading_matrix: NpFloat):
        self.latent_variables: NpFloat = latent_variables
        self.loading_matrix: NpFloat = loading_matrix

    @property
    def n_components(self) -> int:
        """Number of components"""
        return self.loading_matrix.shape[0]

    @property
    def n_data(self) -> int:
        """Number of data"""
        return self.latent_variables.shape[0]

    @property
    def n_features(self) -> int:
        """Number of features"""
        return self.loading_matrix.shape[1]

    @property
    def n_samples(self) -> int:
        """Number of samples"""
        return self.latent_variables.shape[2]

    def explained_variance_ratio_by_factor(self) -> NpFloat:
        """Explained variance ratio by latent factor

        This has been compared with pca.explained_variance_ratio_ and gives the same result.

        Returns:
            Explained variance ratio by latent factor with shape (n_components, n_samples)
        """
        explained_variance_ratio: NpFloat = np.zeros((self.n_components, self.n_samples))

        # Reconstruct the data for each latent factor and compute the explained variance ratio
        for latent_factor_idx in range(self.n_components):
            logger.debug("Working on latent factor %d", latent_factor_idx)
            L_factor: NpFloat = np.zeros_like(self.loading_matrix)
            L_factor[latent_factor_idx, :, :] = self.loading_matrix[latent_factor_idx, :, :]
            logger.debug("L_factor = %s", L_factor)

            explained_variance_ratio_by_feature: NpFloat = (
                self.explained_variance_ratio_by_feature(None, L_factor)
            )
            explained_variance_ratio_by_factor: NpFloat = np.sum(
                explained_variance_ratio_by_feature, axis=0
            ) / np.sum(self.observed_variance_by_feature())
            explained_variance_ratio[latent_factor_idx, :] = explained_variance_ratio_by_factor

        logger.debug("explained_variance_ratio_by_factor = %s", explained_variance_ratio)

        return explained_variance_ratio

    def explained_variance_ratio_by_feature(
        self, latent_variables: Optional[NpFloat] = None, loading_matrix: Optional[NpFloat] = None
    ) -> NpFloat:
        """Explained variance ratio by feature

        Args:
            latent_variables: Latent variables. Defaults to ``None`` to use all values.
            loading_matrix: Loading matrix. Defaults to ``None`` to use all values.

        Returns:
            Explained variance ratio by feature with shape (n_features, n_samples)
        """
        reconstructed_variance: NpFloat = self._reconstruct_variance_by_feature(
            latent_variables, loading_matrix
        )
        explained_variance_ratio: NpFloat = (
            reconstructed_variance / self.observed_variance_by_feature()
        )
        logger.debug("explained_variance_ratio_by_feature = %s", explained_variance_ratio)

        return explained_variance_ratio

    def explained_variance_ratio_total(self) -> NpFloat:
        """Total explained variance ratio across all features and components

        Compute the total variance ratio because in general for factor analysis the components are
        not orthogonal, and therefore the total cannot be determined by summing the contributions
        from the individual latent factors. This sums the variance across all features and latent
        factors, assuming that the reconstructed data includes all factors.

        Returns:
            Total explained variance ratio across all features and latent factors
        """
        explained_variance_ratio_by_feature: NpFloat = self.explained_variance_ratio_by_feature()
        explained_variance_ratio_total: NpFloat = np.sum(
            explained_variance_ratio_by_feature, axis=0
        ) / np.sum(self.observed_variance_by_feature())
        logger.debug("explained_variance_ratio_total = %s", explained_variance_ratio_total)

        return explained_variance_ratio_total

    def observed_variance_by_feature(self) -> NpFloat:
        """Variance of each feature in the standardized observed data

        This is unity by construction for standardized (z-score) data.

        Returns:
            Variance of each feature in the standardized observed data with shape (n_features,)
        """
        # TODO: Unity because we assume standardized data has been used to determine the latent
        # variables and loading matrix, but in general this is a bit dangerous to assume. Also
        # require a column vector for correct broadcasting.
        observed_variance_by_feature: NpFloat = np.ones((self.n_features, 1))
        logger.debug("observed_variance_by_feature = %s", observed_variance_by_feature)

        return observed_variance_by_feature

    def reconstruct_data(
        self, latent_variables: Optional[NpFloat] = None, loading_matrix: Optional[NpFloat] = None
    ) -> NpFloat:
        """Reconstructs data

        Args:
            latent_variables: Latent variables. Defaults to ``None`` to use all values.
            loading_matrix: Loading matrix. Defaults to ``None`` to use all values.

        Returns:
            Reconstructed data, usually with shape (n_data, n_features, n_samples)
        """
        latent_variables_: NpFloat = (
            self.latent_variables if latent_variables is None else latent_variables
        )
        loading_matrix_: NpFloat = (
            self.loading_matrix if loading_matrix is None else loading_matrix
        )

        reconstructed_data: NpFloat = np.einsum("ijk,jlk->ilk", latent_variables_, loading_matrix_)
        logger.debug("reconstructed_data = %s", reconstructed_data)
        logger.debug("reconstructed_data.shape = %s", reconstructed_data.shape)

        return reconstructed_data

    def _reconstruct_variance_by_feature(
        self, latent_variables: Optional[NpFloat] = None, loading_matrix: Optional[NpFloat] = None
    ) -> NpFloat:
        """Variance of each feature in the reconstructed data

        Args:
            latent_variables: Latent variables. Defaults to ``None`` to use all values.
            loading_matrix: Loading matrix. Defaults to ``None`` to use all values.

        Returns:
            Variance of each feature in the reconstructed data with shape (n_features, n_samples)
        """
        reconstructed_data: NpFloat = self.reconstruct_data(latent_variables, loading_matrix)
        reconstructed_variance_by_feature: NpFloat = np.var(reconstructed_data, axis=0)
        logger.debug("reconstructed_variance_by_feature = %s", reconstructed_variance_by_feature)
        logger.debug(
            "reconstructed_variance_by_feature.shape = %s", reconstructed_variance_by_feature.shape
        )

        return reconstructed_variance_by_feature
