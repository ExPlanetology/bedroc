# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Joint Bayesian inference of category differences and population fraction for two categories
with covariance structure shared between the two categories."""

import logging
from pathlib import Path
from typing import Any, Self, Sequence

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bedroc import RANDOM_SEED, override
from bedroc.core.data_container import DataContainer
from bedroc.core.plotting import save_figure
from bedroc.core.type_aliases import NpArray, NpFloat, NpInt
from bedroc.core.utils import SummaryStatistics
from bedroc.difference import DEFAULT_CATEGORY_COLORS, DEFAULT_CATEGORY_NAMES
from bedroc.difference.base import (
    CategoryClassifierProtocol,
    CategoryComparisonBase,
    PipelineProtocol,
    build_pipeline,
)
from bedroc.difference.plotting import plot_group_fraction_posterior, plot_mahalanobis_distance
from bedroc.difference.utils import validate_observation_data

logger: logging.Logger = logging.getLogger(__name__)


class UnifiedCovarianceModel(CategoryComparisonBase, CategoryClassifierProtocol):
    """Joint Bayesian inference of category differences and population fraction for two categories
    with covariance structure shared between the two categories.

    This model simultaneously infers the category parameters and the fraction of samples belonging
    to category 0 (with the category 1 fraction given by 1-pi0) in an unlabeled dataset.

    Args:
        name: Name of the model or analysis
        X_train: Observation data for the labeled training set, shape (n_samples, n_features)
        X_category_idx_train: Category indices for each sample in the training set, shape
            (n_samples,)
        X_unlabeled: Observation data for the unlabeled target set, shape (n_samples, n_features)
        X_sigma: Optional observation uncertainties for the training set, shape
            (n_samples, n_features). Defaults to ``None``, in which case the model assumes that the
            observations are exact.
        X_sigma_unlabeled: Optional observation uncertainties for the unlabeled target set, shape
            (n_samples, n_features). Defaults to ``None``, in which case the model assumes that the
            observations are exact.
        feature_names: Optional names for each feature. If not provided, defaults to
            ``["Feature 0", "Feature 1", ..., "Feature N"]``.
        category_names: Optional names for each category. Defaults to :obj:`DEFAULT_CATEGORY_NAMES`.
    """

    def __init__(
        self,
        name: str,
        X_train: NpFloat,
        X_category_idx_train: NpInt,
        X_unlabeled: NpFloat,
        *,
        X_sigma: NpFloat | None = None,
        X_sigma_unlabeled: NpFloat | None = None,
        feature_names: Sequence | None = None,
        category_names: Sequence = DEFAULT_CATEGORY_NAMES,
    ):
        logger.info("Creating a unified category difference model for %s", name)
        super().__init__(
            name,
            X_train,
            X_category_idx_train,
            X_sigma=X_sigma,
            feature_names=feature_names,
            category_names=category_names,
        )
        self.X_unlabeled, self.X_sigma_unlabeled = validate_observation_data(
            X_unlabeled, X_sigma=X_sigma_unlabeled
        )
        self._prior_alpha: float
        self._prior_beta: float

    @classmethod
    @override
    def from_data_container(
        cls,
        name: str,
        data: DataContainer,
        *,
        unlabeled_data: DataContainer | None = None,
        **kwargs,
    ) -> Self:
        """Creates an instance from a labeled and an unlabeled data container.

        Args:
            name: Name of the model or analysis
            data: Data container providing the standardized, labeled training observations
            unlabeled_data: Data container providing the standardized, unlabeled target
                observations over which the category fraction is jointly inferred. Required for
                this model.
            **kwargs: Additional keyword arguments to pass to the constructor

        Returns:
            Class instance

        Raises:
            ValueError: If ``unlabeled_data`` is not provided
        """
        if unlabeled_data is None:
            raise ValueError(
                f"{cls.__name__}.from_data_container requires 'unlabeled_data' "
                "(a second, unlabeled DataContainer)."
            )

        return super().from_data_container(
            name,
            data,
            X_unlabeled=unlabeled_data.values_std.to_numpy(),
            X_sigma_unlabeled=unlabeled_data.uncertainties_std.to_numpy(),
            **kwargs,
        )

    @override
    def pi_0_samples(self) -> NpFloat:
        """Posterior samples of the fraction of samples belonging to category 0 in the unlabeled
        dataset"""
        pi_0_samples: NpFloat = self.idata.posterior["pi_0"].values.flatten()

        SummaryStatistics(pi_0_samples).log_summary("pi_0 posterior summary")

        return pi_0_samples

    def mahalanobis_distance_samples(self) -> NpFloat:
        """Posterior samples of the Mahalanobis distance between the category means under the
        shared covariance structure."""
        mahalanobis_distance_samples: NpFloat = self.idata.posterior[
            "mahalanobis_distance"
        ].values.flatten()

        SummaryStatistics(mahalanobis_distance_samples).log_summary(
            "mahalanobis_distance posterior summary"
        )

        return mahalanobis_distance_samples

    @override
    def build_model(self, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        """Builds the PyMC model for the category comparison and stores it in ``self._model``.

        Each category's per-feature mean is drawn from the shared reference/difference structure
        built by :meth:`~bedroc.difference.base.CategoryComparisonBase.build_category_mean_priors`.
        Unlike :class:`~bedroc.difference.models.standard_difference.StandardDifferenceModel`,
        features are not independent: a single covariance matrix (``cov_shared``) is shared between
        both categories and jointly used for the labeled training likelihood and the two-component
        mixture likelihood of the unlabeled data, whose mixture weight ``pi_0`` (the category-0
        fraction) is inferred jointly with the rest of the model.

        Args:
            prior_alpha: Alpha parameter of the Beta prior on ``pi_0``. Defaults to ``1.0``.
            prior_beta: Beta parameter of the Beta prior on ``pi_0``. Defaults to ``1.0``.

        Raises:
            ValueError: If ``prior_alpha`` or ``prior_beta`` is not strictly positive.
        """
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("prior_alpha and prior_beta must be > 0.")

        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta

        # Get unique sample indices containing finite values
        train_s_idx = np.unique(np.where(np.isfinite(self.X))[0])
        self._train_sample_idx: NpInt = train_s_idx
        train_c_idx = self.X_category_idx[train_s_idx]

        # Slices maintain correct (N_train, n_features) shape
        X_train_data = self.X[train_s_idx]
        X_train_sigma_data = self.X_sigma[train_s_idx]

        n_features = self.X.shape[1]

        model_coords: dict[str, NpArray] = {
            **self.coords,
            "observation": np.arange(len(train_s_idx)),
            "observation_unlabeled": np.arange(self.X_unlabeled.shape[0]),
            # A second, distinct coordinate over the same feature names as "feature", so the two
            # axes of the covariance matrix can have different dimension names. xarray does not
            # support a variable with a repeated dimension name.
            "feature_bis": self.coords["feature"],
        }

        with pm.Model(coords=model_coords) as model:
            mu_0, delta_scale, delta, mu = self.build_category_mean_priors()

            # Single shared Cholesky factor across both categories
            chol, _, _ = pm.LKJCholeskyCov(
                "chol_shared",
                n=n_features,
                eta=2.0,
                sd_dist=pm.HalfNormal.dist(sigma=0.5),
                compute_corr=True,
            )

            # Full (n_features, n_features) shared covariance matrix
            cov_shared = pm.Deterministic(
                "cov_shared",
                pt.dot(chol, chol.T),  # pyright: ignore
                dims=("feature", "feature_bis"),
            )

            # Intrinsic effect size: separation of the underlying categories in units of the
            # shared within-feature standard deviation (diagonal of the shared covariance matrix).
            # Convenient for downstream plotting to not have underscores in the name since this
            # will be used as the label
            pm.Deterministic(
                "effect_size",
                delta / pm.math.sqrt(pt.diag(cov_shared)),  # pyright: ignore
                dims="feature",
            )

            # Multivariate distinguishability: Mahalanobis distance between the category means
            # under the shared covariance structure, D^2 = delta^T Sigma^-1 delta. Generalizes
            # `effect_size` to account for correlations between features.
            mahalanobis_sq = pm.Deterministic(
                "mahalanobis_sq",
                pt.dot(delta, pt.linalg.solve(cov_shared, delta)),  # pyright: ignore
            )
            pm.Deterministic("mahalanobis_distance", pt.sqrt(mahalanobis_sq))  # pyright: ignore

            # Identity matrix for diagonal masking: shape (D, D)
            eye_D = pt.eye(n_features)

            # Labeled Training Data
            # Broadened addition: (n_features, n_features) + (N_train, n_features, n_features)
            obs_cov_train = (X_train_sigma_data**2)[:, :, None] * eye_D  # pyright: ignore
            cov_train = cov_shared + obs_cov_train  # shape (N_train, n_features, n_features)

            # shape (N_train, n_features, n_features)
            chol_train = pt.linalg.cholesky(cov_train)  # pyright: ignore

            pm.MvNormal(
                "obs_train",
                mu=mu[train_c_idx],  # pyright: ignore
                chol=chol_train,
                observed=X_train_data,
                dims=("observation", "feature"),
            )

            # Unlabeled Data
            # Batch diagonal formation for unlabeled samples: shape (N_unlabeled, D, D)
            obs_cov_unlabeled = (self.X_sigma_unlabeled**2)[:, :, None] * eye_D  # pyright: ignore
            cov_unlabeled = cov_shared + obs_cov_unlabeled

            # Compute batched Cholesky once per step
            # shape (N_unlabeled, D, D)
            chol_unlabeled = pt.linalg.cholesky(cov_unlabeled)  # pyright: ignore[reportPrivateImportUsage]

            comp_0 = pm.MvNormal.dist(mu=mu[0], chol=chol_unlabeled)  # pyright: ignore
            comp_1 = pm.MvNormal.dist(mu=mu[1], chol=chol_unlabeled)  # pyright: ignore

            pi_0 = pm.Beta("pi_0", alpha=prior_alpha, beta=prior_beta)

            pm.CustomDist(
                "obs_unlabeled",
                pi_0,
                comp_0,
                comp_1,
                logp=sample_mixture_logp,
                random=sample_mixture_random,
                observed=self.X_unlabeled,
                dims=("observation_unlabeled", "feature"),
            )

        self._model = model

    def plot_prior_predictive_unlabeled(
        self,
        *,
        sample_kwargs: dict[str, Any] | None = None,
        random_seed: int | None = None,
        x_min: float | None = -4.0,
        x_max: float | None = 4.0,
        figsize: tuple[float, float] = (8, 5),
        legend: bool = True,
        title: bool = True,
    ) -> az.PlotCollection:
        """Plots a prior predictive check for the unlabeled mixture likelihood.

        Unlike the labeled training data, unlabeled samples have no known category, so this is
        faceted by feature alone rather than by category and feature.

        Args:
            sample_kwargs: Keyword arguments for :func:`pymc.sample_prior_predictive`. Defaults
                to ``None``.
            random_seed: Random seed for reproducibility, forwarded to
                :func:`pymc.sample_prior_predictive`. Defaults to ``None``.
            x_min: Minimum value for x-axis limits. Defaults to ``-4.0``.
            x_max: Maximum value for x-axis limits. Defaults to ``4.0``.
            figsize: Size of the figure. Defaults to ``(8, 5)``.
            legend: Whether to include a legend. Defaults to ``True``.
            title: Whether to include a title. Defaults to ``True``.

        Returns:
            Plot collection
        """
        return self.plot_prior_predictive(
            sample_kwargs=sample_kwargs,
            var_names=["obs_unlabeled"],
            cols=["feature"],
            title_prefix="Unlabeled ",
            random_seed=random_seed,
            x_min=x_min,
            x_max=x_max,
            figsize=figsize,
            legend=legend,
            title=title,
        )

    def plot_posterior_predictive_unlabeled(
        self,
        *,
        sample_kwargs: dict[str, Any] | None = None,
        random_seed: int | None = None,
        x_min: float | None = -4.0,
        x_max: float | None = 4.0,
        figsize: tuple[float, float] = (8, 5),
        legend: bool = True,
        title: bool = True,
    ) -> az.PlotCollection:
        """Plots a posterior predictive check for the unlabeled mixture likelihood.

        Unlike the labeled training data, unlabeled samples have no known category, so this is
        faceted by feature alone rather than by category and feature.

        Args:
            sample_kwargs: Keyword arguments for :func:`pymc.sample_posterior_predictive`.
                Defaults to ``None``.
            random_seed: Random seed for reproducibility, forwarded to
                :func:`pymc.sample_posterior_predictive`. Defaults to ``None``.
            x_min: Minimum value for x-axis limits. Defaults to ``-4.0``.
            x_max: Maximum value for x-axis limits. Defaults to ``4.0``.
            figsize: Size of the figure. Defaults to ``(8, 5)``.
            legend: Whether to include a legend. Defaults to ``True``.
            title: Whether to include a title. Defaults to ``True``.

        Returns:
            Plot collection
        """
        return self.plot_posterior_predictive(
            sample_kwargs=sample_kwargs,
            var_names=["obs_unlabeled"],
            cols=["feature"],
            title_prefix="Unlabeled ",
            random_seed=random_seed,
            x_min=x_min,
            x_max=x_max,
            figsize=figsize,
            legend=legend,
            title=title,
        )

    @override
    def _build_plot_dict(
        self, *, title: bool, random_seed: int | None = None
    ) -> dict[str, az.PlotCollection | Figure]:
        """Builds the dictionary of diagnostic plots generated by :meth:`generate_plots`.

        This model has no independent per-feature ``sigma`` (it uses a shared covariance matrix
        instead), so ``effect_size`` and the parameter/posterior-distribution plots use this
        model's own variable names rather than the base class defaults. Two additional
        predictive-check plots are generated for the unlabeled mixture likelihood, alongside the
        usual training-data predictive checks. The category-fraction posterior is intentionally
        not included here since it is handled separately (e.g. by :func:`pipeline`), which can
        supply the true unlabeled category counts for comparison.
        """
        mahalanobis_distance_fig: Figure = (
            self.plot_mahalanobis_distance().get_figure()  # pyright: ignore[reportAssignmentType]
        )

        return {
            "prior_predictive": self.plot_prior_predictive(
                var_names=["obs_train"],
                sample_idx=self._train_sample_idx,
                title=title,
                random_seed=random_seed,
            ),
            "posterior_predictive": self.plot_posterior_predictive(
                var_names=["obs_train"],
                sample_idx=self._train_sample_idx,
                title=title,
                random_seed=random_seed,
            ),
            "prior_predictive_unlabeled": self.plot_prior_predictive_unlabeled(
                title=title, random_seed=random_seed
            ),
            "posterior_predictive_unlabeled": self.plot_posterior_predictive_unlabeled(
                title=title, random_seed=random_seed
            ),
            "parameter_estimates": self.plot_parameter_estimates(
                var_names=["mu_0", "delta_scale", "delta", "cov_shared", "mahalanobis_sq"],
                title=title,
            ),
            "posterior_distributions": self.plot_posterior_distributions(
                var_names=["mu", "cov_shared"], title=title
            ),
            "effect_sizes": self.plot_effect_sizes(title=title),
            "mahalanobis_distance": mahalanobis_distance_fig,
        }

    def plot_group_fraction_posterior(
        self,
        bins: int = 50,
        n_grid: int = 2001,
        category_colors: tuple[str, str] = DEFAULT_CATEGORY_COLORS,
        category_counts: pd.Series | None = None,
        ax: Axes | None = None,
    ) -> Axes:
        """Plots the posterior distribution of the fraction of samples belonging to category 0.

        Args:
            bins: Number of bins for the histogram. Defaults to ``50``.
            n_grid: Number of grid points for the prior and perfect-classification limit. Defaults to
                ``2001``.
            category_colors: Colors for the two categories. Defaults to
                :data:`~bedroc.difference.DEFAULT_CATEGORY_COLORS`.
            category_counts: Known counts for the two categories. If ``None``, the observed
                fractions are not plotted. Defaults to ``None``.
            ax: Matplotlib axes on which to plot. If ``None``, a new figure and axes are created.

        Returns:
            Matplotlib axes containing the posterior group-fraction plot
        """
        return plot_group_fraction_posterior(
            self.pi_0_samples(),
            prior_alpha=self._prior_alpha,
            prior_beta=self._prior_beta,
            bins=bins,
            n_grid=n_grid,
            category_names=self.coords["category"],
            category_colors=category_colors,
            category_counts=category_counts,
            ax=ax,
        )

    def plot_mahalanobis_distance(
        self,
        *,
        reference_value: float | None = None,
        bins: int = 50,
        n_grid: int = 2001,
        ax: Axes | None = None,
    ) -> Axes:
        """Plots the posterior distribution of the Mahalanobis distance between the category
        means under the shared covariance structure.

        Args:
            reference_value: Optional known/ground-truth Mahalanobis distance (e.g. from a
                synthetic data-generating process) to overlay for validation. Defaults to ``None``.
            bins: Number of bins for the histogram. Defaults to ``50``.
            n_grid: Number of grid points for the KDE curve. Defaults to ``2001``.
            ax: Matplotlib axes on which to plot. If ``None``, a new figure and axes are created.

        Returns:
            Matplotlib axes containing the posterior Mahalanobis distance plot
        """
        return plot_mahalanobis_distance(
            self.mahalanobis_distance_samples(),
            reference_value=reference_value,
            bins=bins,
            n_grid=n_grid,
            ax=ax,
        )


def sample_mixture_logp(value, pi_0, comp_0, comp_1):
    r"""Calculates the sample-level mixture log-likelihood for multivariate observations.

    Args:
        value: Observed sample data array of shape ``(n_samples, n_features)``
        pi_0: Mixture prior weight for Component 0 (scalar probability in ``[0, 1]``)
        comp_0: Multivariate distribution for Component 0
        comp_1: Multivariate distribution for Component 1

    Returns:
        Log-likelihood values for each sample, shape ``(n_samples,)``
    """
    # pm.logp(MvNormal, value) produces shape (n_samples,)
    logp_0 = pm.logp(comp_0, value)
    logp_1 = pm.logp(comp_1, value)

    # Apply likelihood tempering to multivariate sample logp
    log_w0 = pt.log(pi_0) + logp_0  # pyright: ignore[reportOperatorIssue]
    log_w1 = pt.log(1.0 - pi_0) + logp_1  # pyright: ignore[reportOperatorIssue]

    return pt.logaddexp(log_w0, log_w1)


def sample_mixture_random(
    pi_0: float | NpArray,
    comp_0: NpArray,
    comp_1: NpArray,
    rng: np.random.Generator | None = None,
    size: tuple[int, ...] | None = None,
) -> NpArray:
    r"""Generates random samples from the multivariate two-component mixture distribution.

    Args:
        pi_0: Mixture prior weight for Component 0 (scalar or array)
        comp_0: Drawn samples from Component 0 distribution
        comp_1: Drawn samples from Component 1 distribution
        rng: Optional NumPy random number generator.
        size: Target output shape, typically ending in (..., n_samples, n_features)

    Returns:
        Random samples from the mixture distribution matching ``size`` shape.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Fall back to comp_0 shape if size is not passed explicitly
    target_shape = comp_0.shape if size is None else size

    # Category assignment is sample-level, so drop the trailing feature dimension (axis=-1)
    # Target binomial shape: (..., n_samples, 1)
    sample_shape = target_shape[:-1] + (1,)

    # Draw binary category selection: 1 = Component 0, 0 = Component 1
    is_comp_0 = rng.binomial(n=1, p=pi_0, size=sample_shape)

    # Broadcast selection across feature dimension (axis=-1)
    return np.where(is_comp_0 == 1, comp_0, comp_1)


_build_pipeline: PipelineProtocol = build_pipeline(UnifiedCovarianceModel)


def pipeline(
    data: DataContainer,
    *,
    output_directory: Path | None = None,
    random_seed: int | None = RANDOM_SEED,
    build_model_kwargs: dict[str, Any] | None = None,
) -> UnifiedCovarianceModel:
    """Pipeline for the unified category difference and covariance model.

    This wraps the generic :func:`~bedroc.difference.base.build_pipeline` pipeline to
    additionally plot the category-fraction posterior, since that plot needs the true unlabeled
    category counts for comparison, which are not available to the generic base-class pipeline.

    Args:
        data: The container holding the input data for the pipeline
        output_directory: Directory to save generated figures. If ``None``, figures are not
            saved.
        random_seed: Random seed for reproducibility. Defaults to :data:`~bedroc.RANDOM_SEED`.
        build_model_kwargs: Optional keyword arguments passed to the model's ``build_model()``
            method (e.g. subclass-specific prior hyperparameters). Defaults to ``None``.

    Returns:
        The fitted :class:`UnifiedCovarianceModel` instance
    """
    model: UnifiedCovarianceModel = _build_pipeline(
        data,
        output_directory=output_directory,
        random_seed=random_seed,
        build_model_kwargs=build_model_kwargs,
    )

    _, test = data.train_test_split(random_state=random_seed)

    ax: Axes = model.plot_group_fraction_posterior(category_counts=test.category_counts)
    save_figure(
        ax.get_figure(),  # pyright: ignore[reportArgumentType]
        Path(f"{data.name}_group_fraction_posterior"),
        output_directory,
    )

    return model
