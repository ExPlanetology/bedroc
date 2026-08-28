# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Joint Bayesian inference of category differences and population fraction for two categories
with covariance structure shared between the two categories."""

import logging
from pathlib import Path
from typing import Any, Sequence

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from matplotlib.axes import Axes

from bedroc import RANDOM_SEED, override
from bedroc.core.data_container import DataContainer
from bedroc.core.plotting import save_figure
from bedroc.core.type_aliases import NpArray, NpFloat, NpInt
from bedroc.difference import DEFAULT_CATEGORY_NAMES
from bedroc.difference.group_base import CategoryClassifierProtocol, CategoryComparisonBase
from bedroc.difference.plotting import plot_group_fraction_posterior
from bedroc.difference.validation import validate_observation_data

logger: logging.Logger = logging.getLogger(__name__)


class UnifiedCategoryDifferenceCovarianceModel(CategoryComparisonBase, CategoryClassifierProtocol):
    """Joint Bayesian inference of category differences and population fraction for two catgories
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

    @override
    def pi_0_samples(self) -> NpFloat:
        """Posterior samples of the fraction of samples belonging to category 0 in the unlabeled
        dataset"""
        return self.idata.posterior["pi_0"].values.flatten()

    @override
    def build_model(self, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:

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
            # Priors on Category Parameters
            mu_0 = pm.Normal("mu_0", mu=0, sigma=0.5, dims="feature")
            delta_scale = pm.HalfNormal("delta_scale", sigma=0.5)
            delta = pm.Normal("delta", mu=0, sigma=delta_scale, dims="feature")

            # Shape: (2, n_features)
            mu = pm.Deterministic(
                "mu", pm.math.stack([mu_0, mu_0 + delta], axis=0), dims=("category", "feature")
            )

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
            x_min: Minimum value for x-axis limits. Defaults to ``-4.0``.
            x_max: Maximum value for x-axis limits. Defaults to ``4.0``.
            figsize: Size of the figure. Defaults to ``(8, 5)``.
            legend: Whether to include a legend. Defaults to ``True``.
            title: Whether to include a title. Defaults to ``True``.

        Returns:
            Plot collection
        """
        if sample_kwargs is None:
            sample_kwargs = {}

        prior_predictive: xr.DataTree = pm.sample_prior_predictive(
            model=self.model, **sample_kwargs
        )

        return self._plot_predictive(
            prior_predictive,
            "prior_predictive",
            var_names=["obs_unlabeled"],
            cols=["feature"],
            title_prefix="Unlabeled ",
            figsize=figsize,
            x_min=x_min,
            x_max=x_max,
            legend=legend,
            title=title,
        )

    def plot_posterior_predictive_unlabeled(
        self,
        *,
        sample_kwargs: dict[str, Any] | None = None,
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
            x_min: Minimum value for x-axis limits. Defaults to ``-4.0``.
            x_max: Maximum value for x-axis limits. Defaults to ``4.0``.
            figsize: Size of the figure. Defaults to ``(8, 5)``.
            legend: Whether to include a legend. Defaults to ``True``.
            title: Whether to include a title. Defaults to ``True``.

        Returns:
            Plot collection
        """
        if sample_kwargs is None:
            sample_kwargs = {}

        posterior_predictive = pm.sample_posterior_predictive(
            self.idata, model=self.model, **sample_kwargs
        )

        return self._plot_predictive(
            posterior_predictive,
            "posterior_predictive",
            var_names=["obs_unlabeled"],
            cols=["feature"],
            title_prefix="Unlabeled ",
            figsize=figsize,
            x_min=x_min,
            x_max=x_max,
            legend=legend,
            title=title,
        )

    @override
    def generate_plots(
        self, output_directory: Path | str | None = None, title: bool = True
    ) -> dict[str, az.PlotCollection]:
        """Wrapper method to generate plots and save them to the specified output directory.

        This model has no independent per-feature ``sigma`` (it uses a shared covariance matrix
        instead), so ``effect_size`` and the parameter/posterior-distribution plots use this
        model's own variable names rather than the base class defaults. Two additional
        predictive-check plots are generated for the unlabeled mixture likelihood, alongside the
        usual training-data predictive checks. The category-fraction posterior is intentionally
        not included here since it is handled separately (e.g. by :func:`pipeline`), which can
        supply the true unlabeled category counts for comparison.

        Args:
            output_directory: Optional path to the directory where output files will be saved. If
                ``None``, no output files will be saved.
            title: Whether to include titles in the plots. Defaults to ``True``.

        Returns:
            Dictionary of plot collections with keys corresponding to plot types
        """
        handle_dict: dict[str, az.PlotCollection] = {}

        handle_dict["prior_predictive"] = self.plot_prior_predictive(
            var_names=["obs_train"], sample_idx=self._train_sample_idx, title=title
        )
        handle_dict["posterior_predictive"] = self.plot_posterior_predictive(
            var_names=["obs_train"], sample_idx=self._train_sample_idx, title=title
        )
        handle_dict["prior_predictive_unlabeled"] = self.plot_prior_predictive_unlabeled(
            title=title
        )
        handle_dict["posterior_predictive_unlabeled"] = self.plot_posterior_predictive_unlabeled(
            title=title
        )
        handle_dict["parameter_estimates"] = self.plot_parameter_estimates(
            var_names=["mu_0", "delta_scale", "delta", "cov_shared"], title=title
        )
        handle_dict["posterior_distributions"] = self.plot_posterior_distributions(
            var_names=["mu", "cov_shared"], title=title
        )
        handle_dict["effect_sizes"] = self.plot_effect_sizes(title=title)

        if output_directory is not None:
            output_directory = Path(output_directory)
            output_directory.mkdir(parents=True, exist_ok=True)

            self.plot_model(output_directory=output_directory)

            for plot_type, pc in handle_dict.items():
                save_figure(pc, f"{self.name}_{plot_type}", output_directory)

        return handle_dict

    def plot_group_fraction_posterior(
        self,
        bins: int = 50,
        n_grid: int = 2001,
        category_colors: tuple[str, str] = ("tab:blue", "tab:orange"),
        category_counts: pd.Series | None = None,
        ax: Axes | None = None,
    ) -> Axes:
        """Plots the posterior distribution of the fraction of samples belonging to category 0.

        Args:
            bins: Number of bins for the histogram. Defaults to ``50``.
            n_grid: Number of grid points for the prior and perfect-classification limit. Defaults to
                ``2001``.
            category_colors: Colors for the two categories. Defaults to
                ``("tab:blue", "tab:orange")``.
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


def pipeline(
    data: DataContainer,
    *,
    output_directory: Path | None = None,
    random_seed: int | None = RANDOM_SEED,
) -> UnifiedCategoryDifferenceCovarianceModel:
    """Pipeline.

    This provides a basic pipeline for running a standard analysis and generating the associated
    figures. For more customized analyses, you may wish to create your own pipeline.

    Args:
        data: The container holding the input data for the pipeline
        output_directory (Path | None): Optional path to the directory where output files will be
            saved. If ``None``, no output files will be saved.
        random_seed: Random seed for reproducible results. Defaults to :obj:`RANDOM_SEED`.
    """
    logger.info("Running pipeline for %s", data.name)

    if output_directory is not None:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory: %s", output_directory)
    else:
        logger.info("Output directory not specified. Figures will not be saved.")

    ax: Axes = data.plot_correlation_coefficient()
    save_figure(
        ax.get_figure(),  # pyright: ignore[reportArgumentType]
        stem=f"{data.name}_correlation_coefficient",
        output_directory=output_directory,
    )

    train, test = data.train_test_split(random_state=random_seed)

    model: UnifiedCategoryDifferenceCovarianceModel = (
        UnifiedCategoryDifferenceCovarianceModel.from_data_container(
            data.name,
            train,
            X_unlabeled=test.values_std.to_numpy(),
            X_sigma_unlabeled=test.uncertainties_std.to_numpy(),
        )
    )

    model.build_model()
    model.run_inference(random_seed=random_seed)
    model.generate_plots(output_directory=output_directory, title=True)

    logger.info("Pipeline completed for %s", data.name)

    return model
