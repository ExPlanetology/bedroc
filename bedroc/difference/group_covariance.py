# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Joint Bayesian inference of group differences and population fraction for two groups with
covariance structure shared between the two groups."""

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from matplotlib.axes import Axes

from bedroc import RANDOM_SEED, override
from bedroc.core.data_container import DataContainer
from bedroc.core.plotting import add_xaxis_labels_to_bottom_row, save_figure
from bedroc.core.type_aliases import NpArray, NpFloat, NpInt
from bedroc.core.utils import SummaryStatistics
from bedroc.difference import DEFAULT_GROUP_NAMES
from bedroc.difference.group_base import GroupClassifierProtocol, GroupComparisonBase
from bedroc.difference.plotting import plot_group_fraction_posterior
from bedroc.difference.validation import validate_observation_data

logger: logging.Logger = logging.getLogger(__name__)


class UnifiedGroupDifferenceCovarianceModel(GroupComparisonBase, GroupClassifierProtocol):
    """Joint Bayesian inference of group differences and population fraction for two groups with
    covariance structure shared between the two groups.

    This model simultaneously infers the group parameters and the fraction of samples belonging to
    group 0 (with the group 1 fraction given by 1-pi0) in an unlabeled dataset.

    Args:
        name: Name of the model or analysis
        X_train: Observation data for the labeled training set, shape (n_samples, n_features)
        X_group_idx_train: Group indices for each sample in the training set, shape (n_samples,)
        X_unlabeled: Observation data for the unlabeled target set, shape (n_samples, n_features)
        X_sigma_train: Optional observation uncertainties for the training set, shape
            (n_samples, n_features). Defaults to ``None``, in which case the model assumes that the
            observations are exact.
        X_sigma_unlabeled: Optional observation uncertainties for the unlabeled target set, shape
            (n_samples, n_features). Defaults to ``None``, in which case the model assumes that the
            observations are exact.
        feature_names: Optional names for each feature. If not provided, defaults to
            ``["Feature 0", "Feature 1", ..., "Feature N"]``.
        group_names: Optional names for each group. Defaults to :obj:`DEFAULT_GROUP_NAMES`.
    """

    def __init__(
        self,
        name: str,
        X_train: NpFloat,
        X_group_idx_train: NpInt,
        X_unlabeled: NpFloat,
        *,
        X_sigma_train: NpFloat | None = None,
        X_sigma_unlabeled: NpFloat | None = None,
        feature_names: Iterable | None = None,
        group_names: Iterable = DEFAULT_GROUP_NAMES,
    ):
        logger.info("Creating a unified group difference model for %s", name)
        super().__init__(
            name,
            X_train,
            X_group_idx_train,
            X_sigma=X_sigma_train,
            feature_names=feature_names,
            group_names=group_names,
        )
        self.X_unlabeled, self.X_sigma_unlabeled = validate_observation_data(
            X_unlabeled, X_sigma=X_sigma_unlabeled
        )
        self._prior_alpha: float
        self._prior_beta: float

    @override
    def pi_0_samples(self) -> NpFloat:
        """Posterior samples of the fraction of samples belonging to group 0 in the unlabeled dataset"""
        return self.idata.posterior["pi_0"].values.flatten()

    @override
    def build_model(self, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:

        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta

        # Get unique sample indices containing finite values
        train_s_idx = np.unique(np.where(np.isfinite(self.X))[0])
        train_g_idx = self.X_group_idx[train_s_idx]

        # Slices maintain correct (N_train, n_features) shape
        X_train_data = self.X[train_s_idx]
        X_train_sigma_data = self.X_sigma[train_s_idx]

        n_features = self.X.shape[1]

        with pm.Model(coords=self.coords) as model:
            # Priors on Group Parameters
            mu_0 = pm.Normal("mu_0", mu=0, sigma=0.5, dims="feature")
            delta_scale = pm.HalfNormal("delta_scale", sigma=0.5)
            delta = pm.Normal("delta", mu=0, sigma=delta_scale, dims="feature")

            # Shape: (2, n_features)
            mu = pm.Deterministic(
                "mu", pm.math.stack([mu_0, mu_0 + delta], axis=0), dims=("group", "feature")
            )

            # Single shared Cholesky factor across both groups
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
                dims=("feature", "feature"),
            )

            # Identity matrix for diagonal masking: shape (D, D)
            eye_D = pt.eye(n_features)

            # Labeled Training Data
            # Broadened addition: (n_features, n_features) + (N_train, n_features, n_features)
            obs_cov_train = (X_train_sigma_data**2)[:, :, None] * eye_D  # pyright: ignore
            cov_train = cov_shared + obs_cov_train  # shape (N_train, n_features, n_features)

            # shape (N_train, n_features, n_features)
            chol_train = pt.linalg.cholesky(cov_train)  # pyright: ignore

            pm.MvNormal("obs_train", mu=mu[train_g_idx], chol=chol_train, observed=X_train_data)  # pyright: ignore

            # ------------------------------------------------------------------
            # Unlabeled Data
            # ------------------------------------------------------------------
            # Batch diagonal formation for unlabeled samples: shape (N_unlabeled, D, D)
            obs_cov_unlabeled = (self.X_sigma_unlabeled**2)[:, :, None] * eye_D  # pyright: ignore
            cov_unlabeled = cov_shared + obs_cov_unlabeled

            # Compute batched Cholesky once per step
            # shape (N_unlabeled, D, D)
            chol_unlabeled = pt.linalg.cholesky(cov_unlabeled)  # pyright: ignore

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
            )

        self._model = model

    def plot_posterior_predictive(
        self,
        *,
        sample_kwargs: dict[str, Any] | None = None,
        x_min: float | None = -5.0,
        x_max: float | None = 5.0,
    ) -> az.PlotCollection:
        """Plots posterior predictive check (in-sample predictions).

        This performs in-sample replicated observations to assess how well the model can generate
        the observed data, i.e., test how well the model can reproduce the data it was trained on.

        Args:
            sample_kwargs: Keyword arguments for :func:`pymc.sample_posterior_predictive`. Defaults
                to ``None``.
            x_min: Minimum value for x-axis limits. Defaults to ``-5.0``.
            x_max: Maximum value for x-axis limits. Defaults to ``5.0``.

        Returns:
            Plot collection
        """
        if sample_kwargs is None:
            sample_kwargs = {}

        pm.sample_posterior_predictive(
            self.idata, model=self.model, extend_inferencedata=True, **sample_kwargs
        )

        sample_idx, feature_idx = np.where(np.isfinite(self.X))
        group_idx: NpInt = self.X_group_idx[sample_idx]

        # There appears to be a limitation in ArviZ's plot_ppc_dist function that prevents it from
        # using a custom observation coordinate. As a workaround, filter the inference data to only
        # include the observed data and posterior predictive groups, then assign a new observation
        # coordinate according to how we wish to facet the plot.
        observation_group_feature = (
            self.coords.group[group_idx] + ", " + self.coords.feature[feature_idx]
        )

        dt_with_observation_coords: xr.DataTree = self.idata.filter(
            lambda node: node.name in ("observed_data", "posterior_predictive")
        ).map_over_datasets(
            lambda node: node.assign_coords(observation=("observation", observation_group_feature))
        )

        # Hist is also not supported with faceting. Perhaps in future versions of ArviZ?
        figsize = (8, 5)
        pc_kwargs: dict = {"figure_kwargs": {"figsize": figsize}}

        pc: az.PlotCollection = az.plot_ppc_dist(
            dt_with_observation_coords,
            group="posterior_predictive",
            cols=["observation"],
            kind="kde",
            # kind="hist",
            visuals={"observed_dist": {"color": "black"}},
            col_wrap=len(self.coords.feature),  # one column per feature
            **pc_kwargs,
        )

        add_xaxis_labels_to_bottom_row(pc, "Standardized units")

        fig = pc.get_viz("figure")
        fig.tight_layout(h_pad=0.3)

        # For comparison with different likelihoods, set x-limits to a common range for all feats
        for ax in fig.axes:
            ax.set_xlim(x_min, x_max)

        return pc

    def plot_group_fraction_posterior(
        self,
        bins: int = 50,
        n_grid: int = 2001,
        group_colors: tuple[str, str] = ("tab:blue", "tab:orange"),
        group_counts: tuple[float, float] | None = None,
        ax: Axes | None = None,
    ) -> Axes:
        """Plots the posterior distribution of the fraction of samples belonging to group 0.

        Args:
            bins: Number of bins for the histogram. Defaults to ``50``.
            n_grid: Number of grid points for the prior and perfect-classification limit. Defaults to
                ``2001``.
            group_colors: Colors for the two groups. Defaults to ``("tab:blue", "tab:orange")``.
            group_counts: Known counts for the two groups. If ``None``, the observed fractions are not
                plotted. Defaults to ``None``.
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
            group_names=self.coords.group,
            group_colors=group_colors,
            group_counts=group_counts,
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

    # Group assignment is sample-level, so drop the trailing feature dimension (axis=-1)
    # Target binomial shape: (..., n_samples, 1)
    sample_shape = target_shape[:-1] + (1,)

    # Draw binary group selection: 1 = Component 0, 0 = Component 1
    is_comp_0 = rng.binomial(n=1, p=pi_0, size=sample_shape)

    # Broadcast selection across feature dimension (axis=-1)
    return np.where(is_comp_0 == 1, comp_0, comp_1)


def pipeline(
    data: DataContainer,
    *,
    group_names: tuple[str, str],
    output_directory: Path | None = None,
    random_seed: int | None = RANDOM_SEED,
) -> None:
    """Pipeline.

    This provides a basic pipeline for running a standard analysis and generating the associated
    figures. For more customized analyses, you may wish to create your own pipeline.

    Args:
        data: The container holding the input data for the pipeline
        group_names: A tuple containing the names of the two groups for classification
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

    model: UnifiedGroupDifferenceCovarianceModel = UnifiedGroupDifferenceCovarianceModel(
        data.name,
        train.values_std.to_numpy(),
        train.metadata[train.group_type_column].to_numpy(),
        test.values_std.to_numpy(),
        X_sigma_train=train.uncertainties_std.to_numpy(),
        X_sigma_unlabeled=test.uncertainties_std.to_numpy(),
        feature_names=train.feature_names,
        group_names=group_names,
    )

    model.build_model()

    if output_directory is not None:
        model.plot_model(output_directory)

    model.run_inference(random_seed=random_seed)

    # Figure generation

    # pc: az.PlotCollection = model.plot_prior_predictive()
    # save_figure(pc, f"{data.name}_prior_predictive", output_directory=output_directory)

    # pc: az.PlotCollection = model.plot_posterior_predictive()
    # fig = pc.get_viz("figure")
    # legend_handles: list = [
    #     Line2D([0], [0], color="black", linewidth=2, label="Observed"),
    #     Line2D([0], [0], color="C0", linewidth=1.5, label="Posterior predictive"),
    # ]
    # fig.legend(handles=legend_handles, frameon=True)
    # save_figure(pc, f"{data.name}_posterior_predictive", output_directory=output_directory)

    # FIXME: This will break if the group_counts are not known
    # Get the true group counts in the test set for plotting the observed fractions
    group_counts = (
        np.sum(test.metadata[test.group_type_column] == 0),
        np.sum(test.metadata[test.group_type_column] == 1),
    )

    # Figure generation
    ax: Axes = model.plot_group_fraction_posterior(group_counts=group_counts)
    save_figure(
        ax.get_figure(),  # pyright: ignore[reportArgumentType]
        stem=f"{data.name}_group_fraction_posterior",
        output_directory=output_directory,
    )

    # Summary stats
    truth_val = group_counts[0] / sum(group_counts)
    pi_0_samples = model.pi_0_samples()
    summary_statistics: SummaryStatistics = SummaryStatistics(
        samples=pi_0_samples, truth=truth_val
    )
    df_summary_stats: pd.DataFrame = summary_statistics.to_dataframe()
    if output_directory is not None:
        df_summary_stats.to_excel(
            output_directory / Path(f"{data.name}_summary_statistics.xlsx"), index=False
        )

    logger.info("Pipeline completed for %s", data.name)

    # plt.show()
