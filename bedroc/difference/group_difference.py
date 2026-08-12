# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Hierarchical Bayesian models for quantifying differences between groups.

This module provides the base model for comparing two groups across multiple features.
Group-specific feature means are expressed relative to a reference group, with feature-wise
differences estimated using a hierarchical prior.

The hierarchical structure enables partial pooling across features, allowing weakly supported group
differences to be shrunk toward zero while permitting stronger differences to deviate from the
shared population scale.

Likelihood-specific implementations, such as Normal, Laplace, or Student-t models, are provided by
subclasses in this package.
"""

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import xarray as xr
from matplotlib.axes import Axes

from bedroc.core import RANDOM_SEED, save_figure
from bedroc.difference.likelihood_models import LaplaceLikelihood, LikelihoodModel
from bedroc.type_aliases import NpArray, NpFloat, NpInt

logger: logging.Logger = logging.getLogger(__name__)


class HierarchicalGroupDifferenceModel:
    """Bayesian hierarchical model for group-centric comparisons of two groups

    The model treats one group as a reference and estimates a mean for each feature in that group
    For the second group, each feature is assigned its own difference parameter (``delta``), such
    that the feature mean is modeled as the reference-group mean plus ``delta``.

    The feature differences are modeled hierarchically. Each ``delta`` is drawn from a shared
    zero-centered Normal distribution with scale ``delta_scale``. This parameter controls the
    typical magnitude of group differences across features and induces partial pooling: features
    with weak signal relative to the shared scale are shrunk toward zero, while features with
    stronger signal are allowed to deviate further.

    This hierarchical structure couples the feature-wise differences through a common scale
    parameter rather than estimating them independently. This can improve stability when data are
    limited and is most appropriate when features are expected to exhibit broadly similar scales of
    group differences.

    The model can be used as a generative classifier via posterior predictive probabilities of
    group membership.

    Note:
        This model assumes that ``X`` has been standardized such that each feature has unit
        variance. All parameters, including ``delta`` and ``feature_sigma``, are therefore
        interpreted in standardized feature units.

    Args:
        name: Name of the dataset
        X: Observations (n_samples, n_features)
        X_group_idx: Group ID of observations, must be 0 or 1 (n_samples,)
        X_sigma: Sigma of observations (n_samples, n_features). Defaults to ``None``.
        sample_names: Sample names. ``None`` defaults to sequential sample names
        feature_names: Feature names. ``None`` defaults to sequential feature names
        group_names: Group names. ``None`` defaults to unique values in ``X_group_idx``.
        output_directory: Optional path to save generated data. Defaults to ``None`` (no saving).
        likelihood_model: Likelihood model class to use for the observation likelihood. Defaults
            to :class:`LaplaceLikelihood`.
    """

    def __init__(
        self,
        name: str,
        X: NpFloat,
        X_group_idx: NpInt,
        *,
        X_sigma: NpFloat | None = None,
        sample_names: Iterable | None = None,
        feature_names: Iterable | None = None,
        group_names: Iterable | None = None,
        output_directory: Path | None = None,
        likelihood_model: type[LikelihoodModel] = LaplaceLikelihood,
    ):
        logger.info("Creating a hierarchical group difference model for %s", name)
        self.name: str = name
        self.X: NpFloat = X
        self.X_group_idx: NpInt = X_group_idx
        self.X_sigma: NpFloat | None = X_sigma
        self.coords: dict = get_coords(
            self.X,
            self.X_group_idx,
            sample_names=sample_names,
            feature_names=feature_names,
            group_names=group_names,
        )
        self.output_directory: Path | None = output_directory
        if self.output_directory is not None:
            self.output_directory.mkdir(parents=True, exist_ok=True)

        self._likelihood_model: LikelihoodModel = likelihood_model()
        self._idata: xr.DataTree | None = None
        self._model: pm.Model | None = None
        self.observation_sample_idx: NpInt
        self.observation_feature_idx: NpInt
        self.observation_group_idx: NpInt

    @property
    def idata(self) -> xr.DataTree:
        """Inference data containing posterior samples"""
        if self._idata is None:
            raise ValueError("Inference has not been run yet. Call `run_inference()` first.")
        else:
            return self._idata

    @property
    def model(self) -> pm.Model:
        """PyMC model object"""
        if self._model is None:
            raise ValueError("Inference has not been run yet. Call `run_inference()` first.")
        else:
            return self._model

    @property
    def difference_string(self) -> str:
        """String representation of the group difference for plotting"""
        return f"({self.coords['group'][1]} - {self.coords['group'][0]})"

    def run_inference(
        self,
        *,
        draws: int = 2000,
        tune: int = 1000,
        target_accept: float = 0.95,
        random_seed: int | None = RANDOM_SEED,
        log_likelihood: bool = True,
    ) -> None:
        """Runs inference on the hierarchical model.

        Args:
            draws: Number of posterior samples to draw. Defaults to ``2000``.
            tune: Number of tuning steps. Defaults to ``1000``.
            target_accept: Target acceptance rate for NUTS sampler. Defaults to ``0.95``.
            random_seed: Random seed for reproducibility. Defaults to :obj:`RANDOM_SEED`.
            log_likelihood: Whether to compute log likelihood. Defaults to ``True``.
        """
        logger.info(
            "Running inference with draws=%d, tune=%d, target_accept=%.2f, random_seed=%s",
            draws,
            tune,
            target_accept,
            random_seed,
        )

        delta_scale_prior_sd: float = 0.5

        with pm.Model(coords=self.coords) as model:
            # Group A feature means (standardized space)
            mu_A = pm.Normal("mu_A", mu=0, sigma=0.5, dims="feature")

            # Hierarchical effect scale
            delta_scale = pm.HalfNormal("delta_scale", sigma=delta_scale_prior_sd)

            # Feature-wise group differences
            delta = pm.Normal("delta", mu=0, sigma=delta_scale, dims="feature")

            # All group feature means
            mu = pm.Deterministic(
                "mu", pm.math.stack([mu_A, mu_A + delta], axis=0), dims=("group", "feature")
            )

            # Intrinsic feature variability/noise is assumed to be shared across both groups,
            # representing irreducible within-feature dispersion independent of group membership.
            # sigma is expressed in standardized feature units and is learned from the data.
            sigma = pm.HalfNormal("sigma", sigma=0.5, dims="feature")

            # Intrinsic effect size: separation of the underlying populations in units of their
            # intrinsic within-feature standard deviation.
            pm.Deterministic("effect_size", delta / sigma, dims="feature")

            self._likelihood_model.add_parameters()

            # Observed data
            # Per-(group, feature) likelihood; missing values contribute no likelihood term
            # (MCAR/MAR).
            sample_idx, feature_idx = np.where(np.isfinite(self.X))
            group_idx: NpInt = self.X_group_idx[sample_idx]

            self.observation_sample_idx = sample_idx
            self.observation_feature_idx = feature_idx
            self.observation_group_idx = group_idx

            X_observed: NpFloat = self.X[sample_idx, feature_idx]

            mu_observed = mu[group_idx, feature_idx]  # pyright: ignore

            if self.X_sigma is not None:
                # Quadrature combination; a reasonable approximation for all likelihood families
                # when measurement uncertainty is small relative to intrinsic feature variability.
                sigma_observed = pm.math.sqrt(
                    self.X_sigma[sample_idx, feature_idx] ** 2 + sigma[feature_idx] ** 2
                )
            else:
                sigma_observed = sigma[feature_idx]

            self._likelihood_model.add_likelihood(
                name="observations",
                mu=mu_observed,
                sigma=sigma_observed,
                observed=X_observed,
                dims="observation",
            )

            # Sampling and store objects for later access
            idata_kwargs = {"log_likelihood": log_likelihood}
            self._idata = pm.sample(
                draws=draws,
                tune=tune,
                target_accept=target_accept,
                random_seed=random_seed,
                idata_kwargs=idata_kwargs,
            )
            self._model = model

        # Add observation metadata coordinates to the inference data for plotting and analysis
        self._idata = self._add_observation_coords(self._idata)

        if self.output_directory is not None:
            graph = pm.model_to_graphviz(model)
            graph.render(
                self.output_directory / Path(f"{self.name}_model_graph"),
                format="pdf",
                cleanup=True,
            )

    def plot_group_corner(
        self,
        *,
        savefig_kwargs: dict[str, Any] | None = None,
        truth_overlay: dict[str, NpArray] | None = None,
        save_fig: bool = True,
    ) -> sns.PairGrid:
        """Plots a corner plot for comparing the two groups with an optional overlay of truth.

        Args:
            savefig_kwargs: Override keyword arguments for :func:`matplotlib.pyplot.savefig`.
                Defaults to ``None``.
            truth_overlay: Optional dictionary containing true values for overlaying on the plot.
                Defaults to ``None``.
            save_fig: Whether to save the figure. Defaults to ``True``.

        Returns:
            Pairgrid
        """
        feature_names: list[str] = self.coords["feature"]
        group1, group2 = self.coords["group"]

        # Build DataFrame for seaborn
        df: pd.DataFrame = pd.DataFrame(self.X, columns=feature_names)
        df["Group"] = np.asarray(self.coords["group"])[self.X_group_idx]

        # Create corner plot
        pairgrid: sns.PairGrid = sns.pairplot(
            df,
            hue="Group",
            hue_order=self.coords["group"],
            corner=True,
            # diag_kind="hist",
            plot_kws=dict(alpha=0.4, s=20),
            diag_kws=dict(alpha=0.6, common_norm=False),
        )

        if truth_overlay is not None:
            # Overlay true means and 1 sigma bands on diagonal
            mu_A: NpFloat | None = truth_overlay.get("mu_A")
            mu_B: NpFloat | None = truth_overlay.get("mu_B")
            sigma: NpFloat | None = truth_overlay.get("sigma")

            def plot_helper(mu: NpFloat | None, color: str) -> None:
                if mu is not None:
                    for i, ax in enumerate(pairgrid.diag_axes):  # pyright: ignore - diag_axes is not None
                        ax.axvline(
                            mu[i],
                            color=color,
                            linestyle="--",
                            linewidth=2,
                            label="_nolegend_",
                            zorder=1,
                        )
                        if sigma is not None:
                            ax.axvspan(
                                mu[i] - sigma[i],
                                mu[i] + sigma[i],
                                color=color,
                                alpha=0.1,
                                zorder=0,
                            )

            plot_helper(mu_A, "blue")
            plot_helper(mu_B, "orange")

            # Off-diagonal: true multivariate centers
            for row in range(len(self.coords["feature"])):  # row index in axes
                for col in range(row):  # col index in axes
                    ax: Axes = pairgrid.axes[row, col]
                    if mu_A is not None:
                        ax.plot(
                            mu_A[col],
                            mu_A[row],
                            "o",
                            color="blue",
                            markersize=8,
                            markeredgecolor="k",
                            label="_nolegend_",
                        )
                    if mu_B is not None:
                        ax.plot(
                            mu_B[col],
                            mu_B[row],
                            "o",
                            color="orange",
                            markersize=8,
                            markeredgecolor="k",
                            label="_nolegend_",
                        )

        sns.move_legend(pairgrid, "upper left", bbox_to_anchor=(0.18, 0.8), frameon=True)

        pairgrid.figure.suptitle(f"{self.name}: {group2} vs {group1}")

        if save_fig:
            save_figure(
                pairgrid.figure,
                f"{group2}_vs_{group1}_pairplot",
                output_directory=self.output_directory,
                savefig_kwargs=savefig_kwargs,
            )

        return pairgrid

    def plot_prior_predictive(
        self,
        *,
        sample_kwargs: dict[str, Any] | None = None,
        savefig_kwargs: dict[str, Any] | None = None,
    ) -> az.PlotCollection:
        """Plots prior predictive check.

        This plot is used to determine if the model can generate data plausibly shaped like the
        observed distributions.

        Args:
            sample_kwargs: Keyword arguments for :func:`pymc.sample_prior_predictive`. Defaults to
                ``None``.
            savefig_kwargs: Override keyword arguments for :func:`matplotlib.pyplot.savefig`.
                Defaults to ``None``.

        Returns:
            Plot collection
        """
        if sample_kwargs is None:
            sample_kwargs = {}

        prior_predictive: xr.DataTree = pm.sample_prior_predictive(
            model=self.model, **sample_kwargs
        )

        pc: az.PlotCollection = az.plot_ppc_dist(
            prior_predictive,
            group="prior_predictive",
            kind="kde",
            # cols=["feature"], # to split by feature
            visuals={"observed_dist": {"color": "black"}},
        )
        pc.get_viz("figure").tight_layout(h_pad=1.0)

        save_figure(
            pc,
            "prior_predictive",
            output_directory=self.output_directory,
            savefig_kwargs=savefig_kwargs,
        )

        return pc

    def plot_posterior_predictive(
        self,
        *,
        sample_kwargs: dict[str, Any] | None = None,
        x_min: float | None = -5.0,
        x_max: float | None = 5.0,
        savefig_kwargs: dict[str, Any] | None = None,
    ) -> az.PlotCollection:
        """Plots posterior predictive check (in-sample predictions).

        This performs in-sample predictions to assess how well the model fits the observed data,
        i.e., test how well the model can reproduce the data it was trained on.

        Args:
            sample_kwargs: Keyword arguments for :func:`pymc.sample_posterior_predictive`. Defaults
                to ``None``.
            x_min: Minimum value for x-axis limits. Defaults to ``-5.0``.
            x_max: Maximum value for x-axis limits. Defaults to ``5.0``.
            savefig_kwargs: Override keyword arguments for :func:`matplotlib.pyplot.savefig`.
                Defaults to ``None``.

        Returns:
            Plot collection
        """
        if sample_kwargs is None:
            sample_kwargs = {}

        pm.sample_posterior_predictive(
            self.idata, model=self.model, extend_inferencedata=True, **sample_kwargs
        )

        # Re-add observation metadata coordinates to the posterior predictive samples for plotting
        # and analysis
        self._idata = self._add_observation_coords(self.idata)

        # There appears to be a limitation in ArviZ's plot_ppc_dist function that prevents it from
        # using a custom observation coordinate. As a workaround, filter the inference data to only
        # include the observed data and posterior predictive groups, then assign a new observation
        # coordinate according to how we wish to facet the plot.
        dt_with_observation_coords: xr.DataTree = self.idata.filter(
            lambda node: node.name in ("observed_data", "posterior_predictive")
        ).map_over_datasets(
            lambda node: node.assign_coords(
                observation=("observation", self.observation_feature_idx)
            )
        )

        pc: az.PlotCollection = az.plot_ppc_dist(
            dt_with_observation_coords,
            group="posterior_predictive",
            cols=["observation"],
            kind="kde",
            # kind="hist",
            visuals={"observed_dist": {"color": "black"}},
        )

        pc.get_viz("figure").tight_layout(h_pad=1.0)

        # For comparison with different likelihoods, set x-limits to a common range for all
        # features
        fig = pc.get_viz("figure")
        for ax in fig.axes:
            ax.set_xlim(x_min, x_max)

        save_figure(
            pc,
            "posterior_predictive",
            output_directory=self.output_directory,
            savefig_kwargs=savefig_kwargs,
        )

        return pc

    def plot_posterior_distributions(
        self,
        *,
        figsize: tuple = (12, 6),
        col_wrap: int = 4,
        savefig_kwargs: dict[str, Any] | None = None,
    ) -> az.PlotCollection:
        """Plots posterior distributions of model parameters.

        Args:
            figsize: Figure size. Defaults to ``(12, 6)``.
            col_wrap: Number of columns to wrap the plots. Defaults to ``4``.
            savefig_kwargs: Override keyword arguments for :func:`matplotlib.pyplot.savefig`.
                Defaults to ``None``.

        Returns:
            Plot collection
        """
        pc_kwargs: dict = {"figure_kwargs": {"figsize": figsize}}
        pc: az.PlotCollection = az.plot_dist(
            self.idata, var_names=["mu"], col_wrap=col_wrap, **pc_kwargs
        )
        pc.get_viz("figure").tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.0)
        pc.add_title("Posterior Distributions", fontsize="xx-large")

        save_figure(
            pc,
            "posterior_distributions",
            output_directory=self.output_directory,
            savefig_kwargs=savefig_kwargs,
        )

        return pc

    def plot_forest(
        self, figsize: tuple = (10, 15), *, savefig_kwargs: dict[str, Any] | None = None
    ) -> az.PlotCollection:
        """Plots forest plot of posterior distributions.

        Args:
            figsize: Figure size. Defaults to ``(10, 15)``.
            savefig_kwargs: Override keyword arguments for :func:`matplotlib.pyplot.savefig`.
                Defaults to ``None``.

        Returns:
            Plot collection
        """
        pc_kwargs: dict = {"figure_kwargs": {"figsize": figsize}}
        pc: az.PlotCollection = az.plot_forest(
            self.idata,
            var_names=["delta_scale", "delta", "sigma", "mu"],
            combined=True,
            **pc_kwargs,
        )

        ax = pc.get_viz("plot").sel(column="forest").item()
        # Strong reference line at zero
        ax.axvline(0, color="black", linewidth=1.5, zorder=1)

        pc.get_viz("figure").tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.0)
        pc.add_title(f"Posterior Differences {self.difference_string}", fontsize="large")

        save_figure(
            pc,
            "posterior_forest",
            output_directory=self.output_directory,
            savefig_kwargs=savefig_kwargs,
        )

        return pc

    def plot_forest_effect_size(
        self, figsize: tuple = (10, 6), *, savefig_kwargs: dict[str, Any] | None = None
    ) -> az.PlotCollection:
        """Forest plot of posterior effect sizes with interpretation bands

        Args:
            figsize: Figure size. Defaults to ``(10, 6)``.
            savefig_kwargs: Override keyword arguments for :func:`matplotlib.pyplot.savefig`.
                Defaults to ``None``.

        Returns:
            Plot collection
        """
        pc_kwargs: dict = {"figure_kwargs": {"figsize": figsize}}
        pc: az.PlotCollection = az.plot_forest(
            self.idata,
            var_names=["effect_size"],
            combined=True,
            **pc_kwargs,
        )

        ax: Axes = pc.get_viz("plot").sel(column="forest").item()

        band_colors: dict[str, str] = {
            "negligible": "#ffffff",
            "small": "#e0e0e0",
            "medium": "#bdbdbd",
            "large": "#9e9e9e",
        }

        # Effect size interpretation bands
        bands: list[tuple[float, float, str]] = [
            (0.0, 0.2, "negligible"),
            (0.2, 0.5, "small"),
            (0.5, 1.0, "medium"),
            # (1.0, 2.0, "large"),
        ]

        for left, right, label in bands:
            ax.axvspan(-right, -left, color=band_colors[label], alpha=1.0, zorder=0)
            ax.axvspan(left, right, color=band_colors[label], alpha=1.0, zorder=0)

        # Strong reference line at zero
        ax.axvline(0, color="black", linewidth=1.5, zorder=1)

        # Optional: annotate regions once (not per feature)
        ylim = ax.get_ylim()
        y_pos = ylim[1] * 0.95

        ax.text(-0.6, y_pos, "medium", ha="center", va="top", fontsize=9, color="0.3", rotation=90)
        ax.text(-0.35, y_pos, "small", ha="center", va="top", fontsize=9, color="0.3", rotation=90)
        ax.text(
            0.0,
            y_pos,
            "negligible",
            ha="center",
            va="top",
            fontsize=9,
            color="0.3",
            rotation=90,
            bbox=dict(facecolor=band_colors["negligible"], edgecolor="none"),
        )
        ax.text(0.35, y_pos, "small", ha="center", va="top", fontsize=9, color="0.3", rotation=90)
        ax.text(0.6, y_pos, "medium", ha="center", va="top", fontsize=9, color="0.3", rotation=90)

        pc.get_viz("figure").tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.0)
        pc.add_title(f"Posterior Effect Sizes {self.difference_string}", fontsize="large")

        save_figure(
            pc,
            "posterior_effect_sizes",
            output_directory=self.output_directory,
            savefig_kwargs=savefig_kwargs,
        )

        return pc

    def run_and_plot(self, *, savefig_kwargs: dict[str, Any] | None = None) -> None:
        """Runs the inference and generates standard plots.

        Args:
            savefig_kwargs: Override keyword arguments for :func:`matplotlib.pyplot.savefig`.
                Defaults to ``None``.
        """
        logger.info("Running analysis for %s", self.name)

        self.run_inference()
        self.plot_prior_predictive(savefig_kwargs=savefig_kwargs)
        self.plot_posterior_predictive(savefig_kwargs=savefig_kwargs)
        self.plot_posterior_distributions(savefig_kwargs=savefig_kwargs)
        self.plot_forest(savefig_kwargs=savefig_kwargs)
        self.plot_forest_effect_size(savefig_kwargs=savefig_kwargs)

        logger.info("Analysis complete for %s", self.name)

    def _add_observation_coords(self, dt: xr.DataTree) -> xr.DataTree:
        """Adds metadata coordinates identifying each flattened observation.

        The likelihood represents each finite sample-feature pair as a single observation along the
        ``observation`` dimension. This method adds coordinates identifying the corresponding
        sample, feature, and group.

        These coordinates are added to the resulting :class:`xarray.DataTree` after sampling
        because they are xarray metadata coordinates rather than PyMC model dimensions.

        Args:
            dt: Inference data containing the sampled model results

        Returns:
            The data tree with observation metadata coordinates added to the relevant groups
        """
        sample_names = self.coords["obs"]
        feature_names = self.coords["feature"]
        group_names = self.coords["group"]

        observation_coords: dict = {
            "observation_sample": ("observation", sample_names[self.observation_sample_idx]),
            "observation_feature": ("observation", feature_names[self.observation_feature_idx]),
            "observation_group": ("observation", group_names[self.observation_group_idx]),
        }

        def add_coords(node: xr.Dataset) -> xr.Dataset:
            """Helper function to add observation metadata coordinates to a dataset if relevant"""
            if "observation" not in node.dims:
                return node
            return node.assign_coords(observation_coords)

        return dt.map_over_datasets(add_coords)


def get_coords(
    X: NpFloat,
    X_group_idx: NpInt,
    *,
    sample_names: Iterable | None = None,
    feature_names: Iterable | None = None,
    group_names: Iterable | None = None,
) -> dict[str, list]:
    """Utility function to generate group and feature names with defaults.

    Args:
        X: Observations (n_samples, n_features)
        X_group_idx: Group ID of observations (n_samples,)
        sample_names: Sample names. Defaults to ``None`` to generate sequential sample names.
        feature_names: Feature names. Defaults to ``None`` to generate sequential feature names.
        group_names: Group names. Defaults to ``None`` to generate generic names.

    Returns:
        Dictionary of coordinates used for PyMC models
    """
    n_samples, n_features = X.shape

    sample_names = (
        np.asarray([f"Sample {i}" for i in range(n_samples)])
        if sample_names is None
        else np.asarray(sample_names)
    )

    feature_names = (
        np.asarray([f"Feature {i}" for i in range(n_features)])
        if feature_names is None
        else np.asarray(feature_names)
    )

    group_names = (
        np.asarray([f"Group {i}" for i in np.unique(X_group_idx)])
        if group_names is None
        else np.asarray(group_names)
    )

    n_groups: int = len(group_names)

    if np.min(X_group_idx) < 0 or np.max(X_group_idx) >= n_groups:
        raise ValueError(f"X_group_idx contains indices outside the range [0, {n_groups - 1}]")

    sample_idx, _ = np.where(np.isfinite(X))

    coords: dict[str, Any] = {
        "group": group_names,
        "feature": feature_names,
        "obs": sample_names,
        "observation": np.arange(len(sample_idx)),
    }

    return coords
