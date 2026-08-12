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
    """Hierarchical Bayesian model for comparing two groups across multiple features.

    Group 0 is treated as the reference group. Each feature has a reference-group mean ``mu_A`` and
    a group difference ``delta``, such that the corresponding group means are

    ``mu[0] = mu_A``
    ``mu[1] = mu_A + delta``.

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
        name: Name of the dataset or analysis
        X: Training observations with shape ``(n_samples, n_features)``
        X_group_idx: Group index for each training sample, with values 0 or 1
        X_sigma: Optional measurement uncertainties with the same shape as ``X``
        feature_names: Names of the features. Defaults to ``"Feature 0"``, etc.
        group_names: Names of the two groups. Defaults to ``"Group 0"`` and ``"Group 1"``.
        output_directory: Directory for generated figures. If ``None``, figures are not saved.
        likelihood_model: Likelihood model implementation used for the observations. Defaults to
            :class:`LaplaceLikelihood`.
    """

    def __init__(
        self,
        name: str,
        X: NpFloat,
        X_group_idx: NpInt,
        *,
        X_sigma: NpFloat | None = None,
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

        self.output_directory: Path | None = output_directory
        if self.output_directory is not None:
            self.output_directory.mkdir(parents=True, exist_ok=True)

        self._likelihood_model: LikelihoodModel = likelihood_model()

        self.coords: dict[str, NpArray] = get_coords(
            self.X, self.X_group_idx, feature_names=feature_names, group_names=group_names
        )
        self._idata: xr.DataTree | None = None

        self._model: pm.Model = self._build_model()

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
        return self._model

    @property
    def difference_string(self) -> str:
        """Return a human-readable representation of group 1 relative to group 0."""
        return f"({self.coords['group'][1]} - {self.coords['group'][0]})"

    def _build_model(self, plot_model: bool = True) -> pm.Model:
        """Builds the hierarchical model in PyMC.

        Args:
            plot_model: Whether to export the model graph. Defaults to ``True``.

        Returns:
            PyMC model object
        """
        # Observed data
        # Flatten finite sample-feature pairs into the observation dimension. Missing values are
        # omitted from the likelihood.
        sample_idx, feature_idx = np.where(np.isfinite(self.X))
        group_idx: NpInt = self.X_group_idx[sample_idx]

        X_data_np: NpFloat = self.X[sample_idx, feature_idx]

        with pm.Model(coords=self.coords) as model:
            # Group A feature means (standardized space)
            mu_A = pm.Normal("mu_A", mu=0, sigma=0.5, dims="feature")

            # Hierarchical effect scale
            delta_scale = pm.HalfNormal("delta_scale", sigma=0.5)

            # Feature-wise group differences
            delta = pm.Normal("delta", mu=0, sigma=delta_scale, dims="feature")

            # All group feature means
            mu = pm.Deterministic(
                "mu", pm.math.stack([mu_A, mu_A + delta], axis=0), dims=("group", "feature")
            )

            # Intrinsic feature variability, shared between groups. ``sigma`` is expressed in
            # standardized feature units.
            sigma = pm.HalfNormal("sigma", sigma=0.5, dims="feature")

            # Intrinsic effect size: separation of the underlying populations in units of their
            # intrinsic within-feature standard deviation.
            pm.Deterministic("effect_size", delta / sigma, dims="feature")

            # Data
            X_data = pm.Data("X_data", X_data_np, dims="observation")
            feature_idx_data = pm.Data("feature_idx", feature_idx, dims="observation")
            group_idx_data = pm.Data("group_idx", group_idx, dims="observation")

            if self.X_sigma is not None:
                # Combine intrinsic variability with per-observation measurement uncertainty.
                X_sigma_observed = self.X_sigma[sample_idx, feature_idx]
                X_sigma_data = pm.Data("X_sigma", X_sigma_observed, dims="observation")
                sigma_observed = pm.math.sqrt(X_sigma_data**2 + sigma[feature_idx_data] ** 2)  # pyright: ignore
            else:
                sigma_observed = sigma[feature_idx_data]

            mu_observed = mu[group_idx_data, feature_idx_data]  # pyright: ignore

            self._likelihood_model.add_parameters()

            self._likelihood_model.add_likelihood(
                name="observations",
                mu=mu_observed,
                sigma=sigma_observed,
                observed=X_data,
                # Allows the observation dimension to change via pm.set_data()
                # https://www.pymc.io/projects/docs/en/latest/api/model/generated/pymc.model.core.set_data.html
                shape=X_data.shape,  # pyright: ignore[reportAttributeAccessIssue]
                dims="observation",
            )

        if plot_model and self.output_directory is not None:
            graph = pm.model_to_graphviz(model)
            graph.render(
                self.output_directory / Path(f"{self.name}_model_graph"),
                format="pdf",
                cleanup=True,
            )

        return model

    def run_inference(
        self,
        *,
        draws: int = 2000,
        tune: int = 1000,
        target_accept: float = 0.95,
        random_seed: int | None = RANDOM_SEED,
        **kwargs,
    ) -> None:
        """Runs inference on the hierarchical model.

        Args:
            draws: Number of posterior samples to draw. Defaults to ``2000``.
            tune: Number of tuning steps. Defaults to ``1000``.
            target_accept: Target acceptance rate for NUTS sampler. Defaults to ``0.95``.
            random_seed: Random seed for reproducibility. Defaults to :obj:`RANDOM_SEED`.
            **kwargs: Arbitrary keyword arguments passed to :func:`pymc.sample`. See PyMC
                documentation for details.
        """
        logger.info(
            "Running inference with draws=%d, tune=%d, target_accept=%.2f, random_seed=%s",
            draws,
            tune,
            target_accept,
            random_seed,
        )

        self._idata = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            random_seed=random_seed,
            model=self.model,
            **kwargs,
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
        feature_names: NpArray = self.coords["feature"]
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

        This performs in-sample replicated observations to assess how well the model can generate
        the observed data, i.e., test how well the model can reproduce the data it was trained on.

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

        sample_idx, feature_idx = np.where(np.isfinite(self.X))
        group_idx: NpInt = self.X_group_idx[sample_idx]

        # There appears to be a limitation in ArviZ's plot_ppc_dist function that prevents it from
        # using a custom observation coordinate. As a workaround, filter the inference data to only
        # include the observed data and posterior predictive groups, then assign a new observation
        # coordinate according to how we wish to facet the plot.
        observation_group_feature = (
            self.coords["group"][group_idx] + " — " + self.coords["feature"][feature_idx]
        )

        dt_with_observation_coords: xr.DataTree = self.idata.filter(
            lambda node: node.name in ("observed_data", "posterior_predictive")
        ).map_over_datasets(
            lambda node: node.assign_coords(observation=("observation", observation_group_feature))
        )

        # Hist is also not supported with faceting. Perhaps in future versions of ArviZ?
        pc: az.PlotCollection = az.plot_ppc_dist(
            dt_with_observation_coords,
            group="posterior_predictive",
            cols=["observation"],
            kind="kde",
            # kind="hist",
            visuals={"observed_dist": {"color": "black"}},
            col_wrap=len(self.coords["feature"]),  # one column per feature
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
        """Runs inference and generates standard plots.

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


def get_coords(
    X: NpFloat,
    X_group_idx: NpInt,
    *,
    feature_names: Iterable | None = None,
    group_names: Iterable | None = None,
) -> dict[str, NpArray]:
    """Generates static coordinates for the PyMC model.

    Only coordinates describing the model structure are included. The ``observation`` dimension is
    intentionally omitted because it is mutable and may change when the fitted model is evaluated
    on new data.

    Args:
        X: Training observations with shape ``(n_samples, n_features)``
        X_group_idx: Group indices for the training samples
        feature_names: Names of the features. Defaults to sequential names.
        group_names: Names of the two groups. Defaults to sequential names.

    Returns:
        Dictionary containing the ``group`` and ``feature`` coordinates
    """
    _, n_features = X.shape

    feature_names = (
        np.asarray([f"Feature {i}" for i in range(n_features)])
        if feature_names is None
        else np.asarray(feature_names)
    )

    unique_groups: NpArray = np.unique(X_group_idx)

    if not np.array_equal(unique_groups, np.array([0, 1])):
        raise ValueError("X_group_idx must contain exactly the two groups 0 and 1.")

    if group_names is None:
        group_names = np.asarray(["Group 0", "Group 1"])
    else:
        group_names = np.asarray(group_names)

    if len(group_names) != 2:
        raise ValueError("group_names must contain exactly two names.")

    if np.min(X_group_idx) < 0 or np.max(X_group_idx) >= len(group_names):
        raise ValueError(
            f"X_group_idx contains indices outside the range [0, {len(group_names) - 1}]"
        )

    return {"group": group_names, "feature": feature_names}
