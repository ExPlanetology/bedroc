# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Base classes and protocols"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, Self

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from bedroc import RANDOM_SEED, override
from bedroc.core.data_container import DataContainer
from bedroc.core.plotting import add_xaxis_labels_to_bottom_row, save_figure
from bedroc.core.type_aliases import NpArray, NpFloat, NpInt
from bedroc.difference import DEFAULT_CATEGORY_COLORS, DEFAULT_CATEGORY_NAMES
from bedroc.difference.plotting import plot_corner, plot_group_fraction_posterior
from bedroc.difference.utils import validate_category_idx, validate_observation_data

logger: logging.Logger = logging.getLogger(__name__)


class CategoryComparisonBase(ABC):
    """Base class for category comparison models.

    Args:
        name: Name of the model or analysis
        X: Observation data, shape (n_samples, n_features)
        X_category_idx: Category indices for each sample, shape (n_samples,)
        X_sigma: Optional observation uncertainties, shape (n_samples, n_features). Defaults to
            ``None``, in which case the model assumes that the observations are exact.
        feature_names: Optional names for each feature. If not provided, defaults to
            ``["Feature 0", "Feature 1", ..., "Feature N"]``.
        category_names: Optional names for each category. Defaults to
            :data:`~bedroc.difference.DEFAULT_CATEGORY_NAMES`.
        **kwargs: Additional keyword arguments to pass to the model's constructor.
    """

    def __init__(
        self,
        name: str,
        X: NpFloat,
        X_category_idx: NpInt,
        *,
        X_sigma: NpFloat | None = None,
        feature_names: Sequence | None = None,
        category_names: Sequence = DEFAULT_CATEGORY_NAMES,
        **kwargs,
    ):
        del kwargs  # Unused in base class, but may be used in subclasses

        self.name: str = name
        self.X, self.X_sigma = validate_observation_data(X, X_sigma=X_sigma)
        self.X_category_idx = validate_category_idx(X_category_idx, n_samples=self.X.shape[0])

        n_features = self.X.shape[1]
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(n_features)]
        feature_arr = np.asarray(list(feature_names), dtype=str)
        if len(feature_arr) != n_features:
            raise ValueError(
                f"Length of feature_names ({len(feature_arr)}) does not match "
                f"n_features in X ({n_features})."
            )
        category_arr = np.asarray(list(category_names), dtype=str)
        if len(category_arr) != 2:
            raise ValueError("category_names must contain exactly two names.")

        self.coords: dict[str, NpArray] = {"feature": feature_arr, "category": category_arr}
        self._idata: xr.DataTree | None = None
        self._model: pm.Model | None = None
        logger.info("Creating %s", self.__class__.__name__)

    @classmethod
    def from_data_container(
        cls,
        name: str,
        data: DataContainer,
        *,
        unlabeled_data: DataContainer | None = None,
        **kwargs,
    ) -> Self:
        """Creates an instance from a data container.

        Args:
            name: Name of the model or analysis
            data: Data container providing the standardized feature values, uncertainties, and
                category information
            unlabeled_data: Optional second data container of unlabeled observations. Unused in
                the base class; subclasses that jointly infer over an unlabeled dataset (e.g.
                :class:`UnifiedCovarianceModel`) should override this method to
                make use of it. Defaults to ``None``.
            **kwargs: Additional keyword arguments to pass to the constructor

        Returns:
            Class instance
        """
        del unlabeled_data  # Unused in base class, but may be used in subclasses

        return cls(
            name,
            data.values_std.to_numpy(),
            data.category_codes.to_numpy(),  # pyright: ignore[reportOptionalMemberAccess]
            X_sigma=data.uncertainties_std.to_numpy(),
            feature_names=data.feature_names,  # pyright: ignore[reportArgumentType]
            category_names=data.category_names,  # pyright: ignore[reportArgumentType]
            **kwargs,
        )

    @property
    def difference_string(self) -> str:
        """Returns a human-readable representation of category 1 relative to category 0."""
        return f"({self.coords['category'][1]} - {self.coords['category'][0]})"

    @property
    def idata(self) -> xr.DataTree:
        """Inference data containing posterior samples."""
        if self._idata is None:
            raise ValueError("Inference has not been run yet. Call `run_inference()` first.")
        else:
            return self._idata

    @property
    def model(self) -> pm.Model:
        """PyMC model object."""
        if self._model is None:
            raise ValueError("Model has not been built yet. Call `build_model()` first.")
        else:
            return self._model

    @abstractmethod
    def build_model(self, **kwargs) -> None:
        """Builds the PyMC model for the category comparison and stores it in ``self._model``.

        Args:
            **kwargs: Additional build-time keyword arguments for subclass-specific hyperparameters
                (e.g. prior parameters). Subclasses without build-time hyperparameters should
                accept and discard ``**kwargs``, so that :func:`build_pipeline`'s generated
                pipeline can call ``build_model()`` uniformly across all subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @staticmethod
    def build_category_mean_priors():
        """Builds the shared reference/difference prior structure for category means.

        Category 0 is treated as the reference category with mean ``mu_0``. Category 1's mean is
        offset from it by a hierarchically-shrunk difference ``delta``, such that
        ``mu = stack([mu_0, mu_0 + delta])``. This structure is shared by every
        :class:`CategoryComparisonBase` subclass regardless of what likelihood or covariance
        structure it builds on top of it.

        Must be called inside an active PyMC model context (e.g. within ``build_model()``).

        Returns:
            Tuple of ``(mu_0, delta_scale, delta, mu)``. ``mu_0`` and ``delta`` have
            ``dims="feature"``; ``mu`` has ``dims=("category", "feature")``.
        """
        # Category 0 feature means (standardized space)
        mu_0 = pm.Normal("mu_0", mu=0, sigma=0.5, dims="feature")

        # Hierarchical effect scale
        delta_scale = pm.HalfNormal("delta_scale", sigma=0.5)

        # Feature-wise category differences
        delta = pm.Normal("delta", mu=0, sigma=delta_scale, dims="feature")

        # All category feature means
        mu = pm.Deterministic(
            "mu", pm.math.stack([mu_0, mu_0 + delta], axis=0), dims=("category", "feature")
        )

        return mu_0, delta_scale, delta, mu

    def add_category_feature_coords(
        self, idata: xr.DataTree, *, sample_idx: NpInt | None = None
    ) -> xr.DataTree:
        """Helper function to attach a category, feature identifier coordinate to the inference
        data.

        This allows easier faceting of plots by category and feature, since there appears to be a
        limitation in ArviZ's plot_ppc_dist function that prevents it from using a custom
        observation coordinate. As a workaround, filter the inference data to only include the
        observed data and predictive groups, then assign a new observation coordinate that combines
        the category and feature names.

        Args:
            idata: Inference data object
            sample_idx: Optional indices into ``X_category_idx`` selecting which samples are
                actually present in the observed likelihood's ``"observation"`` dimension (e.g.
                after a subclass filters out samples with missing values). Defaults to ``None``,
                which uses all samples in their original order.

        Returns:
            Inference data object with 'observation_category_feature' coordinate attached to the
            relevant datasets
        """
        category_idx = (
            self.X_category_idx if sample_idx is None else self.X_category_idx[sample_idx]
        )
        obs_categories = self.coords["category"][category_idx]

        # 2D array matching (observation, feature)
        obs_category_feat_matrix: NpArray = np.strings.add(
            np.strings.add(obs_categories[:, None], ", "), self.coords["feature"][None, :]
        )

        # Target the nodes present in the DataTree
        for group_name in ("observed_data", "prior_predictive", "posterior_predictive"):
            if group_name in idata.children:
                # Access the underlying xarray.Dataset via .ds
                idata[group_name].ds = idata[group_name].ds.assign_coords(
                    observation_category_feature=(
                        ("observation", "feature"),
                        obs_category_feat_matrix,
                    )
                )

        return idata

    def plot_model(self, output_directory: Path | str, *, format: str = "pdf") -> Path:
        """Exports a graph of the PyMC model to a PDF file.

        Args:
            output_directory: Directory to save the model graph. If it does not exist, it will be
                created.
            format: Format of the output file. Defaults to ``"pdf"``. Can be any format supported
                by Graphviz.

        Returns:
            Path to the saved model graph file
        """
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        graph = pm.model_to_graphviz(self.model)

        # Should not include extension for graph.render()
        out_path: Path = output_directory / Path(f"{self.name}_model_graph")

        graph.render(out_path, format=format, cleanup=True)

        # Add format for the return path to match the saved file
        out_path = out_path.with_suffix(f".{format}")
        logger.info("Model graph saved to %s", out_path)

        return out_path

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
            random_seed: Random seed for reproducibility. Defaults to :data:`~bedroc.RANDOM_SEED`.
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

        logger.info("Inference completed successfully.")
        logger.debug("Inference data:\n%s", self._idata)

    def plot_effect_sizes(
        self,
        var_names: list[str] | str | None = None,
        *,
        figsize: tuple = (8, 3),
        positive_labels: bool = True,
        negative_labels: bool = True,
        title: bool = True,
    ) -> az.PlotCollection:
        """Forest plot of posterior effect sizes with interpretation bands.

        Args:
            var_names: List of variable names to plot. Can be a single string or an iterable of
                strings. Defaults to ``None`` to use default values that are typically of interest
                for category comparisons.
            figsize: Figure size. Defaults to ``(8, 3)``.
            positive_labels: Include descriptive labels for positive effect sizes. Defaults to
                ``True``.
            negative_labels: Include descriptive labels for negative effect sizes. Defaults to
                ``True``.
            title: Whether to include a title. Defaults to ``True``.

        Returns:
            Plot collection
        """
        if var_names is None:
            var_names = ["effect_size"]

        pc_kwargs: dict = {"figure_kwargs": {"figsize": figsize}}

        pc: az.PlotCollection = az.plot_forest(
            self.idata, var_names=var_names, combined=True, **pc_kwargs
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
            (0.5, 0.8, "medium"),
            (0.8, 2.0, "large"),
        ]

        for left, right, label in bands:
            ax.axvspan(-right, -left, color=band_colors[label], alpha=1.0, zorder=0)
            ax.axvspan(left, right, color=band_colors[label], alpha=1.0, zorder=0)

        # Strong reference line at zero
        ax.axvline(0, color="black", linewidth=1.5, zorder=1)

        # Optional: annotate regions once (not per feature)
        ylim = ax.get_ylim()
        y_pos = ylim[1] * 0.95

        ax.set_xlabel("Dimensionless")

        text_kwargs: dict[str, Any] = {
            "ha": "center",
            "va": "top",
            "fontsize": 10,
            "color": "0.3",
            "rotation": 90,
        }

        ax.text(
            0.0,
            y_pos,
            "negligible",
            **text_kwargs,
            bbox=dict(facecolor=band_colors["negligible"], edgecolor="none"),
        )

        if negative_labels:
            ax.text(-0.6, y_pos, "medium", **text_kwargs)
            ax.text(-0.35, y_pos, "small", **text_kwargs)

        if positive_labels:
            ax.text(0.35, y_pos, "small", **text_kwargs)
            ax.text(0.6, y_pos, "medium", **text_kwargs)

        fig = pc.get_viz("figure")

        if title:
            fig.suptitle(f"{self.name}: Effect Sizes {self.difference_string}", y=1)

        fig.tight_layout(h_pad=1.0, w_pad=1.0)

        return pc

    def plot_parameter_estimates(
        self,
        var_names: list[str] | str | None = None,
        *,
        figsize: tuple = (8, 5),
        title: bool = True,
    ) -> az.PlotCollection:
        """Plots parameter estimates as a forest plot.

        Args:
            var_names: List of variable names to plot. Can be a single string or an iterable of
                strings. Defaults to ``None`` to use default values that are typically of interest
                for category comparisons.
            figsize: Figure size. Defaults to ``(8, 5)``.
            title: Whether to include a title. Defaults to ``True``.

        Returns:
            Plot collection
        """
        if var_names is None:
            var_names = ["delta_scale", "delta", "sigma", "mu"]

        pc_kwargs: dict = {"figure_kwargs": {"figsize": figsize}}

        pc: az.PlotCollection = az.plot_forest(
            self.idata, var_names=var_names, combined=True, **pc_kwargs
        )

        ax = pc.get_viz("plot").sel(column="forest").item()
        # Strong reference line at zero
        ax.axvline(0, color="black", linewidth=1.5, zorder=1)
        ax.set_xlabel("Standardized units")

        fig = pc.get_viz("figure")

        if title:
            fig.suptitle(f"{self.name}: Parameter Estimates", y=1)

        fig.tight_layout(h_pad=1.0, w_pad=1.0)

        return pc

    def plot_posterior_distributions(
        self,
        var_names: list[str] | str | None = None,
        *,
        figsize: tuple = (8, 5),
        col_wrap: int = 4,
        legend: bool = True,
        title: bool = True,
    ) -> az.PlotCollection:
        """Plots posterior distributions of model parameters.

        Args:
            var_names: List of variable names to plot. Can be a single string or an iterable of
                strings. Defaults to ``None`` to use default values that are typically of interest
                for category comparisons.
            figsize: Figure size. Defaults to ``(8, 5)``.
            col_wrap: Number of columns to wrap the plots. Defaults to ``4``.
            legend: Whether to include a legend. Defaults to ``True``.
            title: Whether to include a title. Defaults to ``True``.

        Returns:
            Plot collection
        """
        if var_names is None:
            var_names = "mu"

        pc_kwargs: dict = {"figure_kwargs": {"figsize": figsize}}

        pc: az.PlotCollection = az.plot_dist(
            self.idata, var_names=var_names, col_wrap=col_wrap, **pc_kwargs
        )

        add_xaxis_labels_to_bottom_row(pc, "Standardized units")

        fig = pc.get_viz("figure")

        if legend:
            legend_handles: list = [
                Line2D([0], [0], color="0.4", linewidth=2, marker="o", label="95% CrI"),
            ]
            fig.legend(handles=legend_handles, frameon=True)

        if title:
            fig.suptitle(f"{self.name}: Posterior Distributions", y=1)

        fig.tight_layout(h_pad=1.0, w_pad=1.0)

        return pc

    def _plot_predictive(
        self,
        predictive_data: xr.DataTree,
        group: Literal["prior_predictive", "posterior_predictive"],
        *,
        var_names: list[str] | None = None,
        sample_idx: NpInt | None = None,
        cols: list[str] | None = None,
        title_prefix: str = "",
        figsize: tuple[float, float] = (8, 5),
        x_min: float | None = -4.0,
        x_max: float | None = 4.0,
        legend: bool = True,
        title: bool = True,
    ) -> az.PlotCollection:
        """Helper function to plot predictive checks (prior or posterior).

        Args:
            predictive_data: Inference data containing predictive samples
            group: Type of predictive check, either ``"prior_predictive"`` or
                ``"posterior_predictive"``
            var_names: Optional list of observed variable names to plot. Defaults to ``None``,
                which lets ArviZ select automatically (only appropriate when the model has a
                single observed variable).
            sample_idx: Optional indices into ``X_category_idx`` selecting which samples are
                actually present in the observed variable being plotted, passed through to
                :meth:`add_category_feature_coords`. Only used when ``cols`` is ``None``.
            cols: Optional column-faceting coordinate name(s) for :func:`arviz.plot_ppc_dist`.
                Defaults to ``None``, which facets by the category/feature coordinate added via
                :meth:`add_category_feature_coords`. Pass e.g. ``["feature"]`` to facet by feature
                alone, for an observed variable with no associated category (e.g. unlabeled data).
            title_prefix: Optional prefix inserted before the group name in the plot title.
                Defaults to ``""``.
            figsize: Size of the figure. Defaults to ``(8, 5)``.
            x_min: Minimum value for x-axis limits. Defaults to ``-4.0``.
            x_max: Maximum value for x-axis limits. Defaults to ``4.0``.
            legend: Whether to include a legend. Defaults to ``True``.
            title: Whether to include a title. Defaults to ``True``.

        Returns:
            Plot collection
        """
        if cols is None:
            predictive_data = self.add_category_feature_coords(
                predictive_data, sample_idx=sample_idx
            )
            cols = ["observation_category_feature"]

        # Hist is also not supported with faceting. Perhaps in future versions of ArviZ?
        pc_kwargs: dict = {"figure_kwargs": {"figsize": figsize}}

        pc: az.PlotCollection = az.plot_ppc_dist(
            predictive_data,
            group=group,
            var_names=var_names,
            kind="kde",
            cols=cols,
            visuals={"observed_dist": {"color": "black"}},
            col_wrap=len(self.coords["feature"]),  # one column per feature
            **pc_kwargs,
        )

        add_xaxis_labels_to_bottom_row(pc, "Standardized units")

        fig = pc.get_viz("figure")

        if legend:
            label: str = f"{group.replace('_', ' ').title()}"
            legend_handles: list = [
                Line2D([0], [0], color="black", linewidth=2, label="Observed"),
                Line2D([0], [0], color="C0", linewidth=1.5, label=label),
            ]
            fig.legend(handles=legend_handles, frameon=True)

        if title:
            fig.suptitle(f"{self.name}: {title_prefix}{group.replace('_', ' ').title()}", y=1)

        fig.tight_layout(h_pad=1.0, w_pad=1.0)

        # For comparison with different likelihoods, set x-limits to a common range for all feats
        for ax in fig.axes:
            ax.set_xlim(x_min, x_max)

        return pc

    def plot_prior_predictive(
        self,
        *,
        sample_kwargs: dict[str, Any] | None = None,
        var_names: list[str] | None = None,
        sample_idx: NpInt | None = None,
        cols: list[str] | None = None,
        title_prefix: str = "",
        random_seed: int | None = None,
        x_min: float | None = -4.0,
        x_max: float | None = 4.0,
        figsize: tuple[float, float] = (8, 5),
        legend: bool = True,
        title: bool = True,
    ) -> az.PlotCollection:
        """Plots prior predictive check.

        This plot is used to determine if the model can generate data plausibly shaped like the
        observed distributions.

        Args:
            sample_kwargs: Keyword arguments for :func:`pymc.sample_prior_predictive`. Defaults
                to ``None``.
            var_names: Optional list of observed variable names to plot, for models with more
                than one observed variable. Defaults to ``None``.
            sample_idx: Optional indices into ``X_category_idx`` selecting which samples are
                actually present in the plotted observed variable. Defaults to ``None``.
            cols: Optional column-faceting coordinate name(s), passed through to
                :meth:`_plot_predictive`. Defaults to ``None``, which facets by category and
                feature.
            title_prefix: Optional prefix inserted before the group name in the plot title.
                Defaults to ``""``.
            random_seed: Random seed for reproducibility, forwarded to
                :func:`pymc.sample_prior_predictive`. Ignored if ``sample_kwargs`` already
                supplies its own ``random_seed``. Defaults to ``None``.
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
        sample_kwargs.setdefault("random_seed", random_seed)

        prior_predictive: xr.DataTree = pm.sample_prior_predictive(
            model=self.model, **sample_kwargs
        )

        pc: az.PlotCollection = self._plot_predictive(
            prior_predictive,
            "prior_predictive",
            var_names=var_names,
            sample_idx=sample_idx,
            cols=cols,
            title_prefix=title_prefix,
            figsize=figsize,
            x_min=x_min,
            x_max=x_max,
            legend=legend,
            title=title,
        )

        return pc

    def plot_posterior_predictive(
        self,
        *,
        sample_kwargs: dict[str, Any] | None = None,
        var_names: list[str] | None = None,
        sample_idx: NpInt | None = None,
        cols: list[str] | None = None,
        title_prefix: str = "",
        random_seed: int | None = None,
        x_min: float | None = -4.0,
        x_max: float | None = 4.0,
        figsize: tuple[float, float] = (8, 5),
        legend: bool = True,
        title: bool = True,
    ) -> az.PlotCollection:
        """Plots posterior predictive check (in-sample predictions).

        This performs in-sample replicated observations to assess how well the model can generate
        the observed data, i.e., test how well the model can reproduce the data it was trained on.

        Args:
            sample_kwargs: Keyword arguments for :func:`pymc.sample_posterior_predictive`. Defaults
                to ``None``.
            var_names: Optional list of observed variable names to plot, for models with more
                than one observed variable. Defaults to ``None``.
            sample_idx: Optional indices into ``X_category_idx`` selecting which samples are
                actually present in the plotted observed variable. Defaults to ``None``.
            cols: Optional column-faceting coordinate name(s), passed through to
                :meth:`_plot_predictive`. Defaults to ``None``, which facets by category and
                feature.
            title_prefix: Optional prefix inserted before the group name in the plot title.
                Defaults to ``""``.
            random_seed: Random seed for reproducibility, forwarded to
                :func:`pymc.sample_posterior_predictive`. Ignored if ``sample_kwargs`` already
                supplies its own ``random_seed``. Defaults to ``None``.
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
        sample_kwargs.setdefault("random_seed", random_seed)

        posterior_predictive = pm.sample_posterior_predictive(
            self.idata, model=self.model, **sample_kwargs
        )

        pc: az.PlotCollection = self._plot_predictive(
            posterior_predictive,
            "posterior_predictive",
            var_names=var_names,
            sample_idx=sample_idx,
            cols=cols,
            title_prefix=title_prefix,
            figsize=figsize,
            x_min=x_min,
            x_max=x_max,
            legend=legend,
            title=title,
        )

        return pc

    def _build_plot_dict(
        self, *, title: bool, random_seed: int | None = None
    ) -> dict[str, az.PlotCollection | Figure]:
        """Builds the dictionary of diagnostic plots generated by :meth:`generate_plots`.

        Subclasses with additional or differently-parameterized plots (e.g. extra predictive
        checks, or non-default ``var_names`` because they don't share this base class's default
        parameterization) should override this method rather than :meth:`generate_plots`, so that
        the plot-saving logic stays shared.

        Args:
            title: Whether to include titles in the plots.
            random_seed: Random seed for reproducibility, forwarded to the prior/posterior
                predictive sampling calls. Defaults to ``None``.

        Returns:
            Dictionary of plot collections with keys corresponding to plot types
        """
        return {
            "prior_predictive": self.plot_prior_predictive(title=title, random_seed=random_seed),
            "posterior_predictive": self.plot_posterior_predictive(
                title=title, random_seed=random_seed
            ),
            "parameter_estimates": self.plot_parameter_estimates(title=title),
            "posterior_distributions": self.plot_posterior_distributions(title=title),
            "effect_sizes": self.plot_effect_sizes(title=title),
        }

    def _save_plots(
        self,
        handle_dict: dict[str, az.PlotCollection | Figure],
        output_directory: Path | str | None,
    ) -> None:
        """Saves the model graph and each plot in ``handle_dict`` to ``output_directory``.

        Args:
            handle_dict: Dictionary of plot collections with keys corresponding to plot types
            output_directory: Directory to save output files. If ``None``, nothing is saved.
        """
        if output_directory is not None:
            output_directory = Path(output_directory)
            output_directory.mkdir(parents=True, exist_ok=True)

            self.plot_model(output_directory=output_directory)

            for plot_type, pc in handle_dict.items():
                save_figure(pc, f"{self.name}_{plot_type}", output_directory)

    def generate_plots(
        self,
        output_directory: Path | str | None = None,
        title: bool = True,
        random_seed: int | None = None,
    ) -> dict[str, az.PlotCollection | Figure]:
        """Wrapper method to generate plots and save them to the specified output directory.

        Args:
            output_directory: Optional path to the directory where output files will be saved. If
                ``None``, no output files will be saved.
            title: Whether to include titles in the plots. Defaults to ``True``.
            random_seed: Random seed for reproducibility, forwarded to the prior/posterior
                predictive sampling calls. Defaults to ``None``.

        Returns:
            Dictionary of plot collections with keys corresponding to plot types
        """
        handle_dict: dict[str, az.PlotCollection | Figure] = self._build_plot_dict(
            title=title, random_seed=random_seed
        )

        self._save_plots(handle_dict, output_directory)

        return handle_dict


class UnlabeledMixtureModelMixin(CategoryComparisonBase):
    """Mixin for category comparison models that jointly infer over an unlabeled target dataset.

    Provides the shared ``from_data_container`` override (requiring a second, unlabeled
    :class:`~bedroc.core.data_container.DataContainer`) and the paired prior/posterior
    predictive-check plotting methods for the unlabeled mixture likelihood, reused identically by
    :class:`~bedroc.difference.models.unified_covariance.UnifiedCovarianceModel`,
    :class:`~bedroc.difference.models.tempered_likelihood.TemperedDifferenceModel`, and
    :class:`~bedroc.difference.models.tempered_full.TemperedFullModel`.

    Subclasses must set ``self.X_unlabeled``/``self.X_sigma_unlabeled`` in their own ``__init__``
    (e.g. via :func:`~bedroc.difference.utils.validate_observation_data`) and observe the
    unlabeled mixture likelihood as a variable named ``"obs_unlabeled"`` with
    ``dims=("observation_unlabeled", "feature")`` in their ``build_model``.
    """

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
        faceted by feature alone rather than by category and feature. (Some subclasses express
        the training likelihood as a tempered :class:`~pymc.Potential` with no associated
        random-sampling method, in which case this is the only predictive check available at
        all — see the subclass's own ``_build_plot_dict`` for details.)

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

        See :meth:`plot_prior_predictive_unlabeled` for why this is faceted by feature alone.

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


class CategoryClassifierBase(ABC):
    """Base class for category classifiers.

    Implementations must also expose a ``coords`` attribute (a ``dict[str, NpArray]`` including a
    ``"category"`` entry), used by :meth:`plot_group_fraction_posterior`. This isn't declared as a
    formal abstract property because :class:`~bedroc.difference.base.CategoryComparisonBase`
    (which :class:`~bedroc.difference.models.unified_covariance.UnifiedCovarianceModel` also
    inherits from) sets it as a plain instance attribute rather than a property; declaring it here
    as a property would shadow that assignment via the descriptor protocol and break it.
    """

    @abstractmethod
    def pi_0_samples(self, *args, **kwargs) -> NpFloat:
        """Posterior samples of the fraction of samples belonging to category 0 in the unlabeled
        dataset.

        Implementations must return exactly one fraction per posterior draw (i.e. shape
        ``(n_chains * n_draws,)``, matching the fitted model's ``chain``/``draw`` dimensions
        flattened), so that samples are index-aligned with the model's other per-draw posterior
        quantities.

        Args:
            *args: Positional arguments passed to the method
            **kwargs: Keyword arguments passed to the method

        Returns:
            Posterior samples of the fraction of samples belonging to category 0 in the unlabeled
            dataset, shape ``(n_chains * n_draws,)``.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def plot_group_fraction_posterior(
        self,
        bins: int = 50,
        n_grid: int = 2001,
        category_colors: tuple[str, str] = DEFAULT_CATEGORY_COLORS,
        category_counts: pd.Series | None = None,
        prior_alpha: float | None = None,
        prior_beta: float | None = None,
        ax: Axes | None = None,
        random_seed: int | None = None,
    ) -> Axes:
        """Plots the posterior distribution of the fraction of samples belonging to category 0.

        The posterior is shown together with the beta prior and, where available, the observed
        group fraction. Shared by every implementation, since it only depends on
        :meth:`pi_0_samples` and :attr:`coords`.

        Args:
            bins: Number of bins for the histogram. Defaults to ``50``.
            n_grid: Number of grid points for the prior and perfect-classification limit. Defaults
                to ``2001``.
            category_colors: Colors for the two categories. Defaults to
                :data:`~bedroc.difference.DEFAULT_CATEGORY_COLORS`.
            category_counts: Known counts for the two categories. If ``None``, the observed
                fractions are not plotted. Defaults to ``None``.
            prior_alpha: Alpha parameter of the Beta prior on the fraction of category 0. Defaults
                to whatever prior the implementation itself was fit with, if it stores one (as
                ``self._prior_alpha``, e.g. :class:`~bedroc.difference.models.unified_covariance.UnifiedCovarianceModel`),
                otherwise ``1.0``.
            prior_beta: Beta parameter of the Beta prior on the fraction of category 0. Same
                fallback behavior as ``prior_alpha``, via ``self._prior_beta``.
            ax: Matplotlib axes on which to plot. If ``None``, a new figure and axes are created.
            random_seed: Random seed for reproducibility, forwarded to :meth:`pi_0_samples`.
                Ignored by implementations whose ``pi_0`` is sampled jointly with the rest of the
                posterior rather than resampled here. Defaults to ``None``.

        Returns:
            Matplotlib axes containing the posterior group-fraction plot
        """
        resolved_prior_alpha: float = (
            prior_alpha if prior_alpha is not None else getattr(self, "_prior_alpha", 1.0)
        )
        resolved_prior_beta: float = (
            prior_beta if prior_beta is not None else getattr(self, "_prior_beta", 1.0)
        )

        return plot_group_fraction_posterior(
            self.pi_0_samples(
                prior_alpha=resolved_prior_alpha,
                prior_beta=resolved_prior_beta,
                random_seed=random_seed,
            ),
            prior_alpha=resolved_prior_alpha,
            prior_beta=resolved_prior_beta,
            bins=bins,
            n_grid=n_grid,
            category_names=self.coords["category"],  # pyright: ignore[reportAttributeAccessIssue]
            category_colors=category_colors,
            category_counts=category_counts,
            ax=ax,
        )


class LogLikelihoodModelProtocol(Protocol):
    """Protocol for a fitted model that can evaluate class-conditional log likelihoods.

    This is the minimal interface a stage-2 classifier (e.g. :class:`StandardClassifierModel`)
    needs from a fitted stage-1 category comparison model, so that stage-2 classifiers are not
    coupled to a specific stage-1 model implementation.
    """

    @property
    def coords(self) -> dict[str, NpArray]:
        """PyMC coordinates used to fit the model (must include a ``"category"`` entry)."""
        ...

    @property
    def name(self) -> str:
        """Name of the fitted model or analysis."""
        ...

    def compute_log_likelihood(
        self, X: NpFloat, *, X_sigma: NpFloat | None = None, category_idx: NpInt
    ) -> xr.Dataset:
        """Computes posterior log likelihoods for new observations under a category assignment.

        Args:
            X: New observations to evaluate, shape (n_samples, n_features)
            X_sigma: Optional uncertainties for the new observations, shape
                (n_samples, n_features). If ``None``, assumes no uncertainty in the new
                observations. Defaults to ``None``.
            category_idx: Category indices for the new observations, shape (n_samples,). Must be
                0 or 1, corresponding to the two categories in the fitted model.
        """
        ...


class PipelineProtocol(Protocol):
    """Protocol for pipelines."""

    def __call__(
        self,
        data: DataContainer,
        *,
        output_directory: Path | None = None,
        random_seed: int | None = RANDOM_SEED,
        build_model_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Runs the pipeline.

        Args:
            data: The container holding the input data for the pipeline
            output_directory (Path | None): Optional path to the directory where output files will
                be saved. If ``None``, no output files will be saved.
            random_seed: Optional random seed for reproducible results. Defaults to
                :data:`~bedroc.RANDOM_SEED`.
            build_model_kwargs: Optional keyword arguments passed to the model's ``build_model()``
                method (e.g. subclass-specific prior hyperparameters). Defaults to ``None``.

        Returns:
            Result of the pipeline, which may vary depending on the specific implementation
        """
        ...


def build_pipeline(model_class: type[CategoryComparisonBase]) -> PipelineProtocol:
    """Builds a pipeline function for a given category comparison model class.

    Args:
        model_class: The class of the category comparison model to be used in the pipeline. Must
            be a subclass of :class:`CategoryComparisonBase`.

    Returns:
        A pipeline function that conforms to the :class:`PipelineProtocol`.
    """

    def pipeline(
        data: DataContainer,
        *,
        output_directory: Path | None = None,
        random_seed: int | None = RANDOM_SEED,
        build_model_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> CategoryComparisonBase:
        """Pipeline for running a category comparison model on a dataset.

        This provides a basic pipeline for running a standard analysis and generating the
        associated figures, including feature correlation-coefficient plots for the full dataset
        and for the train/test split individually. For more customized analyzes, you may wish to
        create your own pipeline.

        Args:
            data: The container containing the dataset to analyze
            output_directory: Directory to save generated figures. If ``None``, figures are not
                saved.
            random_seed: Random seed for reproducibility. Defaults to :data:`~bedroc.RANDOM_SEED`.
            build_model_kwargs: Optional keyword arguments passed to the model's ``build_model()``
                method (e.g. subclass-specific prior hyperparameters). Defaults to ``None``.
            **kwargs: Additional keyword arguments to pass to the model's constructor

        Returns:
            The :class:`CategoryComparisonBase` instance
        """
        logger.info("Running category comparison pipeline for %s", data.name)

        if build_model_kwargs is None:
            build_model_kwargs = {}

        if output_directory is not None:
            output_directory = Path(output_directory)
            output_directory.mkdir(parents=True, exist_ok=True)
            logger.info("Output directory: %s", output_directory)
        else:
            logger.info("Output directory not specified. Figures will not be saved.")

        train, test = data.train_test_split(random_state=random_seed)

        # Plot the feature correlation structure for the full dataset, then the train/test split
        # alone, to check the split didn't skew either subset's correlation structure relative to
        # the full dataset. Also dump the underlying covariance matrix (of the standardized
        # features) to Excel for each: the pooled-across-categories version is a general,
        # as-observed diagnostic, while the within-category version is the one that matches
        # UnifiedCovarianceModel's cov_shared assumption and is the correct choice to reuse as
        # SyntheticDataGenerator's covariance argument (see
        # DataDiagnostics.within_category_covariance_matrix's docstring).
        for subset in (data, train, test):
            # Corner plot
            plot_corner(subset, output_directory=output_directory)

            # Eigenvalue decomposition
            eigen = subset.diagnostics.covariance_eigenanalysis()
            if output_directory is not None:
                eigen.to_excel(output_directory / f"{subset.name}_covariance_eigenanalysis.xlsx")

            # How much of the real category separation (Mahalanobis D^2) lands on each of the
            # above principal directions
            alignment = subset.diagnostics.mahalanobis_alignment()
            if output_directory is not None:
                alignment.to_excel(output_directory / f"{subset.name}_mahalanobis_alignment.xlsx")

            # Correlation coefficient plot
            ax = subset.diagnostics.plot_correlation_coefficient()
            ax.set_title(f"{subset.name}: {ax.get_title()}")
            save_figure(
                ax.get_figure(),  # pyright: ignore[reportArgumentType]
                Path(f"{subset.name}_correlation_coefficient"),
                output_directory,
            )

            # Covariance matrix dump to Excel: pooled (general/raw diagnostic) and within-category
            # (matches UnifiedCovarianceModel's cov_shared assumption)
            if output_directory is not None:
                subset.diagnostics.covariance_matrix().to_excel(
                    output_directory / f"{subset.name}_covariance_matrix.xlsx"
                )
                subset.diagnostics.within_category_covariance_matrix().to_excel(
                    output_directory / f"{subset.name}_covariance_matrix_within_category.xlsx"
                )

        model: CategoryComparisonBase = model_class.from_data_container(
            data.name, train, unlabeled_data=test, **kwargs
        )

        model.build_model(**build_model_kwargs)
        model.run_inference(random_seed=random_seed)
        model.generate_plots(
            output_directory=output_directory, title=True, random_seed=random_seed
        )

        logger.info("Category comparison pipeline completed for %s", data.name)

        return model

    return pipeline
