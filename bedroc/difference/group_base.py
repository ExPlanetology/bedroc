# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Base classes and protocols"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, Protocol

import arviz as az
import numpy as np
import pymc as pm
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from bedroc import RANDOM_SEED
from bedroc.core.data_container import DataContainer
from bedroc.core.plotting import add_xaxis_labels_to_bottom_row, save_figure
from bedroc.core.type_aliases import NpArray, NpFloat, NpInt
from bedroc.difference import DEFAULT_GROUP_NAMES
from bedroc.difference.utils import get_coords
from bedroc.difference.validation import validate_group_idx, validate_observation_data

logger: logging.Logger = logging.getLogger(__name__)


class GroupComparisonBase(ABC):
    """Base class for group comparison models.

    Args:
        name: Name of the model or analysis
        X: Observation data, shape (n_samples, n_features)
        X_group_idx: Group indices for each sample, shape (n_samples,)
        X_sigma: Optional observation uncertainties, shape (n_samples, n_features). Defaults to
            ``None``, in which case the model assumes that the observations are exact.
        feature_names: Optional names for each feature. If not provided, defaults to
            ``["Feature 0", "Feature 1", ..., "Feature N"]``.
        group_names: Optional names for each group. Defaults to
            :data:`~bedroc.difference.DEFAULT_GROUP_NAMES`.
        **kwargs: Additional keyword arguments to pass to the model's constructor.
    """

    def __init__(
        self,
        name: str,
        X: NpFloat,
        X_group_idx: NpInt,
        *,
        X_sigma: NpFloat | None = None,
        feature_names: Iterable[str] | None = None,
        group_names: Iterable[str] = DEFAULT_GROUP_NAMES,
        **kwargs,
    ):
        del kwargs  # Unused in base class, but may be used in subclasses

        self.name: str = name
        self.X, self.X_sigma = validate_observation_data(X, X_sigma=X_sigma)
        self.X_group_idx = validate_group_idx(X_group_idx, n_samples=self.X.shape[0])
        self.coords: dict[str, NpArray] = get_coords(
            self.X, self.X_group_idx, feature_names=feature_names, group_names=group_names
        )
        self._idata: xr.DataTree | None = None
        self._model: pm.Model | None = None
        logger.info("Creating %s", self.__class__.__name__)

    @property
    def difference_string(self) -> str:
        """Returns a human-readable representation of group 1 relative to group 0."""
        return f"({self.coords['group'][1]} - {self.coords['group'][0]})"

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
    def build_model(self) -> pm.Model:
        """Builds the PyMC model for the group comparison and stores it in ``self._model``."""
        raise NotImplementedError("Subclasses must implement this method.")

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
                for group comparisons.
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
                for group comparisons.
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
                for group comparisons.
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
            figsize: Size of the figure. Defaults to ``(8, 5)``.
            x_min: Minimum value for x-axis limits. Defaults to ``-4.0``.
            x_max: Maximum value for x-axis limits. Defaults to ``4.0``.
            legend: Whether to include a legend. Defaults to ``True``.
            title: Whether to include a title. Defaults to ``True``.

        Returns:
            Plot collection
        """
        sample_idx, feature_idx = np.where(np.isfinite(self.X))
        group_idx: NpInt = self.X_group_idx[sample_idx]

        # There appears to be a limitation in ArviZ's plot_ppc_dist function that prevents it from
        # using a custom observation coordinate. As a workaround, filter the inference data to only
        # include the observed data and posterior predictive groups, then assign a new observation
        # coordinate according to how we wish to facet the plot.
        observation_group_feature = (
            self.coords["group"][group_idx] + ", " + self.coords["feature"][feature_idx]
        )

        predictive_data_with_obs_coords: xr.DataTree = predictive_data.filter(
            lambda node: node.name in ("observed_data", group)
        ).map_over_datasets(
            lambda node: node.assign_coords(observation=("observation", observation_group_feature))
        )

        # Hist is also not supported with faceting. Perhaps in future versions of ArviZ?
        pc_kwargs: dict = {"figure_kwargs": {"figsize": figsize}}

        pc: az.PlotCollection = az.plot_ppc_dist(
            predictive_data_with_obs_coords,
            group=group,
            kind="kde",
            cols=["observation"],
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
            fig.suptitle(f"{self.name}: {group.replace('_', ' ').title()}", y=1)

        fig.tight_layout(h_pad=1.0, w_pad=1.0)

        # For comparison with different likelihoods, set x-limits to a common range for all feats
        for ax in fig.axes:
            ax.set_xlim(x_min, x_max)

        return pc

    def plot_prior_predictive(
        self,
        *,
        sample_kwargs: dict[str, Any] | None = None,
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

        pc: az.PlotCollection = self._plot_predictive(
            prior_predictive,
            "prior_predictive",
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

        pc: az.PlotCollection = self._plot_predictive(
            posterior_predictive,
            "posterior_predictive",
            figsize=figsize,
            x_min=x_min,
            x_max=x_max,
            legend=legend,
            title=title,
        )

        return pc

    def generate_plots(
        self, output_directory: Path | str | None = None, title: bool = True
    ) -> dict[str, az.PlotCollection]:
        """Wrapper method to generate plots and save them to the specified output directory.

        Args:
            output_directory: Optional path to the directory where output files will be saved. If
                ``None``, no output files will be saved.
            title: Whether to include titles in the plots. Defaults to ``True``.

        Returns:
            Dictionary of plot collections with keys corresponding to plot types
        """
        handle_dict: dict[str, az.PlotCollection] = {}

        handle_dict["prior_predictive"] = self.plot_prior_predictive(title=title)
        handle_dict["posterior_predictive"] = self.plot_posterior_predictive(title=title)
        handle_dict["parameter_estimates"] = self.plot_parameter_estimates(title=title)
        handle_dict["posterior_distributions"] = self.plot_posterior_distributions(title=title)
        handle_dict["effect_sizes"] = self.plot_effect_sizes(title=title)

        if output_directory is not None:
            output_directory = Path(output_directory)
            output_directory.mkdir(parents=True, exist_ok=True)

            self.plot_model(output_directory=output_directory)

            for plot_type, pc in handle_dict.items():
                save_figure(pc, f"{self.name}_{plot_type}", output_directory)

        return handle_dict


class GroupClassifierProtocol(Protocol):
    """Protocol for group classifiers."""

    def pi_0_samples(self) -> NpFloat:
        """Posterior samples of the fraction of samples belonging to group 0 in the unlabeled
        dataset."""
        ...


class PipelineProtocol(Protocol):
    """Protocol for pipelines."""

    def __call__(
        self,
        data: DataContainer,
        group_data_column: str,
        *,
        group_names: tuple[str, str] = DEFAULT_GROUP_NAMES,
        output_directory: Path | None = None,
        random_seed: int | None = RANDOM_SEED,
    ) -> Any:
        """Runs the pipeline.

        Args:
            data: The container holding the input data for the pipeline
            group_data_column: The name of the column in the metadata that contains the group
                indices
            group_names: A tuple containing the names of the two groups for classification.
                Defaults to :data:`~bedroc.difference.DEFAULT_GROUP_NAMES`.
            output_directory (Path | None): Optional path to the directory where output files will
                be saved. If ``None``, no output files will be saved.
            random_seed: Optional random seed for reproducible results. Defaults to
                :data:`~bedroc.RANDOM_SEED`.

        Returns:
            Result of the pipeline, which may vary depending on the specific implementation
        """
        ...


def build_pipeline(model_class: type[GroupComparisonBase]) -> PipelineProtocol:
    """Builds a pipeline function for a given group comparison model class.

    Args:
        model_class: The class of the group comparison model to be used in the pipeline. Must be a
            subclass of :class:`GroupComparisonBase`.

    Returns:
        A pipeline function that conforms to the :class:`PipelineProtocol`.
    """

    def pipeline(
        data: DataContainer,
        group_data_column: str,
        *,
        group_names: tuple[str, str] = DEFAULT_GROUP_NAMES,
        output_directory: Path | None = None,
        random_seed: int | None = RANDOM_SEED,
        **kwargs,
    ) -> GroupComparisonBase:
        """Pipeline for running a group comparison model on a dataset.

        This provides a basic pipeline for running a standard analysis and generating the
        associated figures. For more customized analyzes, you may wish to create your own pipeline.

        Args:
            data: The container containing the dataset to analyze
            group_data_column: Column name in ``data.metadata`` that contains the group index for
                each sample.
            group_names: Names of the two groups to compare. Defaults to
                :data:`~bedroc.difference.DEFAULT_GROUP_NAMES`.
            output_directory: Directory to save generated figures. If ``None``, figures are not
                saved.
            random_seed: Random seed for reproducibility. Defaults to :data:`~bedroc.RANDOM_SEED`.
            **kwargs: Additional keyword arguments to pass to the model's constructor

        Returns:
            The :class:`GroupComparisonBase` instance
        """
        logger.info("Running group comparison pipeline for %s", data.name)

        if output_directory is not None:
            output_directory = Path(output_directory)
            output_directory.mkdir(parents=True, exist_ok=True)
            logger.info("Output directory: %s", output_directory)
        else:
            logger.info("Output directory not specified. Figures will not be saved.")

        train, _ = data.train_test_split(
            random_state=random_seed, stratify=data.metadata[group_data_column]
        )
        model: GroupComparisonBase = model_class(
            data.name,
            train.values_std.to_numpy(),
            train.metadata[group_data_column].to_numpy(),
            group_names=group_names,
            feature_names=train.feature_names,
            X_sigma=train.uncertainties_std.to_numpy(),
            **kwargs,
        )

        model.build_model()
        model.run_inference(random_seed=random_seed)
        model.generate_plots(output_directory=output_directory, title=True)

        logger.info("Group comparison pipeline completed for %s", data.name)

        return model

    return pipeline
