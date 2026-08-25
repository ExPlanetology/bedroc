# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Base classes and protocols"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

import arviz as az
import pymc as pm
import xarray as xr
from matplotlib.axes import Axes

from bedroc.core.data_container import RANDOM_SEED
from bedroc.core.plotting import add_xaxis_labels_to_bottom_row
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
        group_names: Optional names for each group. If not provided, defaults to
            :obj:`DEFAULT_GROUP_NAMES`.
    """

    def __init__(
        self,
        name: str,
        X: NpFloat,
        X_group_idx: NpInt,
        *,
        X_sigma: NpFloat | None = None,
        feature_names: Iterable | None = None,
        group_names: Iterable = DEFAULT_GROUP_NAMES,
    ):
        self.name: str = name
        self.X, self.X_sigma = validate_observation_data(X, X_sigma=X_sigma)
        self.X_group_idx = validate_group_idx(X_group_idx, n_samples=self.X.shape[0])
        self.coords: dict[str, NpArray] = get_coords(
            self.X, self.X_group_idx, feature_names=feature_names, group_names=group_names
        )
        self._idata: xr.DataTree | None = None
        self._model: pm.Model | None = None

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
            format: Format of the output file. Defaults to ``'pdf'``. Can be any format supported
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

    def plot_effect_size(
        self,
        var_names: list[str] | str | None = None,
        *,
        figsize: tuple = (8, 3),
        positive_labels: bool = True,
        negative_labels: bool = True,
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

        pc.get_viz("figure").tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.0)

        return pc

    def plot_parameter_estimates(
        self, var_names: list[str] | str | None = None, *, figsize: tuple = (8, 5)
    ) -> az.PlotCollection:
        """Plots parameter estimates as a forest plot.

        Args:
            var_names: List of variable names to plot. Can be a single string or an iterable of
                strings. Defaults to ``None`` to use default values that are typically of interest
                for group comparisons.
            figsize: Figure size. Defaults to ``(8, 5)``.

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

        pc.get_viz("figure").tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.0)

        return pc

    def plot_posterior_distributions(
        self,
        var_names: list[str] | str | None = None,
        *,
        figsize: tuple = (8, 5),
        col_wrap: int = 4,
    ) -> az.PlotCollection:
        """Plots posterior distributions of model parameters.

        Args:
            var_names: List of variable names to plot. Can be a single string or an iterable of
                strings. Defaults to ``None`` to use default values that are typically of interest
                for group comparisons.
            figsize: Figure size. Defaults to ``(8, 5)``.
            col_wrap: Number of columns to wrap the plots. Defaults to ``4``.

        Returns:
            Plot collection
        """
        if var_names is None:
            var_names = "mu"

        pc_kwargs: dict = {"figure_kwargs": {"figsize": figsize}}

        pc: az.PlotCollection = az.plot_dist(
            self.idata, var_names=var_names, col_wrap=col_wrap, **pc_kwargs
        )
        pc.get_viz("figure").tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.3)

        add_xaxis_labels_to_bottom_row(pc, "Standardized units")

        return pc

    def plot_prior_predictive(
        self, *, sample_kwargs: dict[str, Any] | None = None
    ) -> az.PlotCollection:
        """Plots prior predictive check.

        This plot is used to determine if the model can generate data plausibly shaped like the
        observed distributions.

        Args:
            sample_kwargs: Keyword arguments for :func:`pymc.sample_prior_predictive`. Defaults to
                ``None``.

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

        return pc


class GroupClassifierProtocol(Protocol):
    """Protocol for group classifiers

    This protocol defines the expected interface for group classifiers. Any class that implements
    this protocol should provide the following methods and properties.
    """

    def pi_0_samples(self) -> NpFloat:
        """Posterior samples of the fraction of samples belonging to group 0 in the unlabeled
        dataset."""
        ...
