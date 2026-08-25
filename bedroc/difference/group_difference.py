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

This model can be used as the first stage of a two-step generative classifier. Once fitted, the
model can evaluate the class-conditional likelihoods for new data points, which, when combined with
class priors, enables Bayesian classification.
"""

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pymc as pm
import xarray as xr
from matplotlib.lines import Line2D

from bedroc import override
from bedroc.core.data_container import RANDOM_SEED, DataContainer
from bedroc.core.plotting import add_xaxis_labels_to_bottom_row, save_figure
from bedroc.core.type_aliases import NpArray, NpFloat, NpInt
from bedroc.difference import DEFAULT_GROUP_NAMES
from bedroc.difference.group_base import GroupComparisonBase
from bedroc.difference.likelihood_models import LikelihoodModel, StudentTLikelihood
from bedroc.difference.validation import validate_group_idx, validate_observation_data

logger: logging.Logger = logging.getLogger(__name__)


class HierarchicalGroupDifferenceModel(GroupComparisonBase):
    """Hierarchical Bayesian model for comparing two groups across multiple features.

    Group 0 is treated as the reference group. Each feature has a reference-group mean ``mu_0`` and
    a group difference ``delta``, such that the corresponding group means are

    ``mu[0] = mu_0``
    ``mu[1] = mu_0 + delta``.

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
        X_sigma: Optional measurement uncertainties with the same shape as ``X``. Defaults to
            ``None``, in which case the model assumes that the observations are exact.
        feature_names: Optional names for each feature. If not provided, defaults to
            ``["Feature 0", "Feature 1", ..., "Feature N"]``.
        group_names: Optional names for each group. Defaults to :obj:`DEFAULT_GROUP_NAMES`.
        likelihood_model: Likelihood model implementation used for the observations. Defaults to
            :class:`StudentTLikelihood`.
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
        likelihood_model: type[LikelihoodModel] = StudentTLikelihood,
    ):
        logger.info("Creating a hierarchical group difference model for %s", name)
        super().__init__(
            name,
            X,
            X_group_idx,
            X_sigma=X_sigma,
            feature_names=feature_names,
            group_names=group_names,
        )
        self._likelihood_model: LikelihoodModel = likelihood_model()

    @override
    def build_model(self) -> None:
        # Observed data
        # Flatten finite sample-feature pairs into the observation dimension. Missing values are
        # omitted from the likelihood.
        sample_idx, feature_idx = np.where(np.isfinite(self.X))
        X_group_idx: NpInt = self.X_group_idx[sample_idx]

        X_data_np: NpFloat = self.X[sample_idx, feature_idx]

        with pm.Model(coords=self.coords) as model:
            # Group 0 feature means (standardized space)
            mu_0 = pm.Normal("mu_0", mu=0, sigma=0.5, dims="feature")

            # Hierarchical effect scale
            delta_scale = pm.HalfNormal("delta_scale", sigma=0.5)

            # Feature-wise group differences
            delta = pm.Normal("delta", mu=0, sigma=delta_scale, dims="feature")

            # All group feature means
            mu = pm.Deterministic(
                "mu", pm.math.stack([mu_0, mu_0 + delta], axis=0), dims=("group", "feature")
            )

            # Intrinsic feature variability. ``sigma`` is expressed in standardized feature units.
            sigma = pm.HalfNormal("sigma", sigma=0.5, dims="feature")

            # Intrinsic effect size: separation of the underlying groups in units of their
            # intrinsic within-feature standard deviation. Convenient for downstream plotting to
            # not have underscores in the name since this will be used as the label
            pm.Deterministic("effect_size", delta / sigma, dims="feature")

            # Data
            X_data = pm.Data("X_data", X_data_np, dims="observation")
            feature_idx_data = pm.Data("feature_idx", feature_idx, dims="observation")
            group_idx_data = pm.Data("group_idx", X_group_idx, dims="observation")

            # Combine intrinsic variability with per-observation measurement uncertainty.
            X_sigma_observed = self.X_sigma[sample_idx, feature_idx]
            X_sigma_data = pm.Data("X_sigma", X_sigma_observed, dims="observation")
            sigma_observed = pm.math.sqrt(X_sigma_data**2 + sigma[feature_idx_data] ** 2)  # pyright: ignore[reportOperatorIssue]

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

        self._model = model

    def compute_log_likelihood(
        self, X: NpFloat, *, X_sigma: NpFloat | None = None, group_idx: NpInt
    ) -> xr.Dataset:
        """Computes posterior log likelihoods for new observations under a group assignment.

        The fitted model parameters are held fixed at each posterior draw while the likelihood
        of each observation is evaluated under the supplied group assignment.

        Args:
            X: Data to evaluate, with shape ``(n_samples, n_features)``. Missing values should
                be represented by ``NaN``.
            X_sigma: Optional 1-sigma uncertainties for ``X``, with shape
                ``(n_samples, n_features)``. If ``None``, observations are treated as exact.
            group_idx: Group index for each sample, with shape ``(n_samples,)``. Values must
                correspond to the groups defined by the fitted model.

        Returns:
            Dataset containing the posterior log likelihood for each finite observation, with
            dimensions ``(chain, draw, observation)``. The ``sample_idx`` and ``feature_idx``
            coordinates map each observation back to the original ``X`` array.

        Raises:
            ValueError: If ``X``, ``X_sigma``, or ``group_idx`` has an invalid shape or
                contains invalid values.
        """
        X, X_sigma = validate_observation_data(X, X_sigma=X_sigma)
        group_idx = validate_group_idx(group_idx, n_samples=X.shape[0])

        # Convert the sample/feature matrix into the observation-level representation expected by
        # the PyMC model.
        sample_idx, feature_idx = np.where(np.isfinite(X))

        X_data: NpFloat = X[sample_idx, feature_idx]
        sigma_data: NpFloat = X_sigma[sample_idx, feature_idx]

        # A group assignment is defined per sample, whereas the likelihood is defined per observed
        # feature. Map the sample-level group index onto observations.
        observation_group_idx: NpInt = group_idx[sample_idx]

        coords: dict[str, NpArray] = {"observation": np.arange(len(X_data))}

        data: dict[str, NpArray] = {
            "X_data": X_data,
            "feature_idx": feature_idx,
            "group_idx": observation_group_idx,
            "X_sigma": sigma_data,
        }

        with self.model:
            pm.set_data(data, coords=coords)

            log_likelihood: xr.Dataset = pm.compute_log_likelihood(
                self.idata,
                var_names=["observations"],
                extend_inferencedata=False,
            )  # pyright: ignore[reportAssignmentType]

        log_likelihood = log_likelihood.rename({"observations": "log_likelihood"})

        # Map the flattened observations back to the original sample/feature matrix.
        log_likelihood = log_likelihood.assign_coords(
            sample_idx=("observation", sample_idx), feature_idx=("observation", feature_idx)
        )

        return log_likelihood

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
            self.coords["group"][group_idx] + ", " + self.coords["feature"][feature_idx]
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
            col_wrap=len(self.coords["feature"]),  # one column per feature
            **pc_kwargs,
        )

        add_xaxis_labels_to_bottom_row(pc, "Standardized units")

        fig = pc.get_viz("figure")
        fig.tight_layout(h_pad=0.3)

        # For comparison with different likelihoods, set x-limits to a common range for all feats
        for ax in fig.axes:
            ax.set_xlim(x_min, x_max)

        return pc


def pipeline(
    data: DataContainer,
    *,
    group_names: tuple[str, str],
    group_data_column: str,
    output_directory: Path | None = None,
    random_seed: int | None = RANDOM_SEED,
    title_fontsize: str = "large",
) -> HierarchicalGroupDifferenceModel:
    """Pipeline for running the hierarchical group difference model on a dataset

    This provides a basic pipeline for running a standard analysis and generating the associated
    figures. For more customized analyses, you may wish to create your own pipeline.

    Args:
        data: The container containing the dataset to analyze
        group_names: Names of the two groups to compare
        group_data_column: Column name in ``data.metadata`` that contains the group index for each
            sample.
        output_directory: Directory to save generated figures. If ``None``, figures are not saved.
        random_seed: Random seed for reproducibility. Defaults to :obj:`RANDOM_SEED`.
        title_fontsize: Font size for plot titles. Defaults to ``large``.

    Returns:
        The fitted :class:`HierarchicalGroupDifferenceModel` instance
    """
    logger.info("Running hierarchical group difference pipeline for %s", data.name)

    if output_directory is not None:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory: %s", output_directory)
    else:
        logger.info("Output directory not specified. Figures will not be saved.")

    train, _ = data.train_test_split(
        random_state=random_seed, stratify=data.metadata[group_data_column]
    )
    fitted_model: HierarchicalGroupDifferenceModel = HierarchicalGroupDifferenceModel(
        data.name,
        train.values_std.to_numpy(),
        train.metadata[group_data_column].to_numpy(),
        group_names=group_names,
        feature_names=train.feature_names,
        X_sigma=train.uncertainties_std.to_numpy(),
    )
    fitted_model.build_model()

    if output_directory is not None:
        fitted_model.plot_model(output_directory)

    fitted_model.run_inference(random_seed=random_seed)

    # Figure generation

    pc: az.PlotCollection = fitted_model.plot_prior_predictive()
    save_figure(pc, f"{data.name}_prior_predictive", output_directory=output_directory)

    pc: az.PlotCollection = fitted_model.plot_posterior_predictive()
    fig = pc.get_viz("figure")
    legend_handles: list = [
        Line2D([0], [0], color="black", linewidth=2, label="Observed"),
        Line2D([0], [0], color="C0", linewidth=1.5, label="Posterior predictive"),
    ]
    fig.legend(handles=legend_handles, frameon=True)
    save_figure(pc, f"{data.name}_posterior_predictive", output_directory=output_directory)

    pc = fitted_model.plot_posterior_distributions()
    pc.add_title("Posterior Distributions", fontsize=title_fontsize)
    fig = pc.get_viz("figure")
    legend_handles: list = [
        Line2D([0], [0], color="0.4", linewidth=2, marker="o", label="95% CrI"),
    ]
    fig.legend(handles=legend_handles, frameon=True)
    save_figure(pc, f"{data.name}_posterior_distributions", output_directory=output_directory)

    pc = fitted_model.plot_parameter_estimates()
    pc.add_title("Posterior parameter estimates", fontsize=title_fontsize)
    save_figure(
        pc, f"{data.name}_posterior_parameter_estimates", output_directory=output_directory
    )

    pc: az.PlotCollection = fitted_model.plot_effect_size()
    pc.add_title(
        f"Posterior effect size {fitted_model.difference_string}", fontsize=title_fontsize
    )
    save_figure(pc, f"{data.name}_posterior_effect_size", output_directory=output_directory)

    logger.info("Hierarchical group difference pipeline completed for %s", data.name)

    return fitted_model
