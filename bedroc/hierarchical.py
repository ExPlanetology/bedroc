# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Utilities for building and working with Bayesian hierarchical models"""

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
from numpy.typing import ArrayLike

from bedroc.core import RANDOM_SEED, save_figure
from bedroc.type_aliases import NpArray, NpFloat, NpInt

logger: logging.Logger = logging.getLogger(__name__)

LOW_PERCENTILE: float = 2.5
"""Low percentile for credible intervals"""
HIGH_PERCENTILE: float = 97.5
"""High percentile for credible intervals"""


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

    if sample_names is None:
        sample_names = [f"Sample {i}" for i in range(n_samples)]
    sample_names = list(sample_names)

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n_features)]
    feature_names = list(feature_names)

    if group_names is None:
        group_names = [f"Group {i}" for i in np.unique(X_group_idx)]
    group_names = list(group_names)

    n_groups: int = len(group_names)

    if np.min(X_group_idx) < 0 or np.max(X_group_idx) >= n_groups:
        raise ValueError(f"X_group_idx contains indices outside the range [0, {n_groups - 1}]")

    coords: dict[str, list] = {
        "obs": sample_names,  # avoid collision with "sample" in PyMC's internal namespace
        "feature": feature_names,
        "group": group_names,
    }

    return coords


def zero_difference_model(
    X: NpFloat,
    X_group_idx: NpInt,
    *,
    group_names: Iterable | None = None,
    feature_names: Iterable | None = None,
    X_sigma: NpFloat | None = None,
    draws: int = 2000,
    tune: int = 1000,
    target_accept: float = 0.95,
    random_seed: int | None = None,
) -> tuple[pm.Model, xr.DataTree]:
    """Model assuming no difference between two groups.

    This model is a "null"-like version of the group-centric hierarchical model: it assumes that
    the feature-wise means of Group B are identical to those of Group A (i.e., delta = 0). Each
    feature has its own observation noise, shared across groups. Observations are modeled as
    independent given their feature means and noise.

    Args:
        X: Observations (n_samples, n_features)
        X_group_idx: Group ID of observations, must be 0 or 1 (n_samples,)
        group_names: Group names. Defaults to unique values in ``X_group_idx``.
        feature_names: Feature names. Defaults to sequential feature names.
        X_sigma: Sigma of observations (n_samples, n_features). Defaults to ``None``.
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
    coords: dict[str, list] = get_coords(
        X, X_group_idx, group_names=group_names, feature_names=feature_names
    )

    with pm.Model(coords=coords) as model:
        # Group A feature means (no pooling across features)
        mu_A = pm.Normal("mu_A", mu=0, sigma=3, dims="feature")

        # All group feature means
        mu = pm.Deterministic(
            "mu", pm.math.stack([mu_A, mu_A], axis=0), dims=("group", "feature")
        )  # No difference between groups

        # Feature-specific observation noise, shared across groups
        feature_sigma = pm.HalfNormal("feature_sigma", sigma=1.0, dims="feature")

        if X_sigma is not None:
            # The actual likelihood noise for each observation
            sigma_obs = pm.math.sqrt(X_sigma**2 + feature_sigma**2)  # pyright: ignore
        else:
            sigma_obs = feature_sigma

        # Build mu_obs with broadcasting
        mu_obs = mu[X_group_idx, ...]  # pyright: ignore

        # Likelihood
        pm.Normal("observed_data", mu=mu_obs, sigma=sigma_obs, observed=X)

        # Sampling
        idata: xr.DataTree = pm.sample(
            draws=draws, tune=tune, target_accept=target_accept, random_seed=random_seed
        )

    return model, idata


# TODO: Needs refreshing to be consistent with group centric model
def feature_centric_hierarchical_model(
    X: NpFloat,
    X_group_idx: NpInt,
    *,
    group_names: Iterable | None = None,
    feature_names: Iterable | None = None,
    X_sigma: NpFloat | None = None,
    draws: int = 2000,
    tune: int = 1000,
    target_accept: float = 0.95,
    random_seed: int | None = None,
) -> tuple[pm.Model, xr.DataTree]:
    """Bayesian hierarchical model for feature-centered group comparisons.

    This model estimates feature-wise latent structure shared across groups, while allowing
    group-specific deviations that are partially pooled across features.

    The model is feature-centric: each feature has a global baseline mean, and each group expresses
    deviations from this baseline with hierarchical shrinkage controlled at the feature level.

    This structure allows:
        - feature-specific heterogeneity in group effects
        - partial pooling of group deviations across features
        - stable estimation of group differences in high-dimensional settings

    Note:
        The variable names in the model are fixed and are propagated downstream and expected by
        helper functions and analysis/plotting utilities.

    Args:
        X: Observations (n_samples, n_features)
        X_group_idx: Group ID of observations (n_samples,)
        group_names: Group labels. Defaults to unique values in X_group_idx.
        feature_names: Feature names. Defaults to sequential feature labels.
        X_sigma: Measurement noise per observation (n_samples, n_features).
            If None, noise is inferred.
        draws: Number of posterior samples.
        tune: Number of warm-up steps.
        target_accept: NUTS target acceptance probability.
        random_seed: RNG seed for reproducibility.

    Returns:
        tuple:
            - PyMC model
            - ArviZ InferenceData
    """
    _, n_features = X.shape

    if group_names is None:
        group_names = np.unique(X_group_idx)
    group_names = list(group_names)

    n_groups = len(group_names)

    if np.min(X_group_idx) < 0 or np.max(X_group_idx) >= n_groups:
        raise ValueError(f"X_group_idx contains indices outside the range [0, {n_groups - 1}]")

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]
    feature_names = list(feature_names)

    coords: dict[str, list] = {"group": group_names, "feature": feature_names}

    group_sigma_prior: int = 5

    sigma_prior: int = 5

    with pm.Model(coords=coords) as model:
        # Global mean for each feature across all groups. This acts as the population-level center
        # toward which individual group means are shrunk.
        mu_global = pm.Normal("mu_global", mu=0, sigma=10, dims="feature")

        # Feature-specific scale describing how much group means are allowed to vary around the
        # global mean. Small values imply strong pooling (group means are similar), while large
        # values imply weak pooling (group means can differ substantially).
        sigma_group = pm.HalfNormal("sigma_group", sigma=group_sigma_prior, dims="feature")

        # Group-specific deviations from the global mean. Groups with limited data are shrunk more
        # strongly toward the population mean, whereas groups with abundant data are more strongly
        # informed by their own observations.
        mu_offset = pm.Normal("mu_offset", mu=0, sigma=sigma_group, dims=("group", "feature"))

        # Mean value of each feature for each group.
        #
        # For feature f and group g:
        #
        #     mu[g, f] = mu_global[f] + mu_offset[g, f]
        #
        # This defines a hierarchical model in which group means are drawn from a common population
        # distribution centred on mu_global.
        mu = pm.Deterministic("mu", mu_global + mu_offset, dims=("group", "feature"))

        # Feature-specific residual scatter (irreducible model + system noise)
        sigma_resid = pm.HalfNormal("sigma_resid", sigma=sigma_prior, dims="feature")

        # Total uncertainty per observation (used in likelihood) and feature-level summary (used
        # for effect sizes / plots).
        if X_sigma is not None:
            sigma_total = pm.math.sqrt(X_sigma**2 + sigma_resid**2)  # type: ignore
            pm.Deterministic(
                "sigma_total_feature",
                pm.math.sqrt(pm.math.mean(X_sigma**2, axis=0) + sigma_resid**2),  # type: ignore
                dims="feature",
            )

        else:
            sigma_total = sigma_resid  # broadcasts to (n_samples, n_features)
            pm.Deterministic("sigma_total_feature", sigma_resid, dims="feature")

        pm.Deterministic("sigma_total", sigma_total)

        mu_obs = mu[X_group_idx, ...]  # type: ignore

        # Likelihood
        # Assume every observed data point was generated from a Gaussian (normal) distribution
        pm.Normal("X_obs", mu=mu_obs, sigma=sigma_total, observed=X)

        # Sampling
        idata: xr.DataTree = pm.sample(
            draws=draws, tune=tune, target_accept=target_accept, random_seed=random_seed
        )

    return model, idata


class SyntheticDataGenerator:
    """Generates synthetic multivariate data for two types (A & B) with configurable parameters.

    Args:
        n_samples: Number of samples per type. Defaults to ``100``.
        n_features: Number of features per sample. Defaults to ``5``.
        feature_offsets: Optional shift to apply to the Type B feature means relative to Type A.
            May be either a scalar (applied to every feature) or an array of shape
            ``(n_features,)`` specifying per-feature offsets. Defaults to ``1.0``.
        feature_sigma: Standard deviation of the noise (stddev) for features. May be either a
            scalar (applied to every feature) or an array of shape ``(n_features,)`` specifying
            per-feature noise. Defaults to ``0.5``.
        random_seed: Optional seed for reproducibility. Defaults to ``None``.
        output_directory: Optional path to save generated data. Defaults to ``None`` (no saving).
    """

    def __init__(
        self,
        n_samples: int = 100,
        *,
        n_features: int = 5,
        feature_offsets: ArrayLike = 1.0,
        feature_sigma: ArrayLike = 0.5,
        random_seed: int | None = None,
        output_directory: Path | None = None,
    ):
        self.n_samples: int = n_samples
        self.n_features: int = n_features
        self.feature_offsets: NpFloat = np.full(self.n_features, feature_offsets, dtype=float)
        self.feature_sigma: NpFloat = np.full(self.n_features, feature_sigma, dtype=float)
        self.random_seed: int | None = random_seed
        self.output_directory: Path | None = output_directory
        self._rng = np.random.default_rng(self.random_seed)

        # For Type A, each feature gets its own true mean (center of distribution)
        self.mu_A: NpFloat = self._rng.normal(loc=0.0, scale=1.0, size=self.n_features)
        logger.debug("mu_A = %s", self.mu_A)

        # Shift distribution of Type B relative to Type A by the specified offsets
        self.mu_B: NpFloat = self.mu_A + self.feature_offsets
        logger.debug("mu_B = %s", self.mu_B)

        # Internal storage for generated data
        self._X: NpFloat | None = None
        self._X_group_idx: NpInt | None = None

    @property
    def X(self) -> NpArray:
        """Type A data (n_samples, n_features)"""
        if self._X is None:
            raise ValueError("Data not yet generated. Call 'generate()' first.")

        return self._X

    @property
    def X_group_idx(self) -> NpInt:
        """Group idx"""
        if self._X_group_idx is None:
            raise ValueError("Data not yet generated. Call 'generate()' first.")

        return self._X_group_idx

    def generate(self) -> None:
        """Generates multivariate data for 2 types (A & B) and stores internally."""

        logger.info("Generating synthetic data with random_seed=%s", self.random_seed)

        # Generate samples
        X_A: NpFloat = self._rng.normal(
            self.mu_A, self.feature_sigma, size=(self.n_samples, self.n_features)
        )
        logger.debug("X_A = %s", X_A)
        X_B: NpFloat = self._rng.normal(
            self.mu_B, self.feature_sigma, size=(self.n_samples, self.n_features)
        )
        logger.debug("X_B = %s", X_B)

        # Store internally
        self._X = np.vstack([X_A, X_B])
        self._X_group_idx = np.hstack(
            [np.zeros(X_A.shape[0], dtype=int), np.ones(X_B.shape[0], dtype=int)]
        )

        logger.info(
            "Synthetic data generation complete. Generated %d samples per type with %d features.",
            self.n_samples,
            self.n_features,
        )

    def generate_out_of_sample_data(self, n_samples: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Generates out-of-sample synthetic data using previously-sampled true parameters.

        Args:
            n_samples: Number of out-of-sample points per type. Defaults to ``100``.

        Returns:
            tuple:
                - Type A data (n_samples, n_features)
                - Type B data (n_samples, n_features)
        """
        # Draw new samples from the same ground-truth distribution
        X_A_test: NpFloat = self._rng.normal(
            self.mu_A, self.feature_sigma, size=(n_samples, self.n_features)
        )
        X_B_test: NpFloat = self._rng.normal(
            self.mu_B, self.feature_sigma, size=(n_samples, self.n_features)
        )

        X_test = np.vstack([X_A_test, X_B_test])
        X_test_group_idx = np.hstack(
            [np.zeros(X_A_test.shape[0], dtype=int), np.ones(X_B_test.shape[0], dtype=int)]
        )

        return X_test, X_test_group_idx


class HierarchicalGroupModel:
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
    group differences. If some features are known a priori to behave fundamentally differently, a
    more flexible hierarchical structure may be preferable.

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
    ):
        logger.info("Creating a hierarchical group model for %s", name)
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

        self._idata: xr.DataTree | None = None
        self._model: pm.Model | None = None

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
    ) -> None:
        """Runs inference on the hierarchical model.

        Args:
            draws: Number of posterior samples to draw. Defaults to ``2000``.
            tune: Number of tuning steps. Defaults to ``1000``.
            target_accept: Target acceptance rate for NUTS sampler. Defaults to ``0.95``.
            random_seed: Random seed for reproducibility. Defaults to :obj:`RANDOM_SEED`.
        """
        logger.info(
            "Running inference with draws=%d, tune=%d, target_accept=%.2f",
            draws,
            tune,
            target_accept,
        )

        # Prior belief about effect sizes in SD units
        delta_scale_prior: float = 0.5

        with pm.Model(coords=self.coords) as model:
            # Group A feature means (standardized space)
            mu_A = pm.Normal("mu_A", mu=0, sigma=1, dims="feature")

            # Hierarchical effect scale
            delta_scale = pm.HalfNormal("delta_scale", sigma=delta_scale_prior)

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

            # Additional parameter compared to the normal distribution
            # nu_minus_1 = pm.Exponential("nu_minus_1", 1 / 50.0)  # , dims="group")

            if self.X_sigma is not None:
                # The actual likelihood noise for each observation
                # sigma_obs = pm.math.sqrt(self.X_sigma**2 + sigma ** 2)  # pyright: ignore
                sigma_total_feature = pm.math.sqrt(
                    pm.math.mean(self.X_sigma**2, axis=0) + sigma**2
                )
            else:
                # sigma_obs = sigma
                sigma_total_feature = sigma

            pm.Deterministic("effect_size", delta / sigma_total_feature, dims="feature")

            # Per-(group, feature) likelihood; missing values contribute no likelihood term (MCAR/MAR).
            for g_idx, g_name in enumerate(self.coords["group"]):
                for f_idx, f_name in enumerate(self.coords["feature"]):
                    obs_idx: NpArray = np.where(
                        np.isfinite(self.X[:, f_idx]) & (self.X_group_idx == g_idx)
                    )[0]
                    if len(obs_idx) == 0:
                        continue
                    if self.X_sigma is not None:
                        sigma_f = pm.math.sqrt(
                            self.X_sigma[obs_idx, f_idx] ** 2 + sigma[f_idx] ** 2
                        )
                    else:
                        sigma_f = sigma[f_idx]

                    # Laplace scale parameter
                    b_f = sigma_f / pm.math.sqrt(2)

                    # pm.Normal(
                    #     f"obs_{g_name}_{f_name}",
                    #     mu=mu[g_idx, f_idx],
                    #     sigma=sigma_f,
                    #     observed=self.X[obs_idx, f_idx],
                    # )
                    # pm.StudentT(
                    #     f"obs_{g_name}_{f_name}",
                    #     nu=nu_minus_1 + 1,
                    #     mu=mu[g_idx, f_idx],
                    #     sigma=sigma_f,
                    #     observed=self.X[obs_idx, f_idx],
                    # )
                    pm.Laplace(
                        f"obs_{g_name}_{f_name}",
                        mu=mu[g_idx, f_idx],
                        b=b_f,
                        observed=self.X[obs_idx, f_idx],
                    )

            # Sampling and store objects for later access
            self._idata = pm.sample(
                draws=draws, tune=tune, target_accept=target_accept, random_seed=random_seed
            )
            self._model = model

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
        savefig_kwargs: dict[str, Any] | None = None,
    ) -> az.PlotCollection:
        """Plots posterior predictive check (in-sample predictions).

        This performs in-sample predictions to assess how well the model fits the observed data,
        i.e., test how well the model can reproduce the data it was trained on.

        Args:
            sample_kwargs: Keyword arguments for :func:`pymc.sample_posterior_predictive`. Defaults
                to ``None``.
            savefig_kwargs: Override keyword arguments for :func:`matplotlib.pyplot.savefig`.
                Defaults to ``None``.

        Returns:
            Plot collection
        """
        if sample_kwargs is None:
            sample_kwargs = {}

        posterior_predictive: xr.DataTree = pm.sample_posterior_predictive(
            self.idata, model=self.model, **sample_kwargs
        )
        pc: az.PlotCollection = az.plot_ppc_dist(
            posterior_predictive,
            group="posterior_predictive",
            # kind="kde",
            kind="hist",
            visuals={"observed_dist": {"color": "black"}},
        )
        pc.get_viz("figure").tight_layout(h_pad=1.0)

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
