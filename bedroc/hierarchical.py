# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Utilities for building and working with Bayesian hierarchical models"""

import logging
from collections.abc import Iterable
from dataclasses import KW_ONLY, dataclass, field
from pprint import pformat

import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import xarray as xr
from matplotlib.axes import Axes

from bedroc.type_aliases import NpArray, NpFloat, NpInt

logger: logging.Logger = logging.getLogger(__name__)


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


@dataclass
class TrueParams:
    """Container for true parameters used in synthetic data generation

    Args:
        mu_A: True means for Type A
        mu_B: True means for Type B
        difference_vector: True difference vector (Type B - Type A)
        sigma_A: True noise (stddev) for Type A
        sigma_B: True noise (stddev) for Type B
    """

    mu_A: NpFloat
    mu_B: NpFloat
    difference_vector: NpFloat
    sigma_A: NpFloat
    sigma_B: NpFloat


@dataclass
class SyntheticDataGenerator:
    """Generates synthetic multivariate data for two types (A & B) with configurable parameters.

    Args:
        n_samples: Number of samples per type. Defaults to ``100``.
        n_features: Number of features per sample. Defaults to ``5``.
        difference_scale: Controls how different Type B is from Type A. Defaults to ``2``.
        type_a_std_of_mean: Standard deviation for Type A feature means. Defaults to ``1``.
        type_b_std_of_mean: Standard deviation for Type B feature means. Defaults to ``1.5``.
        sigma_min: Minimum noise (stddev) for features. Defaults to ``0.5``.
        sigma_max: Maximum noise (stddev) for features. Defaults to ``2``.
        random_seed: Optional seed for reproducibility. Defaults to ``None``.
        heteroscedastic: If ``True``, generate independent sigma per type. However, note that the
            Bayesian models in this module are not configured to recover per-type sigmas. Defaults
            to ``False``.
    """

    n_samples: int = 100
    _: KW_ONLY
    n_features: int = 5
    difference_scale: float = 2.0
    type_a_std_of_mean: float = 1.0
    type_b_std_of_mean: float = 1.5
    sigma_min: float = 0.5
    sigma_max: float = 2.0
    random_seed: int | None = None
    heteroscedastic: bool = False
    # Internal storage for generated data
    _X: NpFloat | None = field(init=False, default=None)
    _X_group_idx: NpInt | None = field(init=False, default=None)
    _true_params: TrueParams | None = field(init=False, default=None)

    @property
    def X(self) -> NpArray:
        """Type A data (n_samples, n_features)"""
        if self._X is None:
            raise ValueError(
                "Data not yet generated. Call 'generate()' first."
            )  # pragma: no cover

        return self._X

    @property
    def X_group_idx(self) -> NpInt:
        """Group idx"""
        if self._X_group_idx is None:
            raise ValueError(
                "Data not yet generated. Call 'generate()' first."
            )  # pragma: no cover

        return self._X_group_idx

    @property
    def true_params(self) -> TrueParams:
        """True parameters used in data generation"""
        if self._true_params is None:
            raise ValueError(
                "Data not yet generated. Call 'generate()' first."
            )  # pragma: no cover

        return self._true_params

    def generate(self) -> None:
        """Generates multivariate data for 2 types (A & B) and stores internally."""

        logger.info("Generating synthetic data with random_seed=%s", self.random_seed)
        rng = np.random.default_rng(self.random_seed)

        # For Type A, each feature gets its own true mean (center of distribution)
        mu_A: NpFloat = rng.normal(loc=0.0, scale=self.type_a_std_of_mean, size=self.n_features)
        logger.debug("mu_A = %s", mu_A)

        # For Type B, each feature mean gets a random shift relative to Type A.
        # Scaling by difference_scale controls overall separation between types.
        raw_shift: NpFloat = rng.normal(
            loc=0.0, scale=self.type_b_std_of_mean, size=self.n_features
        )
        mu_B: NpFloat = mu_A + self.difference_scale * raw_shift
        logger.debug("mu_B = %s", mu_B)

        # Noise (standard deviation) per feature
        if self.heteroscedastic:
            # Noise varies across types as well as features
            sigma_A: NpFloat = rng.uniform(self.sigma_min, self.sigma_max, size=self.n_features)
            sigma_B: NpFloat = rng.uniform(self.sigma_min, self.sigma_max, size=self.n_features)
            logger.debug("sigma_A = %s", sigma_A)
            logger.debug("sigma_B = %s", sigma_B)
        else:
            # Noise only varies across features, not types
            sigma: NpFloat = rng.uniform(self.sigma_min, self.sigma_max, size=self.n_features)
            sigma_A = sigma_B = sigma
            logger.debug("sigma (shared) = %s", sigma)

        # Generate samples
        X_A: NpFloat = rng.normal(mu_A, sigma_A, size=(self.n_samples, self.n_features))
        logger.debug("X_A = %s", X_A)
        X_B: NpFloat = rng.normal(mu_B, sigma_B, size=(self.n_samples, self.n_features))
        logger.debug("X_B = %s", X_B)

        true_params: TrueParams = TrueParams(
            mu_A=mu_A, mu_B=mu_B, difference_vector=mu_B - mu_A, sigma_A=sigma_A, sigma_B=sigma_B
        )

        # Store internally
        self._X = np.vstack([X_A, X_B])
        self._X_group_idx = np.hstack(
            [np.zeros(X_A.shape[0], dtype=int), np.ones(X_B.shape[0], dtype=int)]
        )
        self._true_params = true_params

        logger.info(
            "Synthetic data generation complete. Generated %d samples per type with %d features.",
            self.n_samples,
            self.n_features,
        )
        logger.info("True parameters:\n%s", pformat(true_params))

    def generate_out_of_sample_data(self, n_samples: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Generates out-of-sample synthetic data using previously-sampled true parameters.

        Args:
            n_samples: Number of out-of-sample points per type. Defaults to ``100``.

        Returns:
            tuple:
                - Type A data (n_samples, n_features)
                - Type B data (n_samples, n_features)
        """
        rng = np.random.default_rng(self.random_seed)

        mu_A: NpFloat = self.true_params.mu_A
        mu_B: NpFloat = self.true_params.mu_B
        sigma_A: NpFloat = self.true_params.sigma_A
        sigma_B: NpFloat = self.true_params.sigma_B

        # Draw new samples from the same ground-truth distribution
        X_A_test: NpFloat = rng.normal(mu_A, sigma_A, size=(n_samples, self.n_features))
        X_B_test: NpFloat = rng.normal(mu_B, sigma_B, size=(n_samples, self.n_features))

        X_test = np.vstack([X_A_test, X_B_test])
        X_test_group_idx = np.hstack(
            [np.zeros(X_A_test.shape[0], dtype=int), np.ones(X_B_test.shape[0], dtype=int)]
        )

        return X_test, X_test_group_idx

    def plot(self) -> sns.PairGrid:
        """Plots a corner plot for comparing Type A vs Type B with overlay of true inputs.

        Returns:
            Pairgrid
        """
        feature_labels: pd.Series = pd.Series([f"Feature {i}" for i in range(self.n_features)])

        # Build DataFrame for seaborn
        df_A: pd.DataFrame = pd.DataFrame(self.X[self.X_group_idx == 0], columns=feature_labels)
        df_A["Type"] = "A"
        df_B: pd.DataFrame = pd.DataFrame(self.X[self.X_group_idx == 1], columns=feature_labels)
        df_B["Type"] = "B"
        df: pd.DataFrame = pd.concat([df_A, df_B], ignore_index=True)

        # Create corner plot
        pairgrid: sns.PairGrid = sns.pairplot(
            df, hue="Type", corner=True, plot_kws=dict(alpha=0.4, s=20), diag_kws=dict(alpha=0.6)
        )

        # Overlay true means and 1 sigma bands on diagonal
        mu_A: NpFloat = self.true_params.mu_A
        mu_B: NpFloat = self.true_params.mu_B
        sigma_A: NpFloat = self.true_params.sigma_A
        sigma_B: NpFloat = self.true_params.sigma_B

        for i, ax in enumerate(pairgrid.diag_axes):  # pyright: ignore since diag_axes is not None
            ax.axvline(mu_A[i], color="blue", linestyle="--", linewidth=2, label="_nolegend_")
            ax.axvline(mu_B[i], color="orange", linestyle="--", linewidth=2, label="_nolegend_")
            # Shaded sigma bands
            ax.axvspan(mu_A[i] - sigma_A[i], mu_A[i] + sigma_A[i], color="blue", alpha=0.1)
            ax.axvspan(mu_B[i] - sigma_B[i], mu_B[i] + sigma_B[i], color="orange", alpha=0.1)

        # Off-diagonal: true multivariate centers
        for row in range(self.n_features):  # row index in axes
            for col in range(row):  # col index in axes
                ax: Axes = pairgrid.axes[row, col]
                ax.plot(
                    mu_A[col],
                    mu_A[row],
                    "o",
                    color="blue",
                    markersize=8,
                    markeredgecolor="k",
                    label="_nolegend_",
                )
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

        return pairgrid
