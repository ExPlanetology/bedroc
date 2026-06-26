# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Utilities for building and working with Bayesian hierarchical models"""

import logging
from collections.abc import Iterable
from dataclasses import KW_ONLY, dataclass, field
from pprint import pformat
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
from arviz import InferenceData
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.special import softmax
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from bedroc.type_aliases import NpArray, NpBool, NpFloat, NpInt

logger: logging.Logger = logging.getLogger(__name__)


def get_coords(
    X: NpFloat,
    X_group_idx: NpInt,
    *,
    group_names: Iterable | None = None,
    feature_names: Iterable | None = None,
) -> dict[str, list]:
    """Utility function to generate group and feature names with defaults.

    Args:
        X: Observations (n_samples, n_features)
        X_group_idx: Group ID of observations (n_samples,)
        group_names: Group names. Defaults to ``None`` to generate generic names.
        feature_names: Feature names. Defaults to ``None`` to generate sequential feature names.

    Returns:
        Dictionary of coordinates used for PyMC models
    """
    _, n_features = X.shape

    if group_names is None:
        group_names = [f"Group {i}" for i in np.unique(X_group_idx)]
    group_names = list(group_names)

    n_groups: int = len(group_names)

    if np.min(X_group_idx) < 0 or np.max(X_group_idx) >= n_groups:
        raise ValueError(f"X_group_idx contains indices outside the range [0, {n_groups - 1}]")

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n_features)]
    feature_names = list(feature_names)

    coords: dict[str, list] = {"group": group_names, "feature": feature_names}

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
) -> tuple[pm.Model, InferenceData]:
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
        pm.Normal("X_obs", mu=mu_obs, sigma=sigma_obs, observed=X)

        # Sampling
        idata: InferenceData = pm.sample(
            draws=draws, tune=tune, target_accept=target_accept, random_seed=random_seed
        )

    return model, idata


def group_centric_hierarchical_model(
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
) -> tuple[pm.Model, InferenceData]:
    """Bayesian hierarchical model for group-centric comparisons of two groups

    The model treats one group as a reference and estimates a mean for each feature in that group.
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
        variance (or is approximately on a comparable scale). All parameters, including ``delta``
        and ``feature_sigma``, are therefore interpreted in standardized feature units.

        The variable names in the model are fixed and are propagated downstream and expected by
        helper functions and analysis/plotting utilities.

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

    # Prior belief about effect sizes in SD units
    delta_scale_prior: int = 1

    with pm.Model(coords=coords) as model:
        # Group A feature means (standardized space)
        mu_A = pm.Normal("mu_A", mu=0, sigma=3, dims="feature")

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
        # feature_sigma is expressed in standardized feature units and is learned from the data.
        feature_sigma = pm.HalfNormal("feature_sigma", sigma=1.0, dims="feature")

        if X_sigma is not None:
            # The actual likelihood noise for each observation
            sigma_obs = pm.math.sqrt(X_sigma**2 + feature_sigma**2)  # pyright: ignore
        else:
            sigma_obs = feature_sigma

        mu_obs = mu[X_group_idx, ...]  # pyright: ignore

        # Likelihood
        # Assume every observed data point was generated from a Gaussian (normal) distribution
        # whose standard deviation is sqrt(X_sigma^2 + feature_sigma^2) when measurement error is
        # provided, otherwise feature_sigma.
        pm.Normal("X_obs", mu=mu_obs, sigma=sigma_obs, observed=X)

        # Sampling
        idata: InferenceData = pm.sample(
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
) -> tuple[pm.Model, InferenceData]:
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
        idata: InferenceData = pm.sample(
            draws=draws, tune=tune, target_accept=target_accept, random_seed=random_seed
        )

    return model, idata


def feature_log_likelihood(
    idata: InferenceData, X_new: NpFloat, *, X_new_sigma: NpFloat | None = None
) -> NpFloat:
    """Returns per-feature log likelihood.

    Args:
        idata: Inference data
        X_new: New data (n_samples_new, n_features)
        X_new_sigma: Optional known 1-sigma uncertainties of new data (n_samples_new, n_features).
            Defaults to ``None``.

    Returns:
        log-likelihood for each feature
    """
    # (n_draws, n_groups, n_features)
    mu_samples: NpFloat = idata["posterior"]["mu"].stack(draws=("chain", "draw")).values
    mu_samples = np.transpose(mu_samples, (2, 0, 1))  # (draws, group, feature)
    # logger.debug("mu_A_samples.shape = %s", mu_samples.shape)

    # (n_draws, n_features)
    feature_sigma_samples: NpFloat = (
        idata["posterior"]["feature_sigma"].stack(draws=("chain", "draw")).values
    )
    feature_sigma_samples = np.transpose(feature_sigma_samples, (1, 0))  # (draws, feature)
    # logger.debug("feature_sigma_samples.shape = %s", feature_sigma_samples.shape)

    # Expand data
    X_b: NpFloat = X_new[None, :, None, :]  # (1, samples, 1, features)

    # Total observational noise
    if X_new_sigma is not None:
        sigma_b: NpFloat = np.sqrt(
            feature_sigma_samples[:, None, :] ** 2 + X_new_sigma[None, :, :] ** 2
        )  # (draws, samples, features)
    else:
        sigma_b = feature_sigma_samples[:, None, :]  # (draws, 1, features)

    sigma_b = sigma_b[:, :, None, :]  # (draws, samples, 1, features)

    # Compute log-likelihood:
    # (draws, samples, groups, features)
    log_lik_feat: NpFloat = -0.5 * (
        ((X_b - mu_samples[:, None, :, :]) ** 2) / (sigma_b**2) + np.log(2 * np.pi * sigma_b**2)
    )

    return log_lik_feat


def feature_log_likelihood_diff(
    idata: InferenceData, X_new: NpFloat, *, X_new_sigma: NpFloat | None = None
) -> NpFloat:
    """Computes the log-likelihood difference for each feature between two groups.

    Args:
        idata: Inference data
        X_new: New data (n_samples_new, n_features)
        X_new_sigma: Optional known 1-sigma uncertainties of new data (n_samples_new, n_features).
            Defaults to ``None``.

    Returns:
        log-likelihood difference: log p(X|B) - log p(X|A)
    """
    log_lik_feat: NpFloat = feature_log_likelihood(idata, X_new, X_new_sigma=X_new_sigma)

    # How much each feature prefers group B versus group A in log-likelihood
    delta_log_lik_feat: NpFloat = log_lik_feat[:, :, 1, :] - log_lik_feat[:, :, 0, :]

    return delta_log_lik_feat


def explain_samples(
    idata: InferenceData,
    X_new: NpFloat,
    *,
    X_new_sigma: NpFloat | None = None,
    X_new_group_id: NpInt | None = None,
    prior_A: float = 0.5,
) -> dict[str, Any]:
    """Explains the classification of each sample in terms of feature contributions.

    Args:
        idata: Inference data
        X_new: New data (n_samples_new, n_features)
        X_new_sigma: Optional known 1-sigma uncertainties of new data (n_samples_new, n_features).
            Defaults to ``None``.
        X_new_group_id: Group ID of observations (n_samples,). Defaults to ``None``.
        prior_A: Prior probability of Type A. The prior probability of Type B is
            taken as ``1 - prior_A``. Defaults to ``0.5``.

    Returns:
        Dictionary containing:

        - ``mean``:
            Posterior mean feature contribution
            (n_samples_new, n_features).

        - ``std``:
            Posterior standard deviation of the contribution
            (n_samples_new, n_features).

        - ``ci95``:
            95% credible interval of the contribution
            (2, n_samples_new, n_features), with lower then upper bounds.

        - ``p_support_B``:
            Posterior probability that the feature favours Group B,
            i.e. ``P(log p_B > log p_A)``
            (n_samples_new, n_features).

        - ``posterior``:
            Full posterior feature contributions
            (n_draws, n_samples_new, n_features).

        - ``total``:
            Total log-likelihood difference for each sample
            (n_draws, n_samples_new).

        - ``log_odds``
            Log-odds of Group B vs Group A for each sample

        - ``P_B``
            Posterior probability of Group B for each sample

        - ``P_A``
            Posterior probability of Group A for each sample
    """
    # (draws, samples, features)
    posterior: NpFloat = feature_log_likelihood_diff(idata, X_new, X_new_sigma=X_new_sigma)

    total: NpFloat = posterior.sum(axis=-1)

    log_odds: NpFloat = total + np.log(1 - prior_A) - np.log(prior_A)
    P_A, P_B = predict_type_posterior(idata, X_new, X_new_sigma=X_new_sigma, prior_A=prior_A)

    if X_new_group_id is not None:
        y_true = np.asarray(X_new_group_id)  # (samples,)

        # probability assigned to the TRUE class
        # (samples, draws)
        P_correct = np.where(y_true[:, None] == 1, P_B, P_A)

        # per-sample difficulty / reliability
        per_sample_accuracy = P_correct.mean(axis=1)

        # Bayesian predictive accuracy
        overall_accuracy = P_correct.mean()

        classification_check = {
            "y_true": y_true,  # (samples,)
            "P_correct_draw": P_correct,  # (samples, draws)
            "per_sample_accuracy": per_sample_accuracy,  # (samples,)
            "overall_accuracy": overall_accuracy,  # ()
        }
    else:
        classification_check = None

    return {
        "mean": posterior.mean(axis=0),
        "std": posterior.std(axis=0),
        "ci95": np.quantile(posterior, [0.025, 0.975], axis=0),
        "p_support_B": (posterior > 0).mean(axis=0),
        "posterior": posterior,
        # dataset-level evidence (data only)
        "total": total,
        "total_mean": total.mean(axis=0),
        "total_ci95": np.quantile(total, [0.025, 0.975], axis=0),
        # classifier decision (data+prior)
        "log_odds": log_odds,
        "P_A": P_A,
        "P_B": P_B,
        # evaluation
        "classification_check": classification_check,
    }


def global_feature_importance(idata, X_new, *, X_new_sigma=None):
    llr: NpFloat = feature_log_likelihood_diff(idata, X_new, X_new_sigma=X_new_sigma)

    return {
        # On average, does this feature support A or B, and how strongly?
        "signed_mean": llr.mean(axis=(0, 1)),
        # How much does this feature move the decision boundary? (Absolute discriminative strength)
        "abs_mean": np.abs(llr).mean(axis=(0, 1)),
        # Does this feature consistently favor B or A?
        "consistency": (llr > 0).mean(axis=(0, 1)),
        "std": llr.std(axis=(0, 1)),
    }


def plot_explanation_horizontal_error_bar(
    idata: InferenceData,
    X_new: NpFloat,
    *,
    X_new_sigma: NpFloat | None = None,
    X_new_group_id: NpInt | None = None,
    prior_A: float = 0.5,
    sample_idx: int = 0,
) -> tuple[Figure, Axes]:
    """Plots the feature contributions to the classification of a single sample.

    Args:
        idata: Inference data
        X_new: New data (n_samples_new, n_features)
        X_new_sigma: Optional known 1-sigma uncertainties of new data (n_samples_new, n_features).
            Defaults to ``None``.
        X_new_group_id: Group ID of observations (n_samples,). Defaults to ``None``.
        prior_A: Prior probability of Type A. The prior probability of Type B is
            taken as ``1 - prior_A``. Defaults to ``0.5``.
        sample_idx: Index of the sample to plot. Defaults to ``0``.

    Returns:
        tuple:
            - Matplotlib Figure
            - Matplotlib Axes
    """
    explanation: dict[str, Any] = explain_samples(
        idata, X_new, X_new_sigma=X_new_sigma, X_new_group_id=X_new_group_id, prior_A=prior_A
    )

    mean: NpFloat = explanation["mean"][sample_idx]
    ci: NpFloat = explanation["ci95"][:, sample_idx]

    feature_names: list[str] = list(idata["posterior"].coords["feature"].values)

    order: NpArray = np.argsort(np.abs(mean))

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(mean) + 1))

    ax.errorbar(
        mean[order],
        np.arange(len(mean)),
        xerr=[
            mean[order] - ci[0, order],
            ci[1, order] - mean[order],
        ],
        fmt="o",
        capsize=3,
    )

    ax.axvline(0, color="k", ls="--", alpha=0.5)

    total_mean: float = explanation["total_mean"][sample_idx]
    total_ci: NpFloat = explanation["total_ci95"][:, sample_idx]

    if explanation["classification_check"] is not None:
        correct: NpBool = (
            explanation["classification_check"]["per_sample_accuracy"][sample_idx] > 0.5
        )
        color: str = "lightgreen" if correct else "lightcoral"

        per_sample_accuracy: NpFloat = explanation["classification_check"]["per_sample_accuracy"][
            sample_idx
        ]
        ax.set_title(
            f"Sample {sample_idx} (Total LLR = {total_mean:.2f}, "
            f"95% CI = [{total_ci[0]:.2f}, {total_ci[1]:.2f}], "
            f"Per-sample accuracy = {per_sample_accuracy:.2%})"
        )
    else:
        color = "lightgray"
        ax.set_title(
            f"Sample {sample_idx} (Total LLR = {total_mean:.2f}, "
            f"95% CI = [{total_ci[0]:.2f}, {total_ci[1]:.2f}])"
        )

    ax.axvline(total_mean, color=color, lw=2, alpha=0.8, label="Total")
    ax.axvspan(total_ci[0], total_ci[1], color=color, alpha=0.15)

    ax.set_xlabel("Log-likelihood contribution (B-A)")
    ax.set_yticks(np.arange(len(mean)))
    ax.set_yticklabels(np.array(feature_names)[order])

    return fig, ax


def predict_type_posterior(
    idata: InferenceData,
    X_new: NpFloat,
    *,
    X_new_sigma: NpFloat | None = None,
    prior_A: float = 0.5,
) -> tuple[NpFloat, NpFloat]:
    """Computes posterior probabilities that each row in X_new is Type A or B.

    Args:
        idata: Inference data
        X_new: New data (n_samples_new, n_features)
        X_new_sigma: Optional known 1-sigma uncertainties of new data
            (n_samples_new, n_features). Defaults to ``None``.
        prior_A: Prior probability of Type A. The prior probability of Type B is
            taken as ``1 - prior_A``. Defaults to ``0.5``.

    Returns:
        tuple:
            - Posterior probability of Type A (n_samples_new, n_draws)
            - Posterior probability of Type B (n_samples_new, n_draws)
    """
    log_lik_feat: NpFloat = feature_log_likelihood(idata, X_new, X_new_sigma=X_new_sigma)

    # log_lik: (draws, samples, groups)
    log_lik: NpFloat = log_lik_feat.sum(axis=-1)

    # Add priors
    log_lik[:, :, 0] += np.log(prior_A)
    log_lik[:, :, 1] += np.log(1 - prior_A)

    prob: NpFloat = softmax(log_lik, axis=-1)

    # Return: (samples, draws)
    P_A: NpFloat = prob[:, :, 0].T
    P_B: NpFloat = prob[:, :, 1].T

    return P_A, P_B


def plot_confusion_matrix(
    idata: InferenceData,
    X: NpFloat,
    X_group_idx: NpInt,
    *,
    group_names: Iterable | None = None,
    X_sigma: NpFloat | None = None,
) -> Axes:
    """Plots the confusion matrix and logs metrics.

    Note:
        The predicted type is determined using a Bayesian MAP classifier based on the posterior
        mean probabilities.

    Args:
        idata: Inference data containing posterior samples
        X: Observations (n_samples, n_features)
        X_group_idx: Group ID of observations, must be 0 or 1 (n_samples,)
        X_sigma:  Sigma of observations (n_samples, n_features). Defaults to ``None``.

    Returns:
        Axes
    """
    coords: dict[str, list] = get_coords(X, X_group_idx, group_names=group_names)
    P_A, P_B = predict_type_posterior(idata, X, X_new_sigma=X_sigma)

    group1, group2 = coords["group"]

    # Compute posterior mean probability
    mean_prob_A: NpFloat = P_A.mean(axis=1)
    mean_prob_B: NpFloat = P_B.mean(axis=1)
    logger.debug("Posterior probability of %s = %s", group1, mean_prob_A)
    logger.debug("Posterior probability of %s = %s", group2, mean_prob_B)

    # Choose the most probable type Bayesian MAP classifier: standard Naive Bayes rule
    predicted_type: NpFloat = np.where(mean_prob_A > mean_prob_B, group1, group2)
    groups: NpArray = np.array([group1, group2])
    true_labels: NpFloat = groups[X_group_idx]

    # Build confusion matrix
    cm: NpArray = confusion_matrix(true_labels, predicted_type, labels=[group1, group2])
    logger.debug("Confusion matrix = %s", cm)

    # Compute overall accuracy
    accuracy: float = float(accuracy_score(true_labels, predicted_type))

    # Per-class precision, recall, F1
    # Precision - Of all the points predicted of a type, how many were actually the type? Focus
    # is on avoiding false alarms.
    # Recall - Out of all the points that are truly a certain type, what fraction did the model
    # correctly identify? Focus is to avoid misses.
    # Harmonic mean of precision and recall.
    # High F1 -> the model balances correctness (precision) and completeness (recall)
    # Low F1 -> either precision or recall (or both) is low
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_type, labels=[group1, group2], zero_division="warn"
    )

    # Extract values for clarity
    precision_A, precision_B = precision  # pyright: ignore
    recall_A, recall_B = recall  # pyright: ignore
    f1_A, f1_B = f1  # pyright: ignore

    logger.info("Training classification overall accuracy: %0.3f", accuracy)
    logger.info("Training classification precision (%s): %0.3f", group1, precision_A)
    logger.info("Training classification recall (%s): %0.3f", group1, recall_A)
    logger.info("Training classification f1 score (%s): %0.3f", group1, f1_A)
    logger.info("Training classification precision (%s): %0.3f", group2, precision_B)
    logger.info("Training classification recall (%s): %0.3f", group2, recall_B)
    logger.info("Training classification f1 score (%s): %0.3f", group2, f1_B)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[group1, group2])
    disp.plot(cmap="Blues", values_format="d")

    return disp.ax_


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
