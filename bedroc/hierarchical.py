#
# Copyright 2025 Dan J. Bower
#
# This file is part of Bedroc.
#
# Bedroc is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Bedroc is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Bedroc. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Utilities for building and working with Bayesian hierarchical models.

This module provides reusable components for specifying and fitting hierarchical models.
Hierarchical (multi-level) models allow parameters to vary across groups, while sharing information
through structured priors. This partial pooling leads to more stable estimates and reduces
overfitting, especially when data are sparse or imbalanced across groups.

In addition to model specification and posterior inference, this module also implements supervised
classification derived from the hierarchical model. The classifier is not a
stand-alone machine-learning model; instead, it uses the fitted Bayesian generative model to
compute posterior class probabilities and apply a Bayesian MAP decision rule.

This design keeps the focus on Bayesian hierarchical inference while still supporting
posterior-based prediction, diagnostic visualisations (corner plots, forest plots), and
evaluation tools such as confusion matrices. Classification is therefore a downstream application
of the hierarchical model rather than its primary purpose.

Quick Reference Glossary:
    - Partial Pooling: Parameters vary by group but share information through a common prior,
      stabilizing estimates.
    - Shrinkage: Pulling parameter estimates toward a central value (e.g., zero) when data are weak
      or noisy.
    - Hyperparameter: A parameter of a prior controlling variability or central tendency of
      lower-level parameters
    - Hierarchical / Multi-level Model: Parameters structured at multiple levels (e.g., group and
      observation levels) to share information.
    - Feature-wise noise: Standard deviation of observations per feature; shared across groups
    - Standardized Effect Size (SMD): Dimensionless measure of group difference normalized by
      variability.
    - Random Seed: Fixes sampler randomness to enable reproducible posterior draws.
"""

import logging
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any, Optional, cast

import arviz as az
import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm
import seaborn as sns
from arviz import InferenceData
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

logger: logging.Logger = logging.getLogger(__name__)

SUPTITLE_FONTSIZE: str = "xx-large"
"""Font size for the super title"""
savefig_opts: dict[str, Any] = {"dpi": 300, "bbox_inches": "tight", "format": "pdf"}
"""Figure options for savefig"""


def hierarchical_difference_model(
    X_A: npt.NDArray,
    X_B: npt.NDArray,
    draws: int = 2000,
    tune: int = 1000,
    target_accept: float = 0.95,
    random_seed: Optional[int] = None,
) -> tuple[pm.Model, InferenceData]:
    """Bayesian hierarchical model to estimate feature-wise mean differences between two groups
    with partial pooling.

    The difference parameters (``delta``) for each feature are drawn from a shared prior with
    global scale ``tau``, which induces shrinkage towards zero for features with weak evidence.
    Each feature has its own noise level (``sigma``), but noise is assumed equivalent across
    groups. Observations are modelled as independent given their feature means and noise.

    Note:
        The variable names in the model are fixed: 'mu_A', 'mu_B', 'delta', 'sigma', 'effect'.
        These names are propagated downstream and expected by helper functions and
        analysis/plotting utilities.

    Args:
        X_A: Observations from group A (n_samples, n_features)
        X_B: Observations from group B (n_samples, n_features)
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

    nA, n_features = X_A.shape
    nB, _ = X_B.shape

    # Stack observations once, outside the model
    Y: npt.NDArray = np.vstack([X_A, X_B])  # shape: (nA + nB, n_features)

    # Boolean mask to distinguish groups
    group_idx: npt.NDArray = np.concatenate(
        [
            np.zeros(nA, dtype=int),  # group A
            np.ones(nB, dtype=int),  # group B
        ]
    )  # shape (nA+nB,)

    with pm.Model() as model:
        # Group A feature means (no pooling across features)
        mu_A = pm.Normal("mu_A", mu=0, sigma=10, shape=n_features)

        # Global scale controlling how much deltas vary across features
        tau = pm.HalfNormal("tau", sigma=5)

        # Feature-wise mean differences (hierarchical / partial pooling)
        delta = pm.Normal("delta", mu=0, sigma=tau, shape=n_features)

        # Group B feature means derive from A + delta
        mu_B = pm.Deterministic("mu_B", mu_A + delta)

        # Feature-specific observation noise, shared across groups
        sigma = pm.HalfNormal("sigma", sigma=5, shape=n_features)

        # NOTE: Assumes uncorrelated noise across features
        pooled_sigma = pm.math.sqrt(pm.math.mean(sigma**2))  # pyright: ignore (attr. is available)

        # Standardized effect size (SMD = Cohen's d-like)
        pm.Deterministic("effect_tau", tau / pooled_sigma)
        pm.Deterministic("effect", delta / sigma)

        # Build mu_obs with broadcasting
        mu_obs = pm.math.stack([mu_A, mu_B], axis=0)[group_idx]  # pyright: ignore (attr. is available)

        # Likelihood
        pm.Normal("X_obs", mu=mu_obs, sigma=sigma, observed=Y)

        # Sampling
        idata: InferenceData = pm.sample(
            draws=draws, tune=tune, target_accept=target_accept, random_seed=random_seed
        )

    return model, idata


def zero_difference_model(
    X_A: npt.NDArray,
    X_B: npt.NDArray,
    draws: int = 2000,
    tune: int = 1000,
    target_accept: float = 0.95,
    random_seed: Optional[int] = None,
) -> tuple[pm.Model, InferenceData]:
    """Model assuming no difference between two groups.

    This model is a "null" version of the hierarchical difference model: it assumes that the
    feature-wise means of Group B are identical to those of Group A (i.e., delta = 0). Each feature
    has its own observation noise, shared across groups. Observations are modelled as independent
    given their feature means and noise.

    Note:
        The variable names in the model are fixed: 'mu_A', 'mu_B', 'sigma'. These names are
        expected by downstream analysis/plotting utilities.

    Args:
        X_A: Observations from group A (n_samples, n_features)
        X_B: Observations from group B (n_samples, n_features)
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
    nA, n_features = X_A.shape
    nB, _ = X_B.shape

    # Stack observations once, outside the model
    Y: npt.NDArray = np.vstack([X_A, X_B])  # shape: (nA + nB, n_features)

    # Boolean mask to distinguish groups
    group_idx: npt.NDArray = np.concatenate(
        [
            np.zeros(nA, dtype=int),  # group A
            np.ones(nB, dtype=int),  # group B
        ]
    )  # shape (nA+nB,)

    with pm.Model() as model:
        # Group A feature means (no pooling across features)
        mu_A = pm.Normal("mu_A", mu=0, sigma=10, shape=n_features)

        # Group B feature means fixed equal to mu_A (delta = 0)
        mu_B = pm.Deterministic("mu_B", mu_A)  # No difference between groups

        # Feature-specific observation noise, shared across groups
        sigma = pm.HalfNormal("sigma", sigma=5, shape=n_features)

        # Build mu_obs with broadcasting
        mu_obs = pm.math.stack([mu_A, mu_B], axis=0)[group_idx]  # pyright: ignore (attr. is available)

        # Likelihood
        pm.Normal("X_obs", mu=mu_obs, sigma=sigma, observed=Y)

        # Sampling
        idata: InferenceData = pm.sample(
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

    mu_A: npt.NDArray
    mu_B: npt.NDArray
    difference_vector: npt.NDArray
    sigma_A: npt.NDArray
    sigma_B: npt.NDArray


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
    random_seed: Optional[int] = None
    heteroscedastic: bool = False
    # Internal storage for generated data
    _X_A: Optional[npt.NDArray] = field(init=False, default=None)
    _X_B: Optional[npt.NDArray] = field(init=False, default=None)
    _true_params: Optional[TrueParams] = field(init=False, default=None)

    @property
    def X_A(self) -> npt.NDArray:
        """Type A data (n_samples, n_features)"""
        if self._X_A is None:
            raise ValueError(
                "Data not yet generated. Call 'generate()' first."
            )  # pragma: no cover

        return self._X_A

    @property
    def X_B(self) -> npt.NDArray:
        """Type B data (n_samples, n_features)"""
        if self._X_B is None:
            raise ValueError(
                "Data not yet generated. Call 'generate()' first."
            )  # pragma: no cover

        return self._X_B

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
        mu_A: npt.NDArray = rng.normal(
            loc=0.0, scale=self.type_a_std_of_mean, size=self.n_features
        )
        logger.debug("mu_A = %s", mu_A)

        # For Type B, each feature mean gets a random shift relative to Type A.
        # Scaling by difference_scale controls overall separation between types.
        raw_shift: npt.NDArray = rng.normal(
            loc=0.0, scale=self.type_b_std_of_mean, size=self.n_features
        )
        mu_B: npt.NDArray = mu_A + self.difference_scale * raw_shift
        logger.debug("mu_B = %s", mu_B)

        # Noise (standard deviation) per feature
        if self.heteroscedastic:
            # Noise varies across types as well as features
            sigma_A: npt.NDArray = rng.uniform(
                self.sigma_min, self.sigma_max, size=self.n_features
            )
            sigma_B: npt.NDArray = rng.uniform(
                self.sigma_min, self.sigma_max, size=self.n_features
            )
            logger.debug("sigma_A = %s", sigma_A)
            logger.debug("sigma_B = %s", sigma_B)
        else:
            # Noise only varies across features, not types
            sigma: npt.NDArray = rng.uniform(self.sigma_min, self.sigma_max, size=self.n_features)
            sigma_A = sigma_B = sigma
            logger.debug("sigma (shared) = %s", sigma)

        # Generate samples
        X_A: npt.NDArray = rng.normal(mu_A, sigma_A, size=(self.n_samples, self.n_features))
        logger.debug("X_A = %s", X_A)
        X_B: npt.NDArray = rng.normal(mu_B, sigma_B, size=(self.n_samples, self.n_features))
        logger.debug("X_B = %s", X_B)

        true_params: TrueParams = TrueParams(
            mu_A=mu_A, mu_B=mu_B, difference_vector=mu_B - mu_A, sigma_A=sigma_A, sigma_B=sigma_B
        )

        # Store internally
        self._X_A = X_A
        self._X_B = X_B
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

        mu_A: npt.NDArray = self.true_params.mu_A
        mu_B: npt.NDArray = self.true_params.mu_B
        sigma_A: npt.NDArray = self.true_params.sigma_A
        sigma_B: npt.NDArray = self.true_params.sigma_B

        # Draw new samples from the same ground-truth distribution
        X_A_test: npt.NDArray = rng.normal(mu_A, sigma_A, size=(n_samples, self.n_features))
        X_B_test: npt.NDArray = rng.normal(mu_B, sigma_B, size=(n_samples, self.n_features))

        return X_A_test, X_B_test

    def plot(
        self, savefig: bool = False, filename_prefix: Path | str = "synthetic_data_corner_plot"
    ) -> sns.PairGrid:
        """Plots a corner plot for comparing Type A vs Type B with overlay of true inputs.

        Args:
            savefig: Saves the figure to a file. Defaults to ``False``.
            filename_prefix: Prefix for the saved figure filename. Defaults to
                "synthetic_data_corner_plot".

        Returns:
            Pairgrid
        """
        feature_labels: pd.Series = pd.Series([f"Feature {i}" for i in range(self.n_features)])

        # Build DataFrame for seaborn
        df_A: pd.DataFrame = pd.DataFrame(self.X_A, columns=feature_labels)
        df_A["Type"] = "A"
        df_B: pd.DataFrame = pd.DataFrame(self.X_B, columns=feature_labels)
        df_B["Type"] = "B"
        df: pd.DataFrame = pd.concat([df_A, df_B], ignore_index=True)

        # Create corner plot
        pairgrid: sns.PairGrid = sns.pairplot(
            df, hue="Type", corner=True, plot_kws=dict(alpha=0.4, s=20), diag_kws=dict(alpha=0.6)
        )

        # Overlay true means and 1 sigma bands on diagonal
        mu_A: npt.NDArray = self.true_params.mu_A
        mu_B: npt.NDArray = self.true_params.mu_B
        sigma_A: npt.NDArray = self.true_params.sigma_A
        sigma_B: npt.NDArray = self.true_params.sigma_B

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

        pairgrid.figure.suptitle("Corner Plot: Type A vs Type B", fontsize=SUPTITLE_FONTSIZE)
        sns.move_legend(pairgrid, "upper left", bbox_to_anchor=(0.18, 0.8), frameon=True)

        if savefig:
            pairgrid.savefig(
                f"{filename_prefix}.{savefig_opts['format']}", **savefig_opts
            )  # pragma: no cover

        return pairgrid


class Analyzer:
    """Analyzer for the hierarchical difference model

    Note:
        This Analyzer expects the following variable names in the model: 'mu_A', 'mu_B', 'delta',
        'sigma', 'effect'.  These names are produced by the hierarchical difference model and are
        required by the analysis and plotting utilities.

    Args:
        model: PyMC model object
        idata: Trace data from sampling
    """

    def __init__(self, model: pm.Model, idata: InferenceData):
        self.model: pm.Model = model
        self.idata: InferenceData = idata

    @property
    def n_features(self) -> int:
        """Number of features in the model"""
        return self.idata["posterior"]["delta"].shape[-1]

    def plot_confusion_matrix(
        self,
        X_data: npt.NDArray,
        true_labels: npt.NDArray,
        savefig: bool = False,
        filename_prefix: Path | str = "confusion_matrix",
    ) -> Figure:
        """Plots the confusion matrix and logs metrics.

        Note:
            The predicted type is determined using a Bayesian MAP classifier based on the posterior
            mean probabilities.

        Args:
            X_data: Data
            true_labels: True labels of the data
            savefig: Saves the figure to a file. Defaults to ``False``.
            filename_prefix: Prefix for the saved figure filename. Defaults to
                "confusion_matrix".
        """
        P_A, P_B = self.predict_type_posterior(X_data)

        # Compute posterior mean probability
        mean_prob_A: npt.NDArray = P_A.mean(axis=1)
        mean_prob_B: npt.NDArray = P_B.mean(axis=1)
        logger.debug("Posterior probability of Type A = %s", mean_prob_A)
        logger.debug("Posterior probability of Type B = %s", mean_prob_B)

        # Choose the most probable type Bayesian MAP classifier: standard Naive Bayes rule
        predicted_type: npt.NDArray = np.where(mean_prob_A > mean_prob_B, "A", "B")

        # Build confusion matrix
        cm: npt.NDArray = confusion_matrix(true_labels, predicted_type, labels=["A", "B"])
        logger.debug("Confusion matrix = %s", cm)

        # Type A metrics
        accuracy: npt.NDArray = np.mean(predicted_type == true_labels)
        # Out of all points the model predicted as Type A, what fraction were actually Type A?
        # Focus is to avoid false alarms (FP)
        precision_A: npt.NDArray = cm[0, 0] / cm[:, 0].sum()  # TP / (TP + FP)
        # Out of all the points that are truly Type A, what fraction did the model correctly
        # identify? Focus is to avoid misses (FN)
        recall_A: npt.NDArray = cm[0, 0] / cm[0, :].sum()  # TP / (TP + FN)
        # Harmonic mean of precision and recall.
        # High F1 -> the model balances correctness (precision) and completeness (recall)
        # Low F1 -> either precision or recall (or both) is low
        f1_A: npt.NDArray = 2 * (precision_A * recall_A) / (precision_A + recall_A)

        # Type B metrics
        precision_B: npt.NDArray = cm[1, 1] / cm[:, 1].sum()  # TN / (FP + TN)
        recall_B: npt.NDArray = cm[1, 1] / cm[1, :].sum()  # TP / (TP + FN)
        f1_B: npt.NDArray = 2 * (precision_B * recall_B) / (precision_B + recall_B)

        logger.info("Training classification accuracy: %0.3f", accuracy)
        logger.info("Training classification precision (Type A): %0.3f", precision_A)
        logger.info("Training classification recall (Type A): %0.3f", recall_A)
        logger.info("Training classification f1 score (Type A): %0.3f", f1_A)
        logger.info("Training classification precision (Type B): %0.3f", precision_B)
        logger.info("Training classification recall (Type B): %0.3f", recall_B)
        logger.info("Training classification f1 score (Type B): %0.3f", f1_B)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["A", "B"])
        disp.plot(cmap="Blues", values_format="d")

        disp.ax_.set_title("Confusion Matrix: Type A vs Type B")

        if savefig:  # pragma: no cover
            disp.figure_.savefig(f"{filename_prefix}.{savefig_opts['format']}", **savefig_opts)

        return disp.figure_

    def plot_posterior(self, savefig: bool = False, **kwargs) -> Figure:
        """Plots posterior densities in the style of John K. Kruschke's book.

        Args:
            savefig: Saves the figure to a file. Defaults to ``False``.
            **kwargs: Keyword arguments for :func:`arviz.plot_posterior`

        Returns:
            Figure
        """

        # Corner plot of key parameters
        axes = az.plot_posterior(self.idata, **kwargs)

        # Get the Figure safely
        if isinstance(axes, np.ndarray):
            figure: Figure = axes.flatten()[0].figure
        else:
            figure = axes.figure

        figure.suptitle("Posterior Distributions", fontsize=SUPTITLE_FONTSIZE)

        # Automatically adjust spacing for suptitle
        figure.tight_layout(rect=(0, 0, 1, 0.98))

        if savefig:  # pragma: no cover
            var_names = kwargs.get("var_names", None)
            if var_names is not None:
                var_names_str: str = "_".join(var_names)
            else:
                var_names_str = "all_variables"
            figure.savefig(f"posterior_{var_names_str}.{savefig_opts['format']}", **savefig_opts)

        return figure

    def plot_posterior_differences(
        self,
        hdi_prob: float = 0.94,
        savefig: bool = False,
        filename_prefix: Path | str = "posterior_differences",
    ) -> Figure:
        """Plots posterior distributions of the difference vector (delta) in a forest-style plot.

        Args:
            hdi_prob: Credible interval probability. Defaults to ``0.94``.
            savefig: Saves the figure to a file. Defaults to ``False``.
            filename_prefix: Prefix for the saved figure filename. Defaults to
                "posterior_differences".

        Returns:
            Figure
        """

        axes: tuple[Axes] = az.plot_forest(
            self.idata,
            var_names=["tau", "delta"],
            combined=True,
            hdi_prob=hdi_prob,
            kind="forestplot",
            # r_hat=True,
        )

        axes[0].axvline(0, linestyle="--", linewidth=1, alpha=0.6)

        # Replace default tick labels with feature_labels
        yticklabels: list[str] = ["Tau"] + [f"Feature {i}" for i in range(self.n_features)]
        yticklabels.reverse()
        axes[0].set_yticklabels(yticklabels)
        axes[0].set_title("Posterior Differences (B-A)", fontdict={"fontsize": SUPTITLE_FONTSIZE})

        figure: Figure = cast(Figure, axes[0].figure)

        if savefig:  # pragma: no cover
            figure.savefig(f"{filename_prefix}.{savefig_opts['format']}", **savefig_opts)

        return figure

    def plot_posterior_effect_size(
        self,
        hdi_prob: float = 0.94,
        savefig: bool = False,
        filename_prefix: Path | str = "posterior_effect_sizes",
    ) -> Figure:
        """Plots posterior distributions of the effect size per feature in a forest-style plot.

        Args:
            hdi_prob: Credible interval probability. Defaults to ``0.94``.
            savefig: Saves the figure to a file. Defaults to ``False``.
            filename_prefix: Prefix for the saved figure filename. Defaults to
                "posterior_effect_sizes".

        Returns:
            Figure
        """

        axes: tuple[Axes] = az.plot_forest(
            self.idata,
            var_names=["effect_tau", "effect"],
            combined=True,
            hdi_prob=hdi_prob,
            kind="forestplot",
            # r_hat=True,
        )

        axes[0].axvline(0, linestyle="--", linewidth=1, alpha=0.6)

        # Replace default tick labels with feature_labels
        yticklabels: list[str] = ["Tau"] + [f"Feature {i}" for i in range(self.n_features)]
        yticklabels.reverse()
        axes[0].set_yticklabels(yticklabels)
        axes[0].set_title("Posterior Effect Sizes (B-A)", fontdict={"fontsize": SUPTITLE_FONTSIZE})

        figure: Figure = cast(Figure, axes[0].figure)

        if savefig:  # pragma: no cover
            figure.savefig(f"{filename_prefix}.{savefig_opts['format']}", **savefig_opts)

        return figure

    def plot_posterior_predictive(
        self,
        thinning_factor: int = 5,
        savefig: bool = False,
        filename_prefix: Path | str = "posterior_predictive_check",
    ) -> Figure:
        """Plots posterior predictive check (in-sample predictions).

        This performs in-sample predictions to assess how well the model fits the observed data,
        i.e., test how well the model can reproduce the data it was trained on.

        Args:
            thinning_factor: Thinning factor for posterior samples to reduce overplotting.
                Defaults to ``5``.
            savefig: Saves the figure to a file. Defaults to ``False``.
            filename_prefix: Prefix for the saved figure filename. Defaults to
                "posterior_predictive_check".

        Returns:
            Figure
        """
        thinned_idata: InferenceData = cast(
            InferenceData, self.idata.sel(draw=slice(None, None, thinning_factor))
        )
        posterior_predictive: InferenceData = pm.sample_posterior_predictive(
            thinned_idata, model=self.model
        )

        axes = az.plot_ppc(posterior_predictive, group="posterior", observed=True)

        # Get the Figure safely
        if isinstance(axes, np.ndarray):
            figure: Figure = axes.flatten()[0].figure
        else:
            figure = axes.figure

        figure.suptitle("Posterior Predictive Check", fontsize=SUPTITLE_FONTSIZE)

        if savefig:  # pragma: no cover
            figure.savefig(f"{filename_prefix}.{savefig_opts['format']}", **savefig_opts)

        return figure

    def plot_prior_predictive(
        self,
        savefig: bool = False,
        filename_prefix: Path | str = "prior_predictive_check",
        **kwargs,
    ) -> Figure:
        """Plots prior predictive check.

        This plot is used to determine if the model can generate data plausibly shaped like the
        observed distributions.

        Args:
            savefig: Saves the figure to a file. Defaults to ``False``.
            filename_prefix: Prefix for the saved figure filename. Defaults to
                "prior_predictive_check".
            kwargs: Keyword arguments for :func:`pymc.sample_prior_predictive`

        Returns:
            Figure
        """
        prior_predictive: InferenceData = pm.sample_prior_predictive(model=self.model, **kwargs)

        axes = az.plot_ppc(prior_predictive, group="prior", observed=True)

        # Get the Figure safely
        if isinstance(axes, np.ndarray):
            figure: Figure = axes.flatten()[0].figure
        else:
            figure = axes.figure

        figure.suptitle("Prior Predictive Check", fontsize=SUPTITLE_FONTSIZE)

        if savefig:  # pragma: no cover
            figure.savefig(f"{filename_prefix}.{savefig_opts['format']}", **savefig_opts)

        return figure

    def predict_type_posterior(self, X_new: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """Computes posterior probabilities that each row in X_new is Type A or B.

        Args:
            X_new: New data (n_samples_new, n_features_new)

        Returns:
            tuple:
                - Posterior probability of Type A (n_samples_new, n_draws)
                - Posterior probability of Type B (n_samples_new, n_draws)
        """
        # Extract posterior samples
        # (samples, features)
        mu_A_samples = self.idata["posterior"]["mu_A"].stack(draws=("chain", "draw")).values.T
        mu_B_samples = self.idata["posterior"]["mu_B"].stack(draws=("chain", "draw")).values.T
        sigma_samples = self.idata["posterior"]["sigma"].stack(draws=("chain", "draw")).values.T
        logger.debug("mu_A_samples.shape = %s", mu_A_samples.shape)
        logger.debug("mu_B_samples.shape = %s", mu_B_samples.shape)
        logger.debug("sigma_samples.shape = %s", sigma_samples.shape)

        # Reshape for broadcasting: (draws, samples, features)
        mu_A_b = mu_A_samples[:, None, :]  # (n_draws, 1, n_features)
        mu_B_b = mu_B_samples[:, None, :]
        sigma_b = sigma_samples[:, None, :]
        X_b = X_new[None, :, :]  # (1, n_samples, n_features)

        # Compute log-likelihoods per feature and sum across features
        log_lik_A = -0.5 * np.sum(
            (np.square((X_b - mu_A_b) / sigma_b)) + np.log(2 * np.pi * np.square(sigma_b)), axis=-1
        )
        log_lik_B = -0.5 * np.sum(
            (np.square((X_b - mu_B_b) / sigma_b)) + np.log(2 * np.pi * np.square(sigma_b)), axis=-1
        )
        # Shapes: (n_draws, n_samples)

        # Numerically stable posterior probability
        max_log = np.maximum(log_lik_A, log_lik_B)
        lik_A = np.exp(log_lik_A - max_log)
        lik_B = np.exp(log_lik_B - max_log)

        P_A = lik_A / (lik_A + lik_B)  # (n_draws, n_samples)
        P_B = lik_B / (lik_A + lik_B)

        # Transpose to match original API: (n_samples, n_draws)
        return P_A.T, P_B.T
