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
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm
import seaborn as sns
from arviz import InferenceData
from matplotlib.axes import Axes

from bedroc import debug_logger

logger: logging.Logger = debug_logger()
logger.setLevel(logging.DEBUG)

SUPTITLE_FONTSIZE: int = 14
"""Font size for the super title"""
savefig_opts: dict[str, Any] = {"dpi": 300, "bbox_inches": "tight", "format": "pdf"}
"""Figure options for savefig"""


def hierarchical_difference_model(
    X_A: npt.NDArray,
    X_B: npt.NDArray,
    draws: int = 2000,
    tune: int = 1000,
    target_accept: float = 0.95,
    random_seed: int | None = None,
) -> tuple[pm.Model, InferenceData]:
    """Bayesian hierarchical model to estimate feature-wise mean differences between two groups
    with partial pooling.

    The difference parameters (``delta``) for each feature are drawn from a shared prior with
    global scale ``tau``, which induces shrinkage towards zero for features with weak evidence.
    Each feature has its own noise level (``sigma``), but noise is assumed equivalent across
    groups. Observations are modelled as independent given their feature means and noise.

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
            - model: PyMC model object
            - idata: InferenceData containing posterior samples
    """
    _, n_features = X_A.shape

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

        # Standardised effect size (SMD = Cohen's d-like)
        pm.Deterministic("effect", delta / sigma)

        # Observed data (mutable for predictive use)
        X_A_data = pm.Data("X_A_data", X_A)
        X_B_data = pm.Data("X_B_data", X_B)

        # Likelihoods
        pm.Normal("X_A_obs", mu=mu_A, sigma=sigma, observed=X_A_data)
        pm.Normal("X_B_obs", mu=mu_B, sigma=sigma, observed=X_B_data)

        # Sampling
        idata: InferenceData = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
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
        heteroscedastic: If ``True``, generate independent sigma per type. Defaults to ``False``.
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
            raise ValueError("Data not yet generated. Call 'generate()' first.")

        return self._X_A

    @property
    def X_B(self) -> npt.NDArray:
        """Type B data (n_samples, n_features)"""
        if self._X_B is None:
            raise ValueError("Data not yet generated. Call 'generate()' first.")

        return self._X_B

    @property
    def true_params(self) -> TrueParams:
        """True parameters used in data generation"""
        if self._true_params is None:
            raise ValueError("Data not yet generated. Call 'generate()' first.")

        return self._true_params

    def generate(self) -> None:
        """Generates multivariate data for 2 types (A & B) and stores internally."""
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
        logger.debug("true_params = \n%s", pformat(true_params))

        # Store internally
        self._X_A = X_A
        self._X_B = X_B
        self._true_params = true_params

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
        feature_labels = [f"Feature {i}" for i in range(self.n_features)]

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
                x_idx: int = col  # feature index along x-axis
                y_idx: int = row  # feature index along y-axis in full data
                ax.plot(
                    mu_A[x_idx],
                    mu_A[y_idx],
                    "o",
                    color="blue",
                    markersize=8,
                    markeredgecolor="k",
                    label="_nolegend_",
                )
                ax.plot(
                    mu_B[x_idx],
                    mu_B[y_idx],
                    "o",
                    color="orange",
                    markersize=8,
                    markeredgecolor="k",
                    label="_nolegend_",
                )

        pairgrid.figure.suptitle("Corner Plot: Type A vs Type B", fontsize=SUPTITLE_FONTSIZE)
        sns.move_legend(pairgrid, "upper left", bbox_to_anchor=(0.18, 0.8), frameon=True)

        if savefig:
            pairgrid.savefig(f"{filename_prefix}.{savefig_opts['format']}", **savefig_opts)

        return pairgrid


# class Plotter:
#     """Plotter

#     Args:
#         idata: Trace data from sampling
#     """

#     def __init__(self, idata: InferenceData):
#         self.idata: InferenceData = idata
#         """Trace data from sampling"""

#     def confusion_matrix(self, X_data: npt.NDArray, true_labels: npt.NDArray) -> Figure:
#         """Plots the confusion matrix and logs metrics.

#         Args:
#             trace: InferenceData
#             X_data: Data
#             true_labels: True labels of the data
#         """
#         P_A, P_B = predict_type_posterior(trace, X_data)

#         # Compute posterior mean probability
#         mean_prob_A: npt.NDArray = P_A.mean(axis=1)
#         mean_prob_B: npt.NDArray = P_B.mean(axis=1)
#         logger.debug("Posterior probability of Type A = %s", mean_prob_A)
#         logger.debug("Posterior probability of Type B = %s", mean_prob_B)

#         # Choose the most probable type Bayesian MAP classifier: standard Naive Bayes rule
#         predicted_type: npt.NDArray = np.where(mean_prob_A > mean_prob_B, "A", "B")

#         # Build confusion matrix
#         cm: npt.NDArray = confusion_matrix(true_labels, predicted_type, labels=["A", "B"])
#         logger.debug("Confusion matrix = %s", cm)

#         # Type A metrics
#         accuracy: npt.NDArray = np.mean(predicted_type == true_labels)
#         # Out of all points the model predicted as Type A, what fraction were actually Type A?
#         # Focus is to avoid false alarms (FP)
#         precision_A: npt.NDArray = cm[0, 0] / cm[:, 0].sum()  # TP / (TP + FP)
#         # Out of all the points that are truly Type A, what fraction did the model correctly identify?
#         # Focus is to avoid misses (FN)
#         recall_A: npt.NDArray = cm[0, 0] / cm[0, :].sum()  # TP / (TP + FN)
#         # Harmonic mean of precision and recall.
#         # High F1 -> the model balances correctness (precision) and completeness (recall)
#         # Low F1 -> either precision or recall (or both) is low
#         f1_A: npt.NDArray = 2 * (precision_A * recall_A) / (precision_A + recall_A)

#         # Type B metrics
#         precision_B: npt.NDArray = cm[1, 1] / cm[:, 1].sum()  # TN / (FP + TN)
#         recall_B: npt.NDArray = cm[1, 1] / cm[1, :].sum()  # TP / (TP + FN)
#         f1_B: npt.NDArray = 2 * (precision_B * recall_B) / (precision_B + recall_B)

#         logger.info("Training classification accuracy: %0.3f", accuracy)
#         logger.info("Training classification precision (Type A): %0.3f", precision_A)
#         logger.info("Training classification recall (Type A): %0.3f", recall_A)
#         logger.info("Training classification f1 score (Type A): %0.3f", f1_A)
#         logger.info("Training classification precision (Type B): %0.3f", precision_B)
#         logger.info("Training classification recall (Type B): %0.3f", recall_B)
#         logger.info("Training classification f1 score (Type B): %0.3f", f1_B)

#         disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["A", "B"])
#         disp.plot(cmap="Blues", values_format="d")

#         disp.ax_.set_title("Confusion Matrix: Type A vs Type B")

#         disp.figure_.savefig(f"confusion_matrix.{savefig_opts['format']}", **savefig_opts)

#         return disp.figure_

#     def posterior_differences(self, hdi_prob: float = 0.94) -> Figure:
#         """Plots posterior distributions of the difference vector (delta) in a forest-style plot.

#         Args:
#             hdi_prob: Credible interval probability. Defaults to ``0.94``.

#         Returns:
#             Figure
#         """
#         # Extract delta variable names
#         n_features = self.idata["posterior"]["delta"].shape[-1]

#         # Forest plot
#         axes: tuple[Axes] = az.plot_forest(
#             self.idata,
#             var_names=["tau", "delta"],
#             combined=True,
#             hdi_prob=hdi_prob,
#             kind="forestplot",
#             # r_hat=True,
#         )

#         axes[0].axvline(0, linestyle="--", linewidth=1, alpha=0.6)

#         # Replace default tick labels with feature_labels
#         yticklabels: list[str] = ["Tau"] + [f"Feature {i}" for i in range(n_features)]
#         yticklabels.reverse()
#         axes[0].set_yticklabels(yticklabels)
#         axes[0].set_title(
#             "Posterior Differences (Type B - Type A)",
#             fontdict={"fontsize": SUPTITLE_FONTSIZE},
#         )

#         figure: Figure = cast(Figure, axes[0].figure)
#         figure.savefig(f"posterior_differences.{savefig_opts['format']}", **savefig_opts)

#         return figure

#     def posterior_effect(self, hdi_prob: float = 0.94) -> Figure:
#         """Plots posterior distributions of the effect size per feature in a forest-style plot.

#         Args:
#             hdi_prob: Credible interval probability. Defaults to ``0.94``.

#         Returns:
#             Figure
#         """
#         # Extract delta variable names
#         n_features = self.idata["posterior"]["effect"].shape[-1]

#         # Forest plot
#         axes: tuple[Axes] = az.plot_forest(
#             self.idata,
#             var_names=["effect"],
#             combined=True,
#             hdi_prob=hdi_prob,
#             kind="forestplot",
#             # r_hat=True,
#         )

#         axes[0].axvline(0, linestyle="--", linewidth=1, alpha=0.6)

#         # Replace default tick labels with feature_labels
#         yticklabels: list[str] = [f"Feature {i}" for i in range(n_features)]
#         yticklabels.reverse()
#         axes[0].set_yticklabels(yticklabels)
#         axes[0].set_title("Effect size", fontdict={"fontsize": SUPTITLE_FONTSIZE})

#         figure: Figure = cast(Figure, axes[0].figure)
#         figure.savefig(f"posterior_effect_size.{savefig_opts['format']}", **savefig_opts)

#         return figure
