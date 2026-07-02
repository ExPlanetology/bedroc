# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bayesian hierarchical model for group-centric comparison of two groups"""

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.special import softmax
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from bedroc.hierarchical import get_coords
from bedroc.type_aliases import NpArray, NpFloat, NpInt

logger: logging.Logger = logging.getLogger(__name__)

RANDOM_SEED: int | None = 123
"""Random seed for reproducibility. Set to ``None`` for random behavior."""
SAVEFIG_KWARGS: dict[str, Any] = {"dpi": 300, "bbox_inches": "tight", "format": "pdf"}
"""Default savefig options"""
LOW_PERCENTILE: float = 2.5
"""Low percentile for credible intervals"""
HIGH_PERCENTILE: float = 97.5
"""High percentile for credible intervals"""


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
        sample_names: Sample names. Defaults to sequential sample names.
        feature_names: Feature names. Defaults to sequential feature names.
        group_names: Group names. Defaults to unique values in ``X_group_idx``.
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
        logger.info("Creating an Hierarchical Group Model for %s", name)
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

    def _save_figure(
        self, figure, stem: str, savefig_kwargs: dict[str, Any] | None = None
    ) -> None:
        """Private helper function to save a figure with consistent formatting and naming

        Args:
            figure: Figure object to save.
            stem: Stem of the filename (without extension).
            savefig_kwargs: Keyword arguments for :func:`matplotlib.pyplot.savefig`. Defaults to
                :obj:`SAVEFIG_KWARGS`.
        """
        if self.output_directory is None:
            logger.warning("Output directory is None. Figure will not be saved.")
            return

        kwargs: dict[str, Any] = SAVEFIG_KWARGS.copy()
        if savefig_kwargs:
            kwargs.update(savefig_kwargs)

        # Defaults to pdf if no format is specified in kwargs
        fmt: str = kwargs.get("format", "pdf")

        filename: Path = self.output_directory / Path(f"{self.name}_{stem}.{fmt}")

        figure.savefig(filename, **kwargs)
        logger.info("Figure saved to %s", filename)

    def run_inference(
        self,
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
            sigma = pm.HalfNormal("sigma", sigma=1.0, dims="feature")

            if self.X_sigma is not None:
                # The actual likelihood noise for each observation
                sigma_obs = pm.math.sqrt(self.X_sigma**2 + sigma**2)  # pyright: ignore
                sigma_total_feature = pm.math.sqrt(
                    pm.math.mean(self.X_sigma**2, axis=0) + sigma**2
                )
            else:
                sigma_obs = sigma
                sigma_total_feature = sigma

            mu_obs = mu[self.X_group_idx, ...]  # pyright: ignore

            pm.Deterministic("effect_size", delta / sigma_total_feature, dims="feature")

            # Likelihood
            pm.Normal(
                "observed_data",
                mu=mu_obs,
                sigma=sigma_obs,
                observed=self.X,
                dims=("obs", "feature"),
            )

            # Sampling and store objects for later access
            self._idata = pm.sample(
                draws=draws, tune=tune, target_accept=target_accept, random_seed=random_seed
            )
            self._model = model

    def plot_dist(
        self,
        figsize: tuple = (12, 6),
        col_wrap: int = 4,
        *,
        savefig_kwargs: dict[str, Any] | None = None,
    ) -> az.PlotCollection:
        """Plots posterior distributions of model parameters.

        Args:
            figsize: Figure size. Defaults to ``(12, 6)``.
            col_wrap: Number of columns to wrap the plots. Defaults to ``4``.
            savefig_kwargs: Keyword arguments for :func:`matplotlib.pyplot.savefig`. Defaults to
                ``None`` to use :obj:`SAVEFIG_KWARGS`.

        Returns:
            Plot collection
        """
        pc_kwargs: dict = {"figure_kwargs": {"figsize": figsize}}
        pc: az.PlotCollection = az.plot_dist(
            self.idata, var_names=["mu"], col_wrap=col_wrap, **pc_kwargs
        )
        pc.get_viz("figure").tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.0)
        pc.add_title("Posterior Distributions", fontsize="xx-large")

        self._save_figure(pc, "posterior_distributions", savefig_kwargs=savefig_kwargs)

        return pc

    def plot_posterior_predictive(
        self,
        sample_kwargs: dict[str, Any] | None = None,
        *,
        savefig_kwargs: dict[str, Any] | None = None,
    ) -> az.PlotCollection:
        """Plots posterior predictive check (in-sample predictions).

        This performs in-sample predictions to assess how well the model fits the observed data,
        i.e., test how well the model can reproduce the data it was trained on.

        Args:
            sample_kwargs: Keyword arguments for :func:`pymc.sample_posterior_predictive`. Defaults
                to ``None``.
            savefig_kwargs: Keyword arguments for :func:`matplotlib.pyplot.savefig`. Defaults to
                ``None`` to use :obj:`SAVEFIG_KWARGS`.

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
            kind="kde",
            visuals={"observed_dist": {"color": "black"}},
        )

        self._save_figure(pc, "posterior_predictive", savefig_kwargs=savefig_kwargs)

        return pc

    def plot_prior_predictive(
        self,
        sample_kwargs: dict[str, Any] | None = None,
        *,
        savefig_kwargs: dict[str, Any] | None = None,
    ) -> az.PlotCollection:
        """Plots prior predictive check.

        This plot is used to determine if the model can generate data plausibly shaped like the
        observed distributions.

        Args:
            sample_kwargs: Keyword arguments for :func:`pymc.sample_prior_predictive`. Defaults to
                ``None``.
            savefig_kwargs: Keyword arguments for :func:`matplotlib.pyplot.savefig`. Defaults to
                ``None`` to use :obj:`SAVEFIG_KWARGS`.

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

        self._save_figure(pc, "prior_predictive", savefig_kwargs=savefig_kwargs)

        return pc

    def plot_forest(
        self, figsize: tuple = (10, 15), *, savefig_kwargs: dict[str, Any] | None = None
    ) -> az.PlotCollection:
        """Plots forest plot of posterior distributions.

        Args:
            figsize: Figure size. Defaults to ``(10, 15)``.
            savefig_kwargs: Keyword arguments for :func:`matplotlib.pyplot.savefig`. Defaults to
                ``None`` to use :obj:`SAVEFIG_KWARGS`.

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

        self._save_figure(pc, "posterior_forest", savefig_kwargs=savefig_kwargs)

        return pc

    def plot_forest_effect_size(
        self, figsize: tuple = (10, 6), *, savefig_kwargs: dict[str, Any] | None = None
    ) -> az.PlotCollection:
        """Forest plot of posterior effect sizes with interpretation bands.

        Args:
            figsize: Figure size. Defaults to ``(10, 6)``.
            savefig_kwargs: Keyword arguments for :func:`matplotlib.pyplot.savefig`. Defaults to
                ``None`` to use :obj:`SAVEFIG_KWARGS`.

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

        self._save_figure(pc, "posterior_effect_sizes", savefig_kwargs=savefig_kwargs)

        return pc

    def plot_confusion_matrix(
        self,
        X_test: NpFloat,
        X_test_group_idx: NpInt,
        *,
        X_test_sigma: NpFloat | None = None,
        savefig_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure, Axes]:
        """Plots the confusion matrix and logs metrics.

        Note:
            The predicted type is determined using a Bayesian MAP classifier based on the posterior
            mean probabilities.

        Args:
            X_test: Observations (n_samples, n_features)
            X_test_group_idx: Group ID of observations, must be 0 or 1 (n_samples,)
            X_test_sigma: Sigma of observations (n_samples, n_features). Defaults to ``None``.
            savefig_kwargs: Keyword arguments for :func:`matplotlib.pyplot.savefig`. Defaults to
                ``None`` to use :obj:`SAVEFIG_KWARGS`.

        Returns:
            Figure, Axes
        """
        P_A, P_B = self.predict_type_posterior(X_test, X_sigma=X_test_sigma)

        group1, group2 = self.coords["group"]

        # Compute posterior mean probability
        mean_prob_A: NpFloat = P_A.mean(axis=0)
        mean_prob_B: NpFloat = P_B.mean(axis=0)
        logger.debug("Posterior probability of %s = %s", group1, mean_prob_A)
        logger.debug("Posterior probability of %s = %s", group2, mean_prob_B)

        # Choose the most probable type Bayesian MAP classifier: standard Naive Bayes rule
        predicted_type: NpFloat = np.where(mean_prob_A > mean_prob_B, group1, group2)
        groups: NpArray = np.array([group1, group2])
        true_labels: NpFloat = groups[X_test_group_idx]

        # Build confusion matrix
        cm: NpArray = confusion_matrix(true_labels, predicted_type, labels=[group1, group2])
        logger.debug("Confusion matrix = %s", cm)

        accuracy: float = float(accuracy_score(true_labels, predicted_type))
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

        # Dump interpretation notes to log for user reference
        notes: str = (
            " - TP = True Positives, FP = False Positives, FN = False Negatives\n"
            " - Overall accuracy: Fraction of all samples correctly classified.\n"
            "     accuracy = (TP + TN) / (TP + TN + FP + FN)\n"
            " - Precision: When I identify a type, how often am I correct?\n"
            "     Focus is on avoiding false alarms.\n"
            "     precision = TP / (TP + FP) \n"
            " - Recall: Of all the samples of a certain type, how many did I identify?\n"
            "     Focus is to avoid misses.\n"
            "     recall = TP / (TP + FN) \n"
            " - F1 Score: Harmonic mean of precision and recall.\n"
            "     High F1 -> the model balances correctness (precision) and completeness (recall)\n"
            "     Low F1  -> either precision or recall (or both) is low\n"
        )
        logger.info("Interpretation Notes:\n%s", notes)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[group1, group2])
        disp.plot(cmap="Blues", values_format="d")

        self._save_figure(disp.figure_, "confusion_matrix", savefig_kwargs=savefig_kwargs)

        return disp.figure_, disp.ax_

    def predict_type_posterior(
        self, X: NpFloat, *, X_sigma: NpFloat | None = None, prior_A: float = 0.5
    ) -> tuple[NpFloat, NpFloat]:
        """Computes posterior probabilities that each row in X is Type A or B.

        Args:
            X: Observations (n_samples, n_features)
            X_sigma: Optional known 1-sigma uncertainties of data (n_samples, n_features). Defaults
                to ``None``.
            prior_A: Prior probability of Type A. The prior probability of Type B is
                taken as ``1 - prior_A``. Defaults to ``0.5``.

        Returns:
            tuple:
                - Posterior probability of Type A (n_samples, n_draws)
                - Posterior probability of Type B (n_samples, n_draws)
        """
        # (draws, samples, groups, features)
        log_lik_feat: NpFloat = self.feature_log_likelihood(X, X_sigma=X_sigma)

        # (draws, samples, groups)
        log_lik: NpFloat = log_lik_feat.sum(axis=-1)

        # Add priors
        log_lik[:, :, 0] += np.log(prior_A)
        log_lik[:, :, 1] += np.log(1 - prior_A)

        prob: NpFloat = softmax(log_lik, axis=-1)

        # (draws, samples)
        P_A: NpFloat = prob[:, :, 0]
        P_B: NpFloat = prob[:, :, 1]

        return P_A, P_B

    def feature_log_likelihood(self, X: NpFloat, *, X_sigma: NpFloat | None = None) -> NpFloat:
        """Returns per-feature log likelihood.

        Args:
            X: Observations (n_samples, n_features)
            X_sigma: Optional known 1-sigma uncertainties of data (n_samples, n_features). Defaults
                to ``None``.

        Returns:
            log-likelihood for each feature
        """
        # (n_draws, n_groups, n_features)
        mu_samples: NpFloat = self.idata["posterior"]["mu"].stack(draws=("chain", "draw")).values
        mu_samples = np.transpose(mu_samples, (2, 0, 1))  # (draws, group, feature)
        # logger.debug("mu_A_samples.shape = %s", mu_samples.shape)

        # (n_draws, n_features)
        feature_sigma_samples: NpFloat = (
            self.idata["posterior"]["sigma"].stack(draws=("chain", "draw")).values
        )
        feature_sigma_samples = np.transpose(feature_sigma_samples, (1, 0))  # (draws, feature)
        # logger.debug("feature_sigma_samples.shape = %s", feature_sigma_samples.shape)

        # Expand data
        X_b: NpFloat = X[None, :, None, :]  # (1, samples, 1, features)

        # Total observational noise (independent of group membership)
        if X_sigma is not None:
            sigma_b: NpFloat = np.sqrt(
                feature_sigma_samples[:, None, :] ** 2 + X_sigma[None, :, :] ** 2
            )  # (draws, samples, features)
        else:
            sigma_b = feature_sigma_samples[:, None, :]  # (draws, 1, features)

        sigma_b = sigma_b[:, :, None, :]  # (draws, samples, 1, features)

        # Compute log-likelihood:
        # (draws, samples, groups, features)
        log_lik_feat: NpFloat = -0.5 * (
            ((X_b - mu_samples[:, None, :, :]) ** 2) / (sigma_b**2)
            + np.log(2 * np.pi * sigma_b**2)
        )

        return log_lik_feat

    def feature_log_likelihood_ratio(
        self, X: NpFloat, *, X_sigma: NpFloat | None = None
    ) -> NpFloat:
        """Computes the log-likelihood ratio for each feature.

        Args:
            X: Observations (n_samples, n_features)
            X_sigma: Optional known 1-sigma uncertainties of data (n_samples, n_features). Defaults
                to ``None``.

        Returns:
            log-likelihood ratio: log p(X|B) - log p(X|A)
        """
        # (draws, samples, groups, features)
        log_lik_feat: NpFloat = self.feature_log_likelihood(X, X_sigma=X_sigma)

        # How much each feature prefers group B versus group A in log-likelihood
        delta_log_lik_feat: NpFloat = log_lik_feat[:, :, 1, :] - log_lik_feat[:, :, 0, :]

        return delta_log_lik_feat

    def feature_llr_diagnostics(
        self, X: NpFloat, X_group_idx: NpInt, *, X_sigma: NpFloat | None = None
    ) -> pd.DataFrame:
        """Summarizes the feature-wise log-likelihood ratio (LLR) statistics.

        The LLR is defined as

            log p(x | Group B) - log p(x | Group A),

        so positive values favor Group B and negative values favor Group A. This is also known as
        the weight of evidence or likelihood log-odds. The LLR can be interpreted as the amount of
        evidence in favor of one group over the other, with larger absolute values indicating
        stronger evidence.

        Args:
            X: Observations (n_samples, n_features)
            X_group_idx: True group index for each sample (n_samples,)
            X_sigma: Optional known 1-sigma uncertainties of the observations. Defaults to
                ``None``.

        Returns:
            DataFrame containing feature-wise summary statistics
        """
        # (draws, samples, groups, features)
        evidence_draws: NpFloat = self.feature_log_likelihood_ratio(X, X_sigma=X_sigma)
        evidence_mean: NpFloat = evidence_draws.mean(axis=(0, 1))
        evidence_mag_mean: NpFloat = np.abs(evidence_draws).mean(axis=(0, 1))
        alignment_sign: NpFloat = np.where(X_group_idx == 1, 1.0, -1.0)[None, :, None]
        evidence_aligned_draws: NpFloat = evidence_draws * alignment_sign
        evidence_aligned_mean: NpFloat = evidence_aligned_draws.mean(axis=(0, 1))
        evidence_aligned_std: NpFloat = evidence_aligned_draws.std(axis=(0, 1))
        stability_snr: NpFloat = evidence_aligned_mean / (evidence_aligned_std + 1e-8)
        consistency: NpFloat = (evidence_draws > 0).mean(axis=(0, 1))

        # Average raw LLR isolated by true group membership (which features characterize A vs B)
        group_a_mask: NpInt = X_group_idx == 0
        group_b_mask: NpInt = X_group_idx == 1

        # Highly negative values mean strongly characteristic of Group A
        profile_a: NpFloat = (
            np.mean(evidence_draws[:, group_a_mask, :], axis=(0, 1))
            if np.any(group_a_mask)
            else np.zeros(evidence_draws.shape[2])
        )
        # Highly positive values mean strongly characteristic of Group B
        profile_b: NpFloat = (
            np.mean(evidence_draws[:, group_b_mask, :], axis=(0, 1))
            if np.any(group_b_mask)
            else np.zeros(evidence_draws.shape[2])
        )

        group1, group2 = self.coords["group"]

        diagnostics: dict[str, NpFloat] = {
            "Evidence": evidence_mean,
            "Evidence Magnitude": evidence_mag_mean,
            "Diagnostic Evidence": evidence_aligned_mean,
            "Stability SNR": stability_snr,
            f"{group2} Vote Rate": consistency,
            f"{group1} Conditional Evidence": profile_a,
            f"{group2} Conditional Evidence": profile_b,
        }

        df: pd.DataFrame = pd.DataFrame(diagnostics, index=self.coords["feature"])
        df.index.name = "Feature"
        df = df.sort_values(by="Diagnostic Evidence", ascending=False)
        df = df.reset_index(drop=False)
        df.index += 1  # 1-based indexing for reporting
        df.index.name = "Rank"

        logger.info("Feature-wise log-likelihood ratio diagnostics:\n%s\n", df.round(3))

        # Dump interpretation notes to log for user reference
        notes: str = (
            " - Feature: Name of the feature.\n"
            " - Evidence: Expected LLR across samples and draws, indicating which class the feature supports.\n"
            " - Evidence Magnitude: Expected magnitude of feature evidence (ignoring correctness).\n"
            " - Diagnostic Evidence: How well the feature supports the correct classification.\n"
            " - Stability SNR: How reliably the feature contributes in the right direction.\n"
            f" - {group2} Vote Rate: Proportion of times the feature favous {group2}.\n"
            f" - {group1} Conditional Evidence: Expected LLR for {group1} samples.\n"
            f" - {group2} Conditional Evidence: Expected LLR for {group2} samples.\n"
        )

        logger.info("Interpretation Notes:\n%s", notes)

        if self.output_directory is not None:
            df.to_excel(self.output_directory / Path(f"{self.name}_feature_llr_diagnostics.xlsx"))

        return df

    def sample_llr_diagnostics(
        self, X: NpFloat, X_group_idx: NpInt, *, X_sigma: NpFloat | None = None, ci: bool = False
    ) -> pd.DataFrame:
        """Summarizes the sample-wise log-likelihood ratio (LLR) statistics.

        The LLR is defined as

            log p(x | Group B) - log p(x | Group A),

        so positive values favor Group B and negative values favor Group A. This is also known as
        the weight of evidence or likelihood log-odds. The LLR can be interpreted as the amount of
        evidence in favor of one group over the other, with larger absolute values indicating
        stronger evidence.

        Args:
            X: Observations (n_samples, n_features)
            X_group_idx: True group index for each sample (n_samples,)
            X_sigma: Optional known 1-sigma uncertainties of the observations. Defaults to
                ``None``.
            ci: Whether to compute confidence intervals. Defaults to ``False``.

        Returns:
            DataFrame containing sample-wise summary statistics
        """
        names: list[str] = ["Metric", "Stat", "Variable"]
        _, group2 = self.coords["group"]

        def add_to_cols_and_data(cols: list, data: list, metric: str, values: NpFloat) -> None:
            """Helper function to add columns and data for a given metric"""
            values_mean: NpFloat = values.mean(axis=0)
            for i, f in enumerate(self.coords["feature"]):
                cols.append((metric, "Mean", f))
                data.append(values_mean[:, i])
                if ci:
                    cols.append((metric, "Low", f))
                    data.append(np.percentile(values, LOW_PERCENTILE, axis=0)[:, i])
                    cols.append((metric, "High", f))
                    data.append(np.percentile(values, HIGH_PERCENTILE, axis=0)[:, i])

            cols.append((metric, "Mean", "Total"))
            data.append(values_mean.sum(axis=-1))

        cols: list = []
        data: list = []

        # Local evidence
        # (draws, samples, groups, features)
        evidence_draws: NpFloat = self.feature_log_likelihood_ratio(X, X_sigma=X_sigma)
        add_to_cols_and_data(cols, data, "Evidence", evidence_draws)

        # Local evidence magnitude
        evidence_mag_draws: NpFloat = np.abs(evidence_draws)
        add_to_cols_and_data(cols, data, "Evidence Magnitude", evidence_mag_draws)

        # Aligned evidence (diagnostic evidence)
        # Positive = correct evidence; Negative = misleading evidence
        # Does this feature help classification? Best feature importance metric
        alignment_sign: NpFloat = np.where(X_group_idx == 1, 1.0, -1.0)[None, :, None]
        evidence_aligned_draws: NpFloat = evidence_draws * alignment_sign
        add_to_cols_and_data(cols, data, "Diagnostic Evidence", evidence_aligned_draws)

        # Stability score (signal-to-noise ratio of the correct evidence)
        # How reliably does this feature contribute in the right direction?
        evidence_aligned_mean: NpFloat = evidence_aligned_draws.mean(axis=0)
        evidence_aligned_std: NpFloat = np.std(evidence_aligned_draws, axis=0)
        stability_snr: NpFloat = evidence_aligned_mean / (evidence_aligned_std + 1e-8)
        consistency: NpFloat = (evidence_draws > 0).mean(axis=0)

        # Keep all stability and vote rate metrics clustered together
        for i, f in enumerate(self.coords["feature"]):
            cols.append(("Stability SNR", "Mean", f))
            data.append(stability_snr[:, i])
        for i, f in enumerate(self.coords["feature"]):
            cols.append((f"{group2} Vote Rate", "Mean", f))
            data.append(consistency[:, i])

        # Prediction and truth
        _, P_B = self.predict_type_posterior(X, X_sigma=X_sigma)  # (draws, samples)
        P_true: NpFloat = np.where(X_group_idx[None, :] == 1, P_B, 1 - P_B)  # (draws, samples)
        cols.append(("Prediction", "Mean", "True Class Probability"))
        data.append(P_true.mean(axis=0))
        if ci:
            cols.append(("Prediction", "Low", "True Class Probability"))
            data.append(np.percentile(P_true, LOW_PERCENTILE, axis=0))
            cols.append(("Prediction", "High", "True Class Probability"))
            data.append(np.percentile(P_true, HIGH_PERCENTILE, axis=0))

        # Add predicted group
        cols.append(("Prediction", "Predicted Class", "Name"))
        predicted_group: NpArray = np.where(P_B.mean(axis=0) > 0.5, 1, 0)
        data.append(np.array([self.coords["group"][i] for i in predicted_group]))

        # Add true group
        cols.append(("Prediction", "True Class", "Name"))
        data.append(np.array([self.coords["group"][i] for i in X_group_idx]))

        df: pd.DataFrame = pd.DataFrame(
            np.column_stack(data), columns=pd.MultiIndex.from_tuples(cols, names=names)
        )
        df.index = np.arange(1, evidence_draws.shape[1] + 1)
        df.index.name = "Sample"

        # Sort samples by P_True ascending so the worst misclassifications float to the top. This
        # keeps the multi-index intact while shifting the row order.
        df = df.sort_values(by=("Prediction", "Mean", "True Class Probability"), ascending=True)

        # Re-index the sample row labels to maintain a clean 1-to-N tracking sequence
        df = df.reset_index(drop=False)
        df.index = np.arange(1, len(df) + 1)
        df.index.name = "Ranked Sample (Worst First)"

        # Append the global mean summary row at the very bottom after sorting
        df.loc["Mean"] = df.mean(axis=0, numeric_only=True)

        if self.output_directory is not None:
            df.to_excel(self.output_directory / Path(f"{self.name}_sample_llr_diagnostics.xlsx"))

        logger.info(
            "Note that the confidence intervals are:\n - Low = %0.2f%%\n- High = %0.2f%%",
            LOW_PERCENTILE,
            HIGH_PERCENTILE,
        )

        return df

    # TODO: This needs to use the new sample_llr_diagnostics method to get the data for plotting,
    # and then plot it nicely. The current implementation is commented out because it is not yet
    # updated to use the new method.
    # def plot_explanation(
    #     self,
    #     X: NpFloat,
    #     *,
    #     X_sigma: NpFloat | None = None,
    #     X_group_id: NpInt | None = None,
    #     prior_A: float = 0.5,
    #     sample_idx: int = 0,
    # ) -> tuple[Figure, Axes]:
    #     """Plots the feature contributions to the classification of a single sample.

    #     Args:
    #         X: Data (n_samples, n_features)
    #         X_sigma: Optional known 1-sigma uncertainties of data (n_samples, n_features).
    #             Defaults to ``None``.
    #         X_group_id: Group ID of observations (n_samples,). Defaults to ``None``.
    #         prior_A: Prior probability of Type A. The prior probability of Type B is
    #             taken as ``1 - prior_A``. Defaults to ``0.5``.
    #         sample_idx: Index of the sample to plot. Defaults to ``0``.

    #     Returns:
    #         tuple:
    #             - Matplotlib Figure
    #             - Matplotlib Axes
    #     """
    #     explanation: dict[str, Any] = self.explain_samples(
    #         X, X_sigma=X_sigma, X_group_id=X_group_id, prior_A=prior_A
    #     )

    #     mean: NpFloat = explanation["mean"][sample_idx]
    #     ci: NpFloat = explanation["ci95"][:, sample_idx]

    #     feature_names: list[str] = list(self.idata["posterior"].coords["feature"].values)

    #     order: NpArray = np.argsort(np.abs(mean))

    #     fig, ax = plt.subplots(figsize=(8, 0.45 * len(mean) + 1))

    #     ax.errorbar(
    #         mean[order],
    #         np.arange(len(mean)),
    #         xerr=[
    #             mean[order] - ci[0, order],
    #             ci[1, order] - mean[order],
    #         ],
    #         fmt="o",
    #         capsize=3,
    #     )

    #     ax.axvline(0, color="k", ls="--", alpha=0.5)

    #     total_mean: float = explanation["total_mean"][sample_idx]
    #     total_ci: NpFloat = explanation["total_ci95"][:, sample_idx]

    #     if explanation["classification_check"] is not None:
    #         correct: NpBool = (
    #             explanation["classification_check"]["per_sample_accuracy"][sample_idx] > 0.5
    #         )
    #         color: str = "lightgreen" if correct else "lightcoral"

    #         per_sample_accuracy: NpFloat = explanation["classification_check"][
    #             "per_sample_accuracy"
    #         ][sample_idx]
    #         ax.set_title(
    #             f"Sample {sample_idx} (Total LLR = {total_mean:.2f}, "
    #             f"95% CI = [{total_ci[0]:.2f}, {total_ci[1]:.2f}], "
    #             f"Per-sample accuracy = {per_sample_accuracy:.2%})"
    #         )
    #     else:
    #         color = "lightgray"
    #         ax.set_title(
    #             f"Sample {sample_idx} (Total LLR = {total_mean:.2f}, "
    #             f"95% CI = [{total_ci[0]:.2f}, {total_ci[1]:.2f}])"
    #         )

    #     ax.axvline(total_mean, color=color, lw=2, alpha=0.8, label="Total")
    #     ax.axvspan(total_ci[0], total_ci[1], color=color, alpha=0.15)

    #     ax.set_xlabel(f"Log-likelihood contribution {self.difference_string}")
    #     ax.set_yticks(np.arange(len(mean)))
    #     ax.set_yticklabels(np.array(feature_names)[order])

    #     return fig, ax

    def run_analysis(self, *, savefig_kwargs: dict[str, Any] | None = None) -> None:
        """Runs the analysis: inference, posterior predictive checks, etc. and saves figures.

        Args:
            savefig_kwargs: Keyword arguments for :func:`matplotlib.pyplot.savefig`. Defaults to
                ``None`` to use :obj:`SAVEFIG_KWARGS`.
        """
        logger.info("Running analysis for %s", self.name)
        self.run_inference()
        self.plot_prior_predictive(savefig_kwargs=savefig_kwargs)
        self.plot_posterior_predictive(savefig_kwargs=savefig_kwargs)
        self.plot_dist(savefig_kwargs=savefig_kwargs)
        self.plot_forest(savefig_kwargs=savefig_kwargs)
        self.plot_forest_effect_size(savefig_kwargs=savefig_kwargs)
        logger.info("Analysis complete for %s", self.name)

    def evaluate(
        self,
        X_test: NpFloat,
        X_test_group_idx: NpInt,
        *,
        X_test_sigma: NpFloat | None = None,
        savefig_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Evaluates the model on new data and plots the confusion matrix.

        Args:
            X_test: Observations (n_samples, n_features)
            X_test_group_idx: Group ID of observations, must be 0 or 1 (n_samples,)
            X_test_sigma: Sigma of observations (n_samples, n_features). Defaults to ``None``.
            savefig_kwargs: Keyword arguments for :func:`matplotlib.pyplot.savefig`. Defaults to
                ``None`` to use :obj:`SAVEFIG_KWARGS`.
        """
        logger.info("Evaluating model on new data for %s", self.name)
        self.plot_confusion_matrix(
            X_test, X_test_group_idx, X_test_sigma=X_test_sigma, savefig_kwargs=savefig_kwargs
        )
        self.feature_llr_diagnostics(X_test, X_test_group_idx, X_sigma=X_test_sigma)
        self.sample_llr_diagnostics(X_test, X_test_group_idx, X_sigma=X_test_sigma)
        logger.info("Evaluation complete for %s", self.name)
