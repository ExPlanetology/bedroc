# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bayesian hierarchical model for group-centric comparison of two groups"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.special import softmax
from scipy.stats import beta, gaussian_kde
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from bedroc.core import HIGH_PERCENTILE, LOW_PERCENTILE, RANDOM_SEED, save_figure
from bedroc.difference.group_difference import HierarchicalGroupDifferenceModel
from bedroc.type_aliases import NpArray, NpFloat, NpInt

logger: logging.Logger = logging.getLogger(__name__)


class GroupClassifierModel:
    """Bayesian classifier and population-fraction estimator built on a fitted group model.

    Wraps a trained :class:`HierarchicalGroupModel` to provide classification, diagnostic, and
    prevalence-inference methods for new (likely unlabeled) data. The posterior over model
    parameters learned during training is propagated through all predictions.

    Args:
        fitted_group_model: A :class:`HierarchicalGroupModel` on which ``run_inference`` has
            already been called
        X: Data to classify (n_samples, n_features)
        X_group_idx: Group index for each row of ``X``, must be 0 or 1 (n_samples,). Defaults to
            ``None`` (unlabeled).
        X_sigma: Optional 1-sigma uncertainties for ``X`` (n_samples, n_features). Defaults to
            ``None``.
        output_directory: Optional path to save generated data. Defaults to ``None`` (no saving).
    """

    def __init__(
        self,
        fitted_model: HierarchicalGroupDifferenceModel,
        X: NpFloat,
        *,
        X_group_idx: NpInt | None = None,
        X_sigma: NpFloat | None = None,
        output_directory: Path | None = None,
    ):
        self.fitted_model: HierarchicalGroupDifferenceModel = fitted_model
        self.X: NpFloat = X
        self.X_group_idx: NpInt | None = X_group_idx
        self.X_sigma: NpFloat | None = X_sigma
        self.output_directory: Path | None = output_directory
        if self.output_directory is not None:
            self.output_directory.mkdir(parents=True, exist_ok=True)
        # For caching the feature log-likelihood to avoid recomputation
        # self._feature_log_likelihood: NpFloat = self._compute_feature_log_likelihood()
        self._feature_log_likelihood: NpFloat = self.new_likelihood()

    @property
    def name(self) -> str:
        return self.fitted_model.name

    @property
    def feature_log_likelihood(self) -> NpFloat:
        """Returns the cached per-feature log likelihood."""
        return self._feature_log_likelihood

    def new_likelihood(self):
        """Computes log likelihoood of new data under each group"""

        sample_idx, feature_idx = np.where(np.isfinite(self.X))

        self.observation_sample_idx = sample_idx
        self.observation_feature_idx = feature_idx

        X_data_np = self.X[sample_idx, feature_idx]

        data: dict = {
            "X_data": X_data_np,
            "feature_idx": feature_idx,
            "group_idx": np.zeros(len(X_data_np), dtype=int),
        }

        if self.X_sigma is not None:
            data["X_sigma"] = self.X_sigma[sample_idx, feature_idx]

        with self.fitted_model.model:
            pm.set_data(
                data,
                coords={"observation": np.arange(len(X_data_np))},
            )

            ll_A: xr.Dataset = pm.compute_log_likelihood(
                self.fitted_model.idata, var_names=["observations"], extend_inferencedata=False
            )  # pyright: ignore

            data["group_idx"] = np.ones(len(X_data_np), dtype=int)

            pm.set_data(
                data,
                coords={"observation": np.arange(len(X_data_np))},
            )

            # print(
            #     "group B:",
            #     self.fitted_model.model["group_idx"].get_value()[:20],
            # )

            ll_B: xr.Dataset = pm.compute_log_likelihood(
                self.fitted_model.idata, var_names=["observations"], extend_inferencedata=False
            )  # pyright: ignore

        print("here")
        print(ll_A)

        # Convert to the original form for the interface
        return self._compute_feature_likelihood(ll_A, ll_B)

    def new_compute_log_likelihood(self, idata_A, idata_B):
        """Computes per-feature log likelihood for each group."""

        ll_A = idata_A["observations"]
        ll_B = idata_B["observations"]

        print("A shape:", ll_A.shape)
        print("B shape:", ll_B.shape)

        print("max abs difference:", np.max(np.abs(ll_A - ll_B)))
        print("mean abs difference:", np.mean(np.abs(ll_A - ll_B)))

        print("A first:", ll_A[0, 0, :10])
        print("B first:", ll_B[0, 0, :10])

        n_sample, n_feature = self.X.shape

        log_lik = np.zeros(
            (ll_A.sizes["chain"], ll_A.sizes["draw"], n_sample, 2, n_feature),
            dtype=float,
        )

        sample_idx, feature_idx = np.where(np.isfinite(self.X))

        for observation, (sample, feature) in enumerate(zip(sample_idx, feature_idx)):
            log_lik[:, :, sample, 0, feature] = ll_A.values[:, :, observation]
            log_lik[:, :, sample, 1, feature] = ll_B.values[:, :, observation]

        return xr.DataArray(
            log_lik,
            dims=("chain", "draw", "sample", "group", "feature"),
            coords={
                "chain": ll_A.coords["chain"],
                "draw": ll_A.coords["draw"],
                "sample": np.arange(n_sample),
                "group": self.fitted_model.coords["group"],
                "feature": self.fitted_model.coords["feature"],
            },
            name="log_likelihood",
        )

    def _compute_feature_likelihood(self, idata_A, idata_B) -> NpFloat:
        """Returns per-feature log likelihood in the legacy array format.

        Returns:
            Array with shape ``(draws, samples, groups, features)``.
        """
        log_lik = self.new_compute_log_likelihood(idata_A, idata_B)

        return (
            log_lik.stack(draws=("chain", "draw"))
            .transpose("draws", "sample", "group", "feature")
            .values
        )

    def _compute_feature_log_likelihood(self) -> NpFloat:
        return self._compute_feature_laplace_log_likelihood()  # _null()

    def _compute_feature_laplace_log_likelihood(self) -> NpFloat:
        """Computes per-feature Laplace log likelihood.

        Returns:
            Log-likelihood for each feature
        """
        # (draws, groups, features)
        mu_samples: NpFloat = (
            self.fitted_model.idata["posterior"]["mu"].stack(draws=("chain", "draw")).values
        )
        mu_samples = np.transpose(mu_samples, (2, 0, 1))  # (draws, groups, features)
        # logger.debug("mu_A_samples.shape = %s", mu_samples.shape)

        # (draws, features)
        feature_sigma_samples: NpFloat = (
            self.fitted_model.idata["posterior"]["sigma"].stack(draws=("chain", "draw")).values
        )
        feature_sigma_samples = np.transpose(feature_sigma_samples, (1, 0))
        # logger.debug("feature_sigma_samples.shape = %s", feature_sigma_samples.shape)

        # Expand dimensions
        # X_b:          (1, samples, 1, features)
        # mu_samples:   (draws, 1, groups, features)
        # sigma:        (draws, samples, 1, features)
        X_b: NpFloat = self.X[None, :, None, :]

        # Total standard deviation used by the fitted model
        if self.X_sigma is not None:
            sigma_total: NpFloat = np.sqrt(
                feature_sigma_samples[:, None, :] ** 2 + self.X_sigma[None, :, :] ** 2
            )
        else:
            sigma_total = feature_sigma_samples[:, None, :]

        sigma_total = sigma_total[:, :, None, :]

        # The fitted model uses:
        #
        #     b = sigma_total / sqrt(2)
        #
        # because sigma_total is interpreted as the standard deviation of
        # the Laplace distribution.
        b: NpFloat = sigma_total / np.sqrt(2.0)

        # Laplace log likelihood
        #
        # log p(x | mu, b)
        #     = -log(2 b) - |x - mu| / b
        #
        # Shape: (draws, samples, groups, features)
        log_lik_feat: NpFloat = -np.log(2.0 * b) - np.abs(X_b - mu_samples[:, None, :, :]) / b

        # Missing features contribute zero to the total log likelihood, equivalent to multiplying
        # the likelihood by 1.
        observed: NpFloat = np.isfinite(self.X)

        log_lik_feat = np.where(observed[None, :, None, :], log_lik_feat, 0.0)

        return log_lik_feat

    def _compute_feature_laplace_log_likelihood_null(self) -> NpFloat:
        """Computes an uninformative per-feature Laplace log likelihood.

        The two groups have identical likelihoods, such that

            p(x | A) = p(x | B).

        Consequently, the likelihood contains no information about group
        membership and the inferred group fraction should reproduce its prior.
        """
        n_samples, n_features = self.X.shape

        # No discrimination between groups:
        #
        #     log p(x | A) = log p(x | B)
        #
        # Setting both to zero means that the likelihood ratio is exactly one.
        log_lik_feat = np.zeros((1, n_samples, 2, n_features), dtype=float)

        # Missing features contribute zero to the total log likelihood,
        # consistent with the normal likelihood implementation.
        observed = np.isfinite(self.X)
        log_lik_feat = np.where(observed[None, :, None, :], log_lik_feat, 0.0)

        return log_lik_feat

    def _compute_feature_gaussian_log_likelihood(self) -> NpFloat:
        """Computes per-feature Gaussian log likelihood.

        Returns:
            log-likelihood for each feature
        """
        # (draws, groups, features)
        mu_samples: NpFloat = (
            self.fitted_model.idata["posterior"]["mu"].stack(draws=("chain", "draw")).values
        )
        mu_samples = np.transpose(mu_samples, (2, 0, 1))  # (draws, groups, features)
        # logger.debug("mu_A_samples.shape = %s", mu_samples.shape)

        # (draws, features)
        feature_sigma_samples: NpFloat = (
            self.fitted_model.idata["posterior"]["sigma"].stack(draws=("chain", "draw")).values
        )
        feature_sigma_samples = np.transpose(feature_sigma_samples, (1, 0))  # (draws, features)
        # logger.debug("feature_sigma_samples.shape = %s", feature_sigma_samples.shape)

        # Expand data
        X_b: NpFloat = self.X[None, :, None, :]  # (1, samples, 1, features)

        # Total observational noise (independent of group membership)
        if self.X_sigma is not None:
            sigma_b: NpFloat = np.sqrt(
                feature_sigma_samples[:, None, :] ** 2 + self.X_sigma[None, :, :] ** 2
            )  # (draws, samples, features)
        else:
            sigma_b = feature_sigma_samples[:, None, :]  # (draws, 1, features)

        sigma_b = sigma_b[:, :, None, :]  # (draws, samples, 1, features)

        # Compute log-likelihood
        # (draws, samples, groups, features)
        log_lik_feat: NpFloat = -0.5 * (
            ((X_b - mu_samples[:, None, :, :]) ** 2) / (sigma_b**2)
            + np.log(2 * np.pi * sigma_b**2)
        )

        # Missing features (NaN in X) contribute 0 to the log-likelihood sum (i.e., factor of 1).
        observed: NpFloat = np.isfinite(self.X)  # (samples, features)
        log_lik_feat = np.where(observed[None, :, None, :], log_lik_feat, 0.0)

        return log_lik_feat

    def feature_log_likelihood_ratio(self) -> NpFloat:
        """Computes the log-likelihood ratio for each feature.

        Returns:
            log-likelihood ratio: log p(X|B) - log p(X|A)
        """
        # (draws, samples, groups, features)
        log_lik_feat: NpFloat = self.feature_log_likelihood

        # How much each feature prefers group B versus group A in log-likelihood
        delta_log_lik_feat: NpFloat = log_lik_feat[:, :, 1, :] - log_lik_feat[:, :, 0, :]

        return delta_log_lik_feat

    def predict_type_posterior(self, *, prior_A: float = 0.5) -> tuple[NpFloat, NpFloat]:
        """Posterior for the classification of individual samples

        This infers the class of each sample assuming a known class prior. This is the standard
        Bayesian classifier implementing Bayes' theorem where:

            p(X | A) comes from the hierarchical model
            P(A) is the prior probability of group A membership
            P(B) = 1 - P(A) is the prior probability of group B membership

        Args:
            prior_A: For an individual sample, absent its measurements, defines the prior
                probability of membership in the first group. The prior probability of membership
                in the second group is taken as ``1 - prior_A``. Defaults to ``0.5``.

        Returns:
            tuple:
                - Posterior probability of the first group (n_samples, n_draws)
                - Posterior probability of the second group (n_samples, n_draws)
        """
        # (draws, samples, groups, features)
        log_lik_feat: NpFloat = self.feature_log_likelihood

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

    def infer_group_fraction(
        self,
        *,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        n_grid: int = 2001,
        n_posterior_samples: int = 10_000,
        random_seed: int | None = RANDOM_SEED,
    ) -> dict[str, Any]:
        """Infers the group fractions (i.e. prevalence of classes) in an unlabeled dataset.

        Asks the question, "What value of the common population fraction of group A (pi) best
        explains the entire unlabeled dataset?"

        The fraction of the first group is treated as an unknown population parameter and inferred
        jointly from all observations. The likelihood is a two-component mixture,

            p(x | pi) = pi * p(x | A) + (1 - pi) * p(x | B),

        where ``pi`` is the fraction of the dataset belonging to group A.

        Posterior uncertainty in the learned group distributions is propagated by evaluating the
        mixture likelihood for every posterior draw of the fitted model.

        A Beta prior is used for the group-A fraction:

            pi ~ Beta(prior_alpha, prior_beta)

        Args:
            prior_alpha: Alpha parameter of the Beta prior on the fraction of group A. Defaults to
                ``1.0``.
            prior_beta: Beta parameter of the Beta prior on the fraction of group A. Defaults to
                ``1.0``.
            n_grid: Number of points used to represent the posterior distribution of the group-A
                fraction. Defaults to ``2001``.
            n_posterior_samples: Number of samples drawn from the resulting posterior distribution.
                Defaults to ``10000``.
            random_seed: Random seed for reproducibility. Defaults to :obj:`RANDOM_SEED`.

        Returns:
            Dictionary containing:

            ``fraction_A_samples``
                Posterior samples of the fraction belonging to group A

            ``fraction_B_samples``
                Posterior samples of the fraction belonging to group B

            ``summary``
                Dictionary containing posterior mean, median, and 95% credible interval for both
                groups

            ``grid``
                Grid of group-A fractions

        Raises:
            ValueError: If the Beta prior parameters or grid size are invalid
        """
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("prior_alpha and prior_beta must be > 0.")

        if n_grid < 2:
            raise ValueError("n_grid must be at least 2.")

        if n_posterior_samples < 1:
            raise ValueError("n_posterior_samples must be at least 1.")

        group1, group2 = self.fitted_model.coords["group"]

        logger.info(
            "Inferring group fractions for %d unlabeled samples using Beta(%g, %g) prior",
            self.X.shape[0],
            prior_alpha,
            prior_beta,
        )

        # (draws, samples, groups, features)
        log_lik_feat: NpFloat = self.feature_log_likelihood

        # Sum over features: (draws, samples, groups)
        log_lik: NpFloat = log_lik_feat.sum(axis=-1)

        # Separate class-conditional likelihoods: (draws, samples)
        log_lik_A: NpFloat = log_lik[:, :, 0]
        log_lik_B: NpFloat = log_lik[:, :, 1]

        n_draws: int = log_lik_A.shape[0]

        # Grid over population fraction of group A. Avoid exactly 0 and 1 because the logarithm of
        # the mixture weights would otherwise contain -inf.
        eps: np.float64 = np.finfo(float).eps
        fraction_A_grid: NpFloat = np.linspace(eps, 1.0 - eps, n_grid)

        log_fraction_A: NpFloat = np.log(fraction_A_grid)
        log_fraction_B: NpFloat = np.log1p(-fraction_A_grid)

        # Beta prior
        # Normalization constant of the Beta distribution is irrelevant because we normalize the
        # posterior below.
        log_prior: NpFloat = (prior_alpha - 1.0) * log_fraction_A + (
            prior_beta - 1.0
        ) * log_fraction_B

        # Compute p(pi | X, theta) for every posterior draw theta.
        #
        # For each posterior draw:
        #
        #   p(X | pi, theta)
        #       = product_i [ pi p(x_i | A, theta) + (1-pi) p(x_i | B, theta) ]
        #
        # We work in log space for numerical stability, looping over draws to bound memory.
        #
        # Result:
        #   (draws, grid)

        # Intermediate (draws, samples, grid) arrays would be ~tens of GB; loop over draws instead.
        log_likelihood_fraction: NpFloat = np.empty((n_draws, n_grid))
        for d in range(n_draws):
            log_comp_A = log_lik_A[d, :, None] + log_fraction_A[None, :]  # (samples, grid)
            log_comp_B = log_lik_B[d, :, None] + log_fraction_B[None, :]  # (samples, grid)
            log_likelihood_fraction[d] = np.logaddexp(log_comp_A, log_comp_B).sum(axis=0)
        # Each row is a posterior for pi conditional on one posterior draw of the trained model.
        log_posterior_draws: NpFloat = (
            log_likelihood_fraction + log_prior[None, :]
        )  # (draws, grid)

        # Normalize each posterior distribution over the fraction grid (i.e., axis=1)
        posterior_draws = np.exp(
            log_posterior_draws - np.max(log_posterior_draws, axis=1, keepdims=True)
        )

        # Normalize using trapezoidal integration
        normalization = np.trapezoid(posterior_draws, fraction_A_grid, axis=1)  # (draws,)
        posterior_draws /= normalization[:, None]  # (draws, grid)

        # Build CDF for each model posterior draw
        cdf_draws = np.zeros_like(posterior_draws)  # (draws, grid)
        cdf_draws[:, 1:] = np.cumsum(
            0.5
            * (posterior_draws[:, 1:] + posterior_draws[:, :-1])
            * np.diff(fraction_A_grid)[None, :],
            axis=1,
        )
        cdf_draws /= cdf_draws[:, -1, None]

        # Draw samples from the posterior
        rng = np.random.default_rng(random_seed)

        # Randomly select posterior draws of the trained model
        selected_draws = rng.integers(0, n_draws, size=n_posterior_samples)

        # Random probabilities for inverse-CDF sampling.
        u = rng.random(n_posterior_samples)

        fraction_A_samples = np.empty(n_posterior_samples)

        for i, draw_idx in enumerate(selected_draws):
            fraction_A_samples[i] = np.interp(u[i], cdf_draws[draw_idx], fraction_A_grid)

        fraction_B_samples = 1.0 - fraction_A_samples

        # Summarize
        summary: dict[str, dict] = {
            group1: {
                "mean": np.mean(fraction_A_samples),
                "median": np.median(fraction_A_samples),
                "lower_95": np.percentile(fraction_A_samples, LOW_PERCENTILE),
                "upper_95": np.percentile(fraction_A_samples, HIGH_PERCENTILE),
            },
            group2: {
                "mean": np.mean(fraction_B_samples),
                "median": np.median(fraction_B_samples),
                "lower_95": np.percentile(fraction_B_samples, LOW_PERCENTILE),
                "upper_95": np.percentile(fraction_B_samples, HIGH_PERCENTILE),
            },
        }

        logger.info(
            "Inferred %s fraction = %.3f (95%% CI: %.3f - %.3f)",
            group1,
            summary[group1]["median"],
            summary[group1]["lower_95"],
            summary[group1]["upper_95"],
        )

        logger.info(
            "Inferred %s fraction = %.3f (95%% CI: %.3f - %.3f)",
            group2,
            summary[group2]["median"],
            summary[group2]["lower_95"],
            summary[group2]["upper_95"],
        )

        output = {
            "fraction_A_samples": fraction_A_samples,
            "fraction_B_samples": fraction_B_samples,
            "posterior_draw_indices": selected_draws,
            "summary": summary,
            "grid": fraction_A_grid,
        }

        if self.X_group_idx is not None:
            # Compute the true fraction of each group in the dataset
            true_fraction_A = np.mean(self.X_group_idx == 0)
            true_fraction_B = np.mean(self.X_group_idx == 1)
            summary[group1]["true"] = true_fraction_A
            summary[group2]["true"] = true_fraction_B

            logger.info(
                "True %s fraction = %.3f, %s fraction = %.3f",
                group1,
                true_fraction_A,
                group2,
                true_fraction_B,
            )

            output["true_fraction_A"] = true_fraction_A
            output["true_fraction_B"] = true_fraction_B

        return output

    def plot_group_fraction_posterior(
        self,
        result: dict,
        *,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        bins: int = 50,
        savefig_kwargs: dict[str, Any] | None = None,
    ) -> Axes:
        """Plot posterior group fractions with Beta prior and true fractions."""

        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("prior_alpha and prior_beta must be > 0.")

        fig, ax = plt.subplots(figsize=(8, 5))

        fraction_A = result["fraction_A_samples"]
        fraction_B = result["fraction_B_samples"]

        group_A, group_B = result["summary"].keys()

        color_A = "tab:blue"
        color_B = "tab:orange"

        x = np.linspace(0, 1, 1000)

        # ------------------------------------------------------------------
        # Posterior distributions
        # ------------------------------------------------------------------
        ax.hist(
            fraction_A,
            bins=bins,
            density=True,
            alpha=0.25,
            color=color_A,
            label=f"{group_A} (samples)",
        )
        ax.hist(
            fraction_B,
            bins=bins,
            density=True,
            alpha=0.25,
            color=color_B,
            label=f"{group_B} (samples)",
        )

        kde_A = gaussian_kde(fraction_A)
        kde_B = gaussian_kde(fraction_B)

        ax.plot(
            x,
            kde_A(x),
            color=color_A,
            linewidth=2,
            label=f"{group_A} (posterior)",
        )
        ax.plot(
            x,
            kde_B(x),
            color=color_B,
            linewidth=2,
            label=f"{group_B} (posterior)",
        )

        # ------------------------------------------------------------------
        # Beta prior
        # ------------------------------------------------------------------
        prior_pdf = beta.pdf(x, prior_alpha, prior_beta)

        ax.plot(
            x,
            prior_pdf,
            color="black",
            linestyle="--",
            linewidth=1.8,
            label=rf"Beta prior ($\alpha={prior_alpha:g},\ \beta={prior_beta:g}$)",
        )

        # ------------------------------------------------------------------
        # True fractions
        # ------------------------------------------------------------------
        true_fraction_A = result.get("true_fraction_A")
        true_fraction_B = result.get("true_fraction_B")

        ymax = max(
            np.max(kde_A(x)),
            np.max(kde_B(x)),
            np.max(prior_pdf),
        )

        if true_fraction_A is not None:
            ax.annotate(
                f"True {group_A}\n{true_fraction_A:.2f}",
                xy=(true_fraction_A, ymax * 0.72),
                xytext=(true_fraction_A, ymax * 1.05),
                ha="center",
                va="bottom",
                color=color_A,
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color_A,
                    lw=1.5,
                ),
            )

        if true_fraction_B is not None:
            ax.annotate(
                f"True {group_B}\n{true_fraction_B:.2f}",
                xy=(true_fraction_B, ymax * 0.72),
                xytext=(true_fraction_B, ymax * 1.05),
                ha="center",
                va="bottom",
                color=color_B,
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color_B,
                    lw=1.5,
                ),
            )

        # ------------------------------------------------------------------
        # Formatting
        # ------------------------------------------------------------------
        ax.set(
            title="Inferred Group Fractions",
            xlabel="Population fraction",
            ylabel="Density",
            xlim=(0, 1),
        )

        ax.legend(loc="lower left")
        ax.margins(y=0.15)

        save_figure(
            fig,
            "group_fraction_posterior",
            output_directory=self.output_directory,
            savefig_kwargs=savefig_kwargs,
        )

        return ax

    def plot_confusion_matrix(
        self, *, prior_A: float = 0.5, savefig_kwargs: dict[str, Any] | None = None
    ) -> tuple[Figure, Axes]:
        """Plots the confusion matrix and logs metrics.

        The predicted group is determined using a Bayesian MAP classifier based on the posterior
        mean probabilities.

        Args:
            prior_A: Prior probability of the first group. The prior probability of the second
                group is taken as ``1 - prior_A``. Defaults to ``0.5``.
            savefig_kwargs: Override keyword arguments for :func:`matplotlib.pyplot.savefig`.
                Defaults to ``None``.

        Returns:
            Figure, Axes
        """
        # (draws, samples)
        P_A, P_B = self.predict_type_posterior(prior_A=prior_A)

        group1, group2 = self.fitted_model.coords["group"]

        # Compute posterior mean probability
        mean_prob_A: NpFloat = P_A.mean(axis=0)
        mean_prob_B: NpFloat = P_B.mean(axis=0)
        logger.debug("Posterior probability of %s = %s", group1, mean_prob_A)
        logger.debug("Posterior probability of %s = %s", group2, mean_prob_B)

        # Choose the most probable type Bayesian MAP classifier: standard Naive Bayes rule
        predicted_type: NpFloat = np.where(mean_prob_A > mean_prob_B, group1, group2)
        groups: NpArray = np.array([group1, group2])
        true_labels: NpFloat = groups[self.X_group_idx]

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

        save_figure(
            disp.figure_,
            "confusion_matrix",
            output_directory=self.output_directory,
            savefig_kwargs=savefig_kwargs,
        )

        return disp.figure_, disp.ax_

    def classification_value(
        self, *, prior_alpha: float = 1.0, prior_beta: float = 1.0, n_grid: int = 501
    ) -> pd.DataFrame:
        """Estimates how valuable perfect classification of each sample would be.

        For each sample, calculate the population-fraction posterior under two hypothetical
        scenarios:

            1. The sample is known with certainty to belong to Group A.
            2. The sample is known with certainty to belong to Group B.

        The difference between these two estimates measures how strongly the population-fraction
        inference depends on resolving that sample's classification.

        A large value means that improving the classification of the sample could substantially
        affect the inferred population fraction.

        Args:
            prior_alpha: Alpha parameter of the Beta prior on the fraction of group A. Defaults to
                ``1.0``.
            prior_beta: Beta parameter of the Beta prior on the fraction of group A. Defaults to
                ``1.0``.
            n_grid: Number of points used to represent the posterior distribution of the group-A
                fraction. Defaults to ``501``.

        Returns:
            DataFrame containing the classification value for each sample, sorted by descending
            classification value
        """
        group1, group2 = self.fitted_model.coords["group"]

        # Get class-conditional likelihoods from the trained model
        # (draws, samples, group)
        log_lik = self.feature_log_likelihood.sum(axis=-1)

        log_lik_A = log_lik[:, :, 0]
        log_lik_B = log_lik[:, :, 1]

        n_draws, n_samples = log_lik_A.shape

        # Population-fraction grid and prior
        eps = np.finfo(float).eps
        pi = np.linspace(eps, 1 - eps, n_grid)

        log_pi = np.log(pi)
        log_1mpi = np.log1p(-pi)

        log_prior = (prior_alpha - 1) * log_pi + (prior_beta - 1) * log_1mpi

        # Precompute total log-mixture sum over all samples: (draws, grid).
        # Loop over draws to avoid a (draws, samples, grid) intermediate array.
        log_likelihood_all = np.empty((n_draws, n_grid))
        for d in range(n_draws):
            log_comp_A = log_lik_A[d, :, None] + log_pi[None, :]  # (samples, grid)
            log_comp_B = log_lik_B[d, :, None] + log_1mpi[None, :]  # (samples, grid)
            log_likelihood_all[d] = np.logaddexp(log_comp_A, log_comp_B).sum(axis=0)

        results = []

        for i in range(n_samples):
            # Per-sample mixture log-likelihood: (draws, grid)
            log_mix_i = np.logaddexp(
                log_lik_A[:, i, None] + log_pi[None, :],
                log_lik_B[:, i, None] + log_1mpi[None, :],
            )

            # Remove sample i by subtracting its log term (valid: sum of logs, not log-sum-exp)
            log_mix_other = log_likelihood_all - log_mix_i  # (draws, grid)

            # Scenario 1:
            # sample i is known to be Group A.
            #
            # Its likelihood contribution is therefore simply:
            #
            #     pi * p(x_i | A)
            log_post_A = (
                log_mix_other + log_lik_A[:, i, None] + log_pi[None, :] + log_prior[None, :]
            )

            # Scenario 2:
            # sample i is known to be Group B.
            log_post_B = (
                log_mix_other + log_lik_B[:, i, None] + log_1mpi[None, :] + log_prior[None, :]
            )

            # Calculate the posterior median for each scenario.
            # We use the median over the mixture of posterior draws.
            def posterior_median(log_post: NpFloat) -> float:
                posterior = np.exp(log_post - np.max(log_post, axis=1, keepdims=True))

                posterior /= np.trapezoid(posterior, pi, axis=1)[:, None]

                cdf = np.zeros_like(posterior)
                cdf[:, 1:] = np.cumsum(
                    0.5 * (posterior[:, 1:] + posterior[:, :-1]) * np.diff(pi)[None, :],
                    axis=1,
                )

                cdf /= cdf[:, -1, None]

                # Combine posterior draws by averaging their CDFs.
                cdf_mean = cdf.mean(axis=0)

                return float(np.interp(0.5, cdf_mean, pi))

            median_if_A = posterior_median(log_post_A)
            median_if_B = posterior_median(log_post_B)

            results.append(
                {
                    "Sample": i,
                    f"Fraction A if {group1}": median_if_A,
                    f"Fraction A if {group2}": median_if_B,
                    "Classification Value": abs(median_if_A - median_if_B),
                }
            )

        df: pd.DataFrame = pd.DataFrame(results)
        df = df.sort_values("Classification Value", ascending=False)
        df = df.reset_index(drop=True)

        logger.info("Classification value:\n%s\n", df.round(3))

        if self.output_directory is not None:
            df.to_excel(self.output_directory / Path(f"{self.name}_classification_value.xlsx"))

        return df

    def feature_llr_diagnostics(self) -> pd.DataFrame:
        """Summarizes the feature-wise log-likelihood ratio (LLR) statistics.

        The LLR is defined as

            log p(x | Group B) - log p(x | Group A),

        so positive values favor Group B and negative values favor Group A. This is also known as
        the weight of evidence or likelihood log-odds. The LLR can be interpreted as the amount of
        evidence in favor of one group over the other, with larger absolute values indicating
        stronger evidence.

        Returns:
            DataFrame containing feature-wise summary statistics
        """
        # (draws, samples, groups, features)
        evidence_draws: NpFloat = self.feature_log_likelihood_ratio()
        consistency: NpFloat = (evidence_draws > 0).mean(axis=(0, 1))
        evidence_mean: NpFloat = evidence_draws.mean(axis=(0, 1))
        evidence_mag_mean: NpFloat = np.abs(evidence_draws).mean(axis=(0, 1))

        group1, group2 = self.fitted_model.coords["group"]

        diagnostics: dict[str, NpFloat] = {
            "Evidence": evidence_mean,
            "Evidence Magnitude": evidence_mag_mean,
        }

        if self.X_group_idx is not None:
            alignment_sign: NpFloat = np.where(self.X_group_idx == 1, 1.0, -1.0)[None, :, None]
            evidence_aligned_draws: NpFloat = evidence_draws * alignment_sign
            evidence_aligned_mean: NpFloat = evidence_aligned_draws.mean(axis=(0, 1))
            evidence_aligned_std: NpFloat = evidence_aligned_draws.std(axis=(0, 1))
            directional_stability: NpFloat = evidence_aligned_mean / (evidence_aligned_std + 1e-8)

            # Average raw LLR isolated by true group membership (which features characterize A vs B)
            group_a_mask: NpInt = self.X_group_idx == 0
            group_b_mask: NpInt = self.X_group_idx == 1

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

            diagnostics.update(
                {
                    "Evidence Aligned": evidence_aligned_mean,
                    "Directional Stability": directional_stability,
                    f"{group2} Vote Rate": consistency,
                    f"{group1} Conditional Evidence": profile_a,
                    f"{group2} Conditional Evidence": profile_b,
                }
            )

        df: pd.DataFrame = pd.DataFrame(diagnostics, index=self.fitted_model.coords["feature"])
        df.index.name = "Feature"
        if self.X_group_idx is not None:
            df = df.sort_values(by="Evidence Aligned", ascending=False)
        else:
            df = df.sort_values(by="Evidence", ascending=False)
        df = df.reset_index(drop=False)
        df.index.name = "Rank"

        logger.info("Feature-wise log-likelihood ratio diagnostics:\n%s\n", df.round(3))

        # Dump interpretation notes to log for user reference
        notes: str = (
            " - Feature: Name of the feature.\n"
            " - Evidence: Expected LLR across samples and draws, indicating which class the feature supports.\n"
            " - Evidence Magnitude: Expected magnitude of feature evidence (ignoring correctness).\n"
            " - Evidence Aligned: How well the feature supports the correct classification.\n"
            " - Directional Stability: How reliably the feature contributes in the right direction.\n"
            f" - {group2} Vote Rate: Proportion of times the feature favors {group2}.\n"
            f" - {group1} Conditional Evidence: Expected LLR for {group1} samples.\n"
            f" - {group2} Conditional Evidence: Expected LLR for {group2} samples.\n"
        )

        logger.info("Interpretation Notes:\n%s", notes)

        if self.output_directory is not None:
            df.to_excel(self.output_directory / Path(f"{self.name}_feature_llr_diagnostics.xlsx"))

        return df

    def sample_llr_diagnostics(
        self,
        *,
        ci: bool = False,
        index: pd.Index | None = None,
        sample_df_append: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Summarizes the sample-wise log-likelihood ratio (LLR) statistics.

        The LLR is defined as

            log p(x | Group B) - log p(x | Group A),

        so positive values favor Group B and negative values favor Group A. This is also known as
        the weight of evidence or likelihood log-odds. The LLR can be interpreted as the amount of
        evidence in favor of one group over the other, with larger absolute values indicating
        stronger evidence.

        Args:
            ci: Whether to compute confidence intervals. Defaults to ``False``.
            index: Optional index for the output DataFrame. Defaults to ``None`` to use the default
                index of sample numbers.
            sample_df_append: Optional DataFrame to append to the output. Defaults to ``None``.

        Returns:
            DataFrame containing sample-wise summary statistics
        """
        names: list[str] = ["Metric", "Stat", "Variable"]
        _, group2 = self.fitted_model.coords["group"]

        def add_to_cols_and_data(cols: list, data: list, metric: str, values: NpFloat) -> None:
            """Helper function to add columns and data for a given metric"""
            values_mean: NpFloat = values.mean(axis=0)
            for i, f in enumerate(self.fitted_model.coords["feature"]):
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

        # Raw data values
        for i, f in enumerate(self.fitted_model.coords["feature"]):
            cols.append(("Raw Data", "Value", f))
            data.append(self.X[:, i])

        # Local evidence
        # (draws, samples, groups, features)
        evidence_draws: NpFloat = self.feature_log_likelihood_ratio()
        add_to_cols_and_data(cols, data, "Evidence", evidence_draws)

        # Local evidence magnitude
        evidence_mag_draws: NpFloat = np.abs(evidence_draws)
        add_to_cols_and_data(cols, data, "Evidence Magnitude", evidence_mag_draws)

        consistency: NpFloat = (evidence_draws > 0).mean(axis=0)

        # Prediction and truth
        _, P_B = self.predict_type_posterior()  # (draws, samples)

        if self.X_group_idx is not None:
            # Aligned evidence (diagnostic evidence)
            # Positive = correct evidence; Negative = misleading evidence
            # Does this feature help classification? Best feature importance metric
            alignment_sign: NpFloat = np.where(self.X_group_idx == 1, 1.0, -1.0)[None, :, None]
            evidence_aligned_draws: NpFloat = evidence_draws * alignment_sign
            add_to_cols_and_data(cols, data, "Evidence Aligned", evidence_aligned_draws)

            # Stability score (signal-to-noise ratio of the correct evidence)
            # How reliably does this feature contribute in the right direction?
            evidence_aligned_mean: NpFloat = evidence_aligned_draws.mean(axis=0)
            evidence_aligned_std: NpFloat = np.std(evidence_aligned_draws, axis=0)
            directional_stability: NpFloat = evidence_aligned_mean / (evidence_aligned_std + 1e-8)

            for i, f in enumerate(self.fitted_model.coords["feature"]):
                cols.append(("Directional Stability", "Mean", f))
                data.append(directional_stability[:, i])

        for i, f in enumerate(self.fitted_model.coords["feature"]):
            cols.append((f"{group2} Vote Rate", "Mean", f))
            data.append(consistency[:, i])

        if self.X_group_idx is not None:
            P_true: NpFloat = np.where(
                self.X_group_idx[None, :] == 1, P_B, 1 - P_B
            )  # (draws, samples)
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
        data.append(np.array([self.fitted_model.coords["group"][i] for i in predicted_group]))

        if self.X_group_idx is not None:
            # Add true group
            cols.append(("Prediction", "True Class", "Name"))
            data.append(np.array([self.fitted_model.coords["group"][i] for i in self.X_group_idx]))

        df: pd.DataFrame = pd.DataFrame(data).T
        df.columns = pd.MultiIndex.from_tuples(cols, names=names)

        df = df.convert_dtypes()

        if index is not None:
            df.index = index
        else:
            df.index.name = "Sample Index"

        if sample_df_append is not None:
            sample_df_append.columns = pd.MultiIndex.from_tuples(
                [("Appended", "Metadata", col) for col in sample_df_append.columns],
                names=df.columns.names,
            )
            sample_df_append = sample_df_append.convert_dtypes()
            df = pd.concat([df, sample_df_append], axis=1, join="inner")

        if self.X_group_idx is not None:
            # Sort samples by P_True ascending so the worst misclassifications float to the top
            # This keeps the multi-index intact while shifting the row order.
            df = df.sort_values(
                by=("Prediction", "Mean", "True Class Probability"), ascending=True
            )
            # Re-index the sample row labels
            df = df.reset_index(drop=False)
            df.index.name = "Ranked Sample (Worst First)"

        # Append the global mean summary row at the very bottom after sorting
        # df.loc["Mean"] = df.mean(axis=0, numeric_only=True)

        if self.output_directory is not None:
            df.to_excel(self.output_directory / Path(f"{self.name}_sample_llr_diagnostics.xlsx"))

        return df

    def plot_sample_explanation_corner(
        self,
        df_samples: pd.Series | pd.DataFrame,
        annotation_column: tuple[str, str, str] | None = None,
        *,
        savefig_kwargs: dict[str, Any] | None = None,
    ) -> sns.PairGrid:
        """Plots the corner plot with sample overlay.

        Args:
            df_samples: DataFrame containing sample-wise log-likelihood ratio diagnostics for a
                single or several sample(s).
            annotation_column: Column name in ``df_samples`` to use for annotating the points in
                the plot. This should be a column in the DataFrame that contains the sample names
                or any other relevant information you want to display as annotations. Defaults to
                    ``None`` to not annotate the points.
            savefig_kwargs: Override keyword arguments for :func:`matplotlib.pyplot.savefig`.
                Defaults to ``None``.

        Returns:
            PairGrid object containing the corner plot of feature contributions for the samples
        """
        features = self.fitted_model.coords["feature"]

        pairgrid: sns.PairGrid = self.fitted_model.plot_group_corner(save_fig=False)

        # TODO: Could probably just make work for a pandas series and ditch support for dataframes
        # Hack to allow a single series to also work
        if isinstance(df_samples, pd.Series):
            series_name = df_samples.name
            df_samples = df_samples.to_frame().T
        else:
            series_name = None

        X: NpFloat = df_samples.loc[:, pd.IndexSlice["Raw Data", "Value", features]].to_numpy()

        labels: list[str] | None = None
        if annotation_column is not None:
            annotation_values = df_samples.loc[:, annotation_column]
            labels = [str(value) for value in annotation_values.tolist()]  # type: ignore is series

        # Off-diagonal: true multivariate centers
        for row in range(len(self.fitted_model.coords["feature"])):
            for col in range(row):
                ax: Axes = pairgrid.axes[row, col]

                ax.plot(
                    X[:, col],
                    X[:, row],
                    "x",
                    color="black",
                    markersize=8,
                    markeredgecolor="k",
                    label="_nolegend_",
                )

                if labels:
                    # annotations
                    for i in range(X.shape[0]):
                        ax.annotate(
                            labels[i],
                            (X[i, col], X[i, row]),
                            xytext=(7, 7),
                            textcoords="offset points",
                            fontsize=8,
                            color="black",
                            bbox=dict(boxstyle="round,pad=0.1", fc="yellow", alpha=0.5),
                        )

        for feature_idx, ax in enumerate(pairgrid.diag_axes):  # type: ignore
            for sample_idx in range(X.shape[0]):
                x = X[sample_idx, feature_idx]

                ax.plot(
                    x,
                    0,  # rug baseline
                    marker="|",
                    color="black",
                    markersize=10,
                    # alpha=0.8,
                    label="_nolegend_",
                )

                ymin, ymax = ax.get_ylim()

                if labels is not None:
                    ax.text(
                        x,
                        ymin + 0.05 * (ymax - ymin),
                        labels[sample_idx],
                        rotation=90,
                        fontsize=8,
                        va="bottom",
                        ha="center",
                        # alpha=0.8,
                        bbox=dict(boxstyle="round,pad=0.1", fc="yellow", alpha=0.5),
                    )

        title: str = ""
        if isinstance(series_name, str):
            title = series_name
        pairgrid.figure.suptitle(title)
        sns.move_legend(pairgrid, "upper right")

        save_figure(
            pairgrid.figure,
            f"{series_name}_corner",
            output_directory=self.output_directory,
            savefig_kwargs=savefig_kwargs,
        )

        return pairgrid

    def plot_sample_dashboard(
        self, sample: pd.Series, *, savefig_kwargs: dict[str, Any] | None = None
    ) -> Figure:
        """Plots a dashboard of feature contributions to the classification of a single sample.

        Args:
            sample: Series containing the sample data
            savefig_kwargs: Override keyword arguments for :func:`matplotlib.pyplot.savefig`.
                Defaults to ``None`` to use :obj:`SAVEFIG_KWARGS`.

        Returns:
            Figure
        """
        features: NpArray = self.fitted_model.coords["feature"]
        group1, group2 = self.fitted_model.coords["group"]
        y: NpInt = np.arange(len(features))

        # Extract data
        de_mean = sample.loc[("Evidence Aligned", "Mean", features)]
        de_low = sample.loc[("Evidence Aligned", "Low", features)]
        de_high = sample.loc[("Evidence Aligned", "High", features)]

        evidence = sample.loc[("Evidence", "Mean", features)].to_numpy()
        evidence_total = sample.loc[("Evidence", "Mean", "Total")]
        stability = sample.loc[("Directional Stability", "Mean", features)].to_numpy()

        p_true = sample.loc[("Prediction", "Mean", "True Class Probability")]
        p_true_low = sample.loc[("Prediction", "Low", "True Class Probability")]
        p_true_high = sample.loc[("Prediction", "High", "True Class Probability")]
        pred = sample.loc[("Prediction", "Predicted Class", "Name")]
        true = sample.loc[("Prediction", "True Class", "Name")]

        # Figure layout
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 3, width_ratios=[2.2, 1, 1])

        ax_forest = fig.add_subplot(gs[:, 0])  # big left panel
        ax_prob = fig.add_subplot(gs[0, 1])
        ax_stab = fig.add_subplot(gs[0, 2])
        ax_raw = fig.add_subplot(gs[1, 1])
        ax_align = fig.add_subplot(gs[1, 2])

        # Forest plot — main explanation
        ax_forest.hlines(y, de_low, de_high, color="gray", alpha=0.5, zorder=0)
        ax_forest.plot(de_mean, y, "o", color="black", zorder=2)
        ax_forest.axvline(0, linestyle="--", color="black", zorder=1)

        ax_forest.set_yticks(y)
        ax_forest.set_yticklabels(features)
        ax_forest.set_title("Aligned Evidence")
        ax_forest.set_xlabel("Aligned log-likelihood ratio\n(positive = correct evidence)")

        # Stability coloring
        sc = ax_forest.scatter(de_mean, y, c=stability, cmap="coolwarm", s=50, zorder=3)
        cbar = plt.colorbar(sc, ax=ax_forest, fraction=0.02)
        cbar.set_label("Directional Stability")

        ax_prob.axvspan(0, 0.5, color="red", alpha=0.05)
        ax_prob.axvspan(0.5, 1, color="blue", alpha=0.05)
        ax_prob.axvspan(p_true_low, p_true_high, color="black", alpha=0.2)
        ax_prob.axvline(p_true, color="black", linewidth=2, zorder=2)
        ax_prob.axvline(0.5, linestyle="--", color="black")

        ax_prob.set_xlim(0, 1)
        ax_prob.set_title(f"Classification and confidence\nTrue: {true} | Pred: {pred}")

        ax_prob.text(
            p_true,
            0.5,
            f"{p_true:.2f}",
            va="center",
            ha="left" if p_true < 0.5 else "right",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.8),
        )

        # Stability summary
        ax_stab.barh(features, stability, color="purple", alpha=0.7)
        ax_stab.axvline(0, linestyle="--", color="black")
        ax_stab.set_title("Directional Stability")
        ax_stab.set_xlabel("Stability of evidence direction")

        # Raw evidence
        ax_raw.barh(features, evidence, color="black", alpha=0.7)
        ax_raw.axvline(0, linestyle="--", color="black")
        ax_raw.set_title(f"Raw Evidence (Total: {evidence_total:.2f})")
        ax_raw.set_xlabel(f"Log-likelihood ratio\n(neg = {group1}, pos = {group2})")

        # Aligned evidence
        ax_align.barh(features, de_mean, color="teal", alpha=0.7)
        ax_align.axvline(0, linestyle="--", color="black")
        ax_align.set_title("Aligned Evidence")
        ax_align.set_xlabel("Aligned log-likelihood ratio\n(pos = correct)")

        fig.suptitle(sample.name)  # type: ignore

        fig.tight_layout()

        save_figure(
            fig,
            f"{sample.name}",
            output_directory=self.output_directory,
            savefig_kwargs=savefig_kwargs,
        )

        return fig

    def bayes_performance(self, *, prior_A: float | None = None) -> dict[str, float]:
        """Estimate Bayesian classification performance on held-out data.

        The class prior is taken from the known class fraction of the held-out dataset unless
        explicitly supplied.

        The observed classifier and posterior expected Bayes accuracy are both calculated from
        posterior-marginalised class probabilities.

        Returns:
            Dictionary containing observed accuracy, posterior expected Bayes accuracy,
            posterior expected Bayes error, classification headroom, and classification efficiency.
        """
        if self.X_group_idx is None:
            raise ValueError("X_group_idx is required to assess classification accuracy.")

        # Class prior. For held-out data, the prevalence of A and B is known.
        if prior_A is None:
            prior_A = float(np.mean(self.X_group_idx == 0))

        if not 0.0 < prior_A < 1.0:
            raise ValueError("prior_A must be between 0 and 1.")

        # Posterior class probabilities
        # (draws, samples)
        # Each posterior draw represents one possible set of parameters for the fitted
        # class-conditional distributions.
        P_A, P_B = self.predict_type_posterior(prior_A=prior_A)
        true_A = self.X_group_idx == 0

        # Posterior-marginalised class probabilities
        # We marginalise over uncertainty in the fitted model before making classification
        # decisions.
        mean_P_A = P_A.mean(axis=0)
        mean_P_B = P_B.mean(axis=0)

        # Actual classifier. MAP decision based on posterior-marginalised probabilities.
        predicted_A = mean_P_A > mean_P_B
        observed_accuracy = float(np.mean(predicted_A == true_A))

        # Posterior expected Bayes accuracy.
        # For each sample, the Bayes decision is the class with the higher posterior probability.
        # The probability that this decision is correct is therefore:
        #
        #     max(P(A | x, D), P(B | x, D))
        #
        # where D represents the training data and posterior uncertainty in the fitted model has
        # already been marginalised.
        bayes_probability_correct = np.maximum(mean_P_A, mean_P_B)

        # Average the expected probability of correctness over the held-out samples
        bayes_expected_accuracy = float(bayes_probability_correct.mean())

        # Derived quantities
        bayes_expected_error = 1.0 - bayes_expected_accuracy
        headroom = bayes_expected_accuracy - observed_accuracy
        efficiency = observed_accuracy / bayes_expected_accuracy

        # Logging
        logger.info("Held-out Group A fraction: %.3f", prior_A)
        logger.info("Observed classification accuracy: %.3f", observed_accuracy)
        logger.info("Posterior expected Bayes accuracy: %.3f", bayes_expected_accuracy)
        logger.info("Posterior expected Bayes error: %.3f", bayes_expected_error)
        logger.info("Estimated classification headroom: %.3f", headroom)
        logger.info("Classification efficiency: %.1f%%", 100.0 * efficiency)

        return {
            "prior_A": prior_A,
            "observed_accuracy": observed_accuracy,
            "bayes_expected_accuracy": bayes_expected_accuracy,
            "bayes_expected_error": bayes_expected_error,
            "headroom": headroom,
            "efficiency": efficiency,
        }

    def run_and_plot(
        self,
        *,
        index: pd.Index | None = None,
        sample_df_append: pd.DataFrame | None = None,
        savefig_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Runs the classifier and and generates plots.

        Args:
            index: Optional index for the sample diagnostics DataFrame. Defaults to ``None`` to use
                the default index of sample numbers.
            sample_df_append: Optional DataFrame to append to the sample diagnostics DataFrame.
                Defaults to ``None``.
            savefig_kwargs: Override keyword arguments for :func:`matplotlib.pyplot.savefig`.
                Defaults to ``None`` to use :obj:`SAVEFIG_KWARGS`.
        """
        logger.info("Running classifier on data for %s", self.name)

        self.plot_confusion_matrix(savefig_kwargs=savefig_kwargs)
        # self.bayes_performance()
        result = self.infer_group_fraction()

        self.plot_group_fraction_posterior(result, savefig_kwargs=savefig_kwargs)
        self.classification_value()
        self.feature_llr_diagnostics()

        df: pd.DataFrame = self.sample_llr_diagnostics(
            ci=True, index=index, sample_df_append=sample_df_append
        )

        # Loop over all test samples and generate a dashboard and corner plot for each sample to
        # explain the classification and feature contributions.
        for sample_id in range(len(df)):
            sample_series: pd.Series = df.iloc[sample_id]
            row_index = sample_series.name
            orig_index = sample_series.loc[("_index", "", "")]
            sample_name: str = sample_series.loc[("Appended", "Metadata", "Sample_name")]
            locality = sample_series.loc[("Appended", "Metadata", "Locality")]
            name: str = f"{row_index}-{orig_index}-{sample_name}-{locality}"
            sample_series.name = name

            self.plot_sample_explanation_corner(
                sample_series,
                annotation_column=("Appended", "Metadata", "Sample_name"),
                savefig_kwargs=savefig_kwargs,
            )
            self.plot_sample_dashboard(sample_series, savefig_kwargs=savefig_kwargs)

        logger.info("Evaluation complete for %s", self.name)
