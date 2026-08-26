# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bayesian classification and group-fraction inference based on hierarchical group models.

This module acts as stage 2 of a two-step generative classifier, taking a fitted
`HierarchicalGroupDifferenceModel` from stage 1 and using its posterior samples to evaluate
class-conditional likelihoods for classification and group-fraction inference.

In practice, a two-stage classifier may fail to accurately determine the group fraction due to
several key trade-offs:

1. Fixed Stage 1 Posterior: The model assumes the feature distribution parameters fitted in stage 1
   are fixed with respect to stage 2, preventing feedback from the target dataset during inference.
   While this separation decouples feature learning from target fraction estimation, it misses
   joint updating opportunities.
2. Naive Bayes Assumption: Conditional independence across features given the group label is
   typically assumed, which can severely underperform when features exhibit significant covariance.
3. Lack of Likelihood Curvature Under Overlap: In regimes with heavy class overlap, the mixture
   likelihood loses curvature. Consequently, posterior estimates for the group fraction collapse
   toward the prior mean/median (e.g., 0.5 under a uniform prior) and flatten out into prior
   dominance. This drives an asymmetric bias in the inferred group fraction, where the model
   overestimates the prevalence of the minor group when its true fraction is below 0.5, and
   underestimates it when it is above 0.5.

Overall, while a two-stage model is appealing for its modular architecture and computational
separation, it can perform significantly worse in practice than a one-step joint inference model,
particularly when data features are correlated or the group distributions strongly overlap.
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from pprint import pformat
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from scipy.special import expit, logsumexp
from scipy.stats import beta
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from bedroc.core.data_container import (
    HIGH_CI_PERCENTILE,
    LOW_CI_PERCENTILE,
    RANDOM_SEED,
    DataContainer,
)
from bedroc.core.plotting import save_figure
from bedroc.core.type_aliases import NpArray, NpFloat, NpInt
from bedroc.difference import DEFAULT_GROUP_COLORS
from bedroc.difference.group_difference import HierarchicalGroupDifferenceModel
from bedroc.difference.validation import validate_group_idx, validate_observation_data

logger: logging.Logger = logging.getLogger(__name__)


class GroupClassifierModel:
    """Bayesian classifier and group-fraction estimator built on a fitted group model.

    Wraps a fitted :class:`HierarchicalGroupDifferenceModel` to classify new observations and infer
    group prevalence. Posterior uncertainty in the fitted model is propagated through all
    predictions.

    Samples containing no finite observations do not contribute to the likelihood and therefore
    cannot be classified or contribute to group-fraction inference.

    Args:
        fitted_model: A :class:`HierarchicalGroupDifferenceModel` on which ``run_inference`` has
            already been called
        X: Data to classify (n_samples, n_features)
        X_sigma: Optional 1-sigma uncertainties for ``X`` (n_samples, n_features). Defaults to
            ``None``, in which case the model assumes that the observations are exact.
    """

    def __init__(
        self,
        fitted_model: HierarchicalGroupDifferenceModel,
        X: NpFloat,
        *,
        X_sigma: NpFloat | None = None,
    ):
        self.fitted_model: HierarchicalGroupDifferenceModel = fitted_model
        self.X, self.X_sigma = validate_observation_data(X, X_sigma=X_sigma)
        self._prediction_data: xr.DataTree | None = None

    @property
    def coords(self) -> dict[str, NpArray]:
        return self.fitted_model.coords

    @property
    def name(self) -> str:
        return self.fitted_model.name

    @property
    def prediction_data(self) -> xr.DataTree:
        if self._prediction_data is None:
            self._prediction_data = self._compute_prediction()
        return self._prediction_data

    def _compute_prediction(self) -> xr.DataTree:
        """Computes posterior likelihoods and the likelihood ratio for new data.

        Returns:
            Prediction data
        """
        group_0: NpInt = np.zeros(self.X.shape[0], dtype=int)
        group_1: NpInt = np.ones(self.X.shape[0], dtype=int)

        ll_0: xr.Dataset = self.fitted_model.compute_log_likelihood(
            self.X, X_sigma=self.X_sigma, group_idx=group_0
        ).rename({"log_likelihood": "log_likelihood_0"})

        ll_1: xr.Dataset = self.fitted_model.compute_log_likelihood(
            self.X, X_sigma=self.X_sigma, group_idx=group_1
        ).rename({"log_likelihood": "log_likelihood_1"})

        log_likelihood: xr.Dataset = xr.merge([ll_0, ll_1], compat="override")

        log_likelihood["log_likelihood_ratio"] = (
            log_likelihood["log_likelihood_1"] - log_likelihood["log_likelihood_0"]
        )

        return xr.DataTree.from_dict({"log_likelihood": log_likelihood})

    def predict_group_posterior(
        self, *, prior_0: float | ArrayLike = 0.5
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Computes posterior group probabilities for each sample containing at least one finite
        observation.

        Args:
            prior_0: Prior probability of group 0. May be a scalar, in which case the same prior is
                applied to every sample, or an array with shape ``(n_samples,)`` specifying a
                separate prior for each sample. The prior probability of group 1 is
                ``1 - prior_0``. Defaults to ``0.5``.

        Returns:
            Tuple containing posterior probabilities for group 0 and group 1. Both arrays have
            dimensions ``(chain, draw, sample_idx)`` and contain only samples with at least one
            finite observation.

        Raises:
            ValueError: If ``prior_0`` is not strictly between 0 and 1, or if an array prior does
            not have shape ``(n_samples,)``.
        """
        ll = self.prediction_data["log_likelihood"]
        llr = ll["log_likelihood_ratio"]

        sample_llr = llr.groupby(ll["sample_idx"]).sum(dim="observation")  # pyright: ignore[reportArgumentType]

        prior_0 = np.asarray(prior_0, dtype=float)

        if prior_0.ndim == 0:
            if not 0.0 < prior_0 < 1.0:
                raise ValueError("prior_0 must be strictly between 0 and 1.")

        elif prior_0.ndim == 1:
            if prior_0.shape != (self.X.shape[0],):
                raise ValueError(
                    f"Array prior_0 must have shape ({self.X.shape[0]},), got {prior_0.shape}."
                )

            if not np.all((prior_0 > 0.0) & (prior_0 < 1.0)):
                raise ValueError("All values of prior_0 must be strictly between 0 and 1.")

            # Align the sample-level prior with the sample_idx coordinate.
            prior_0 = xr.DataArray(
                prior_0,
                dims="sample_idx",
                coords={"sample_idx": np.arange(self.X.shape[0])},
            )

        else:
            raise ValueError("prior_0 must be a scalar or a 1-dimensional array.")

        # Log prior odds: log(P(Group 1) / P(Group 0)) = log(1 - prior_0) - log(prior_0)
        log_prior_odds = np.log1p(-prior_0) - np.log(prior_0)

        # Posterior probability for Group 1: P(Group 1 | X) = expit(LLR + log_prior_odds)
        p_1: xr.DataArray = expit(sample_llr + log_prior_odds)
        p_0: xr.DataArray = 1.0 - p_1

        return p_0, p_1

    def infer_group_fraction(
        self, *, prior_alpha: float = 1.0, prior_beta: float = 1.0, n_grid: int = 2001
    ) -> dict[str, Any]:
        """Infers the group fractions of the two groups in an unlabeled dataset.

        Asks the question, "What value of the common group fraction of group 0 (pi) best
        explains the entire unlabeled dataset?"

        The fraction of the first group is treated as an unknown group parameter and inferred
        jointly from all observations. The likelihood is a two-component mixture,

            p(x | pi) = pi * p(x | 0) + (1 - pi) * p(x | 1),

        where ``pi`` is the fraction of the dataset belonging to group 0.

        Posterior uncertainty in the learned group distributions is propagated by evaluating the
        mixture likelihood for every posterior draw of the fitted model.

        A Beta prior is used for the group-0 fraction:

            pi ~ Beta(prior_alpha, prior_beta)

        Args:
            prior_alpha: Alpha parameter of the Beta prior on the fraction of group 0. Defaults to
                ``1.0``.
            prior_beta: Beta parameter of the Beta prior on the fraction of group 0. Defaults to
                ``1.0``.
            n_grid: Number of points used to represent the posterior distribution of the group-0
                fraction. Defaults to ``2001``.

        Returns:
            Dictionary of results

        Raises:
            ValueError: If the Beta prior parameters or grid size are invalid
        """
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("prior_alpha and prior_beta must be > 0.")

        if n_grid < 2:
            raise ValueError("n_grid must be at least 2.")

        group_0, group_1 = self.fitted_model.coords["group"]

        logger.info(
            "Inferring group fractions for %d unlabeled samples using Beta(%g, %g) prior",
            self.X.shape[0],
            prior_alpha,
            prior_beta,
        )

        ll = self.prediction_data["log_likelihood"]

        # Combine chain and draw into a single posterior-draw dimension
        sample_log_lik_0: xr.DataArray = (
            ll["log_likelihood_0"]
            .groupby(ll["sample_idx"])  # pyright: ignore[reportArgumentType]
            .sum(dim="observation")
            .stack(draws=("chain", "draw"))
        )
        sample_log_lik_1: xr.DataArray = (
            ll["log_likelihood_1"]
            .groupby(ll["sample_idx"])  # pyright: ignore[reportArgumentType]
            .sum(dim="observation")
            .stack(draws=("chain", "draw"))
        )

        n_draws: int = sample_log_lik_0.sizes["draws"]

        # Grid over group fraction of group 0. Avoid exactly 0 and 1 because the logarithm of the
        # mixture weights would otherwise contain -inf
        eps: np.float64 = np.finfo(float).eps
        fraction_0_grid: NpFloat = np.linspace(eps, 1.0 - eps, n_grid)

        log_fraction_0: NpFloat = np.log(fraction_0_grid)
        log_fraction_1: NpFloat = np.log1p(-fraction_0_grid)

        # Beta prior
        # Normalization constant of the Beta distribution is irrelevant because we normalize the
        # posterior below
        log_prior: NpFloat = (prior_alpha - 1.0) * log_fraction_0 + (
            prior_beta - 1.0
        ) * log_fraction_1

        # Compute p(pi | X, theta) for every posterior draw theta.
        #
        # For each posterior draw:
        #
        #   p(X | pi, theta)
        #       = product_i [ pi p(x_i | 0, theta) + (1-pi) p(x_i | 1, theta) ]
        #
        # We work in log space for numerical stability, looping over draws to bound memory.
        #
        # Result:
        #   (draws, grid)

        # Intermediate (draws, samples, grid) arrays would be ~tens of GB; loop over draws instead
        log_likelihood_fraction: NpFloat = np.empty((n_draws, n_grid))

        for d in range(n_draws):
            log_lik_0: NpFloat = sample_log_lik_0.isel(draws=d).values  # (sample_idx,)
            log_lik_1: NpFloat = sample_log_lik_1.isel(draws=d).values  # (sample_idx,)

            # (sample_idx, 1) + (1, grid) -> (sample_idx, grid)
            log_comp_0: NpFloat = log_lik_0[:, None] + log_fraction_0[None, :]  # (samples, grid)
            log_comp_1: NpFloat = log_lik_1[:, None] + log_fraction_1[None, :]  # (samples, grid)
            log_likelihood_fraction[d] = np.logaddexp(log_comp_0, log_comp_1).sum(axis=0)

        # Marginalize over posterior uncertainty in the fitted model parameters
        #
        # The desired marginal posterior is
        #
        #   p(pi | X, D) \propto p(pi) * \int p(X | pi, theta) p(theta | D) dtheta
        #
        # We approximate the integral by averaging over posterior draws:
        #
        #   p(pi | X, D) \propto p(pi) * mean_d[p(X | pi, theta_d)]
        #
        # Importantly, we must NOT normalize each posterior draw separately before averaging
        #
        # logsumexp is used to perform the average in log space without numerical underflow

        log_marginal_likelihood: NpFloat = logsumexp(log_likelihood_fraction, axis=0) - np.log(
            n_draws
        )

        log_marginal_posterior: NpFloat = log_marginal_likelihood + log_prior

        # Normalize the marginal posterior for numerical stability
        log_marginal_posterior -= np.max(log_marginal_posterior)

        marginal_posterior: NpFloat = np.exp(log_marginal_posterior)

        # Normalize using trapezoidal integration
        marginal_posterior /= np.trapezoid(marginal_posterior, fraction_0_grid)

        # Construct the CDF of the marginal posterior
        marginal_cdf: NpFloat = np.zeros_like(marginal_posterior)
        marginal_cdf[1:] = np.cumsum(
            0.5 * (marginal_posterior[1:] + marginal_posterior[:-1]) * np.diff(fraction_0_grid)
        )
        marginal_cdf /= marginal_cdf[-1]

        # Calculate posterior mean and quantiles for group 0
        fraction_0_mean = np.trapezoid(fraction_0_grid * marginal_posterior, fraction_0_grid)

        fraction_0_lower, fraction_0_median, fraction_0_upper = np.interp(
            [LOW_CI_PERCENTILE / 100, 0.5, HIGH_CI_PERCENTILE / 100], marginal_cdf, fraction_0_grid
        )

        # Group 1 is complementary to group 0
        fraction_1_mean = 1.0 - fraction_0_mean
        fraction_1_median = 1.0 - fraction_0_median
        fraction_1_lower = 1.0 - fraction_0_upper
        fraction_1_upper = 1.0 - fraction_0_lower

        # Summarize
        summary: dict[str, dict] = {
            group_0: {
                "mean": fraction_0_mean,
                "median": fraction_0_median,
                "lower_95": fraction_0_lower,
                "upper_95": fraction_0_upper,
            },
            group_1: {
                "mean": fraction_1_mean,
                "median": fraction_1_median,
                "lower_95": fraction_1_lower,
                "upper_95": fraction_1_upper,
            },
        }

        logger.info(
            "Inferred %s fraction = %.3f (95%% CI: %.3f - %.3f)",
            group_0,
            summary[group_0]["mean"],
            summary[group_0]["lower_95"],
            summary[group_0]["upper_95"],
        )

        logger.info(
            "Inferred %s fraction = %.3f (95%% CI: %.3f - %.3f)",
            group_1,
            summary[group_1]["mean"],
            summary[group_1]["lower_95"],
            summary[group_1]["upper_95"],
        )

        output: dict[str, Any] = {
            "fraction_0_posterior": marginal_posterior,
            "summary": summary,
            "grid": fraction_0_grid,
        }

        return output

    def evaluate_group_fraction(
        self,
        X_group_idx: NpInt,
        *,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        n_grid: int = 2001,
    ) -> dict[str, Any]:
        """Compares inferred group fractions with known group labels.

        Args:
            X_group_idx: Known group index for each row of ``X``, which must be 0 or 1.
            prior_alpha: Alpha parameter of the Beta prior on the fraction of group 0. Defaults to
                ``1.0``.
            prior_beta: Beta parameter of the Beta prior on the fraction of group 0. Defaults to
                ``1.0``.
            n_grid: Number of points used to represent the posterior distribution of the group-0
                fraction. Defaults to ``2001``.

        Returns:
            Dictionary containing the inferred group fractions and the observed group fractions.
        """
        X_group_idx = validate_group_idx(X_group_idx, n_samples=self.X.shape[0])

        group_0, group_1 = self.fitted_model.coords["group"]

        result: dict[str, Any] = self.infer_group_fraction(
            prior_alpha=prior_alpha, prior_beta=prior_beta, n_grid=n_grid
        )

        # Compute the observed fraction of each group in the dataset
        observed_fraction_0: NpFloat = np.mean(X_group_idx == 0)
        observed_fraction_1: NpFloat = np.mean(X_group_idx == 1)

        # For analysis use group 0
        group0_dict = result["summary"][group_0]
        group0_dict["observed"] = observed_fraction_0
        group0_dict["error"] = group0_dict["mean"] - group0_dict["observed"]
        group0_dict["absolute_error"] = np.abs(group0_dict["error"])
        group0_dict["squared_error"] = np.square(group0_dict["error"])
        group0_dict["ci_width"] = group0_dict["upper_95"] - group0_dict["lower_95"]
        group0_dict["covered_95"] = (
            group0_dict["lower_95"] <= group0_dict["observed"] <= group0_dict["upper_95"]
        )

        result["summary"][group_1]["observed"] = observed_fraction_1

        logger.info(
            "Observed %s fraction = %.3f, %s fraction = %.3f",
            group_0,
            observed_fraction_0,
            group_1,
            observed_fraction_1,
        )

        return result

    def plot_confusion_matrix(
        self,
        *,
        X_group_idx: NpInt,
        prior_0: float | ArrayLike = 0.5,
        normalize: Literal["true", "pred", "all"] | None = "true",
    ) -> Figure:
        """Plots the confusion matrix and logs metrics.

        Args:
            X_group_idx: Array of group indices for the samples
            prior_0: Prior probability of the first group. The prior probability of the second
                group is taken as ``1 - prior_0``. Defaults to ``0.5``.
            normalize: Normalization mode for the confusion matrix. Defaults to ``true``.

        Returns:
            Figure
        """
        X_group_idx = validate_group_idx(X_group_idx, n_samples=self.X.shape[0])

        # (chain, draw, sample_idx)
        P_0, P_1 = self.predict_group_posterior(prior_0=prior_0)
        P_0: xr.DataArray = P_0.stack(draws=("chain", "draw"))
        P_1: xr.DataArray = P_1.stack(draws=("chain", "draw"))

        group_0, group_1 = self.fitted_model.coords["group"]

        # Compute posterior mean probability
        mean_prob_0: xr.DataArray = P_0.mean(dim="draws")
        mean_prob_1: xr.DataArray = P_1.mean(dim="draws")
        logger.debug("Posterior probability of %s = %s", group_0, mean_prob_0)
        logger.debug("Posterior probability of %s = %s", group_1, mean_prob_1)

        # Choose the most probable type Bayesian MAP classifier: standard Naive Bayes rule
        predicted_type: xr.DataArray = np.where(mean_prob_0 > mean_prob_1, group_0, group_1)
        groups: NpArray = np.array([group_0, group_1])
        true_labels: NpFloat = groups[X_group_idx]

        # Build confusion matrix
        cm: NpArray = confusion_matrix(
            true_labels, predicted_type, labels=[group_0, group_1], normalize=normalize
        )
        logger.debug("Confusion matrix = %s", cm)

        accuracy: float = float(accuracy_score(true_labels, predicted_type))
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predicted_type, labels=[group_0, group_1], zero_division="warn"
        )

        # Extract values for clarity
        precision_0, precision_1 = precision  # pyright: ignore[reportGeneralTypeIssues]
        recall_0, recall_1 = recall  # pyright: ignore[reportGeneralTypeIssues]
        f1_0, f1_1 = f1  # pyright: ignore[reportGeneralTypeIssues]

        logger.info("Training classification overall accuracy: %0.3f", accuracy)
        logger.info("Training classification precision (%s): %0.3f", group_0, precision_0)
        logger.info("Training classification recall (%s): %0.3f", group_0, recall_0)
        logger.info("Training classification f1 score (%s): %0.3f", group_0, f1_0)
        logger.info("Training classification precision (%s): %0.3f", group_1, precision_1)
        logger.info("Training classification recall (%s): %0.3f", group_1, recall_1)
        logger.info("Training classification f1 score (%s): %0.3f", group_1, f1_1)

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

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[group_0, group_1])
        disp.plot(cmap="Blues", values_format="0.2f")

        return disp.figure_

    def plot_group_fraction_posterior(
        self,
        *,
        X_group_idx: NpInt | None = None,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        n_grid: int = 2001,
        group_colors: Sequence[str] = DEFAULT_GROUP_COLORS,
    ) -> Figure:
        """Plot the posterior distribution of group fractions.

        The posterior is shown together with the beta prior and, where available, the observed
        group fraction.

        Args:
            X_group_idx: Optional known group index for each row of ``X``, which must be 0 or 1.
                If provided, the observed group fraction is shown on the plot. Defaults to
                ``None`` (no observed fraction).
            prior_alpha: Alpha parameter of the beta prior. Defaults to ``1.0``.
            prior_beta: Beta parameter of the beta prior. Defaults to ``1.0``.
            n_grid: Number of points used to represent the posterior distribution of the group-0
                fraction. Defaults to ``2001``.
            group_colors: Optional sequence of colors for the two groups. Defaults to
                :obj:`DEFAULT_GROUP_COLORS`.

        Returns:
            Matplotlib figure containing the posterior group-fraction plot
        """
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("prior_alpha and prior_beta must be > 0.")

        fig, ax = plt.subplots(figsize=(8, 5))

        if X_group_idx is not None:
            result: dict[str, Any] = self.evaluate_group_fraction(
                X_group_idx, prior_alpha=prior_alpha, prior_beta=prior_beta, n_grid=n_grid
            )
        else:
            result: dict[str, Any] = self.infer_group_fraction(
                prior_alpha=prior_alpha, prior_beta=prior_beta, n_grid=n_grid
            )

        grid: NpFloat = result["grid"]
        fraction_0_posterior: NpFloat = result["fraction_0_posterior"]
        # Same as 1 - fraction_0_posterior
        fraction_1_posterior: NpFloat = fraction_0_posterior[::-1]

        group_0, group_1 = self.coords["group"]
        color_0, color_1 = group_colors

        summary: dict[str, Any] = result.get("summary", {})

        def plot_posterior(
            label: str,
            grid: NpFloat,
            posterior: NpFloat,
            group_summary: dict[str, float],
            color: str,
        ) -> None:
            """Plot posterior with reduced opacity outside the 95% credible interval."""
            lower = group_summary["lower_95"]
            upper = group_summary["upper_95"]

            left = grid < lower
            central = (grid >= lower) & (grid <= upper)
            right = grid > upper

            ax.plot(grid[left], posterior[left], color=color, alpha=0.5, linewidth=2)
            ax.plot(grid[central], posterior[central], color=color, linewidth=2, label=label)
            ax.plot(grid[right], posterior[right], color=color, alpha=0.5, linewidth=2)

        plot_posterior(f"{group_0}", grid, fraction_0_posterior, summary[group_0], color_0)
        plot_posterior(f"{group_1}", grid, fraction_1_posterior, summary[group_1], color_1)

        # Beta prior
        prior_pdf: NpArray = beta.pdf(grid, prior_alpha, prior_beta)

        ax.plot(
            grid,
            prior_pdf,
            color="black",
            linestyle="--",
            linewidth=2,
            label=rf"{group_0} prior",  #: beta($\alpha={prior_alpha:g},\ \beta={prior_beta:g}$)",
        )

        if X_group_idx is not None:
            n_group_0: int = np.sum(X_group_idx == 0)
            n_group_1: int = np.sum(X_group_idx == 1)

            # Perfect-classification limit for group 0
            limiting_posterior_0: NpFloat = beta.pdf(
                grid, prior_alpha + n_group_0, prior_beta + n_group_1
            )

            ax.plot(
                grid,
                limiting_posterior_0,
                color="tab:blue",
                linestyle="--",
                linewidth=2,
                label="Perfect-classification limit",
            )

        # Observed fractions, if available
        observed_fraction_0: NpFloat | None = summary.get(group_0, {}).get("observed")
        observed_fraction_1: NpFloat | None = summary.get(group_1, {}).get("observed")

        if observed_fraction_0 is not None:
            ax.annotate(
                f"Obs\n{observed_fraction_0:.2f}",
                xy=(observed_fraction_0, 0.6),
                xytext=(observed_fraction_0, 1.8),
                ha="center",
                va="bottom",
                color=color_0,
                bbox=dict(
                    boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.9
                ),
                arrowprops=dict(arrowstyle="-|>", color=color_0, lw=1.5),
            )

        if observed_fraction_1 is not None:
            ax.annotate(
                f"Obs\n{observed_fraction_1:.2f}",
                xy=(observed_fraction_1, 0.8),
                xytext=(observed_fraction_1, 2.2),
                ha="center",
                va="bottom",
                color=color_1,
                bbox=dict(
                    boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.9
                ),
                arrowprops=dict(arrowstyle="-|>", color=color_1, lw=1.5),
            )

        def plot_credible_interval(summary: dict[str, float], y: float, color: str) -> None:
            """Plots the 95% credible interval and median for a group."""
            ax.errorbar(
                summary["median"],
                y,
                xerr=[
                    [summary["median"] - summary["lower_95"]],
                    [summary["upper_95"] - summary["median"]],
                ],
                fmt="o",
                color=color,
                capsize=4,
                capthick=2,
                elinewidth=2,
            )

        plot_credible_interval(summary[group_0], 0.4, color_0)
        plot_credible_interval(summary[group_1], 0.6, color_1)

        # Dummy line for legend entry for credible interval
        ax.plot([], [], color="black", linewidth=2, marker="o", label="95% CrI")

        ax.set(xlabel="Population fraction", ylabel="Density", xlim=(0, 1))
        ax.legend(loc="upper right")
        ax.margins(y=0.15)

        logger.info("Group-fraction summary statistics:\n%s", pformat(summary, indent=4))

        return fig


def pipeline(
    data: DataContainer,
    *,
    fitted_model: HierarchicalGroupDifferenceModel,
    group_data_column: str,
    output_directory: Path | None = None,
    random_seed: int | None = RANDOM_SEED,
    title_fontsize: str = "large",
) -> GroupClassifierModel:
    """Pipeline for Bayesian classification and group-fraction inference

    Args:
        data: The container holding the input data for the pipeline
        fitted_model: A fitted :class:`HierarchicalGroupDifferenceModel` on which ``run_inference``
            has already been called
        group_data_column: Column name in ``data.metadata`` that contains the group index for each
            sample
        output_directory: Directory to save output files. Defaults to ``None``, in which case no
            output files will be saved.
        random_seed: Optional random seed for reproducible results. Defaults to :obj:`RANDOM_SEED`.
        title_fontsize: Font size for plot titles. Defaults to ``large``.

    Returns:
        A :class:`GroupClassifierModel` instance containing the fitted model and prediction data
    """
    logger.info("Running group classifier pipeline for %s", data.name)

    if output_directory is not None:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory: %s", output_directory)
    else:
        logger.info("Output directory not specified. Figures will not be saved.")

    _, test = data.train_test_split(
        random_state=random_seed, stratify=data.metadata[group_data_column]
    )

    classifier: GroupClassifierModel = GroupClassifierModel(
        fitted_model, test.values_std.to_numpy(), X_sigma=test.uncertainties_std.to_numpy()
    )

    fig: Figure = classifier.plot_confusion_matrix(
        X_group_idx=test.metadata[group_data_column].to_numpy()
    )
    fig.suptitle(f"{data.name} Confusion Matrix", fontsize=title_fontsize)
    save_figure(fig, Path(f"{data.name}_confusion_matrix"), output_directory)

    fig = classifier.plot_group_fraction_posterior(
        X_group_idx=test.metadata[group_data_column].to_numpy()
    )
    fig.suptitle(f"{data.name} Group Fraction Posterior", fontsize=title_fontsize)
    save_figure(fig, Path(f"{data.name}_group_fraction_posterior"), output_directory)

    logger.info("Group classifier pipeline completed for %s", data.name)

    return classifier
