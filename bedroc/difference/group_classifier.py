# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bayesian classification and group-fraction inference based on hierarchical group models."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from scipy.special import expit, logsumexp

from bedroc.core.data_container import HIGH_CI_PERCENTILE, LOW_CI_PERCENTILE
from bedroc.core.type_aliases import NpFloat, NpInt
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
        output_directory: Optional path to save generated data. Defaults to ``None`` (no saving).
    """

    def __init__(
        self,
        fitted_model: HierarchicalGroupDifferenceModel,
        X: NpFloat,
        *,
        X_sigma: NpFloat | None = None,
        output_directory: Path | None = None,
    ):
        self.fitted_model: HierarchicalGroupDifferenceModel = fitted_model
        self.X, self.X_sigma = validate_observation_data(X, X_sigma=X_sigma)

        self.output_directory: Path | None = output_directory

        if self.output_directory is not None:
            self.output_directory.mkdir(parents=True, exist_ok=True)

        self._prediction_data: xr.DataTree | None = None

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
            self.X,
            X_sigma=self.X_sigma,
            group_idx=group_0,
        ).rename({"log_likelihood": "log_likelihood_0"})

        ll_1: xr.Dataset = self.fitted_model.compute_log_likelihood(
            self.X,
            X_sigma=self.X_sigma,
            group_idx=group_1,
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
                ValueError: If ``prior_0`` is not strictly between 0 and 1, or if an array prior
                does not have shape ``(n_samples,)``.
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

        log_prior_odds = np.log1p(-prior_0) - np.log(prior_0)

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
        # mixture weights would otherwise contain -inf.
        eps: np.float64 = np.finfo(float).eps
        fraction_0_grid: NpFloat = np.linspace(eps, 1.0 - eps, n_grid)

        log_fraction_0: NpFloat = np.log(fraction_0_grid)
        log_fraction_1: NpFloat = np.log1p(-fraction_0_grid)

        # Beta prior
        # Normalization constant of the Beta distribution is irrelevant because we normalize the
        # posterior below.
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

        # Intermediate (draws, samples, grid) arrays would be ~tens of GB; loop over draws instead.
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
        # Importantly, we must NOT normalize each posterior draw separately before averaging.
        #
        # logsumexp is used to perform the average in log space without numerical underflow.

        log_marginal_likelihood: NpFloat = logsumexp(log_likelihood_fraction, axis=0) - np.log(
            n_draws
        )

        log_marginal_posterior: NpFloat = log_marginal_likelihood + log_prior

        # Normalize the marginal posterior for numerical stability.
        log_marginal_posterior -= np.max(log_marginal_posterior)

        marginal_posterior: NpFloat = np.exp(log_marginal_posterior)

        # Normalize using trapezoidal integration.
        marginal_posterior /= np.trapezoid(marginal_posterior, fraction_0_grid)

        # Construct the CDF of the marginal posterior
        marginal_cdf: NpFloat = np.zeros_like(marginal_posterior)
        marginal_cdf[1:] = np.cumsum(
            0.5 * (marginal_posterior[1:] + marginal_posterior[:-1]) * np.diff(fraction_0_grid)
        )
        marginal_cdf /= marginal_cdf[-1]

        # Calculate posterior mean and quantiles for group 0.
        fraction_0_mean = np.trapezoid(fraction_0_grid * marginal_posterior, fraction_0_grid)

        fraction_0_lower, fraction_0_median, fraction_0_upper = np.interp(
            [LOW_CI_PERCENTILE / 100, 0.5, HIGH_CI_PERCENTILE / 100], marginal_cdf, fraction_0_grid
        )

        # Group 1 is complementary to group 0.
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
