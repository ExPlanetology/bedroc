# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bayesian classification and category-fraction inference based on hierarchical category models.

This module acts as stage 2 of a two-step generative classifier, taking a fitted
`StandardDifferenceModel` from stage 1 and using its posterior samples to evaluate
class-conditional likelihoods for classification and category-fraction inference.

In practice, a two-stage classifier may fail to accurately determine the category fraction due to
several key trade-offs:

1. Fixed Stage 1 Posterior: The model assumes the feature distribution parameters fitted in stage 1
   are fixed with respect to stage 2, preventing feedback from the target dataset during inference.
   While this separation decouples feature learning from target fraction estimation, it misses
   joint updating opportunities.
2. Naive Bayes Assumption: Conditional independence across features given the category label is
   typically assumed, which can severely underperform when features exhibit significant covariance.
3. Lack of Likelihood Curvature Under Overlap: In regimes with heavy class overlap, the mixture
   likelihood loses curvature. Consequently, posterior estimates for the category fraction collapse
   toward the prior mean/median (e.g., 0.5 under a uniform prior) and flatten out into prior
   dominance. This drives an asymmetric bias in the inferred category fraction, where the model
   overestimates the prevalence of the minor category when its true fraction is below 0.5, and
   underestimates it when it is above 0.5.

Overall, while a two-stage model is appealing for its modular architecture and computational
separation, it can perform significantly worse in practice than a one-step joint inference model,
particularly when data features are correlated or the category distributions strongly overlap.
"""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from scipy.special import expit
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from typing_extensions import override

from bedroc import RANDOM_SEED
from bedroc.core.data_container import DataContainer
from bedroc.core.plotting import save_figure
from bedroc.core.type_aliases import NpArray, NpFloat, NpInt
from bedroc.core.utils import SummaryStatistics
from bedroc.difference import DEFAULT_CATEGORY_COLORS
from bedroc.difference.group_base import CategoryClassifierProtocol
from bedroc.difference.models.standard_difference import StandardDifferenceModel
from bedroc.difference.plotting import plot_group_fraction_posterior
from bedroc.difference.validation import validate_category_idx, validate_observation_data

logger: logging.Logger = logging.getLogger(__name__)


class StandardClassifierModel(CategoryClassifierProtocol):
    """Bayesian classifier and category-fraction estimator built on a fitted category model.

    Wraps a fitted :class:`StandardDifferenceModel` to classify new observations and infer
    category prevalence. Posterior uncertainty in the fitted model is propagated through all
    predictions.

    Samples containing no finite observations do not contribute to the likelihood and therefore
    cannot be classified or contribute to category-fraction inference.

    Args:
        fitted_model: A :class:`StandardDifferenceModel` on which ``run_inference`` has already
            been called.
        X: Data to classify (n_samples, n_features)
        X_sigma: Optional 1-sigma uncertainties for ``X`` (n_samples, n_features). Defaults to
            ``None``, in which case the model assumes that the observations are exact.
    """

    def __init__(
        self, fitted_model: StandardDifferenceModel, X: NpFloat, *, X_sigma: NpFloat | None = None
    ):
        self.fitted_model: StandardDifferenceModel = fitted_model
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

    @override
    def pi_0_samples(
        self, *, prior_alpha: float = 1.0, prior_beta: float = 1.0, n_grid: int = 2001
    ) -> NpFloat:
        """Draws one pi_0 posterior sample for each model posterior draw theta^(s).

        Args:
            prior_alpha: Alpha parameter of the Beta prior on the fraction of category 0. Defaults
                to ``1.0``.
            prior_beta: Beta parameter of the Beta prior on the fraction of category 0. Defaults
                to ``1.0``.
            n_grid: Number of points used to represent the posterior distribution of the
                category-0 fraction. Defaults to ``2001``.

        Returns:
            Posterior samples of the fraction of samples belonging to category 0 in the unlabeled
            dataset.
        """
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("prior_alpha and prior_beta must be > 0.")

        ll = self.prediction_data["log_likelihood"]

        # Sum log likelihoods across features per sample under Naive Bayes conditional independence
        # and stack chain + draw into a single flattened posterior draw dimension (draws)
        sample_log_lik_0: xr.DataArray = (
            ll["log_likelihood_0"].sum(dim="feature").stack(draws=("chain", "draw"))
        )
        sample_log_lik_1: xr.DataArray = (
            ll["log_likelihood_1"].sum(dim="feature").stack(draws=("chain", "draw"))
        )

        n_draws: int = sample_log_lik_0.sizes["draws"]

        eps: float = np.finfo(float).eps
        grid: NpFloat = np.linspace(eps, 1.0 - eps, n_grid)
        log_grid_0: NpFloat = np.log(grid)
        log_grid_1: NpFloat = np.log1p(-grid)

        # Log Prior
        log_prior: NpFloat = (prior_alpha - 1.0) * log_grid_0 + (prior_beta - 1.0) * log_grid_1

        # Calculate per-draw PMFs
        log_post_per_draw = np.empty((n_draws, n_grid))

        for d in range(n_draws):
            log_lik_0 = sample_log_lik_0.isel(draws=d).values  # (observation,)
            log_lik_1 = sample_log_lik_1.isel(draws=d).values  # (observation,)

            # (observation, 1) + (1, grid) -> (observation, grid)
            log_comp_0 = log_lik_0[:, None] + log_grid_0[None, :]
            log_comp_1 = log_lik_1[:, None] + log_grid_1[None, :]
            log_post_per_draw[d] = np.logaddexp(log_comp_0, log_comp_1).sum(axis=0) + log_prior

        # Normalize PMF per draw
        log_post_per_draw -= np.max(log_post_per_draw, axis=1, keepdims=True)
        pmfs = np.exp(log_post_per_draw)
        pmfs /= pmfs.sum(axis=1, keepdims=True)

        # Sample 1 value of pi_0 per draw via CDF inversion
        cdfs = np.cumsum(pmfs, axis=1)
        u = np.random.uniform(0, 1, size=(n_draws, 1))
        indices = np.clip((cdfs < u).sum(axis=1), 0, n_grid - 1)

        pi_0_samples: NpFloat = grid[indices]

        SummaryStatistics(pi_0_samples).log_summary("pi_0 posterior summary")

        return pi_0_samples

    def _compute_prediction(self) -> xr.DataTree:
        """Computes posterior likelihoods and the likelihood ratio for new data.

        Returns:
            Prediction data
        """
        category_0_idx: NpInt = np.zeros(self.X.shape[0], dtype=int)
        category_1_idx: NpInt = np.ones(self.X.shape[0], dtype=int)

        ll_0: xr.Dataset = self.fitted_model.compute_log_likelihood(
            self.X, X_sigma=self.X_sigma, category_idx=category_0_idx
        ).rename({"log_likelihood": "log_likelihood_0"})

        ll_1: xr.Dataset = self.fitted_model.compute_log_likelihood(
            self.X, X_sigma=self.X_sigma, category_idx=category_1_idx
        ).rename({"log_likelihood": "log_likelihood_1"})

        log_likelihood: xr.Dataset = xr.merge([ll_0, ll_1], compat="override")

        log_likelihood["log_likelihood_ratio"] = (
            log_likelihood["log_likelihood_1"] - log_likelihood["log_likelihood_0"]
        )

        return xr.DataTree.from_dict({"log_likelihood": log_likelihood})

    def predict_category_posterior(
        self, *, prior_0: float | ArrayLike = 0.5
    ) -> tuple[xr.DataArray, xr.DataArray]:
        r"""Computes posterior category probabilities for each sample containing at least one
        finite observation.

        .. note::
            Summing log-likelihood ratios across features assumes conditional independence
            between features given the category label (Naive Bayes assumption):

            .. math::
                \\log p(X_i \\mid C) = \\sum_{j=1}^{n_{\\text{features}}} \\log p(X_{i,j} \\mid C)

        Args:
            prior_0: Prior probability of category 0. May be a scalar, in which case the same
                prior is applied to every sample, or an array with shape ``(n_samples,)``
                specifying a separate prior for each sample. The prior probability of category 1
                is ``1 - prior_0``. Defaults to ``0.5``.

        Returns:
            Tuple containing posterior probabilities for category 0 and category 1. Both arrays
            have dimensions ``(chain, draw, observation)`` and contain only samples with at least
            one finite observation.

        Raises:
            ValueError: If ``prior_0`` is not strictly between 0 and 1, or if an array prior does
            not have shape ``(n_samples,)``.
        """
        ll = self.prediction_data["log_likelihood"]
        llr = ll["log_likelihood_ratio"]

        # Sum LLR across features; dimensions become (chain, draw, observation)
        sample_llr = llr.sum(dim="feature")

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
                dims="observation",
                coords={"observation": np.arange(self.X.shape[0])},
            )
        else:
            raise ValueError("prior_0 must be a scalar or a 1-dimensional array.")

        # Log prior odds: log(P(Category 1) / P(Category 0)) = log(1 - prior_0) - log(prior_0)
        log_prior_odds = np.log1p(-prior_0) - np.log(prior_0)

        # Posterior probability for Category 1: P(Category 1 | X) = expit(LLR + log_prior_odds)
        p_1: xr.DataArray = expit(sample_llr + log_prior_odds)
        p_0: xr.DataArray = 1.0 - p_1

        return p_0, p_1

    def plot_confusion_matrix(
        self,
        *,
        X_category_idx: NpInt,
        prior_0: float | ArrayLike = 0.5,
        normalize: Literal["true", "pred", "all"] | None = "true",
    ) -> Figure:
        """Plots the confusion matrix and logs metrics.

        Args:
            X_category_idx: Array of category indices for the samples
            prior_0: Prior probability of the first category. The prior probability of the second
                category is taken as ``1 - prior_0``. Defaults to ``0.5``.
            normalize: Normalization mode for the confusion matrix. Defaults to ``true``.

        Returns:
            Figure
        """
        X_category_idx = validate_category_idx(X_category_idx, n_samples=self.X.shape[0])

        # (chain, draw, sample_idx)
        P_0, P_1 = self.predict_category_posterior(prior_0=prior_0)
        P_0: xr.DataArray = P_0.stack(draws=("chain", "draw"))
        P_1: xr.DataArray = P_1.stack(draws=("chain", "draw"))

        category_0, category_1 = self.fitted_model.coords["category"]

        # Compute posterior mean probability
        mean_prob_0: xr.DataArray = P_0.mean(dim="draws")
        mean_prob_1: xr.DataArray = P_1.mean(dim="draws")
        logger.debug("Posterior probability of %s = %s", category_0, mean_prob_0)
        logger.debug("Posterior probability of %s = %s", category_1, mean_prob_1)

        # Choose the most probable type Bayesian MAP classifier: standard Naive Bayes rule
        predicted_type: xr.DataArray = np.where(mean_prob_0 > mean_prob_1, category_0, category_1)
        categories: NpArray = np.array([category_0, category_1])
        true_labels: NpFloat = categories[X_category_idx]

        # Build confusion matrix
        cm: NpArray = confusion_matrix(
            true_labels, predicted_type, labels=[category_0, category_1], normalize=normalize
        )
        logger.debug("Confusion matrix = %s", cm)

        accuracy: float = float(accuracy_score(true_labels, predicted_type))
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predicted_type, labels=[category_0, category_1], zero_division="warn"
        )

        # Extract values for clarity
        precision_0, precision_1 = precision  # pyright: ignore[reportGeneralTypeIssues]
        recall_0, recall_1 = recall  # pyright: ignore[reportGeneralTypeIssues]
        f1_0, f1_1 = f1  # pyright: ignore[reportGeneralTypeIssues]

        logger.info("Training classification overall accuracy: %0.3f", accuracy)
        logger.info("Training classification precision (%s): %0.3f", category_0, precision_0)
        logger.info("Training classification recall (%s): %0.3f", category_0, recall_0)
        logger.info("Training classification f1 score (%s): %0.3f", category_0, f1_0)
        logger.info("Training classification precision (%s): %0.3f", category_1, precision_1)
        logger.info("Training classification recall (%s): %0.3f", category_1, recall_1)
        logger.info("Training classification f1 score (%s): %0.3f", category_1, f1_1)

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

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[category_0, category_1])
        disp.plot(cmap="Blues", values_format="0.2f")

        return disp.figure_

    def plot_group_fraction_posterior(
        self,
        bins: int = 50,
        n_grid: int = 2001,
        category_colors: tuple[str, str] = DEFAULT_CATEGORY_COLORS,
        category_counts: pd.Series | None = None,
        ax: Axes | None = None,
    ) -> Axes:
        """Plots the posterior distribution of the fraction of samples belonging to category 0.

        Args:
            bins: Number of bins for the histogram. Defaults to ``50``.
            n_grid: Number of grid points for the prior and perfect-classification limit. Defaults to
                ``2001``.
            category_colors: Colors for the two categories. Defaults to
                :data:`~bedroc.difference.DEFAULT_CATEGORY_COLORS`.
            category_counts: Known counts for the two categories. If ``None``, the observed
                fractions are not plotted. Defaults to ``None``.
            ax: Matplotlib axes on which to plot. If ``None``, a new figure and axes are created.

        Returns:
            Matplotlib axes containing the posterior group-fraction plot
        """
        return plot_group_fraction_posterior(
            self.pi_0_samples(),
            prior_alpha=1,
            prior_beta=1,
            bins=bins,
            n_grid=n_grid,
            category_names=self.coords["category"],
            category_colors=category_colors,
            category_counts=category_counts,
            ax=ax,
        )


def pipeline(
    data: DataContainer,
    *,
    fitted_model: StandardDifferenceModel,
    output_directory: Path | None = None,
    random_seed: int | None = RANDOM_SEED,
) -> StandardClassifierModel:
    """Pipeline for Bayesian classification and category-fraction inference

    Args:
        data: The container holding the input data for the pipeline
        fitted_model: A fitted :class:`StandardDifferenceModel` on which ``run_inference`` has
            already been called
        output_directory: Directory to save output files. Defaults to ``None``, in which case no
            output files will be saved.
        random_seed: Optional random seed for reproducible results. Defaults to
            :data:`~bedroc.RANDOM_SEED`.

    Returns:
        A :class:`StandardClassifierModel` instance containing the fitted model and prediction
        data
    """
    logger.info("Running standard category classifier pipeline for %s", data.name)

    if output_directory is not None:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory: %s", output_directory)
    else:
        logger.info("Output directory not specified. Figures will not be saved.")

    _, test = data.train_test_split(random_state=random_seed)

    classifier: StandardClassifierModel = StandardClassifierModel(
        fitted_model, test.values_std.to_numpy(), X_sigma=test.uncertainties_std.to_numpy()
    )

    fig: Figure = classifier.plot_confusion_matrix(X_category_idx=test.category_codes.to_numpy())  # pyright: ignore[reportOptionalMemberAccess]
    fig.suptitle(f"{data.name} Confusion Matrix")
    save_figure(fig, Path(f"{data.name}_confusion_matrix"), output_directory)

    ax: Axes = classifier.plot_group_fraction_posterior(category_counts=test.category_counts)
    fig = ax.get_figure()  # pyright: ignore[reportAssignmentType]
    save_figure(fig, Path(f"{data.name}_group_fraction_posterior"), output_directory)

    logger.info("Standard category classifier pipeline completed for %s", data.name)

    return classifier
