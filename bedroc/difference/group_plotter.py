# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Plotting of results for Bayesian hierarchical model for group-centric comparison of two groups"""

import logging
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from scipy.stats import beta
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from bedroc.core import save_figure
from bedroc.difference.group_classifier import GroupClassifierModel
from bedroc.type_aliases import NpArray, NpFloat, NpInt

logger: logging.Logger = logging.getLogger(__name__)


class GroupPlotter:
    """Plotter for visualizing the results of a Bayesian hierarchical model for group-centric
    comparison of two groups.

    Args:
        classifier_model: An instance of :class:`GroupClassifierModel` containing the classifier
            model.
        group_idx: Optional array of group indices if known. Defaults to ``None`` to mean unlabeled
            data.
        output_directory: Optional path to the directory where plots will be saved. Defaults to
            ``None``, meaing plots will not be saved.
    """

    def __init__(
        self,
        classifier_model: GroupClassifierModel,
        *,
        group_idx: NpInt | None = None,
        output_directory: Path | None = None,
    ) -> None:
        self.classifier_model: GroupClassifierModel = classifier_model

        self._group_idx: NpInt | None = group_idx
        self.output_directory: Path | None = output_directory

        if self.output_directory is not None:
            self.output_directory.mkdir(parents=True, exist_ok=True)

    @property
    def group_idx(self) -> NpInt | None:
        if self._group_idx is None:
            raise ValueError("Group indices have not been set. Please provide group_idx.")
        return self._group_idx

    def confusion_matrix(
        self,
        *,
        prior_0: float | ArrayLike = 0.5,
        normalize: Literal["true", "pred", "all"] | None = "true",
        savefig_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure, Axes]:
        """Plots the confusion matrix and logs metrics.

        Args:
            prior_0: Prior probability of the first group. The prior probability of the second
                group is taken as ``1 - prior_0``. Defaults to ``0.5``.
            normalize: Normalization mode for the confusion matrix. Defaults to ``None``.
            savefig_kwargs: Override keyword arguments for :func:`matplotlib.pyplot.savefig`.
                Defaults to ``None``.

        Returns:
            Figure, Axes
        """
        # (chain, draw, sample_idx)
        P_0, P_1 = self.classifier_model.predict_group_posterior(prior_0=prior_0)
        P_0: xr.DataArray = P_0.stack(draws=("chain", "draw"))
        P_1: xr.DataArray = P_1.stack(draws=("chain", "draw"))

        group_0, group_1 = self.classifier_model.fitted_model.coords["group"]

        # Compute posterior mean probability
        mean_prob_0: xr.DataArray = P_0.mean(dim="draws")
        mean_prob_1: xr.DataArray = P_1.mean(dim="draws")
        logger.debug("Posterior probability of %s = %s", group_0, mean_prob_0)
        logger.debug("Posterior probability of %s = %s", group_1, mean_prob_1)

        # Choose the most probable type Bayesian MAP classifier: standard Naive Bayes rule
        predicted_type: xr.DataArray = np.where(mean_prob_0 > mean_prob_1, group_0, group_1)
        groups: NpArray = np.array([group_0, group_1])
        true_labels: NpFloat = groups[self.group_idx]

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

        save_figure(
            disp.figure_,
            "confusion_matrix",
            output_directory=self.output_directory,
            savefig_kwargs=savefig_kwargs,
        )

        return disp.figure_, disp.ax_

    def plot_group_fraction_posterior(
        self,
        *,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        n_grid: int = 2001,
        savefig_kwargs: dict[str, Any] | None = None,
    ) -> Axes:
        """Plot the posterior distribution of group fractions.

        The posterior is shown together with the beta prior and, where available, the observed
        group fraction.

        Args:
            prior_alpha: Alpha parameter of the beta prior. Defaults to ``1.0``.
            prior_beta: Beta parameter of the beta prior. Defaults to ``1.0``.
            n_grid: Number of points used to represent the posterior distribution of the group-0
                fraction. Defaults to ``2001``.
            savefig_kwargs: Override keyword arguments for :func:`matplotlib.pyplot.savefig`.
                Defaults to ``None``.

        Returns:
            Matplotlib axes containing the posterior group-fraction plot
        """
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("prior_alpha and prior_beta must be > 0.")

        fig, ax = plt.subplots(figsize=(8, 5))

        if self.group_idx is not None:
            result: dict[str, Any] = self.classifier_model.evaluate_group_fraction(
                self.group_idx, prior_alpha=prior_alpha, prior_beta=prior_beta, n_grid=n_grid
            )
        else:
            result: dict[str, Any] = self.classifier_model.infer_group_fraction(
                prior_alpha=prior_alpha, prior_beta=prior_beta, n_grid=n_grid
            )

        grid: NpFloat = result["grid"]
        fraction_0_posterior: NpFloat = result["fraction_0_posterior"]
        # Same as 1 - fraction_0_posterior
        fraction_1_posterior: NpFloat = fraction_0_posterior[::-1]

        group_0, group_1 = self.classifier_model.fitted_model.coords["group"]

        color_0: str = "tab:blue"
        color_1: str = "tab:orange"

        summary: dict[str, dict[str, float]] = result["summary"]

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

        if self.group_idx is not None:
            n_group_0 = np.sum(self.group_idx == 0)
            n_group_1 = np.sum(self.group_idx == 1)

            limiting_posterior: NpArray = beta.pdf(
                grid, prior_alpha + n_group_0, prior_beta + n_group_1
            )

            ax.plot(
                grid,
                limiting_posterior,
                color="black",
                linestyle=":",
                linewidth=2,
                label="Perfect classification",
            )

        # Observed fractions, if available
        observed_fraction_0: NpFloat | None = result.get("observed_fraction_0")
        observed_fraction_1: NpFloat | None = result.get("observed_fraction_1")

        coords: tuple[str, str] = ("data", "axes fraction")

        if observed_fraction_0 is not None:
            ax.annotate(
                f"Observed\n{observed_fraction_0:.2f}",
                xy=(observed_fraction_0, 0.17),
                xycoords=coords,
                xytext=(observed_fraction_0, 0.27),
                textcoords=coords,
                ha="center",
                va="bottom",
                color=color_0,
                arrowprops=dict(arrowstyle="-|>", color=color_0, lw=1.5),
            )

        if observed_fraction_1 is not None:
            ax.annotate(
                f"Observed\n{observed_fraction_1:.2f}",
                xy=(observed_fraction_1, 0.17),
                xycoords=coords,
                xytext=(observed_fraction_1, 0.27),
                textcoords=coords,
                ha="center",
                va="bottom",
                color=color_1,
                arrowprops=dict(arrowstyle="-|>", color=color_1, lw=1.5),
            )

        def plot_credible_interval(summary: dict[str, float], y: float, color: str) -> None:
            """Plots the 95% credible interval and median for a group."""
            ax.plot(
                [summary["lower_95"], summary["upper_95"]],
                [y, y],
                color=color,
                linewidth=2,
                solid_capstyle="round",
            )
            ax.plot(summary["median"], y, color=color, marker="o", markersize=6)

        plot_credible_interval(summary[group_0], 0.5, color_0)
        plot_credible_interval(summary[group_1], 0.5, color_1)

        # Dummy line for legend entry for credible interval
        ax.plot([], [], color="black", linewidth=2, marker="o", label="95% CrI")

        ax.set(
            # Titles not required for publications
            # title="Posterior Group Fractions",
            xlabel="Population fraction",
            ylabel="Density",
            xlim=(0, 1),
        )
        ax.legend(loc="upper left")
        ax.margins(y=0.15)

        save_figure(
            fig,
            "group_fraction_posterior",
            output_directory=self.output_directory,
            savefig_kwargs=savefig_kwargs,
        )

        return ax
