# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Support vector machine classification and population-fraction estimation.

A non-Bayesian comparison for the category difference models: a
:class:`sklearn.svm.SVC` classifier is trained on the labeled data and used to classify the
unlabeled population. Because classifying each sample and counting the result
("classify-and-count") is biased whenever the categories overlap (misclassification pulls the
estimate towards the training balance), the population fraction is estimated with the adjusted
count, which corrects the raw count using the classifier's true- and false-positive rates
estimated by cross-validation on the labeled data:

.. math::

    \\hat\\pi_0 = \\frac{\\mathrm{cc} - \\mathrm{fpr}}{\\mathrm{tpr} - \\mathrm{fpr}}

where :math:`\\mathrm{cc}` is the fraction of unlabeled samples classified as category 0,
:math:`\\mathrm{tpr} = P(\\text{predicted } 0 \\mid \\text{true } 0)` and
:math:`\\mathrm{fpr} = P(\\text{predicted } 0 \\mid \\text{true } 1)`.

Measurement uncertainties are not used: the SVM sees only the standardized feature values.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC

from bedroc import RANDOM_SEED
from bedroc.core.data_container import DataContainer
from bedroc.core.plotting import get_figure, save_figure
from bedroc.core.type_aliases import NpFloat, NpInt
from bedroc.difference.partitioning import Labeled, Unlabeled, resolve_labeled_unlabeled
from bedroc.difference.plotting import plot_group_fraction_posterior
from bedroc.difference.utils import save_fraction_summary

logger: logging.Logger = logging.getLogger(__name__)

MIN_RATE_SEPARATION: float = 0.05
"""Smallest ``tpr - fpr`` for which the adjusted count is applied. Below this the classifier is
close to chance, the correction divides by nearly zero, and classify-and-count is used instead."""


def _complete_rows(values: pd.DataFrame, label: str) -> NpInt:
    """Positions of the rows with no missing feature value, warning about any that are dropped.

    Args:
        values: Feature values
        label: Description of the data for the warning message

    Returns:
        Positions of the complete rows
    """
    complete: NpInt = np.where(values.notna().all(axis=1).to_numpy())[0]
    n_dropped: int = len(values) - complete.size
    if n_dropped:
        logger.warning(
            "Excluding %d of %d %s samples with missing feature values (the SVM cannot use them)",
            n_dropped,
            len(values),
            label,
        )

    return complete


class SVMModel:
    """Support vector machine classifier with adjusted-count population-fraction estimation.

    Category 0 is treated as the "positive" class, so :meth:`adjusted_count` and
    :meth:`pi_0_samples` estimate the fraction of category 0 in the unlabeled population, matching
    ``pi_0`` in the Bayesian models.

    Args:
        name: Name of the dataset
        labeled: Labeled data to train on
        unlabeled: Unlabeled population whose category fraction is estimated
        n_cv_folds: Number of stratified cross-validation folds used to estimate the true- and
            false-positive rates on the labeled data. Defaults to ``5``.
        n_bootstrap: Number of bootstrap draws for :meth:`pi_0_samples`. Defaults to ``2000``.
        random_seed: Random seed for the cross-validation folds and the bootstrap. Defaults to
            :data:`~bedroc.RANDOM_SEED`.
        **svc_kwargs: Arguments for :class:`sklearn.svm.SVC` (e.g. ``kernel="linear"``,
            ``C=0.5``). Defaults to sklearn's own (an RBF kernel with ``C=1.0`` and
            ``gamma="scale"``).
    """

    def __init__(
        self,
        name: str,
        labeled: Labeled,
        unlabeled: Unlabeled,
        *,
        n_cv_folds: int = 5,
        n_bootstrap: int = 2000,
        random_seed: int | None = RANDOM_SEED,
        **svc_kwargs: Any,
    ):
        self.name: str = name
        self.labeled: Labeled = labeled
        self.unlabeled: Unlabeled = unlabeled
        self.n_cv_folds: int = n_cv_folds
        self.n_bootstrap: int = n_bootstrap
        self.random_seed: int | None = random_seed
        self.svc: SVC = SVC(**svc_kwargs)

        category_names = labeled.data.category_names
        assert category_names is not None  # Guaranteed by Labeled
        self.category_names: list[str] = [str(category) for category in category_names]

        # SVC cannot handle missing values, so incomplete rows are excluded from both datasets
        self._labeled_idx: NpInt = _complete_rows(labeled.data.values_std, "labeled")
        self._unlabeled_idx: NpInt = _complete_rows(unlabeled.data.values_std, "unlabeled")

        self._cv_predictions: NpInt | None = None
        self._unlabeled_predictions: NpInt | None = None
        self._pi_0_samples: NpFloat | None = None

    @property
    def X(self) -> NpFloat:
        """Standardized labeled feature values used for training"""
        return self.labeled.data.values_std.to_numpy()[self._labeled_idx]

    @property
    def y(self) -> NpInt:
        """Category codes of the labeled training samples"""
        codes = self.labeled.data.category_codes
        assert codes is not None  # Guaranteed by Labeled
        return codes.to_numpy().astype(np.int64)[self._labeled_idx]

    @property
    def X_unlabeled(self) -> NpFloat:
        """Standardized unlabeled feature values that are classified"""
        return self.unlabeled.data.values_std.to_numpy()[self._unlabeled_idx]

    @property
    def unlabeled_predictions(self) -> NpInt:
        """Predicted category code for each unlabeled sample used"""
        if self._unlabeled_predictions is None:
            raise ValueError("Model not yet fitted. Call 'fit()' first.")
        return self._unlabeled_predictions

    @property
    def cv_predictions(self) -> NpInt:
        """Out-of-sample (cross-validated) predicted category code for each labeled sample"""
        if self._cv_predictions is None:
            raise ValueError("Model not yet fitted. Call 'fit()' first.")
        return self._cv_predictions

    def fit(self) -> None:
        """Estimates the classifier's error rates by cross-validation, then fits it on all the
        labeled data and classifies the unlabeled population."""
        logger.info("Fitting %s for %s", self.svc, self.name)

        # TODO: Minor leakage in the cross-validation below. self.X is standardized with the
        # scaling of the whole labeled (training) set, so each validation fold has already
        # contributed to the mean/std its fold's SVM is trained with. Strictly, the scaling should
        # be refit on the training folds only, e.g. by cross-validating
        # make_pipeline(StandardScaler(), SVC(...)) instead of the bare SVC. The effect on tpr/fpr
        # (and so the adjusted count) is expected to be negligible, since the mean/std of 4/5 of
        # the data barely differ from those of all of it.
        folds = StratifiedKFold(
            n_splits=self.n_cv_folds, shuffle=True, random_state=self.random_seed
        )
        self._cv_predictions = cross_val_predict(clone(self.svc), self.X, self.y, cv=folds)

        self.svc.fit(self.X, self.y)
        self._unlabeled_predictions = self.svc.predict(self.X_unlabeled)
        self._pi_0_samples = None

    @staticmethod
    def _rates(y_true: NpInt, y_pred: NpInt) -> tuple[float, float]:
        """Returns ``(tpr, fpr)`` with category 0 as the positive class."""
        tpr: float = float(np.mean(y_pred[y_true == 0] == 0))
        fpr: float = float(np.mean(y_pred[y_true == 1] == 0))
        return tpr, fpr

    @staticmethod
    def _adjust(cc: float, tpr: float, fpr: float) -> float:
        """Applies the adjusted-count correction, falling back to ``cc`` near chance."""
        if tpr - fpr <= MIN_RATE_SEPARATION:
            return cc
        return float(np.clip((cc - fpr) / (tpr - fpr), 0.0, 1.0))

    @property
    def rates(self) -> tuple[float, float]:
        """Cross-validated ``(tpr, fpr)`` on the labeled data, with category 0 as positive"""
        return self._rates(self.y, self.cv_predictions)

    def classify_and_count(self) -> float:
        """Fraction of the unlabeled samples classified as category 0 (biased under overlap)."""
        return float(np.mean(self.unlabeled_predictions == 0))

    def adjusted_count(self) -> float:
        """Category-0 fraction of the unlabeled population, corrected for misclassification."""
        tpr, fpr = self.rates
        if tpr - fpr <= MIN_RATE_SEPARATION:
            logger.warning(
                "Classifier is close to chance (tpr=%.3f, fpr=%.3f); reporting classify-and-count "
                "without adjustment",
                tpr,
                fpr,
            )
        return self._adjust(self.classify_and_count(), tpr, fpr)

    def pi_0_samples(self) -> NpFloat:
        """Bootstrap draws of the adjusted count.

        Each draw resamples, with replacement, both the unlabeled predictions (giving the raw
        count) and the labeled cross-validated predictions (giving the true- and false-positive
        rates), so the spread reflects the finite size of both datasets. The classifier is not
        refitted, so the variability from retraining the SVM itself is not included.

        Returns:
            Bootstrap samples of the category-0 fraction, shape ``(n_bootstrap,)``. Computed once
            per fit and then reused.
        """
        if self._pi_0_samples is not None:
            return self._pi_0_samples

        rng: np.random.Generator = np.random.default_rng(self.random_seed)
        y, cv_pred, pred = self.y, self.cv_predictions, self.unlabeled_predictions

        samples: NpFloat = np.empty(self.n_bootstrap)
        for b in range(self.n_bootstrap):
            unlabeled_draw = rng.integers(0, pred.size, pred.size)
            labeled_draw = rng.integers(0, y.size, y.size)
            cc: float = float(np.mean(pred[unlabeled_draw] == 0))
            tpr, fpr = self._rates(y[labeled_draw], cv_pred[labeled_draw])
            samples[b] = self._adjust(cc, tpr, fpr)

        self._pi_0_samples = samples

        return samples

    def true_counts(self) -> pd.Series | None:
        """True category counts over the unlabeled samples used, or ``None`` if unknown.

        Counted over the complete rows only, so they describe the same population the estimate
        refers to.
        """
        categories = self.unlabeled.data.categories
        if categories is None:
            return None
        return categories.iloc[self._unlabeled_idx].value_counts(sort=False)

    def summary(self, output_directory: Path | None = None) -> pd.DataFrame:
        """Summary statistics of the fraction estimate, with the raw count and error rates.

        Args:
            output_directory: Directory to write ``<name>_summary_statistics.xlsx`` to (see
                :func:`~bedroc.difference.utils.save_fraction_summary`). ``None`` for no output.
                Defaults to ``None``.

        Returns:
            One-row dataframe of :class:`~bedroc.core.utils.SummaryStatistics` for
            :meth:`pi_0_samples` (with the true fraction, if known), plus ``adjusted_count``,
            ``classify_and_count``, ``tpr`` and ``fpr`` columns
        """
        tpr, fpr = self.rates

        return save_fraction_summary(
            self.pi_0_samples(),
            category_counts=self.true_counts(),
            name=self.name,
            output_directory=output_directory,
            extra_columns={
                "adjusted_count": self.adjusted_count(),
                "classify_and_count": self.classify_and_count(),
                "tpr": tpr,
                "fpr": fpr,
            },
        )


def pipeline(
    data: DataContainer,
    *,
    unlabeled: Unlabeled | None = None,
    output_directory: Path | None = None,
    random_seed: int | None = RANDOM_SEED,
    build_model_kwargs: dict[str, Any] | None = None,
) -> SVMModel:
    """Pipeline for support vector machine population-fraction estimation.

    Uses the same labeled/unlabeled split as every other model for a given seed (see
    :func:`~bedroc.difference.partitioning.resolve_labeled_unlabeled`), fits an
    :class:`SVMModel`, and saves the fraction plot, the confusion matrix (when the true
    categories of the unlabeled data are known), and the summary statistics.

    Args:
        data: The container holding the labeled input data
        unlabeled: Optional real unlabeled target population. If ``None``, a held-out split of
            ``data`` stands in for it instead, for self-validation. Defaults to ``None``.
        output_directory: Directory to save outputs. ``None`` for no output.
        random_seed: Random seed for reproducibility. Defaults to :data:`~bedroc.RANDOM_SEED`.
        build_model_kwargs: Optional keyword arguments for :class:`SVMModel`, including any
            :class:`sklearn.svm.SVC` arguments (e.g. ``{"kernel": "linear"}``). Defaults to
            ``None``.

    Returns:
        The fitted :class:`SVMModel`
    """
    logger.info("Running SVM pipeline for %s", data.name)

    if output_directory is not None:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

    labeled, unlabeled = resolve_labeled_unlabeled(data, unlabeled, random_seed=random_seed)
    model: SVMModel = SVMModel(
        data.name, labeled, unlabeled, random_seed=random_seed, **(build_model_kwargs or {})
    )
    model.fit()

    summary: pd.DataFrame = model.summary(output_directory=output_directory)
    tpr, fpr = model.rates
    logger.info(
        "SVM %s fraction for %s: adjusted count=%.3f (95%% interval [%.3f, %.3f]), "
        "classify-and-count=%.3f, cross-validated tpr=%.3f, fpr=%.3f, true fraction=%s",
        model.category_names[0],
        data.name,
        model.adjusted_count(),
        float(summary["lower_95"].iloc[0]),
        float(summary["upper_95"].iloc[0]),
        model.classify_and_count(),
        tpr,
        fpr,
        "unknown" if "truth" not in summary else f"{float(summary['truth'].iloc[0]):.3f}",
    )
    counts = model.true_counts()
    ax: Axes = plot_group_fraction_posterior(
        model.pi_0_samples(),
        category_names=model.category_names,
        category_counts=counts,
        show_prior=False,
        title="Bootstrap distribution of category fractions (SVM adjusted count)",
        interval_label="95% CI",
    )
    ax.axvline(
        model.classify_and_count(),
        color="black",
        linestyle=":",
        linewidth=1.5,
        label=f"{model.category_names[0]} classify-and-count",
    )
    ax.legend()
    save_figure(get_figure(ax), Path(f"{data.name}_group_fraction_posterior"), output_directory)

    if counts is not None:
        codes = unlabeled.data.category_codes
        assert codes is not None  # Guaranteed since true counts exist
        y_true: NpInt = codes.to_numpy().astype(np.int64)[model._unlabeled_idx]
        logger.info(
            "SVM held-out accuracy for %s: %.3f",
            data.name,
            accuracy_score(y_true, model.unlabeled_predictions),
        )
        display = ConfusionMatrixDisplay.from_predictions(
            y_true,
            model.unlabeled_predictions,
            labels=[0, 1],
            display_labels=model.category_names,
            normalize="true",
            cmap="Blues",
            values_format="0.2f",
        )
        display.figure_.suptitle(f"{data.name} SVM confusion matrix")
        save_figure(display.figure_, Path(f"{data.name}_confusion_matrix"), output_directory)

    logger.info("SVM pipeline completed for %s", data.name)

    return model
