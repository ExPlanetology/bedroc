# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Untempered naive baseline for joint semi-supervised category difference and population
fraction inference.

This module provides joint semi-supervised inference of category-specific parameters and the
population mixing fraction (:math:`\pi_0`) across two categories, structurally identical to
:class:`~bedroc.difference.models.tempered_likelihood.TemperedLikelihoodModel` except that it
applies **no** tempering correction (:math:`\alpha = 1`) to compensate for the conditional-
independence ("naive") assumption across features. Comparing this model's posteriors against the
tempered variants isolates exactly what tempering changes.

Because the training likelihood here is a genuine observed random variable rather than a tempered
:class:`~pymc.Potential`, this model additionally supports prior/posterior predictive checks on
the labeled training data (see :meth:`UnifiedNaiveModel._build_plot_dict`), which the tempered
variants give up. Samples with a missing value in any feature are excluded from the training
likelihood, rather than relying on PyMC's automatic per-element imputation (see
:meth:`UnifiedNaiveModel.build_model` for why).
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pymc as pm
from matplotlib.axes import Axes

from bedroc import RANDOM_SEED, override
from bedroc.core.data_container import DataContainer
from bedroc.core.plotting import save_figure
from bedroc.core.type_aliases import NpArray, NpFloat, NpInt
from bedroc.core.utils import SummaryStatistics
from bedroc.difference import DEFAULT_CATEGORY_NAMES
from bedroc.difference.base import (
    CategoryClassifierBase,
    PipelineProtocol,
    UnlabeledMixtureModelMixin,
    build_pipeline,
)
from bedroc.difference.models.tempered_mixture import build_unlabeled_mixture
from bedroc.difference.utils import oracle_pi0_posterior, validate_observation_data

logger: logging.Logger = logging.getLogger(__name__)


class UnifiedNaiveModel(UnlabeledMixtureModelMixin, CategoryClassifierBase):
    r"""Joint Bayesian inference of category differences and population fraction using an
    untempered, naive (conditionally-independent) likelihood.

    This model simultaneously infers category-specific feature parameters and the mixture
    fraction :math:`\pi_0` for category 0 in an unlabeled target dataset (with category 1's
    fraction given by :math:`1 - \pi_0`). Feature likelihoods are Normal and treated as
    conditionally independent, exactly as in
    :class:`~bedroc.difference.models.tempered_likelihood.TemperedLikelihoodModel`, but with no
    tempering correction applied — this is the direct baseline against which the tempered models'
    effect can be judged. Samples with a missing value in any feature are excluded from the
    training likelihood (see :meth:`build_model`), matching
    :class:`~bedroc.difference.models.unified_covariance.UnifiedCovarianceModel`'s handling of
    missing training data.

    Args:
        name: Name of the model or analysis.
        X_train: Labeled observation data for the training set, shape ``(n_samples, n_features)``.
        X_category_idx_train: Category indices for each sample in the training set, shape
            ``(n_samples,)``.
        X_unlabeled: Unlabeled observation data for the target set, shape
            ``(n_samples, n_features)``.
        X_sigma: Optional observation uncertainties for the training set, shape
            ``(n_samples, n_features)``. Defaults to ``None``, in which case observations are
            assumed exact.
        X_sigma_unlabeled: Optional observation uncertainties for the target set, shape
            ``(n_samples, n_features)``. Defaults to ``None``, in which case observations are
            assumed exact.
        feature_names: Optional feature names. Defaults to ``["Feature 0", "Feature 1", ...]``.
        category_names: Optional category names. Defaults to :obj:`DEFAULT_CATEGORY_NAMES`.
    """

    def __init__(
        self,
        name: str,
        X_train: NpFloat,
        X_category_idx_train: NpInt,
        X_unlabeled: NpFloat,
        *,
        X_sigma: NpFloat | None = None,
        X_sigma_unlabeled: NpFloat | None = None,
        feature_names: Sequence | None = None,
        category_names: Sequence = DEFAULT_CATEGORY_NAMES,
    ):
        logger.info("Creating a unified naive category difference model for %s", name)
        super().__init__(
            name,
            X_train,
            X_category_idx_train,
            X_sigma=X_sigma,
            feature_names=feature_names,
            category_names=category_names,
        )

        self.X_unlabeled, self.X_sigma_unlabeled = validate_observation_data(
            X_unlabeled, X_sigma=X_sigma_unlabeled
        )
        self._prior_alpha: float
        self._prior_beta: float

    @override
    def pi_0_samples(
        self,
        *,
        prior_alpha: float | None = None,
        prior_beta: float | None = None,
        random_seed: int | None = None,
    ) -> NpFloat:
        """Posterior samples of the fraction of samples belonging to category 0 in the unlabeled
        dataset.

        ``prior_alpha``, ``prior_beta``, and ``random_seed`` are accepted only for interface
        compatibility with :class:`~bedroc.difference.base.CategoryClassifierBase`. This model's
        ``pi_0`` is sampled jointly with the rest of the posterior during ``run_inference()``
        using the prior set in ``build_model()``, so there is nothing left to resample here and
        these arguments are ignored.
        """
        del prior_alpha, prior_beta, random_seed

        pi_0_samples: NpFloat = self.idata.posterior["pi_0"].values.flatten()

        SummaryStatistics(pi_0_samples).log_summary("pi_0 posterior summary")

        return pi_0_samples

    def oracle_ceiling_pdf(self) -> tuple[NpFloat, NpFloat]:
        """Conditional posterior density of the unlabeled category-0 fraction, given every other
        parameter fixed at its posterior mean, for
        :meth:`~bedroc.difference.base.CategoryClassifierBase.plot_group_fraction_posterior`'s
        ``oracle_pdf`` argument.

        A plug-in oracle benchmark: holds every parameter except ``pi_0`` fixed at its posterior
        mean and infers ``pi_0`` by evaluating this model's own fitted PyMC graph directly (see
        :func:`~bedroc.difference.utils.oracle_pi0_posterior`) — i.e. "how would this model's own
        ``pi_0`` inference look if its other parameters were known exactly?" This model's
        per-feature ``sigma`` (features assumed conditionally independent) makes the result *not*
        directly comparable to
        :class:`~bedroc.difference.models.unified_covariance.UnifiedCovarianceModel`'s same-named
        method, which uses the fitted full ``cov_shared`` instead — each reflects that model's own
        structurally-constrained fit, not one universal oracle floor.
        """
        return oracle_pi0_posterior(
            self.model, self.idata, prior_alpha=self._prior_alpha, prior_beta=self._prior_beta
        )

    @override
    def build_model(self, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        """Builds the PyMC model for the category comparison and stores it in ``self._model``.

        Each category's per-feature mean is drawn from the shared reference/difference structure
        built by :meth:`~bedroc.difference.base.CategoryComparisonBase.build_category_mean_priors`.
        Features are treated as conditionally independent given the category (unlike
        :class:`~bedroc.difference.models.unified_covariance.UnifiedCovarianceModel`), with no
        tempering correction applied to compensate for the resulting overcounting of correlated
        feature information (cf.
        :class:`~bedroc.difference.models.tempered_likelihood.TemperedLikelihoodModel`).

        Samples with a missing value in any feature are excluded from the training likelihood
        (rather than relying on PyMC's automatic per-element missing-value imputation): imputation
        would replace the observed variable with a flattened 1-D reconstruction that no longer
        carries the ``("observation", "feature")`` dims this class's shared plotting
        infrastructure needs to facet predictive checks by category and feature (verified
        directly — the resulting variable is a ``Deterministic`` living in the ``"prior"``/
        ``"posterior"`` group under a different name, not the ``"prior_predictive"``/
        ``"posterior_predictive"`` groups the plots read from). Excluding incomplete rows instead
        keeps the observed likelihood's dims intact, matching
        :class:`~bedroc.difference.models.unified_covariance.UnifiedCovarianceModel`'s equivalent
        handling of missing training data.

        Args:
            prior_alpha: Alpha parameter of the Beta prior on ``pi_0``. Defaults to ``1.0``.
            prior_beta: Beta parameter of the Beta prior on ``pi_0``. Defaults to ``1.0``.
        """
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta

        # Rows with a missing value in any feature are excluded from the training likelihood; see
        # the build_model docstring for why this is preferred over automatic imputation here.
        train_s_idx: NpInt = np.where(np.all(np.isfinite(self.X), axis=1))[0]
        self._train_sample_idx: NpInt = train_s_idx
        train_c_idx = self.X_category_idx[train_s_idx]

        X_train_data = self.X[train_s_idx]
        X_train_sigma_data = self.X_sigma[train_s_idx]

        model_coords: dict[str, NpArray] = {
            **self.coords,
            "observation": np.arange(len(train_s_idx)),
            "observation_unlabeled": np.arange(self.X_unlabeled.shape[0]),
        }

        with pm.Model(coords=model_coords) as model:
            mu_0, delta_scale, delta, mu = self.build_category_mean_priors()

            # Intrinsic feature variability. ``sigma`` is expressed in standardized feature units.
            sigma = pm.HalfNormal("sigma", sigma=0.5, dims="feature")

            # Intrinsic effect size: separation of the underlying categories in units of their
            # intrinsic within-feature standard deviation. Convenient for downstream plotting to
            # not have underscores in the name since this will be used as the label
            pm.Deterministic("effect_size", delta / sigma, dims="feature")

            # Fraction prior
            pi_0 = pm.Beta("pi_0", alpha=prior_alpha, beta=prior_beta)

            # Labeled training likelihood: a plain observed random variable (complete cases only,
            # see above), unlike the tempered variants' Potential-based construction over the full,
            # NaN-masked dataset. This is the source of this model's restored prior/posterior
            # predictive-check support for training data (see _build_plot_dict).
            X_sigma_data = pm.Data("X_sigma", X_train_sigma_data, dims=("observation", "feature"))
            category_idx_data = pm.Data("category_idx", train_c_idx, dims="observation")

            mu_observed = mu[category_idx_data, :]  # pyright: ignore
            sigma_observed = pm.math.sqrt(X_sigma_data**2 + sigma[None, :] ** 2)  # pyright: ignore

            pm.Normal(
                "obs_train",
                mu=mu_observed,
                sigma=sigma_observed,
                observed=X_train_data,
                dims=("observation", "feature"),
            )

            # Unlabeled test likelihood (sample-level mixture), untempered (alpha=1.0).
            build_unlabeled_mixture(
                mu,  # pyright: ignore[reportArgumentType]
                sigma,
                self.X_unlabeled,
                self.X_sigma_unlabeled,
                pi_0,
                1.0,
                dims=("observation_unlabeled", "feature"),
            )

        self._model = model

    @override
    def _build_plot_dict(self, *, title: bool, random_seed: int | None = None) -> dict[str, Any]:
        """Builds the dictionary of diagnostic plots generated by :meth:`generate_plots`.

        Unlike the tempered variants, the labeled training likelihood here is a genuine observed
        random variable, so it can also be predictive-checked (in addition to the unlabeled
        mixture likelihood) — the concrete benefit of dropping tempering's Potential-based
        construction. This model also has no independent ``cov_shared`` (features are
        conditionally independent given the category), so ``effect_size`` and the
        parameter/posterior-distribution plots use this model's own variable names rather than
        the base class defaults. The category-fraction posterior is intentionally not included
        here since it is handled separately (e.g. by :func:`pipeline`), which can supply the true
        unlabeled category counts for comparison.
        """
        return {
            "prior_predictive": self.plot_prior_predictive(
                var_names=["obs_train"],
                sample_idx=self._train_sample_idx,
                title=title,
                random_seed=random_seed,
            ),
            "posterior_predictive": self.plot_posterior_predictive(
                var_names=["obs_train"],
                sample_idx=self._train_sample_idx,
                title=title,
                random_seed=random_seed,
            ),
            "prior_predictive_unlabeled": self.plot_prior_predictive_unlabeled(
                title=title, random_seed=random_seed
            ),
            "posterior_predictive_unlabeled": self.plot_posterior_predictive_unlabeled(
                title=title, random_seed=random_seed
            ),
            "parameter_estimates": self.plot_parameter_estimates(
                var_names=["mu_0", "delta_scale", "delta", "sigma"], title=title
            ),
            "posterior_distributions": self.plot_posterior_distributions(
                var_names=["mu", "sigma"], title=title
            ),
            "effect_sizes": self.plot_effect_sizes(title=title),
        }


_build_pipeline: PipelineProtocol = build_pipeline(UnifiedNaiveModel)


def pipeline(
    data: DataContainer,
    *,
    output_directory: Path | None = None,
    random_seed: int | None = RANDOM_SEED,
    build_model_kwargs: dict[str, Any] | None = None,
) -> UnifiedNaiveModel:
    """Pipeline for the untempered naive category difference model.

    This wraps the generic :func:`~bedroc.difference.base.build_pipeline` pipeline to
    additionally plot the category-fraction posterior, since that plot needs the true unlabeled
    category counts for comparison, which are not available to the generic base-class pipeline.

    Args:
        data: The container holding the input data for the pipeline
        output_directory: Directory to save generated figures. If ``None``, figures are not
            saved.
        random_seed: Random seed for reproducibility. Defaults to :data:`~bedroc.RANDOM_SEED`.
        build_model_kwargs: Optional keyword arguments passed to the model's ``build_model()``
            method (e.g. subclass-specific prior hyperparameters). Defaults to ``None``.

    Returns:
        The fitted :class:`UnifiedNaiveModel` instance
    """
    model: UnifiedNaiveModel = _build_pipeline(
        data,
        output_directory=output_directory,
        random_seed=random_seed,
        build_model_kwargs=build_model_kwargs,
    )

    _, test = data.train_test_split(random_state=random_seed)

    ax: Axes = model.plot_group_fraction_posterior(
        category_counts=test.category_counts,
        oracle_pdf=model.oracle_ceiling_pdf(),
    )
    save_figure(
        ax.get_figure(),  # pyright: ignore[reportArgumentType]
        Path(f"{data.name}_group_fraction_posterior"),
        output_directory,
    )

    return model
