# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Fully-tempered Bayesian category difference and population fraction model.

This module provides joint semi-supervised inference of category-specific parameters and the
population mixing fraction (:math:`\pi_0`) across two categories. Feature-level log-likelihoods
are treated as conditionally independent (to avoid the high variance and estimation overhead of a
full covariance matrix, cf.
:class:`~bedroc.difference.models.unified_covariance.UnifiedCovarianceModel`), which overcounts
evidence from correlated features. As in
:class:`~bedroc.difference.models.tempered_likelihood.TemperedLikelihoodModel`, both the labeled
training likelihood and the unlabeled mixture likelihood are tempered by the same scaling factor
:math:`\alpha \in (0, 1]` (estimated from the empirical intra-category feature correlation, see
:func:`~bedroc.difference.utils.compute_tempering_scale`). Unlike that model, every prior here is
*also* rescaled by :math:`\alpha` (raising each prior density to the power :math:`\alpha`, the
standard power-posterior/generalized-Bayes recipe), so the prior-to-likelihood balance stays
coherent once the likelihood's effective weight has changed.
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from matplotlib.axes import Axes

from bedroc import RANDOM_SEED, override
from bedroc.core.data_container import DataContainer
from bedroc.core.plotting import get_figure, save_figure
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
from bedroc.difference.partitioning import train_test_split
from bedroc.difference.utils import (
    compute_tempering_scale,
    oracle_pi0_posterior,
    validate_observation_data,
)

logger: logging.Logger = logging.getLogger(__name__)


class TemperedFullModel(UnlabeledMixtureModelMixin, CategoryClassifierBase):
    r"""Joint Bayesian inference of category differences and population fraction using a fully
    tempered model.

    This model simultaneously infers category-specific feature parameters and the mixture
    fraction :math:`\pi_0` for category 0 in an unlabeled target dataset (with category 1's
    fraction given by :math:`1 - \pi_0`). Feature likelihoods are Normal and treated as
    conditionally independent; both the labeled training likelihood and the unlabeled mixture
    likelihood are tempered by a scaling factor :math:`\alpha \in (0, 1]`, and every prior is
    rescaled by the same :math:`\alpha` to keep the prior-to-likelihood balance coherent. Missing
    values in ``X_train`` are omitted from the likelihood.

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
        logger.info("Creating a fully-tempered category difference model for %s", name)
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
        ``pi_0`` inference look if its other parameters (including the likelihood tempering,
        already baked into this model's ``obs_unlabeled`` definition) were known exactly?" Unlike
        the other three joint models, this one also tempers its *priors* (see ``build_model``), so
        the ``pi_0`` prior actually fitted is the tempered
        ``Beta(alpha_val*(prior_alpha-1)+1, alpha_val*(prior_beta-1)+1)``, not the untransformed
        ``Beta(self._prior_alpha, self._prior_beta)`` — recomputing ``alpha_val`` here and passing
        the tempered prior keeps this a like-for-like comparison with what was actually fitted.
        This model's per-feature ``sigma`` (features assumed conditionally independent) makes the
        result *not* directly comparable to
        :class:`~bedroc.difference.models.unified_covariance.UnifiedCovarianceModel`'s same-named
        method, which uses the fitted full ``cov_shared`` instead — each reflects that model's own
        structurally-constrained fit, not one universal oracle floor.
        """
        alpha_val: float = compute_tempering_scale(self.X, self.X_category_idx)

        return oracle_pi0_posterior(
            self.model,
            self.idata,
            prior_alpha=alpha_val * (self._prior_alpha - 1.0) + 1.0,
            prior_beta=alpha_val * (self._prior_beta - 1.0) + 1.0,
        )

    @override
    def build_model(self, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        """Builds the fully-tempered PyMC model and stores it in ``self._model``.

        Unlike
        :meth:`~bedroc.difference.base.CategoryComparisonBase.build_category_mean_priors`, the
        category-mean priors here are rescaled by ``alpha_val`` (as is every other prior), so that
        helper cannot be reused directly and the equivalent structure is rebuilt inline.

        Args:
            prior_alpha: Alpha parameter of the Beta prior on ``pi_0``, before tempering. Defaults
                to ``1.0``.
            prior_beta: Beta parameter of the Beta prior on ``pi_0``, before tempering. Defaults
                to ``1.0``.
        """
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta

        # Missing values are replaced with a safe finite placeholder before entering the PyMC
        # graph (their exact value is irrelevant since their contribution is masked out below);
        # this keeps the tempered log-density graph NaN-free, avoiding NaN silently propagating
        # through the gradient of the masking operation even though it's excluded from the
        # forward-pass sum.
        finite_mask: NpArray = np.isfinite(self.X)
        X_safe: NpFloat = np.where(finite_mask, self.X, 0.0)

        # Compute empirical likelihood scaling factor from the labeled data's intra-category
        # feature correlation
        alpha_val: float = compute_tempering_scale(self.X, self.X_category_idx)

        model_coords: dict[str, NpArray] = {
            **self.coords,
            "observation": np.arange(self.X.shape[0]),
            "observation_unlabeled": np.arange(self.X_unlabeled.shape[0]),
        }

        with pm.Model(coords=model_coords) as model:
            # Every prior below is "raised to the power alpha_val" (the standard power-posterior
            # recipe), but how that translates into a parameter transformation depends on where
            # the shape parameter sits in the distribution's density, so it looks different for
            # each family used here:
            #   - Normal/HalfNormal: sigma appears squared in the exponent (~exp(-x^2/(2*sigma^2))),
            #     so raising to a power scales the *variance* by 1/alpha_val, i.e. sigma scales by
            #     1/sqrt(alpha_val) (verified numerically against a directly-exponentiated,
            #     renormalized density).
            #   - Beta: its shape parameters (a, b) appear *linearly* in the exponent
            #     (~x^(a-1)(1-x)^(b-1)), so raising to a power shifts them additively instead:
            #     (a, b) -> (alpha_val*(a-1)+1, alpha_val*(b-1)+1), not (alpha_val*a, alpha_val*b)
            #     (see the pi_0 prior below).
            # Category-mean priors, rescaled by alpha_val accordingly.
            mu_0 = pm.Normal("mu_0", mu=0, sigma=0.5 / np.sqrt(alpha_val), dims="feature")
            delta_scale = pm.HalfNormal("delta_scale", sigma=0.5 / np.sqrt(alpha_val))
            delta = pm.Normal(
                "delta", mu=0, sigma=delta_scale / np.sqrt(alpha_val), dims="feature"
            )
            mu = pm.Deterministic(
                "mu", pm.math.stack([mu_0, mu_0 + delta], axis=0), dims=("category", "feature")
            )

            # Intrinsic feature variability, also rescaled by alpha_val.
            sigma = pm.HalfNormal("sigma", sigma=0.5 / np.sqrt(alpha_val), dims="feature")

            # Intrinsic effect size: separation of the underlying categories in units of their
            # intrinsic within-feature standard deviation. Convenient for downstream plotting to
            # not have underscores in the name since this will be used as the label
            pm.Deterministic("effect_size", delta / sigma, dims="feature")

            # Fraction prior. Raising a Beta(a, b) density to the power alpha gives
            # Beta(alpha*(a-1)+1, alpha*(b-1)+1), not Beta(alpha*a, alpha*b): e.g. for the default
            # uniform Beta(1, 1) prior, alpha*(a-1)+1 = 1 for any alpha, i.e. tempering a uniform
            # prior correctly leaves it uniform (verified numerically), whereas alpha*a = alpha*b
            # would instead turn it into a U-shaped distribution concentrated at 0 and 1.
            pi_0 = pm.Beta(
                "pi_0",
                alpha=alpha_val * (prior_alpha - 1.0) + 1.0,
                beta=alpha_val * (prior_beta - 1.0) + 1.0,
            )

            # Labeled Training Likelihood, tempered the same way as
            # TemperedLikelihoodModel.build_model (see that docstring for why both the training
            # and unlabeled likelihoods must be tempered together).
            X_data = pm.Data("X_data", X_safe, dims=("observation", "feature"))
            X_sigma_data = pm.Data("X_sigma", self.X_sigma, dims=("observation", "feature"))
            category_idx_data = pm.Data("category_idx", self.X_category_idx, dims="observation")
            finite_mask_data = pm.Data(
                "X_finite_mask", finite_mask, dims=("observation", "feature")
            )

            mu_observed = mu[category_idx_data, :]  # pyright: ignore
            sigma_observed = pm.math.sqrt(X_sigma_data**2 + sigma[None, :] ** 2)  # pyright: ignore

            train_dist = pm.Normal.dist(
                mu=mu_observed,
                sigma=sigma_observed,
                shape=X_data.shape,  # pyright: ignore
            )
            logp_train = pm.logp(train_dist, X_data)
            masked_logp_train = pt.where(finite_mask_data, logp_train, 0.0)
            # Unlike the priors above, this needs no family-specific reparametrization: X_data is
            # fixed/observed rather than a free variable being sampled, so directly scaling the
            # computed log-density by alpha_val is already exact for any distribution family (see
            # TemperedLikelihoodModel.build_model's identical Potential for the full explanation).
            pm.Potential("obs_train_tempered", alpha_val * pt.sum(masked_logp_train))  # pyright: ignore[reportOperatorIssue]

            # Unlabeled test likelihood (sample-level mixture)
            build_unlabeled_mixture(
                mu,  # pyright: ignore[reportArgumentType]
                sigma,
                self.X_unlabeled,
                self.X_sigma_unlabeled,
                pi_0,
                alpha_val,
                dims=("observation_unlabeled", "feature"),
            )

        self._model = model

    @override
    def _build_plot_dict(self, *, title: bool, random_seed: int | None = None) -> dict[str, Any]:
        """Builds the dictionary of diagnostic plots generated by :meth:`generate_plots`.

        See
        :meth:`~bedroc.difference.models.tempered_likelihood.TemperedLikelihoodModel._build_plot_dict`
        for why only the unlabeled mixture likelihood (and not the labeled training data) can be
        predictive-checked, and why this model's own variable names are used instead of the base
        class defaults.
        """
        return {
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


_build_pipeline: PipelineProtocol = build_pipeline(TemperedFullModel)


def pipeline(
    data: DataContainer,
    *,
    output_directory: Path | None = None,
    random_seed: int | None = RANDOM_SEED,
    build_model_kwargs: dict[str, Any] | None = None,
) -> TemperedFullModel:
    """Pipeline for the fully-tempered category difference model.

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
        The fitted :class:`TemperedFullModel` instance
    """
    model: TemperedFullModel = _build_pipeline(
        data,
        output_directory=output_directory,
        random_seed=random_seed,
        build_model_kwargs=build_model_kwargs,
    )

    _, test = train_test_split(data, random_state=random_seed)

    ax: Axes = model.plot_group_fraction_posterior(
        category_counts=test.category_counts,
        oracle_pdf=model.oracle_ceiling_pdf(),
    )
    save_figure(
        get_figure(ax),
        Path(f"{data.name}_group_fraction_posterior"),
        output_directory,
    )

    return model
