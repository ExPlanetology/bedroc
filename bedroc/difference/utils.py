# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Utility functions for quantifying and analysing differences between populations, including
validation of the observation data used in category difference modeling."""

import logging
from collections.abc import Generator
from contextlib import contextmanager

import numpy as np
import pymc as pm
import xarray as xr
from scipy.integrate import simpson
from scipy.stats import gaussian_kde, norm

from bedroc import RANDOM_SEED
from bedroc.core.type_aliases import NpArray, NpFloat, NpInt
from bedroc.core.utils import pooled_within_category_covariance

logger: logging.Logger = logging.getLogger(__name__)


@contextmanager
def log_pipeline_run(label: str) -> Generator[None]:
    """Logs a consistent start/completion message pair around a pipeline run.

    Args:
        label: Description of the pipeline run, included verbatim in both the start and
            completion log messages (e.g. ``f"SRMVF zircon analysis pipeline with inference:
            {inference}"``).
    """
    logger.info("Running %s", label)
    yield
    logger.info("%s completed", label)


def validate_observation_data(
    X: NpFloat, *, X_sigma: NpFloat | None = None
) -> tuple[NpFloat, NpFloat]:
    """Validates observation data.

    Args:
        X: Observation matrix with shape ``(n_samples, n_features)``. Missing values should be
            represented by ``NaN``.
        X_sigma: Optional 1-sigma uncertainties with the same shape as ``X``. ``NaN`` values are
            treated as zero uncertainty. If ``None``, uncertainties are assumed to be zero.

    Returns:
        Tuple containing validated ``X`` and ``X_sigma`` arrays

    Raises:
        ValueError: If the input arrays have invalid dimensions, shapes, or uncertainties
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array.")

    if np.any(np.isinf(X)):
        raise ValueError("X must not contain infinite values; use NaN for missing values.")

    if X_sigma is None:
        X_sigma = np.zeros_like(X, dtype=float)
    else:
        X_sigma = np.asarray(X_sigma, dtype=float)

        if X_sigma.shape != X.shape:
            raise ValueError(
                f"X_sigma must have the same shape as X ({X.shape}), got {X_sigma.shape}."
            )

        if np.any(np.isinf(X_sigma)):
            raise ValueError(
                "X_sigma must not contain infinite values; use NaN for missing values."
            )

        if np.any(X_sigma < 0):
            raise ValueError("X_sigma must contain only non-negative values.")

        X_sigma = np.nan_to_num(X_sigma, nan=0.0)

    return X, X_sigma


def validate_category_idx(category_idx: NpInt, n_samples: int) -> NpInt:
    """Validates a sample-level binary category index.

    Args:
        category_idx: Array of shape ``(n_samples,)`` containing binary category indices (0 or 1).
        n_samples: Number of samples in the observation data

    Returns:
        Validated ``category_idx`` array

    Raises:
        ValueError: If ``category_idx`` has an invalid shape or contains values other than 0 or 1
    """
    category_idx = np.asarray(
        category_idx
    )  # avoid silently coercing to int, which can change values

    if category_idx.shape != (n_samples,):
        raise ValueError(f"category_idx must have shape ({n_samples},), got {category_idx.shape}.")

    if not np.all(np.isin(category_idx, [0, 1])):
        raise ValueError("category_idx must contain only 0 or 1.")

    return category_idx


def distribution_overlap(
    values_0: NpArray, values_1: NpArray, *, n_grid: int = 2000
) -> tuple[NpArray, NpArray, NpArray, NpArray, float]:
    """Calculates KDEs and overlap data for two 1D distributions.

    The probability density functions are estimated using Gaussian kernel density estimation (KDE).
    The overlap coefficient is then calculated as the integral of the pointwise minimum of the two
    estimated probability density functions.

    For multimodal or strongly irregular distributions, the estimated overlap may be sensitive to
    the KDE bandwidth.

    Args:
        values_0: Samples from the first distribution.
        values_1: Samples from the second distribution.
        n_grid: Number of points to use for the grid over which to evaluate the PDFs. Defaults to
            ``2000``.

    Returns:
        Tuple containing the evaluation grid, first PDF, second PDF, overlap density, and overlap
        coefficient.
    """
    values_0 = np.asarray(values_0, dtype=float)
    values_1 = np.asarray(values_1, dtype=float)

    values_0 = values_0[np.isfinite(values_0)]
    values_1 = values_1[np.isfinite(values_1)]

    if len(values_0) < 2 or len(values_1) < 2:
        raise ValueError("Both populations require at least two finite observations.")

    lower = min(values_0.min(), values_1.min())
    upper = max(values_0.max(), values_1.max())

    x: NpArray = np.linspace(lower, upper, n_grid)

    pdf_0 = gaussian_kde(values_0)(x)
    pdf_1 = gaussian_kde(values_1)(x)

    overlap_density = np.minimum(pdf_0, pdf_1)

    overlap = simpson(overlap_density, x=x)

    logger.info("Distribution overlap coefficient: %.4f", overlap)

    return x, pdf_0, pdf_1, overlap_density, float(overlap)


def effect_size_from_overlap(overlap: float, *, sigma: float = 1.0) -> float:
    """Computes the standardized mean difference between two equal-variance Gaussians that
    produces a given overlap coefficient.

    Inverts the closed-form overlap of two equal-variance normal distributions,
    ``overlap = 2 * Phi(-|delta| / (2 * sigma))`` (where ``Phi`` is the standard normal CDF), so
    that :func:`distribution_overlap` on two Gaussian samples with this ``delta`` will reproduce
    the requested ``overlap``. Useful for calibrating a Gaussian synthetic-data generator (e.g.
    :class:`~bedroc.difference.group_synthetic.SyntheticDataGenerator`) so that its marginal
    per-feature overlap visually matches a real (generally non-Gaussian) empirical overlap
    coefficient, rather than reusing the real data's raw mean difference (which, passed through an
    idealized Gaussian, generally produces a different, smoother overlap than the real data shows).

    Args:
        overlap: Target overlap coefficient, between 0 and 1.
        sigma: Shared standard deviation of the two distributions. Defaults to ``1.0``.

    Returns:
        Non-negative standardized mean difference (``delta``) producing the requested overlap.

    Raises:
        ValueError: If ``overlap`` is not in ``(0, 1]``.
    """
    if not 0.0 < overlap <= 1.0:
        raise ValueError(f"overlap must be in (0, 1], got {overlap}.")

    return -2.0 * sigma * float(norm.ppf(overlap / 2.0))


def joint_overlap(
    values_0: NpArray,
    values_1: NpArray,
    *,
    n_samples: int = 500_000,
    random_seed: int | None = RANDOM_SEED,
) -> float:
    """Calculates the joint overlap coefficient of two empirical distributions.

    A multivariate Gaussian KDE is fitted directly to the observations of each population,
    preserving correlations between features. The overlap coefficient is estimated using Monte
    Carlo integration.

    If the two populations are exactly multivariate normal and share a single covariance matrix
    Sigma (the assumption underlying
    :class:`~bedroc.difference.models.unified_covariance.UnifiedCovarianceModel`), this integral
    collapses to a closed form depending only on the Mahalanobis distance D between the means,
    ``D = sqrt(delta^T Sigma^-1 delta)``: ``OVL = 2 * Phi(-D / 2)``, where ``Phi`` is the standard
    normal CDF. This is because the component of the data orthogonal to the discriminant direction
    ``Sigma^-1 @ delta`` is identically distributed under both populations and integrates out,
    leaving only the univariate overlap along that direction. Comparing this empirical estimate to
    ``2 * scipy.stats.norm.cdf(-D_posterior / 2)`` (using the model's posterior
    ``mahalanobis_distance``) is therefore a useful, assumption-free cross-check on whether the
    shared-covariance Gaussian assumption actually holds for the real data: a substantial
    mismatch indicates the categories' true covariance structure or shape differs from what the
    model assumes.

    Args:
        values_0: Array of shape (n_observations_0, n_features) for population 0.
        values_1: Array of shape (n_observations_1, n_features) for population 1.
        n_samples: Number of Monte Carlo samples used for integration. Defaults to ``500_000``.
        random_seed: Seed for random number generation to enable reproducibility. Defaults to
            :obj:`RANDOM_SEED`.

    Returns:
        Joint overlap coefficient, between 0 and 1.
    """
    values_0 = np.asarray(values_0, dtype=float)
    values_1 = np.asarray(values_1, dtype=float)

    if values_0.ndim != 2 or values_1.ndim != 2:
        raise ValueError("Input arrays must be two-dimensional.")

    if values_0.shape[1] != values_1.shape[1]:
        raise ValueError("Both populations must have the same number of features.")

    # A joint (correlated) KDE needs each row's features to co-occur, so a row with any missing
    # feature must be dropped entirely rather than filtered independently per feature.
    values_0 = values_0[np.all(np.isfinite(values_0), axis=1)]
    values_1 = values_1[np.all(np.isfinite(values_1), axis=1)]

    if values_0.shape[0] < 2 or values_1.shape[0] < 2:
        raise ValueError("Both populations require at least two observations.")

    # Fit multivariate KDEs directly to the observations
    kde_0 = gaussian_kde(values_0.T)
    kde_1 = gaussian_kde(values_1.T)

    # Generate samples from population 0's empirical KDE
    rng = np.random.default_rng(random_seed)
    samples = kde_0.resample(n_samples, seed=rng)

    # Evaluate both joint PDFs at the sampled points.
    pdf_0 = kde_0(samples)
    pdf_1 = kde_1(samples)

    # OVL = E_{x~p0}[min(1, p1/p0)]
    overlap = np.mean(np.minimum(1.0, pdf_1 / pdf_0))

    logger.info("Joint overlap coefficient: %.4f", overlap)

    return float(overlap)


def joint_naive_bayes_overlap(
    values_0: NpArray,
    values_1: NpArray,
    *,
    n_samples: int = 500_000,
    random_seed: int | None = RANDOM_SEED,
) -> float:
    r"""Calculate the joint overlap coefficient for a Naive Bayes model.

    For example, with two populations, 0 and 1, and three features,

        x = (f1, f2, f3)

    each population has a joint probability distribution,

        p0(f1, f2, f3)
        p1(f1, f2, f3)

    The overlap coefficient is defined as

    .. math::

        \mathrm{OVL} = \int \min[p_0(\mathbf{x}), p_1(\mathbf{x})] \, d\mathbf{x}

    The difficulty is that this is a 3D (or, in general, ND) integral, which becomes increasingly
    difficult to evaluate as the number of features increases. Here, the joint distributions are
    constructed as the product of the marginal KDEs for each feature by applying the Naive Bayes
    assumption of conditional independence.

    The overlap is estimated using Monte Carlo integration by sampling from the Naive Bayes joint
    distribution of population 0.

    Args:
        values_0: Array of shape (n_samples_0, n_features) for population 0
        values_1: Array of shape (n_samples_1, n_features) for population 1
        n_samples: Number of Monte Carlo samples used for integration. Defaults to ``500_000``.
        random_seed: Seed for random number generation to enable reproducibility. Defaults to
            :obj:`RANDOM_SEED`.

    Returns:
        Joint overlap coefficient, between 0 and 1.
    """
    values_0 = np.asarray(values_0, dtype=float)
    values_1 = np.asarray(values_1, dtype=float)

    if values_0.ndim != 2 or values_1.ndim != 2:
        raise ValueError("Input arrays must be two-dimensional.")

    if values_0.shape[1] != values_1.shape[1]:
        raise ValueError("Both populations must have the same number of features.")

    # Fit marginal KDEs, dropping each feature's own non-finite values independently: the Naive
    # Bayes assumption treats features as independent, so a missing value in one feature shouldn't
    # exclude that row's other features from their own marginals.
    def _finite_column(values: NpArray, column_index: int) -> NpArray:
        column = values[:, column_index]
        finite_column: NpArray = column[np.isfinite(column)]
        if len(finite_column) < 2:
            raise ValueError(
                f"Feature at index {column_index} requires at least two finite observations."
            )
        return finite_column

    kde_0 = [gaussian_kde(_finite_column(values_0, i)) for i in range(values_0.shape[1])]
    kde_1 = [gaussian_kde(_finite_column(values_1, i)) for i in range(values_1.shape[1])]

    rng = np.random.default_rng(random_seed)

    # Sample from population 0's Naive Bayes joint distribution
    # Each feature is sampled independently from its marginal KDE
    samples = np.column_stack([kde.resample(n_samples, seed=rng).ravel() for kde in kde_0])

    # Evaluate the Naive Bayes joint PDFs
    pdf_0 = np.ones(n_samples)
    pdf_1 = np.ones(n_samples)

    for i in range(values_0.shape[1]):
        pdf_0 *= kde_0[i](samples[:, i])
        pdf_1 *= kde_1[i](samples[:, i])

    # OVL = E_{x~p0}[min(1, p1/p0)]
    overlap = np.mean(np.minimum(1.0, pdf_1 / pdf_0))

    logger.info("Joint Naive Bayes overlap coefficient: %.4f", overlap)

    return float(overlap)


def participation_ratio(correlation_matrix: NpArray) -> float:
    r"""Calculates the participation ratio of a correlation matrix.

    The participation ratio is a measure of the effective number of independent features in a
    correlated system. It is defined as:

    .. math::

        N_\mathrm{eff} = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}

    where :math:`\lambda_i` are the eigenvalues of the correlation matrix.

    Args:
        correlation_matrix: Correlation matrix of shape (n_features, n_features)

    Returns:
        Effective number of independent features
    """
    n_features = correlation_matrix.shape[0]
    eigenvalues = np.linalg.eigvalsh(correlation_matrix)

    # For a correlation matrix, sum(eigenvalues) == n_features
    n_eff = (n_features**2) / np.sum(eigenvalues**2)

    logger.info("Actual number of features: %d", correlation_matrix.shape[0])
    logger.info("Participation ratio (effective number of independent features): %.2f", n_eff)

    return float(n_eff)


def compute_tempering_scale(X: NpArray, category_idx: NpArray) -> float:
    r"""Computes a likelihood tempering factor from intra-category correlation structures.

    The effective number of independent features :math:`N_\mathrm{eff}` is estimated using the
    participation ratio of the pooled intra-category correlation matrix. The tempering scaling
    factor :math:`\alpha` is defined as the fraction of effective independent feature dimensions
    relative to total feature dimensions :math:`F`:

    .. math::

        N_\mathrm{eff} = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}

    .. math::

        \alpha = \frac{N_\mathrm{eff}}{F}

    Args:
        X: Training data matrix of shape `(n_samples, n_features)`. May contain ``NaN`` for
            missing values; the pooled covariance is computed pairwise over the observations
            where both features are finite.
        category_idx: Category labels array of shape `(n_samples,)`

    Returns:
        Likelihood tempering factor :math:`\alpha \in [1/F, 1.0]`
    """
    # 1. Separate training data by category to avoid potential category differences from
    # contaminating or artificially inflating the feature correlation.
    X_c0: NpArray = X[category_idx == 0]
    X_c1: NpArray = X[category_idx == 1]

    # 2. Compute the intra-category correlation structure via the same sample-size-weighted
    # pooled-covariance estimator used elsewhere for this exact "shared within-category
    # association" quantity (see
    # bedroc.core.data_container.DataDiagnostics.within_category_covariance_matrix), rather than
    # averaging each category's own correlation matrix as if they carried equal weight — the two
    # categories can have substantially different sample sizes and within-category variances (see
    # the alignment/covariance discussion this factor feeds into), so a flat average would
    # over-weight the smaller/less variable category. participation_ratio needs a genuine
    # correlation matrix (trace == n_features) rather than a covariance matrix to correctly
    # represent "effective number of independent features", so the pooled covariance is
    # normalized to a correlation matrix afterward.
    cov_avg: NpArray = pooled_within_category_covariance(X_c0, X_c1).to_numpy()
    feature_std: NpArray = np.sqrt(np.diag(cov_avg))
    corr_avg: NpArray = cov_avg / np.outer(feature_std, feature_std)
    n_features: int = corr_avg.shape[0]

    n_eff: float = participation_ratio(corr_avg)

    # 4. Compute linear ratio and clip to valid bounds [1/F, 1.0]
    raw_alpha: float = n_eff / n_features
    alpha: float = float(np.clip(raw_alpha, 1.0 / n_features, 1.0))

    logger.info("Tempering factor alpha = %.4f", alpha)

    return alpha


def oracle_pi0_posterior(
    model: pm.Model,
    idata: xr.DataTree,
    *,
    obs_var_name: str = "obs_unlabeled",
    pi0_var_name: str = "pi_0",
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_grid: int = 2001,
) -> tuple[NpFloat, NpFloat]:
    r"""Computes the *conditional* posterior density of the category-0 mixing fraction ``pi_0``
    given every other model parameter fixed at a point estimate — a plug-in oracle benchmark for
    what's achievable if those nuisance parameters were known exactly.

    This is the mathematical conditional posterior :math:`p(\pi_0 \mid \hat\theta, X)`, where
    :math:`\hat\theta` is each other free parameter's own posterior mean (from ``idata``) rather
    than the true (unknown) value — hence "oracle": it answers "if this fitted model's other
    parameters were known exactly, how would inferring ``pi_0`` actually look?", using an estimate
    as a stand-in for that unavailable ground truth. Every free random variable in ``model`` other
    than ``pi0_var_name`` is held fixed this way, broadcast across a synthetic set of ``n_grid``
    "draws"; ``pi0_var_name`` is swept over ``np.linspace(eps, 1 - eps, n_grid)`` across those same
    synthetic draws. :func:`pymc.compute_log_likelihood` then evaluates ``obs_var_name``'s
    log-likelihood at each synthetic draw — i.e. as a function of the swept ``pi_0`` grid, with
    every other parameter held at its point estimate — using the *model's own* fitted PyMC graph
    directly, the same machinery
    :meth:`~bedroc.difference.models.standard_difference.StandardDifferenceModel.compute_log_likelihood`
    uses. This keeps the result fully consistent with whatever ``model`` actually defines
    (likelihood family, tempering, uncertainty combination, etc.), with no separate reimplementation
    of that logic to maintain. ``pm.Data``/``pm.set_data`` is not needed: that mechanism is for
    *observed* data (already fixed as ``model``'s own ``X_unlabeled``), whereas this substitutes
    values for free RVs directly via a hand-built posterior-shaped dataset, exactly as
    :func:`pymc.compute_log_likelihood` already does for real MCMC draws.

    Holding every parameter except ``pi_0`` fixed removes parameter uncertainty from the result:
    the resulting curve is a genuine floor on achievable precision given known parameters — the
    model's actual posterior (which also carries parameter uncertainty) cannot be narrower than
    this.

    Args:
        model: The fitted PyMC model (e.g. a joint semi-supervised model's ``self.model``).
        idata: The model's inference data (e.g. ``self.idata``), used only to read off each free
            RV's posterior mean.
        obs_var_name: Name of the observed variable whose log-likelihood is evaluated. Defaults
            to ``"obs_unlabeled"``, matching
            :class:`~bedroc.difference.base.UnlabeledMixtureModelMixin`'s naming contract.
        pi0_var_name: Name of the free RV swept over the grid; every other free RV is held fixed.
            Defaults to ``"pi_0"``.
        prior_alpha: Alpha parameter of the Beta prior on ``pi_0``. Should match the fitted
            model's own prior for a like-for-like comparison. Defaults to ``1.0``.
        prior_beta: Beta parameter of the Beta prior on ``pi_0``. Defaults to ``1.0``.
        n_grid: Number of grid points spanning ``[0, 1]`` used to represent the posterior.
            Defaults to ``2001``.

    Returns:
        Tuple ``(grid, density)``, each of shape ``(n_grid,)``: the grid of candidate ``pi_0``
        values and the (normalized) posterior density evaluated at each.
    """
    eps: float = np.finfo(float).eps
    grid: NpFloat = np.linspace(eps, 1.0 - eps, n_grid)

    data_vars: dict[str, tuple[tuple, NpArray]] = {}
    for rv in model.free_RVs:
        name = rv.name
        if name == pi0_var_name:
            data_vars[name] = (("chain", "draw"), grid[None, :])
            continue
        draws: xr.DataArray = idata.posterior[name]
        mean_val: NpArray = draws.mean(dim=("chain", "draw")).values
        extra_dims = [d for d in draws.dims if d not in ("chain", "draw")]
        tiled: NpArray = np.broadcast_to(mean_val, (n_grid, *mean_val.shape)).copy()
        data_vars[name] = (("chain", "draw", *extra_dims), tiled[None, ...])

    coords: dict[str, NpArray] = {"chain": np.array([0]), "draw": np.arange(n_grid)}
    for extra_coord in ("feature",):
        if extra_coord in idata.posterior.coords:
            coords[extra_coord] = idata.posterior.coords[extra_coord].values

    synthetic_posterior: xr.Dataset = xr.Dataset(data_vars, coords=coords)
    synthetic_idata: xr.DataTree = xr.DataTree.from_dict({"posterior": synthetic_posterior})

    with model:
        log_likelihood: xr.Dataset = pm.compute_log_likelihood(
            synthetic_idata, var_names=[obs_var_name], extend_inferencedata=False
        )  # pyright: ignore[reportAssignmentType]

    sum_dims = [d for d in log_likelihood[obs_var_name].dims if d not in ("chain", "draw")]
    log_lik: NpFloat = log_likelihood[obs_var_name].sum(dim=sum_dims).isel(chain=0).values

    log_grid_0: NpFloat = np.log(grid)
    log_grid_1: NpFloat = np.log1p(-grid)

    # Log Beta(prior_alpha, prior_beta) prior (up to a normalizing constant, which cancels out
    # when the PMF is normalized below). Generic Bayesian bookkeeping, not model-specific, so
    # unlike the likelihood term above there is no risk of this drifting from the model.
    log_prior: NpFloat = (prior_alpha - 1.0) * log_grid_0 + (prior_beta - 1.0) * log_grid_1

    log_post: NpFloat = log_lik + log_prior

    # Normalize to a proper density over the grid
    log_post -= log_post.max()
    unnormalized: NpFloat = np.exp(log_post)
    density: NpFloat = unnormalized / np.trapezoid(unnormalized, grid)

    return grid, density
