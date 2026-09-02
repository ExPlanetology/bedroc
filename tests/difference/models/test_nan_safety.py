# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""NaN-handling smoke tests for the two distinct missing-value strategies used by the joint
semi-supervised models: row-exclusion (UnifiedNaiveModel, UnifiedCovarianceModel) and
finite_mask/pt.where masking (TemperedLikelihoodModel, TemperedFullModel, and, via the shared
build_unlabeled_mixture, all three of those models' unlabeled mixture likelihood too)."""

import numpy as np
import pytest

from bedroc.difference.models.tempered_full import TemperedFullModel
from bedroc.difference.models.tempered_likelihood import TemperedLikelihoodModel
from bedroc.difference.models.unified_covariance import UnifiedCovarianceModel
from bedroc.difference.models.unified_naive import UnifiedNaiveModel


@pytest.mark.pymc
def test_unified_naive_handles_missing_training_values(make_synthetic_two_category) -> None:
    """UnifiedNaiveModel excludes incomplete rows from the training likelihood; a NaN in
    X_train should not prevent sampling or produce NaN in the posterior."""
    X_train, X_category_idx_train, X_unlabeled = make_synthetic_two_category(
        n_train_per_category=40, n_unlabeled=30, n_features=3, effect_size=2.0, random_seed=2
    )
    X_train[3, 1] = np.nan

    model = UnifiedNaiveModel("t", X_train, X_category_idx_train, X_unlabeled)
    model.build_model()
    model.run_inference(draws=200, tune=200, chains=1, cores=1, progressbar=False)

    pi_0_samples = model.pi_0_samples()
    assert not np.any(np.isnan(pi_0_samples))


@pytest.mark.pymc
def test_unified_covariance_handles_missing_training_values(make_synthetic_two_category) -> None:
    """UnifiedCovarianceModel excludes incomplete rows from the joint MvNormal training
    likelihood (a partial feature vector cannot be evaluated under a joint density); a NaN in
    X_train should not prevent sampling or produce NaN in the posterior."""
    X_train, X_category_idx_train, X_unlabeled = make_synthetic_two_category(
        n_train_per_category=40, n_unlabeled=30, n_features=3, effect_size=2.0, random_seed=4
    )
    X_train[3, 1] = np.nan

    model = UnifiedCovarianceModel("t", X_train, X_category_idx_train, X_unlabeled)
    model.build_model()
    model.run_inference(draws=200, tune=200, chains=1, cores=1, progressbar=False)

    pi_0_samples = model.pi_0_samples()
    assert not np.any(np.isnan(pi_0_samples))


@pytest.mark.pymc
def test_tempered_likelihood_handles_missing_training_values(make_synthetic_two_category) -> None:
    """TemperedLikelihoodModel masks NaN entries out of the tempered training-likelihood
    Potential (finite_mask/pt.where); a NaN in X_train should not prevent sampling or produce
    NaN in the posterior."""
    X_train, X_category_idx_train, X_unlabeled = make_synthetic_two_category(
        n_train_per_category=40, n_unlabeled=30, n_features=3, effect_size=2.0, random_seed=3
    )
    X_train[3, 1] = np.nan

    model = TemperedLikelihoodModel("t", X_train, X_category_idx_train, X_unlabeled)
    model.build_model()
    model.run_inference(draws=200, tune=200, chains=1, cores=1, progressbar=False)

    pi_0_samples = model.pi_0_samples()
    assert not np.any(np.isnan(pi_0_samples))


@pytest.mark.pymc
def test_unified_naive_handles_missing_unlabeled_values(make_synthetic_two_category) -> None:
    """UnifiedNaiveModel's unlabeled mixture (build_unlabeled_mixture) masks NaN entries out of
    the per-sample feature sum; a NaN in X_unlabeled should not prevent sampling or produce NaN
    in the posterior."""
    X_train, X_category_idx_train, X_unlabeled = make_synthetic_two_category(
        n_train_per_category=40, n_unlabeled=30, n_features=3, effect_size=2.0, random_seed=5
    )
    X_unlabeled[3, 1] = np.nan

    model = UnifiedNaiveModel("t", X_train, X_category_idx_train, X_unlabeled)
    model.build_model()
    model.run_inference(draws=200, tune=200, chains=1, cores=1, progressbar=False)

    pi_0_samples = model.pi_0_samples()
    assert not np.any(np.isnan(pi_0_samples))


@pytest.mark.pymc
def test_tempered_likelihood_handles_missing_unlabeled_values(make_synthetic_two_category) -> None:
    """TemperedLikelihoodModel's unlabeled mixture masks NaN entries out of the per-sample
    feature sum; a NaN in X_unlabeled should not prevent sampling or produce NaN in the
    posterior."""
    X_train, X_category_idx_train, X_unlabeled = make_synthetic_two_category(
        n_train_per_category=40, n_unlabeled=30, n_features=3, effect_size=2.0, random_seed=6
    )
    X_unlabeled[3, 1] = np.nan

    model = TemperedLikelihoodModel("t", X_train, X_category_idx_train, X_unlabeled)
    model.build_model()
    model.run_inference(draws=200, tune=200, chains=1, cores=1, progressbar=False)

    pi_0_samples = model.pi_0_samples()
    assert not np.any(np.isnan(pi_0_samples))


@pytest.mark.pymc
def test_tempered_full_handles_missing_unlabeled_values(make_synthetic_two_category) -> None:
    """TemperedFullModel's unlabeled mixture masks NaN entries out of the per-sample feature sum;
    a NaN in X_unlabeled should not prevent sampling or produce NaN in the posterior."""
    X_train, X_category_idx_train, X_unlabeled = make_synthetic_two_category(
        n_train_per_category=40, n_unlabeled=30, n_features=3, effect_size=2.0, random_seed=7
    )
    X_unlabeled[3, 1] = np.nan

    model = TemperedFullModel("t", X_train, X_category_idx_train, X_unlabeled)
    model.build_model()
    model.run_inference(draws=200, tune=200, chains=1, cores=1, progressbar=False)

    pi_0_samples = model.pi_0_samples()
    assert not np.any(np.isnan(pi_0_samples))


@pytest.mark.pymc
def test_unified_covariance_handles_missing_unlabeled_values(make_synthetic_two_category) -> None:
    """UnifiedCovarianceModel excludes incomplete rows from the joint MvNormal unlabeled mixture
    (same constraint as its training likelihood); a NaN in X_unlabeled should not prevent
    sampling or produce NaN in the posterior."""
    X_train, X_category_idx_train, X_unlabeled = make_synthetic_two_category(
        n_train_per_category=40, n_unlabeled=30, n_features=3, effect_size=2.0, random_seed=8
    )
    X_unlabeled[3, 1] = np.nan

    model = UnifiedCovarianceModel("t", X_train, X_category_idx_train, X_unlabeled)
    model.build_model()
    model.run_inference(draws=200, tune=200, chains=1, cores=1, progressbar=False)

    pi_0_samples = model.pi_0_samples()
    assert not np.any(np.isnan(pi_0_samples))
