# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for separate training/test sets in bedroc.difference.group_synthetic (category
proportions that differ between training and target data). No MCMC."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bedroc.applications.zircons import run as zircon_run
from bedroc.applications.zircons.synthetic import split_name
from bedroc.difference import group_synthetic
from bedroc.difference.group_synthetic import SyntheticDataGenerator


def _generator(**kwargs: Any) -> SyntheticDataGenerator:
    return SyntheticDataGenerator(
        n_samples=400,
        n_features=2,
        feature_offsets=np.full(2, 1.5),
        feature_sigma=np.ones(2),
        category_0_fraction=0.5,
        random_seed=0,
        **kwargs,
    )


def test_train_and_test_sets_have_requested_sizes_and_proportions() -> None:
    generator = _generator(n_test=200, test_category_0_fraction=0.2)
    generator.generate()

    assert generator.X.shape == (400, 2)
    assert np.bincount(generator.X_category_idx).tolist() == [200, 200]
    assert generator.X_test.shape == (200, 2)
    assert np.bincount(generator.X_test_category_idx).tolist() == [40, 160]


def test_train_and_test_sets_share_category_distributions() -> None:
    generator = SyntheticDataGenerator(
        n_samples=20_000,
        n_features=2,
        feature_offsets=np.full(2, 1.5),
        feature_sigma=np.ones(2),
        random_seed=1,
        n_test=20_000,
        test_category_0_fraction=0.2,
    )
    generator.generate()

    for code in (0, 1):
        train_mean = generator.X[generator.X_category_idx == code].mean(axis=0)
        test_mean = generator.X_test[generator.X_test_category_idx == code].mean(axis=0)
        np.testing.assert_allclose(train_mean, test_mean, atol=0.05)


def test_to_train_test_uses_training_scaling_and_keeps_true_counts() -> None:
    generator = _generator(n_test=200, test_category_0_fraction=0.2)
    generator.generate()
    train, unlabeled = generator.to_train_test(name="Synth")

    assert train.name == "Synth"
    assert unlabeled.data.name == "Synth_test"
    np.testing.assert_allclose(unlabeled.data.scaling.means, train.scaling.means)
    np.testing.assert_allclose(unlabeled.data.scaling.stds, train.scaling.stds)
    counts = unlabeled.data.category_counts
    assert counts is not None
    assert counts.tolist() == [40, 160]


def test_to_train_test_keeps_both_categories_when_test_has_one() -> None:
    generator = _generator(n_test=50, test_category_0_fraction=1.0)
    generator.generate()
    _, unlabeled = generator.to_train_test()

    counts = unlabeled.data.category_counts
    assert counts is not None
    assert counts.tolist() == [50, 0]


def test_to_train_test_requires_a_test_set() -> None:
    generator = _generator()
    generator.generate()
    with pytest.raises(ValueError, match="No test data"):
        generator.to_train_test()


@pytest.mark.parametrize("n_test", [None, 120])
def test_run_pipeline_passes_test_set_as_unlabeled(
    n_test: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        group_synthetic, "_run_pipeline", lambda data, **kwargs: calls.append(kwargs)
    )

    group_synthetic.run_pipeline(
        _generator(n_test=n_test, test_category_0_fraction=0.3), output_directory=None
    )

    unlabeled = calls[0]["unlabeled"]
    if n_test is None:
        assert unlabeled is None
    else:
        assert unlabeled.data.values.shape[0] == n_test


def test_split_name() -> None:
    assert split_name((0.5, 0.2), (942, 236)) == "train0.50x942_test0.20x236"


def test_final_stats_runs_finds_nested_synthetic_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(zircon_run, "OUTPUT_ROOT", tmp_path)
    for case in ("nocov", "withcov", "nocov/train0.50x942_test0.20x236"):
        (tmp_path / "synthetic" / case / "svm_seed_1").mkdir(parents=True)
    (tmp_path / "synthetic" / "withcov" / "covariance_seed_1").mkdir(parents=True)

    runs = zircon_run.final_stats_runs("synthetic", "svm")

    assert [path.relative_to(tmp_path / "synthetic").as_posix() for path, _ in runs] == [
        "nocov",
        "nocov/train0.50x942_test0.20x236",
        "withcov",
    ]
    assert {name for _, name in runs} == {"Synthetic"}
