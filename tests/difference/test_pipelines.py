# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for model selection (bedroc.difference.pipelines.MODEL_PIPELINES/run_pipeline) and the
shared self-validation split (bedroc.difference.partitioning.resolve_labeled_unlabeled)."""

from typing import Any, get_args

import numpy as np
import pandas as pd
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from bedroc.core.data_container import DataContainer
from bedroc.difference import FitModel
from bedroc.difference import pipelines
from bedroc.difference.partitioning import (
    Unlabeled,
    resolve_labeled_unlabeled,
    train_test_split,
)


def _make_data_container(n: int = 40, *, random_seed: int = 0) -> DataContainer:
    rng = np.random.default_rng(random_seed)
    values = pd.DataFrame(rng.normal(size=(n, 2)), columns=["feature_0", "feature_1"])
    metadata = pd.DataFrame({"Type": np.where(np.arange(n) % 2 == 0, "a", "b")})
    return DataContainer(values, metadata=metadata, category_column="Type")


def test_model_pipelines_match_fit_model() -> None:
    assert set(pipelines.MODEL_PIPELINES) == set(get_args(FitModel))


@pytest.mark.parametrize("model", get_args(FitModel))
def test_run_pipeline_dispatches_to_selected_model(
    model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    for name in pipelines.MODEL_PIPELINES:
        monkeypatch.setitem(
            pipelines.MODEL_PIPELINES,
            name,
            lambda data, _name=name, **kwargs: calls.append((_name, kwargs)),
        )

    pipelines.run_pipeline(
        _make_data_container(),
        model=model,  # pyright: ignore[reportArgumentType]
        random_seed=3,
    )

    assert [name for name, _ in calls] == [model]
    assert set(calls[0][1]) == {
        "unlabeled",
        "output_directory",
        "random_seed",
        "build_model_kwargs",
    }
    assert calls[0][1]["random_seed"] == 3


def test_run_pipeline_does_not_compute_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    overlap_calls: list[str] = []
    monkeypatch.setattr(
        pipelines, "pipeline_OVL", lambda data, **kwargs: overlap_calls.append(data.name)
    )
    monkeypatch.setitem(pipelines.MODEL_PIPELINES, "covariance", lambda data, **kwargs: None)

    pipelines.run_pipeline(_make_data_container(), model="covariance")

    assert overlap_calls == []


def test_run_pipeline_rejects_unknown_model() -> None:
    # Under the beartype test plugin the FitModel type hint rejects the name first; without it,
    # run_pipeline's own check raises ValueError. Either way an unknown model is refused.
    with pytest.raises((ValueError, BeartypeCallHintParamViolation), match="not-a-model"):
        pipelines.run_pipeline(
            _make_data_container(),
            model="not-a-model",  # pyright: ignore[reportArgumentType]
        )


def test_resolve_labeled_unlabeled_matches_train_test_split() -> None:
    data = _make_data_container()
    labeled, unlabeled = resolve_labeled_unlabeled(data, None, random_seed=7)
    train, test = train_test_split(data, random_state=7)

    assert labeled.data.values.index.equals(train.values.index)
    assert unlabeled.data.values.index.equals(test.values.index)


def test_resolve_labeled_unlabeled_keeps_given_unlabeled() -> None:
    data = _make_data_container()
    given = Unlabeled(_make_data_container(10, random_seed=1))
    labeled, unlabeled = resolve_labeled_unlabeled(data, given, random_seed=7)

    assert labeled.data is data
    assert unlabeled is given
