# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the selectable Michigan data sources and per-type renaming. No real data files."""

import re
from pathlib import Path

import pandas as pd
import pytest

from bedroc.applications.zircons import run as zircon_run
from bedroc.applications.zircons.michigan import (
    DEFAULT_MICHIGAN_SOURCE,
    MICHIGAN_DATA_SOURCES,
    michigan_source,
)


def test_michigan_data_sources_registry() -> None:
    names = [source.name for source in MICHIGAN_DATA_SOURCES.values()]
    assert DEFAULT_MICHIGAN_SOURCE in MICHIGAN_DATA_SOURCES
    assert len(set(names)) == len(names)
    assert all(re.fullmatch(r"[a-z0-9_]+", name) for name in names)
    assert MICHIGAN_DATA_SOURCES[1].type_renames == {"Vulcan": "Plutonic"}


def test_type_renames_apply_after_capitalisation(tmp_path: Path) -> None:
    filepath = tmp_path / "michigan.xlsx"
    df = pd.DataFrame(
        {
            "Sample": ["s1", "s2", "s3"],
            "Type": ["vulcan", "plutonic", "volcanic"],
            "Unit": ["u", "u", "u"],
            "Zircon_number": [1, 2, 3],
            **{feature: [10.0, 20.0, 30.0] for feature in ("Ti", "Hf", "Th", "U")},
            **{f"{feature}±2SE(int)": [1.0, 1.0, 1.0] for feature in ("Ti", "Hf", "Th", "U")},
        }
    )
    with pd.ExcelWriter(filepath) as writer:
        df.to_excel(writer, sheet_name="Data", index=False)

    loaded, _ = michigan_source("test", filepath, type_renames={"Vulcan": "Plutonic"}).load()

    assert loaded["Type"].tolist() == ["Plutonic", "Plutonic", "Volcanic"]


def test_final_stats_runs_finds_each_michigan_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(zircon_run, "OUTPUT_ROOT", tmp_path)
    for source in ("all", "filters_p_ti_al"):
        (tmp_path / "Michigan" / source / "validation" / "kde_seed_1").mkdir(parents=True)
    # Detrital runs (not under validation/) have no true fraction and are not summarized
    (tmp_path / "Michigan" / "all" / "kde_seed_1").mkdir(parents=True)

    runs = zircon_run.final_stats_runs("michigan", "kde")

    assert [path.relative_to(tmp_path / "Michigan").as_posix() for path, _ in runs] == [
        "all/validation",
        "filters_p_ti_al/validation",
    ]
    assert {name for _, name in runs} == {"Michigan_labeled"}
