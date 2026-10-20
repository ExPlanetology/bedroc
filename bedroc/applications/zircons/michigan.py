#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Michigan dataset processing and plotting functions"""

import logging
from pathlib import Path

import pandas as pd

from bedroc import OUTPUT_ROOT, RANDOM_SEED
from bedroc.applications.zircons import michigan_all
from bedroc.applications.zircons.utils import (
    PLOT_FEATURE_LABELS,
    ZIRCON_FILTER,
    ZirconFilter,
    ZirconSource,
    dump_zircon_excel,
    export_zircon_summary,
    log_tick_overrides,
)
from bedroc.core.data_container import DataContainer
from bedroc.difference import DEFAULT_FIT_MODEL, FitModel
from bedroc.difference.partitioning import LabeledUnlabeledSplit
from bedroc.difference.pipelines import run_pipeline as _run_pipeline
from bedroc.difference.plotting import (
    plot_corner,
    plot_corner_by_category,
    plot_corner_with_unlabeled,
)
from bedroc.difference.utils import log_pipeline_run, run_output_directories

logger: logging.Logger = logging.getLogger(__name__)

DATASET_NAME: str = "Michigan"
"""Name for the Michigan zircon dataset analysis"""

DEFAULT_FEATURE_COLUMNS: list[str] = [
    "Ti",
    "Hf",
    "Th",
    "U",
    # "Eu/Eu*",
    # "Ce/Ce*",
    # Extra features
    # "207Pb/206Pb",
    # "Y",
    # "Nb",
    # "Zr",
]
"""Default feature columns for Michigan zircon dataset"""
NAME_COLUMNS: list[str] = ["Sample", "Type", "Unit", "Zircon_number"]
UNCERTAINTY_SUFFIXES: tuple[str, ...] = ("±2SE(int)", "±Error", "±2SE(prop)")
"""Candidate suffixes for a feature's uncertainty column. The Michigan dataset does not use a
single uncertainty suffix."""
LABELED_CATEGORIES: tuple[str, str] = ("Plutonic", "Volcanic")
"""The two ``Type`` values treated as the labeled comparison pair. The remaining ``Type`` value
(``Detrital``, zircons of unknown provenance) is pooled into the unlabeled population."""


def michigan_source(name: str, filepath: Path) -> ZirconSource:
    """Describes a Michigan zircon source spreadsheet.

    All Michigan sources share the same layout, so only the name and path differ.

    Args:
        name: Name for the dataset
        filepath: Path to the Michigan zircon dataset (Excel file)

    Returns:
        The source description
    """
    return ZirconSource(
        name=name,
        filepath=filepath,
        sheet_name="Data",
        feature_columns={feature: feature for feature in DEFAULT_FEATURE_COLUMNS},
        name_columns=NAME_COLUMNS,
        uncertainty_suffixes=UNCERTAINTY_SUFFIXES,
        groupby_columns=["Type", "Unit"],
    )


MICHIGAN_SOURCES: tuple[ZirconSource, ...] = (
    # Barth is missing Th, so we skip it for now. It can be added back in later if needed.
    # michigan_source("barth", michigan_barth),
    # michigan_source("hendrickx", michigan_hendrickx),
    # michigan_source("foldenauer", michigan_foldenauer),
    # michigan_source("petryk", michigan_petryk),
    # michigan_source("pray", michigan_pray),
    # michigan_source("staudenmann", michigan_staudenmann),
    michigan_source("all", michigan_all),
)
"""The Michigan zircon source spreadsheets, combined by :func:`build_michigan_dataset`"""


def process_michigan(
    name: str,
    filepath: Path,
    *,
    output_directory: Path | None = None,
    zircon_filter: ZirconFilter = ZIRCON_FILTER,
) -> DataContainer:
    """Processes a Michigan zircon dataset into a :obj:`DataContainer`.

    Args:
        name: Name for the dataset
        filepath: Path to the Michigan zircon dataset (Excel file)
        output_directory: Directory to save the processed data. Defaults to ``None`` (no saving).
        zircon_filter: Filtering criteria and transforms to apply. Defaults to
            :obj:`ZIRCON_FILTER`.

    Returns:
        A :obj:`DataContainer` containing the processed Michigan zircon dataset.
    """
    return michigan_source(name, filepath).process(
        output_directory=output_directory, zircon_filter=zircon_filter
    )


def build_michigan_dataset(
    *, output_directory: Path | None = None, zircon_filter: ZirconFilter = ZIRCON_FILTER
) -> LabeledUnlabeledSplit:
    """Builds the combined Michigan zircon dataset from every source spreadsheet.

    Processes each of :obj:`MICHIGAN_SOURCES`, concatenates them into a single
    :obj:`DataContainer` via :meth:`DataContainer.concat`, and splits the result into a labeled
    comparison pair (:obj:`LABELED_CATEGORIES`) plus a pooled unlabeled remainder (the
    ``Detrital`` zircons, whose provenance is unknown) via
    :meth:`LabeledUnlabeledSplit.from_data_container`.

    Args:
        output_directory: Directory to save each source's processed data and the combined dataset.
            Defaults to ``None`` (no saving).
        zircon_filter: Filtering criteria and transforms to apply. Defaults to
            :obj:`ZIRCON_FILTER`.

    Returns:
        The :obj:`LabeledUnlabeledSplit` for all Michigan zircon sources.
    """
    data: DataContainer = DataContainer.concat(
        [
            source.process(output_directory=output_directory, zircon_filter=zircon_filter)
            for source in MICHIGAN_SOURCES
        ],
        name=DATASET_NAME,
        category_column="Type",
    )

    dump_zircon_excel(
        data.get_dataframe(),
        output_directory,
        f"{DATASET_NAME}_combined.xlsx",
        sheet_name="data",
    )
    export_zircon_summary(
        pd.concat([data.metadata, data.values], axis=1),
        output_directory=output_directory,
        name=DATASET_NAME,
        groupby_columns=["Type", "Unit"],
        feature_columns=data.values.columns.tolist(),
    )

    split: LabeledUnlabeledSplit = LabeledUnlabeledSplit.from_data_container(
        data, categories=LABELED_CATEGORIES, name=DATASET_NAME
    )

    dump_zircon_excel(
        split.labeled.data.get_dataframe(),
        output_directory,
        f"{DATASET_NAME}_labeled.xlsx",
        sheet_name="data",
    )
    dump_zircon_excel(
        split.unlabeled.data.get_dataframe(),
        output_directory,
        f"{DATASET_NAME}_unlabeled.xlsx",
        sheet_name="data",
    )

    return split


def run_pipeline(
    model: FitModel = DEFAULT_FIT_MODEL,
    *,
    output_directory: Path | None = OUTPUT_ROOT / DATASET_NAME,
    random_seed: int | None = RANDOM_SEED,
    validate: bool = False,
):
    """Runs the analysis pipeline for the Michigan zircon dataset analysis.

    By default the models are fitted to the labeled Plutonic/Volcanic zircons and estimate the
    category fraction of the Detrital zircons, whose true fraction is unknown. With ``validate``,
    the Detrital zircons are left out and a held-out split of the labeled zircons stands in for
    them instead (self-validation, as for the SRMVF data), so the estimate can be scored against
    the known true fraction.

    Args:
        model: Model to fit. Defaults to :obj:`DEFAULT_FIT_MODEL`.
        output_directory: Directory to save the processed data. Defaults to
            ``OUTPUT_ROOT / DATASET_NAME``. With ``validate``, its ``validation`` subdirectory is
            used instead, keeping these runs apart from the Detrital runs.
        random_seed: Seed for random number generation to enable reproducibility. Defaults to
            :obj:`RANDOM_SEED`.
        validate: Whether to self-validate on the labeled zircons only, leaving out the Detrital
            zircons. Defaults to ``False``.
    """
    label: str = "self-validation" if validate else "analysis"
    with log_pipeline_run(f"Michigan zircon {label} pipeline with model: {model}"):
        if validate and output_directory is not None:
            output_directory = output_directory / "validation"
        output_directory, output_directory_data = run_output_directories(
            output_directory, model, random_seed
        )

        split: LabeledUnlabeledSplit = build_michigan_dataset(
            output_directory=output_directory_data
        )

        # Self-validation passes only the labeled zircons, so a held-out split of them (with
        # known categories) stands in for the unlabeled population
        _run_pipeline(
            split.labeled.data if validate else split,
            model=model,
            output_directory=output_directory,
            random_seed=random_seed,
            unlabeled_label="Detrital",
            feature_labels=PLOT_FEATURE_LABELS,
            tick_overrides=log_tick_overrides(),
        )

        # Corner plots of the labeled data with display labels and ticks in original units
        plot_corner(
            split.labeled.data,
            feature_labels=PLOT_FEATURE_LABELS,
            tick_overrides=log_tick_overrides(),
            output_directory=output_directory,
        )
        plot_corner_by_category(
            split.labeled.data,
            hue_column="Unit",
            feature_labels=PLOT_FEATURE_LABELS,
            tick_overrides=log_tick_overrides(),
            output_directory=output_directory,
        )


def plot_detrital_vs_groups(*, output_directory: Path | None = OUTPUT_ROOT / DATASET_NAME) -> float:
    """Plots the Detrital zircons against the labeled Plutonic and Volcanic distributions.

    Shows whether the Detrital zircons can be explained as a mixture of the two labeled groups:
    the group contours overlain on the Detrital zircons, and, for each feature, the best single
    mixture of the group densities against the Detrital density (see
    :func:`~bedroc.difference.plotting.plot_corner_with_unlabeled`).

    Args:
        output_directory: Base output directory; the plot goes to its ``overlap`` subdirectory.
            ``None`` for no output. Defaults to ``OUTPUT_ROOT / DATASET_NAME``.

    Returns:
        The best single Plutonic fraction explaining the Detrital zircons as a mixture
    """
    split: LabeledUnlabeledSplit = build_michigan_dataset(output_directory=None)
    weight: float = plot_corner_with_unlabeled(
        split.labeled.data,
        split.unlabeled.data,
        unlabeled_label="Detrital",
        feature_labels=PLOT_FEATURE_LABELS,
        tick_overrides=log_tick_overrides(),
        output_directory=None if output_directory is None else output_directory / "overlap",
    )
    logger.info("Best single-mixture Plutonic fraction for the Detrital zircons: %.3f", weight)

    return weight
