#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Michigan dataset processing and plotting functions"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from bedroc import OUTPUT_ROOT, RANDOM_SEED
from bedroc.applications.zircons import (
    michigan_all,
    michigan_all_vulcan,
    michigan_filters_p_ti_al,
    michigan_lree_discordance,
    michigan_lree_discordance_thu_barth,
    michigan_lree_discordance_thu_barth_vulcan,
)
from bedroc.applications.zircons.utils import (
    PLOT_FEATURE_LABELS,
    ZIRCON_FILTER,
    ZirconFilter,
    ZirconSource,
    dump_zircon_excel,
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


def michigan_source(
    name: str, filepath: Path, *, type_renames: Mapping[str, str] | None = None
) -> ZirconSource:
    """Describes a Michigan zircon source spreadsheet.

    All Michigan sources share the same layout, so only the name, path, and any type renames
    differ.

    Args:
        name: Name for the dataset
        filepath: Path to the Michigan zircon dataset (Excel file)
        type_renames: Optional ``Type`` values to rename after loading (e.g.
            ``{"Vulcan": "Plutonic"}``). Defaults to ``None``.

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
        type_renames=type_renames or {},
    )


@dataclass(frozen=True)
class MichiganDataSource:
    """A selectable Michigan data file.

    Args:
        name: Short name, also used as the output subdirectory (e.g. ``output/Michigan/<name>``)
        filepath: Path to the Excel file
        type_renames: ``Type`` values to rename after loading. Defaults to no renames.
    """

    name: str
    filepath: Path
    type_renames: Mapping[str, str] = field(default_factory=dict)


MICHIGAN_DATA_SOURCES: Mapping[int, MichiganDataSource] = {
    0: MichiganDataSource("all", michigan_all),
    # The separate Vulcan area samples are counted as plutonic, as in the filtered Vulcan file
    1: MichiganDataSource("all_vulcan", michigan_all_vulcan, {"Vulcan": "Plutonic"}),
    2: MichiganDataSource("lree_discordance", michigan_lree_discordance),
    3: MichiganDataSource("lree_discordance_thu_barth", michigan_lree_discordance_thu_barth),
    4: MichiganDataSource(
        "lree_discordance_thu_barth_vulcan", michigan_lree_discordance_thu_barth_vulcan
    ),
    5: MichiganDataSource("filters_p_ti_al", michigan_filters_p_ti_al),
}
"""Selectable Michigan data files, keyed by the number given on the command line (``-k``). The
unfiltered combined file (0) is the default."""

DEFAULT_MICHIGAN_SOURCE: int = 0
"""Default Michigan data source (see :obj:`MICHIGAN_DATA_SOURCES`)"""

UNLABELED_CATEGORIES: tuple[str, ...] = ("Detrital",)
"""The ``Type`` values that make up the unlabeled target population. Other unlabeled types (e.g.
``Unknown``) are left out of the analysis."""


def michigan_source_directory(
    source: int = DEFAULT_MICHIGAN_SOURCE, *, output_directory: Path = OUTPUT_ROOT / DATASET_NAME
) -> Path:
    """Output directory for a Michigan data source.

    Args:
        source: Key of the data source in :obj:`MICHIGAN_DATA_SOURCES`. Defaults to
            :obj:`DEFAULT_MICHIGAN_SOURCE`.
        output_directory: Michigan base output directory. Defaults to
            ``OUTPUT_ROOT / DATASET_NAME``.

    Returns:
        ``<output_directory>/<source name>``
    """
    return output_directory / MICHIGAN_DATA_SOURCES[source].name


def build_michigan_dataset(
    *,
    source: int = DEFAULT_MICHIGAN_SOURCE,
    output_directory: Path | None = None,
    zircon_filter: ZirconFilter = ZIRCON_FILTER,
) -> LabeledUnlabeledSplit:
    """Builds the Michigan zircon dataset from one data source.

    Processes the chosen source spreadsheet and splits it into a labeled comparison pair
    (:obj:`LABELED_CATEGORIES`) and the unlabeled target population (:obj:`UNLABELED_CATEGORIES`,
    i.e. the ``Detrital`` zircons, whose provenance is unknown) via
    :meth:`LabeledUnlabeledSplit.from_data_container`. Zircons of any other type are left out.

    Args:
        source: Key of the data source in :obj:`MICHIGAN_DATA_SOURCES`. Defaults to
            :obj:`DEFAULT_MICHIGAN_SOURCE`.
        output_directory: Directory to save the processed data. Defaults to ``None`` (no saving).
        zircon_filter: Filtering criteria and transforms to apply. Defaults to
            :obj:`ZIRCON_FILTER`.

    Returns:
        The :obj:`LabeledUnlabeledSplit` for the chosen Michigan data source.
    """
    data_source: MichiganDataSource = MICHIGAN_DATA_SOURCES[source]
    logger.info("Michigan data source %d: %s (%s)", source, data_source.name, data_source.filepath)
    data: DataContainer = michigan_source(
        DATASET_NAME, data_source.filepath, type_renames=data_source.type_renames
    ).process(output_directory=output_directory, zircon_filter=zircon_filter)

    split: LabeledUnlabeledSplit = LabeledUnlabeledSplit.from_data_container(
        data,
        categories=LABELED_CATEGORIES,
        name=DATASET_NAME,
        unlabeled_categories=UNLABELED_CATEGORIES,
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
    source: int = DEFAULT_MICHIGAN_SOURCE,
):
    """Runs the analysis pipeline for the Michigan zircon dataset analysis.

    By default the models are fitted to the labeled Plutonic/Volcanic zircons and estimate the
    category fraction of the Detrital zircons, whose true fraction is unknown. With ``validate``,
    the Detrital zircons are left out and a held-out split of the labeled zircons stands in for
    them instead (self-validation, as for the SRMVF data), so the estimate can be scored against
    the known true fraction.

    Args:
        model: Model to fit. Defaults to :obj:`DEFAULT_FIT_MODEL`.
        output_directory: Base directory to save the output. Defaults to
            ``OUTPUT_ROOT / DATASET_NAME``. Each data source gets its own subdirectory (see
            :func:`michigan_source_directory`), and with ``validate`` a further ``validation``
            subdirectory, keeping these runs apart from the Detrital runs.
        random_seed: Seed for random number generation to enable reproducibility. Defaults to
            :obj:`RANDOM_SEED`.
        validate: Whether to self-validate on the labeled zircons only, leaving out the Detrital
            zircons. Defaults to ``False``.
        source: Key of the data source in :obj:`MICHIGAN_DATA_SOURCES`. Defaults to
            :obj:`DEFAULT_MICHIGAN_SOURCE`.
    """
    label: str = "self-validation" if validate else "analysis"
    source_name: str = MICHIGAN_DATA_SOURCES[source].name
    with log_pipeline_run(f"Michigan zircon {label} pipeline ({source_name}) with model: {model}"):
        if output_directory is not None:
            output_directory = michigan_source_directory(source, output_directory=output_directory)
            if validate:
                output_directory = output_directory / "validation"
        output_directory, output_directory_data = run_output_directories(
            output_directory, model, random_seed
        )

        split: LabeledUnlabeledSplit = build_michigan_dataset(
            source=source, output_directory=output_directory_data
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


def plot_detrital_vs_groups(
    *,
    source: int = DEFAULT_MICHIGAN_SOURCE,
    output_directory: Path | None = OUTPUT_ROOT / DATASET_NAME,
) -> float:
    """Plots the Detrital zircons against the labeled Plutonic and Volcanic distributions.

    Shows whether the Detrital zircons can be explained as a mixture of the two labeled groups:
    the group contours overlain on the Detrital zircons, and, for each feature, the best single
    mixture of the group densities against the Detrital density (see
    :func:`~bedroc.difference.plotting.plot_corner_with_unlabeled`).

    Args:
        source: Key of the data source in :obj:`MICHIGAN_DATA_SOURCES`. Defaults to
            :obj:`DEFAULT_MICHIGAN_SOURCE`.
        output_directory: Michigan base output directory; the plot goes to the source's
            ``overlap`` subdirectory (see :func:`michigan_source_directory`). ``None`` for no
            output. Defaults to ``OUTPUT_ROOT / DATASET_NAME``.

    Returns:
        The best single Plutonic fraction explaining the Detrital zircons as a mixture
    """
    split: LabeledUnlabeledSplit = build_michigan_dataset(source=source, output_directory=None)
    weight: float = plot_corner_with_unlabeled(
        split.labeled.data,
        split.unlabeled.data,
        unlabeled_label="Detrital",
        feature_labels=PLOT_FEATURE_LABELS,
        tick_overrides=log_tick_overrides(),
        output_directory=(
            None
            if output_directory is None
            else michigan_source_directory(source, output_directory=output_directory) / "overlap"
        ),
    )
    logger.info(
        "Best single-mixture Plutonic fraction for the Detrital zircons (%s): %.3f",
        MICHIGAN_DATA_SOURCES[source].name,
        weight,
    )

    return weight
