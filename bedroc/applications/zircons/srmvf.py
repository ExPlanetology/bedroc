# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""San Juan volcanic field zircon dataset processing and plotting functions"""

import dataclasses
import logging
from pathlib import Path

from bedroc import OUTPUT_ROOT, RANDOM_SEED
from bedroc.applications.zircons import srmvf_filepath
from bedroc.applications.zircons.filters import ZIRCON_FILTER, ZirconFilter
from bedroc.applications.zircons.utils import (
    PLOT_FEATURE_LABELS,
    ZirconSource,
    log_tick_overrides,
    zircon_output_directories,
)
from bedroc.core.data_container import DataContainer
from bedroc.difference import DEFAULT_INFERENCE_MODEL, InferenceModel
from bedroc.difference.partitioning import train_test_split
from bedroc.difference.pipelines import run_pipeline as _run_pipeline
from bedroc.difference.plotting import plot_corner, plot_corner_by_category
from bedroc.difference.utils import log_pipeline_run

logger: logging.Logger = logging.getLogger(__name__)

DATASET_NAME: str = "SRMVF"
"""Name for the San Juan volcanic field zircon dataset analysis"""

SRMVF_SOURCE: ZirconSource = ZirconSource(
    name=DATASET_NAME,
    filepath=srmvf_filepath,
    sheet_name="Table S1_SRMVF Zircons",
    feature_columns={
        "Ti_ppm_m49": "Ti",
        "Hf_ppm_m178": "Hf",
        "Th_ppm_m232": "Th",
        "U_ppm_m238": "U",
        # "Ce_ppm_m140", "Eu_ppm_m151" # not available for plutonic
    },
    name_columns=["Sample_name", "Type", "alternate_id"],
    uncertainty_suffixes=("_Int2SE",),
    groupby_columns=["Type", "Locality"],
    extra_renames={"alternate_id": "Locality"},
    # The Pomeroy Inner Border Subunit locality is probably a mixture of plutonic and volcanic
    # zircons (not a simple label)
    exclude={"Locality": ["Pomeroy Inner Border Subunit"]},
    select_data_column="Sample_name",
    dump_raw=True,
)
"""The San Juan volcanic field zircon dataset"""


def process_SRMVF(
    name: str = DATASET_NAME,
    *,
    output_directory: Path | None,
    zircon_filter: ZirconFilter = ZIRCON_FILTER,
) -> DataContainer:
    """Processes the San Juan volcanic field zircon dataset.

    Args:
        name: Name for the dataset. Defaults to :obj:`DATASET_NAME`.
        output_directory: Directory to save the processed data. ``None`` for no output.
        zircon_filter: Filtering criteria and transforms to apply. Defaults to
            :obj:`ZIRCON_FILTER`.

    Returns:
        A DataContainer object containing the data
    """
    source: ZirconSource = SRMVF_SOURCE
    if name != source.name:
        source = dataclasses.replace(source, name=name)

    return source.process(output_directory=output_directory, zircon_filter=zircon_filter)


def run_pipeline(
    inference: InferenceModel = DEFAULT_INFERENCE_MODEL,
    *,
    output_directory: Path | None = OUTPUT_ROOT / DATASET_NAME,
    random_seed: int | None = RANDOM_SEED,
) -> None:
    """Runs the inference pipeline for the San Juan volcanic field zircon dataset analysis.

    Args:
        inference: Type of inference to run. Defaults to :obj:`DEFAULT_INFERENCE_MODEL`.
        output_directory: Directory to save the processed data. Defaults to
            ``OUTPUT_ROOT / DATASET_NAME``.
        random_seed: Seed for random number generation to enable reproducibility. Defaults to
            :obj:`RANDOM_SEED`.
    """
    with log_pipeline_run(f"SRMVF zircon analysis pipeline with inference: {inference}"):
        output_directory, output_directory_data = zircon_output_directories(
            output_directory, inference, random_seed
        )

        data: DataContainer = process_SRMVF(output_directory=output_directory_data)

        # Neither the "covariance" nor "two-stage" pipelines take an explicit category_names
        # argument: both build their model via CategoryComparisonBase.from_data_container(), which
        # always derives category_names from the DataContainer itself (whose category ordering is
        # locked and preserved across train/test splits), so passing it explicitly here would
        # collide with that.
        _run_pipeline(
            data, inference=inference, output_directory=output_directory, random_seed=random_seed
        )

        # Corner plots for the full dataset, then the train/test split alone, to check the split
        # didn't skew either subset's feature distributions relative to the full dataset
        for subset in (data, *train_test_split(data, random_state=random_seed)):
            # Although `_run_pipeline` generates the full corner plot, this implements
            # customizations for the labels and ticks. This is code duplication, technically.
            plot_corner(
                subset,
                feature_labels=PLOT_FEATURE_LABELS,
                tick_overrides=log_tick_overrides(),
                output_directory=output_directory,
            )
            plot_corner_by_category(
                subset,
                hue_column="Locality",
                feature_labels=PLOT_FEATURE_LABELS,
                output_directory=output_directory,
            )
