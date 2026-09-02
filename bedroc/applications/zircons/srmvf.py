# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""San Juan volcanic field zircon dataset processing and plotting functions"""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from bedroc import RANDOM_SEED
from bedroc.applications.zircons import srmvf_filepath
from bedroc.applications.zircons.utils import (
    dump_zircon_excel,
    export_zircon_summary,
    load_zircon_excel,
    require_features_present,
)
from bedroc.core.data_container import DataContainer
from bedroc.core.type_aliases import NpArray
from bedroc.difference import DEFAULT_INFERENCE_MODEL, InferenceModel
from bedroc.difference.partitioning import train_test_split
from bedroc.difference.pipelines import run_pipeline as _run_pipeline
from bedroc.difference.plotting import plot_corner, plot_corner_by_category
from bedroc.difference.utils import log_pipeline_run

logger: logging.Logger = logging.getLogger(__name__)

DATASET_NAME: str = "SRMVF"
"""Name for the San Juan volcanic field zircon dataset analysis"""

PLOT_FEATURE_LABELS: Mapping[str, str] = {
    "Ti": "Ti (ppm)",
    "Hf": "Hf (ppm)",
    "Th": "Th (ppm)",
    "U": "U (ppm)",
}
"""Display labels (with units) for the San Juan volcanic field zircon dataset features"""

_LOG_TICK_VALUES: Mapping[str, Sequence[int]] = {
    "Ti (ppm)": (10, 100, 500),
    "Th (ppm)": (10, 100, 1000, 5000),
    "U (ppm)": (10, 100, 1000, 5000),
}
PLOT_TICK_OVERRIDES: Mapping[str, tuple[NpArray, Sequence[str]]] = {
    label: (np.log(values), [f"{v:g}" for v in values])
    for label, values in _LOG_TICK_VALUES.items()
}
"""Un-transforms the log-scaled features back to their original concentration units for display"""


def process_SRMVF(name: str, *, output_directory: Path | None) -> DataContainer:
    """Processes the San Juan volcanic field zircon dataset.

    Processes the raw Excel data into a form that can be used for analysis and creates summary
    statistics.

    Args:
        name: Name for the dataset
        output_directory: Directory to save the processed data. ``None`` for no output.

    Returns:
        A DataContainer object containing the data
    """
    # Parameters
    name_columns: list[str] = ["Sample_name", "Type", "alternate_id"]
    """Extra columns to keep in addition to the feature columns"""
    feature_columns_map: dict[str, str] = {
        "Ti_ppm_m49": "Ti",
        "Hf_ppm_m178": "Hf",
        "Th_ppm_m232": "Th",
        "U_ppm_m238": "U",
        # "Ce_ppm_m140", "Eu_ppm_m151" # not available for plutonic
    }
    """Feature columns to use for analysis. Keys are original names and values are the new names to
    use"""
    feature_columns: list[str] = list(feature_columns_map.keys())
    uncertainty_suffix: str = "_Int2SE"
    """Suffix for uncertainty columns, as they appear in the raw sheet."""

    df, uncertainty_columns = load_zircon_excel(
        srmvf_filepath,
        sheet_name="Table S1_SRMVF Zircons",
        name_columns=name_columns,
        feature_columns=feature_columns,
        uncertainty_suffixes=(uncertainty_suffix,),
        extra_renames={"alternate_id": "Locality"},
    )

    # Raw data is always raw  (not log transformed)
    dump_zircon_excel(df, output_directory, f"{name}_raw.xlsx")

    # Require all these features to be present
    df = require_features_present(df, feature_columns)

    # Filtering criteria from Olivier and Tobias (7/8/2026)
    logger.info("Applying filtering criteria to the data")

    if "Ti_ppm_m49" in feature_columns:
        Ti_max = 200  # or 300
        logger.info("Removing Ti_ppm_m49 values greater than %d ppm", Ti_max)
        ti = df["Ti_ppm_m49"]
        mask = ti.isna() | ((ti < Ti_max) & (ti > 0))
        df = df.loc[mask]
        # Log transform to mitigate right skewness
        df[uncertainty_columns["Ti_ppm_m49"]] = df[uncertainty_columns["Ti_ppm_m49"]] / df["Ti_ppm_m49"]
        df["Ti_ppm_m49"] = np.log(df["Ti_ppm_m49"])

    if "Hf_ppm_m178" in feature_columns:
        Hf_min = 5000
        logger.info("Removing Hf_ppm_m178 values less than %d ppm", Hf_min)
        hf = df["Hf_ppm_m178"]
        mask = hf.isna() | (hf > Hf_min)
        df = df.loc[mask]

    if "Th_ppm_m232" in feature_columns:
        Th_max = 2000
        logger.info("Removing Th_ppm_m232 values greater than %d ppm", Th_max)
        th = df["Th_ppm_m232"]
        mask = th.isna() | (th < Th_max)
        df = df.loc[mask]
        # Log transform to mitigate right skewness
        df[uncertainty_columns["Th_ppm_m232"]] = (
            df[uncertainty_columns["Th_ppm_m232"]] / df["Th_ppm_m232"]
        )
        df["Th_ppm_m232"] = np.log(df["Th_ppm_m232"])

    if "U_ppm_m238" in feature_columns:
        U_max = 2000
        logger.info("Removing U_ppm_m238 values greater than %d ppm", U_max)
        u = df["U_ppm_m238"]
        mask = u.isna() | (u < U_max)
        df = df.loc[mask]
        # Log transform to mitigate right skewness
        df[uncertainty_columns["U_ppm_m238"]] = (
            df[uncertainty_columns["U_ppm_m238"]] / df["U_ppm_m238"]
        )
        df["U_ppm_m238"] = np.log(df["U_ppm_m238"])

    # NOTE: Remove the Pomeroy Inner Border Subunit locality because it is probably a mixture of
    # plutonic and volcanic zircons (not a simple label).
    df = df.loc[df["Locality"] != "Pomeroy Inner Border Subunit"]

    dump_zircon_excel(df, output_directory, f"{name}_processed.xlsx")
    export_zircon_summary(
        df,
        output_directory=output_directory,
        name=name,
        groupby_columns=["Type", "Locality"],
        feature_columns=feature_columns,
    )

    # Create a DataContainer to hold the data and feature information
    data_container: DataContainer = DataContainer.from_dataframe(
        df,
        name=name,
        feature_columns=feature_columns_map,
        uncertainty_columns={
            raw_uncertainty: feature_columns_map[raw_feature]
            for raw_feature, raw_uncertainty in uncertainty_columns.items()
        },
        select_data_column="Sample_name",
        uncertainty_scale=2,
        category_column="Type",
    )

    return data_container


def run_pipeline(
    inference: InferenceModel = DEFAULT_INFERENCE_MODEL,
    *,
    output_directory: Path | None = Path(DATASET_NAME),
    random_seed: int | None = RANDOM_SEED,
) -> None:
    """Runs the inference pipeline for the San Juan volcanic field zircon dataset analysis.

    Args:
        inference: Type of inference to run. Defaults to :obj:`DEFAULT_INFERENCE_MODEL`.
        output_directory: Directory to save the processed data. Defaults to :obj:`DATASET_NAME`.
        random_seed: Seed for random number generation to enable reproducibility. Defaults to
            :obj:`RANDOM_SEED`.
    """
    with log_pipeline_run(f"SRMVF zircon analysis pipeline with inference: {inference}"):
        if output_directory is not None:
            output_directory = output_directory / Path(f"{inference}_seed_{random_seed}")
            output_directory.mkdir(parents=True, exist_ok=True)

        data: DataContainer = process_SRMVF(name=DATASET_NAME, output_directory=output_directory)

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
                tick_overrides=PLOT_TICK_OVERRIDES,
                output_directory=output_directory,
            )
            plot_corner_by_category(
                subset,
                hue_column="Locality",
                feature_labels=PLOT_FEATURE_LABELS,
                output_directory=output_directory,
            )
