# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""San Juan volcanic field zircon dataset processing and plotting functions"""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from bedroc import RANDOM_SEED
from bedroc.applications.zircons import srmvf_filepath
from bedroc.core.data_container import DataContainer
from bedroc.core.type_aliases import NpArray
from bedroc.difference import DEFAULT_INFERENCE_MODEL, InferenceModel
from bedroc.difference.pipelines import run_pipeline as _run_pipeline
from bedroc.difference.plotting import plot_corner

logger: logging.Logger = logging.getLogger(__name__)

DATASET_NAME: str = "SRMVF"
"""Name for the San Juan volcanic field zircon dataset analysis"""
CATEGORY_NAMES: tuple[str, str] = ("Plutonic", "Volcanic")
"""Category names for the San Juan volcanic field zircon dataset analysis

Anchoring the category names prevents the order from changing, which can then feed into color
changes in plots rendering them inconsistent with each other."""

logger.info("Category names: %s", CATEGORY_NAMES)

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
    datapath: Path = srmvf_filepath
    """Data path for the San Juan volcanic field zircon dataset"""
    name_columns: list[str] = ["Sample_name", "Type", "alternate_id"]
    """Extra columns to keep in addition to the feature columns"""
    feature_columns: dict[str, str] = {
        "Ti_ppm_m49": "Ti",
        "Hf_ppm_m178": "Hf",
        "Th_ppm_m232": "Th",
        "U_ppm_m238": "U",
        # "Ce_ppm_m140", "Eu_ppm_m151" # not available for plutonic
    }
    """Feature columns to use for analysis. Keys are original names and values are the new names to
    use"""
    uncertainty_suffix: str = "_Int2SE"
    """Original suffix for uncertainty columns, which is appended to the feature column names"""
    feature_suffix: str = "_feature"
    """Output suffix for feature columns, which is appended to the feature column names"""

    # Process the Excel data so it can be used for analysis
    logger.info("Reading data: %s", datapath)
    df: pd.DataFrame = pd.read_excel(datapath, sheet_name="Table S1_SRMVF Zircons")

    # Important to lock in the index name for later use in the analysis, Underscore denotes private
    # usage to avoid conflicts with other columns
    df.index.name = "_index"

    # Select required columns for analysis
    std_columns: list[str] = [
        f"{feature}{uncertainty_suffix}" for feature in feature_columns.keys()
    ]
    df = df.loc[:, name_columns + list(feature_columns.keys()) + std_columns]

    # Capitalize volcanic and plutonic group names for consistency
    df["Type"] = df["Type"].str.capitalize()

    # Rename alternate_id to Locality for clarity
    df.rename(columns={"alternate_id": "Locality"}, inplace=True)

    # We must append a suffix to identify the feature columns from the other columns
    rename_map: dict[str, str] = {col: f"{col}{feature_suffix}" for col in feature_columns.keys()}
    df.rename(columns=rename_map, inplace=True)
    new_feature_columns: list[str] = list(rename_map.values())

    # Raw data is always raw  (not log transformed)
    if output_directory is not None:
        df.to_excel(output_directory / Path(f"{name}_raw.xlsx"))

    # Require all these features to be present
    required_features: list[str] = [
        f"Ti_ppm_m49{feature_suffix}",
        f"Hf_ppm_m178{feature_suffix}",
        f"Th_ppm_m232{feature_suffix}",
        f"U_ppm_m238{feature_suffix}",
    ]
    df.dropna(subset=required_features, how="any", inplace=True)

    # Filtering criteria from Olivier and Tobias (7/8/2026)
    logger.info("Applying filtering criteria to the data")

    if f"Ti_ppm_m49{feature_suffix}" in new_feature_columns:
        Ti_max = 200  # or 300
        logger.info("Removing Ti_ppm_m49 values greater than %d ppm", Ti_max)
        ti = df[f"Ti_ppm_m49{feature_suffix}"]
        mask = ti.isna() | ((ti < Ti_max) & (ti > 0))
        df = df.loc[mask]
        # Log transform to mitigate right skewness
        df[f"Ti_ppm_m49{uncertainty_suffix}"] = (
            df[f"Ti_ppm_m49{uncertainty_suffix}"] / df[f"Ti_ppm_m49{feature_suffix}"]
        )
        df[f"Ti_ppm_m49{feature_suffix}"] = np.log(df[f"Ti_ppm_m49{feature_suffix}"])

    if f"Hf_ppm_m178{feature_suffix}" in new_feature_columns:
        Hf_min = 5000
        logger.info("Removing Hf_ppm_m178 values less than %d ppm", Hf_min)
        hf = df[f"Hf_ppm_m178{feature_suffix}"]
        mask = hf.isna() | (hf > Hf_min)
        df = df.loc[mask]

    if f"Th_ppm_m232{feature_suffix}" in new_feature_columns:
        Th_max = 2000
        logger.info("Removing Th_ppm_m232 values greater than %d ppm", Th_max)
        th = df[f"Th_ppm_m232{feature_suffix}"]
        mask = th.isna() | (th < Th_max)
        df = df.loc[mask]
        # Log transform to mitigate right skewness
        df[f"Th_ppm_m232{uncertainty_suffix}"] = (
            df[f"Th_ppm_m232{uncertainty_suffix}"] / df[f"Th_ppm_m232{feature_suffix}"]
        )
        df[f"Th_ppm_m232{feature_suffix}"] = np.log(df[f"Th_ppm_m232{feature_suffix}"])

    if f"U_ppm_m238{feature_suffix}" in new_feature_columns:
        U_max = 2000
        logger.info("Removing U_ppm_m238 values greater than %d ppm", U_max)
        u = df[f"U_ppm_m238{feature_suffix}"]
        mask = u.isna() | (u < U_max)
        df = df.loc[mask]
        # Log transform to mitigate right skewness
        df[f"U_ppm_m238{uncertainty_suffix}"] = (
            df[f"U_ppm_m238{uncertainty_suffix}"] / df[f"U_ppm_m238{feature_suffix}"]
        )
        df[f"U_ppm_m238{feature_suffix}"] = np.log(df[f"U_ppm_m238{feature_suffix}"])

    # NOTE: Remove the Pomeroy Inner Border Subunit locality because it is probably a mixture of
    # plutonic and volcanic zircons (not a simple label).
    df = df.loc[df["Locality"] != "Pomeroy Inner Border Subunit"]

    if output_directory is not None:
        df.to_excel(output_directory / Path(f"{name}_processed.xlsx"))

    # Output summary statistics to Excel
    if output_directory is not None:
        summary = df.groupby(["Type", "Locality"])[new_feature_columns].describe()
        summary_filepath: Path = output_directory / Path(f"{name}_summary.xlsx")
        summary.to_excel(summary_filepath)
        logger.info("Summary statistics saved to %s", summary_filepath)

    # Create a DataContainer to hold the data and feature information
    data_container: DataContainer = DataContainer.from_dataframe(
        df,
        name=name,
        feature_suffix=feature_suffix,
        uncertainty_suffix=uncertainty_suffix,
        select_data_column="Sample_name",
        uncertainty_scale=2,
        feature_renames=feature_columns,
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
    logger.info("Running SRMVF zircon analysis pipeline with inference: %s", inference)

    if output_directory is not None:
        output_directory = output_directory / Path(f"{inference}_seed_{random_seed}")
        output_directory.mkdir(parents=True, exist_ok=True)

    data: DataContainer = process_SRMVF(name=DATASET_NAME, output_directory=output_directory)

    plot_corner(
        data,
        feature_labels=PLOT_FEATURE_LABELS,
        hue_column="Locality",
        tick_overrides=PLOT_TICK_OVERRIDES,
        output_directory=output_directory,
    )

    train, test = data.train_test_split(random_state=random_seed)

    # Corner plots for the train/test split alone, to check the split didn't skew either subset's
    # feature distributions relative to the full dataset plotted above
    plot_corner(
        train,
        feature_labels=PLOT_FEATURE_LABELS,
        hue_column="Locality",
        tick_overrides=PLOT_TICK_OVERRIDES,
        output_directory=output_directory,
    )
    plot_corner(
        test,
        feature_labels=PLOT_FEATURE_LABELS,
        hue_column="Locality",
        tick_overrides=PLOT_TICK_OVERRIDES,
        output_directory=output_directory,
    )

    kwargs: dict = {
        "output_directory": output_directory,
        "random_seed": random_seed,
    }

    # The "covariance" pipeline derives category names directly from the DataContainer (whose
    # category ordering is locked and preserved across train/test splits), so it no longer takes
    # an explicit category_names argument. The other inference modes still need it.
    if inference != "covariance":
        kwargs["category_names"] = CATEGORY_NAMES

    _run_pipeline(data, inference=inference, **kwargs)

    logger.info("SRMVF zircon analysis pipeline completed with inference: %s", inference)
