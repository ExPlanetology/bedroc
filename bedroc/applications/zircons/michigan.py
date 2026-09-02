#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Michigan dataset processing and plotting functions"""

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from bedroc import RANDOM_SEED
from bedroc.applications.zircons import michigan_barth, michigan_hendrickx
from bedroc.core.data_container import DataContainer
from bedroc.difference import DEFAULT_INFERENCE_MODEL, InferenceModel
from bedroc.difference.utils import log_pipeline_run

logger: logging.Logger = logging.getLogger(__name__)

DATASET_NAME: str = "Michigan"
"""Name for the Michigan zircon dataset analysis"""

DEFAULT_FEATURE_COLUMNS: list[str] = ["Ti", "Hf", "U", "Eu/Eu*", "Ce/Ce*"]
"""Default feature columns for Michigan zircon dataset"""
NAME_COLUMNS: list[str] = ["Sample", "Type", "Unit", "Zircon_number"]
UNCERTAINTY_SUFFIXES: tuple[str, ...] = ("±2SE(int)", "±Error")
"""Candidate suffixes for a feature's uncertainty column. The Michigan dataset does not use a
single uncertainty suffix: element columns (``Ti``, ``Hf``, ``Th``, ``U``) use ``"±2SE(int)"``
while ratio columns (``Eu/Eu*``, ``Ce/Ce*``) use ``"±Error"``."""


def find_uncertainty_column(feature: str, columns: Iterable[str]) -> str:
    """Finds a feature's uncertainty column by trying each of :obj:`UNCERTAINTY_SUFFIXES` in turn.

    Args:
        feature: Bare feature column name (e.g. ``"Ti"`` or ``"Eu/Eu*"``).
        columns: Columns to search for a match (e.g. a dataframe's ``.columns``).

    Returns:
        The matching uncertainty column name.

    Raises:
        ValueError: If no candidate suffix produces a column present in ``columns``.
    """
    columns = set(columns)
    for suffix in UNCERTAINTY_SUFFIXES:
        candidate: str = f"{feature}{suffix}"
        if candidate in columns:
            return candidate

    raise ValueError(
        f"No uncertainty column found for feature {feature!r} "
        f"(tried suffixes: {UNCERTAINTY_SUFFIXES})"
    )


def process_michigan(
    name: str, filepath: Path, *, output_directory: Path | None = None
) -> DataContainer:
    """Processes a Michigan zircon dataset into a :obj:`DataContainer`.

    Processes a Michigan zircon dataset into a :obj:`DataContainer` for downstream analysis.

    Args:
        name: Name for the dataset
        filepath: Path to the Michigan zircon dataset (Excel file)
        output_directory: Directory to save the processed data. Defaults to ``None`` (no saving).

    Returns:
        A :obj:`DataContainer` containing the processed Michigan zircon dataset.
    """

    # Parameters
    logger.info("Reading data: %s", filepath)
    name_columns: list[str] = NAME_COLUMNS
    feature_columns: list[str] = DEFAULT_FEATURE_COLUMNS
    uncertainty_suffix: str = "_Int2SE"
    """Original suffix for uncertainty columns, which is appended to the feature column names"""
    feature_suffix: str = "_feature"
    """Output suffix for feature columns, which is appended to the feature column names"""

    # Process the Excel data so it can be used for analysis
    logger.info("Reading data: %s", filepath)
    df: pd.DataFrame = pd.read_excel(filepath, sheet_name="Data")

    uncertainty_columns: dict[str, str] = {
        feature: find_uncertainty_column(feature, df.columns) for feature in feature_columns
    }
    # Important to lock in the index name for later use in the analysis, Underscore denotes private
    # usage to avoid conflicts with other columns
    df.index.name = "_index"

    # Select required columns for analysis
    df = df.loc[:, name_columns + feature_columns + list(uncertainty_columns.values())]

    # Capitalize type names for consistency
    df["Type"] = df["Type"].str.capitalize()

    # Some feature values are reported as below-detection-limit strings (e.g. "< 3.72"); strip
    # the "<"/">" so they parse as plain floats.
    for feature in feature_columns:
        if df[feature].dtype == object:
            df[feature] = df[feature].astype(str).str.replace(r"[<>]", "", regex=True).str.strip()
            df[feature] = pd.to_numeric(df[feature])

    # Some ratio columns (e.g. Ce/Ce*) contain literal inf from a near-zero denominator in the
    # source spreadsheet; treat these as missing rather than propagating inf into the analysis.
    df[feature_columns] = df[feature_columns].replace([np.inf, -np.inf], np.nan)

    # We must append a suffix to identify the feature columns from the other columns
    rename_map: dict[str, str] = {col: f"{col}{feature_suffix}" for col in feature_columns}
    df.rename(columns=rename_map, inplace=True)
    new_feature_columns: list[str] = list(rename_map.values())

    if output_directory is not None:
        df.to_excel(output_directory / Path(f"{name}_processed.xlsx"))

    # Output summary statistics to Excel
    if output_directory is not None:
        summary = df.groupby(["Type", "Unit"])[new_feature_columns].describe()
        summary_filepath: Path = output_directory / Path(f"{name}_summary.xlsx")
        summary.to_excel(summary_filepath)
        logger.info("Summary statistics saved to %s", summary_filepath)

    # Create a DataContainer to hold the data and feature information
    data_container: DataContainer = DataContainer.from_dataframe(
        df,
        name=name,
        feature_suffix=feature_suffix,
        uncertainty_suffix=uncertainty_suffix,
        # select_data_column=None,
        uncertainty_scale=2,
        category_column="Type",
    )

    return data_container


def run_pipeline(
    inference: InferenceModel = DEFAULT_INFERENCE_MODEL,
    *,
    output_directory: Path | None = Path(DATASET_NAME),
    random_seed: int | None = RANDOM_SEED,
):
    """Runs the inference pipeline for the Michigan zircon dataset analysis.

    Args:
        inference: Type of inference to run. Defaults to :obj:`DEFAULT_INFERENCE_MODEL`.
        output_directory: Directory to save the processed data. Defaults to :obj:`DATASET_NAME`.
        random_seed: Seed for random number generation to enable reproducibility. Defaults to
            :obj:`RANDOM_SEED`.
    """
    with log_pipeline_run(f"Michigan zircon analysis pipeline with inference: {inference}"):
        if output_directory is not None:
            output_directory = output_directory / Path(f"{inference}_seed_{random_seed}")
            output_directory.mkdir(parents=True, exist_ok=True)

        data_barth: DataContainer = process_michigan(
            "barth", michigan_barth, output_directory=output_directory
        )
        data_hendrickx: DataContainer = process_michigan(
            "hendrickx", michigan_hendrickx, output_directory=output_directory
        )

        # _run_pipeline(
        #     data.data,
        #     inference=inference,
        #     output_directory=output_directory,
        #     random_seed=random_seed,
        # )


if __name__ == "__main__":
    run_pipeline()
