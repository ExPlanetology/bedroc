#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Michigan dataset processing and plotting functions"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bedroc import RANDOM_SEED
from bedroc.applications.zircons import (
    michigan_foldenauer,
    michigan_hendrickx,
    michigan_petryk,
    michigan_pray,
    michigan_staudenmann,
)
from bedroc.applications.zircons.utils import (
    dump_zircon_excel,
    export_zircon_summary,
    finalize_feature_columns,
    load_zircon_excel,
    require_features_present,
)
from bedroc.core.data_container import DataContainer
from bedroc.difference import DEFAULT_INFERENCE_MODEL, InferenceModel
from bedroc.difference.partitioning import LabeledUnlabeledSplit
from bedroc.difference.pipelines import run_pipeline as _run_pipeline
from bedroc.difference.utils import log_pipeline_run

logger: logging.Logger = logging.getLogger(__name__)

DATASET_NAME: str = "Michigan"
"""Name for the Michigan zircon dataset analysis"""

DEFAULT_FEATURE_COLUMNS: list[str] = ["Ti", "Hf", "U", "Th", "Eu/Eu*", "Ce/Ce*"]
"""Default feature columns for Michigan zircon dataset"""
NAME_COLUMNS: list[str] = ["Sample", "Type", "Unit", "Zircon_number"]
UNCERTAINTY_SUFFIXES: tuple[str, ...] = ("±2SE(int)", "±Error")
"""Candidate suffixes for a feature's uncertainty column. The Michigan dataset does not use a
single uncertainty suffix: element columns (``Ti``, ``Hf``, ``Th``, ``U``) use ``"±2SE(int)"``
while ratio columns (``Eu/Eu*``, ``Ce/Ce*``) use ``"±Error"``."""
UNCERTAINTY_SUFFIX: str = "_Int2SE"
"""Output suffix for uncertainty columns, which is appended to the feature column names"""
FEATURE_SUFFIX: str = "_feature"
"""Output suffix for feature columns, which is appended to the feature column names"""
LABELED_CATEGORIES: tuple[str, str] = ("Plutonic", "Volcanic")
"""The two ``Type`` values treated as the labeled comparison pair. The remaining ``Type`` value
(``Detrital``, zircons of unknown provenance) is pooled into the unlabeled population."""


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
    feature_columns: list[str] = DEFAULT_FEATURE_COLUMNS

    df, uncertainty_columns = load_zircon_excel(
        filepath,
        sheet_name="Data",
        name_columns=NAME_COLUMNS,
        feature_columns=feature_columns,
        uncertainty_suffixes=UNCERTAINTY_SUFFIXES,
    )

    # Some feature values are reported as below-detection-limit strings (e.g. "< 3.72"); strip
    # the "<"/">" so they parse as plain floats.
    for feature in feature_columns:
        if df[feature].dtype == object:
            df[feature] = df[feature].astype(str).str.replace(r"[<>]", "", regex=True).str.strip()
            df[feature] = pd.to_numeric(df[feature])

    # Some ratio columns (e.g. Ce/Ce*) contain literal inf from a near-zero denominator in the
    # source spreadsheet; treat these as missing rather than propagating inf into the analysis.
    df[feature_columns] = df[feature_columns].replace([np.inf, -np.inf], np.nan)

    df, new_feature_columns = finalize_feature_columns(
        df,
        feature_columns=feature_columns,
        uncertainty_columns=uncertainty_columns,
        feature_suffix=FEATURE_SUFFIX,
        uncertainty_suffix=UNCERTAINTY_SUFFIX,
    )

    # Require all these features to be present
    required_features: list[str] = [f"{feature}{FEATURE_SUFFIX}" for feature in feature_columns]
    df = require_features_present(df, required_features)

    # For compatibility with SRMVF processing log-transform Ti, Th, and U to mitigate right
    # skewness
    for column in ("Ti", "Th", "U"):
        df[f"{column}{UNCERTAINTY_SUFFIX}"] = (
            df[f"{column}{UNCERTAINTY_SUFFIX}"] / df[f"{column}{FEATURE_SUFFIX}"]
        )
        df[f"{column}{FEATURE_SUFFIX}"] = np.log(df[f"{column}{FEATURE_SUFFIX}"])

    dump_zircon_excel(df, output_directory, f"{name}_processed.xlsx")
    export_zircon_summary(
        df,
        output_directory=output_directory,
        name=name,
        groupby_columns=["Type", "Unit"],
        feature_columns=new_feature_columns,
    )

    # Create a DataContainer to hold the data and feature information
    data_container: DataContainer = DataContainer.from_dataframe(
        df,
        name=name,
        feature_suffix=FEATURE_SUFFIX,
        uncertainty_suffix=UNCERTAINTY_SUFFIX,
        uncertainty_scale=2,
        category_column="Type",
    )

    return data_container


def build_michigan_dataset(*, output_directory: Path | None = None) -> LabeledUnlabeledSplit:
    """Builds the combined Michigan zircon dataset from every source spreadsheet.

    Processes each Michigan source dataset via :func:`process_michigan`, concatenates them into a
    single :obj:`DataContainer` via :meth:`DataContainer.concat`, and splits the result into a
    labeled comparison pair (:obj:`LABELED_CATEGORIES`) plus a pooled unlabeled remainder (the
    ``Detrital`` zircons, whose provenance is unknown) via
    :meth:`LabeledUnlabeledSplit.from_data_container`.

    Args:
        output_directory: Directory to save each source's processed data and the combined dataset.
            Defaults to ``None`` (no saving).

    Returns:
        The :obj:`LabeledUnlabeledSplit` for all Michigan zircon sources.
    """
    # Barth is missing Th, so we skip it for now. It can be added back in later if needed.
    # data_barth: DataContainer = process_michigan(
    #    "barth", michigan_barth, output_directory=output_directory
    # )
    data_hendrickx: DataContainer = process_michigan(
        "hendrickx", michigan_hendrickx, output_directory=output_directory
    )
    data_foldenauer: DataContainer = process_michigan(
        "foldenauer", michigan_foldenauer, output_directory=output_directory
    )
    data_petryk: DataContainer = process_michigan(
        "petryk", michigan_petryk, output_directory=output_directory
    )
    data_pray: DataContainer = process_michigan(
        "pray", michigan_pray, output_directory=output_directory
    )
    data_staudenmann: DataContainer = process_michigan(
        "staudenmann", michigan_staudenmann, output_directory=output_directory
    )

    data: DataContainer = DataContainer.concat(
        [
            # data_barth,
            data_hendrickx,
            data_foldenauer,
            data_petryk,
            data_pray,
            data_staudenmann,
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

        split: LabeledUnlabeledSplit = build_michigan_dataset(output_directory=output_directory)

        _run_pipeline(
            split.labeled.data,
            inference=inference,
            unlabeled=split.unlabeled,
            output_directory=output_directory,
            random_seed=random_seed,
        )


if __name__ == "__main__":
    run_pipeline()
