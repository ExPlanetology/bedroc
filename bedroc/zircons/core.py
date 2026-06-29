# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Helper functions and classes for zircons"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from bedroc import debug_logger
from bedroc.core import DataContainer
from bedroc.hierarchical_group import HierarchicalGroupModel
from bedroc.type_aliases import NpFloat, NpInt
from bedroc.zircons import srmvf_filepath

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

RANDOM_SEED: int = 123

savefig_kwargs = {"dpi": 300, "bbox_inches": "tight", "format": "pdf"}
"""Figure options for savefig"""


def process_SRMVF() -> tuple[HierarchicalGroupModel, NpFloat, NpInt, NpFloat]:
    """Process the San Juan volanic field zircon dataset"""

    # Parameters
    name: str = "SRMVF"
    """Name of the dataset"""
    datapath: Path = srmvf_filepath
    """Data path for the San Juan volcanic field zircon dataset"""
    name_columns: list[str] = ["Sample_name", "Type", "alternate_id"]
    """Extra columns to keep in addition to the feature columns"""
    feature_columns: list[str] = [
        "Ti_ppm_m49",
        "Hf_ppm_m178",
        "Th_ppm_m232",
        "U_ppm_m238",
        # "Ce_ppm_m140", "Eu_ppm_m151" # not available for plutonic
    ]
    """Feature columns to use for analysis"""
    uncertainty_suffix: str = "_Int2SE"
    """Suffix for uncertainty columns, which is appended to the feature column names"""
    group_names = ["plutonic", "volcanic"]
    """Group names"""
    test_size = 0.2
    """Test size for train-test split"""
    output_directory: Path = Path(f"{name}")
    """Output directory for saving summary statistics and figures"""

    # Process the Excel data so it can be used for analysis
    logger.info("Reading data: %s", datapath)
    df: pd.DataFrame = pd.read_excel(datapath, sheet_name="Table S1_SRMVF Zircons")

    # Keep uncertainty columns
    std_columns: list[str] = [f"{feature}{uncertainty_suffix}" for feature in feature_columns]
    df = df[name_columns + feature_columns + std_columns]

    # We must append a suffix to identify the feature columns from the other columns
    rename_map: dict[str, str] = {col: f"{col}_feature" for col in feature_columns}
    df.rename(columns=rename_map, inplace=True)

    # TODO: Below is some basic filtering to remove outliers and missing data

    # Drop any rows with missing column data
    df.dropna(inplace=True)

    # TODO: Check with Tobias about filtering criteria for Ti and Hf values
    Ti_max = 200000
    df = df[df["Ti_ppm_m49_feature"] < Ti_max]  # Remove some high Ti values (outliers?)
    Hf_min = 2000
    df = df[df["Hf_ppm_m178_feature"] > Hf_min]  # Remove some low Hf values (outliers?)

    # Convenient to have a numerical column for the group label to use for analysis
    group_map: dict[str, int] = {name: index for index, name in enumerate(group_names)}
    df["group_idx"] = df["Type"].map(group_map)

    # Create a DataContainer to hold the data and feature information
    data: DataContainer = DataContainer(
        df, data_column="Sample_name", feature_std_suffix=uncertainty_suffix, std_scale=2
    )
    df = data.get_dataframe(standardized=True)

    # Plotting and outputs
    pair_grid: sns.PairGrid = sns.pairplot(
        df,
        hue="Type",
        vars=data.feature_columns,
        corner=True,
        plot_kws=dict(alpha=0.4, s=20),
        diag_kws=dict(alpha=0.6),
    )

    # Output summary statistics to Excel
    summary: pd.DataFrame = df.groupby(["Type", "alternate_id"])[data.feature_columns].describe()
    summary_filepath: Path = output_directory / Path(f"{name}_summary.xlsx")
    summary.to_excel(summary_filepath)
    logger.info("Summary statistics saved to %s", summary_filepath)

    # Split the data into training and test sets, preserving the original class proportions
    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=RANDOM_SEED, stratify=df["group_idx"]
    )
    train_value_np = df_train[data.feature_columns].to_numpy()
    train_std_np = df_train[data.feature_std_columns].to_numpy()
    train_group_idx = df_train["group_idx"].to_numpy(dtype=int)
    test_value_np = df_test[data.feature_columns].to_numpy()
    test_std_np = df_test[data.feature_std_columns].to_numpy()
    test_group_idx = df_test["group_idx"].to_numpy(dtype=int)

    # Create a hierarchical group model using the training data
    model = HierarchicalGroupModel(
        train_value_np,
        train_group_idx,
        group_names=group_names,
        feature_names=data.feature_names,
        X_sigma=train_std_np,
    )

    return model, test_value_np, test_group_idx, test_std_np


def main():
    """Main function to run the analysis"""

    debug_logger()

    # Run the analysis for the San Juan volcanic field zircon dataset
    model, test_value_np, test_group_idx, test_std_np = process_SRMVF()
    model.run_pipeline(test_value_np, test_group_idx, X_sigma=test_std_np)

    plt.show()


if __name__ == "__main__":
    main()
