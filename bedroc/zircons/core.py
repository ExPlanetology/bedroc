# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Helper functions and classes for zircons"""

import logging
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split

from bedroc.core import DataContainer
from bedroc.hierarchical_group import HierarchicalGroupModel
from bedroc.type_aliases import NpFloat, NpInt
from bedroc.zircons import srmvf_filepath

logger: logging.Logger = logging.getLogger(__name__)

RANDOM_SEED: int = 123

savefig_kwargs = {"dpi": 300, "bbox_inches": "tight", "format": "pdf"}
"""Figure options for savefig"""
ext: str = savefig_kwargs["format"]
"""Extension for figures"""
save_data: bool = True
"""Save data if True"""


def process_SRMVF() -> tuple[HierarchicalGroupModel, NpFloat, NpInt, NpFloat]:
    """Processes and plots the San Juan volcanic field zircon dataset

    Processes the raw Excel data into a form that can be used for analysis and creates summary
    statistics and plots.

    Returns:
        A HierarchicalGroupModel trained on the data along with the test data for evaluation
    """

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
    test_size = 0.2
    """Test size for train-test split"""
    output_directory: Path = Path(f"{name}")
    """Output directory for saving summary statistics and figures"""

    # Process the Excel data so it can be used for analysis
    logger.info("Reading data: %s", datapath)
    df: pd.DataFrame = pd.read_excel(datapath, sheet_name="Table S1_SRMVF Zircons")

    # Select required columns for analysis
    std_columns: list[str] = [f"{feature}{uncertainty_suffix}" for feature in feature_columns]
    df = df[name_columns + feature_columns + std_columns]

    # Capitalize volcanic and plutonic group names for consistency
    df["Type"] = df["Type"].str.capitalize()
    group_names: list[str] = df["Type"].unique().tolist()

    # Rename alternate_id to locality for clarity
    df.rename(columns={"alternate_id": "Locality"}, inplace=True)

    # We must append a suffix to identify the feature columns from the other columns
    rename_map: dict[str, str] = {col: f"{col}_feature" for col in feature_columns}
    df.rename(columns=rename_map, inplace=True)

    if save_data:
        df.to_excel(output_directory / Path(f"{name}_raw.xlsx"))

    # TODO: Basic filtering to remove outliers and missing data, but could be improved.
    # Drop any rows with missing column data
    df.dropna(inplace=True)

    # TODO: Check with Tobias about filtering criteria for Ti and Hf values
    Ti_max = 200000
    df = df[df["Ti_ppm_m49_feature"] < Ti_max]  # Remove some high Ti values (outliers?)
    Hf_min = 2000
    df = df[df["Hf_ppm_m178_feature"] > Hf_min]  # Remove some low Hf values (outliers?)
    Th_max = 3000
    df = df[df["Th_ppm_m232_feature"] < Th_max]  # Remove some high Th values (outliers?)
    U_max = 3000
    df = df[df["U_ppm_m238_feature"] < U_max]  # Remove some high U values (outliers?)

    # Convenient to have a numerical column for the group label to use for analysis
    group_map: dict[str, int] = {name: index for index, name in enumerate(group_names)}
    logger.info("Group mapping: %s", group_map)
    df["group_idx"] = df["Type"].map(group_map)

    # Create a DataContainer to hold the data and feature information
    data: DataContainer = DataContainer(
        df, data_column="Sample_name", feature_std_suffix=uncertainty_suffix, std_scale=2
    )

    # This is the main dataframe that will be used for plotting and analysis
    df = data.get_dataframe(standardized=True)
    if save_data:
        df.to_excel(output_directory / Path(f"{name}_processed.xlsx"))

    # Plotting and outputs

    # Plutonic versus volcanic pairplot
    g: sns.PairGrid = sns.PairGrid(
        df, hue="Type", vars=data.feature_columns, corner=False, diag_sharey=False
    )
    g.map_diag(sns.kdeplot, fill=True, alpha=0.6, common_norm=False)
    g.map_upper(sns.scatterplot, alpha=0.4, s=20)
    g.map_lower(sns.kdeplot, levels=4)
    g.add_legend()
    sns.move_legend(g, "upper left", bbox_to_anchor=(0.18, 0.9), frameon=True)
    g.figure.suptitle(f"{name}: Volcanic vs Plutonic")
    g.figure.tight_layout()
    if save_data:
        g.figure.savefig(
            output_directory / Path(f"{name}_volcanic_vs_plutonic_pairplot.{ext}"),
            **savefig_kwargs,
        )

    # Pair plots for each type, colored by locality, with KDE overlays
    for typ in group_names:
        g: sns.PairGrid = sns.PairGrid(
            df[df["Type"] == typ],
            hue="Locality",
            vars=data.feature_columns,
            corner=False,
            diag_sharey=False,
        )
        g.map_diag(sns.kdeplot, fill=True, alpha=0.6, common_norm=False)
        g.map_lower(sns.scatterplot, alpha=0.4, s=20)
        g.map_upper(sns.kdeplot, levels=4)

        for ax, var in zip(g.diag_axes, data.feature_columns):  # type: ignore
            sns.kdeplot(
                data=df[df["Type"] != typ],
                x=var,
                ax=ax,
                color="black",
                linewidth=2,
                fill=False,
                linestyle="--",
            )
            sns.kdeplot(
                data=df[df["Type"] == typ], x=var, ax=ax, color="black", linewidth=2, fill=False
            )

        for row, yvar in enumerate(data.feature_columns):
            for col, xvar in enumerate(data.feature_columns):
                if row <= col:
                    continue

                ax = g.axes[row, col]

                sns.kdeplot(
                    data=df[df["Type"] == typ],
                    x=xvar,
                    y=yvar,
                    ax=ax,
                    color="black",
                    levels=4,
                    linewidths=1,
                    fill=False,
                )

        g.add_legend()
        sns.move_legend(g, "upper left", bbox_to_anchor=(0.13, 0.95), frameon=True)

        other_type_name: str = [name_ for name_ in group_names if name_ != typ][0]

        line_legend = [
            Line2D([0], [0], color="black", lw=2, label=f"{typ} overall"),
            Line2D([0], [0], color="black", lw=2, ls="--", label=f"{other_type_name} overall"),
        ]

        g.figure.legend(
            handles=line_legend,
            loc="upper left",
            title="Reference",
            frameon=True,
            bbox_to_anchor=(0.16, 0.7),
        )

        g.figure.suptitle(f"{name}: {typ.capitalize()} by locality")
        g.figure.tight_layout()

        if save_data:
            g.figure.savefig(
                output_directory / Path(f"{name}_{typ.lower()}_by_locality_pairplot.{ext}"),
                **savefig_kwargs,
            )

    # Output summary statistics to Excel
    if save_data:
        summary: pd.DataFrame = df.groupby(["Type", "Locality"])[data.feature_columns].describe()
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
