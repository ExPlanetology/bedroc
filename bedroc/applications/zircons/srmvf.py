# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""San Juan volcanic field zircon dataset processing and plotting functions"""

import logging
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from bedroc.applications.zircons import srmvf_filepath
from bedroc.core import RANDOM_SEED, DataContainer, save_figure
from bedroc.difference.group_classifier import GroupClassifierModel
from bedroc.difference.group_difference import HierarchicalGroupDifferenceModel
from bedroc.difference.group_plotter import GroupPlotter

logger: logging.Logger = logging.getLogger(__name__)

GROUP_NAMES: tuple[str, str] = ("Plutonic", "Volcanic")
"""Group names for the San Juan volcanic field zircon dataset analysis

Anchoring the group names prevents the order from changing, which can then feed into color changes
in plots rendering them inconsistent with each other."""

logger.info("Group names: %s", GROUP_NAMES)


def process_SRMVF(
    name: str = "SRMVF",
    *,
    output_directory: Path | None = None,
    dropna_how: Literal["any", "all"] = "any",
    log_transform: bool = False,
) -> DataContainer:
    """Processes the San Juan volcanic field zircon dataset.

    Processes the raw Excel data into a form that can be used for analysis and creates summary
    statistics.

    Args:
        name: Name for the dataset. Defaults to ``SRMVF``.
        output_directory: Directory to save the processed data. Defaults to ``None`` for no output.
        dropna_how: How to drop rows with NaN values. Use``all`` to drop rows with all NaN values
            and ``any`` to drop rows with any NaN values. Defaults to ``any``.
        log_transform: Whether to log transform the numerical data. Defaults to ``True``.

    Returns:
        A DataContainer object containing the data
    """
    # Parameters
    datapath: Path = srmvf_filepath
    """Data path for the San Juan volcanic field zircon dataset"""
    name_columns: list[str] = ["Sample_name", "Type", "alternate_id"]
    """Extra columns to keep in addition to the feature columns"""
    feature_columns: dict[str, str] = {
        "Ti_ppm_m49": "Ti (standardized)",
        "Hf_ppm_m178": "Hf (standardized)",
        "Th_ppm_m232": "Th (standardized)",
        "U_ppm_m238": "U (standardized)",
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
    df = df[name_columns + list(feature_columns.keys()) + std_columns]

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

    # Drop NaN values in the feature columns based on the specified dropna_how parameter
    df.dropna(subset=new_feature_columns, how=dropna_how, inplace=True)

    # Filtering criteria from Olivier and Tobias (7/8/2026)
    Ti_max = 200  # or 200
    ti = df[f"Ti_ppm_m49{feature_suffix}"]
    df = df[ti.isna() | (ti < Ti_max)]
    Hf_min = 5000
    hf = df[f"Hf_ppm_m178{feature_suffix}"]
    df = df[hf.isna() | (hf > Hf_min)]
    Th_max = 2000
    th = df[f"Th_ppm_m232{feature_suffix}"]
    df = df[th.isna() | (th < Th_max)]
    U_max = 2000
    u = df[f"U_ppm_m238{feature_suffix}"]
    df = df[u.isna() | (u < U_max)]

    if log_transform:
        numeric_cols: pd.Index[str] = df.select_dtypes(include="number").columns
        df[numeric_cols] = np.log(df[numeric_cols])

    # Convenient to have a numerical column for the group label to use for analysis
    group_map: dict[str, int] = {name: index for index, name in enumerate(GROUP_NAMES)}
    logger.info("Group mapping: %s", group_map)
    df["group_idx"] = df["Type"].map(group_map)

    # NOTE: Remove the Pomeroy Inner Border Subunit locality because it drives significant overlap
    # with volcanic zircons in the feature space, which is not representative of the overall
    # dataset
    # df = df[df["Locality"] != "Pomeroy Inner Border Subunit"]

    if output_directory is not None:
        df.to_excel(output_directory / Path(f"{name}_processed.xlsx"))

    # Output summary statistics to Excel
    if output_directory is not None:
        summary: pd.DataFrame = df.groupby(["Type", "Locality"])[new_feature_columns].describe()
        summary_filepath: Path = output_directory / Path(f"{name}_summary.xlsx")
        summary.to_excel(summary_filepath)
        logger.info("Summary statistics saved to %s", summary_filepath)

    # Create a DataContainer to hold the data and feature information
    data_container: DataContainer = DataContainer.from_dataframe(
        df,
        name="SRMVF",
        feature_suffix=feature_suffix,
        uncertainty_suffix=uncertainty_suffix,
        data_column="Sample_name",
        uncertainty_scale=2,
        feature_renames=feature_columns,
    )

    return data_container


def plot_SRMVF_corner(
    data: DataContainer,
    *,
    output_directory: Path | None = None,
    savefig_kwargs: dict[str, Any] | None = None,
) -> None:
    """Plots a corner plot of the data using seaborn PairGrid.

    Args:
        data: DataContainer object containing the data to plot
        output_directory: Directory to save the plot. Defaults to ``None`` for no output.
        savefig_kwargs: Override keyword arguments for :func:`matplotlib.pyplot.savefig`.
            Defaults to ``None``.
    """
    # Plotting and outputs
    df: pd.DataFrame = data.get_dataframe()

    # Extract feature values for plotting
    plot_df: pd.DataFrame = cast(pd.DataFrame, df["Values"].copy())

    # Add metadata columns required for grouping
    if "Type" in df["Metadata"]:
        plot_df["Type"] = df["Metadata"]["Type"]

    if "Locality" in df["Metadata"]:
        plot_df["Locality"] = df["Metadata"]["Locality"]

    group1, group2 = GROUP_NAMES

    def filter_type(typ: str, *, exclude: bool = False) -> pd.DataFrame:
        """Helper function to filter the DataFrame by type, with an option to exclude the type."""
        mask = plot_df["Type"] == typ

        if exclude:
            mask = ~mask

        return plot_df[mask]

    # Plutonic versus volcanic pairplot
    g: sns.PairGrid = sns.PairGrid(
        plot_df,
        hue="Type",
        hue_order=GROUP_NAMES,
        vars=data.feature_names,
        corner=False,
        diag_sharey=False,
    )
    # Histogram to reveal any truncation effects in the data, with KDE overlay to show the smoothed
    # distribution shape
    g.map_diag(sns.histplot, fill=True, alpha=0.6, common_norm=False, stat="density")
    g.map_diag(sns.kdeplot, linewidth=2, linestyle="-", common_norm=False)
    g.map_upper(sns.scatterplot, alpha=0.4, s=20)
    g.map_lower(sns.kdeplot, levels=4)
    g.add_legend()
    sns.move_legend(g, "upper left", bbox_to_anchor=(0.18, 0.9), frameon=True)
    g.figure.suptitle(f"{data.name}: {group2} vs {group1}")
    g.figure.tight_layout()

    save_figure(
        g.figure, Path(f"{data.name}_volcanic_vs_plutonic"), output_directory, savefig_kwargs
    )

    # Pair plots for each type, colored by locality, with KDE overlays
    for typ in GROUP_NAMES:
        g: sns.PairGrid = sns.PairGrid(
            filter_type(typ),
            hue="Locality",
            vars=data.feature_names,
            corner=False,
            diag_sharey=False,
        )
        g.map_diag(sns.kdeplot, fill=True, alpha=0.6, common_norm=False)
        g.map_lower(sns.scatterplot, alpha=0.4, s=20)
        g.map_upper(sns.kdeplot, levels=4)

        for ax, var in zip(g.diag_axes, data.feature_names):  # pyright: ignore
            sns.kdeplot(
                data=filter_type(typ, exclude=True),
                x=var,
                ax=ax,
                color="black",
                linewidth=2,
                fill=False,
                linestyle="--",
            )
            sns.kdeplot(
                data=filter_type(typ),
                x=var,
                ax=ax,
                color="black",
                linewidth=2,
                fill=False,
            )

        for row, yvar in enumerate(data.feature_names):
            for col, xvar in enumerate(data.feature_names):
                if row <= col:
                    continue

                ax = g.axes[row, col]

                sns.kdeplot(
                    data=filter_type(typ),
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

        other_type_name: str = [name_ for name_ in GROUP_NAMES if name_ != typ][0]

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

        g.figure.suptitle(f"{data.name}: {typ.capitalize()} by locality")
        g.figure.tight_layout()

        save_figure(
            g.figure,
            Path(f"{data.name}_{typ.lower()}_by_locality"),
            output_directory,
            savefig_kwargs,
        )


def run_SRMVF(output_directory: Path | None = Path("SRMVF")) -> None:
    """Main function to run the San Juan volcanic field zircon dataset analysis

    Args:
        output_directory: Directory to save the processed data. Defaults to ``SRMVF``.
    """
    if output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)

    # Run the analysis for the San Juan volcanic field zircon dataset
    data: DataContainer = process_SRMVF(output_directory=output_directory)

    # Plot a corner plot of the data
    plot_SRMVF_corner(data, output_directory=output_directory)

    train, test = data.train_test_split(
        random_state=RANDOM_SEED, stratify=data.metadata["group_idx"]
    )

    if output_directory is not None:
        train.to_excel(output_directory / f"{data.name}_train.xlsx")
        test.to_excel(output_directory / f"{data.name}_test.xlsx")

    # Train a hierarchical group model
    fitted_model = HierarchicalGroupDifferenceModel(
        train.name,
        train.values_std.to_numpy(),
        train.metadata["group_idx"].to_numpy(),
        group_names=GROUP_NAMES,
        feature_names=train.feature_names,
        X_sigma=train.uncertainties_std.to_numpy(),
        output_directory=output_directory,
    )
    fitted_model.run_and_plot()

    classifier: GroupClassifierModel = GroupClassifierModel(
        fitted_model,
        test.values_std.to_numpy(),
        X_sigma=test.uncertainties_std.to_numpy(),
        output_directory=output_directory,
    )

    plotter: GroupPlotter = GroupPlotter(
        classifier,
        group_idx=test.metadata["group_idx"].to_numpy(),
        output_directory=output_directory,
    )
    plotter.confusion_matrix()
    plotter.plot_group_fraction_posterior(prior_alpha=1, prior_beta=1)

    # plt.show()
