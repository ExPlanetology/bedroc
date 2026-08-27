# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""San Juan volcanic field zircon dataset processing and plotting functions"""

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from bedroc import RANDOM_SEED
from bedroc.applications.zircons import srmvf_filepath
from bedroc.core.data_container import DataContainer
from bedroc.core.plotting import save_figure
from bedroc.difference import DEFAULT_INFERENCE_MODEL, InferenceModel
from bedroc.difference.pipelines import run_pipeline as _run_pipeline

logger: logging.Logger = logging.getLogger(__name__)

DATASET_NAME: str = "SRMVF"
"""Name for the San Juan volcanic field zircon dataset analysis"""
GROUP_NAMES: tuple[str, str] = ("Plutonic", "Volcanic")
"""Group names for the San Juan volcanic field zircon dataset analysis

Anchoring the group names prevents the order from changing, which can then feed into color changes
in plots rendering them inconsistent with each other."""

logger.info("Group names: %s", GROUP_NAMES)


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


def plot_SRMVF_corner(
    data: DataContainer,
    *,
    output_directory: Path | None,
    savefig_kwargs: dict[str, Any] | None = None,
) -> None:
    """Plots a corner plot of the data using seaborn PairGrid.

    Args:
        data: DataContainer object containing the data to plot
        output_directory: Directory to save the plot. ``None`` for no output.
        savefig_kwargs: Override keyword arguments for :func:`matplotlib.pyplot.savefig`.
            Defaults to ``None``.
    """
    # Plotting and outputs
    df: pd.DataFrame = data.get_dataframe()

    plot_feature_names: dict[str, str] = {
        "Ti": "Ti (ppm)",
        "Hf": "Hf (ppm)",
        "Th": "Th (ppm)",
        "U": "U (ppm)",
    }

    # Extract feature values for plotting
    plot_df: pd.DataFrame = cast(pd.DataFrame, df["Values"].copy())

    # Add metadata columns required for grouping
    if "Type" in df["Metadata"]:
        plot_df["Population"] = df["Metadata"]["Type"]

    if "Locality" in df["Metadata"]:
        plot_df["Locality"] = df["Metadata"]["Locality"]

    group1, group2 = data.category_names  # pyright: ignore[reportGeneralTypeIssues]

    def filter_type(typ: str, *, exclude: bool = False) -> pd.DataFrame:
        """Helper function to filter the DataFrame by type, with an option to exclude the type."""
        mask = plot_df["Population"] == typ

        if exclude:
            mask = ~mask

        return plot_df.loc[mask]

    # Add ppm for the features to the column names for clarity in the plots
    plot_df.rename(columns=plot_feature_names, inplace=True)

    # Plutonic versus volcanic pairplot
    g: sns.PairGrid = sns.PairGrid(
        plot_df,
        hue="Population",
        hue_order=data.category_names,
        vars=plot_feature_names.values(),
        corner=False,
        diag_sharey=False,
    )
    # Histogram to reveal any truncation effects in the data, with KDE overlay to show the smoothed
    # distribution shape
    g.map_diag(sns.histplot, fill=True, alpha=0.6, common_norm=True, stat="density")
    g.map_diag(sns.kdeplot, linewidth=2, linestyle="-", common_norm=True)
    g.map_upper(sns.scatterplot, alpha=0.4, s=20)
    g.map_lower(sns.kdeplot, levels=4)  # [0.25, 0.5, 0.75])

    # Replace log-transformed feature tick labels with original concentration values
    log_features: dict[str, tuple[int, ...]] = {
        "Ti (ppm)": (10, 100, 1000, 500),
        "Th (ppm)": (10, 100, 1000, 5000),
        "U (ppm)": (10, 100, 1000, 5000),
    }

    for ax in g.figure.axes:
        for feature, values in log_features.items():
            if ax.get_xlabel() == feature:
                ax.set_xticks(np.log(values))
                ax.set_xticklabels([f"{v:g}" for v in values])

            if ax.get_ylabel() == feature:
                ax.set_yticks(np.log(values))
                ax.set_yticklabels([f"{v:g}" for v in values])

    g.add_legend()
    sns.move_legend(g, "upper right", bbox_to_anchor=(0.4, 0.98), frameon=True)
    # No title required for publication
    # g.figure.suptitle(f"{data.name}: {group2} vs {group1}")
    g.figure.tight_layout()

    save_figure(
        g.figure, Path(f"{data.name}_{group2}_vs_{group1}"), output_directory, savefig_kwargs
    )

    # Pair plots for each type, colored by locality, with KDE overlays
    for typ in data.category_names:  # pyright: ignore[reportOptionalIterable]
        g: sns.PairGrid = sns.PairGrid(
            filter_type(typ),
            hue="Locality",
            vars=plot_feature_names.values(),
            corner=False,
            diag_sharey=False,
        )
        g.map_diag(sns.kdeplot, fill=True, alpha=0.6, common_norm=True)
        g.map_lower(sns.scatterplot, alpha=0.4, s=20)
        g.map_upper(sns.kdeplot, levels=4, common_norm=True)

        for ax, var in zip(g.diag_axes, plot_feature_names.values()):  # pyright: ignore
            sns.kdeplot(
                data=filter_type(typ, exclude=True),
                x=var,
                ax=ax,
                color="black",
                linewidth=2,
                fill=False,
                linestyle="--",
                common_norm=True,
            )
            sns.kdeplot(
                data=filter_type(typ),
                x=var,
                ax=ax,
                color="black",
                linewidth=2,
                fill=False,
                common_norm=True,
            )

        for row, yvar in enumerate(plot_feature_names.values()):
            for col, xvar in enumerate(plot_feature_names.values()):
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
                    common_norm=True,
                )

        g.add_legend()
        sns.move_legend(g, "upper right", bbox_to_anchor=(0.4, 0.98), frameon=True)

        other_type_name: str = [name_ for name_ in data.category_names if name_ != typ][0]

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


def run_pipeline(
    inference: InferenceModel = DEFAULT_INFERENCE_MODEL,
    output_directory: Path | None = Path(DATASET_NAME),
    *,
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

    plot_SRMVF_corner(data, output_directory=output_directory)

    kwargs: dict = {"output_directory": output_directory, "random_seed": random_seed}

    _run_pipeline(data, inference=inference, **kwargs)

    logger.info("SRMVF zircon analysis pipeline completed with inference: %s", inference)
