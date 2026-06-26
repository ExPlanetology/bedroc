# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Helper functions and classes for zircons"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import pymc as pm
import seaborn as sns
from matplotlib.axes import Axes
from sklearn.model_selection import train_test_split

from bedroc import debug_logger, override
from bedroc.containers import DataContainer
from bedroc.core import plot_posterior_predictive, plot_prior_predictive
from bedroc.hierarchical import group_centric_hierarchical_model, plot_confusion_matrix
from bedroc.zircons import srmvf_filepath

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

RANDOM_SEED: int = 123

savefig_kwargs = {"dpi": 300, "bbox_inches": "tight", "format": "pdf"}
"""Figure options for savefig"""


class ZirconDataBase(ABC):
    """Base class for a zircon dataset"""

    name: str
    data: DataContainer
    model: pm.Model
    idata: az.InferenceData

    @property
    def group_names(self) -> list[str]:
        """Returns the group names for the dataset"""
        return ["plutonic", "volcanic"]

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Initializes the dataset and stores the data in a DataContainer"""

    @abstractmethod
    def preprocess(self, *args, **kwargs) -> pd.DataFrame:
        """Preprocesses the data from the raw Excel file and returns a DataFrame"""

    def pairplot(self, standardized: bool = True) -> sns.PairGrid:
        """Creates a pairplot for the features in the data.

        Args:
            standardized: Plot standardized values. Defaults to ``True``.

        Returns:
            PairGrid object from seaborn
        """
        data: pd.DataFrame = self.data.get_dataframe(standardized=standardized)

        pair_grid: sns.PairGrid = sns.pairplot(
            data,
            hue="Type",
            vars=self.data.feature_columns,
            corner=True,
            plot_kws=dict(alpha=0.4, s=20),
            diag_kws=dict(alpha=0.6),
        )

        return pair_grid

    def summary(
        self, to_excel: bool = False, output_dir: Path = Path(), standardized: bool = False
    ) -> pd.DataFrame:
        """Gets a summary of the dataset by type and location.

        Args:
            to_excel: Whether to save the summary to an Excel file. Defaults to ``False``.
            output_dir: Directory to save the Excel file. Defaults to the current directory.
            standardized: Use standardized values. Defaults to ``False``.

        Returns:
            DataFrame of the summary statistics
        """
        data: pd.DataFrame = self.data.get_dataframe(standardized=standardized)
        summary: pd.DataFrame = data.groupby(["Type", "alternate_id"])[
            self.data.feature_columns
        ].describe()

        if to_excel:
            summary_filepath: Path = output_dir / Path(f"{self.name}_summary.xlsx")
            summary.to_excel(summary_filepath)
            logger.info("Summary statistics saved to %s", summary_filepath)

        return summary

    def train_test_split(
        self, test_size: float = 0.2, standardized: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Splits the data into test and training sets.

        This preserves the original class proportions in both the training and test set.

        Args:
            test_size: Size of the test set. Defaults to ``0.2``.
            standardized: Use standardized values. Defaults to ``True``.

        Returns:
            Tuple of (training DataFrame, test DataFrame)
        """
        df_data: pd.DataFrame = self.data.get_dataframe(standardized=standardized)
        df_train, df_test = train_test_split(
            df_data, test_size=test_size, random_state=RANDOM_SEED, stratify=df_data["group_idx"]
        )

        return df_train, df_test

    def plot_confusion_matrix(
        self, idata: az.InferenceData, test_size: float = 0.2, standardized: bool = True
    ):
        _, df_test = self.train_test_split(test_size=test_size, standardized=standardized)

        test_value_np = df_test[self.data.feature_columns].to_numpy()
        test_std_np = df_test[self.data.feature_std_columns].to_numpy()
        test_group_idx = df_test["group_idx"].to_numpy(dtype=int)

        return plot_confusion_matrix(idata, test_value_np, test_group_idx, X_sigma=test_std_np)

    def hierachical_group_model(
        self, test_size: float = 0.2, standardized: bool = True, draws: int = 3000
    ) -> tuple[pm.Model, az.InferenceData]:
        """Trains a hierarchical group model on the data.

        Args:
            test_size: Size of the test set. Defaults to ``0.2``.
            standardized: Use standardized values. Defaults to ``True``.
            draws: Number of draws for the model. Defaults to ``3000``.
        """
        df_train, _ = self.train_test_split(test_size=test_size, standardized=standardized)

        train_value_np = df_train[self.data.feature_columns].to_numpy()
        train_std_np = df_train[self.data.feature_std_columns].to_numpy()
        train_group_idx = df_train["group_idx"].to_numpy(dtype=int)

        # TODO: compare different models
        # model, idata = feature_centric_hierarchical_model(
        model, idata = group_centric_hierarchical_model(
            train_value_np,
            train_group_idx,
            group_names=self.group_names,
            feature_names=self.data.feature_names,
            X_sigma=train_std_np,
            random_seed=RANDOM_SEED,
            draws=draws,
        )

        return model, idata


class ZirconDataSRMVF(ZirconDataBase):
    """Container for Zircon data from the San Juan volcanic field, curated by Tobias Hendrickx"""

    @override
    def __init__(self):
        self.name: str = "SRMVF"
        """Name of the dataset"""
        self.datapath: Path = srmvf_filepath
        """Path to the Excel file containing the data"""
        self.sheet_name: str = "Table S1_SRMVF Zircons"
        """Sheet name in the Excel file containing the data"""
        self.name_columns: list[str] = ["Sample_name", "Type", "alternate_id"]
        """Extra columns to keep in addition to the feature columns"""
        self.feature_columns: list[str] = [
            "Ti_ppm_m49",
            "Hf_ppm_m178",
            "Th_ppm_m232",
            "U_ppm_m238",
            # "Ce_ppm_m140", "Eu_ppm_m151" # not available for plutonic
        ]
        """Feature columns to use for analysis"""
        self.uncertainty_suffix: str = "_Int2SE"
        """Suffix for uncertainty columns, which is appended to the feature column names"""

        df: pd.DataFrame = self.preprocess()

        self.data: DataContainer = DataContainer(
            df, data_column="Sample_name", feature_std_suffix=self.uncertainty_suffix, std_scale=2
        )

    @override
    def preprocess(self) -> pd.DataFrame:
        """Reads and preprocesses the data from the Excel file.

        Returns:
            DataFrame of the data
        """
        logger.info("Reading data: %s", self.datapath)
        df: pd.DataFrame = pd.read_excel(self.datapath, sheet_name=self.sheet_name)

        # Also keep uncertainty columns
        std_columns: list[str] = [
            f"{feature}{self.uncertainty_suffix}" for feature in self.feature_columns
        ]
        df = df[self.name_columns + self.feature_columns + std_columns]

        # We must append a suffix to identify the feature columns from the other columns.
        rename_map: dict[str, str] = {col: f"{col}_feature" for col in self.feature_columns}
        df.rename(columns=rename_map, inplace=True)

        # TODO: Below is some basic filtering to remove outliers and missing data, but this can be
        # improved and extended

        # Drop any rows with missing column data
        df.dropna(inplace=True)

        # TODO: Check with Tobias about filtering criteria for Ti and Hf values
        Ti_max = 200000
        df = df[df["Ti_ppm_m49_feature"] < Ti_max]  # Remove some high Ti values (outliers?)
        Hf_min = 2000
        df = df[df["Hf_ppm_m178_feature"] > Hf_min]  # Remove some low Hf values (outliers?)

        # Convenient to have a numerical column for the group label to use for analysis
        group_map: dict[str, int] = {name: index for index, name in enumerate(self.group_names)}
        df["group_idx"] = df["Type"].map(group_map)

        return df


# TODO
class ZirconDataMichigan(ZirconDataBase):
    """Container for Zircon data from Michigan, curated by Tobias Hendrickx"""


def main():
    """Main function to run the analysis"""

    debug_logger()

    srmvf: ZirconDataBase = ZirconDataSRMVF()
    pipeline(srmvf)
    plt.show()


def pipeline(zircondata: ZirconDataBase):
    """Pipeline function to run the analysis for a given dataset and generate figures

    Args:
        zircondata: ZirconDataBase object containing the dataset and analysis methods
    """
    output_dir: Path = Path(zircondata.name)
    output_dir.mkdir(exist_ok=True)

    zircondata.summary(to_excel=True, output_dir=output_dir)
    model, idata = zircondata.hierachical_group_model()

    # Figures
    ext: str = savefig_kwargs["format"]

    pair_grid: sns.PairGrid = zircondata.pairplot(standardized=True)
    pair_grid.savefig(output_dir / f"{zircondata.name}_pairplot.{ext}", **savefig_kwargs)

    ax: Axes = plot_prior_predictive(model)
    ax.figure.suptitle(f"{zircondata.name} Prior Predictive Check")
    ax.figure.savefig(output_dir / f"{zircondata.name}_prior_predictive.{ext}", **savefig_kwargs)  # type: ignore

    ax: Axes = plot_posterior_predictive(model, idata)
    ax.figure.suptitle(f"{zircondata.name} Posterior Predictive Check")
    ax.figure.savefig(  # type: ignore
        output_dir / f"{zircondata.name}_posterior_predictive.{ext}", **savefig_kwargs
    )

    axes = az.plot_posterior(idata)  # , var_names=["mu"])
    ax = axes.flatten()[0]
    ax.figure.suptitle("Posterior Distributions", fontsize="xx-large")
    # Adjust spacing for suptitle
    ax.figure.tight_layout(rect=(0, 0, 1, 0.98))  # pyright: ignore
    ax.figure.savefig(output_dir / f"{zircondata.name}_posterior.{ext}", **savefig_kwargs)  # pyright: ignore

    axes = az.plot_forest(
        idata,
        var_names=["mu", "delta"],
        combined=True,
        hdi_prob=0.94,
        kind="forestplot",
        # r_hat=True,
    )
    axes[0].axvline(0, linestyle="--", linewidth=1, alpha=0.6)
    axes[0].set_title(
        f"{zircondata.name} Posterior Differences (Volcanic-Plutonic)",
        fontdict={"fontsize": "xx-large"},
    )
    axes[0].figure.savefig(  # pyright: ignore
        output_dir / f"{zircondata.name}_posterior_differences.{ext}", **savefig_kwargs
    )

    # feature_importance(idata, zircondata.X, zircondata.X_sigma)

    zircondata.plot_confusion_matrix(idata)

    # axes = zircondata.analyzer.plot_posterior_effect_size()
    # axes[0].set_title(
    #     f"{zircondata.name} Posterior Effect Sizes ({zircondata.analyzer.difference_str})",
    #     fontdict={"fontsize": "xx-large"},
    # )
    # axes[0].figure.savefig(  # pyright: ignore
    #     output_dir / f"{zircondata.name}_posterior_effect_size.{ext}", **savefig_kwargs
    # )


if __name__ == "__main__":
    main()
