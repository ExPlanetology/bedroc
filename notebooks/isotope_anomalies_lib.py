#
# Copyright 2025 Dan J. Bower
#
# This file is part of Bedroc.
#
# Bedroc is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Bedroc is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Bedroc. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Bespoke helpers for the isotope anomalies analysis"""

import logging
import pickle
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from matplotlib.axes import Axes
from sklearn.decomposition import PCA

from bedroc.containers import DataContainer
from bedroc.pca import bayesian_pca
from bedroc.type_aliases import NpFloat

# TODO: Clean up these file names.

# Isotope data for EC and OC groups in the 'all elements' case
# TODO: Check with Paolo if this can be consolidated with alL_datafile
ISOTOPE_GROUP_FILENAME: str = "data/TableS_NC-CC_multivariate_input_11Jan2025_clean_definitive_final_DJB_VG_03March_OC-EC.csv"

# Data for Bayesian regression (compatible with York regression analysis)
# TODO: Probably to remove
REGRESSION_FILENAME: str = "data/multivariate_input_11Jan2025_clean_definitive_ehaub_EM+FE_DJB.csv"

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class GroupData:
    """Grouped isotope data

    Args:
        name: Name of the group
        datafile: Path to the datafile, which will have ``data_dir`` appended.
        chondrites: Chondries to select. Defaults to ``None`` to select all.
        elements: Elements to select. Defaults to ``None`` to select all.
        data_dir: Data directory. Defaults to ``data``.
    """

    name: str
    datafile: Path
    chondrites: Optional[Iterable[str]] = None
    elements: Optional[Iterable[str]] = None
    data_dir: Path = Path("data")
    data: DataContainer = field(init=False)
    _model: Optional[pm.Model] = field(init=False, default=None)
    _idata: Optional[az.InferenceData] = field(init=False, default=None)

    def __post_init__(self):
        logger.info("Reading data: %s", self.datapath)
        logger.info("Create data container")
        self.data = DataContainer.from_csv(
            self.datapath,
            name=self.name,
            select_features=self.elements,
            select_data=self.chondrites,
            data_column="Chondrite",
        )

    @property
    def datapath(self) -> Path:
        return self.data_dir / self.datafile

    @property
    def idata(self) -> az.InferenceData:
        if self._idata is None:
            raise ValueError(
                "Data not yet generated. Call 'run_bayesian_pca()' first"
            )  # pragma: no cover

        return self._idata

    @property
    def model(self) -> pm.Model:
        if self._model is None:
            raise ValueError(
                "Model not yet generated. Call 'run_bayesian_pca()' first"
            )  # pragma: no cover

        return self._model

    def to_excel(self, dir: Path = Path(".")) -> Path:
        """Exports the inference data to Excel

        Args:
            dir: Directory to export to. Defaults to the current directory.
        """
        filepath: Path = dir / Path(f"{self.name}_idata_summary.xlsx")
        az.summary(self.idata).to_excel(filepath)
        logger.info("Inference summary exported to Excel: %s", filepath)

        return filepath

    def to_pickle(self, dir: Path = Path(".")) -> Path:
        """Exports the inference data to a pickle file

        Args:
            dir: Directory to export to. Defaults to the current directory.
        """
        filepath: Path = dir / Path(f"{self.name}_idata.pkl")
        with open(filepath, "wb") as handle:
            pickle.dump(self.idata, handle, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Inference data exported to pickle: %s", filepath)

        return filepath

    def run_bayesian_pca(self, random_seed: Optional[int] = None) -> None:
        """Runs the Bayesian PCA

        Args:
            random_seed: Optional random seed.
        """
        model, idata = bayesian_pca(
            self.data.get_feature_values(),
            self.data.get_feature_stds(),
            random_seed=random_seed,
            feature_labels=self.elements,
        )

        # Store internally
        self._model = model
        self._idata = idata


# Old
# all_datafile = data_dir / Path(
#     "TableS_NC-CC_multivariate_input_11Jan2025_clean_definitive_final_DJB_VG_03March.csv"
# )
group_all: GroupData = GroupData(
    name="all",
    datafile=Path("NEW_TableS_NC-CC.csv"),
    chondrites=(
        "CI",
        "CM",
        "CO",
        "CV",
        "CR",
        "H",
        "L",
        "LL",
        "EH",
        "EL",
        "Ureilites",
        "Mars",
        "Earth",
        "Vesta Group",
    ),
    elements=("48Ca", "50Ti", "54Cr", "54Fe", "64Ni", "66Zn", "94Mo", "95Mo", "96Zr", "100Ru"),
)
group_vesta_3_systems: GroupData = GroupData(
    name="vesta_3_systems",
    # TODO: Check with Paolo if this can be consolidated with alL_datafile
    datafile=Path("NC_CC_AllData_R1_NCbodies_3_systems.csv"),
    elements=("50Ti", "54Cr", "96Zr"),
)
group_vesta_4_systems: GroupData = GroupData(
    name="vesta_4_systems",
    datafile=Path("NC_CC_AllData_R1_NCbodies_4_systems.csv"),
    elements=("48Ca", "50Ti", "54Cr", "96Zr"),
)
group_heavy: GroupData = GroupData(
    name="heavy", datafile=Path("NEW_TableS_NC-CC.csv"), elements=("94Mo", "95Mo", "96Zr", "100Ru")
)
group_iron_peak: GroupData = GroupData(
    name="iron-peak",
    datafile=Path("NEW_TableS_NC-CC.csv"),
    elements=("48Ca", "50Ti", "54Cr", "54Fe", "64Ni", "66Zn"),
)
group_siderophile: GroupData = GroupData(
    name="siderophile",
    datafile=Path("NEW_TableS_NC-CC.csv"),
    elements=("54Fe", "64Ni", "94Mo", "95Mo", "100Ru"),
)
group_lithophile: GroupData = GroupData(
    name="lithophile",
    datafile=Path("NEW_TableS_NC-CC.csv"),
    elements=("48Ca", "50Ti", "54Cr", "66Zn", "96Zr"),
)
group_iron_set: GroupData = GroupData(
    name="iron-set",
    datafile=Path("NEW_TableS_NC-CC.csv"),
    elements=("54Fe", "64Ni", "94Mo", "100Ru"),
    chondrites=(
        "CI",
        "CM",
        "CO",
        "CV",
        "CR",
        "IIC",
        "IID",
        "IVB",
        "H",
        "L",
        "LL",
        "EH",
        "EL",
        "Ureilites",
        "Mars",
        "IAB",
        "IC",
        "IIAB",
        "IIIAB",
        "IVA",
        "Earth",
    ),
)
group_chromium_set: GroupData = GroupData(
    name="chromium-set",
    datafile=Path("NEW_TableS_NC-CC.csv"),
    elements=("54Cr", "94Mo", "95Mo", "100Ru"),
    chondrites=(
        "CI",
        "CM",
        "CO",
        "CV",
        "CK",
        "CR",
        "CH",
        "H",
        "L",
        "LL",
        "EH",
        "EL",
        "Ureilites",
        "Acapulcoite",
        "Aubrites",
        "Winonanites",
        "MG Pallasites",
        "Mesosiderite",
        "Mars",
        "IIAB",
        "IIIAB",
        "IVA",
        "Earth",
    ),
)


def get_group_data() -> dict[str, GroupData]:
    """Gets the group data"""

    group_data: dict[str, GroupData] = {
        "all": group_all,
        "vesta_3_systems": group_vesta_3_systems,
        "vesta_4_systems": group_vesta_4_systems,
        "heavy": group_heavy,
        "iron-peak": group_iron_peak,
        "siderophile": group_siderophile,
        "lithophile": group_lithophile,
        "iron-set": group_iron_set,
        "chromium-set": group_chromium_set,
    }

    return group_data


def is_vesta_group(chondrite_name: str) -> bool:
    """Checks if a chondrite is part of the Vesta Group.

    Args:
        chondrite_name: Chondrite name

    Returns:
        ``True`` if part of the Vesta Group, otherwise ``False``
    """
    vesta_group: tuple[str, ...] = (
        "Vesta Group",
        "Ureilites",
        "Vesta",
        "Angrites",
        "MG Pallasites",
        "Mesosiderite",
        "Acapulcoite",
    )

    is_vesta_group: bool = chondrite_name in vesta_group

    return is_vesta_group


def get_color(chondrite_name: str, reservoir_name: str) -> tuple[str, str]:
    """Gets the color and the light color associated with the chondrite and/or reservoir.

    Args:
        chondrite_name: Chondrite name
        reservoir_name: Reservoir name

    Returns:
        tuple:
            - Color
            - Light color
    """
    if is_vesta_group(chondrite_name):
        return ("orangered", "orange")
    elif chondrite_name == "CI":
        return ("purple", "orchid")
    elif reservoir_name == "CC":
        return ("blue", "lightblue")
    elif reservoir_name == "NC":
        return ("red", "lightsalmon")
    elif reservoir_name == "BSE":
        return ("green", "lightgreen")
    else:
        raise ValueError(
            "chondrite: %s and reservoir: %s have no color assignment",
            chondrite_name,
            reservoir_name,
        )


def plot_pca(
    data: DataContainer,
    idata: az.InferenceData,
    ax: Axes,
    plot_legend: bool = True,
    x_label: bool = True,
    y_label: bool = True,
    title_prefix: str = "",
    title: Optional[str] = None,
    skip: int = 10,
    n_components: int = 2,
    plot_eigenvectors: bool = True,
) -> None:
    """Plots the Bayesian and deterministic PCA

    Args:
        data: Data container
        idata: Inference data
        ax: Axis
        plot_legend: Plots the legend. Defaults to ``True``.
        x_label: Plot the x-axis label. Defaults to ``True``.
        y_label: Plot the y-axis label. Defaults to ``True``.
        title_prefix: Prefix for the title. Defaults to an empty string.
        title: Or alternatively, use this exact title. Defaults to ``None``.
        skip: Take every `skip`-th sample. Defaults to ``10``.
        n_components: Number of PCA components. Defaults to ``2``.
        plot_eigenvectors: Plot and label the eigenvectors. Defaults to ``True``.
    """
    df: pd.DataFrame = data.get_dataframe()
    Z_samples: NpFloat = idata["posterior"]["Z"].stack(samples=("chain", "draw")).values
    Z_mean: NpFloat = np.mean(Z_samples, axis=-1)
    logger.debug("Z_mean = %s", Z_mean)

    logger.info("Plotting posterior samples of latent variables")
    for ii, row in enumerate(df.itertuples(index=False)):
        x_data = Z_samples[ii][0][0::skip]
        y_data = Z_samples[ii][1][0::skip]
        _, lightcolor = get_color(row.Chondrite, row.Reservoir)  # pyright: ignore
        ax.scatter(x_data, y_data, c=lightcolor, alpha=0.2)

    # Calculate and plot deterministic PCA
    pca: PCA = PCA(n_components=n_components)
    latent_variables: NpFloat = pca.fit_transform(data.get_feature_values())

    logger.info("Plotting deterministic PCA")
    logger.info("Plotting posterior mean (expected value) of latent variables")
    for ii, row in enumerate(df.itertuples(index=False)):
        color, _ = get_color(row.Chondrite, row.Reservoir)  # pyright: ignore
        ax.scatter(
            latent_variables[ii, 0],
            latent_variables[ii, 1],
            marker="o",
            facecolors="none",
            edgecolor=color,
        )

        # TODO: deal with labels at end
        #     label = self.data.get_reservoir_label(ii)
        #     if label is None:
        #         label = None
        #     else:
        #         label = f"{label} Mean"

        ax.scatter(Z_mean[ii, 0], Z_mean[ii, 1], color=color, marker="s")  #  label=label)

    logger.info("Plotting chondrite labels")
    for ii, label in enumerate(df["Chondrite"]):
        # TODO: Reincorporate text offset?
        # try:
        #    text_offset = self.data.get_label_offsets()[label]
        # except KeyError:
        # Required for lithophile
        text_offset = (10, -10)
        ax.annotate(
            label,
            (Z_mean[ii, 0], Z_mean[ii, 1]),
            textcoords="offset points",
            xytext=text_offset,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5, edgecolor="gray"),
        )

    if plot_eigenvectors:
        scaled_eigenvectors = pca.components_.T * np.sqrt(pca.explained_variance_)
        logger.debug("scaled_eigenvectors = %s", scaled_eigenvectors)

        # Plot eigenvectors (loadings)
        ax.quiver(
            np.zeros(data.n_features),
            np.zeros(data.n_features),
            scaled_eigenvectors[:, 0],
            scaled_eigenvectors[:, 1],
            color="k",
            angles="xy",
            scale_units="xy",
            scale=1,
            alpha=0.5,
        )

        # for i, feature_name in enumerate(self.data.data.feature_names):
        #     try:
        #         text_offset = self.data.get_eigenvector_label_offsets()[feature_name]
        #     except KeyError:
        #         text_offset = (0, 0)
        #     ax.annotate(
        #         feature_name,
        #         (scaled_eigenvectors[i, 0], scaled_eigenvectors[i, 1]),
        #         textcoords="offset points",
        #         xytext=text_offset,
        #         color="k",
        #         ha="center",
        #         va="center",
        #     )

    # # Get the existing handles and labels
    # handles, labels = ax.get_legend_handles_labels()
    # # Manually add a custom legend entry
    # custom_entry = Line2D(
    #     [0],
    #     [0],
    #     marker="o",
    #     label="Posterior sample",
    #     markersize=6,
    #     markerfacecolor="grey",
    #     markeredgecolor="k",
    #     linestyle="None",
    #     alpha=0.5,
    # )

    # custom_entry2 = Line2D(
    #     [0],
    #     [0],
    #     marker="s",
    #     label="Posterior mean",
    #     markersize=6,
    #     markerfacecolor="grey",
    #     markeredgecolor="k",
    #     linestyle="None",
    # )

    # custom_entry3 = Line2D(
    #     [0],
    #     [0],
    #     marker="o",
    #     label="Deterministic PCA",
    #     markersize=6,
    #     markerfacecolor="none",
    #     markeredgecolor="k",
    #     linestyle="None",
    # )

    # # Append the custom entry to the handles and labels
    # handles.extend([custom_entry, custom_entry2, custom_entry3])
    # labels.extend(["Posterior sample", "Posterior mean", "Deterministic PCA"])

    # Update the legend with the new handles and labels
    # if plot_legend:
    #     ax.legend(handles=handles, labels=labels, loc="lower left")

    if x_label:
        ax.set_xlabel("Latent Factor 1")
    if y_label:
        ax.set_ylabel("Latent Factor 2")

    # if title is None:
    #    title_str: str = f"{title_prefix} {self.data.get_title()}"
    # else:
    #    title_str = title

    # ax.set_title(title_str)
    # ax.grid(True)
