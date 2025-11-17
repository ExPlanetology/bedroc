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
from matplotlib.lines import Line2D
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
        label_offsets: Offsets for plotting the feature (isotope) labels. Defaults to ``None``.
        eigvec_label_offsets. Offsets for plotting the eigenvectors. Defaults to ``None``.
        data_dir: Data directory. Defaults to ``data``.
    """

    name: str
    datafile: Path
    chondrites: Optional[Iterable[str]] = None
    elements: Optional[Iterable[str]] = None
    label_offsets: Optional[dict[str, tuple[float, float]]] = None
    eigvec_label_offsets: Optional[dict[str, tuple[float, float]]] = None
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

    def plot_pca(
        self,
        ax: Axes,
        plot_legend: bool = True,
        include_title: bool = True,
        skip: int = 10,
        n_components: int = 2,
        plot_eigenvectors: bool = True,
    ) -> Axes:
        """Plots the Bayesian and deterministic PCA

        Args:
            ax: Axis
            plot_legend: Plots the legend. Defaults to ``True``.
            include_title: Adds a title. Defaults to ``True``.
            skip: Take every `skip`-th sample. Defaults to ``10``.
            n_components: Number of PCA components. Defaults to ``2``.
            plot_eigenvectors: Plot and label the eigenvectors. Defaults to ``True``.
        """
        df: pd.DataFrame = self.data.get_dataframe()
        Z_samples: NpFloat = self.idata["posterior"]["Z"].stack(samples=("chain", "draw")).values
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
        latent_variables: NpFloat = pca.fit_transform(self.data.get_feature_values())

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

            ax.scatter(Z_mean[ii, 0], Z_mean[ii, 1], color=color, marker="s")

        logger.info("Plotting chondrite labels")
        for ii, chondrite_name in enumerate(df["Chondrite"]):
            text_offset: tuple[float, float] = (0, 0)
            if self.label_offsets is not None:
                try:
                    text_offset = self.label_offsets[chondrite_name]
                except KeyError:
                    continue  # No offset

            ax.annotate(
                chondrite_name,
                (Z_mean[ii, 0], Z_mean[ii, 1]),
                textcoords="offset points",
                xytext=text_offset,
                ha="center",
                va="center",
            )

        if plot_eigenvectors:
            scaled_eigenvectors = pca.components_.T * np.sqrt(pca.explained_variance_)
            logger.debug("scaled_eigenvectors = %s", scaled_eigenvectors)

            # Plot eigenvectors (loadings)
            ax.quiver(
                np.zeros(self.data.n_features),
                np.zeros(self.data.n_features),
                scaled_eigenvectors[:, 0],
                scaled_eigenvectors[:, 1],
                color="k",
                angles="xy",
                scale_units="xy",
                scale=1,
                alpha=0.5,
            )

            for i, element_name in enumerate(self.data.feature_names):
                text_offset: tuple[float, float] = (0, 0)
                if self.eigvec_label_offsets is not None:
                    try:
                        text_offset = self.eigvec_label_offsets[element_name]
                    except KeyError:
                        continue  # No Offset

                ax.annotate(
                    element_name,
                    (scaled_eigenvectors[i, 0], scaled_eigenvectors[i, 1]),
                    textcoords="offset points",
                    xytext=text_offset,
                    color="k",
                    ha="center",
                    va="center",
                )

        # Get the existing handles and labels
        handles, labels = ax.get_legend_handles_labels()
        # Manually add a custom legend entry
        custom_entry = Line2D(
            [0],
            [0],
            marker="o",
            label="Posterior sample",
            markersize=6,
            markerfacecolor="grey",
            markeredgecolor="k",
            linestyle="None",
            alpha=0.5,
        )

        custom_entry2 = Line2D(
            [0],
            [0],
            marker="s",
            label="Posterior mean",
            markersize=6,
            markerfacecolor="grey",
            markeredgecolor="k",
            linestyle="None",
        )

        custom_entry3 = Line2D(
            [0],
            [0],
            marker="o",
            label="Deterministic PCA",
            markersize=6,
            markerfacecolor="none",
            markeredgecolor="k",
            linestyle="None",
        )

        # Append the custom entry to the handles and labels
        handles.extend([custom_entry, custom_entry2, custom_entry3])
        labels.extend(["Posterior sample", "Posterior mean", "Deterministic PCA"])

        if plot_legend:
            ax.legend(handles=handles, labels=labels, loc="lower left")

        ax.set_xlabel("Latent Factor 1")
        ax.set_ylabel("Latent Factor 2")

        if include_title:
            title: str = f"{self.name.capitalize()} elements"
            ax.set_title(title)

        return ax


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
    label_offsets={
        "CI": (-10, 10),
        "CM": (17, 0),
        "CO": (-5, -15),
        "CV": (-15, 0),
        "CR": (-15, 0),
        "H": (15, 2),
        "L": (-15, 0),
        "LL": (15, 0),
        "EH": (5, -13),
        "EL": (-12, 0),
        "Ureilites": (0, 12),
        "Mars": (0, 10),
        "Vesta Group": (45, 0),
        "Earth": (-25, 0),
        "Earth (Berm.)": (0, -15),
    },
    eigvec_label_offsets={
        "48Ca": (20, 3),
        "50Ti": (20, -7),
        "54Cr": (20, -21),
        "54Fe": (20, 15),
        "64Ni": (20, -24),
        "66Zn": (20, -14),
        "94Mo": (20, 11),
        "95Mo": (20, 8),
        "96Zr": (20, -1),
        "100Ru": (-10, -10),
    },
)
group_vesta_3_systems: GroupData = GroupData(
    name="vesta_3_systems",
    # TODO: Check with Paolo if this can be consolidated with alL_datafile
    datafile=Path("NC_CC_AllData_R1_NCbodies_3_systems.csv"),
    elements=("50Ti", "54Cr", "96Zr"),
    label_offsets={
        "H": (0, 12),
        "L": (0, 12),
        "LL": (0, -15),
        "EH": (15, 0),
        "EL": (0, 10),
        "Angrites": (0, 12),
        "Ureilites": (0, 12),
        "Vesta": (-25, 5),
        "Rumuruti": (0, 15),
        "Acapulcoite": (0, -12),
        "Aubrites": (32, 0),
        "Winonanites": (0, -12),
        "MG Pallasites": (0, 10),
        "Mesosiderite": (-30, -10),
        "Mars": (0, 5),
        "BSE": (20, 0),
    },
    eigvec_label_offsets={"50Ti": (0, -15), "54Cr": (0, 5), "96Zr": (-10, 5)},
)
group_vesta_4_systems: GroupData = GroupData(
    name="vesta_4_systems",
    datafile=Path("NC_CC_AllData_R1_NCbodies_4_systems.csv"),
    elements=("48Ca", "50Ti", "54Cr", "96Zr"),
    label_offsets={
        "H": (0, 12),
        "L": (0, 10),
        "LL": (12, 0),
        "EH": (0, -12),
        "EL": (22, 0),
        "Angrites": (0, 12),
        "Ureilites": (0, 12),
        "Vesta": (0, -12),
        "Aubrites": (0, 12),
        "Winonanites": (0, -15),
        "Mars": (0, 12),
        "BSE": (18, 0),
    },
    eigvec_label_offsets={"48Ca": (0, 5), "50Ti": (10, 0), "54Cr": (0, -15), "96Zr": (-5, 5)},
)
group_heavy: GroupData = GroupData(
    name="heavy",
    datafile=Path("NEW_TableS_NC-CC.csv"),
    elements=("94Mo", "95Mo", "96Zr", "100Ru"),
    label_offsets={
        "CI": (10, -10),
        "CM": (20, 0),
        "CO": (-20, -5),
        "CV": (20, 0),
        "CR": (20, -10),
        "H": (0, 10),
        "L": (-4, 15),
        "LL": (-4, 22),
        "EH": (-5, 15),
        "EL": (-10, -5),
        "Ureilites": (-5, 15),
        "Mars": (3, -13),
        "Vesta Group": (45, 0),
        "Earth": (-25, 0),
        "Earth (Berm.)": (-20, -15),
    },
    eigvec_label_offsets={
        "100Ru": (25, -30),
        "54Fe": (20, 15),
        "94Mo": (20, 6),
        "95Mo": (20, -2),
        "96Zr": (20, -10),
        "48Ca": (20, 3),
        "50Ti": (20, -7),
        "66Zn": (20, -11),
        "54Cr": (20, -19),
        "64Ni": (20, -22),
    },
)
group_iron_peak: GroupData = GroupData(
    name="iron-peak",
    datafile=Path("NEW_TableS_NC-CC.csv"),
    elements=("48Ca", "50Ti", "54Cr", "54Fe", "64Ni", "66Zn"),
    label_offsets={
        "CI": (-15, 0),
        "CM": (17, 0),
        "CO": (0, -12),
        "CV": (0, 10),
        "CR": (0, -15),
        "H": (15, 0),
        "L": (15, 0),
        "LL": (-13, 5),
        "EH": (-20, 0),
        "EL": (-20, -2),
        "Ureilites": (0, 12),
        "Mars": (-25, 2),
        "Vesta Group": (-40, -10),
        "Earth": (-25, 0),
        "Earth (Berm.)": (-25, -15),
    },
    eigvec_label_offsets={
        "100Ru": (-10, -10),
        "54Fe": (20, 15),
        "94Mo": (20, 11),
        "95Mo": (20, 8),
        "96Zr": (20, -1),
        "48Ca": (20, 9),
        "50Ti": (20, -3),
        "66Zn": (20, -9),
        "54Cr": (20, -17),
        "64Ni": (20, -22),
    },
)
group_siderophile: GroupData = GroupData(
    name="siderophile",
    datafile=Path("NEW_TableS_NC-CC.csv"),
    elements=("54Fe", "64Ni", "94Mo", "95Mo", "100Ru"),
    label_offsets={
        "CI": (0, 10),
        "CM": (0, 10),
        "CO": (5, 10),
        "CV": (0, 10),
        "CR": (15, 0),
        "H": (0, -15),
        "L": (-10, 5),
        "LL": (-15, -5),
        "EH": (-15, -5),
        "EL": (-15, 0),
        "Ureilites": (30, -10),
        "Mars": (0, 13),
        "Vesta Group": (45, 0),
        "Earth": (-25, 0),
        "Earth (Berm.)": (0, 10),
    },
    eigvec_label_offsets={
        "100Ru": (15, 12),
        "54Fe": (20, -5),
        "94Mo": (17, 0),
        "95Mo": (-3, 10),
        "96Zr": (20, -1),
        "48Ca": (20, 3),
        "50Ti": (20, -7),
        "66Zn": (20, -11),
        "54Cr": (20, -19),
        "64Ni": (5, 5),
    },
)
group_lithophile: GroupData = GroupData(
    name="lithophile",
    datafile=Path("NEW_TableS_NC-CC.csv"),
    elements=("48Ca", "50Ti", "54Cr", "66Zn", "96Zr"),
    label_offsets={
        "CI": (-15, 0),
        "CR": (15, 0),
        "CM": (15, 0),
        "CO": (-15, 5),
        "CV": (-15, -5),
        "Vesta Group": (0, 10),
        "Ureilites": (0, -15),
        "H": (10, 5),
        "LL": (-15, 0),
        "L": (15, 0),
        "Mars": (-20, -5),
        "EL": (-15, 0),
        "EH": (15, 0),
        "Earth": (0, -15),
    },
    eigvec_label_offsets={
        "96Zr": (15, 5),
        "54Cr": (15, -10),
        "66Zn": (15, -5),
        "50Ti": (15, 0),
        "48Ca": (15, 7),
    },
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
    label_offsets={
        "CI": (0, 10),
        "CM": (-8, 10),
        "CO": (0, 10),
        "CV": (3, 10),
        "CR": (0, -15),
        "IIC": (-15, 0),
        "IID": (-15, 0),
        "IVB": (0, -15),
        "H": (-15, 0),
        "L": (15, 0),
        "LL": (15, 0),
        "EH": (-15, 0),
        "EL": (0, -15),
        "Ureilites": (-30, 8),
        "Mars": (0, 10),
        "IAB": (15, -3),
        "IC": (10, -10),
        "IIAB": (0, -17),
        "IIIAB": (10, -15),
        "IVA": (15, 0),
        "Earth": (0, 10),
    },
    eigvec_label_offsets={"54Fe": (-15, -5), "94Mo": (-15, 5), "64Ni": (3, 5), "100Ru": (-25, 10)},
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
    label_offsets={
        "CI": (0, 10),
        "CM": (0, 12),
        "CO": (-3, 10),
        "CV": (3, 10),
        "CK": (0, 12),
        "CR": (0, 10),
        "CH": (0, 10),
        "H": (-5, -10),
        "L": (-10, 2),
        "LL": (-10, -5),
        "EH": (0, 10),
        "EL": (0, 10),
        "Ureilites": (-30, 0),
        "Acapulcoite": (60, 7),
        "Aubrites": (-30, -5),
        "Winonanites": (-40, -5),
        "MG Pallasites": (-50, 5),
        "Mesosiderite": (60, -7),
        "Mars": (25, 3),
        "IIAB": (10, -15),
        "IIIAB": (-5, -17),
        "IVA": (15, 0),
        "Earth": (0, 10),
    },
    eigvec_label_offsets={"54Cr": (5, 5), "94Mo": (15, 0), "95Mo": (15, 5), "100Ru": (30, 5)},
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
