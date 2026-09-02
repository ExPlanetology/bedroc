# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Zircon datasets"""

from pathlib import Path

root_dir: Path = Path("/Users/dan/Documents/academic/projects/zircons/volcanic_plutonic")
"""Root directory for zircon datasets"""

michigan_rootpath: Path = root_dir / Path("Naive Bayes Zircon Classifier") / Path("Michigan Data")
"""Dataset for Michigan zircons, curated by Tobias Hendrickx"""

michigan_barth: Path = michigan_rootpath / Path(
    "Processed_Barth_BellCreekMichigan_20260810_1049.xlsx"
)
michigan_foldenauer: Path = michigan_rootpath / Path(
    "Processed_Foldenauer_Michigan_20260810_1059.xlsx"
)
michigan_hendrickx: Path = michigan_rootpath / Path(
    "Processed_Hendrickx_Michigan_20260810_1424.xlsx"
)
michigan_petryk: Path = michigan_rootpath / Path("Processed_Petryk_Michigan_20260810_1408.xlsx")
michigan_pray: Path = michigan_rootpath / Path("Processed_Pray_Michigan_20260810_1355.xlsx")
michigan_staudenmann: Path = michigan_rootpath / Path(
    "Processed_Staudenmann_Michigan_20260810_1409.xlsx"
)
srmvf_filepath: Path = (
    root_dir
    / Path("251105_data")
    / Path("SRMVF Zircon Geochronology and Geochemistry Volcanic PLutonic.xlsx")
)
"""Dataset for the San Juan volcanic field"""
