# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Zircon datasets"""

from pathlib import Path

root_dir: Path = Path("/Users/dan/Documents/academic/projects/zircons/volcanic_plutonic")
"""Root directory for zircon datasets"""

michigan_rootpath: Path = root_dir / Path("Naive Bayes Zircon Classifier") / Path("Michigan Data")
"""Dataset for Michigan zircons, curated by Tobias Hendrickx"""

michigan_date: str = "20260918"
"""Date stamp (YYYYMMDD) in the filenames of the current Michigan data export"""

michigan_barth: Path = michigan_rootpath / Path(
    f"Barth_BellCreekMichigan_Zircon_ALL_{michigan_date}_1622.xlsx"
)
michigan_foldenauer: Path = michigan_rootpath / Path(
    f"Processed_Foldenauer_Michigan_ALL_{michigan_date}_1626.xlsx"
)
michigan_hendrickx: Path = michigan_rootpath / Path(
    f"Processed_Hendrickx_Michigan_ALL_{michigan_date}_1629.xlsx"
)
michigan_petryk: Path = michigan_rootpath / Path(
    f"Processed_Petryk_Michigan_ALL_{michigan_date}_1619.xlsx"
)
michigan_pray: Path = michigan_rootpath / Path(
    f"Processed_Pray_Michigan_ALL_{michigan_date}_1623.xlsx"
)
michigan_staudenmann: Path = michigan_rootpath / Path(
    f"Processed_Staudenmann_Michigan_ALL_{michigan_date}_1631.xlsx"
)
michigan_all: Path = michigan_rootpath / Path("MICHIGAN_ALL_ZIRCON.xlsx")
srmvf_filepath: Path = (
    root_dir
    / Path("251105_data")
    / Path("SRMVF Zircon Geochronology and Geochemistry Volcanic PLutonic.xlsx")
)
"""Dataset for the San Juan volcanic field"""
