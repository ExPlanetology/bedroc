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
"""Combined Michigan dataset (unfiltered)"""

michigan_filtered_path: Path = michigan_rootpath / Path("Filtered data")
"""Filtered variants of :obj:`michigan_all` (see ``README_filtering_updates.md`` there)"""
michigan_all_vulcan: Path = michigan_filtered_path / Path("MICHIGAN_ALL_ZIRCON_+vulcan.xlsx")
"""Unfiltered, including the Vulcan area samples (Type "Vulcan")"""
michigan_lree_discordance: Path = michigan_filtered_path / Path(
    "MICHIGAN_ALL_ZIRCON_LREE-Discordance_filtered.xlsx"
)
"""Filtered on Zr content, LREE index and discordance"""
michigan_lree_discordance_thu_barth: Path = michigan_filtered_path / Path(
    "MICHIGAN_ALL_ZIRCON_LREE-Discordance_filtered_ThU-Barth.xlsx"
)
"""As :obj:`michigan_lree_discordance`, with Th/U from U-Pb dating added for the Barth data"""
michigan_lree_discordance_thu_barth_vulcan: Path = michigan_filtered_path / Path(
    "MICHIGAN_ALL_ZIRCON_LREE-Discordance_filtered_ThU-Barth_Vulcan.xlsx"
)
"""As :obj:`michigan_lree_discordance_thu_barth`, including the Vulcan area samples"""
michigan_filters_p_ti_al: Path = michigan_filtered_path / Path(
    "MICHIGAN_ALL_ZIRCON_filters+P-Ti-Al.xlsx"
)
"""As the filtered files above, additionally removing P > 3000, Ti > 1000 and Al > 5000 ug/g"""
srmvf_filepath: Path = (
    root_dir
    / Path("251105_data")
    / Path("SRMVF Zircon Geochronology and Geochemistry Volcanic PLutonic.xlsx")
)
"""Dataset for the San Juan volcanic field"""
