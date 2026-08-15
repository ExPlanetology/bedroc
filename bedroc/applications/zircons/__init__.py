# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Zircon datasets"""

from pathlib import Path

root_dir: Path = Path("/Users/dan/Documents/academic/projects/zircons/volcanic_plutonic")
"""Root directory for zircon datasets"""

zircon_michigan_filepath: Path = (
    root_dir / Path("260526_michigan") / Path("Michigan_Zircons_ALL_cleaned.xlsx")
)
"""Dataset for Michigan zircons, curated by Tobias Hendrickx"""

srmvf_filepath: Path = (
    root_dir
    / Path("251105_data")
    / Path("SRMVF Zircon Geochronology and Geochemistry Volcanic PLutonic.xlsx")
)
"""Dataset for the San Juan volcanic field"""
