#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run Zircon analyses"""

from bedroc import debug_logger
from bedroc.applications.zircons.srmvf import run_analysis as run_analysis_SRMVF

random_seeds: list[int] = [321, 123]

if __name__ == "__main__":
    # Create the logger
    debug_logger()
    run_analysis_SRMVF(random_seeds=random_seeds)
