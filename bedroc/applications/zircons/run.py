#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run Zircon analyses"""

import numpy as np

from bedroc import debug_logger
from bedroc.applications.zircons.srmvf import run_analysis

# random_seeds: list[int] = [321, 123]

random_seeds = np.arange(1000)

# debug_logger()
# run_analysis_SRMVF()  # random_seeds=random_seeds)

if __name__ == "__main__":
    # Create the logger
    debug_logger()
    run_analysis(random_seeds=random_seeds)

    # final_stats()
