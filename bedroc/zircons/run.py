#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run Zircon analysis"""

from pathlib import Path

import matplotlib.pyplot as plt

from bedroc import debug_logger
from bedroc.zircons.core import process_SRMVF


def main():
    """Main function to run the analysis"""

    # Create the logger
    debug_logger()

    output_directory: Path = Path("SRMVF")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Run the analysis for the San Juan volcanic field zircon dataset
    model, test_value_np, test_group_idx, test_std_np = process_SRMVF(output_directory)
    model.run_pipeline(test_value_np, test_group_idx, X_sigma=test_std_np)

    plt.show()


if __name__ == "__main__":
    main()
