#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run Zircon or synthetic analyses for comparison"""

import argparse
import logging
import sys
from pathlib import Path

from bedroc import debug_logger
from bedroc.applications.zircons.srmvf import run_inference_pipeline
from bedroc.core.data_container import RANDOM_SEED, DataContainer
from bedroc.difference.group_synthetic import SyntheticDataGenerator
from bedroc.difference.pipelines import pipeline_two_stage_inference


def run_zircon_analysis():
    """Runs the zircon analysis pipeline."""
    # SRMVF zircon analysis
    run_inference_pipeline()
    # TODO: Add Michigan zircon analysis


def run_synthetic_analysis(random_seed: int | None = RANDOM_SEED):
    """Runs the synthetic analysis pipeline."""

    group_names: tuple[str, str] = ("Group 0", "Group 1")
    output_directory = Path("synthetic")

    generator: SyntheticDataGenerator = SyntheticDataGenerator(
        n_samples=1000,
        n_features=4,
        feature_offsets=2.0,  # [0.5, 0.5, 0.2, 0.2],
        feature_sigma=0.5,
        group_0_fraction=0.32,
        random_seed=random_seed,
    )
    generator.generate()

    data: DataContainer = generator.to_data_container(name="Synthetic")

    # TODO: So far only configured to run the two-stage inference
    pipeline_two_stage_inference(
        data,
        group_names=group_names,
        group_data_column="group_idx",
        output_directory=output_directory,
        random_seed=random_seed,
    )


if __name__ == "__main__":
    logger: logging.Logger = debug_logger()
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Run zircon and synthetic pipelines.")
    parser.add_argument(
        "-z", "--zircon", action="store_true", help="Run the zircon analysis pipeline"
    )
    parser.add_argument(
        "-s", "--synthetic", action="store_true", help="Run the synthetic analysis pipeline"
    )

    args = parser.parse_args()

    if not args.zircon and not args.synthetic:
        print("No analysis flag provided. Please specify -z (zircon) or -s (synthetic).")
        parser.print_help()
        sys.exit(0)

    if args.zircon:
        run_zircon_analysis()

    if args.synthetic:
        run_synthetic_analysis()
