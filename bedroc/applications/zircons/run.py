#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Runs Zircon or synthetic analyses for comparison"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from setuptools import glob

from bedroc import RANDOM_SEED, debug_logger
from bedroc.applications.zircons.srmvf import run_pipeline as srmvf_run_pipeline
from bedroc.core.data_container import DataContainer
from bedroc.difference import DEFAULT_INFERENCE_MODEL, InferenceModel
from bedroc.difference.group_synthetic import SyntheticDataGenerator
from bedroc.difference.pipelines import run_pipeline


def run_zircon_analysis(
    inference: InferenceModel = DEFAULT_INFERENCE_MODEL, *, random_seed: int | None = RANDOM_SEED
):
    """Runs the zircon analysis pipeline.

    Args:
        inference: Type of inference to run. Defaults to :obj:`DEFAULT_INFERENCE_MODEL`.
        random_seed: Random seed for reproducibility. Defaults to :obj:`RANDOM_SEED`.
    """
    # SRMVF zircon analysis
    srmvf_run_pipeline(inference=inference, random_seed=random_seed)
    # TODO: Add Michigan zircon analysis


def run_zircon_analysis_loop(
    inference: InferenceModel = DEFAULT_INFERENCE_MODEL, n_seeds: int = 1000
):
    """Runs the zircon analysis pipeline in a loop for multiple random seeds.

    Args:
        inference: Type of inference to run. Defaults to :obj:`DEFAULT_INFERENCE_MODEL`.
        n_seeds: Number of random seeds to run. Defaults to ``1000``.
    """
    for seed in range(0, n_seeds):
        logger.info("Running zircon analysis with random seed: %d", seed)
        run_zircon_analysis(inference=inference, random_seed=seed)


def run_synthetic_analysis(
    inference: InferenceModel = DEFAULT_INFERENCE_MODEL, *, random_seed: int | None = RANDOM_SEED
):
    """Runs the synthetic analysis pipeline.

    Args:
        inference: Type of inference to run. Defaults to :obj:`DEFAULT_INFERENCE_MODEL`.
        random_seed: Random seed for reproducibility. Defaults to :obj:`RANDOM_SEED`.
    """
    logger.info("Running synthetic analysis pipeline with inference: %s", inference)

    category_names: tuple[str, str] = ("Group 0", "Group 1")

    output_directory = Path("synthetic") / Path(f"{inference}_seed_{random_seed}")

    generator: SyntheticDataGenerator = SyntheticDataGenerator(
        n_samples=1000,
        n_features=4,
        feature_offsets=2.0,  # [0.5, 0.5, 0.2, 0.2],
        feature_sigma=0.5,
        group_0_fraction=0.32,
        random_seed=random_seed,
        output_directory=output_directory,
    )
    generator.generate()

    data: DataContainer = generator.to_data_container(
        name="Synthetic", category_names=category_names
    )

    run_pipeline(
        data,
        inference=inference,
        output_directory=output_directory,
        random_seed=random_seed,
    )

    logger.info("Synthetic analysis pipeline completed with inference: %s", inference)


def final_stats():

    files = sorted(glob.glob("SRMVF/unified_seed_*/SRMVF_summary_statistics.xlsx"))

    results = pd.concat([pd.read_excel(file) for file in files], ignore_index=True)

    summary = pd.Series(
        {
            "Number of splits": len(results),
            "Mean bias": results["error_mean"].mean(),
            "Median bias": (results["median"] - results["truth"]).median(),
            "MAE": results["mae"].mean(),
            "RMSE": np.sqrt((results["rmse"] ** 2).mean()),  # Root Mean Squared Error across seeds
            "95% coverage": results["within_ci"].astype(bool).mean(),
            "Mean 95% CI width": results["ci_width"].mean(),
        }
    )

    print(summary)

    fig, ax = plt.subplots()

    ax.scatter(results["truth"], results["mean"])

    limits = [
        min(results["truth"].min(), results["mean"].min()),
        max(results["truth"].max(), results["mean"].max()),
    ]

    ax.plot(limits, limits, linestyle="--", color="black")

    ax.set_xlabel("Observed Plutonic fraction")
    ax.set_ylabel("Inferred Plutonic fraction")
    ax.set_title("Population fraction inference")

    fig.tight_layout()
    plt.show()


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
    parser.add_argument(
        "-i",
        "--inference",
        choices=["covariance", "tempered", "two-stage"],
        default=DEFAULT_INFERENCE_MODEL,
        help="Type of inference to run. Defaults to :obj:`DEFAULT_INFERENCE_MODEL`.",
    )
    parser.add_argument(
        "-l",
        "--zircon-loop",
        action="store_true",
        help="Run zircon analysis in a loop for multiple seeds",
    )
    parser.add_argument(
        "-r",
        "--random-seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for reproducibility. Defaults to {RANDOM_SEED}.",
    )
    parser.add_argument(
        "-f",
        "--final-stats",
        action="store_true",
        help="Compute final statistics from previous runs",
    )

    args = parser.parse_args()

    if args.synthetic:
        run_synthetic_analysis(inference=args.inference, random_seed=args.random_seed)

    if args.zircon:
        run_zircon_analysis(inference=args.inference, random_seed=args.random_seed)

    if args.zircon_loop:
        run_zircon_analysis_loop(inference=args.inference)

    if args.final_stats:
        final_stats()
