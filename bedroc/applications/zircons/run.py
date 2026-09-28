#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Runs Zircon or synthetic analyses for comparison"""

import os

# Must be set before numpy (and anything that imports it, e.g. matplotlib/pandas/pymc) is
# imported: on macOS, numpy's Accelerate BLAS backend spins up its internal thread pool at
# import time, and letting each PyMC multiprocessing sampling worker do so independently causes
# workers to crash silently (surfacing as an unhelpful EOFError from pm.sample()).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import glob
import logging
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bedroc import OUTPUT_ROOT, RANDOM_SEED, debug_logger
from bedroc.applications.zircons.michigan import run_pipeline as michigan_run_pipeline
from bedroc.applications.zircons.srmvf import DATASET_NAME, process_SRMVF
from bedroc.applications.zircons.srmvf import run_pipeline as srmvf_run_pipeline
from bedroc.core.data_container import DataContainer
from bedroc.difference import DEFAULT_INFERENCE_MODEL, InferenceModel
from bedroc.difference.group_synthetic import SyntheticDataGenerator
from bedroc.difference.group_synthetic import run_pipeline as synthetic_run_pipeline
from bedroc.difference.utils import distribution_overlap, effect_size_from_overlap

logger: logging.Logger = logging.getLogger(__name__)

ZIRCON_PIPELINES: Mapping[str, Callable[..., None]] = {
    "san-juan": srmvf_run_pipeline,
    "michigan": michigan_run_pipeline,
}
"""Zircon dataset pipelines, keyed by the dataset names accepted on the command line"""


def run_zircon_analysis(
    inference: InferenceModel = DEFAULT_INFERENCE_MODEL,
    *,
    datasets: Sequence[str] = tuple(ZIRCON_PIPELINES),
    random_seed: int | None = RANDOM_SEED,
) -> None:
    """Runs the zircon analysis pipeline for each of ``datasets``.

    Args:
        inference: Type of inference to run. Defaults to :obj:`DEFAULT_INFERENCE_MODEL`.
        datasets: Datasets to run, from :obj:`ZIRCON_PIPELINES`. Defaults to all of them.
        random_seed: Random seed for reproducibility. Defaults to :obj:`RANDOM_SEED`.
    """
    for dataset in datasets:
        ZIRCON_PIPELINES[dataset](inference=inference, random_seed=random_seed)


def run_zircon_analysis_loop(
    inference: InferenceModel = DEFAULT_INFERENCE_MODEL,
    *,
    datasets: Sequence[str] = tuple(ZIRCON_PIPELINES),
    n_seeds: int = 1000,
) -> None:
    """Runs the zircon analysis pipeline in a loop for multiple random seeds.

    Args:
        inference: Type of inference to run. Defaults to :obj:`DEFAULT_INFERENCE_MODEL`.
        datasets: Datasets to run, from :obj:`ZIRCON_PIPELINES`. Defaults to all of them.
        n_seeds: Number of random seeds to run. Defaults to ``1000``.
    """
    for seed in range(0, n_seeds):
        logger.info("Running zircon analysis with random seed: %d", seed)
        run_zircon_analysis(inference=inference, datasets=datasets, random_seed=seed)


def run_synthetic_analysis(
    inference: InferenceModel = DEFAULT_INFERENCE_MODEL, *, random_seed: int | None = RANDOM_SEED
) -> None:
    """Runs the synthetic analysis pipeline for two cases: with and without the real SRMVF
    zircon covariance structure.

    Both cases are calibrated from the real SRMVF data (covariance matrix, per-feature effect
    size, sample count, and category balance), so the "with covariance" case statistically
    resembles the real data by construction, and the "without covariance" case is a true
    ablation of it (identical feature offsets, sample count, and category balance; only the
    covariance structure is removed).

    Args:
        inference: Type of inference to run. Defaults to :obj:`DEFAULT_INFERENCE_MODEL`.
        random_seed: Random seed for reproducibility. Defaults to :obj:`RANDOM_SEED`.
    """
    real_data: DataContainer = process_SRMVF(name=DATASET_NAME, output_directory=None)

    covariance_matrix = real_data.diagnostics.within_category_covariance_matrix().to_numpy()
    # SRMVF has exactly two categories (Plutonic/Volcanic); category_mean_difference()'s row 0
    # is category 0's (Plutonic's) all-zero self-offset, row 1 is category 1's (Volcanic's)
    # mean offset from category 0's.
    raw_feature_offsets = real_data.diagnostics.category_mean_difference().iloc[1]
    category_0_fraction = (
        real_data.category_counts.iloc[0]  # pyright: ignore[reportOptionalMemberAccess]
        / real_data.n_data
    )

    # The real per-feature mean difference, taken at face value, understates the effect size
    # needed for a *Gaussian* synthetic replica to visually match real data: Gaussian marginals
    # are smoother-tailed than the real (empirical) marginals, so the same raw delta produces
    # systematically more overlap in the synthetic case. Back-solve the effect size that
    # reproduces the real per-feature overlap coefficient instead, so the synthetic "with
    # covariance" case's marginal overlap actually matches what the real data shows.
    values_std = real_data.values_std
    codes = real_data.category_codes
    feature_offsets = raw_feature_offsets.copy()
    for feature in real_data.feature_names:
        _, _, _, _, real_overlap = distribution_overlap(
            values_std.loc[codes == 0, feature].to_numpy(),
            values_std.loc[codes == 1, feature].to_numpy(),
        )
        effective_delta = effect_size_from_overlap(real_overlap)
        feature_offsets[feature] = np.copysign(effective_delta, raw_feature_offsets[feature])

    feature_offsets = feature_offsets.to_numpy()

    logger.info(
        "Calibrated synthetic analysis from real SRMVF data: n_samples=%d, "
        "category_0_fraction=%.4f, raw feature_offsets=%s, "
        "overlap-matched feature_offsets=%s, covariance=\n%s",
        real_data.n_data,
        category_0_fraction,
        raw_feature_offsets.to_numpy(),
        feature_offsets,
        covariance_matrix,
    )

    for with_covariance in (True, False):
        if with_covariance:
            covariance = covariance_matrix
            output_directory = OUTPUT_ROOT / "synthetic" / f"{inference}_withcov_seed_{random_seed}"
        else:
            covariance = None
            output_directory = OUTPUT_ROOT / "synthetic" / f"{inference}_nocov_seed_{random_seed}"

        generator: SyntheticDataGenerator = SyntheticDataGenerator(
            n_samples=real_data.n_data,
            n_features=real_data.n_features,
            feature_offsets=feature_offsets,
            feature_sigma=1.0,  # Exact for standardized data; unused when covariance is given
            covariance=covariance,
            category_0_fraction=category_0_fraction,
            random_seed=random_seed,
            output_directory=output_directory,
        )

        synthetic_run_pipeline(generator, inference=inference)


def final_stats(
    inference: InferenceModel = DEFAULT_INFERENCE_MODEL, *, dataset_name: str = DATASET_NAME
) -> None:
    """Summarizes the population-fraction inference across the seeds of previous runs.

    Reads each run's ``<dataset_name>_summary_statistics.xlsx`` from where the dataset's
    ``run_pipeline`` writes it: ``OUTPUT_ROOT / dataset_name / f"{inference}_seed_<seed>"``.

    Args:
        inference: Type of inference whose runs to summarize. Defaults to
            :obj:`DEFAULT_INFERENCE_MODEL`.
        dataset_name: Name of the dataset whose runs to summarize. Defaults to the SRMVF
            :obj:`DATASET_NAME`.

    Raises:
        FileNotFoundError: If no run's summary statistics file is found.
    """
    run_directories: Path = OUTPUT_ROOT / dataset_name / f"{inference}_seed_*"
    pattern: Path = run_directories / f"{dataset_name}_summary_statistics.xlsx"
    files: list[str] = sorted(glob.glob(str(pattern)))
    if not files:
        raise FileNotFoundError(f"No summary statistics files found matching {pattern}")
    logger.info("Summarizing %d runs matching %s", len(files), pattern)

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
    logger = debug_logger()
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Run zircon and synthetic pipelines.")
    parser.add_argument(
        "-z",
        "--zircon",
        nargs="*",
        choices=list(ZIRCON_PIPELINES),
        metavar="DATASET",
        help="Run the zircon analysis pipeline for one or more datasets "
        f"({', '.join(ZIRCON_PIPELINES)}), e.g. -z michigan. With no datasets, runs all of them.",
    )
    parser.add_argument(
        "-s", "--synthetic", action="store_true", help="Run the synthetic analysis pipeline"
    )
    parser.add_argument(
        "-i",
        "--inference",
        nargs="+",
        choices=["covariance", "tempered", "tempered-full", "naive", "two-stage"],
        default=[DEFAULT_INFERENCE_MODEL],
        help="Type(s) of inference to run. Accepts one or more values, run in turn (e.g. "
        "-i tempered naive). Defaults to :obj:`DEFAULT_INFERENCE_MODEL`.",
    )
    parser.add_argument(
        "-l",
        "--zircon-loop",
        nargs="*",
        choices=list(ZIRCON_PIPELINES),
        metavar="DATASET",
        help="Run zircon analysis in a loop for multiple seeds, for the given datasets (as for -z)",
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

    for inference in args.inference:
        logger.info("Running with inference: %s", inference)

        if args.synthetic:
            run_synthetic_analysis(inference=inference, random_seed=args.random_seed)

        # An empty list means the flag was given without datasets, so run all of them
        if args.zircon is not None:
            run_zircon_analysis(
                inference=inference,
                datasets=args.zircon or tuple(ZIRCON_PIPELINES),
                random_seed=args.random_seed,
            )

        if args.zircon_loop is not None:
            run_zircon_analysis_loop(
                inference=inference, datasets=args.zircon_loop or tuple(ZIRCON_PIPELINES)
            )

        if args.final_stats:
            final_stats(inference=inference)
