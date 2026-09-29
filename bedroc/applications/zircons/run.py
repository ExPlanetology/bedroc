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
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bedroc import OUTPUT_ROOT, RANDOM_SEED, debug_logger
from bedroc.applications.zircons.michigan import run_pipeline as michigan_run_pipeline
from bedroc.applications.zircons.srmvf import DATASET_NAME as SRMVF_DATASET_NAME
from bedroc.applications.zircons.srmvf import run_pipeline as srmvf_run_pipeline
from bedroc.applications.zircons.synthetic import DATASET_NAME as SYNTHETIC_DATASET_NAME
from bedroc.applications.zircons.synthetic import run_pipeline as synthetic_run_pipeline
from bedroc.core.plotting import save_figure
from bedroc.difference import DEFAULT_FIT_MODEL, FitModel
from bedroc.difference.pipelines import MODEL_PIPELINES

logger: logging.Logger = logging.getLogger(__name__)

ZIRCON_PIPELINES: Mapping[str, Callable[..., None]] = {
    "san-juan": srmvf_run_pipeline,
    "michigan": michigan_run_pipeline,
    "synthetic": synthetic_run_pipeline,
}
"""Zircon dataset pipelines, keyed by the dataset names accepted on the command line"""


def run_zircon_analysis(
    model: FitModel = DEFAULT_FIT_MODEL,
    *,
    datasets: Sequence[str] = tuple(ZIRCON_PIPELINES),
    random_seed: int | None = RANDOM_SEED,
) -> None:
    """Runs the zircon analysis pipeline for each of ``datasets``.

    Args:
        model: Model to fit. Defaults to :obj:`DEFAULT_FIT_MODEL`.
        datasets: Datasets to run, from :obj:`ZIRCON_PIPELINES`. Defaults to all of them.
        random_seed: Random seed for reproducibility. Defaults to :obj:`RANDOM_SEED`.
    """
    for dataset in datasets:
        ZIRCON_PIPELINES[dataset](model=model, random_seed=random_seed)


def run_zircon_analysis_loop(
    model: FitModel = DEFAULT_FIT_MODEL,
    *,
    datasets: Sequence[str] = tuple(ZIRCON_PIPELINES),
    n_seeds: int = 1000,
) -> None:
    """Runs the zircon analysis pipeline in a loop for multiple random seeds.

    Args:
        model: Model to fit. Defaults to :obj:`DEFAULT_FIT_MODEL`.
        datasets: Datasets to run, from :obj:`ZIRCON_PIPELINES`. Defaults to all of them.
        n_seeds: Number of random seeds to run. Defaults to ``1000``.
    """
    for seed in range(0, n_seeds):
        logger.info("Running zircon analysis with random seed: %d", seed)
        run_zircon_analysis(model=model, datasets=datasets, random_seed=seed)


FINAL_STATS_RUNS: Mapping[str, tuple[tuple[Path, str], ...]] = {
    "san-juan": ((OUTPUT_ROOT / SRMVF_DATASET_NAME, SRMVF_DATASET_NAME),),
    # "Synthetic" is the container name group_synthetic.run_pipeline gives the generated data
    "synthetic": (
        (OUTPUT_ROOT / SYNTHETIC_DATASET_NAME / "withcov", "Synthetic"),
        (OUTPUT_ROOT / SYNTHETIC_DATASET_NAME / "nocov", "Synthetic"),
    ),
}
"""Where each dataset with a known true fraction writes its runs, as ``(base output directory,
summary file name prefix)`` pairs (one per case). Michigan is absent: its unlabeled Detrital
zircons have no known true fraction to compare against."""


def final_stats(
    model: FitModel = DEFAULT_FIT_MODEL, *, output_directory: Path, name: str
) -> pd.Series:
    """Summarizes the population-fraction inference across the seeds of previous runs.

    Reads each run's ``<name>_summary_statistics.xlsx`` from
    ``<output_directory>/<model>_seed_<seed>/``, i.e. every seed found there, and writes the
    summary (``<model>_final_stats.xlsx``) and a true-vs-inferred scatter plot
    (``<model>_final_stats``) to ``output_directory``.

    Args:
        model: Model whose runs to summarize. Defaults to :obj:`DEFAULT_FIT_MODEL`.
        output_directory: Base output directory of the dataset (or dataset case)
        name: Prefix of the summary statistics files

    Raises:
        FileNotFoundError: If no run's summary statistics file is found.

    Returns:
        The summary across seeds
    """
    pattern: Path = output_directory / f"{model}_seed_*" / f"{name}_summary_statistics.xlsx"
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

    label: str = f"{model} ({output_directory})"
    print(f"{label}:")
    print(summary)
    summary.to_frame(name=model).to_excel(output_directory / f"{model}_final_stats.xlsx")

    fig, ax = plt.subplots()

    ax.scatter(results["truth"], results["mean"])

    limits = [
        min(results["truth"].min(), results["mean"].min()),
        max(results["truth"].max(), results["mean"].max()),
    ]

    ax.plot(limits, limits, linestyle="--", color="black")

    ax.set_xlabel("Observed category-0 fraction")
    ax.set_ylabel("Inferred category-0 fraction")
    ax.set_title(f"Population fraction inference: {label}")

    fig.tight_layout()
    save_figure(fig, Path(f"{model}_final_stats"), output_directory)

    return summary


def run_final_stats(
    model: FitModel = DEFAULT_FIT_MODEL, *, datasets: Sequence[str] = tuple(FINAL_STATS_RUNS)
) -> None:
    """Runs :func:`final_stats` for each dataset (and each of its cases), skipping any with no
    runs yet.

    Args:
        model: Model whose runs to summarize. Defaults to :obj:`DEFAULT_FIT_MODEL`.
        datasets: Datasets to summarize, from :obj:`FINAL_STATS_RUNS`. Defaults to all of them.
    """
    for dataset in datasets:
        for output_directory, name in FINAL_STATS_RUNS[dataset]:
            try:
                final_stats(model, output_directory=output_directory, name=name)
            except FileNotFoundError as error:
                logger.warning("Skipping %s: %s", dataset, error)


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
        f"({', '.join(ZIRCON_PIPELINES)}), e.g. -z michigan. With no datasets, runs all of them "
        "(including synthetic).",
    )
    parser.add_argument(
        "-m",
        "--model",
        nargs="+",
        choices=list(MODEL_PIPELINES),
        default=[DEFAULT_FIT_MODEL],
        metavar="MODEL",
        help=f"Model(s) to fit ({', '.join(MODEL_PIPELINES)}). Accepts one or more values, run "
        f"in turn (e.g. -m tempered naive). Defaults to {DEFAULT_FIT_MODEL}.",
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
        "-n",
        "--n-seeds",
        type=int,
        default=1000,
        help="Number of random seeds (0 to N-1) for the -l loop. Defaults to 1000.",
    )
    parser.add_argument(
        "-f",
        "--final-stats",
        nargs="*",
        choices=list(FINAL_STATS_RUNS),
        metavar="DATASET",
        help="Summarize every seed run found on disk for the given datasets "
        f"({', '.join(FINAL_STATS_RUNS)}), per model. With no datasets, summarizes all of them. "
        "Writes <model>_final_stats.xlsx and a plot to each dataset's output directory.",
    )

    args = parser.parse_args()

    for model in args.model:
        logger.info("Running with model: %s", model)

        # An empty list means the flag was given without datasets, so run all of them
        if args.zircon is not None:
            run_zircon_analysis(
                model=model,
                datasets=args.zircon or tuple(ZIRCON_PIPELINES),
                random_seed=args.random_seed,
            )

        if args.zircon_loop is not None:
            run_zircon_analysis_loop(
                model=model,
                datasets=args.zircon_loop or tuple(ZIRCON_PIPELINES),
                n_seeds=args.n_seeds,
            )

        if args.final_stats is not None:
            run_final_stats(model=model, datasets=args.final_stats or tuple(FINAL_STATS_RUNS))
