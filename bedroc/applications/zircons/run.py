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
from bedroc.applications.zircons.michigan import DATASET_NAME as MICHIGAN_DATASET_NAME
from bedroc.applications.zircons.michigan import build_michigan_dataset
from bedroc.applications.zircons.michigan import run_pipeline as michigan_run_pipeline
from bedroc.applications.zircons.srmvf import DATASET_NAME as SRMVF_DATASET_NAME
from bedroc.applications.zircons.srmvf import process_SRMVF
from bedroc.applications.zircons.srmvf import run_pipeline as srmvf_run_pipeline
from bedroc.applications.zircons.synthetic import DATASET_NAME as SYNTHETIC_DATASET_NAME
from bedroc.applications.zircons.synthetic import build_synthetic_dataset, case_name
from bedroc.applications.zircons.synthetic import run_pipeline as synthetic_run_pipeline
from bedroc.core.data_container import DataContainer
from bedroc.core.plotting import save_figure
from bedroc.difference import DEFAULT_FIT_MODEL, FitModel
from bedroc.difference.pipelines import MODEL_PIPELINES, pipeline_OVL
from bedroc.difference.utils import log_pipeline_run

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
    synthetic_covariance: bool = False,
) -> None:
    """Runs the zircon analysis pipeline for each of ``datasets``.

    Args:
        model: Model to fit. Defaults to :obj:`DEFAULT_FIT_MODEL`.
        datasets: Datasets to run, from :obj:`ZIRCON_PIPELINES`. Defaults to all of them.
        random_seed: Random seed for reproducibility. Defaults to :obj:`RANDOM_SEED`.
        synthetic_covariance: For the synthetic dataset, whether to generate the features with
            the real SRMVF within-category covariance rather than independently. Ignored by the
            other datasets. Defaults to ``False``.
    """
    for dataset in datasets:
        kwargs: dict[str, bool] = (
            {"with_covariance": synthetic_covariance} if dataset == "synthetic" else {}
        )
        ZIRCON_PIPELINES[dataset](model=model, random_seed=random_seed, **kwargs)


def run_zircon_analysis_loop(
    model: FitModel = DEFAULT_FIT_MODEL,
    *,
    datasets: Sequence[str] = tuple(ZIRCON_PIPELINES),
    n_seeds: int = 1000,
    start_seed: int = 0,
    synthetic_covariance: bool = False,
) -> None:
    """Runs the zircon analysis pipeline in a loop over consecutive random seeds.

    Args:
        model: Model to fit. Defaults to :obj:`DEFAULT_FIT_MODEL`.
        datasets: Datasets to run, from :obj:`ZIRCON_PIPELINES`. Defaults to all of them.
        n_seeds: Number of random seeds to run. Defaults to ``1000``.
        start_seed: First seed; the seeds run are ``start_seed`` to
            ``start_seed + n_seeds - 1``. Defaults to ``0``.
        synthetic_covariance: See :func:`run_zircon_analysis`. Defaults to ``False``.
    """
    for seed in range(start_seed, start_seed + n_seeds):
        logger.info("Running zircon analysis with random seed: %d", seed)
        run_zircon_analysis(
            model=model,
            datasets=datasets,
            random_seed=seed,
            synthetic_covariance=synthetic_covariance,
        )


def overlap_data(
    dataset: str, *, random_seed: int | None, synthetic_covariance: bool
) -> tuple[DataContainer, Path]:
    """Returns the labeled data to compare, and its base output directory, for a dataset.

    Args:
        dataset: Dataset name, from :obj:`ZIRCON_PIPELINES`
        random_seed: Seed for generating the synthetic data
        synthetic_covariance: See :func:`run_zircon_analysis`

    Raises:
        ValueError: If ``dataset`` is not recognized.

    Returns:
        The data (with both categories known) and the directory under which its ``overlap``
        subdirectory is written
    """
    if dataset == "san-juan":
        return process_SRMVF(output_directory=None), OUTPUT_ROOT / SRMVF_DATASET_NAME
    if dataset == "michigan":
        # The labeled Plutonic/Volcanic pair; the Detrital zircons have no known category
        data: DataContainer = build_michigan_dataset(output_directory=None).labeled.data
        return data, OUTPUT_ROOT / MICHIGAN_DATASET_NAME
    if dataset == "synthetic":
        data = build_synthetic_dataset(
            random_seed=random_seed, with_covariance=synthetic_covariance
        )
        return data, OUTPUT_ROOT / SYNTHETIC_DATASET_NAME / case_name(synthetic_covariance)
    raise ValueError(f"Unrecognized dataset {dataset!r}")


def run_zircon_overlap(
    datasets: Sequence[str] = tuple(ZIRCON_PIPELINES),
    *,
    random_seed: int | None = RANDOM_SEED,
    synthetic_covariance: bool = False,
) -> None:
    """Computes the distribution overlap (OVL) diagnostics once for each of ``datasets``.

    These depend only on the data, not on a model or train/test split, so they are computed once
    per dataset (into ``<dataset directory>/overlap/``) rather than for every model run.

    Args:
        datasets: Datasets to compute them for, from :obj:`ZIRCON_PIPELINES`. Defaults to all of
            them.
        random_seed: Seed for the Monte Carlo overlap estimates (and for generating the synthetic
            data). Defaults to :obj:`RANDOM_SEED`.
        synthetic_covariance: See :func:`run_zircon_analysis`. Defaults to ``False``.
    """
    for dataset in datasets:
        with log_pipeline_run(f"{dataset} distribution overlap"):
            data, base_directory = overlap_data(
                dataset, random_seed=random_seed, synthetic_covariance=synthetic_covariance
            )
            pipeline_OVL(data, output_directory=base_directory / "overlap", random_seed=random_seed)


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

    parser = argparse.ArgumentParser(description="Run the zircon and synthetic analysis pipelines.")
    parser.add_argument(
        "-d",
        "--data",
        nargs="*",
        choices=list(ZIRCON_PIPELINES),
        metavar="DATASET",
        help="Run the analysis pipeline once (seed -r) for one or more datasets "
        f"({', '.join(ZIRCON_PIPELINES)}), e.g. -d michigan. With no datasets, runs all of them "
        "(including synthetic).",
    )
    parser.add_argument(
        "-m",
        "--model",
        nargs="+",
        choices=list(MODEL_PIPELINES),
        default=None,
        metavar="MODEL",
        help=f"Model(s) to fit ({', '.join(MODEL_PIPELINES)}). Accepts one or more values, run "
        f"in turn (e.g. -m tempered naive). Defaults to {DEFAULT_FIT_MODEL}, except that with -o "
        "and no -m no model is fitted.",
    )
    parser.add_argument(
        "-o",
        "--overlap",
        action="store_true",
        help="Compute the distribution overlap (OVL) diagnostics once for each -d dataset, into "
        "output/<dataset>/overlap/, using seed -r (and -c for synthetic). No model is fitted "
        "unless -m is also given.",
    )
    parser.add_argument(
        "-l",
        "--loop",
        type=int,
        default=None,
        metavar="N",
        help="Number of consecutive seeds to run the -d datasets for, starting at -r (seeds -r "
        "to -r + N - 1), e.g. -d san-juan -l 50. Requires -d. Defaults to 1 (a single run).",
    )
    parser.add_argument(
        "-r",
        "--random-seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for reproducibility. Defaults to {RANDOM_SEED}.",
    )
    parser.add_argument(
        "-c",
        "--covariance",
        action="store_true",
        help="For the synthetic dataset, generate the features with the real SRMVF "
        "within-category covariance (output in synthetic/withcov). By default they are "
        "independent (synthetic/nocov). Ignored by the other datasets.",
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
    if args.loop is not None and args.data is None:
        parser.error("-l/--loop requires -d/--data to choose the datasets to run")
    if args.loop is not None and args.loop < 1:
        parser.error("-l/--loop must be at least 1")
    if args.overlap and args.data is None:
        parser.error("-o/--overlap requires -d/--data to choose the datasets")

    if args.overlap:
        run_zircon_overlap(
            args.data or tuple(ZIRCON_PIPELINES),
            random_seed=args.random_seed,
            synthetic_covariance=args.covariance,
        )

    # -o on its own only computes the overlap diagnostics; otherwise fit the default model
    fit_models: list[str] = args.model or ([] if args.overlap else [DEFAULT_FIT_MODEL])
    for model in fit_models:
        logger.info("Running with model: %s", model)

        # An empty list means -d was given without names, so run all datasets
        if args.data is not None:
            run_zircon_analysis_loop(
                model=model,
                datasets=args.data or tuple(ZIRCON_PIPELINES),
                n_seeds=args.loop or 1,
                start_seed=args.random_seed,
                synthetic_covariance=args.covariance,
            )

    if args.final_stats is not None:
        for model in args.model or [DEFAULT_FIT_MODEL]:
            run_final_stats(model=model, datasets=args.final_stats or tuple(FINAL_STATS_RUNS))
