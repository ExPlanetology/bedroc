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
from typing import Any

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
    synthetic_options: Mapping[str, Any] | None = None,
) -> None:
    """Runs the zircon analysis pipeline for each of ``datasets``.

    Args:
        model: Model to fit. Defaults to :obj:`DEFAULT_FIT_MODEL`.
        datasets: Datasets to run, from :obj:`ZIRCON_PIPELINES`. Defaults to all of them.
        random_seed: Random seed for reproducibility. Defaults to :obj:`RANDOM_SEED`.
        synthetic_options: Keyword arguments for the synthetic dataset's pipeline only (see
            :func:`bedroc.applications.zircons.synthetic.run_pipeline`: ``with_covariance``,
            ``proportions``, ``sizes``). Ignored by the other datasets. Defaults to ``None``.
    """
    for dataset in datasets:
        kwargs: dict[str, Any] = dict(synthetic_options or {}) if dataset == "synthetic" else {}
        ZIRCON_PIPELINES[dataset](model=model, random_seed=random_seed, **kwargs)


def run_zircon_analysis_loop(
    model: FitModel = DEFAULT_FIT_MODEL,
    *,
    datasets: Sequence[str] = tuple(ZIRCON_PIPELINES),
    n_seeds: int = 1000,
    start_seed: int = 0,
    synthetic_options: Mapping[str, Any] | None = None,
) -> None:
    """Runs the zircon analysis pipeline in a loop over consecutive random seeds.

    Args:
        model: Model to fit. Defaults to :obj:`DEFAULT_FIT_MODEL`.
        datasets: Datasets to run, from :obj:`ZIRCON_PIPELINES`. Defaults to all of them.
        n_seeds: Number of random seeds to run. Defaults to ``1000``.
        start_seed: First seed; the seeds run are ``start_seed`` to
            ``start_seed + n_seeds - 1``. Defaults to ``0``.
        synthetic_options: See :func:`run_zircon_analysis`. Defaults to ``None``.
    """
    for seed in range(start_seed, start_seed + n_seeds):
        logger.info("Running zircon analysis with random seed: %d", seed)
        run_zircon_analysis(
            model=model,
            datasets=datasets,
            random_seed=seed,
            synthetic_options=synthetic_options,
        )


def overlap_data(
    dataset: str, *, random_seed: int | None, synthetic_covariance: bool
) -> tuple[DataContainer, Path]:
    """Returns the labeled data to compare, and its base output directory, for a dataset.

    Args:
        dataset: Dataset name, from :obj:`ZIRCON_PIPELINES`
        random_seed: Seed for generating the synthetic data
        synthetic_covariance: For the synthetic dataset, whether to use the real SRMVF
            within-category covariance

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
        synthetic_covariance: For the synthetic dataset, whether to use the real SRMVF
            within-category covariance. Defaults to ``False``.
    """
    for dataset in datasets:
        with log_pipeline_run(f"{dataset} distribution overlap"):
            data, base_directory = overlap_data(
                dataset, random_seed=random_seed, synthetic_covariance=synthetic_covariance
            )
            pipeline_OVL(data, output_directory=base_directory / "overlap", random_seed=random_seed)


FINAL_STATS_DATASETS: tuple[str, ...] = ("san-juan", "synthetic")
"""Datasets with a known true fraction whose runs can be summarized. Michigan is absent: its
unlabeled Detrital zircons have no known true fraction to compare against."""


def final_stats_runs(dataset: str, model: FitModel) -> list[tuple[Path, str]]:
    """Finds where a dataset's runs of a model were written.

    Args:
        dataset: Dataset name, from :obj:`FINAL_STATS_DATASETS`
        model: Model whose runs to find

    Raises:
        ValueError: If ``dataset`` is not recognized.

    Returns:
        ``(base output directory, summary file name prefix)`` pairs, one per case. For synthetic
        data, every directory under ``OUTPUT_ROOT / "synthetic"`` holding runs of ``model`` is a
        separate case (e.g. ``nocov``, ``withcov``, or a ``-p``/``-s`` training/test split).
    """
    if dataset == "san-juan":
        return [(OUTPUT_ROOT / SRMVF_DATASET_NAME, SRMVF_DATASET_NAME)]
    if dataset == "synthetic":
        synthetic_root: Path = OUTPUT_ROOT / SYNTHETIC_DATASET_NAME
        cases: set[Path] = {run.parent for run in synthetic_root.rglob(f"{model}_seed_*")}
        # "Synthetic" is the container name group_synthetic.run_pipeline gives the generated data
        return [(case, "Synthetic") for case in sorted(cases)]
    raise ValueError(f"Unrecognized dataset {dataset!r}")


def final_stats(
    model: FitModel = DEFAULT_FIT_MODEL, *, output_directory: Path, name: str
) -> pd.Series:
    r"""Summarizes the population-fraction inference across the seeds of previous runs.

    Reads each run's ``<name>_summary_statistics.xlsx`` from
    ``<output_directory>/<model>_seed_<seed>/``, i.e. every seed found there, and writes the
    summary (``<model>_final_stats.xlsx``) and a true-vs-inferred scatter plot
    (``<model>_final_stats``) to ``output_directory``.

    Every error metric uses one point estimate per run, the median of its samples (posterior
    draws for the Bayesian models, bootstrap draws for the SVM), so the metrics are comparable
    across models. With :math:`\hat\pi_{0,k}` the median and :math:`\pi_{0,k}` the true fraction
    of run :math:`k`, over :math:`K` runs:

    - Bias: :math:`\frac{1}{K} \sum_k (\hat\pi_{0,k} - \pi_{0,k})`
    - RMSE: :math:`\sqrt{\frac{1}{K} \sum_k (\hat\pi_{0,k} - \pi_{0,k})^2}`
    - MAE: :math:`\frac{1}{K} \sum_k |\hat\pi_{0,k} - \pi_{0,k}|`
    - 95% coverage: fraction of runs whose 95% interval contains the true fraction
    - Mean 95% CI width: average width of those intervals

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

    # Error of each run's point estimate (the median of its samples)
    error: pd.Series = results["median"] - results["truth"]
    summary = pd.Series(
        {
            "Number of splits": len(results),
            "Bias": error.mean(),
            "RMSE": np.sqrt((error**2).mean()),
            "MAE": error.abs().mean(),
            "95% coverage": results["within_ci"].astype(bool).mean(),
            "Mean 95% CI width": results["ci_width"].mean(),
        }
    )

    label: str = f"{model} ({output_directory})"
    print(f"{label}:")
    print(summary)
    summary.to_frame(name=model).to_excel(output_directory / f"{model}_final_stats.xlsx")

    fig, ax = plt.subplots()

    ax.scatter(results["truth"], results["median"])

    limits = [
        min(results["truth"].min(), results["median"].min()),
        max(results["truth"].max(), results["median"].max()),
    ]

    ax.plot(limits, limits, linestyle="--", color="black")

    ax.set_xlabel("Observed category-0 fraction")
    ax.set_ylabel("Inferred category-0 fraction (median)")
    ax.set_title(f"Population fraction inference: {label}")

    fig.tight_layout()
    save_figure(fig, Path(f"{model}_final_stats"), output_directory)

    return summary


def run_final_stats(
    model: FitModel = DEFAULT_FIT_MODEL, *, datasets: Sequence[str] = FINAL_STATS_DATASETS
) -> None:
    """Runs :func:`final_stats` for each dataset (and each of its cases), skipping any with no
    runs yet.

    Args:
        model: Model whose runs to summarize. Defaults to :obj:`DEFAULT_FIT_MODEL`.
        datasets: Datasets to summarize, from :obj:`FINAL_STATS_DATASETS`. Defaults to all of
            them.
    """
    for dataset in datasets:
        runs: list[tuple[Path, str]] = final_stats_runs(dataset, model)
        if not runs:
            logger.warning("Skipping %s: no %s runs found", dataset, model)
        for output_directory, name in runs:
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
        "-p",
        "--proportions",
        nargs=2,
        type=float,
        metavar=("TRAIN", "TEST"),
        help="For the synthetic dataset, generate separate training and test sets with these "
        "category-0 (plutonic) fractions, e.g. -p 0.5 0.2, so the target population's balance "
        "can differ from training. Output in synthetic/<case>/train<p>x<n>_test<p>x<n>/. "
        "Without -p or -s, one dataset is split 80/20 at the SRMVF-calibrated fraction. Ignored "
        "by -o and the other datasets.",
    )
    parser.add_argument(
        "-s",
        "--sizes",
        nargs=2,
        type=int,
        metavar=("NTRAIN", "NTEST"),
        help="For the synthetic dataset, the sizes of separate training and test sets (see -p). "
        "Defaults to the SRMVF sample count split 80/20. Ignored by -o and the other datasets.",
    )
    parser.add_argument(
        "-f",
        "--final-stats",
        nargs="*",
        choices=list(FINAL_STATS_DATASETS),
        metavar="DATASET",
        help="Summarize every seed run found on disk for the given datasets "
        f"({', '.join(FINAL_STATS_DATASETS)}), per model and per case (each synthetic "
        "-c/-p/-s setting separately). With no datasets, summarizes all of them. Writes "
        "<model>_final_stats.xlsx and a plot to each case's output directory.",
    )

    args = parser.parse_args()
    if args.loop is not None and args.data is None:
        parser.error("-l/--loop requires -d/--data to choose the datasets to run")
    if args.loop is not None and args.loop < 1:
        parser.error("-l/--loop must be at least 1")
    if args.overlap and args.data is None:
        parser.error("-o/--overlap requires -d/--data to choose the datasets")
    if args.proportions is not None and not all(0 <= p <= 1 for p in args.proportions):
        parser.error("-p/--proportions must be between 0 and 1")
    if args.sizes is not None and not all(n >= 2 for n in args.sizes):
        parser.error("-s/--sizes must be at least 2")

    synthetic_options: dict[str, Any] = {
        "with_covariance": args.covariance,
        "proportions": None if args.proportions is None else tuple(args.proportions),
        "sizes": None if args.sizes is None else tuple(args.sizes),
    }

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
                synthetic_options=synthetic_options,
            )

    if args.final_stats is not None:
        for model in args.model or [DEFAULT_FIT_MODEL]:
            run_final_stats(model=model, datasets=args.final_stats or FINAL_STATS_DATASETS)
