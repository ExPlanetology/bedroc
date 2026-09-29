# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Synthetic zircon-like datasets calibrated from the San Juan volcanic field data"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from bedroc import OUTPUT_ROOT, RANDOM_SEED
from bedroc.applications.zircons.srmvf import process_SRMVF
from bedroc.core.data_container import DataContainer
from bedroc.difference import DEFAULT_FIT_MODEL, FitModel
from bedroc.difference.group_synthetic import SyntheticDataGenerator
from bedroc.difference.group_synthetic import run_pipeline as _run_synthetic_pipeline
from bedroc.difference.utils import (
    distribution_overlap,
    effect_size_from_overlap,
    log_pipeline_run,
)

logger: logging.Logger = logging.getLogger(__name__)

DATASET_NAME: str = "synthetic"
"""Name for the synthetic zircon-like dataset analysis"""


def srmvf_calibration() -> dict[str, Any]:
    """Calibrates synthetic data generation from the real SRMVF data.

    Matches the SRMVF data's covariance matrix, per-feature effect size, sample count, and
    category balance, so that synthetic data generated with the returned settings statistically
    resembles the real data by construction.

    Returns:
        Keyword arguments for :class:`~bedroc.difference.group_synthetic.SyntheticDataGenerator`
        (excluding ``random_seed``), with ``covariance`` set to the real within-category
        covariance matrix
    """
    real_data: DataContainer = process_SRMVF(output_directory=None)

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

    return {
        "n_samples": real_data.n_data,
        "n_features": real_data.n_features,
        "feature_offsets": feature_offsets,
        "feature_sigma": 1.0,  # Exact for standardized data; unused when covariance is given
        "category_0_fraction": category_0_fraction,
        "covariance": covariance_matrix,
    }


def run_pipeline(
    model: FitModel = DEFAULT_FIT_MODEL,
    *,
    output_directory: Path | None = OUTPUT_ROOT / DATASET_NAME,
    random_seed: int | None = RANDOM_SEED,
    with_covariance: bool = False,
    proportions: tuple[float, float] | None = None,
    sizes: tuple[int, int] | None = None,
) -> None:
    """Runs the analysis pipeline for SRMVF-calibrated synthetic data, with or without the real
    SRMVF covariance structure.

    The data are generated from :func:`srmvf_calibration`, so the "with covariance" case
    statistically resembles the real data by construction, and the "without covariance" case is
    a true ablation of it (identical feature offsets, sample count, and category balance; only
    the covariance structure is removed, leaving independent features). The output goes to the
    ``withcov`` or ``nocov`` subdirectory of ``output_directory``, respectively.

    By default a single dataset is generated and a held-out split of it, with the same category
    proportions, is the target population. If ``proportions`` or ``sizes`` is given, separate
    training and test sets are generated instead, so their category proportions can differ; the
    output then goes to a further subdirectory named by :func:`split_name`.

    Args:
        model: Model to fit. Defaults to :obj:`DEFAULT_FIT_MODEL`.
        output_directory: Directory to save the output. Defaults to
            ``OUTPUT_ROOT / DATASET_NAME``.
        random_seed: Seed for random number generation to enable reproducibility. Defaults to
            :obj:`RANDOM_SEED`.
        with_covariance: Whether to generate the features with the real SRMVF within-category
            covariance (``True``) or independently (``False``). Defaults to ``False``.
        proportions: Category-0 (plutonic) fractions of the training and test sets. Defaults to
            ``None``, meaning the SRMVF-calibrated fraction for both.
        sizes: Numbers of samples in the training and test sets. Defaults to ``None``, meaning
            the SRMVF sample count split 80/20.
    """
    generator: SyntheticDataGenerator = _generator(
        random_seed=random_seed,
        with_covariance=with_covariance,
        proportions=proportions,
        sizes=sizes,
    )
    case: Path = Path(case_name(with_covariance))
    if generator.n_test is not None:
        case = case / split_name(
            (generator.category_0_fraction, generator.test_category_0_fraction),
            (generator.n_samples, generator.n_test),
        )

    with log_pipeline_run(f"SRMVF-calibrated synthetic analysis ({case}) with model: {model}"):
        _run_synthetic_pipeline(
            generator,
            model=model,
            output_directory=None if output_directory is None else output_directory / case,
        )


def build_synthetic_dataset(
    *, random_seed: int | None = RANDOM_SEED, with_covariance: bool = False
) -> DataContainer:
    """Generates the SRMVF-calibrated synthetic dataset that :func:`run_pipeline` analyses.

    Args:
        random_seed: Seed for the data generation. Defaults to :obj:`RANDOM_SEED`.
        with_covariance: See :func:`run_pipeline`. Defaults to ``False``.

    Returns:
        The generated data, with a known category for every sample
    """
    generator: SyntheticDataGenerator = _generator(
        random_seed=random_seed, with_covariance=with_covariance
    )
    generator.generate()

    return generator.to_data_container(name="Synthetic")


def case_name(with_covariance: bool) -> str:
    """Output subdirectory name for the covariance setting."""
    return "withcov" if with_covariance else "nocov"


def split_name(proportions: tuple[float, float], sizes: tuple[int, int]) -> str:
    """Output subdirectory name for separate training and test sets.

    Args:
        proportions: Category-0 fractions of the training and test sets
        sizes: Numbers of samples in the training and test sets

    Returns:
        e.g. ``"train0.50x942_test0.20x236"``
    """
    return f"train{proportions[0]:.2f}x{sizes[0]}_test{proportions[1]:.2f}x{sizes[1]}"


def _generator(
    *,
    random_seed: int | None,
    with_covariance: bool,
    proportions: tuple[float, float] | None = None,
    sizes: tuple[int, int] | None = None,
) -> SyntheticDataGenerator:
    """Builds the SRMVF-calibrated synthetic data generator (see :func:`srmvf_calibration`).

    Args:
        random_seed: Seed for the data generation
        with_covariance: Whether to use the real SRMVF within-category covariance
        proportions: Category-0 fractions of separate training and test sets. Defaults to
            ``None``; see :func:`run_pipeline`.
        sizes: Sizes of separate training and test sets. Defaults to ``None``; see
            :func:`run_pipeline`.

    Returns:
        The configured (not yet generated) generator. It has a separate test set only if
        ``proportions`` or ``sizes`` is given.
    """
    calibration: dict[str, Any] = srmvf_calibration()
    covariance = calibration.pop("covariance")

    if proportions is not None or sizes is not None:
        if sizes is None:
            # Match the self-validation split (80/20) of the calibrated sample count
            n_train: int = round(0.8 * calibration["n_samples"])
            sizes = (n_train, calibration["n_samples"] - n_train)
        if proportions is None:
            fraction: float = calibration["category_0_fraction"]
            proportions = (fraction, fraction)
        calibration.update(
            n_samples=sizes[0],
            category_0_fraction=proportions[0],
            n_test=sizes[1],
            test_category_0_fraction=proportions[1],
        )

    return SyntheticDataGenerator(
        **calibration,
        covariance=covariance if with_covariance else None,
        random_seed=random_seed,
    )
