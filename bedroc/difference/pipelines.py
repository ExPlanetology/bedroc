# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pipelines"""

import logging
from pathlib import Path

from bedroc.core.data_container import RANDOM_SEED, DataContainer
from bedroc.core.plotting import save_figure
from bedroc.difference import DEFAULT_INFERENCE_MODEL, InferenceModel
from bedroc.difference.group_classifier import GroupClassifierModel
from bedroc.difference.group_classifier import pipeline as pipeline_group_classifier
from bedroc.difference.group_covariance import pipeline as pipeline_covariance
from bedroc.difference.group_difference import HierarchicalGroupDifferenceModel
from bedroc.difference.group_difference import pipeline as pipeline_hierarchical_group_difference
from bedroc.difference.group_plotter import plot_distribution_overlap
from bedroc.difference.group_tempered import pipeline as pipeline_tempered
from bedroc.difference.utils import joint_naive_bayes_overlap, joint_overlap

logger: logging.Logger = logging.getLogger(__name__)


def pipeline_OVL(
    data: DataContainer,
    *,
    group_names: tuple[str, str],
    group_data_column: str,
    output_directory: Path | None = None,
):
    """Calculates distribution overlaps (OVL) for each feature.

    Args:
        data: The container holding the input data for the pipeline
        group_names: A tuple containing the names of the two groups for classification
        group_data_column: The name of the column in the metadata that contains the group indices
        output_directory: Path to the directory where output files will be saved. If ``None``, no
            output files will be saved.
    """
    logger.info("Running pipeline for distribution overlaps (OVL) for %s", data.name)

    if output_directory is not None:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

    # Plot distribution overlaps for each feature (OVL coefficient)
    for feature in data.feature_names:
        logger.info("Calculating distribution overlap for feature: %s", feature)
        fig, _, overlap = plot_distribution_overlap(
            data.values_std.loc[data.metadata[group_data_column] == 0, feature].to_numpy(),
            data.values_std.loc[data.metadata[group_data_column] == 1, feature].to_numpy(),
            group_names=group_names,
        )
        fig.suptitle(f"{data.name}: {feature} distribution overlap (OVL = {overlap:.2f})")
        save_figure(fig, Path(f"{data.name}_{feature}_distribution_overlap"), output_directory)

    # This breaks with NaNs values in the data, like for Ti
    logger.info("Calculating joint naive Bayes overlap")
    values_0 = data.values_std.loc[
        data.metadata[group_data_column] == 0, data.feature_names
    ].to_numpy()
    values_1 = data.values_std.loc[
        data.metadata[group_data_column] == 1, data.feature_names
    ].to_numpy()
    joint_naive_bayes_overlap(values_0, values_1)  # Outputs to the logger

    # Joint empirical overlap
    logger.info("Calculating joint empirical overlap")
    joint_overlap(values_0, values_1)  # Outputs to the logger

    logger.info("Pipeline for distribution overlaps (OVL) completed for %s", data.name)


def pipeline_two_stage_inference(
    data: DataContainer,
    *,
    group_names: tuple[str, str],
    group_data_column: str,
    output_directory: Path | None = None,
    random_seed: int | None = RANDOM_SEED,
    title_fontsize: str = "large",
) -> GroupClassifierModel:
    """Two-stage pipeline for Bayesian classification and group-fraction inference.

    This function orchestrates the two-stage process of Bayesian classification and group-fraction
    inference based on hierarchical group models.

    Args:
        data: The container holding the input data for the pipeline
        group_names: A tuple containing the names of the two groups for classification
        group_data_column: The name of the column in the metadata that contains the group indices
        output_directory (Path | None): Optional path to the directory where output files will be
            saved. If ``None``, no output files will be saved.
        random_seed: Optional random seed for reproducible results. Defaults to :obj:`RANDOM_SEED`.
        title_fontsize: Font size for plot titles. Defaults to ``large``.

    Returns:
        Group classifier model
    """
    logger.info("Running two-stage inference pipeline for %s", data.name)

    fitted_model: HierarchicalGroupDifferenceModel = pipeline_hierarchical_group_difference(
        data,
        group_names=group_names,
        group_data_column=group_data_column,
        output_directory=output_directory,
        random_seed=random_seed,
        title_fontsize=title_fontsize,
    )

    classifier_model: GroupClassifierModel = pipeline_group_classifier(
        data,
        fitted_model=fitted_model,
        group_data_column=group_data_column,
        output_directory=output_directory,
        random_seed=random_seed,
        title_fontsize=title_fontsize,
    )

    logger.info("Two-stage inference pipeline completed for %s", data.name)

    return classifier_model


def run_pipeline(
    data: DataContainer,
    inference: InferenceModel = DEFAULT_INFERENCE_MODEL,
    *,
    group_names: tuple[str, str],
    group_data_column: str,
    output_directory: Path | None = None,
    random_seed: int | None = RANDOM_SEED,
    title_fontsize: str = "large",
    OVL: bool = True,
) -> None:
    """Runs the full analysis pipeline for a dataset.

    This function orchestrates the entire analysis pipeline, including distribution overlap
    calculations, hierarchical group difference modeling, and Bayesian classification.

    Args:
        data: The container holding the input data for the pipeline
        inference: Type of inference to run. Defaults to :obj:`DEFAULT_INFERENCE_MODEL`.
        group_names: A tuple containing the names of the two groups for classification
        group_data_column: The name of the column in the metadata that contains the group indices
        output_directory: Optional path to the directory where output files will be saved. If
            ``None``, no output files will be saved.
        random_seed: Optional random seed for reproducible results. Defaults to :obj:`RANDOM_SEED`.
        title_fontsize: Font size for plot titles. Defaults to ``large``.
        OVL: Whether to calculate distribution overlaps (OVL) for each feature. Defaults to
            ``True``.
    """
    logger.info("Running full analysis pipeline for %s", data.name)

    if OVL:
        pipeline_OVL(
            data,
            group_names=group_names,
            group_data_column=group_data_column,
            output_directory=output_directory,
        )

    if inference == "covariance":
        pipeline_covariance(
            data,
            group_names=group_names,
            group_data_column=group_data_column,
            output_directory=output_directory,
            random_seed=random_seed,
            title_fontsize=title_fontsize,
        )
    elif inference == "tempered":
        pipeline_tempered(
            data,
            group_names=group_names,
            group_data_column=group_data_column,
            output_directory=output_directory,
            random_seed=random_seed,
            title_fontsize=title_fontsize,
        )
    elif inference == "two-stage":
        pipeline_two_stage_inference(
            data,
            group_names=group_names,
            group_data_column=group_data_column,
            output_directory=output_directory,
            random_seed=random_seed,
            title_fontsize=title_fontsize,
        )

    logger.info("Full analysis pipeline completed for %s", data.name)
