# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pipelines"""

import logging
from pathlib import Path

from bedroc import RANDOM_SEED
from bedroc.core.data_container import DataContainer
from bedroc.core.plotting import save_figure
from bedroc.difference import DEFAULT_INFERENCE_MODEL, InferenceModel
from bedroc.difference.group_tempered import pipeline as pipeline_tempered
from bedroc.difference.models.unified_covariance import pipeline as pipeline_covariance
from bedroc.difference.models.standard_classifier import StandardClassifierModel
from bedroc.difference.models.standard_classifier import pipeline as pipeline_standard_classifier
from bedroc.difference.models.standard_difference import StandardDifferenceModel
from bedroc.difference.models.standard_difference import pipeline as pipeline_category_difference
from bedroc.difference.plotting import plot_distribution_overlap
from bedroc.difference.utils import joint_naive_bayes_overlap, joint_overlap

logger: logging.Logger = logging.getLogger(__name__)


def pipeline_OVL(data: DataContainer, *, output_directory: Path | None = None):
    """Calculates distribution overlaps (OVL) for each feature.

    This function will only compute the overlap for two categories in the data.

    Args:
        data: The container holding the input data for the pipeline
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
            data.values_std.loc[data.category_codes == 0, feature].to_numpy(),
            data.values_std.loc[data.category_codes == 1, feature].to_numpy(),
            group_names=data.category_names,  # pyright: ignore[reportArgumentType]
        )
        fig.suptitle(f"{data.name}: {feature} distribution overlap (OVL = {overlap:.2f})")
        save_figure(fig, Path(f"{data.name}_{feature}_distribution_overlap"), output_directory)

    # This breaks with NaNs values in the data, like for Ti
    logger.info("Calculating joint naive Bayes overlap")
    values_0 = data.values_std.loc[data.category_codes == 0, data.feature_names].to_numpy()
    values_1 = data.values_std.loc[data.category_codes == 1, data.feature_names].to_numpy()
    joint_naive_bayes_overlap(values_0, values_1)  # Outputs to the logger

    # Joint empirical overlap
    logger.info("Calculating joint empirical overlap")
    joint_overlap(values_0, values_1)  # Outputs to the logger

    logger.info("Pipeline for distribution overlaps (OVL) completed for %s", data.name)


def pipeline_two_stage_inference(
    data: DataContainer,
    *,
    output_directory: Path | None = None,
    random_seed: int | None = RANDOM_SEED,
    **kwargs,
) -> StandardClassifierModel:
    """Two-stage pipeline for Bayesian classification and category-fraction inference.

    This function orchestrates the two-stage process of Bayesian classification and
    category-fraction inference based on hierarchical category models.

    Args:
        data: The container holding the input data for the pipeline
        output_directory (Path | None): Optional path to the directory where output files will be
            saved. If ``None``, no output files will be saved.
        random_seed: Optional random seed for reproducible results. Defaults to :obj:`RANDOM_SEED`.
        **kwargs: Additional keyword arguments to pass to the underlying pipeline functions.

    Returns:
        StandardClassifierModel: The fitted category classifier model.
    """
    logger.info("Running two-stage inference pipeline for %s", data.name)

    fitted_model: StandardDifferenceModel = pipeline_category_difference(
        data, output_directory=output_directory, random_seed=random_seed, **kwargs
    )

    classifier_model: StandardClassifierModel = pipeline_standard_classifier(
        data,
        fitted_model=fitted_model,
        output_directory=output_directory,
        random_seed=random_seed,
        **kwargs,
    )

    logger.info("Two-stage inference pipeline completed for %s", data.name)

    return classifier_model


def run_pipeline(
    data: DataContainer,
    *,
    inference: InferenceModel = DEFAULT_INFERENCE_MODEL,
    output_directory: Path | None = None,
    random_seed: int | None = RANDOM_SEED,
    OVL: bool = True,
    **pipeline_kwargs,
) -> None:
    """Runs the full analysis pipeline for a dataset.

    This function orchestrates the entire analysis pipeline, including distribution overlap
    calculations, hierarchical category difference modeling, and Bayesian classification.

    Args:
        data: The container holding the input data for the pipeline
        inference: Type of inference to run. Defaults to :obj:`DEFAULT_INFERENCE_MODEL`.
        output_directory: Optional path to the directory where output files will be saved. If
            ``None``, no output files will be saved.
        random_seed: Optional random seed for reproducible results. Defaults to :obj:`RANDOM_SEED`.
        OVL: Whether to calculate distribution overlaps (OVL) for each feature. Defaults to
            ``True``.
        **pipeline_kwargs: Additional keyword arguments to pass to the underlying pipeline
            functions.
    """
    logger.info("Running full analysis pipeline for %s", data.name)

    if OVL:
        pipeline_OVL(data, output_directory=output_directory)

    if inference == "covariance":
        pipeline_covariance(
            data, output_directory=output_directory, random_seed=random_seed, **pipeline_kwargs
        )
    elif inference == "tempered":
        pipeline_tempered(
            data, output_directory=output_directory, random_seed=random_seed, **pipeline_kwargs
        )
    # TODO: Must pass in **kwargs to allow for additional parameters
    elif inference == "two-stage":
        pipeline_two_stage_inference(
            data, output_directory=output_directory, random_seed=random_seed, **pipeline_kwargs
        )

    logger.info("Full analysis pipeline completed for %s", data.name)
