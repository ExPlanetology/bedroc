# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Joint Bayesian inference of group differences and population fraction for two groups"""

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pymc as pm
from matplotlib.axes import Axes

from bedroc import override
from bedroc.core.data_container import RANDOM_SEED, DataContainer
from bedroc.core.plotting import save_figure
from bedroc.core.type_aliases import NpFloat, NpInt
from bedroc.difference.group_base import GroupClassifierProtocol, GroupComparisonBase
from bedroc.difference.plotting import plot_group_fraction_posterior
from bedroc.difference.validation import validate_observation_data

logger: logging.Logger = logging.getLogger(__name__)


class UnifiedGroupDifferenceModel(GroupComparisonBase, GroupClassifierProtocol):
    """Joint Bayesian inference of group differences and population fraction for two groups

    This model simultaneously infers the group parameters and the fraction of samples belonging to
    each group in an unlabeled dataset.

    Args:
        name: Name of the model or analysis
        X_train: Observation data for the labeled training set, shape (n_samples, n_features)
        X_group_idx_train: Group indices for each sample in the training set, shape (n_samples,)
        X_unlabeled: Observation data for the unlabeled target set, shape (n_samples, n_features)
        X_sigma_train: Optional observation uncertainties for the training set, shape
            (n_samples, n_features). Defaults to ``None``, in which case the model assumes that the
            observations are exact.
        X_sigma_unlabeled: Optional observation uncertainties for the unlabeled target set, shape
            (n_samples, n_features). Defaults to ``None``, in which case the model assumes that the
            observations are exact.
        feature_names: Optional names for each feature. If not provided, defaults to
            ``["Feature 0", "Feature 1", ..., "Feature N"]``.
        group_names: Optional names for each group. If not provided, defaults to
            ``["Group 0", "Group 1"]``.
    """

    def __init__(
        self,
        name: str,
        X_train: NpFloat,
        X_group_idx_train: NpInt,
        X_unlabeled: NpFloat,
        *,
        X_sigma_train: NpFloat | None = None,
        X_sigma_unlabeled: NpFloat | None = None,
        feature_names: Iterable | None = None,
        group_names: Iterable | None = None,
    ):
        logger.info("Creating a unified group difference model for %s", name)
        super().__init__(
            name,
            X_train,
            X_group_idx_train,
            X_sigma=X_sigma_train,
            feature_names=feature_names,
            group_names=group_names,
        )
        self.X_unlabeled, self.X_sigma_unlabeled = validate_observation_data(
            X_unlabeled, X_sigma=X_sigma_unlabeled
        )
        self._prior_alpha: float
        self._prior_beta: float

    @override
    def pi_0_samples(self) -> NpFloat:
        """Posterior samples of the fraction of samples belonging to group 0 in the unlabeled dataset"""
        return self.idata.posterior["pi_0"].values.flatten()

    @override
    def build_model(self, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        """Builds the PyMC model and stores it in ``self._model``.

        Args:
            prior_alpha: Alpha parameter for the Beta prior on the group fraction. Defaults to
                ``1.0``.
            prior_beta: Beta parameter for the Beta prior on the group fraction. Defaults to
                ``1.0``.
        """
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta

        # Flatten finite elements for labeled training data
        train_s_idx, train_f_idx = np.where(np.isfinite(self.X))
        train_g_idx = self.X_group_idx[train_s_idx]
        X_train_data = self.X[train_s_idx, train_f_idx]
        X_train_sigma_data = self.X_sigma[train_s_idx, train_f_idx]

        # Flatten finite elements for unlabeled target data
        unlab_s_idx, unlab_f_idx = np.where(np.isfinite(self.X_unlabeled))
        X_unlab_data = self.X_unlabeled[unlab_s_idx, unlab_f_idx]
        X_unlab_sigma_data = self.X_sigma_unlabeled[unlab_s_idx, unlab_f_idx]

        with pm.Model(coords=self.coords) as model:
            # Priors on Group Parameters (Shared)
            mu_0 = pm.Normal("mu_0", mu=0, sigma=0.5, dims="feature")
            delta_scale = pm.HalfNormal("delta_scale", sigma=0.5)
            delta = pm.Normal("delta", mu=0, sigma=delta_scale, dims="feature")

            # Group means: shape (2, n_features)
            mu = pm.Deterministic(
                "mu", pm.math.stack([mu_0, mu_0 + delta], axis=0), dims=("group", "feature")
            )
            sigma = pm.HalfNormal("sigma", sigma=0.5, dims="group")

            # Strongly constrain nu >= 2.0 to ensure finite variance and clean gradients
            nu_minus_2 = pm.Exponential("nu_minus_2", 1 / 29.0, dims="group")
            nu = pm.Deterministic("nu", nu_minus_2 + 2.0, dims="group")

            # Fraction prior
            pi_0 = pm.Beta("pi_0", alpha=prior_alpha, beta=prior_beta)
            # Stack weights into shape (2,) for PyMC Mixture input
            w = pm.math.stack([pi_0, 1.0 - pi_0])

            # Labeled Training Likelihood
            sigma_obs_train = pm.math.sqrt(X_train_sigma_data**2 + sigma[train_g_idx] ** 2)
            mu_obs_train = mu[train_g_idx, train_f_idx]  # pyright: ignore
            nu_obs_train = nu[train_g_idx]  # pyright: ignore

            pm.StudentT(
                "obs_train",
                mu=mu_obs_train,
                sigma=sigma_obs_train,
                nu=nu_obs_train,
                observed=X_train_data,
                shape=X_train_data.shape,
            )

            # Unlabeled Mixture Likelihood using pm.Mixture
            mu_unlab_0 = mu[0, unlab_f_idx]  # pyright: ignore
            sigma_unlab_0 = pm.math.sqrt(X_unlab_sigma_data**2 + sigma[0] ** 2)
            nu_unlab_0 = nu[0]  # pyright: ignore

            mu_unlab_1 = mu[1, unlab_f_idx]  # pyright: ignore
            sigma_unlab_1 = pm.math.sqrt(X_unlab_sigma_data**2 + sigma[1] ** 2)
            nu_unlab_1 = nu[1]  # pyright: ignore

            # Define component distributions
            comp_0 = pm.StudentT.dist(nu=nu_unlab_0, mu=mu_unlab_0, sigma=sigma_unlab_0)
            comp_1 = pm.StudentT.dist(nu=nu_unlab_1, mu=mu_unlab_1, sigma=sigma_unlab_1)

            # Create mixture
            pm.Mixture("obs_unlabeled", w=w, comp_dists=[comp_0, comp_1], observed=X_unlab_data)

        self._model = model

    def plot_group_fraction_posterior(
        self,
        bins: int = 50,
        n_grid: int = 2001,
        group_colors: tuple[str, str] = ("tab:blue", "tab:orange"),
        group_counts: tuple[float, float] | None = None,
        ax: Axes | None = None,
    ) -> Axes:
        """Plots the posterior distribution of the fraction of samples belonging to group 0.

        Args:
            bins: Number of bins for the histogram. Defaults to ``50``.
            n_grid: Number of grid points for the prior and perfect-classification limit. Defaults to
                ``2001``.
            group_colors: Colors for the two groups. Defaults to ``("tab:blue", "tab:orange")``.
            group_counts: Known counts for the two groups. If ``None``, the observed fractions are not
                plotted. Defaults to ``None``.
            ax: Matplotlib axes on which to plot. If ``None``, a new figure and axes are created.

        Returns:
            Matplotlib axes containing the posterior group-fraction plot
        """
        return plot_group_fraction_posterior(
            self.pi_0_samples(),
            prior_alpha=self._prior_alpha,
            prior_beta=self._prior_beta,
            bins=bins,
            n_grid=n_grid,
            group_names=self.coords["group"],
            group_colors=group_colors,
            group_counts=group_counts,
            ax=ax,
        )


def pipeline(
    data: DataContainer,
    *,
    group_names: tuple[str, str],
    group_data_column: str,
    output_directory: Path | None = None,
    random_seed: int | None = RANDOM_SEED,
    title_fontsize: str = "large",
) -> None:
    """Pipeline for running the unified group difference model on a dataset.

    This provides a basic pipeline for running a standard analysis and generating the associated
    figures. For more customized analyses, you may wish to create your own pipeline.

    Args:
        data: The container holding the input data for the pipeline
        group_names: A tuple containing the names of the two groups for classification
        group_data_column: The name of the column in the metadata that contains the group indices
        output_directory (Path | None): Optional path to the directory where output files will be
            saved. If ``None``, no output files will be saved.
        random_seed: Random seed for reproducible results. Defaults to :obj:`RANDOM_SEED`.
        title_fontsize: Font size for plot titles. Defaults to ``large``.
    """
    logger.info("Running unified group difference pipeline for %s", data.name)

    if output_directory is not None:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory: %s", output_directory)
    else:
        logger.info("Output directory not specified. Figures will not be saved.")

    train, test = data.train_test_split(
        random_state=random_seed, stratify=data.metadata[group_data_column]
    )

    fitted_model: UnifiedGroupDifferenceModel = UnifiedGroupDifferenceModel(
        data.name,
        train.values_std.to_numpy(),
        train.metadata[group_data_column].to_numpy(),
        test.values_std.to_numpy(),
        X_sigma_train=train.uncertainties_std.to_numpy(),
        X_sigma_unlabeled=test.uncertainties_std.to_numpy(),
        feature_names=train.feature_names,
        group_names=group_names,
    )

    fitted_model.build_model()

    if output_directory is not None:
        fitted_model.plot_model(output_directory)

    fitted_model.run_inference(random_seed=random_seed)

    logger.info("Unified group difference pipeline completed for %s", data.name)

    # Get the true group counts in the test set for plotting the observed fractions
    group_counts = (
        np.sum(test.metadata[group_data_column] == 0),
        np.sum(test.metadata[group_data_column] == 1),
    )

    # Figure generation
    ax: Axes = fitted_model.plot_group_fraction_posterior(group_counts=group_counts)
    save_figure(
        ax.get_figure(),  # pyright: ignore[reportArgumentType]
        stem=f"{data.name}_group_fraction_posterior",
        output_directory=output_directory,
    )

    logger.info("Unified group difference pipeline completed for %s", data.name)

    # plt.show()
