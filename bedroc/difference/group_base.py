# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Base classes and protocols"""

import logging
from abc import abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import pymc as pm
import xarray as xr

from bedroc.core.data_container import RANDOM_SEED
from bedroc.core.type_aliases import NpArray, NpFloat, NpInt
from bedroc.difference.utils import get_coords
from bedroc.difference.validation import validate_group_idx, validate_observation_data

logger: logging.Logger = logging.getLogger(__name__)


class GroupComparisonBase:
    """Base class for group comparison models

    Args:
        name: Name of the model or analysis
        X: Observation data, shape (n_samples, n_features)
        X_group_idx: Group indices for each sample, shape (n_samples,)
        X_sigma: Optional observation uncertainties, shape (n_samples, n_features). Defaults to
            ``None``, in which case the model assumes that the observations are exact.
        feature_names: Optional names for each feature. If not provided, defaults to
            ``["Feature 0", "Feature 1", ..., "Feature N"]``.
        group_names: Optional names for each group. If not provided, defaults to
            ``["Group 0", "Group 1"]``.
    """

    def __init__(
        self,
        name: str,
        X: NpFloat,
        X_group_idx: NpInt,
        *,
        X_sigma: NpFloat | None = None,
        feature_names: Iterable | None = None,
        group_names: Iterable | None = None,
    ):
        self.name: str = name
        self.X, self.X_sigma = validate_observation_data(X, X_sigma=X_sigma)
        self.X_group_idx = validate_group_idx(X_group_idx, n_samples=self.X.shape[0])
        self.coords: dict[str, NpArray] = get_coords(
            self.X, self.X_group_idx, feature_names=feature_names, group_names=group_names
        )
        self._idata: xr.DataTree | None = None
        self._model: pm.Model | None = None

    @property
    def difference_string(self) -> str:
        """Return a human-readable representation of group 1 relative to group 0."""
        return f"({self.coords['group'][1]} - {self.coords['group'][0]})"

    @property
    def idata(self) -> xr.DataTree:
        """Inference data containing posterior samples"""
        if self._idata is None:
            raise ValueError("Inference has not been run yet. Call `run_inference()` first.")
        else:
            return self._idata

    @property
    def model(self) -> pm.Model:
        """PyMC model object"""
        if self._model is None:
            raise ValueError("Model has not been built yet. Call `build_model()` first.")
        else:
            return self._model

    @abstractmethod
    def build_model(self) -> pm.Model:
        """Builds the PyMC model for the group comparison and stores it in ``self._model``."""
        raise NotImplementedError("Subclasses must implement this method.")

    def plot_model(self, output_directory: Path | str, *, format: str = "pdf") -> Path:
        """Exports a graph of the PyMC model to a PDF file.

        Args:
            output_directory: Directory to save the model graph. If it does not exist, it will be
                created.
            format: Format of the output file. Defaults to ``'pdf'``. Can be any format supported
                by Graphviz.

        Returns:
            Path to the saved model graph file
        """
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        graph = pm.model_to_graphviz(self.model)

        # Should not include extension for graph.render()
        out_path: Path = output_directory / Path(f"{self.name}_model_graph")

        graph.render(out_path, format=format, cleanup=True)

        # Add format for the return path to match the saved file
        out_path = out_path.with_suffix(f".{format}")

        return out_path

    def run_inference(
        self,
        *,
        draws: int = 2000,
        tune: int = 1000,
        target_accept: float = 0.95,
        random_seed: int | None = RANDOM_SEED,
        **kwargs,
    ) -> None:
        """Runs inference on the hierarchical model.

        Args:
            draws: Number of posterior samples to draw. Defaults to ``2000``.
            tune: Number of tuning steps. Defaults to ``1000``.
            target_accept: Target acceptance rate for NUTS sampler. Defaults to ``0.95``.
            random_seed: Random seed for reproducibility. Defaults to :obj:`RANDOM_SEED`.
            **kwargs: Arbitrary keyword arguments passed to :func:`pymc.sample`. See PyMC
                documentation for details.
        """
        logger.info(
            "Running inference with draws=%d, tune=%d, target_accept=%.2f, random_seed=%s",
            draws,
            tune,
            target_accept,
            random_seed,
        )

        self._idata = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            random_seed=random_seed,
            model=self.model,
            **kwargs,
        )


class GroupClassifierProtocol(Protocol):
    """Protocol for group classifiers

    This protocol defines the expected interface for group classifiers. Any class that implements
    this protocol should provide the following methods and properties.
    """

    def pi_0_samples(self) -> NpFloat:
        """Posterior samples of the fraction of samples belonging to group 0 in the unlabeled
        dataset."""
        ...
