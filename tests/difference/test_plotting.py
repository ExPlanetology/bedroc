# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Rendering tests for bedroc.difference.plotting's corner plots."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from bedroc.core.data_container import DataContainer
from bedroc.difference.plotting import plot_corner_with_unlabeled

matplotlib.use("Agg")


def _labeled_and_mixture(
    fraction_0: float, *, n_labeled: int = 600, n_unlabeled: int = 2000
) -> tuple[DataContainer, DataContainer]:
    """Two labeled groups and an unlabeled set drawn as an exact mixture of the same groups."""
    rng = np.random.default_rng(0)
    means = (np.array([0.0, 0.0, 0.0]), np.array([2.0, -1.5, 1.0]))

    def draw(n: int, fraction: float) -> tuple[np.ndarray, np.ndarray]:
        n_0 = round(n * fraction)
        X = np.vstack(
            [rng.normal(means[0], 1.0, (n_0, 3)), rng.normal(means[1], 1.0, (n - n_0, 3))]
        )
        return X, np.array(["a"] * n_0 + ["b"] * (n - n_0))

    X, labels = draw(n_labeled, 0.5)
    labeled = DataContainer(
        pd.DataFrame(X, columns=["f0", "f1", "f2"]),
        metadata=pd.DataFrame({"Type": labels}),
        category_column="Type",
        name="test",
    )
    X_unlabeled, _ = draw(n_unlabeled, fraction_0)
    unlabeled = DataContainer(
        pd.DataFrame(X_unlabeled, columns=["f0", "f1", "f2"]),
        name="test_unlabeled",
        scaling_params=labeled.scaling,
    )
    return labeled, unlabeled


def test_plot_corner_with_unlabeled_saves_plot_and_recovers_mixture_weight(
    tmp_path: Path,
) -> None:
    labeled, unlabeled = _labeled_and_mixture(0.7)

    weight = plot_corner_with_unlabeled(
        labeled,
        unlabeled,
        unlabeled_label="Mixture",
        output_directory=tmp_path,
        savefig_kwargs={"format": "png", "dpi": 40},
    )

    assert abs(weight - 0.7) < 0.05
    assert (tmp_path / "test_Mixture_vs_groups.png").exists()


def test_mixture_weight_recovers_known_fraction() -> None:
    from scipy.stats import norm

    from bedroc.difference.utils import mixture_weight

    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(0.0, 1.0, 3000), rng.normal(3.0, 1.0, 7000)])

    weight = mixture_weight(norm(0.0, 1.0).logpdf(x), norm(3.0, 1.0).logpdf(x))

    assert abs(weight - 0.3) < 0.02
