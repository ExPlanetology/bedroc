#
# Copyright 2025 Dan J. Bower
#
# This file is part of Bedroc.
#
# Bedroc is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Bedroc is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Bedroc. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Tests for hierarchical module."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.testing import assert_allclose

from bedroc import debug_logger
from bedroc.hierarchical import SyntheticDataGenerator

logger: logging.Logger = debug_logger()
logger.setLevel(logging.WARNING)

RANDOM_SEED: int = 123


def test_synthetic_data_generation() -> None:
    """Tests the generation of synthetic data."""

    data_generator: SyntheticDataGenerator = SyntheticDataGenerator(random_seed=RANDOM_SEED)
    data_generator.generate()

    assert_allclose(
        data_generator.true_params.mu_A,
        np.array([-0.98912135, -0.36778665, 1.28792526, 0.19397442, 0.9202309]),
    )

    assert_allclose(
        data_generator.true_params.mu_B,
        np.array([0.74219002, -2.27717759, 2.91378192, -0.75581193, -0.04693645]),
    )

    assert_allclose(
        data_generator.true_params.sigma_A,
        np.array([1.26945568, 0.8674469, 1.73636239, 0.82064445, 1.61220058]),
    )

    assert_allclose(
        data_generator.true_params.sigma_B,
        np.array([1.26945568, 0.8674469, 1.73636239, 0.82064445, 1.61220058]),
    )


def test_plot_pairgrid_structure():
    """Tests the structure of the PairGrid returned by plot()."""

    data_generator: SyntheticDataGenerator = SyntheticDataGenerator(random_seed=RANDOM_SEED)
    data_generator.generate()
    grid: sns.PairGrid = data_generator.plot()

    # 1. Correct return type
    assert isinstance(grid, sns.PairGrid)

    # 2. Correct number of diagonal axes (1 per feature)
    assert grid.diag_axes is not None
    assert len(grid.diag_axes) == data_generator.n_features

    # 3. Off-diagonal grid shape
    assert grid.axes.shape == (data_generator.n_features, data_generator.n_features)

    # 4. Check that diagonal axes received vertical lines
    for ax in grid.diag_axes:
        lines = ax.lines
        # 2 vertical lines total: mu_A, mu_B
        assert len(lines) == 2

    # 5. Check sigma shading exists (patches)
    for ax in grid.diag_axes:
        patches = ax.patches
        # two sigma bands: A and B
        assert len(patches) == 2

    # 6. Off-diagonal: means plotted as points
    for row in range(data_generator.n_features):
        for col in range(row):
            ax = grid.axes[row, col]
            line_count: int = len(ax.lines)
            # Expect:
            # - 2 scatter artists for Type A/B
            # - 2 overlay mean markers
            assert line_count == 4

    plt.close(grid.figure)  # avoid figure buildup in pytest runs
