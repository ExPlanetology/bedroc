# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the separability summary on the PCA biplot
(bedroc.core.data_container.DataDiagnostics.separability_table/pca_biplot)"""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from bedroc.core.data_container import DataContainer


def _container(shifts: list[np.ndarray], n: int = 400) -> DataContainer:
    """Categories drawn from Gaussians with a shared, anisotropic covariance, one per shift."""
    rng = np.random.default_rng(0)
    covariance = np.diag([4.0, 1.0, 0.25])
    values = np.vstack([rng.multivariate_normal(shift, covariance, n) for shift in shifts])
    labels = np.repeat([f"c{i}" for i in range(len(shifts))], n)
    return DataContainer(
        pd.DataFrame(values, columns=["f0", "f1", "f2"]),
        metadata=pd.DataFrame({"Type": labels}),
        category_column="Type",
        name="test",
    )


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_separability_table_totals_match_mahalanobis_distance() -> None:
    data = _container([np.zeros(3), np.array([2.0, 0.0, 0.5])])
    alignment = data.diagnostics.mahalanobis_alignment()
    total = float(alignment.loc["mahalanobis_sq contribution"].to_numpy().sum())

    table = data.diagnostics.separability_table()

    # One row per component, all of them
    assert all(f"PC{k}" in table for k in (1, 2, 3))
    match = re.search(r"D² = ([0-9.]+), D = ([0-9.]+)", table)
    assert match is not None
    assert float(match.group(1)) == pytest.approx(total, abs=0.005)
    overlap = re.search(r"OVL = 2Φ\(−D/2\) = ([0-9.]+)", table)
    assert overlap is not None
    assert float(overlap.group(1)) == pytest.approx(2 * norm.cdf(-np.sqrt(total) / 2), abs=0.005)


def test_pca_biplot_shows_effect_sizes_and_table() -> None:
    data = _container([np.zeros(3), np.array([2.0, 0.0, 0.5])])

    ax = data.diagnostics.pca_biplot()

    assert "d² =" in ax.get_xlabel()
    assert "d² =" in ax.get_ylabel()
    texts = [text.get_text() for text in ax.texts]
    assert data.diagnostics.separability_table() in texts
    assert any(re.search(r"D² = [0-9.]+ in PC1\+PC2", text) for text in texts)


def test_pca_biplot_with_three_categories() -> None:
    data = _container([np.zeros(3), np.array([2.0, 0.0, 0.5]), np.array([0.0, 1.0, 0.0])])

    ax = data.diagnostics.pca_biplot()

    # A single d² per axis is ambiguous with several shifts, so it is left to the table
    assert "d²" not in ax.get_xlabel()
    table = data.diagnostics.separability_table()
    assert "c1: D² =" in table
    assert "c2: D² =" in table
