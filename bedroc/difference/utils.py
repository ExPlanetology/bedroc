# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Difference utils"""

import numpy as np
from scipy.integrate import simpson
from scipy.stats import gaussian_kde

from bedroc.core.type_aliases import NpArray


def distribution_overlap_data(
    values_0: NpArray, values_1: NpArray, *, n_grid: int = 2000
) -> tuple[NpArray, NpArray, NpArray, NpArray, float]:
    """Calculates KDEs and overlap data for two 1D distributions.

    The probability density functions are estimated using Gaussian kernel density estimation (KDE).
    The overlap coefficient is then calculated as the integral of the pointwise minimum of the two
    estimated probability density functions.

    For multimodal or strongly irregular distributions, the estimated overlap may be sensitive to
    the KDE bandwidth.

    Args:
        values_0: Samples from the first distribution.
        values_1: Samples from the second distribution.
        n_grid: Number of points to use for the grid over which to evaluate the PDFs. Defaults to
            ``2000``.

    Returns:
        Tuple containing the evaluation grid, first PDF, second PDF, overlap density, and overlap
        coefficient.
    """
    values_0 = np.asarray(values_0, dtype=float)
    values_1 = np.asarray(values_1, dtype=float)

    values_0 = values_0[np.isfinite(values_0)]
    values_1 = values_1[np.isfinite(values_1)]

    if len(values_0) < 2 or len(values_1) < 2:
        raise ValueError("Both populations require at least two finite observations.")

    lower = min(values_0.min(), values_1.min())
    upper = max(values_0.max(), values_1.max())

    x: NpArray = np.linspace(lower, upper, n_grid)

    pdf_0 = gaussian_kde(values_0)(x)
    pdf_1 = gaussian_kde(values_1)(x)

    overlap_density = np.minimum(pdf_0, pdf_1)

    overlap = simpson(overlap_density, x=x)

    return x, pdf_0, pdf_1, overlap_density, float(overlap)
