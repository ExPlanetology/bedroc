# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bayesian hierarchical models for quantifying group differences and classification."""

from typing import Literal

DEFAULT_CATEGORY_NAMES: tuple[str, str] = ("Category 0", "Category 1")
"""Default category names"""
DEFAULT_CATEGORY_COLORS: tuple[str, str] = ("tab:blue", "tab:orange")
"""Default category colors"""
InferenceModel = Literal["covariance", "tempered", "tempered-full", "two-stage"]
"""Inference models"""
DEFAULT_INFERENCE_MODEL: InferenceModel = "covariance"
"""Default inference model"""
