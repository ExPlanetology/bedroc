# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bayesian hierarchical models for quantifying group differences and classification."""

from typing import Literal

DEFAULT_CATEGORY_NAMES: tuple[str, str] = ("Category 0", "Category 1")
"""Default category names"""
DEFAULT_CATEGORY_COLORS: tuple[str, str] = ("tab:blue", "tab:orange")
"""Default category colors"""
FitModel = Literal["covariance", "tempered", "tempered-full", "naive", "two-stage"]
"""Models that can be fitted, each registered in
:obj:`~bedroc.difference.pipelines.MODEL_PIPELINES`. Not restricted to Bayesian models: any fit
following :class:`~bedroc.difference.base.PipelineProtocol` can be added."""
DEFAULT_FIT_MODEL: FitModel = "covariance"
"""Default model to fit"""
