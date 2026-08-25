# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bayesian hierarchical models for quantifying group differences and classification."""

from typing import Literal

DEFAULT_GROUP_NAMES: tuple[str, str] = ("Group 0", "Group 1")
"""Default group names"""
InferenceModel = Literal["covariance", "tempered", "two-stage"]
"""Inference models"""
DEFAULT_INFERENCE_MODEL: InferenceModel = "covariance"
"""Default inference model"""
