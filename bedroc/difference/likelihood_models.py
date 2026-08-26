# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Likelihood models for Bayesian hierarchical group-difference models.

Provides alternative probability distributions for modeling observations around group-specific
feature means.
"""

import logging

logger: logging.Logger = logging.getLogger(__name__)
