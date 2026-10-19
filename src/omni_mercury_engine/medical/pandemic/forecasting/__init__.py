# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pandemic Forecasting Module (Medical Interdiction).

SEIR forecasting with R0 = β/γ, an epidemic-threshold test and R estimated
from observed daily incidence.

Part of Medical Interdiction and Intervention framework.
"""

from __future__ import annotations

from omni_mercury_engine.medical.pandemic.forecasting.epidemic_model import (
    EpidemicForecaster,
    PandemicForecast,
)

__all__ = ["EpidemicForecaster", "PandemicForecast"]
