# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""GUI Module for Mercury Agent.

Interactive Plotly dashboards (the ``[gui]`` extra) over detection scores and
hazard-detector diagnostics, exported as self-contained HTML -- see
``mercury-agent hazard-viz --format html``.
"""

from __future__ import annotations

from omni_mercury_engine.gui.visualization_dashboard import (
    PLOTLY_AVAILABLE,
    AnomalyVisualizer,
    DashboardBuilder,
    DashboardConfig,
    HazardDiagnosticsVisualizer,
    create_quick_dashboard,
)

__all__ = [
    "PLOTLY_AVAILABLE",
    "AnomalyVisualizer",
    "DashboardBuilder",
    "DashboardConfig",
    "HazardDiagnosticsVisualizer",
    "create_quick_dashboard",
]
