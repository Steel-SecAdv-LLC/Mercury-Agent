# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cross-domain anomaly fusion subpackage for Mercury Agent.

This package is the home of detectors that fuse two or more
``detectors/<domain>/`` outputs into one decision-support assessment.
Single-mode algorithmic detectors (statistical, spectral, dimensional) live
in :mod:`omni_mercury_engine.detectors`; single-domain telemetry detectors
live in :mod:`omni_mercury_engine.detectors.<domain>` (for example
:mod:`omni_mercury_engine.detectors.drone`,
:mod:`omni_mercury_engine.detectors.marine`,
:mod:`omni_mercury_engine.detectors.energy`).

The cross-domain cascades that ship, each exported here and implemented once
in its domain module:

* :class:`EqTsunamiCascadeDetector` -- an earthquake (USGS catalog) screened
  for tsunami potential, confirmed or cleared on sea-level records
  (``detectors.geological.eq_tsunami_cascade``).
* :class:`SolarGICCascadeDetector` -- solar flare and solar wind through the
  geomagnetic response to geomagnetically induced currents on the grid
  (``space.solar_gic_cascade``).
* :class:`SurgeFloodCascade` -- tropical-cyclone storm surge through coastal
  water levels to flooding (``detectors.meteorological.surge_flood_cascade``).

They are imported on first access (PEP 562), so importing this package does
not load torch or the domain detectors.
"""

from __future__ import annotations

import importlib
from typing import Any

#: Exported cascade -> the domain module that implements it.
_CASCADES = {
    "EqTsunamiCascadeDetector": "omni_mercury_engine.detectors.geological.eq_tsunami_cascade",
    "SolarGICCascadeDetector": "omni_mercury_engine.space.solar_gic_cascade",
    "SurgeFloodCascade": "omni_mercury_engine.detectors.meteorological.surge_flood_cascade",
}


def __getattr__(name: str) -> Any:
    """Lazily import a cross-domain cascade on first attribute access (PEP 562)."""
    if name in _CASCADES:
        cascade = getattr(importlib.import_module(_CASCADES[name]), name)
        globals()[name] = cascade
        return cascade
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["EqTsunamiCascadeDetector", "SolarGICCascadeDetector", "SurgeFloodCascade"]
