"""Drone single-domain anomaly detector subpackage.

Hosts the RADD + sklearn-ensemble + DronLomaly drone anomaly detector,
co-located with Mercury's other per-domain detector subpackages
(``marine``, ``economic``, ``energy``, ``geological``, …).

Cross-domain fusion detectors that consume drone faults *together with*
another single-domain stream live in
:mod:`omni_mercury_engine.anomaly`, not here.
"""

# Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.

from __future__ import annotations

from omni_mercury_engine.detectors.drone.detector import (
    DroneAnomalyDetector,
    DroneFault,
    DroneState,
    FaultType,
    MissionPhase,
    get_drone_detector,
)

__all__ = [
    "DroneAnomalyDetector",
    "DroneFault",
    "DroneState",
    "FaultType",
    "MissionPhase",
    "get_drone_detector",
]
