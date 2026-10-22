# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adapters serving domain detectors through the engine and registry seams.

Each adapter wraps one domain detector over its true input (named rows, a
time-ordered window, a record, an event sequence or text) and maps it
one-to-one onto that detector's own entry point; see
:mod:`omni_mercury_engine.detectors.hazard_adapter` for the contracts.
"""

from omni_mercury_engine.detectors.adapters.earth import (
    AvalancheProfileAdapter,
    EqTsunamiCascadeAdapter,
    FireDebrisFlowAdapter,
    LandslideAdapter,
    MarineEcosystemAdapter,
    RockfallAdapter,
    SubsidenceAdapter,
    SurgeFloodCascadeAdapter,
    VolcanicAdapter,
    WildfireSceneAdapter,
)
from omni_mercury_engine.detectors.adapters.society import (
    EMPRecordAdapter,
    FinancialCrisisRecordAdapter,
    ThreatPayloadAdapter,
)
from omni_mercury_engine.detectors.adapters.space import (
    CMEArrivalAdapter,
    GICWindowAdapter,
    ScintillationAdapter,
    SEPProtonWindowAdapter,
    SolarGICCascadeAdapter,
    SolarStormSnapshotAdapter,
)

__all__ = [
    "AvalancheProfileAdapter",
    "CMEArrivalAdapter",
    "EMPRecordAdapter",
    "EqTsunamiCascadeAdapter",
    "FinancialCrisisRecordAdapter",
    "FireDebrisFlowAdapter",
    "GICWindowAdapter",
    "LandslideAdapter",
    "MarineEcosystemAdapter",
    "RockfallAdapter",
    "SEPProtonWindowAdapter",
    "ScintillationAdapter",
    "SolarGICCascadeAdapter",
    "SolarStormSnapshotAdapter",
    "SubsidenceAdapter",
    "SurgeFloodCascadeAdapter",
    "ThreatPayloadAdapter",
    "VolcanicAdapter",
    "WildfireSceneAdapter",
]
