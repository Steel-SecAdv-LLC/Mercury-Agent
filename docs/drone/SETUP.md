# Drone modules — operator setup guide

Applies to Mercury Agent **v1.6.x and the v1.7 development cycle**. Last updated: 2026-05-19.

Mercury Agent's drone detection stack is **integration-ready, not
pre-integrated**. The platform never ships vendor SDKs, telemetry
endpoints, or live aircraft credentials. This document is written for
the operator who deploys Mercury Agent in their own environment and
supplies their own telemetry source.

> **Decision support only.** The drone anomaly detector is a
> decision-support tool. Acting on its output (recall-to-home,
> emergency landing, payload abort) requires operator authority and a
> validated airworthiness procedure — Mercury never commands a
> vehicle directly.

---

## Contents

- [Architecture](#architecture)
- [Quick start](#quick-start)
- [The DroneState contract](#the-dronestate-contract)
- [Ingest examples](#ingest-examples)
  - [PX4 ULog](#px4-ulog)
  - [MAVLink](#mavlink)
- [Provenance](#provenance)

---

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                Your ingest layer (operator-owned)         │
│   PX4 ULog file  /  MAVLink stream  /  vendor SDK         │
└──────────────────────────────┬────────────────────────────┘
                               │ DroneState objects
                               ▼
┌───────────────────────────────────────────────────────────┐
│           omni_mercury_engine.detectors.drone             │
│                                                           │
│    DroneAnomalyDetector                                   │
│    ├── RADD invariant rules (altitude, battery, GPS, …)   │
│    ├── MercuryAnomalyDetector ensemble                    │
│    │     ├── ResonanceScore (40%) — FFT harmonic OOD      │
│    │     ├── KinematicScore  (30%) — jerk / curvature     │
│    │     └── InfoGeometryScore (30%) — Fisher OOD         │
│    └── DronLomaly-style Bi-LSTM log path (optional)       │
└──────────────────────────────┬────────────────────────────┘
                               │ DroneFault list + scores
                               ▼
                Your action layer (operator-owned)
```

The detector itself is **transport-agnostic**. Mercury does not ship
the ingest layer or the action layer; it consumes `DroneState`
objects and emits typed `DroneFault` records.

## Quick start

```python
import numpy as np
from datetime import datetime, timezone

from omni_mercury_engine.detectors.drone.detector import (
    DroneAnomalyDetector,
    DroneState,
    MissionPhase,
)

detector = DroneAnomalyDetector()

state = DroneState(
    position=np.array([0.0, 0.0, 50.0]),
    velocity=np.array([5.0, 0.0, -0.2]),
    attitude=np.array([0.0, 0.05, 0.0]),
    battery_level=0.72,
    altitude=50.0,
    gps_satellites=11,
    signal_strength=0.88,
    motor_speeds=np.array([7800.0, 7820.0, 7790.0, 7810.0]),
    temperature=42.0,
    mission_phase=MissionPhase.ON_MISSION,
    # The kinematic fields below are derived from ``velocity`` /
    # ``position`` when omitted; supplying them explicitly is preferred
    # when the ingest layer measures them directly.
    altitude_rate=-0.2,
    horizontal_velocity=5.0,
    vertical_velocity=0.2,
    distance_to_home=125.0,
)

faults = detector.detect(state)
for fault in faults:
    print(fault.fault_type, fault.severity, fault.description)
```

## The DroneState contract

`DroneState` is a frozen-shape snapshot of vehicle telemetry. The
`__post_init__` validator rejects mis-shaped inputs at the source
rather than leaking obscure `IndexError`s into the rule engine.

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `position` | `ndarray[float64]` shape `(3,)` | yes | World-frame x/y/z in metres. |
| `velocity` | `ndarray[float64]` shape `(3,)` | yes | World-frame vx/vy/vz in metres/sec. |
| `attitude` | `ndarray[float64]` shape `(3,)` | yes | Euler angles (roll/pitch/yaw) in radians. |
| `battery_level` | `float` ∈ [0, 1] | yes | State of charge. |
| `altitude` | `float` | yes | Above-ground metres. |
| `gps_satellites` | `int` | yes | Sat count locked. |
| `signal_strength` | `float` ∈ [0, 1] | yes | Command-link strength. |
| `motor_speeds` | `ndarray[float64]` shape `(4,)` | yes | Per-motor RPM. Quadrotor only — extend the rules for hex/oct. |
| `temperature` | `float` | yes | Component temperature in °C. |
| `mission_phase` | `MissionPhase` | yes | `INIT`, `TAKEOFF`, `ON_MISSION`, `RETURN`, `LANDING`, `EMERGENCY`. |
| `altitude_rate` | `float \| None` | derived | Climb rate (positive = ascent). Derived from `velocity[2]` when `None`. |
| `horizontal_velocity` | `float \| None` | derived | Horizontal speed magnitude. Derived from `velocity[:2]` when `None`. |
| `vertical_velocity` | `float \| None` | derived | Vertical speed (positive = descent, matches LANDING rule semantics). |
| `distance_to_home` | `float \| None` | derived | Horizontal distance to launch. Derived from `position` and `home_position` when `None`. |
| `home_position` | `ndarray[float64] \| None` | optional | Launch position, used to derive `distance_to_home`. |
| `timestamp` | `datetime` (UTC) | default | Defaults to `datetime.now(UTC)`. |

### Why the four "derived" fields exist

In the upstream Omni-AXA-Engine implementation, `altitude_rate`,
`horizontal_velocity`, `vertical_velocity`, and `distance_to_home`
were referenced by the invariant rules but **not defined on
`DroneState`** — so the rules silently no-op'd. The port adds them as
first-class fields with automatic derivation from `velocity` /
`position` so a rule can no longer fail by missing-attribute lookup.

Supply these fields explicitly when the ingest layer measures them
directly (most PX4 / MAVLink telemetry does); the derived values are
correct but lose any sensor smoothing the autopilot already applied.

---

## Ingest examples

### PX4 ULog

```python
from pyulog import ULog
import numpy as np
from datetime import datetime, timezone

from omni_mercury_engine.detectors.drone.detector import (
    DroneState, MissionPhase,
)

def states_from_ulog(path: str) -> list[DroneState]:
    log = ULog(path, message_name_filter_list=[
        "vehicle_local_position", "vehicle_attitude",
        "battery_status", "vehicle_gps_position",
        "actuator_outputs", "vehicle_status",
    ])
    out: list[DroneState] = []
    # ... iterate aligned timestamps, build one DroneState per sample ...
    return out

states = states_from_ulog("/path/to/log_001.ulg")
faults_per_state = [detector.detect(s) for s in states]
```

`pyulog` is a runtime-optional dependency: install with
`pip install pyulog` (not vendored by Mercury). The `vehicle_status`
field maps to `MissionPhase`; the recommended PX4 nav-state mapping
(`commander/state_machine_helper.cpp`) is:

| PX4 nav state | `MissionPhase` |
|---------------|---------------|
| `MANUAL`, `STAB`, `ACRO`, `POSCTL`, `ALTCTL`, `OFFBOARD` (pre-arm) | `INIT` |
| `AUTO_TAKEOFF` | `TAKEOFF` |
| `AUTO_MISSION`, `AUTO_LOITER`, `AUTO_FOLLOW_TARGET`, `OFFBOARD` (armed) | `ON_MISSION` |
| `AUTO_RTL`, `AUTO_RTGS` | `RETURN` |
| `AUTO_LAND`, `AUTO_PRECLAND` | `LANDING` |
| `TERMINATION`, `AUTO_LANDENGFAIL`, `AUTO_LANDGPSFAIL` | `EMERGENCY` |

`AUTO_LOITER` collapses into `ON_MISSION` rather than a dedicated
hover state because the Mercury `MissionPhase` enum intentionally
does not split loiter from mission cruise — the rule engine treats
them identically. If your operational protocol requires
distinguishing them, encode the discriminator in your ingest layer's
`extra` metadata rather than extending `MissionPhase` here.

### MAVLink

```python
from pymavlink import mavutil
import numpy as np

connection = mavutil.mavlink_connection("udpin:0.0.0.0:14550")

while True:
    msg = connection.recv_match(blocking=True)
    if msg is None:
        continue
    if msg.get_type() == "GLOBAL_POSITION_INT":
        # build / update a DroneState; call detector.detect() at the
        # cadence that matches your operational protocol (typically
        # 1–10 Hz)
        ...
```

`pymavlink` is a runtime-optional dependency. The detector is happy
to be called at any cadence; the rule engine is stateless per call,
and the Mercury ensemble can be re-`fit()` on a sliding window if
your telemetry shifts characteristics over a mission.

---

## Provenance

Ported from the verified Omni-AXA-Engine `drone_anomaly_detector.py`.
The port corrects three defects from the original implementation:

1. **Missing `DroneState` fields.** Added `altitude_rate`,
   `horizontal_velocity`, `vertical_velocity`, and
   `distance_to_home`; the invariant rules referenced these fields,
   but the upstream dataclass did not define them, so the rules
   silently no-op'd.
2. **Ensemble rebuild.** Replaced the hand-coded z-score "K-Means /
   DBSCAN / OPTICS / LOF / OCSVM" ensemble with Mercury Agent's
   first-party `MercuryAnomalyDetector` — three deterministic
   `numpy`/`scipy` scorers (Resonance 40%, Kinematic 30%,
   InfoGeometry 30%). The drone detector therefore carries **no
   scikit-learn runtime dependency**; sklearn lives in the
   `benchmark-comparison` extra only.
3. **Removed unvalidated benchmark claim.** The upstream docstring
   carried an unsourced "93.84% average recall" paper-citation
   claim; no reproduction dataset existed in either tree, so the
   claim was removed. Any future quantitative claim must be backed
   by a reproducible benchmark in `benchmarks/`.

Live telemetry adapters (PX4 ULog, MAVLink, vendor SDK) are
intentionally **not** shipped — adopters supply their own. The
detector is transport-agnostic; the contract is the `DroneState`
dataclass.

---

## See also

- [`docs/API_REFERENCE.md`](../API_REFERENCE.md) — quick-import index.
- `tests/test_drone_detector.py` — 16+ tests covering the rule engine
  and ensemble integration.
- `RADD: Rule-based Anomaly Detection for Drones` — academic reference
  for the invariant-rule approach.
- `DronLomaly: Drone Log Anomaly Detection with Bi-LSTM` — academic
  reference for the optional log-based path.
