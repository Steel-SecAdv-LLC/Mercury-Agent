# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Deterministic scenario-set builder for the hazard regression guard.

Two kinds of committed scenario sets, transparently labelled in the manifest:

**Real recorded data** (``label_source: "measured"``) -- the solar domain.
Raw NOAA SWPC snapshots (planetary Kp, propagated real-time solar wind, GOES
primary X-ray flux, GOES flare-event list) are processed once into small
window files with full provenance (source URL, fetch timestamp, raw-payload
sha256, row counts, flare event evidence). Nothing is interpolated or
fabricated: windows without sufficient real coverage are dropped. Flare-class
labels are *definitional* -- the NOAA class of a window IS a threshold on its
measured peak long-channel flux -- so that set gates the classification
chain's correctness; the Kp set is genuine forecast skill (Boyle-index
coupling from measured solar wind vs the independently measured ground-based
Kp index).

**Constructed physics scenarios** (``label_source: "constructed"``) -- the
domains whose detectors consume raw sensor series (Doppler radar velocity
fields, seismic traces, DART sea-level records, gridded wind fields,
gauge/precip observations, multi-parameter volcano monitoring) for which no
allow-listed feed provides labelled raw series. Each scenario is built with
a fixed seed (NumPy ``default_rng`` -- stream-stable across versions by
NumPy's RNG policy) against the detector's DOCUMENTED input contract,
mirroring the transparency-test fixtures (``tests/detectors/test_*_honesty.py``).
Labels are the physical ground truth of the constructed situation (e.g. a
25 m/s Doppler velocity couplet IS a mesocyclone signature; a flat noise
field is not), never the detector's own output.

Usage::

    python benchmarks/hazard_scenarios/generate_scenarios.py --constructed
    python benchmarks/hazard_scenarios/generate_scenarios.py \
        --solar-from-raw /path/to/swpc/snapshots --fetched-at 2026-07-09T05:45:00Z

Regenerating any file changes its manifest hash, and the guard baseline
(``benchmarks/hazard_domain_baseline.json``) must then be re-pinned via
``python benchmarks/hazard_regression_guard.py --update`` and reviewed.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from scenario_io import (
    MANIFEST_PATH,
    SCENARIO_DIR,
    hash_scenario_file,
    load_manifest,
    sha256_file,
    write_canonical_json,
)

# Fixed per-domain seeds: one seed per constructed set so a change to one
# generator cannot silently shift another set's stream.
SEEDS = {
    "tornado": 4207,
    "earthquake": 4208,
    "tsunami": 4209,
    "hurricane": 4210,
    "volcano": 4211,
}

SWPC_BASE = "https://services.swpc.noaa.gov"
SWPC_SOURCES = {
    "kp": f"{SWPC_BASE}/products/noaa-planetary-k-index.json",
    "wind": f"{SWPC_BASE}/products/geospace/propagated-solar-wind.json",
    "xray": f"{SWPC_BASE}/json/goes/primary/xrays-7-day.json",
    "flares": f"{SWPC_BASE}/json/goes/primary/xray-flares-7-day.json",
}

# NOAA GOES X-ray flare classification thresholds (W/m^2, long channel).
FLARE_THRESHOLDS = (("X", 1e-4), ("M", 1e-5), ("C", 1e-6), ("B", 1e-7))


def _flare_class(flux_wm2: float) -> str:
    """NOAA flare class of a long-channel X-ray flux (definitional mapping).

    Args:
        flux_wm2: GOES 0.1-0.8 nm flux in W/m^2.

    Returns:
        One of ``"A"``, ``"B"``, ``"C"``, ``"M"``, ``"X"``.
    """
    for label, threshold in FLARE_THRESHOLDS:
        if flux_wm2 >= threshold:
            return label
    return "A"


# ---------------------------------------------------------------------------
# Constructed sets (seeded physics scenarios, detector input contracts)
# ---------------------------------------------------------------------------


def build_tornado() -> dict[str, Any]:
    """Doppler-radar mesocyclone scenarios for ``TornadoDetector``.

    Input contract (see ``predict_tornado`` + transparency tests): a radar
    velocity sequence ``[frames, gates]`` in m/s. A mesocyclone appears as an
    inbound/outbound couplet whose rotational velocity ``(Vmax - Vmin)/2``
    exceeds ~15 m/s (WSR-88D operational threshold).

    Events: couplet onsets at frame ``onset`` and ramps to full strength over
    6 frames; the labelled touchdown is ``onset + 15`` (frames = minutes).
    Nulls: calm noise, strong turbulence, uniform (non-rotational) flow, and
    sub-threshold couplets -- all physically mesocyclone-free.

    Returns:
        Manifest entry for the written NPZ.
    """
    rng = np.random.default_rng(SEEDS["tornado"])
    n_frames, gates, sigma = 48, 64, 2.0
    frames, labels, onsets, events, kinds = [], [], [], [], []

    for i in range(12):  # events
        onset = 10 + (i % 8)
        v_rot = 18.0 + 1.5 * i
        seq = rng.normal(0.0, sigma, (n_frames, gates))
        for f in range(onset, n_frames):
            strength = v_rot * min(1.0, (f - onset + 1) / 6.0)
            seq[f] += np.linspace(-strength, strength, gates)
        frames.append(seq)
        labels.append(1)
        onsets.append(onset)
        events.append(onset + 15)
        kinds.append(f"mesocyclone_v{v_rot:.1f}")

    null_specs = (
        [("calm", 0.0, sigma)] * 4
        + [("turbulence", 0.0, 5.0)] * 4
        + [("uniform_flow", 15.0, 3.0)] * 2
    )
    for kind, mean, s in null_specs:
        frames.append(rng.normal(mean, s, (n_frames, gates)))
        labels.append(0)
        onsets.append(-1)
        events.append(-1)
        kinds.append(kind)
    for v_sub in (7.0, 9.0):  # rotation present but physically sub-mesocyclone
        seq = rng.normal(0.0, sigma, (n_frames, gates))
        seq += np.linspace(-v_sub, v_sub, gates)[None, :]
        frames.append(seq)
        labels.append(0)
        onsets.append(-1)
        events.append(-1)
        kinds.append(f"subthreshold_couplet_v{v_sub:.0f}")

    path = SCENARIO_DIR / "tornado_scenarios.npz"
    np.savez(
        path,
        frames=np.array(frames),
        labels=np.array(labels, dtype=np.int64),
        onset_frame=np.array(onsets, dtype=np.int64),
        event_frame=np.array(events, dtype=np.int64),
        kinds=np.array(kinds),
    )
    return {
        "sha256": hash_scenario_file(path),
        "hash_policy": "npz_content",
        "label_source": "constructed",
        "seed": SEEDS["tornado"],
        "n_scenarios": len(labels),
        "n_events": int(sum(labels)),
        "construction": (
            "48-frame x 64-gate Doppler velocity sequences (m/s), 1 frame/min. "
            "Events: inbound/outbound couplet ramping to V_rot 18-34.5 m/s over 6 "
            "frames from a known onset; touchdown labelled onset+15 min. Nulls: "
            "calm (sigma 2), turbulence (sigma 5), uniform 15 m/s flow, and "
            "sub-threshold couplets (V_rot 7/9 m/s). Mirrors "
            "tests/detectors/test_meteorological_honesty.py fixtures."
        ),
    }


def build_earthquake() -> dict[str, Any]:
    """Single-station seismic traces for ``EarthquakeDetector``.

    Input contract: a raw seismic trace at 100 Hz. Events carry an impulsive
    P burst (short coda) followed by a stronger S coda at a known S-P delay;
    the true epicentral distance follows the standard single-station rule
    ``d = dt_SP * Vp*Vs/(Vp-Vs)`` with the detector's documented velocities
    (6.0 / 3.5 km/s => 8.4 km/s per second of S-P time). Nulls are quiet
    noise, slowly modulated noise, and a steady harmonic hum -- none has an
    impulsive arrival.

    Returns:
        Manifest entry for the written NPZ.
    """
    rng = np.random.default_rng(SEEDS["earthquake"])
    n, fs, sigma = 6000, 100.0, 0.1
    sp_rule_km_per_s = (6.0 * 3.5) / (6.0 - 3.5)  # 8.4
    traces, labels, p_onsets, s_onsets, distances, kinds = [], [], [], [], [], []

    for i in range(14):  # events
        p0 = 1200 + 220 * i
        sp_sec = 2.0 + 0.35 * i
        s0 = p0 + round(sp_sec * fs)
        amp_p = 1.2 + 0.2 * (i % 5)
        trace = rng.normal(0.0, sigma, n)
        p_env = amp_p * np.exp(-np.arange(120) / 40.0)
        trace[p0 : p0 + 120] += p_env * rng.normal(0.0, 1.0, 120)
        s_env = 2.0 * amp_p * np.exp(-np.arange(600) / 150.0)
        trace[s0 : s0 + 600] += s_env * rng.normal(0.0, 1.0, 600)
        traces.append(trace)
        labels.append(1)
        p_onsets.append(p0)
        s_onsets.append(s0)
        distances.append(sp_sec * sp_rule_km_per_s)
        kinds.append(f"local_event_sp{sp_sec:.2f}s")

    for kind_idx in range(10):  # nulls
        if kind_idx < 4:
            trace = rng.normal(0.0, sigma, n)
            kind = "quiet"
        elif kind_idx < 7:
            envelope = np.linspace(sigma, 3.0 * sigma, n)
            trace = rng.normal(0.0, 1.0, n) * envelope
            kind = "slow_modulation"
        else:
            t = np.arange(n) / fs
            trace = rng.normal(0.0, sigma, n) + 0.15 * np.sin(2 * np.pi * 2.0 * t)
            kind = "harmonic_hum"
        traces.append(trace)
        labels.append(0)
        p_onsets.append(-1)
        s_onsets.append(-1)
        distances.append(-1.0)
        kinds.append(kind)

    path = SCENARIO_DIR / "earthquake_scenarios.npz"
    np.savez(
        path,
        traces=np.array(traces),
        labels=np.array(labels, dtype=np.int64),
        p_onset=np.array(p_onsets, dtype=np.int64),
        s_onset=np.array(s_onsets, dtype=np.int64),
        sp_distance_km=np.array(distances),
        kinds=np.array(kinds),
    )
    return {
        "sha256": hash_scenario_file(path),
        "hash_policy": "npz_content",
        "label_source": "constructed",
        "seed": SEEDS["earthquake"],
        "n_scenarios": len(labels),
        "n_events": int(sum(labels)),
        "construction": (
            "60 s single-station traces at 100 Hz, background sigma 0.1. Events: "
            "impulsive P burst (12-20x background, 1.2 s exponential coda) then S "
            "coda at 2x P amplitude, S-P delay 2.0-6.55 s; true distance = "
            "dt_SP * 8.4 km/s (detector's documented Vp=6.0/Vs=3.5). Nulls: quiet "
            "noise, slow amplitude modulation, steady 2 Hz hum. Mirrors "
            "tests/detectors/test_geophysical_honesty.py fixtures."
        ),
    }


def build_tsunami() -> dict[str, Any]:
    """DART-style sea-level records for ``TsunamiDetector``.

    Input contract: a 1 Hz sea-level/pressure record. Events are long-period
    (0.005 Hz) waves of known amplitude arriving mid-record over a 5 cm noise
    floor; the 1.2 m case is deliberately borderline for the detector's 0.96
    confidence threshold and is documented as such. Nulls: quiet sea, wind
    swell (0.08 Hz -- outside the tsunami band), and a tidal ramp.

    Returns:
        Manifest entry for the written NPZ.
    """
    rng = np.random.default_rng(SEEDS["tsunami"])
    n = 4096
    t = np.arange(n)
    records, labels, amplitudes, kinds = [], [], [], []

    for amp in (1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.2, 3.5):
        rec = rng.normal(0.0, 0.05, n) + amp * np.sin(2 * np.pi * 0.005 * t) * (t > 2000)
        records.append(rec)
        labels.append(1)
        amplitudes.append(amp)
        kinds.append(f"long_period_wave_{amp:.1f}m")

    for kind_idx in range(10):
        if kind_idx < 4:
            rec = rng.normal(0.0, 0.05, n)
            kind = "quiet_sea"
        elif kind_idx < 7:
            rec = rng.normal(0.0, 0.05, n) + 0.3 * np.sin(2 * np.pi * 0.08 * t)
            kind = "wind_swell"
        else:
            rec = rng.normal(0.0, 0.05, n) + np.linspace(0.0, 0.5, n)
            kind = "tidal_ramp"
        records.append(rec)
        labels.append(0)
        amplitudes.append(0.0)
        kinds.append(kind)

    path = SCENARIO_DIR / "tsunami_scenarios.npz"
    np.savez(
        path,
        records=np.array(records),
        labels=np.array(labels, dtype=np.int64),
        amplitude_m=np.array(amplitudes),
        kinds=np.array(kinds),
    )
    return {
        "sha256": hash_scenario_file(path),
        "hash_policy": "npz_content",
        "label_source": "constructed",
        "seed": SEEDS["tsunami"],
        "n_scenarios": len(labels),
        "n_events": int(sum(labels)),
        "construction": (
            "4096 s sea-level records at 1 Hz, 5 cm noise floor. Events: 0.005 Hz "
            "long-period waves of 1.2-3.5 m arriving at t>2000 s (1.2 m is a "
            "documented borderline case for the 0.96 detection threshold). Nulls: "
            "quiet sea, 0.08 Hz wind swell (outside the tsunami band), 0.5 m tidal "
            "ramp. Mirrors tests/detectors/test_geophysical_honesty.py fixtures."
        ),
    }


def build_hurricane() -> dict[str, Any]:
    """Gridded wind fields + central pressure for ``HurricaneDetector``.

    Input contract: ``wind_field`` with 2-D u/v components (m/s) and grid
    spacing, plus ``pressure_data`` (central/environmental pressure, mb).
    Events are Rankine vortices (solid-body core, 1/r decay) whose core
    relative vorticity ``zeta = 2*Vmax/rc`` exceeds the detector's documented
    2e-3 s^-1 closed-circulation criterion, with a Dvorak-consistent pressure
    deficit. Nulls: uniform flow, linear shear, a weak broad low, and
    disorganised noise -- no closed circulation, deficits < 10 mb.

    Returns:
        Manifest entry for the written NPZ.
    """
    rng = np.random.default_rng(SEEDS["hurricane"])
    n_grid, spacing = 50, 4000.0
    half = (n_grid - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(n_grid) - half, np.arange(n_grid) - half, indexing="ij")
    x_m, y_m = xx * spacing, yy * spacing
    r = np.hypot(x_m, y_m)
    theta = np.arctan2(y_m, x_m)

    us, vs, centrals, envs, labels, kinds = [], [], [], [], [], []

    for i in range(10):  # events
        v_max = 30.0 + 4.0 * i
        rc = (12.0 + 4.0 * (i % 4)) * 1000.0
        v_theta = np.where(r <= rc, v_max * r / rc, v_max * rc / np.maximum(r, 1.0))
        u = -v_theta * np.sin(theta) + rng.normal(0.0, 0.5, r.shape)
        v = v_theta * np.cos(theta) + rng.normal(0.0, 0.5, r.shape)
        v_max_kt = v_max * 1.9438
        deficit = (v_max_kt / 6.7) ** (1.0 / 0.644)  # inverse Dvorak wind-pressure
        us.append(u)
        vs.append(v)
        centrals.append(1013.0 - deficit)
        envs.append(1013.0)
        labels.append(1)
        kinds.append(f"rankine_vortex_v{v_max:.0f}_rc{rc / 1000:.0f}km")

    null_specs = (
        [("uniform_flow", 2.0)] * 3
        + [("linear_shear", 4.0)] * 3
        + [("weak_broad_low", 6.0)] * 2
        + [("disorganised_noise", 0.0)] * 2
    )
    for kind, deficit in null_specs:
        if kind == "uniform_flow":
            u = np.full(r.shape, 12.0) + rng.normal(0.0, 0.5, r.shape)
            v = rng.normal(0.0, 0.5, r.shape)
        elif kind == "linear_shear":
            u = 5.0 + 10.0 * (yy + half) / (n_grid - 1) + rng.normal(0.0, 0.5, r.shape)
            v = rng.normal(0.0, 0.5, r.shape)
        elif kind == "weak_broad_low":
            rc = 60_000.0
            v_theta = np.where(r <= rc, 8.0 * r / rc, 8.0 * rc / np.maximum(r, 1.0))
            u = -v_theta * np.sin(theta) + rng.normal(0.0, 0.5, r.shape)
            v = v_theta * np.cos(theta) + rng.normal(0.0, 0.5, r.shape)
        else:
            # sigma 1.0 keeps the pointwise max noise vorticity (~1.3e-3 s^-1)
            # below the 2e-3 closed-circulation criterion: disorganised noise
            # is physically circulation-free and must stay so numerically.
            u = rng.normal(0.0, 1.0, r.shape)
            v = rng.normal(0.0, 1.0, r.shape)
        us.append(u)
        vs.append(v)
        centrals.append(1013.0 - deficit)
        envs.append(1013.0)
        labels.append(0)
        kinds.append(kind)

    path = SCENARIO_DIR / "hurricane_scenarios.npz"
    np.savez(
        path,
        u=np.array(us),
        v=np.array(vs),
        grid_spacing_m=np.array(spacing),
        central_pressure_mb=np.array(centrals),
        environmental_pressure_mb=np.array(envs),
        labels=np.array(labels, dtype=np.int64),
        kinds=np.array(kinds),
    )
    return {
        "sha256": hash_scenario_file(path),
        "hash_policy": "npz_content",
        "label_source": "constructed",
        "seed": SEEDS["hurricane"],
        "n_scenarios": len(labels),
        "n_events": int(sum(labels)),
        "construction": (
            "50x50 wind grids at 4 km spacing. Events: Rankine vortices, Vmax "
            "30-66 m/s, core radius 12-24 km (core zeta = 2*Vmax/rc >= 2.5e-3 "
            "s^-1), central pressure from the inverse Dvorak wind-pressure "
            "relationship. Nulls: uniform 12 m/s flow, linear shear (zeta ~5e-5), "
            "weak broad low (Vmax 8 m/s, zeta ~2.7e-4), disorganised noise; "
            "deficits <= 6 mb. Mirrors "
            "tests/detectors/test_meteorological_honesty.py fixtures."
        ),
    }


def build_flood() -> dict[str, Any]:
    """Gauge/precipitation/soil scenarios + timed series for ``FloodDetector``.

    Input contract: ``precip_data`` (in/hr and 24 h accumulations),
    ``gauge_data`` (stages in ft vs NWS-style action/flood stages), and
    ``soil_data``. Static scenarios use NWS flash-flood (2 in/hr) and flood
    (4 in/24 h) climatology and gauge exceedance physics; timed series ramp
    rainfall ahead of the river crest so the first transparent alert precedes
    stage exceedance by a known margin. Fully explicit -- no RNG.

    Returns:
        Manifest entry for the written JSON.
    """
    gauge_base = {
        "action_stage_ft": 10.0,
        "flood_stage_ft": 15.0,
        "moderate_flood_stage_ft": 20.0,
        "major_flood_stage_ft": 25.0,
        "record_stage_ft": 30.0,
    }

    def gauge(stage: float, history: list[float]) -> dict[str, Any]:
        return {**gauge_base, "current_stage_ft": stage, "stage_history_ft": history}

    scenarios: list[dict[str, Any]] = []
    # Events: flash-flood rain rates, river exceedance, compound situations.
    for p1, p24 in ((2.4, 5.0), (2.8, 6.0), (3.2, 8.0)):
        scenarios.append(
            {
                "kind": f"flash_flood_{p1}in_hr",
                "label": 1,
                "data": {
                    "precip_data": {
                        "precipitation_1h_inches": p1,
                        "precipitation_24h_inches": p24,
                    },
                    "soil_data": {"soil_moisture_pct": 88.0, "soil_type": "clay_loam"},
                },
            }
        )
    for above in (2.5, 4.0, 6.0):
        stage = gauge_base["flood_stage_ft"] + above
        scenarios.append(
            {
                "kind": f"river_flood_{above}ft_above",
                "label": 1,
                "data": {
                    "precip_data": {
                        "precipitation_1h_inches": 0.4,
                        "precipitation_24h_inches": 2.5,
                    },
                    "gauge_data": gauge(stage, [stage - 3.0, stage - 1.5, stage]),
                },
            }
        )
    scenarios.append(
        {
            "kind": "compound_flash_and_river",
            "label": 1,
            "data": {
                "precip_data": {
                    "precipitation_1h_inches": 2.6,
                    "precipitation_24h_inches": 7.0,
                },
                "gauge_data": gauge(18.0, [14.0, 16.0, 18.0]),
                "soil_data": {"soil_moisture_pct": 92.0, "soil_type": "clay"},
            },
        }
    )
    scenarios.append(
        {
            "kind": "compound_moderate",
            "label": 1,
            "data": {
                "precip_data": {
                    "precipitation_1h_inches": 1.2,
                    "precipitation_24h_inches": 5.5,
                },
                "gauge_data": gauge(16.0, [12.5, 14.5, 16.0]),
            },
        }
    )
    scenarios.append(
        {
            "kind": "river_at_flood_rising_rapidly",
            "label": 1,
            "data": {
                "precip_data": {
                    "precipitation_1h_inches": 0.8,
                    "precipitation_24h_inches": 3.0,
                },
                "gauge_data": gauge(15.5, [13.0, 14.2, 15.5]),
            },
        }
    )
    scenarios.append(
        {
            "kind": "flash_marginal_saturated",
            "label": 1,
            "data": {
                "precip_data": {
                    "precipitation_1h_inches": 2.1,
                    "precipitation_24h_inches": 4.5,
                },
                "soil_data": {"soil_moisture_pct": 95.0, "soil_type": "clay"},
            },
        }
    )
    # Nulls: sub-threshold rain, normal or falling stages, near-misses.
    null_specs: list[tuple[str, dict[str, Any]]] = [
        (
            "light_rain",
            {
                "precip_data": {
                    "precipitation_1h_inches": 0.3,
                    "precipitation_24h_inches": 1.0,
                },
                "gauge_data": gauge(6.0, [5.8, 5.9, 6.0]),
            },
        ),
        (
            "moderate_rain_below_thresholds",
            {
                "precip_data": {
                    "precipitation_1h_inches": 1.2,
                    "precipitation_24h_inches": 3.5,
                },
                "gauge_data": gauge(8.0, [7.8, 7.9, 8.0]),
            },
        ),
        (
            "dry_normal_stage",
            {
                "precip_data": {
                    "precipitation_1h_inches": 0.0,
                    "precipitation_24h_inches": 0.0,
                },
                "gauge_data": gauge(5.0, [5.0, 5.0, 5.0]),
                "soil_data": {"soil_moisture_pct": 25.0, "soil_type": "sand"},
            },
        ),
        (
            "near_threshold_rain_and_stage",
            {
                "precip_data": {
                    "precipitation_1h_inches": 1.8,
                    "precipitation_24h_inches": 3.9,
                },
                "gauge_data": gauge(13.8, [13.4, 13.6, 13.8]),
                "soil_data": {"soil_moisture_pct": 55.0, "soil_type": "loam"},
            },
        ),
        (
            "falling_after_crest",
            {
                "precip_data": {
                    "precipitation_1h_inches": 0.1,
                    "precipitation_24h_inches": 2.0,
                },
                "gauge_data": gauge(14.5, [15.6, 15.0, 14.5]),
            },
        ),
        (
            "action_stage_stable",
            {
                "precip_data": {
                    "precipitation_1h_inches": 0.5,
                    "precipitation_24h_inches": 1.8,
                },
                "gauge_data": gauge(11.0, [10.9, 11.0, 11.0]),
            },
        ),
    ]
    for kind, data in null_specs:
        scenarios.append({"kind": kind, "label": 0, "data": data})
    for moisture, soil in ((45.0, "sandy_loam"), (60.0, "loam")):
        scenarios.append(
            {
                "kind": f"damp_soil_only_{int(moisture)}pct",
                "label": 0,
                "data": {
                    "precip_data": {
                        "precipitation_1h_inches": 0.6,
                        "precipitation_24h_inches": 1.5,
                    },
                    "soil_data": {"soil_moisture_pct": moisture, "soil_type": soil},
                },
            }
        )
    for stage in (7.0, 9.0):
        scenarios.append(
            {
                "kind": f"quiet_gauge_{int(stage)}ft",
                "label": 0,
                "data": {"gauge_data": gauge(stage, [stage, stage, stage])},
            }
        )

    # Timed series: rainfall ramps ahead of the river crest, so a transparent
    # detector alerts (flash-flood rain rate) before stage exceedance.
    series = []
    for idx, (ramp_start, ramp_rate, crest_hour) in enumerate(
        [(2, 0.30, 15), (3, 0.28, 16), (2, 0.35, 14), (4, 0.25, 18), (3, 0.32, 17), (2, 0.26, 19)]
    ):
        base_stage = 8.0
        flood_stage = gauge_base["flood_stage_ft"]
        rise_start = 6
        rate = (flood_stage - base_stage) / (crest_hour - rise_start)
        steps = []
        cumulative = 0.0
        stages: list[float] = []
        for hour in range(24):
            p1 = min(3.2, max(0.2, (hour - ramp_start) * ramp_rate))
            cumulative += p1
            stage = base_stage + max(0.0, hour - rise_start) * rate
            stages.append(round(stage, 3))
            history = stages[-4:]
            steps.append(
                {
                    "hour": hour,
                    "precip_data": {
                        "precipitation_1h_inches": round(p1, 3),
                        "precipitation_24h_inches": round(min(cumulative, 12.0), 3),
                    },
                    "gauge_data": gauge(round(stage, 3), history),
                }
            )
        event_hour = next(h for h, s in enumerate(stages) if s >= flood_stage)
        series.append({"kind": f"rain_then_crest_{idx}", "event_hour": event_hour, "steps": steps})

    path = SCENARIO_DIR / "flood_scenarios.json"
    write_canonical_json(path, {"scenarios": scenarios, "series": series})
    return {
        "sha256": hash_scenario_file(path),
        "hash_policy": "file_bytes",
        "label_source": "constructed",
        "seed": None,
        "n_scenarios": len(scenarios),
        "n_events": sum(s["label"] for s in scenarios),
        "n_series": len(series),
        "construction": (
            "Static scenarios against NWS flash-flood (2 in/hr) and 24 h flood "
            "(4 in) climatology plus gauge stage exceedance; nulls include "
            "near-threshold rain/stage and a falling post-crest river. Timed "
            "series ramp rainfall ahead of a river crest (known crest hour) to "
            "measure warning lead time. Fully explicit; no RNG."
        ),
    }


def build_volcano() -> dict[str, Any]:
    """Multi-parameter unrest scenarios for ``VolcanicEruptionDetector``.

    Input contract (see ``predict_eruption`` + transparency tests): seismic
    sequence, gas fluxes (SO2/CO2 t/d vs 100/500 baselines), InSAR
    displacement (cm), and thermal brightness temperatures (K). Four
    scenarios per intended USGS alert level, constructed from the number and
    strength of genuinely present precursors: quiet -> ``normal``; a single
    modest seismic swarm -> ``advisory``; strong swarm + degassing ->
    ``watch``; swarm + degassing + deformation + thermal anomaly ->
    ``warning``.

    Returns:
        Manifest entry for the written JSON.
    """
    rng = np.random.default_rng(SEEDS["volcano"])
    scenarios: list[dict[str, Any]] = []

    def seismic(n_spikes: int, spike_amp: float) -> list[float]:
        seq = rng.normal(0.0, 1.0, 200)
        if n_spikes:
            positions = rng.choice(200, size=n_spikes, replace=False)
            seq[positions] = spike_amp
        return [round(float(x), 6) for x in seq]

    quiet_gas = {"so2_tons_per_day": 100.0, "co2_tons_per_day": 500.0}
    quiet_insar = {"vertical_displacement_cm": 0.5, "deformation_rate_cm_day": 0.0}
    quiet_thermal = {"brightness_temperature_k": [293.0, 295.0, 296.0, 298.0]}

    for i in range(4):  # normal: no genuine precursor
        scenarios.append(
            {
                "kind": f"quiescent_{i}",
                "alert_level": "normal",
                "data": {
                    "seismic_sequence": seismic(0, 0.0),
                    "gas_data": dict(quiet_gas),
                    "insar_data": dict(quiet_insar),
                    "thermal_data": dict(quiet_thermal),
                },
            }
        )
    for i, spikes in enumerate((9, 10, 10, 11)):  # advisory: one modest swarm
        scenarios.append(
            {
                "kind": f"modest_swarm_{i}",
                "alert_level": "advisory",
                "data": {
                    "seismic_sequence": seismic(spikes, 12.0),
                    "gas_data": dict(quiet_gas),
                    "insar_data": dict(quiet_insar),
                    "thermal_data": dict(quiet_thermal),
                },
            }
        )
    for i, (so2, co2) in enumerate(
        ((400.0, 1200.0), (450.0, 1250.0), (500.0, 1400.0), (420.0, 1300.0))
    ):
        scenarios.append(
            {
                "kind": f"swarm_plus_degassing_{i}",
                "alert_level": "watch",
                "data": {
                    "seismic_sequence": seismic(30, 12.0),
                    "gas_data": {"so2_tons_per_day": so2, "co2_tons_per_day": co2},
                    "insar_data": dict(quiet_insar),
                    "thermal_data": dict(quiet_thermal),
                },
            }
        )
    for i, (so2, co2, disp, heat) in enumerate(
        (
            (800.0, 2000.0, 22.0, 800.0),
            (900.0, 2200.0, 25.0, 1000.0),
            (1000.0, 2400.0, 28.0, 1200.0),
            (850.0, 2100.0, 30.0, 900.0),
        )
    ):
        scenarios.append(
            {
                "kind": f"multi_precursor_unrest_{i}",
                "alert_level": "warning",
                "data": {
                    "seismic_sequence": seismic(40, 15.0),
                    "gas_data": {"so2_tons_per_day": so2, "co2_tons_per_day": co2},
                    "insar_data": {
                        "vertical_displacement_cm": disp,
                        "deformation_rate_cm_day": 2.0,
                    },
                    "thermal_data": {
                        "brightness_temperature_k": [300.0, 305.0, 410.0, 420.0],
                        "radiant_heat_mw": heat,
                    },
                },
            }
        )

    path = SCENARIO_DIR / "volcano_scenarios.json"
    write_canonical_json(path, {"scenarios": scenarios})
    return {
        "sha256": hash_scenario_file(path),
        "hash_policy": "file_bytes",
        "label_source": "constructed",
        "seed": SEEDS["volcano"],
        "n_scenarios": len(scenarios),
        "levels": ["normal", "advisory", "watch", "warning"],
        "construction": (
            "Four scenarios per intended USGS alert level, labelled by which "
            "genuine precursors are present: none (normal); one modest seismic "
            "swarm, ~5% robust-outlier fraction (advisory); strong swarm + SO2 "
            "4-5x baseline degassing (watch); swarm + degassing + 22-30 cm InSAR "
            "displacement + >400 K thermal anomaly (warning). Mirrors "
            "tests/detectors/test_volcanic_honesty.py fixtures."
        ),
    }


# ---------------------------------------------------------------------------
# Real recorded data (NOAA SWPC snapshots -> window files)
# ---------------------------------------------------------------------------


def _parse_iso(ts: str) -> datetime:
    """Parse an SWPC timestamp (with or without trailing ``Z``).

    Args:
        ts: Timestamp string.

    Returns:
        Naive UTC datetime.
    """
    return datetime.fromisoformat(ts.replace("Z", ""))


def process_solar_kp(raw_dir: Path, fetched_at: str) -> dict[str, Any]:
    """Build Kp forecast-skill windows from real SWPC snapshots.

    For every observed 3-hour planetary-Kp interval, averages the real
    propagated solar wind (speed, IMF By/Bz) whose *arrival* time
    (``propagated_time_tag``) falls inside the interval. Windows with fewer
    than 60 valid wind minutes are dropped (never imputed). The observed Kp
    is the ground-based label; the guard drives the detector's Boyle-index
    physics with the measured wind and scores Kp MAE + G-bucket accuracy.

    Args:
        raw_dir: Directory containing the raw SWPC snapshot files.
        fetched_at: ISO-8601 UTC timestamp of the one-time fetch.

    Returns:
        Manifest entry for the written JSON.

    Raises:
        FileNotFoundError: If a required raw snapshot is missing.
        ValueError: If fewer than 20 usable windows result (data problem).
    """
    import json as _json

    kp_raw_path = raw_dir / "noaa-planetary-k-index.json"
    wind_raw_path = raw_dir / "propagated-solar-wind.json"
    for p in (kp_raw_path, wind_raw_path):
        if not p.exists():
            raise FileNotFoundError(f"raw SWPC snapshot missing: {p}")
    kp_rows = _json.loads(kp_raw_path.read_text())
    wind_rows = _json.loads(wind_raw_path.read_text())

    header, wind_data = wind_rows[0], wind_rows[1:]
    idx = {name: header.index(name) for name in ("speed", "by", "bz", "propagated_time_tag")}
    samples = []
    for row in wind_data:
        if row[idx["speed"]] is None or row[idx["by"]] is None or row[idx["bz"]] is None:
            continue  # drop, never impute
        samples.append(
            (
                _parse_iso(row[idx["propagated_time_tag"]]),
                float(row[idx["speed"]]),
                float(row[idx["by"]]),
                float(row[idx["bz"]]),
            )
        )
    samples.sort(key=lambda s: s[0])

    windows: list[dict[str, Any]] = []
    for row in kp_rows:
        start = _parse_iso(row["time_tag"])
        end = start + timedelta(hours=3)
        in_window = [s for s in samples if start <= s[0] < end]
        if len(in_window) < 60:
            continue
        windows.append(
            {
                "window_start": start.isoformat() + "Z",
                "kp_observed": float(row["Kp"]),
                "n_wind_minutes": len(in_window),
                "solar_wind_speed_km_s": round(float(np.mean([s[1] for s in in_window])), 3),
                "by_imf_nt": round(float(np.mean([s[2] for s in in_window])), 4),
                "bz_imf_nt": round(float(np.mean([s[3] for s in in_window])), 4),
            }
        )
    if len(windows) < 20:
        raise ValueError(f"only {len(windows)} usable Kp windows; refusing to build a thin set")

    payload = {
        "windows": windows,
        "provenance": {
            "label_source": "measured",
            "labels": "NOAA/GFZ planetary Kp index (ground magnetometer network)",
            "inputs": "SWPC propagated real-time solar wind (speed, IMF By/Bz), "
            "averaged per 3-hour Kp interval by propagated arrival time; "
            "windows with <60 valid minutes dropped, never imputed",
            "fetched_at": fetched_at,
            "sources": [
                {
                    "url": SWPC_SOURCES["kp"],
                    "sha256": sha256_file(kp_raw_path),
                    "rows": len(kp_rows),
                },
                {
                    "url": SWPC_SOURCES["wind"],
                    "sha256": sha256_file(wind_raw_path),
                    "rows": len(wind_data),
                },
            ],
        },
    }
    path = SCENARIO_DIR / "solar_kp_windows.json"
    write_canonical_json(path, payload)
    return {
        "sha256": hash_scenario_file(path),
        "hash_policy": "file_bytes",
        "label_source": "measured",
        "n_windows": len(windows),
        "kp_range": [
            min(w["kp_observed"] for w in windows),
            max(w["kp_observed"] for w in windows),
        ],
        "provenance": payload["provenance"],
    }


def process_solar_flares(raw_dir: Path, fetched_at: str) -> dict[str, Any]:
    """Build flare-class windows from real GOES X-ray flux.

    Splits the 7-day GOES primary X-ray record into 6-hour windows; each
    window's label is the NOAA class of its measured peak long-channel
    (0.1-0.8 nm) flux -- a *definitional* label (the NOAA class IS a flux
    threshold), so this set gates the classification chain, not forecast
    skill, and is documented as such. Windows with under 300 long-channel
    minutes are dropped. The independent SWPC flare-event list is attached
    as evidence per window.

    Args:
        raw_dir: Directory containing the raw SWPC snapshot files.
        fetched_at: ISO-8601 UTC timestamp of the one-time fetch.

    Returns:
        Manifest entry for the written JSON.

    Raises:
        FileNotFoundError: If a required raw snapshot is missing.
        ValueError: If fewer than 3 flare classes are represented (a flat-sun
            week cannot pin a meaningful classification baseline).
    """
    import json as _json

    xray_raw_path = raw_dir / "xrays-7-day.json"
    flares_raw_path = raw_dir / "xray-flares-7-day.json"
    for p in (xray_raw_path, flares_raw_path):
        if not p.exists():
            raise FileNotFoundError(f"raw SWPC snapshot missing: {p}")
    xray_rows = _json.loads(xray_raw_path.read_text())
    flare_events = _json.loads(flares_raw_path.read_text())

    by_window: dict[str, dict[str, list[float]]] = {}
    for row in xray_rows:
        if row.get("flux") is None:
            continue
        ts = _parse_iso(row["time_tag"])
        key = f"{ts.date().isoformat()}T{(ts.hour // 6) * 6:02d}"
        channel = "long" if row["energy"] == "0.1-0.8nm" else "short"
        by_window.setdefault(key, {"long": [], "short": []})[channel].append(float(row["flux"]))

    windows: list[dict[str, Any]] = []
    for key in sorted(by_window):
        chans = by_window[key]
        if len(chans["long"]) < 300:
            continue  # partial window at the record edge: drop, never pad
        peak_long = max(chans["long"])
        peak_short = max(chans["short"]) if chans["short"] else 0.0
        window_start = _parse_iso(key + ":00:00")
        window_end = window_start + timedelta(hours=6)
        events_in = [
            {"max_time": ev["max_time"], "max_class": ev["max_class"]}
            for ev in flare_events
            if window_start <= _parse_iso(ev["max_time"]) < window_end
        ]
        windows.append(
            {
                "window_start": window_start.isoformat() + "Z",
                "flare_class": _flare_class(peak_long),
                "peak_flux_long_wm2": peak_long,
                "peak_flux_short_wm2": peak_short,
                "n_minutes_long": len(chans["long"]),
                "swpc_flare_events": events_in,
            }
        )
    classes: set[str] = {w["flare_class"] for w in windows}
    if len(classes) < 3:
        raise ValueError(
            f"flare windows span only {sorted(classes)}; a flat week cannot pin "
            "a meaningful classification baseline -- refetch during activity"
        )

    payload = {
        "windows": windows,
        "provenance": {
            "label_source": "measured",
            "labels": (
                "NOAA class of each 6 h window's measured peak GOES 0.1-0.8 nm "
                "flux. DEFINITIONAL label (the NOAA class is a flux threshold): "
                "this set gates the classification chain end-to-end on real "
                "measurements, not forecast skill."
            ),
            "fetched_at": fetched_at,
            "sources": [
                {
                    "url": SWPC_SOURCES["xray"],
                    "sha256": sha256_file(xray_raw_path),
                    "rows": len(xray_rows),
                },
                {
                    "url": SWPC_SOURCES["flares"],
                    "sha256": sha256_file(flares_raw_path),
                    "rows": len(flare_events),
                },
            ],
        },
    }
    path = SCENARIO_DIR / "solar_flare_windows.json"
    write_canonical_json(path, payload)
    return {
        "sha256": hash_scenario_file(path),
        "hash_policy": "file_bytes",
        "label_source": "measured",
        "n_windows": len(windows),
        "class_counts": {
            c: sum(1 for w in windows if w["flare_class"] == c) for c in sorted(classes)
        },
        "provenance": payload["provenance"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _update_manifest(entries: dict[str, dict[str, Any]]) -> None:
    """Merge new file entries into the manifest and rewrite it canonically.

    Args:
        entries: Mapping of scenario file name to manifest entry.
    """
    try:
        manifest = load_manifest()
    except FileNotFoundError:
        manifest = {
            "generator": "benchmarks/hazard_scenarios/generate_scenarios.py",
            "seeds": SEEDS,
            "files": {},
        }
    manifest["seeds"] = SEEDS
    manifest["numpy_version_at_generation"] = np.__version__
    manifest.setdefault("files", {}).update(entries)
    write_canonical_json(MANIFEST_PATH, manifest)


def main() -> int:
    """CLI entry point: regenerate constructed sets and/or real solar windows."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--constructed",
        action="store_true",
        help="regenerate all constructed (seeded physics) scenario sets",
    )
    ap.add_argument(
        "--solar-from-raw",
        type=Path,
        metavar="DIR",
        help="process real SWPC raw snapshots from DIR into solar window files",
    )
    ap.add_argument(
        "--fetched-at",
        help="ISO-8601 UTC fetch timestamp of the raw snapshots (required with --solar-from-raw)",
    )
    args = ap.parse_args()

    if not args.constructed and args.solar_from_raw is None:
        ap.print_help()
        return 1
    if args.solar_from_raw is not None and not args.fetched_at:
        ap.error("--fetched-at is required with --solar-from-raw (fetch provenance)")

    entries: dict[str, dict[str, Any]] = {}
    if args.constructed:
        entries["tornado_scenarios.npz"] = build_tornado()
        entries["earthquake_scenarios.npz"] = build_earthquake()
        entries["tsunami_scenarios.npz"] = build_tsunami()
        entries["hurricane_scenarios.npz"] = build_hurricane()
        entries["flood_scenarios.json"] = build_flood()
        entries["volcano_scenarios.json"] = build_volcano()
    if args.solar_from_raw is not None:
        entries["solar_kp_windows.json"] = process_solar_kp(args.solar_from_raw, args.fetched_at)
        entries["solar_flare_windows.json"] = process_solar_flares(
            args.solar_from_raw, args.fetched_at
        )

    _update_manifest(entries)
    for name, entry in entries.items():
        print(f"wrote {name}: sha256={entry['sha256'][:12]}...")
    print(f"manifest updated: {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
