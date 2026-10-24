# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every series detector's false-alarm rate on a normal regime that does not drift.

A detector is fitted on the first 400 samples of a stationary AR(1) series and
asked about the next 400 from the same process.  Nothing in the held-out half is
anomalous, so whatever it flags there is a false alarm -- and because the
process is stationary, none of it can be attributed to the level drift that the
recorded NAB files do carry.  That makes this the one place where the squash
tier's documented contract, ``1 - calibration_quantile`` as the normal-regime
false-alarm rate, can be read directly.

The rates below are the measured behaviour, pinned so a change that worsens one
-- or improves one -- fails here and has to be recorded.  Several are pinned
*high*: a detector whose default threshold floods is recorded as flooding, in
the same spirit as ``test_recorded_nab_series.py``.  Two defects were found by
this measurement, both of them a manifest entry declaring a modality its class
cannot consume (``rca`` and ``kmeans_distance`` are ``matrix_2d`` detectors that
were declared ``series_1d``, so a 1-D series reshaped to a single row, every
node's or feature's training spread collapsed to its floor, and every held-out
point flagged); both are fixed, and both now sit under their correct modality
and outside this gate.

Seeds are fixed and the measurement is repeatable: two runs are byte-identical.
The whole sweep takes about 15 s.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST

SEEDS = (7, 11, 23, 101, 999)
FIT = 400
TOLERANCE = 0.03

#: entry -> mean share of a stationary held-out half it flags, as measured.
#: ``1 - calibration_quantile`` is 0.02 for the squash tier.
MEASURED: dict[str, float] = {
    # Flooding at the default threshold -- recorded, not accepted.
    "nano_safeguard": 1.0000,
    "acceleration_dynamics": 1.0000,
    "physics_integrated": 1.0000,
    "deeplog_sequence": 0.9965,
    "lstm_autoencoder": 0.6265,
    "srcnn": 0.2725,
    # Above the 0.02 the squash anchor targets, but bounded.
    "statistical": 0.0980,
    "energy_based": 0.0755,
    "deep_svdd": 0.0470,
    "extended_statistical": 0.0460,
    # At or under the contract.
    "particle_filter": 0.0305,
    "survival": 0.0270,
    "temporal": 0.0265,
    "sprt": 0.0250,
    "gaussian_process": 0.0245,
    "hawkes": 0.0240,
    "imm": 0.0235,
    "digital_twin": 0.0235,
    "diffusion_ad": 0.0230,
    "bocpd": 0.0205,
    "spectral_residual": 0.0205,
    "spiking": 0.0180,
    "echo_state": 0.0110,
    "spot_evt": 0.0075,
    "page_hinkley": 0.0025,
    # Flags nothing on a 1-D series: measured and pinned as the limit it is.
    "frequent_pattern": 0.0000,
    "spectral_vibration": 0.0000,
    "uiux_anomaly": 0.0000,
    "spectral_domain_frequency": 0.0000,
}

#: ``series_1d`` entries this gate cannot drive, and why.  Each serves its
#: domain API rather than ``BaseDetector.fit`` / ``detect``, so a plain 1-D
#: array is not the call it takes; they are covered by their own tests.
NO_BASE_DETECTOR_SEAM = frozenset(
    {
        "affective",
        "astrophysical",
        "atmospheric_river",
        "biometric",
        "chemistry",
        "chronos",
        "consciousness",
        "derecho",
        "drone",
        "drought",
        "dust_storm",
        "earthquake",
        "flood",
        "foundation_ensemble",
        "hail",
        "heatwave",
        "hurricane",
        "lightning",
        "matrix_profile",
        "meteor",
        "neural_cognitive",
        "neurosymbolic",
        "parapsychology",
        "psyop",
        "quantum",
        "schumann_resonance",
        "solar_flare",
        "tornado",
        "tsunami",
        "winter_storm",
    }
)


def stationary_ar1(seed: int, n: int = 800, phi: float = 0.6) -> np.ndarray[Any, Any]:
    """A normal regime with no level drift: ``x[t] = phi * x[t-1] + e[t]``."""
    rng = np.random.default_rng(seed)
    series = np.zeros(n)
    for t in range(1, n):
        series[t] = phi * series[t - 1] + rng.normal()
    return series


def _candidates() -> list[Any]:
    """Manifest entries this gate drives: 1-D series, registry seam."""
    return [
        e
        for e in DETECTOR_MANIFEST
        if e.modality == "series_1d"
        and "registry" in e.seams
        and e.name not in NO_BASE_DETECTOR_SEAM
    ]


def measure(name: str) -> float:
    """Mean share of the stationary held-out half that ``name`` flags."""
    entry = next(e for e in DETECTOR_MANIFEST if e.name == name)
    cls = getattr(importlib.import_module(entry.module_path), entry.class_name)
    rates = []
    for seed in SEEDS:
        series = stationary_ar1(seed)
        torch.manual_seed(0)
        np.random.seed(0)
        detector = cls()
        detector.fit(series[:FIT])
        flags = np.asarray(detector.detect(series[FIT:])["is_anomaly"], dtype=bool)
        rates.append(float(flags.mean()))
    return float(np.mean(rates))


@pytest.mark.parametrize("name", sorted(MEASURED))
def test_false_alarm_rate_matches_the_measurement(name: str) -> None:
    got = measure(name)
    assert got == pytest.approx(MEASURED[name], abs=TOLERANCE), (
        f"{name}: {got:.4f} of a stationary held-out half flagged, measured "
        f"{MEASURED[name]:.4f}; record the change if it is real"
    )


def test_every_drivable_series_entry_is_pinned() -> None:
    """A new ``series_1d`` entry has to land in ``MEASURED`` or in the exclusion set."""
    drivable = {e.name for e in _candidates()}
    assert drivable == set(MEASURED), (
        f"unpinned: {sorted(drivable - set(MEASURED))}; "
        f"pinned but no longer drivable: {sorted(set(MEASURED) - drivable)}"
    )


def test_no_excluded_entry_silently_gained_the_base_seam() -> None:
    """If one of these grows a ``fit``/``detect`` pair, it belongs in the gate."""
    gained = []
    for entry in DETECTOR_MANIFEST:
        if entry.name not in NO_BASE_DETECTOR_SEAM:
            continue
        cls = getattr(importlib.import_module(entry.module_path), entry.class_name)
        if callable(getattr(cls, "fit", None)) and callable(getattr(cls, "detect", None)):
            gained.append(entry.name)
    assert not gained, f"now drivable, pin them in MEASURED: {sorted(gained)}"
