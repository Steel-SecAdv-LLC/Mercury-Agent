# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Offline, deterministic tests for the Schumann weak-supervision labeller (WS-C).

These inject fixed NOAA-shaped catalogs (no network), so the labeling logic --
NOAA G1+ storm windows (Kp 5- and up, via noaa_g_scale), M/X-class flare
windows, documented lag, provenance, and label-noise disclosure -- is verified
reproducibly.
"""

from __future__ import annotations

from datetime import UTC, datetime

from omni_mercury_engine.space.schumann_labeling import (
    FLARE_M_CLASS,
    FLARE_X_CLASS,
    KP_STORM_THRESHOLD,
    fetch_catalogs,
    label_noise_disclosure,
)

# Fixed NOAA-shaped fixtures (no network). The 09:00 row is a 5- (G1) bin in
# the live feed's shape: the thirds string and estimated_kp carry 4.67 while
# the filing integer kp_index reads 5 -- a literal "kp_index >= 5" check only
# caught it by that filing convention, and recorded Kp 5.0 instead of 4.67.
_KP = [
    {"time_tag": "2026-01-01T00:00:00", "kp_index": 2},  # quiet
    {"time_tag": "2026-01-01T03:00:00", "kp_index": 6},  # storm (G2)
    {"time_tag": "2026-01-01T06:00:00", "kp_index": 3},  # quiet
    {"time_tag": "2026-01-01T09:00:00", "kp_index": 5, "estimated_kp": 4.67, "kp": "5M"},  # G1
    {"time_tag": "2026-01-01T12:00:00", "kp_index": 4, "estimated_kp": 4.33, "kp": "4P"},  # 4+
]
_XRAY = [
    {"time_tag": "2026-01-02T00:00:00Z", "energy": "0.1-0.8nm", "flux": 1.0e-7},  # below M
    {"time_tag": "2026-01-02T01:00:00Z", "energy": "0.1-0.8nm", "flux": 2.0e-5},  # M-flare
    {"time_tag": "2026-01-02T02:00:00Z", "energy": "0.1-0.8nm", "flux": 3.0e-4},  # X-flare
    {"time_tag": "2026-01-02T02:00:00Z", "energy": "0.05-0.4nm", "flux": 9.9e-3},  # wrong band
]


def _catalog():
    return fetch_catalogs(kp_json=_KP, xray_json=_XRAY)


def test_storm_window_from_high_kp() -> None:
    cat = _catalog()
    storms = [w for w in cat.windows if w.driver == "geomagnetic_storm"]
    # the integer-6 bin and the 5- (4.67) bin; the 4+ (4.33) bin is sub-G1
    assert len(storms) == 2
    assert sorted(w.magnitude for w in storms) == [4.666666666666667, 6.0]
    # labelled positive at the storm times
    assert cat.label(datetime(2026, 1, 1, 3, 0, tzinfo=UTC)) == 1
    assert cat.label(datetime(2026, 1, 1, 9, 0, tzinfo=UTC)) == 1
    # quiet hours far from any driver are negative
    assert cat.label(datetime(2026, 1, 1, 0, 0, tzinfo=UTC)) == 0


def test_flare_windows_m_and_x() -> None:
    cat = _catalog()
    flares = [w for w in cat.windows if w.driver.endswith("flare")]
    assert {w.driver for w in flares} == {"M_flare", "X_flare"}
    # sub-M flux and the wrong energy band do not create windows
    assert len(flares) == 2
    # flare onset is labelled positive; well after the lag it is negative
    assert cat.label(datetime(2026, 1, 2, 1, 0, tzinfo=UTC)) == 1
    assert cat.label(datetime(2026, 1, 2, 5, 0, tzinfo=UTC)) == 0


def test_thresholds_are_documented_constants() -> None:
    assert KP_STORM_THRESHOLD == 14.0 / 3.0  # NOAA G1 onset: Kp 5- (4.67)
    assert FLARE_M_CLASS == 1e-5
    assert FLARE_X_CLASS == 1e-4


def test_provenance_and_license() -> None:
    cat = _catalog()
    p = cat.provenance
    assert p["label_kind"] == "proxy_event_coincidence"
    assert "public domain" in p["license"].lower()
    assert p["thresholds"]["kp_storm"] == KP_STORM_THRESHOLD
    assert p["n_windows"] == len(cat.windows)
    assert p["n_storm_windows"] == 2
    assert p["n_flare_windows"] == 2


def test_label_noise_is_disclosed() -> None:
    nz = label_noise_disclosure()
    assert {"false_positive", "false_negative", "timing", "implication"} <= set(nz)
    # transparency: it must say these are not ground truth
    assert "ground truth" in nz["implication"].lower()
