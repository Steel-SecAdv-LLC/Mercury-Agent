# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""EnergyLoader synthetic fallback: G-scale thresholds and flare/storm causality.

The synthetic series is gated behind MERCURY_ALLOW_SYNTHETIC; these tests call
the generator directly, so no gate toggling is needed.
"""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.evaluation.hazard_metrics import noaa_g_scale
from omni_mercury_engine.loaders.energy_loader import EnergyLoader, _g_levels

_EVENT = {
    "name": "test storm",
    "date": "1989-03-13",
    "start": "1989-02-01T00:00:00Z",
    "synthetic_hours": 2160,
    "peak_kp": 9,
    "storm_day": 20,
}


def test_g_levels_follow_noaa_g_scale() -> None:
    kp = np.array([4.33, 4.67, 6.33, 6.67, 8.67, 9.0])
    assert _g_levels(kp).tolist() == [noaa_g_scale(k) for k in kp]
    assert _g_levels(kp).tolist() == [0, 1, 2, 3, 4, 5]


def test_flares_lead_storms_not_track_them() -> None:
    """M/X flares precede geomagnetic storm steps by a 1-3 day CME transit.

    Regression: the fallback set the X-ray class from the SIMULTANEOUS Kp
    (X-class whenever Kp >= 7), teaching the anti-physical shortcut that
    flares and storms peak together.
    """
    df = EnergyLoader()._generate_synthetic_kp_series(_EVENT)
    kp = df["kp"].to_numpy()
    xray = df["xray_class"].to_numpy()
    storm = _g_levels(kp) >= 1
    onsets = np.flatnonzero(storm & ~np.concatenate(([False], storm[:-1])))
    flares = np.flatnonzero(xray >= 4.0)
    assert onsets.size > 0 and flares.size > 0

    # Every storm episode was preceded by an eruption 8-24 steps (1-3 days)
    # earlier, clipped to the series start for an early onset.
    for onset in onsets:
        leads = onset - flares[flares <= onset]
        assert leads.size > 0
        assert ((leads >= 8) & (leads <= 24)).any() or (onset < 8 and (flares == 0).any())

    # The old generator set M/X on every simultaneous storm step: now storm
    # steps are mostly flare-free, and eruptions sit on pre-storm steps.
    assert (xray[storm] < 4.0).any()
    assert (~storm[flares]).any() or (flares == 0).any()


def test_ground_truth_labels_are_g3_per_noaa_scale() -> None:
    """Labels follow noaa_g_scale >= G3 (Kp 7- = 6.67 is severe, 6.33 is not)."""
    df = EnergyLoader()._generate_synthetic_kp_series(_EVENT)
    kp = df["kp"].to_numpy()
    labels = (_g_levels(kp) >= 3).astype(int)
    assert labels.sum() > 0
    covered = kp[labels == 1]
    assert covered.min() >= 6.5  # G3 decision bound (7- files as G3)
    uncovered = kp[labels == 0]
    assert uncovered.max() < 6.5
