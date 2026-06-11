# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the geographic movement-plausibility detector.

The casework scenarios reproduce defects measured in an external detector
during the 2026-06-10 audit: a weighted-sum fusion (0.4/0.3/0.3 against a
0.72 threshold) that no single channel could ever trip, and a jump
statistic measured from the candidate point that suppressed its own
signal.  The remaining tests lock the BaseDetector interface contract.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST, DetectorCategory
from omni_mercury_engine.core.exceptions import DetectorException
from omni_mercury_engine.detectors.geo_movement import (
    GeoMovementAnomalyDetector,
    MovementAssessment,
)

T0 = datetime(2026, 6, 1, 8, 0)


def _steady_history(n: int = 20) -> list[tuple[float, float, datetime]]:
    """Hourly sightings drifting ~1.4 km NE per hour."""
    return [(40.0 + 0.01 * i, -105.0 + 0.01 * i, T0 + timedelta(hours=i)) for i in range(n)]


def _steady_track(n: int = 20) -> np.ndarray:
    return np.array(
        [[40.0 + 0.01 * i, -105.0 + 0.01 * i, i * 3600.0] for i in range(n)],
        dtype=np.float64,
    )


class TestCaseworkAssess:
    """assess(): one candidate observation against a sighting history."""

    def test_impossible_jump_fires(self) -> None:
        """Audit defect 3: a ~450 km/h teleport scored 0.30 < 0.72 -> NORMAL."""
        det = GeoMovementAnomalyDetector()
        res = det.assess(_steady_history(), (45.0, -100.0), T0 + timedelta(hours=20))
        assert isinstance(res, MovementAssessment)
        assert res.is_anomalous
        assert res.channels["velocity"] == 1.0
        assert res.reason == "velocity"

    def test_normal_step_stays_quiet(self) -> None:
        det = GeoMovementAnomalyDetector()
        res = det.assess(_steady_history(), (40.20, -104.80), T0 + timedelta(hours=20))
        assert not res.is_anomalous
        assert res.score < 0.3

    def test_insufficient_history(self) -> None:
        det = GeoMovementAnomalyDetector()
        res = det.assess([(40.0, -105.0, T0)], (41.0, -105.0), T0 + timedelta(hours=1))
        assert res.score == 0.0
        assert not res.is_anomalous
        assert res.reason == "insufficient_history"

    def test_single_saturated_channel_fires_alone(self) -> None:
        """Structural noisy-OR property: velocity alone trips the threshold
        while the jump and gap channels stay near zero (impossible under
        the audited 0.4/0.3/0.3 weighted sum, where one channel maxes the
        score at 0.4)."""
        # 100 km steps every 4 h: large, well-spread step distribution.
        history = [(40.0 + 0.9 * i, -105.0, T0 + timedelta(hours=4 * i)) for i in range(10)]
        det = GeoMovementAnomalyDetector()
        # Same 100 km step length, but covered in 30 minutes: 200 km/h.
        res = det.assess(history, (40.0 + 0.9 * 9 + 0.9, -105.0), T0 + timedelta(hours=36.5))
        assert res.channels["velocity"] == 1.0
        assert res.channels["jump"] < 0.1
        assert res.channels["time_gap"] < 0.1
        assert res.is_anomalous

    def test_jump_scored_against_history_not_candidate(self) -> None:
        """Audit defect 4: spatial spread measured from the candidate point
        suppressed the jump signal.  A step 20x the historical mean must
        saturate the jump channel even at feasible speed."""
        # 1.4 km hourly steps; candidate step ~31 km in one hour (31 km/h).
        det = GeoMovementAnomalyDetector()
        res = det.assess(_steady_history(), (40.39, -104.99), T0 + timedelta(hours=20))
        assert res.channels["jump"] == 1.0
        assert res.channels["velocity"] < 0.3
        assert res.is_anomalous


class TestBaseDetectorContract:
    """fit/detect/extract_features interface compliance."""

    def test_detect_contract_keys(self) -> None:
        det = GeoMovementAnomalyDetector().fit(_steady_track())
        res = det.detect(_steady_track())
        assert 0.0 <= res["anomaly_score"] <= 1.0
        assert isinstance(res["is_anomaly"], bool)
        assert len(res["scores"]) == 20
        assert res["detector_type"] == "geo_movement"
        assert res["reason"] == "normal"

    def test_detect_flags_teleport_step(self) -> None:
        track = np.vstack([_steady_track(), [45.0, -100.0, 20 * 3600.0]])
        det = GeoMovementAnomalyDetector().fit(_steady_track())
        res = det.detect(track)
        assert res["is_anomaly"]
        assert res["anomaly_score"] > 0.99
        assert res["metadata"]["worst_step_index"] == 20
        assert res["reason"] == "velocity"

    def test_unfitted_detect_raises(self) -> None:
        with pytest.raises(DetectorException, match="fitted"):
            GeoMovementAnomalyDetector().detect(_steady_track())

    def test_fit_validation(self) -> None:
        det = GeoMovementAnomalyDetector()
        with pytest.raises(DetectorException, match="at least 2"):
            det.fit(np.array([[40.0, -105.0, 0.0]]))
        with pytest.raises(DetectorException, match="shape"):
            det.fit(np.zeros((5, 2)))
        with pytest.raises(DetectorException, match="non-decreasing"):
            det.fit(np.array([[40.0, -105.0, 3600.0], [40.1, -105.0, 0.0]]))
        with pytest.raises(DetectorException, match="non-finite"):
            det.fit(np.array([[40.0, np.nan, 0.0], [40.1, -105.0, 3600.0]]))
        with pytest.raises(DetectorException, match="out of range"):
            det.fit(np.array([[95.0, -105.0, 0.0], [40.1, -105.0, 3600.0]]))

    def test_config_validation(self) -> None:
        with pytest.raises(ValueError):
            GeoMovementAnomalyDetector({"threshold": 1.5})
        with pytest.raises(ValueError, match="max_feasible_kmh"):
            GeoMovementAnomalyDetector({"max_feasible_kmh": -5.0})
        with pytest.raises(ValueError, match="expected_gap_hours"):
            GeoMovementAnomalyDetector({"expected_gap_hours": 0.0})

    def test_extract_features_shape(self) -> None:
        det = GeoMovementAnomalyDetector()
        feats = det.extract_features(_steady_track())
        assert tuple(feats.shape) == (20, 8)
        arr = feats.numpy() if hasattr(feats, "numpy") else np.asarray(feats)
        assert np.all(np.isfinite(arr))
        # The first point has no incoming step: channel features are zero.
        assert np.allclose(arr[0, :6], 0.0)

    def test_single_point_detect_is_insufficient_history(self) -> None:
        det = GeoMovementAnomalyDetector().fit(_steady_track())
        res = det.detect(np.array([[40.0, -105.0, 0.0]]))
        assert res["anomaly_score"] == 0.0
        assert not res["is_anomaly"]
        assert res["reason"] == "insufficient_history"


class TestRegistration:
    """Package and manifest wiring."""

    def test_lazy_package_import(self) -> None:
        from omni_mercury_engine.detectors import GeoMovementAnomalyDetector as lazy_cls

        assert lazy_cls is GeoMovementAnomalyDetector

    def test_manifest_entry(self) -> None:
        entry = next(e for e in DETECTOR_MANIFEST if e.name == "geo_movement")
        assert entry.class_name == "GeoMovementAnomalyDetector"
        assert entry.category == DetectorCategory.BASE
        assert entry.feature_dim == 8
        assert "geospatial" in entry.tags
