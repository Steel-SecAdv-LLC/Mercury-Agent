# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the earthquake → tsunami cascade detector.

Real-data fixtures (committed, offline-deterministic):

- ``tests/fixtures/geological/usgs_tohoku_2011_event.json``: the archived
  USGS GeoJSON response for the 2011 M9.1 Great Tōhoku earthquake (event id
  ``official20110311054624120_30``, hypocentre 38.297N 142.373E, depth
  29 km). Provenance is embedded in the file's ``metadata`` block (the USGS
  FDSN query URL and generation timestamp).
- ``tests/fixtures/geological/usgs_m7plus_2011_catalog.json``: the archived
  USGS FDSN catalog of all 20 M >= 7.0 events of 2011, same embedded
  provenance.

The Tōhoku screening expectations are the published PTWC criteria (IOC
Technical Series No. 87 §4.3.2): M9.1 at 29 km depth is regional-expanding
warning class. Note the archived catalog's ``tsunami`` flag is 0 even for
Tōhoku — it reflects real-time alert state, which is exactly why the
detector must not use it as a criterion.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from omni_mercury_engine.detectors.geological.eq_tsunami_cascade import (
    DART_DETECTION_THRESHOLD_M,
    EXPANDING_WARNING_MIN_MAGNITUDE,
    INFO_BULLETIN_MIN_MAGNITUDE,
    MAX_TSUNAMIGENIC_DEPTH_KM,
    REGIONAL_WARNING_MIN_MAGNITUDE,
    CascadeStage,
    EqTsunamiCascadeDetector,
    ScreeningProduct,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "geological"


def _tohoku_feature() -> dict:
    data = json.loads((FIXTURES / "usgs_tohoku_2011_event.json").read_text())
    assert data["metadata"]["count"] == 1
    return data["features"][0]


def _catalog_2011() -> dict:
    return json.loads((FIXTURES / "usgs_m7plus_2011_catalog.json").read_text())


def _tide_series(
    n: int = 240, interval_s: float = 60.0, tsunami_amplitude_m: float = 0.0
) -> np.ndarray:
    """A slow tidal signal with an optional tsunami arrival in the last 25%."""
    t = np.arange(n) * interval_s
    tide = 0.5 * np.sin(2.0 * np.pi * t / (12.42 * 3600.0))  # M2 tide
    series = tide.copy()
    if tsunami_amplitude_m > 0.0:
        onset = int(n * 0.85)
        wave_t = t[onset:] - t[onset]
        series[onset:] += tsunami_amplitude_m * np.sin(2.0 * np.pi * wave_t / (20.0 * 60.0))
    return series


class TestPublishedCriteria:
    """The embedded thresholds are the published PTWC values."""

    def test_constants_match_ioc_ts87(self) -> None:
        assert INFO_BULLETIN_MIN_MAGNITUDE == 6.5
        assert REGIONAL_WARNING_MIN_MAGNITUDE == 7.6
        assert EXPANDING_WARNING_MIN_MAGNITUDE == 7.9
        assert MAX_TSUNAMIGENIC_DEPTH_KM == 100.0
        assert DART_DETECTION_THRESHOLD_M == 0.03


class TestTohokuScreening:
    """PTWC screening on the real archived Tōhoku 2011 event."""

    def test_tohoku_metadata_parsed_from_fixture(self) -> None:
        parsed = EqTsunamiCascadeDetector.parse_event(_tohoku_feature())
        assert parsed["event_id"] == "official20110311054624120_30"
        assert parsed["magnitude"] == pytest.approx(9.1)
        assert parsed["depth_km"] == pytest.approx(29.0)
        assert "Tohoku" in parsed["place"]

    def test_tohoku_is_regional_expanding_warning(self) -> None:
        result = EqTsunamiCascadeDetector().screen_event(_tohoku_feature(), offshore=True)
        assert result.product is ScreeningProduct.REGIONAL_EXPANDING_WARNING
        assert result.criteria["depth_lt_100km"] is True
        assert result.criteria["location_criterion"] == "offshore"
        assert result.criteria["focal_mechanism"] == "unavailable_in_basic_feed"

    def test_archived_tsunami_flag_recorded_but_not_used(self) -> None:
        """The archive flags Tōhoku tsunami=0; screening must still warn."""
        feature = _tohoku_feature()
        assert feature["properties"]["tsunami"] == 0
        result = EqTsunamiCascadeDetector().screen_event(feature, offshore=True)
        assert result.criteria["usgs_tsunami_flag"] is False
        assert result.product is ScreeningProduct.REGIONAL_EXPANDING_WARNING

    def test_caller_supplied_rake_recorded_but_mechanism_free_table_decides(self) -> None:
        """Tōhoku was a thrust event (rake ~ +90°, USGS W-phase solution);
        supplying that knowledge is recorded as evidence but the published
        mechanism-free PTWC table still decides the product."""
        detector = EqTsunamiCascadeDetector()
        with_rake = detector.screen_event(
            {"magnitude": 9.1, "depth_km": 29.0, "rake_deg": 90.0}, offshore=True
        )
        without = detector.screen_event({"magnitude": 9.1, "depth_km": 29.0}, offshore=True)
        assert with_rake.criteria["focal_mechanism"] == {
            "rake_deg": 90.0,
            "source": "caller_supplied",
        }
        assert without.criteria["focal_mechanism"] == "unavailable_in_basic_feed"
        assert with_rake.product is without.product


class TestCatalogScreening:
    """Screening the real 2011 M7+ catalog window."""

    def test_all_twenty_events_screen(self) -> None:
        results = EqTsunamiCascadeDetector().screen_catalog(_catalog_2011())
        assert len(results) == 20
        # Every event is M >= 7.0 >= info floor: none may screen as NONE.
        assert all(r.product is not ScreeningProduct.NONE for r in results)

    def test_deep_events_capped_at_information(self) -> None:
        """usp000hsdc (M7.0, 576.8 km) and usp000j83e (M7.3, 644.6 km) are
        far below the 100 km tsunamigenic limit."""
        by_id = {r.event_id: r for r in EqTsunamiCascadeDetector().screen_catalog(_catalog_2011())}
        assert by_id["usp000hsdc"].product is ScreeningProduct.INFORMATION_BULLETIN
        assert by_id["usp000j83e"].product is ScreeningProduct.INFORMATION_BULLETIN

    def test_warning_class_partition_matches_published_table(self) -> None:
        """Recompute the expected product for each event independently."""
        detector = EqTsunamiCascadeDetector()
        for result in detector.screen_catalog(_catalog_2011()):
            if result.magnitude < 7.6 or result.depth_km >= 100.0:
                expected = ScreeningProduct.INFORMATION_BULLETIN
            elif result.magnitude < 7.9:
                expected = ScreeningProduct.REGIONAL_FIXED_WARNING
            else:
                expected = ScreeningProduct.REGIONAL_EXPANDING_WARNING
            assert result.product is expected, result.event_id

    def test_tohoku_and_honshu_aftershock_are_warning_class(self) -> None:
        results = EqTsunamiCascadeDetector().screen_catalog(_catalog_2011())
        warning_ids = {
            r.event_id
            for r in results
            if r.product
            in (
                ScreeningProduct.REGIONAL_FIXED_WARNING,
                ScreeningProduct.REGIONAL_EXPANDING_WARNING,
            )
        }
        # Tōhoku mainshock (M9.1/29km), the M7.9 (42.6km) and M7.7 (18.6km)
        # aftershocks, and the M7.6 Kermadec event (17km).
        assert "official20110311054624120_30" in warning_ids
        assert "usp000hvpa" in warning_ids
        assert "usp000hvpg" in warning_ids
        assert "usp000j48h" in warning_ids


class TestStateMachine:
    """Stage transitions require real evidence at every step."""

    def test_initial_state_quiet(self) -> None:
        assert EqTsunamiCascadeDetector().state().stage == CascadeStage.QUIET.value

    def test_small_event_stays_quiet(self) -> None:
        detector = EqTsunamiCascadeDetector()
        state = detector.process_event({"magnitude": 5.8, "depth_km": 10.0})
        assert state.stage == CascadeStage.QUIET.value

    def test_information_event_reaches_evaluating_only(self) -> None:
        detector = EqTsunamiCascadeDetector()
        state = detector.process_event({"magnitude": 7.0, "depth_km": 30.0})
        assert state.stage == CascadeStage.EVALUATING.value

    def test_deep_major_event_does_not_watch(self) -> None:
        detector = EqTsunamiCascadeDetector()
        state = detector.process_event({"magnitude": 8.2, "depth_km": 550.0})
        assert state.stage == CascadeStage.EVALUATING.value
        assert state.screening is not None
        assert state.screening.product is ScreeningProduct.INFORMATION_BULLETIN

    def test_known_inland_event_does_not_watch(self) -> None:
        detector = EqTsunamiCascadeDetector()
        state = detector.process_event({"magnitude": 7.9, "depth_km": 15.0}, offshore=False)
        assert state.stage == CascadeStage.EVALUATING.value

    def test_tohoku_reaches_watch(self) -> None:
        detector = EqTsunamiCascadeDetector()
        state = detector.process_event(_tohoku_feature(), offshore=True)
        assert state.stage == CascadeStage.TSUNAMI_WATCH.value

    def test_threat_requires_water_level_evidence(self) -> None:
        """WATCH + sub-threshold water level stays WATCH."""
        detector = EqTsunamiCascadeDetector()
        detector.process_event(_tohoku_feature(), offshore=True)
        state = detector.confirm_water_level(
            _tide_series(tsunami_amplitude_m=0.0),
            sampling_interval_s=60.0,
            run_spectral_analysis=False,
        )
        assert state.stage == CascadeStage.TSUNAMI_WATCH.value
        assert state.water_level is not None
        assert not state.water_level.confirmed
        assert state.water_level.max_residual_m < DART_DETECTION_THRESHOLD_M

    def test_threat_on_dart_style_confirmation(self) -> None:
        detector = EqTsunamiCascadeDetector()
        detector.process_event(_tohoku_feature(), offshore=True)
        state = detector.confirm_water_level(
            _tide_series(tsunami_amplitude_m=0.5),
            sampling_interval_s=60.0,
            run_spectral_analysis=False,
        )
        assert state.stage == CascadeStage.TSUNAMI_THREAT.value
        assert state.water_level is not None
        assert state.water_level.confirmed
        assert state.water_level.max_residual_m >= DART_DETECTION_THRESHOLD_M

    def test_confirmation_without_watch_raises(self) -> None:
        detector = EqTsunamiCascadeDetector()
        with pytest.raises(RuntimeError, match="TSUNAMI_WATCH"):
            detector.confirm_water_level(_tide_series(), 60.0)

    def test_confirmed_observation_path(self) -> None:
        detector = EqTsunamiCascadeDetector()
        detector.process_event(_tohoku_feature(), offshore=True)
        state = detector.confirm_observation(
            "1.2 m wave observed at coastal gauge", "national tide gauge network"
        )
        assert state.stage == CascadeStage.TSUNAMI_THREAT.value

    def test_empty_observation_refused(self) -> None:
        detector = EqTsunamiCascadeDetector()
        detector.process_event(_tohoku_feature(), offshore=True)
        with pytest.raises(ValueError, match="non-empty"):
            detector.confirm_observation("  ", "someone")

    def test_reset_preserves_evidence_chain(self) -> None:
        detector = EqTsunamiCascadeDetector()
        detector.process_event(_tohoku_feature(), offshore=True)
        detector.reset()
        state = detector.state()
        assert state.stage == CascadeStage.QUIET.value
        kinds = [e["kind"] for e in state.evidence_chain]
        assert kinds == ["event_intake", "screening", "reset"]


class TestEvidenceChain:
    """Every transition carries a full evidence record."""

    def test_full_chain_through_threat(self) -> None:
        detector = EqTsunamiCascadeDetector()
        detector.process_event(_tohoku_feature(), offshore=True)
        state = detector.confirm_water_level(
            _tide_series(tsunami_amplitude_m=0.5),
            sampling_interval_s=60.0,
            run_spectral_analysis=False,
        )
        kinds = [e["kind"] for e in state.evidence_chain]
        assert kinds == ["event_intake", "screening", "water_level_confirmation"]
        intake, screening, confirmation = state.evidence_chain
        assert intake["detail"]["source"] == "USGS FDSN event feed"
        assert screening["detail"]["product"] == "regional_expanding_warning"
        assert "IOC Technical Series No. 87" in screening["detail"]["citation"]
        assert confirmation["detail"]["confirmed"] is True
        assert "Meinig" in confirmation["detail"]["citation"]

    def test_supplementary_spectral_evidence_is_deterministic_only(self) -> None:
        pytest.importorskip("torch")
        detector = EqTsunamiCascadeDetector()
        detector.process_event(_tohoku_feature(), offshore=True)
        state = detector.confirm_water_level(
            _tide_series(tsunami_amplitude_m=0.5),
            sampling_interval_s=60.0,
            run_spectral_analysis=True,
        )
        assert state.water_level is not None
        supplementary = state.water_level.supplementary_spectral
        assert supplementary is not None
        assert "resonance_score" in supplementary
        assert "neural confidence" in supplementary["note"]
        # The stage decision is identical with and without the supplement.
        assert state.stage == CascadeStage.TSUNAMI_THREAT.value


class TestFailLoud:
    """Input contracts."""

    def test_event_without_magnitude_raises(self) -> None:
        with pytest.raises(ValueError, match="magnitude"):
            EqTsunamiCascadeDetector().screen_event({"depth_km": 10.0})

    def test_event_without_depth_raises(self) -> None:
        with pytest.raises(ValueError, match="depth"):
            EqTsunamiCascadeDetector().screen_event({"magnitude": 8.0})

    def test_empty_catalog_raises(self) -> None:
        with pytest.raises(ValueError, match="features"):
            EqTsunamiCascadeDetector().screen_catalog({"features": []})

    def test_short_water_level_series_raises(self) -> None:
        detector = EqTsunamiCascadeDetector()
        detector.process_event(_tohoku_feature(), offshore=True)
        with pytest.raises(ValueError, match=">= 40 samples"):
            detector.confirm_water_level(np.zeros(10), 60.0)

    def test_nonfinite_water_level_raises(self) -> None:
        detector = EqTsunamiCascadeDetector()
        detector.process_event(_tohoku_feature(), offshore=True)
        series = _tide_series()
        series[50] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            detector.confirm_water_level(series, 60.0)

    def test_bad_sampling_interval_raises(self) -> None:
        detector = EqTsunamiCascadeDetector()
        detector.process_event(_tohoku_feature(), offshore=True)
        with pytest.raises(ValueError, match="sampling_interval_s"):
            detector.confirm_water_level(_tide_series(), 0.0)
