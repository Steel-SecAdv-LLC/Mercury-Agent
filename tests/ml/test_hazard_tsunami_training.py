# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the tsunami DART training pipeline (hook ``tsunami_waveform``).

Offline: every test runs against committed real-data fixtures in
``tests/fixtures/hazard_training/tsunami/`` -- byte-exact row excerpts from
the real NOAA NDBC DART bottom-pressure archive for station 21419
(https://www.ndbc.noaa.gov/data/historical/dart/21419t2011.txt.gz, SHA-256
pinned in ``provenance.json`` next to the fixtures). The January excerpt is a
quiet tidal span including a real 9999.000 sentinel gap; the March excerpt
covers the 2011-03-11 Tohoku tsunami with event-mode (T=2/3) rows and the
blanked standard-mode stream. The labeled arrival used below (2011-03-11
07:06 UTC, runupHt 0.54 m) is the NCEI HazEL runup record for event 5413 at
D21419 (see ``provenance.json``).

Covers: DART parsing (sentinels, mode handling, exact-grid-slot event-mode
fill), deterministic detiding on a real tidal window, window gap rejection,
arrival-time resolution, temporal-split wiring, checkpoint operating-point
consumption (the validation-selected threshold governs the learned path's
``tsunami_detected`` decision; nonsensical thresholds refuse to load;
checkpoints without one keep the constructor threshold), the
measured-wave-height contract (``estimated_wave_height_m`` is the record's
peak deviation on both paths; the NN head is diagnostic-only), and the
differential physics-vs-shipped-checkpoint comparison through the public
detector API (skipped while no ``tsunami_dart`` checkpoint has been
shipped).
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from omni_mercury_engine.detectors.geological.disaster_detectors import TsunamiDetector
from omni_mercury_engine.ml.hazard_training import tsunami_waveform as tw
from omni_mercury_engine.models.checkpoint_paths import shipped_checkpoint_path

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "hazard_training" / "tsunami"
FIXTURE_JAN = FIXTURE_DIR / "21419t2011_jan01-09_excerpt.txt.gz"
FIXTURE_MAR = FIXTURE_DIR / "21419t2011_mar10-12_excerpt.txt.gz"

#: NCEI HazEL event 5413 (Tohoku) arrival at D21419 -- see provenance.json.
TOHOKU_ARRIVAL_UTC = dt.datetime(2011, 3, 11, 7, 6, tzinfo=dt.UTC)
TOHOKU_RUNUP_HT_M = 0.54

_SHIPPED = shipped_checkpoint_path("tsunami_dart")


def _fixture_rows(path: Path) -> list[list[str]]:
    """Raw measurement rows (tokenized) of a fixture, headers skipped."""
    with gzip.open(path, "rt") as fh:
        return [line.split() for line in fh if not line.startswith("#")]


def _tohoku_residual() -> np.ndarray:
    """The real detided Tohoku arrival window from the March fixture."""
    grid = tw.parse_dart_file(FIXTURE_MAR, "21419", 2011)
    anchor = tw._slot_of(TOHOKU_ARRIVAL_UTC, 2011)
    residual = tw.extract_residual_window(grid.values, anchor - 4 * tw.SLOTS_PER_HOUR)
    assert residual is not None
    return residual


def _write_checkpoint(path: Path, operating_point: dict[str, float] | None) -> Path:
    """Write a contract-complete checkpoint with seeded (untrained) weights.

    The operating-point machinery is decision plumbing: it must govern the
    ``tsunami_detected`` decision for ANY loaded weights, so hermetic seeded
    weights are sufficient (no candidate/shipped artifact required).
    """
    from omni_mercury_engine.detectors.geological.disaster_detectors import (
        WaveformFFTAnalyzer,
    )

    torch.manual_seed(20260709)
    payload: dict[str, object] = {
        "waveform_analyzer": WaveformFFTAnalyzer().state_dict(),
        "feature_spec": tw.FEATURE_SPEC_VERSION,
        "window_samples": tw.WINDOW_SAMPLES,
        "sampling_period_s": tw.SAMPLE_PERIOD_S,
        "detide": tw.DETIDE_METHOD,
    }
    if operating_point is not None:
        payload["operating_point"] = operating_point
    torch.save(payload, path)
    return path


class TestDartParsing:
    """The archive parser against independent recounts of the fixture rows."""

    def test_fixture_provenance_pins_source(self) -> None:
        prov = json.loads((FIXTURE_DIR / "provenance.json").read_text())
        assert prov["source_url"].startswith("https://www.ndbc.noaa.gov/data/historical/dart/")
        assert len(prov["source_sha256"]) == 64
        assert prov["files"]["21419t2011_jan01-09_excerpt.txt.gz"]["rows"] == len(
            _fixture_rows(FIXTURE_JAN)
        )
        assert prov["files"]["21419t2011_mar10-12_excerpt.txt.gz"]["rows"] == len(
            _fixture_rows(FIXTURE_MAR)
        )

    def test_standard_mode_grid_and_sentinels(self) -> None:
        """T==1 rows land on the grid; 9999.000 sentinels stay missing."""
        grid = tw.parse_dart_file(FIXTURE_JAN, "21419", 2011)
        assert grid.values.shape == (365 * 96,)
        rows = _fixture_rows(FIXTURE_JAN)
        assert all(r[6] == "1" for r in rows), "January excerpt is pure standard mode"
        sentinels = [r for r in rows if r[7] == "9999.000"]
        assert len(sentinels) == 4, "the real Jan 8 sentinel gap must be present"
        finite = [r for r in rows if r[7] != "9999.000"]
        # Independent recount: every finite standard row occupies its slot.
        assert int(np.sum(~np.isnan(grid.values))) == len(finite)
        first = finite[0]
        slot = tw._slot_of(
            dt.datetime(2011, 1, int(first[2]), int(first[3]), int(first[4]), tzinfo=dt.UTC),
            2011,
        )
        assert grid.values[slot] == pytest.approx(float(first[7]))
        assert not grid.from_event_mode.any(), "no event-mode rows -> no filled slots"

    def test_event_mode_fills_only_exact_grid_slots(self) -> None:
        """Off-grid T=2/3 rows are never used; exact-slot ones fill gaps only."""
        grid = tw.parse_dart_file(FIXTURE_MAR, "21419", 2011)
        rows = _fixture_rows(FIXTURE_MAR)
        standard_present: set[int] = set()
        event_on_grid: set[int] = set()
        n_event_rows = 0
        for r in rows:
            if r[7] == "9999.000":
                continue
            ts = dt.datetime(2011, 3, int(r[2]), int(r[3]), int(r[4]), int(r[5]), tzinfo=dt.UTC)
            on_grid = ts.second == 0 and ts.minute % 15 == 0
            if r[6] == "1":
                if on_grid:
                    standard_present.add(tw._slot_of(ts, 2011))
            else:
                n_event_rows += 1
                if on_grid:
                    event_on_grid.add(tw._slot_of(ts, 2011))
        expected_fill = event_on_grid - standard_present
        assert n_event_rows > 1000, "the Tohoku excerpt carries the event-mode burst"
        filled = set(np.flatnonzero(grid.from_event_mode).tolist())
        assert filled == expected_fill
        # The leakage guard: only a small on-grid subset of event rows is
        # ever consulted -- their sub-15-min structure never enters the grid.
        assert len(filled) < n_event_rows / 10


class TestDetideAndWindows:
    """Detiding and the gap rule on real tidal data."""

    def test_detide_reduces_rms_on_real_tidal_window(self) -> None:
        """OLS harmonic detiding must shrink a metre-scale tide to cm-scale."""
        grid = tw.parse_dart_file(FIXTURE_JAN, "21419", 2011)
        start = tw._slot_of(dt.datetime(2011, 1, 2, tzinfo=dt.UTC), 2011)
        raw = grid.values[start : start + tw.WINDOW_SAMPLES]
        assert not np.isnan(raw).any()
        residual = tw.extract_residual_window(grid.values, start)
        assert residual is not None
        assert residual.dtype == np.float32
        raw_rms = float(np.sqrt(np.mean((raw - raw.mean()) ** 2)))
        res_rms = float(np.sqrt(np.mean(residual.astype(np.float64) ** 2)))
        assert raw_rms > 0.2, "the real tide is decimetre-to-metre scale"
        assert res_rms < 0.02, "the residual must be cm-scale (detiding worked)"

    def test_window_with_long_gap_is_rejected(self) -> None:
        """The real 3-slot Jan 8 sentinel gap exceeds MAX_GAP_SLOTS=2."""
        grid = tw.parse_dart_file(FIXTURE_JAN, "21419", 2011)
        start = tw._slot_of(dt.datetime(2011, 1, 8, 12, 0, tzinfo=dt.UTC), 2011)
        covered = grid.values[start : start + tw.WINDOW_SAMPLES]
        assert int(np.isnan(covered).sum()) >= 3, "window must cover the sentinel gap"
        assert tw.extract_residual_window(grid.values, start) is None

    def test_out_of_range_window_is_rejected(self) -> None:
        grid = tw.parse_dart_file(FIXTURE_JAN, "21419", 2011)
        assert tw.extract_residual_window(grid.values, -1) is None
        assert tw.extract_residual_window(grid.values, grid.values.size - 10) is None

    def test_positive_window_recovers_tohoku_amplitude(self) -> None:
        """A real arrival window's peak residual matches the HazEL runupHt."""
        grid = tw.parse_dart_file(FIXTURE_MAR, "21419", 2011)
        anchor = tw._slot_of(TOHOKU_ARRIVAL_UTC, 2011)
        residual = tw.extract_residual_window(grid.values, anchor - 4 * tw.SLOTS_PER_HOUR)
        assert residual is not None, "event-mode fill must make the Tohoku window buildable"
        peak = float(np.max(np.abs(residual)))
        assert 0.3 < peak < 0.8, f"peak residual {peak:.3f} m vs runupHt {TOHOKU_RUNUP_HT_M} m"


class TestLabelResolution:
    """HazEL arrival-time resolution rules."""

    def test_arrival_day_rolls_into_next_month(self) -> None:
        origin = dt.datetime(2011, 3, 31, 23, 0, tzinfo=dt.UTC)
        resolved = tw._arrival_datetime(origin, {"arrDay": 1, "arrHour": 2, "arrMin": 30})
        assert resolved == dt.datetime(2011, 4, 1, 2, 30, tzinfo=dt.UTC)

    def test_arrival_far_from_origin_is_dropped(self) -> None:
        origin = dt.datetime(2011, 3, 11, 5, 46, tzinfo=dt.UTC)
        assert tw._arrival_datetime(origin, {"arrDay": 25, "arrHour": 0, "arrMin": 0}) is None

    def test_same_month_arrival_resolves(self) -> None:
        origin = dt.datetime(2011, 3, 11, 5, 46, tzinfo=dt.UTC)
        resolved = tw._arrival_datetime(origin, {"arrDay": 11, "arrHour": 7, "arrMin": 6})
        assert resolved == TOHOKU_ARRIVAL_UTC


class TestTemporalSplitWiring:
    """The module's split constants obey the anti-leakage contract."""

    def test_split_is_strictly_ordered(self) -> None:
        assert max(tw.SPLIT.train_years) < min(tw.SPLIT.val_years)
        assert max(tw.SPLIT.val_years) < min(tw.SPLIT.test_years)
        assert tw.SPLIT.test_years[-1] <= tw.EVENTS_MAX_YEAR

    def test_masks_route_window_years(self) -> None:
        years = np.array([2010, 2015, 2019, 2020, 2025])
        train, val, test = tw.SPLIT.masks(years)
        assert train.tolist() == [True, False, False, False, False]
        assert val.tolist() == [False, True, True, False, False]
        assert test.tolist() == [False, False, False, True, True]


class TestOperatingPointConsumption:
    """The checkpoint's ratified threshold governs the learned decision.

    Mirrors the solar-storm dual-rule operating-point tests: the
    validation-selected tau carried by the checkpoint is part of the
    deployed rule, so loading it must (a) validate it, (b) apply it to the
    learned path's ``tsunami_detected`` decision without touching the
    confidence estimate, and (c) leave the constructor threshold in charge
    for checkpoints that predate the convention.
    """

    def test_operating_point_drives_learned_decision(self, tmp_path: Path) -> None:
        """tau just below/above the emitted confidence must flip detection."""
        residual = _tohoku_residual()
        detector = TsunamiDetector(sampling_rate=1.0 / tw.SAMPLE_PERIOD_S)
        detector.load_neural_weights(
            str(_write_checkpoint(tmp_path / "op.pt", {"detection_threshold": 0.5}))
        )
        assert detector._operating_point == {"detection_threshold": 0.5}

        base = detector.predict_tsunami(residual)
        conf = float(base.confidence)
        assert 0.0 < conf < 1.0, "sigmoid + bounded resonance keeps confidence interior"

        detector._operating_point = {"detection_threshold": max(conf - 1e-6, 1e-9)}
        below = detector.predict_tsunami(residual)
        assert below.tsunami_detected is True
        assert below.confidence == pytest.approx(
            conf
        ), "the operating point changes the DECISION, never the confidence estimate"

        detector._operating_point = {"detection_threshold": conf + (1.0 - conf) / 2.0}
        above = detector.predict_tsunami(residual)
        assert above.tsunami_detected is False
        assert above.confidence == pytest.approx(conf)

        # Without an operating point the constructor threshold governs.
        detector._operating_point = None
        default = detector.predict_tsunami(residual)
        assert default.tsunami_detected is (conf > detector.detection_threshold)

    def test_operating_point_threshold_validated_on_load(self, tmp_path: Path) -> None:
        """A checkpoint carrying a nonsensical tau must refuse to load."""
        detector = TsunamiDetector(sampling_rate=1.0 / tw.SAMPLE_PERIOD_S)
        bad = _write_checkpoint(tmp_path / "bad_op.pt", {"detection_threshold": 1.5})
        with pytest.raises(ValueError, match=r"not a\s+probability"):
            detector.load_neural_weights(str(bad))
        zero = _write_checkpoint(tmp_path / "zero_op.pt", {"detection_threshold": 0.0})
        with pytest.raises(ValueError, match=r"not a\s+probability"):
            detector.load_neural_weights(str(zero))

    def test_checkpoint_without_operating_point_keeps_constructor_rule(
        self, tmp_path: Path
    ) -> None:
        """Explicit-path backward compat: pre-convention checkpoints load."""
        detector = TsunamiDetector(sampling_rate=1.0 / tw.SAMPLE_PERIOD_S)
        detector.load_neural_weights(str(_write_checkpoint(tmp_path / "old.pt", None)))
        assert detector._neural_trained is True
        assert detector._operating_point is None
        out = detector.predict_tsunami(_tohoku_residual())
        assert out.tsunami_detected is (out.confidence > detector.detection_threshold)


class TestMeasuredWaveHeight:
    """estimated_wave_height_m is a MEASUREMENT on both paths.

    The peak sea-level deviation from the median baseline is what the DART
    record itself shows; the NN wave-height head (held-out MAE ~4x the
    measurement's) is surfaced only as the diagnostic ``nn_wave_height_m``.
    """

    def test_learned_path_reports_measured_peak_deviation(self, tmp_path: Path) -> None:
        residual = _tohoku_residual()
        measured = float(np.max(np.abs(residual - np.median(residual))))

        detector = TsunamiDetector(sampling_rate=1.0 / tw.SAMPLE_PERIOD_S)
        detector.load_neural_weights(
            str(_write_checkpoint(tmp_path / "ckpt.pt", {"detection_threshold": 0.5}))
        )
        out = detector.predict_tsunami(residual)
        assert out.estimated_wave_height_m == pytest.approx(measured, rel=1e-6)
        assert out.nn_wave_height_m is not None, "NN estimate kept as a diagnostic"
        assert np.isfinite(out.nn_wave_height_m) and out.nn_wave_height_m >= 0.0

    def test_physics_path_reports_same_measurement_without_nn_diagnostic(self) -> None:
        residual = _tohoku_residual()
        measured = float(np.max(np.abs(residual - np.median(residual))))
        physics = TsunamiDetector(sampling_rate=1.0 / tw.SAMPLE_PERIOD_S)
        out = physics.predict_tsunami(residual)
        assert out.estimated_wave_height_m == pytest.approx(measured, rel=1e-6)
        assert out.nn_wave_height_m is None


class TestShippedCheckpointDifferential:
    """Learned vs physics through the public API on a real positive window."""

    @pytest.mark.skipif(
        not _SHIPPED.exists(),
        reason="no shipped tsunami_dart checkpoint yet (ship stage not merit-gated through)",
    )
    def test_learned_path_differs_from_physics_on_real_window(self) -> None:
        grid = tw.parse_dart_file(FIXTURE_MAR, "21419", 2011)
        anchor = tw._slot_of(TOHOKU_ARRIVAL_UTC, 2011)
        residual = tw.extract_residual_window(grid.values, anchor - 4 * tw.SLOTS_PER_HOUR)
        assert residual is not None

        physics_det = TsunamiDetector(sampling_rate=1.0 / tw.SAMPLE_PERIOD_S)
        learned_det = TsunamiDetector(sampling_rate=1.0 / tw.SAMPLE_PERIOD_S)
        learned_det.load_neural_weights()  # no path -> shipped default
        assert learned_det._neural_trained is True
        assert learned_det._feature_spec == tw.FEATURE_SPEC_VERSION
        assert physics_det._neural_trained is False

        physics_out = physics_det.predict_tsunami(residual)
        learned_out = learned_det.predict_tsunami(residual)
        assert np.isfinite(learned_out.confidence) and np.isfinite(physics_out.confidence)
        assert learned_out.confidence != pytest.approx(physics_out.confidence), (
            "the learned analyzer must actually be consulted (identical confidence would "
            "mean the physics fallback silently stayed in charge)"
        )

    @pytest.mark.skipif(
        not _SHIPPED.exists(),
        reason="no shipped tsunami_dart checkpoint yet (ship stage not merit-gated through)",
    )
    def test_shipped_payload_carries_input_contract(self) -> None:
        payload = torch.load(_SHIPPED, map_location="cpu", weights_only=True)
        assert payload["feature_spec"] == tw.FEATURE_SPEC_VERSION
        assert payload["window_samples"] == tw.WINDOW_SAMPLES
        assert payload["sampling_period_s"] == tw.SAMPLE_PERIOD_S
        assert payload["detide"] == tw.DETIDE_METHOD
        # A shipped checkpoint must carry the ratified deployed rule.
        assert 0.0 < float(payload["operating_point"]["detection_threshold"]) < 1.0
