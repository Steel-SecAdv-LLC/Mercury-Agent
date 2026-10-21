# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioural tests for the visual anomaly analyser: calibration, detection, localisation, classification, explanation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from scipy import ndimage

from omni_mercury_engine.ml.mercury_ml import roc_auc_score
from omni_mercury_engine.verifiers.three_state import ThreeState
from omni_mercury_engine.vision import (
    AnalyzerConfig,
    StreamingVisualMonitor,
    VisualAnomalyAnalyzer,
    degrade,
    features as features_module,
)
from omni_mercury_engine.vision.simulation import SIMULATED_TYPES, simulate_anomaly

from .conftest import inject, woven_surface

if TYPE_CHECKING:
    from pathlib import Path

#: Which taxonomy types are physically correct for each injected defect. A
#: dark flat fill is both an occluder and a local exposure drop.
EXPECTED_TYPES = {
    "occlusion": {"occlusion_or_missing", "exposure_anomaly"},
    "blur": {"blur_or_defocus"},
    "noise": {"noise_or_sensor_corruption"},
    "stain": {"color_shift"},
    "dark": {"exposure_anomaly"},
}


class TestCalibration:
    def test_conformal_false_alarm_rate_holds_across_calibration_splits(
        self, fitted_analyzer: VisualAnomalyAnalyzer
    ) -> None:
        # The guarantee is marginal over the calibration draw: average the
        # rank rule over many random splits of exchangeable normal images.
        scores = np.array([fitted_analyzer.score(woven_surface(s)) for s in range(600, 660)])
        alpha, n_cal = 0.2, 17
        rng = np.random.default_rng(0)
        rates = []
        for _ in range(400):
            order = rng.permutation(scores.size)
            cal, test = np.sort(scores[order[:n_cal]]), scores[order[n_cal:]]
            threshold = cal[int(np.ceil((n_cal + 1) * (1 - alpha))) - 1]
            rates.append(np.mean(test > threshold))
        assert float(np.mean(rates)) <= alpha + 0.02

    def test_recalibrate_sets_exactly_the_rank_rule_threshold(
        self, fitted_analyzer: VisualAnomalyAnalyzer
    ) -> None:
        analyzer = VisualAnomalyAnalyzer.load(_roundtrip(fitted_analyzer))
        images = [woven_surface(s) for s in range(700, 709)]
        analyzer.recalibrate(images)
        scores = np.sort([analyzer.score(im) for im in images])
        assert analyzer._image_threshold_raw == pytest.approx(scores[int(np.ceil(10 * 0.8)) - 1])

    def test_p_value_is_the_conformal_rank_formula(
        self, fitted_analyzer: VisualAnomalyAnalyzer
    ) -> None:
        image = woven_surface(710)
        report = fitted_analyzer.analyze(image)
        raw = fitted_analyzer.score(image)
        cal = fitted_analyzer._calibration_scores
        assert report.p_value == pytest.approx((1 + np.sum(cal >= raw)) / (cal.size + 1))

    def test_insufficient_calibration_withholds_instead_of_pretending(self) -> None:
        analyzer = VisualAnomalyAnalyzer(AnalyzerConfig(alpha=0.01, type_simulations=0))
        analyzer.fit([woven_surface(s) for s in range(12)])
        report = analyzer.analyze(woven_surface(99))
        assert not analyzer.calibration_sufficient
        assert report.state is ThreeState.UNAVAILABLE
        assert report.is_anomalous is None
        assert "cannot certify" in report.reason


class TestDetectionAndLocalisation:
    @pytest.mark.parametrize("kind", sorted(EXPECTED_TYPES))
    def test_every_defect_is_detected_localised_and_correctly_typed(
        self, fitted_analyzer: VisualAnomalyAnalyzer, kind: str
    ) -> None:
        for i in range(4):
            image, mask = inject(woven_surface(900 + i), kind, 50 + i)
            report = fitted_analyzer.analyze(image)
            assert report.state is ThreeState.GROUNDED
            assert report.is_anomalous
            region = report.regions[0]
            assert mask[region.peak[1], region.peak[0]], "peak must lie inside the defect"
            assert roc_auc_score(mask.ravel().astype(int), report.anomaly_map.ravel()) > 0.97
            assert region.evidence, "every region carries its evidence"
            if region.anomaly_type not in EXPECTED_TYPES[kind]:
                # The only acceptable alternative is an explicit open-set
                # rejection whose stated closest type is physically correct.
                rejection = " ".join(region.evidence)
                assert "no known type fits" in rejection
                assert any(f"closest: {t}" in rejection for t in EXPECTED_TYPES[kind])

    def test_region_box_tracks_the_defect_extent(
        self, fitted_analyzer: VisualAnomalyAnalyzer
    ) -> None:
        ious = []
        for i in range(6):
            image, mask = inject(woven_surface(950 + i), "noise", 70 + i)
            x0, y0, x1, y1 = fitted_analyzer.analyze(image).regions[0].box
            ys, xs = np.nonzero(mask)
            g = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
            inter = max(0, min(x1, g[2]) - max(x0, g[0])) * max(0, min(y1, g[3]) - max(y0, g[1]))
            ious.append(inter / ((x1 - x0) * (y1 - y0) + (g[2] - g[0]) * (g[3] - g[1]) - inter))
        assert float(np.mean(ious)) > 0.4

    def test_normal_images_produce_no_regions(self, fitted_analyzer: VisualAnomalyAnalyzer) -> None:
        reports = [fitted_analyzer.analyze(woven_surface(s)) for s in range(800, 806)]
        assert all(not r.regions for r in reports if not r.is_anomalous)

    def test_invalid_pixels_are_their_own_definitive_type(
        self, fitted_analyzer: VisualAnomalyAnalyzer
    ) -> None:
        image = woven_surface(960).astype(np.float64)
        image[30:50, 30:50] = np.nan
        report = fitted_analyzer.analyze(image)
        assert report.provenance["nonfinite_pixels"] == 400
        assert report.regions[0].anomaly_type == "invalid_pixels"


class TestClassification:
    def test_domain_adapted_types_recognise_unseen_synthetic_anomalies(
        self, fitted_analyzer: VisualAnomalyAnalyzer
    ) -> None:
        rng = np.random.default_rng(123)
        correct = total = 0
        for anomaly_type in SIMULATED_TYPES:
            for i in range(5):
                image, _ = simulate_anomaly(woven_surface(1200 + i), anomaly_type, rng)
                report = fitted_analyzer.analyze(image)
                correct += bool(report.regions) and report.regions[0].anomaly_type == anomaly_type
                total += 1
        assert correct / total >= 0.85

    def test_open_set_rejects_what_no_prototype_explains(self) -> None:
        analyzer = VisualAnomalyAnalyzer(
            AnalyzerConfig(alpha=0.2, type_simulations=6, open_set_factor=1e-6)
        ).fit([woven_surface(s) for s in range(30)])
        image, _ = inject(woven_surface(970), "noise", 3)
        region = analyzer.analyze(image).regions[0]
        assert region.anomaly_type in ("unclassified_deviation", "splice_candidate")
        assert any("no known type fits" in e for e in region.evidence)

    def test_rule_fallback_without_simulation(self) -> None:
        analyzer = VisualAnomalyAnalyzer(AnalyzerConfig(alpha=0.2, type_simulations=0))
        analyzer.fit([woven_surface(s) for s in range(30)])
        image, _ = inject(woven_surface(971), "stain", 4)
        assert analyzer.analyze(image).regions[0].anomaly_type == "color_shift"

    def test_learned_types_from_labelled_examples(
        self, fitted_analyzer: VisualAnomalyAnalyzer
    ) -> None:
        analyzer = VisualAnomalyAnalyzer.load(_roundtrip(fitted_analyzer))
        examples = []
        for i in range(4):
            for kind, label in (("stain", "dye_spill"), ("blur", "lens_smear")):
                image, mask = inject(woven_surface(1300 + i), kind, 90 + i)
                examples.append((image, mask, label))
        accuracy = analyzer.learn_anomaly_types(examples)
        assert accuracy is not None and accuracy >= 0.75
        image, _ = inject(woven_surface(1400), "stain", 99)
        region = analyzer.analyze(image).regions[0]
        assert region.learned_type == "dye_spill"
        assert region.learned_confidence is not None and region.learned_confidence > 0.5


class TestEnvironment:
    def test_global_lighting_change_is_not_a_content_anomaly(
        self, fitted_analyzer: VisualAnomalyAnalyzer
    ) -> None:
        dimmer = [fitted_analyzer.analyze(woven_surface(s) * 0.8) for s in range(1500, 1506)]
        assert np.mean([bool(r.is_anomalous) for r in dimmer]) <= 0.34

    def test_out_of_envelope_capture_withholds_the_decision(
        self, fitted_analyzer: VisualAnomalyAnalyzer
    ) -> None:
        dark = degrade(woven_surface(1510), "underexpose", 0.95)
        report = fitted_analyzer.analyze(dark)
        assert report.state is ThreeState.UNAVAILABLE
        assert report.is_anomalous is None
        assert not report.environment_in_envelope
        assert "outside the calibrated envelope" in report.reason

    def test_report_policy_decides_but_flags_conditions(
        self, fitted_analyzer: VisualAnomalyAnalyzer
    ) -> None:
        analyzer = VisualAnomalyAnalyzer.load(_roundtrip(fitted_analyzer))
        analyzer.config.environment_policy = "report"
        report = analyzer.analyze(degrade(woven_surface(1511), "underexpose", 0.95))
        assert report.state is ThreeState.GROUNDED
        assert any("unusual capture conditions" in e for e in report.explanation)

    def test_layout_mismatch_is_refused_with_the_reason(
        self, fitted_analyzer: VisualAnomalyAnalyzer
    ) -> None:
        with pytest.raises(ValueError, match="does not match the normal model"):
            fitted_analyzer.analyze(woven_surface(1).mean(axis=2))


def _grain(seed: int, size: int = 96) -> np.ndarray[Any, Any]:
    """An aperiodic, isotropic grain (leather-like): no translation symmetry at all."""
    rng = np.random.default_rng(seed)
    grain = ndimage.gaussian_filter(rng.normal(size=(size, size)), 2.0)
    lum = 0.45 + 0.09 * grain / grain.std()
    return np.asarray(np.clip(lum[..., None] * np.array([1.15, 0.95, 0.8]), 0, 1))


@pytest.fixture(scope="module")
def grain_analyzer() -> VisualAnomalyAnalyzer:
    """An analyser fitted on the aperiodic grain domain."""
    config = AnalyzerConfig(alpha=0.2, type_simulations=0, calibration_fraction=0.5)
    return VisualAnomalyAnalyzer(config).fit([_grain(s) for s in range(24)])


class TestLatticeRelevance:
    """The lattice group is used only where the normal data has a lattice."""

    def test_a_periodic_domain_keeps_the_lattice_group(
        self, fitted_analyzer: VisualAnomalyAnalyzer
    ) -> None:
        assert fitted_analyzer.lattice_active
        periodicity = fitted_analyzer.lattice_periodicity
        assert periodicity is not None and periodicity < 0.1

    def test_an_aperiodic_domain_switches_it_off_and_still_detects(
        self, grain_analyzer: VisualAnomalyAnalyzer
    ) -> None:
        assert not grain_analyzer.lattice_active
        periodicity = grain_analyzer.lattice_periodicity
        assert periodicity is not None and periodicity > 0.4
        image, _ = inject(_grain(500), "noise", 3)
        report = grain_analyzer.analyze(image)
        assert report.is_anomalous
        deviation = report.regions[0].feature_deviation
        assert deviation["lattice_any_axis"] == 0.0
        assert deviation["lattice_both_axes"] == 0.0

    def test_the_decision_survives_save_and_load(
        self, grain_analyzer: VisualAnomalyAnalyzer, tmp_path: Path
    ) -> None:
        loaded = VisualAnomalyAnalyzer.load(grain_analyzer.save(tmp_path / "grain"))
        assert loaded.lattice_active is False
        assert loaded.lattice_periodicity == grain_analyzer.lattice_periodicity
        image, _ = inject(_grain(501), "stain", 4)
        assert loaded.analyze(image).image_score == pytest.approx(
            grain_analyzer.analyze(image).image_score
        )

    def test_an_aperiodic_domain_never_computes_the_lattice_after_fit(
        self, grain_analyzer: VisualAnomalyAnalyzer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _refuse(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("lattice computed for a domain fitted as aperiodic")

        monkeypatch.setattr(features_module, "lattice_structure", _refuse)
        image, _ = inject(_grain(502), "noise", 5)
        assert grain_analyzer.analyze(image).is_anomalous

    def test_a_periodic_domain_still_computes_it(
        self, fitted_analyzer: VisualAnomalyAnalyzer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _refuse(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("computed")

        monkeypatch.setattr(features_module, "lattice_structure", _refuse)
        with pytest.raises(AssertionError, match="computed"):
            fitted_analyzer.analyze(woven_surface(1700))

    def test_switched_off_features_take_no_part_in_the_gaussian(
        self, grain_analyzer: VisualAnomalyAnalyzer
    ) -> None:
        off = np.array([g == "lattice" for g in grain_analyzer._groups])
        precision = grain_analyzer._gauss_precision
        assert off.any()
        assert not precision[off].any() and not precision[:, off].any()
        assert not grain_analyzer._gauss_mean[off].any()
        # The active block is a proper (positive-definite) precision matrix.
        active_block = precision[np.ix_(~off, ~off)]
        assert float(np.linalg.eigvalsh(active_block).min()) > 0.0


class TestEvidenceChannelsAndPersistence:
    def test_report_is_json_serialisable(self, fitted_analyzer: VisualAnomalyAnalyzer) -> None:
        image, _ = inject(woven_surface(1600), "blur", 5)
        payload = fitted_analyzer.analyze(image).to_dict(include_map=True)
        restored = json.loads(json.dumps(payload))
        assert restored["state"] == "grounded"
        assert len(restored["anomaly_map"]) == 96

    def test_save_load_round_trip_reproduces_reports(
        self, fitted_analyzer: VisualAnomalyAnalyzer, tmp_path: Path
    ) -> None:
        path = fitted_analyzer.save(tmp_path / "normal_model")
        loaded = VisualAnomalyAnalyzer.load(path)
        image, _ = inject(woven_surface(1610), "noise", 6)
        a, b = fitted_analyzer.analyze(image), loaded.analyze(image)
        assert a.image_score == pytest.approx(b.image_score)
        assert a.p_value == b.p_value
        assert [r.anomaly_type for r in a.regions] == [r.anomaly_type for r in b.regions]

    def test_load_rejects_a_foreign_schema(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.npz"
        np.savez(path, meta=np.array(json.dumps({"schema": "other/v9"})))
        with pytest.raises(ValueError, match="unsupported visual model schema"):
            VisualAnomalyAnalyzer.load(path)

    def test_deep_evidence_channel_is_fused_and_calibrated(self) -> None:
        def provider(batch: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            # A stand-in "deep" detector that responds to bright red pixels.
            return np.asarray((batch[..., 0] - batch[..., 2] > 0.6).astype(np.float32))

        analyzer = VisualAnomalyAnalyzer(AnalyzerConfig(alpha=0.2, type_simulations=0))
        analyzer.attach_deep_evidence(provider, name="red_detector")
        analyzer.fit([woven_surface(s) for s in range(30)])
        image = woven_surface(1620)
        image[40:56, 40:56] = [1.0, 0.0, 0.0]
        report = analyzer.analyze(image)
        assert report.deep_evidence is not None and report.deep_evidence["name"] == "red_detector"
        assert "deep" in analyzer._component_norm
        with pytest.raises(RuntimeError, match="cannot be saved"):
            analyzer.save("unused")

    def test_deep_evidence_must_be_attached_before_fit(
        self, fitted_analyzer: VisualAnomalyAnalyzer
    ) -> None:
        with pytest.raises(RuntimeError, match="before fit"):
            fitted_analyzer.attach_deep_evidence(lambda b: b[..., 0])

    def test_unfitted_analyser_refuses(self) -> None:
        with pytest.raises(RuntimeError, match="not fitted"):
            VisualAnomalyAnalyzer().analyze(woven_surface(0))


class TestRealTime:
    def test_batch_equals_individual_analysis(self, fitted_analyzer: VisualAnomalyAnalyzer) -> None:
        images = [inject(woven_surface(1700 + i), "stain", i)[0] for i in range(3)]
        batch = fitted_analyzer.analyze_batch(images)
        single = [fitted_analyzer.analyze(im) for im in images]
        assert [r.image_score for r in batch] == [r.image_score for r in single]

    def test_stream_alerts_only_on_persistent_regions(
        self, fitted_analyzer: VisualAnomalyAnalyzer
    ) -> None:
        monitor = StreamingVisualMonitor(
            VisualAnomalyAnalyzer.load(_roundtrip(fitted_analyzer)), persistence=2
        )
        defective, _ = inject(woven_surface(1800), "noise", 9)
        events = [
            monitor.process(woven_surface(1801)),
            monitor.process(defective),
            monitor.process(defective),
        ]
        assert not events[1].persistent_alert
        assert events[2].persistent_alert
        assert events[2].latency_p95_ms >= events[2].latency_p50_ms > 0

    def test_stream_sheds_forensics_when_over_budget(
        self, fitted_analyzer: VisualAnomalyAnalyzer
    ) -> None:
        monitor = StreamingVisualMonitor(
            VisualAnomalyAnalyzer.load(_roundtrip(fitted_analyzer)), latency_budget_ms=0.001
        )
        for s in range(3):
            monitor.process(woven_surface(1900 + s))
        assert monitor.forensics_shed
        assert not monitor.analyzer.config.forensics

    def test_fast_profile_is_faster_and_still_detects(
        self, fitted_analyzer: VisualAnomalyAnalyzer
    ) -> None:
        normals = [woven_surface(s) for s in range(30)]
        fast = VisualAnomalyAnalyzer(AnalyzerConfig.fast(alpha=0.2, type_simulations=0)).fit(
            normals
        )
        image, mask = inject(woven_surface(1950), "noise", 1)
        report = fast.analyze(image)
        assert report.is_anomalous
        # The real-time profile localises to patch precision: its region must
        # overlap the defect.
        x0, y0, x1, y1 = report.regions[0].box
        assert mask[y0:y1, x0:x1].any()
        # The speed-up comes from doing less work: one coarse pyramid scale and
        # a sparser patch grid. Assert that, deterministically. Wall-clock
        # medians of a few calls on a shared CI runner flip under parallel
        # load since the default profile stopped computing the lattice group
        # on aperiodic textures (39a7fa9e) and the gap narrowed; latency itself
        # is measured on an idle machine by benchmarks/vision_latency.py.
        fitted_analyzer.analyze(image)
        full = fitted_analyzer._last_features
        fast_feats = fast._last_features
        fast_cells = fast_feats.values.shape[0] * fast_feats.values.shape[1]
        full_cells = full.values.shape[0] * full.values.shape[1]
        assert set(fast_feats.scales) == {0.5} and len(set(full.scales)) > 1
        assert fast_cells * fast_feats.values.shape[2] < full_cells * full.values.shape[2]
        assert fast_cells < full_cells


def _roundtrip(analyzer: VisualAnomalyAnalyzer) -> Any:
    import tempfile
    from pathlib import Path

    directory = Path(tempfile.mkdtemp())
    return analyzer.save(directory / "copy")
