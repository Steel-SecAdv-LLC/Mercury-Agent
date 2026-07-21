# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the unified :mod:`omni_mercury_engine.biometric` detector.

Covers the symbols defined directly in ``biometric/__init__.py``:
the :class:`BiometricAnomalyDetector` (construction, enroll, verify,
detect_anomaly, and the multi-modal fusion logic) together with the
enums and result dataclasses (:class:`BiometricModality`,
:class:`FusionStrategy`, :class:`BiometricEnrollment`,
:class:`BiometricVerificationResult`, :class:`BiometricAnomalyResult`).

All tests are deterministic (seeded numpy RNG, no torch, no network, no
wall-clock reliance) and assert on behavior observed from the real
recognizer backends. Iris images are kept tiny (16x16) because iris
segmentation is an O(n^3) search; fingerprint and voice paths are cheap.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.biometric import (
    BiometricAnomalyDetector,
    BiometricAnomalyResult,
    BiometricEnrollment,
    BiometricModality,
    BiometricVerificationResult,
    FusionStrategy,
)

SEED = 20250721


def _rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def _iris_image(seed: int = SEED) -> np.ndarray:
    # 16x16 keeps the integro-differential search fast (~0.8s).
    return _rng(seed).integers(0, 256, size=(16, 16)).astype(np.float64)


def _fingerprint_image(seed: int = SEED) -> np.ndarray:
    return _rng(seed).integers(0, 256, size=(64, 64)).astype(np.float64)


def _voice_sample(seed: int = SEED, n: int = 8192) -> np.ndarray:
    return _rng(seed).standard_normal(n).astype(np.float64)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class TestEnums:
    def test_modality_members(self) -> None:
        names = {m.name for m in BiometricModality}
        assert names == {"IRIS", "FINGERPRINT", "VOICE", "FACE"}

    def test_modality_values_distinct(self) -> None:
        values = [m.value for m in BiometricModality]
        assert len(set(values)) == len(values)

    def test_fusion_strategy_members(self) -> None:
        names = {s.name for s in FusionStrategy}
        assert names == {"SCORE_LEVEL", "DECISION_LEVEL", "QUALITY_WEIGHTED"}

    def test_fusion_strategy_values_distinct(self) -> None:
        values = [s.value for s in FusionStrategy]
        assert len(set(values)) == len(values)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
class TestDataclasses:
    def test_enrollment_defaults(self) -> None:
        enr = BiometricEnrollment(identity="alice")
        assert enr.identity == "alice"
        assert enr.iris_features is None
        assert enr.fingerprint_features is None
        assert enr.voice_features is None
        assert enr.enrollment_timestamp == 0.0
        assert enr.metadata == {}

    def test_enrollment_metadata_factory_independent(self) -> None:
        a = BiometricEnrollment(identity="a")
        b = BiometricEnrollment(identity="b")
        a.metadata["k"] = 1
        assert b.metadata == {}

    def test_verification_result_fields(self) -> None:
        res = BiometricVerificationResult(
            identity="bob",
            is_verified=True,
            confidence=0.9,
            modality_results={"voice": {}},
            liveness_results={},
            fusion_method="SCORE_LEVEL",
        )
        assert res.is_verified is True
        assert res.confidence == 0.9
        assert res.fusion_method == "SCORE_LEVEL"
        assert res.details == {}

    def test_anomaly_result_fields(self) -> None:
        res = BiometricAnomalyResult(
            is_anomaly=False,
            anomaly_score=0.1,
            anomaly_type=None,
            modality_scores={"voice": 0.8},
            liveness_scores={"voice": 0.7},
        )
        assert res.is_anomaly is False
        assert res.anomaly_type is None
        assert res.details == {}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
class TestConstruction:
    def test_default_uses_all_supported_modalities(self) -> None:
        det = BiometricAnomalyDetector()
        assert det._modalities == ["iris", "fingerprint", "voice"]
        assert set(det._recognizers) == {"iris", "fingerprint", "voice"}

    def test_supported_modalities_constant(self) -> None:
        assert BiometricAnomalyDetector.SUPPORTED_MODALITIES == [
            "iris",
            "fingerprint",
            "voice",
        ]

    def test_subset_of_modalities(self) -> None:
        det = BiometricAnomalyDetector(modalities=["voice"])
        assert det._modalities == ["voice"]
        assert set(det._recognizers) == {"voice"}

    def test_modalities_are_lowercased_and_filtered(self) -> None:
        det = BiometricAnomalyDetector(modalities=["IRIS", "Voice", "FACE", "bogus"])
        # FACE and bogus are not supported and are dropped; case is normalized.
        assert det._modalities == ["iris", "voice"]
        assert set(det._recognizers) == {"iris", "voice"}

    def test_empty_after_filtering(self) -> None:
        det = BiometricAnomalyDetector(modalities=["face", "unknown"])
        assert det._modalities == []
        assert det._recognizers == {}

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("score_level", FusionStrategy.SCORE_LEVEL),
            ("decision_level", FusionStrategy.DECISION_LEVEL),
            ("quality_weighted", FusionStrategy.QUALITY_WEIGHTED),
        ],
    )
    def test_fusion_strategy_mapping(self, name: str, expected: FusionStrategy) -> None:
        det = BiometricAnomalyDetector(modalities=[], fusion_strategy=name)
        assert det._fusion_strategy is expected

    def test_unknown_fusion_strategy_defaults_to_quality_weighted(self) -> None:
        det = BiometricAnomalyDetector(modalities=[], fusion_strategy="nonsense")
        assert det._fusion_strategy is FusionStrategy.QUALITY_WEIGHTED

    def test_liveness_and_threshold_stored(self) -> None:
        det = BiometricAnomalyDetector(
            modalities=[], liveness_required=False, anomaly_threshold=0.75
        )
        assert det._liveness_required is False
        assert det._anomaly_threshold == 0.75


# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------
class TestEnroll:
    def test_enroll_voice_sets_only_voice_features(self) -> None:
        det = BiometricAnomalyDetector(modalities=["voice"], liveness_required=False)
        enr = det.enroll("u1", voice_sample=_voice_sample(), tag="demo")
        assert isinstance(enr, BiometricEnrollment)
        assert enr.identity == "u1"
        assert enr.voice_features is not None
        assert enr.iris_features is None
        assert enr.fingerprint_features is None
        assert enr.voice_features.quality_score >= 0.0
        # kwargs are captured as metadata; timestamp is a positive epoch value.
        assert enr.metadata == {"tag": "demo"}
        assert enr.enrollment_timestamp > 0.0

    def test_enroll_stores_and_returns_same_object(self) -> None:
        det = BiometricAnomalyDetector(modalities=["voice"], liveness_required=False)
        enr = det.enroll("u2", voice_sample=_voice_sample())
        assert det._enrollments["u2"] is enr

    def test_enroll_fingerprint_sets_fingerprint_features(self) -> None:
        det = BiometricAnomalyDetector(modalities=["fingerprint"], liveness_required=False)
        enr = det.enroll("u3", fingerprint_image=_fingerprint_image())
        assert enr.fingerprint_features is not None
        assert enr.voice_features is None
        assert 0.0 <= enr.fingerprint_features.overall_quality <= 1.0

    def test_enroll_without_samples_leaves_features_none(self) -> None:
        det = BiometricAnomalyDetector(liveness_required=False)
        enr = det.enroll("empty")
        assert enr.iris_features is None
        assert enr.fingerprint_features is None
        assert enr.voice_features is None

    def test_enroll_ignores_sample_for_unconfigured_modality(self) -> None:
        # Detector only handles voice; an iris image must be ignored.
        det = BiometricAnomalyDetector(modalities=["voice"], liveness_required=False)
        enr = det.enroll("u4", iris_image=_iris_image(), voice_sample=_voice_sample())
        assert enr.iris_features is None
        assert enr.voice_features is not None


# ---------------------------------------------------------------------------
# verify() error / trivial paths
# ---------------------------------------------------------------------------
class TestVerifyGuards:
    def test_unknown_identity_returns_error_result(self) -> None:
        det = BiometricAnomalyDetector(modalities=["voice"], fusion_strategy="score_level")
        res = det.verify("ghost", voice_sample=_voice_sample())
        assert isinstance(res, BiometricVerificationResult)
        assert res.is_verified is False
        assert res.confidence == 0.0
        assert res.modality_results == {"error": "Identity not enrolled"}
        assert res.liveness_results == {}
        assert res.fusion_method == "SCORE_LEVEL"

    def test_enrolled_but_no_probe_gives_empty_no_match(self) -> None:
        det = BiometricAnomalyDetector(modalities=["voice"], liveness_required=False)
        det.enroll("u", voice_sample=_voice_sample())
        res = det.verify("u")
        assert res.is_verified is False
        assert res.confidence == 0.0
        assert res.modality_results == {}

    def test_probe_skipped_when_enrollment_feature_missing(self) -> None:
        # Enroll voice only, then also probe fingerprint: the fingerprint
        # branch is skipped because enrollment has no fingerprint features.
        det = BiometricAnomalyDetector(modalities=["voice", "fingerprint"], liveness_required=False)
        det.enroll("u", voice_sample=_voice_sample())
        res = det.verify(
            "u",
            voice_sample=_voice_sample(seed=SEED + 1),
            fingerprint_image=_fingerprint_image(),
        )
        assert "voice" in res.modality_results
        assert "fingerprint" not in res.modality_results


# ---------------------------------------------------------------------------
# verify() modality paths
# ---------------------------------------------------------------------------
class TestVerifyVoice:
    def test_voice_quality_weighted_confidence_equals_similarity(self) -> None:
        det = BiometricAnomalyDetector(modalities=["voice"], liveness_required=False)
        det.enroll("u", voice_sample=_voice_sample())
        res = det.verify("u", voice_sample=_voice_sample(seed=SEED + 2))
        voice = res.modality_results["voice"]
        assert set(voice) == {"similarity_score", "is_match", "embedding_distance"}
        # Single-modality quality-weighted fusion: weighted score reduces to
        # the similarity itself, and no liveness is produced.
        assert res.confidence == pytest.approx(voice["similarity_score"])
        assert res.is_verified is (res.confidence >= 0.5)
        assert res.liveness_results == {}
        assert res.fusion_method == "QUALITY_WEIGHTED"

    def test_voice_score_level_confidence_equals_similarity(self) -> None:
        det = BiometricAnomalyDetector(
            modalities=["voice"], fusion_strategy="score_level", liveness_required=False
        )
        det.enroll("u", voice_sample=_voice_sample())
        res = det.verify("u", voice_sample=_voice_sample())
        assert res.confidence == pytest.approx(res.modality_results["voice"]["similarity_score"])
        assert res.fusion_method == "SCORE_LEVEL"

    def test_voice_verify_with_liveness_populates_liveness_results(self) -> None:
        det = BiometricAnomalyDetector(modalities=["voice"], liveness_required=True)
        det.enroll("u", voice_sample=_voice_sample())
        res = det.verify("u", voice_sample=_voice_sample(seed=SEED + 3))
        assert "voice" in res.modality_results
        assert set(res.liveness_results["voice"]) == {"is_live", "confidence"}
        # is_live originates from numpy comparisons, so it may be a numpy bool.
        assert res.liveness_results["voice"]["is_live"] in (True, False)
        assert 0.0 <= res.confidence <= 1.0


class TestVerifyFingerprint:
    def test_fingerprint_result_shape(self) -> None:
        det = BiometricAnomalyDetector(modalities=["fingerprint"], liveness_required=False)
        det.enroll("u", fingerprint_image=_fingerprint_image())
        res = det.verify("u", fingerprint_image=_fingerprint_image(seed=SEED + 4))
        fp = res.modality_results["fingerprint"]
        assert set(fp) == {"match_score", "is_match", "matched_minutiae"}
        assert isinstance(fp["is_match"], bool)
        assert isinstance(fp["matched_minutiae"], int)
        assert fp["match_score"] >= 0.0

    def test_fingerprint_verify_with_liveness_populates_liveness_results(self) -> None:
        det = BiometricAnomalyDetector(modalities=["fingerprint"], liveness_required=True)
        det.enroll("u", fingerprint_image=_fingerprint_image())
        res = det.verify("u", fingerprint_image=_fingerprint_image(seed=SEED + 8))
        assert "fingerprint" in res.modality_results
        assert set(res.liveness_results["fingerprint"]) == {"is_live", "confidence"}


class TestVerifyMultiModal:
    def test_voice_and_fingerprint_fuse_two_scores(self) -> None:
        det = BiometricAnomalyDetector(modalities=["voice", "fingerprint"], liveness_required=False)
        det.enroll(
            "u",
            voice_sample=_voice_sample(),
            fingerprint_image=_fingerprint_image(),
        )
        res = det.verify(
            "u",
            voice_sample=_voice_sample(seed=SEED + 5),
            fingerprint_image=_fingerprint_image(seed=SEED + 6),
        )
        assert set(res.modality_results) == {"voice", "fingerprint"}
        assert res.liveness_results == {}
        assert 0.0 <= res.confidence <= 1.0
        assert res.is_verified is (res.confidence >= 0.5)


class TestVerifyIris:
    def test_iris_enroll_and_verify(self) -> None:
        det = BiometricAnomalyDetector(modalities=["iris"], liveness_required=False)
        enr = det.enroll("u", iris_image=_iris_image())
        assert enr.iris_features is not None
        res = det.verify("u", iris_image=_iris_image(seed=SEED + 7))
        iris = res.modality_results["iris"]
        assert set(iris) == {"match_score", "is_match", "hamming_distance"}
        assert isinstance(iris["is_match"], bool)
        assert 0.0 <= iris["hamming_distance"] <= 1.0

    def test_iris_verify_with_liveness_gate_fails_single_image(self) -> None:
        # Liveness required but only one image is available, so pupil dynamics
        # cannot be established: the liveness gate forces a non-verification.
        det = BiometricAnomalyDetector(modalities=["iris"], liveness_required=True)
        det.enroll("u", iris_image=_iris_image())
        res = det.verify("u", iris_image=_iris_image(seed=SEED + 9))
        assert set(res.liveness_results["iris"]) == {"is_live", "confidence"}
        assert bool(res.liveness_results["iris"]["is_live"]) is False
        assert res.is_verified is False
        assert res.confidence == 0.0


# ---------------------------------------------------------------------------
# detect_anomaly()
# ---------------------------------------------------------------------------
class TestDetectAnomaly:
    def test_empty_inputs_flag_presentation_attack(self) -> None:
        det = BiometricAnomalyDetector(modalities=["fingerprint"])
        res = det.detect_anomaly()
        assert isinstance(res, BiometricAnomalyResult)
        assert bool(res.is_anomaly) is True
        assert res.anomaly_score == pytest.approx(1.0)
        assert res.anomaly_type == "presentation_attack"
        assert res.modality_scores == {}
        assert res.liveness_scores == {}

    def test_fingerprint_below_threshold_not_anomaly(self) -> None:
        det = BiometricAnomalyDetector(modalities=["fingerprint"], anomaly_threshold=0.5)
        res = det.detect_anomaly(fingerprint_image=_fingerprint_image(seed=1234))
        # score ~0.44 < 0.5 -> not anomalous, so no type is assigned.
        assert bool(res.is_anomaly) is False
        assert res.anomaly_type is None
        assert "fingerprint" in res.modality_scores
        assert "fingerprint" in res.liveness_scores
        assert "minutiae_count" in res.details["fingerprint"]

    def test_fingerprint_low_threshold_poor_quality(self) -> None:
        det = BiometricAnomalyDetector(modalities=["fingerprint"], anomaly_threshold=0.3)
        # A genuinely low-contrast image has few resolvable ridges, so the
        # minutiae extractor (fixed to be dtype-safe) recovers a low-quality
        # score while liveness still passes -> the poor_quality branch. (A
        # high-entropy random image now scores higher quality and lands in
        # suspicious_pattern instead -- see test_fingerprint_random_suspicious.)
        low_contrast = np.clip(128.0 + 2.0 * _rng(3).standard_normal((64, 64)), 0.0, 255.0)
        res = det.detect_anomaly(fingerprint_image=low_contrast)
        # quality < 0.5 but liveness >= 0.5 -> classified as poor_quality.
        assert bool(res.is_anomaly) is True
        assert res.anomaly_type == "poor_quality"

    def test_fingerprint_random_suspicious(self) -> None:
        # A high-entropy random fingerprint yields detectable minutiae (quality
        # >= 0.5) yet still trips the low anomaly threshold, so it is classified
        # as suspicious_pattern rather than poor_quality.
        det = BiometricAnomalyDetector(modalities=["fingerprint"], anomaly_threshold=0.3)
        res = det.detect_anomaly(fingerprint_image=_fingerprint_image(seed=1234))
        assert bool(res.is_anomaly) is True
        assert res.anomaly_type == "suspicious_pattern"

    def test_ridge_pattern_suspicious(self) -> None:
        # Deterministic sinusoidal ridge field: high quality AND high liveness,
        # yet still above the (low) threshold -> suspicious_pattern branch.
        xx = np.mgrid[0:96, 0:96][1]
        ridge = (128 + 100 * np.sin(2 * np.pi * xx / 8.0)).astype(np.float64)
        det = BiometricAnomalyDetector(modalities=["fingerprint"], anomaly_threshold=0.3)
        res = det.detect_anomaly(fingerprint_image=ridge)
        assert bool(res.is_anomaly) is True
        assert res.anomaly_type == "suspicious_pattern"

    def test_voice_anomaly_reports_scores(self) -> None:
        det = BiometricAnomalyDetector(modalities=["voice"], anomaly_threshold=0.3)
        res = det.detect_anomaly(voice_sample=_voice_sample(seed=1234))
        assert "voice" in res.modality_scores
        assert "voice" in res.liveness_scores
        assert "duration" in res.details["voice"]
        assert 0.0 <= res.anomaly_score <= 1.0

    def test_iris_anomaly_single_image_is_presentation_attack(self) -> None:
        # A single iris image cannot establish pupil dynamics, so liveness
        # confidence is 0.0 -> presentation_attack.
        det = BiometricAnomalyDetector(modalities=["iris"])
        res = det.detect_anomaly(iris_image=_iris_image())
        assert res.liveness_scores["iris"] == pytest.approx(0.0)
        assert bool(res.is_anomaly) is True
        assert res.anomaly_type == "presentation_attack"
        assert "pupil_response" in res.details["iris"]


# ---------------------------------------------------------------------------
# _fuse_results() — exhaustive, fast, deterministic branch coverage
# ---------------------------------------------------------------------------
class TestFuseResults:
    @staticmethod
    def _detector(strategy: str, liveness: bool = True) -> BiometricAnomalyDetector:
        # Empty modality list -> no recognizers built, so fusion is isolated.
        return BiometricAnomalyDetector(
            modalities=[], fusion_strategy=strategy, liveness_required=liveness
        )

    def test_empty_scores(self) -> None:
        det = self._detector("score_level")
        assert det._fuse_results([], {}, {}) == (False, 0.0)

    def test_liveness_gate_blocks_when_not_live(self) -> None:
        det = self._detector("score_level", liveness=True)
        verified, conf = det._fuse_results(
            [(0.9, 1.0)], {"voice": {"is_match": True}}, {"voice": {"is_live": False}}
        )
        assert (verified, conf) == (False, 0.0)

    def test_liveness_gate_passes_when_live(self) -> None:
        det = self._detector("score_level", liveness=True)
        verified, conf = det._fuse_results(
            [(0.9, 1.0)], {"voice": {"is_match": True}}, {"voice": {"is_live": True}}
        )
        assert verified is True
        assert conf == pytest.approx(0.9)

    def test_liveness_not_required_ignores_dead_liveness(self) -> None:
        det = self._detector("score_level", liveness=False)
        verified, conf = det._fuse_results(
            [(0.8, 1.0)], {"voice": {}}, {"voice": {"is_live": False}}
        )
        assert verified is True
        assert conf == pytest.approx(0.8)

    def test_score_level_average_above_threshold(self) -> None:
        det = self._detector("score_level")
        verified, conf = det._fuse_results([(0.8, 1.0), (0.6, 1.0)], {}, {})
        assert verified is True
        assert conf == pytest.approx(0.7)

    def test_score_level_average_below_threshold(self) -> None:
        det = self._detector("score_level")
        verified, conf = det._fuse_results([(0.2, 1.0), (0.4, 1.0)], {}, {})
        assert verified is False
        assert conf == pytest.approx(0.3)

    def test_decision_level_majority_match(self) -> None:
        det = self._detector("decision_level")
        modality_results = {
            "a": {"is_match": True},
            "b": {"is_match": True},
            "c": {"is_match": False},
        }
        verified, conf = det._fuse_results([(0.9, 1.0)], modality_results, {})
        assert verified is True
        assert conf == pytest.approx(2 / 3)

    def test_decision_level_minority_match(self) -> None:
        det = self._detector("decision_level")
        modality_results = {
            "a": {"is_match": True},
            "b": {"is_match": False},
            "c": {"is_match": False},
        }
        verified, conf = det._fuse_results([(0.9, 1.0)], modality_results, {})
        assert verified is False
        assert conf == pytest.approx(1 / 3)

    def test_quality_weighted_above_threshold(self) -> None:
        det = self._detector("quality_weighted")
        # (0.8*2 + 0.2*1) / 3 = 0.6
        verified, conf = det._fuse_results([(0.8, 2.0), (0.2, 1.0)], {}, {})
        assert verified is True
        assert conf == pytest.approx(0.6)

    def test_quality_weighted_below_threshold(self) -> None:
        det = self._detector("quality_weighted")
        verified, conf = det._fuse_results([(0.2, 1.0), (0.3, 1.0)], {}, {})
        assert verified is False
        assert conf == pytest.approx(0.25)

    def test_quality_weighted_zero_total_weight(self) -> None:
        det = self._detector("quality_weighted")
        assert det._fuse_results([(0.9, 0.0)], {}, {}) == (False, 0.0)
