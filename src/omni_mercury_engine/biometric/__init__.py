"""
Biometric Recognition Module for Mercury Agent.

Provides multi-modal biometric authentication including iris, fingerprint,
and voice recognition with comprehensive liveness detection.

Key Components:
- IrisRecognizer: Daugman IrisCode-based iris recognition
- FingerprintRecognizer: Minutiae-based fingerprint matching
- VoiceRecognizer: MFCC and embedding-based speaker verification
- BiometricAnomalyDetector: Unified multi-modal biometric system
"""

from omni_mercury_engine.biometric.iris_recognition import (
    GaborFilter,
    IrisEncoder,
    IrisFeatures,
    IrisLivenessDetector,
    IrisMatcher,
    IrisMatchResult,
    IrisNormalizer,
    IrisRecognizer,
    IrisSegmenter,
    LivenessResult as IrisLivenessResult,
)
from omni_mercury_engine.biometric.fingerprint_recognition import (
    FingerprintFeatures,
    FingerprintLivenessDetector,
    FingerprintLivenessResult,
    FingerprintMatcher,
    FingerprintMatchResult,
    FingerprintRecognizer,
    GaborEnhancer,
    Minutia,
    MinutiaeExtractor,
    MinutiaeType,
    OrientationFieldEstimator,
    RidgeFrequencyEstimator,
    Singularity,
    SingularityType,
)
from omni_mercury_engine.biometric.voice_recognition import (
    AudioPreprocessor,
    EnergyExtractor,
    MFCCExtractor,
    PitchExtractor,
    SpeakerEmbedding,
    VoiceActivityDetector,
    VoiceFeatures,
    VoiceLivenessDetector,
    VoiceLivenessResult,
    VoiceMatcher,
    VoiceMatchResult,
    VoiceRecognizer,
)

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np


class BiometricModality(Enum):
    """Supported biometric modalities."""

    IRIS = auto()
    FINGERPRINT = auto()
    VOICE = auto()
    FACE = auto()


class FusionStrategy(Enum):
    """Multi-modal fusion strategy."""

    SCORE_LEVEL = auto()
    DECISION_LEVEL = auto()
    QUALITY_WEIGHTED = auto()


@dataclass
class BiometricEnrollment:
    """Enrolled biometric data for an identity."""

    identity: str
    iris_features: IrisFeatures | None = None
    fingerprint_features: FingerprintFeatures | None = None
    voice_features: VoiceFeatures | None = None
    enrollment_timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BiometricVerificationResult:
    """Result of biometric verification."""

    identity: str
    is_verified: bool
    confidence: float
    modality_results: dict[str, Any]
    liveness_results: dict[str, Any]
    fusion_method: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class BiometricAnomalyResult:
    """Result of biometric anomaly detection."""

    is_anomaly: bool
    anomaly_score: float
    anomaly_type: str | None
    modality_scores: dict[str, float]
    liveness_scores: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)


class BiometricAnomalyDetector:
    """
    Multi-modal biometric anomaly detection.

    Provides unified interface for biometric verification across
    iris, fingerprint, and voice modalities with liveness detection.

    Example:
        detector = BiometricAnomalyDetector(
            modalities=["iris", "fingerprint", "voice"],
            fusion_strategy="quality_weighted",
            liveness_required=True,
        )

        result = detector.verify(
            iris_image=iris_scan,
            fingerprint_image=fingerprint_scan,
            voice_sample=audio_recording,
            claimed_identity="user_123",
        )
    """

    SUPPORTED_MODALITIES = ["iris", "fingerprint", "voice"]

    def __init__(
        self,
        modalities: list[str] | None = None,
        fusion_strategy: str = "quality_weighted",
        liveness_required: bool = True,
        anomaly_threshold: float = 0.5,
    ) -> None:
        """
        Initialize the biometric anomaly detector.

        Args:
            modalities: List of modalities to use (default: all)
            fusion_strategy: score_level, decision_level, or quality_weighted
            liveness_required: Whether to require liveness detection
            anomaly_threshold: Threshold for anomaly detection
        """
        if modalities is None:
            modalities = self.SUPPORTED_MODALITIES.copy()

        self._modalities = [m.lower() for m in modalities if m.lower() in self.SUPPORTED_MODALITIES]
        self._liveness_required = liveness_required
        self._anomaly_threshold = anomaly_threshold

        strategy_map = {
            "score_level": FusionStrategy.SCORE_LEVEL,
            "decision_level": FusionStrategy.DECISION_LEVEL,
            "quality_weighted": FusionStrategy.QUALITY_WEIGHTED,
        }
        self._fusion_strategy = strategy_map.get(fusion_strategy, FusionStrategy.QUALITY_WEIGHTED)

        self._recognizers: dict[str, Any] = {}
        if "iris" in self._modalities:
            self._recognizers["iris"] = IrisRecognizer(liveness_required=liveness_required)
        if "fingerprint" in self._modalities:
            self._recognizers["fingerprint"] = FingerprintRecognizer(liveness_required=liveness_required)
        if "voice" in self._modalities:
            self._recognizers["voice"] = VoiceRecognizer(liveness_required=liveness_required)

        self._enrollments: dict[str, BiometricEnrollment] = {}

    def enroll(
        self,
        identity: str,
        iris_image: np.ndarray | None = None,
        fingerprint_image: np.ndarray | None = None,
        voice_sample: np.ndarray | None = None,
        **kwargs: Any,
    ) -> BiometricEnrollment:
        """
        Enroll a new identity with biometric samples.

        Args:
            identity: Unique identifier for the person
            iris_image: Iris image (if using iris modality)
            fingerprint_image: Fingerprint image (if using fingerprint modality)
            voice_sample: Voice audio sample (if using voice modality)

        Returns:
            BiometricEnrollment with extracted features
        """
        import time

        enrollment = BiometricEnrollment(
            identity=identity,
            enrollment_timestamp=time.time(),
            metadata=kwargs,
        )

        if "iris" in self._modalities and iris_image is not None:
            enrollment.iris_features = self._recognizers["iris"].extract_features(iris_image)

        if "fingerprint" in self._modalities and fingerprint_image is not None:
            enrollment.fingerprint_features = self._recognizers["fingerprint"].extract_features(fingerprint_image)

        if "voice" in self._modalities and voice_sample is not None:
            enrollment.voice_features = self._recognizers["voice"].extract_features(voice_sample)

        self._enrollments[identity] = enrollment
        return enrollment

    def verify(
        self,
        claimed_identity: str,
        iris_image: np.ndarray | None = None,
        fingerprint_image: np.ndarray | None = None,
        voice_sample: np.ndarray | None = None,
        iris_liveness_images: list[np.ndarray] | None = None,
        fingerprint_liveness_images: list[np.ndarray] | None = None,
        voice_liveness_samples: list[np.ndarray] | None = None,
    ) -> BiometricVerificationResult:
        """
        Verify a claimed identity against enrolled biometrics.

        Args:
            claimed_identity: Identity to verify against
            iris_image: Probe iris image
            fingerprint_image: Probe fingerprint image
            voice_sample: Probe voice sample
            *_liveness_images: Additional images/samples for liveness detection

        Returns:
            BiometricVerificationResult with verification outcome
        """
        if claimed_identity not in self._enrollments:
            return BiometricVerificationResult(
                identity=claimed_identity,
                is_verified=False,
                confidence=0.0,
                modality_results={"error": "Identity not enrolled"},
                liveness_results={},
                fusion_method=self._fusion_strategy.name,
            )

        enrollment = self._enrollments[claimed_identity]
        modality_results: dict[str, Any] = {}
        liveness_results: dict[str, Any] = {}
        scores: list[tuple[float, float]] = []

        if "iris" in self._modalities and iris_image is not None and enrollment.iris_features is not None:
            match_result, liveness_result = self._recognizers["iris"].verify(
                iris_image,
                enrollment.iris_features,
                iris_liveness_images,
            )
            modality_results["iris"] = {
                "match_score": match_result.match_score,
                "is_match": match_result.is_match,
                "hamming_distance": match_result.hamming_distance,
            }
            if liveness_result:
                liveness_results["iris"] = {
                    "is_live": liveness_result.is_live,
                    "confidence": liveness_result.confidence,
                }
            quality = enrollment.iris_features.quality_score
            scores.append((match_result.match_score, quality))

        if "fingerprint" in self._modalities and fingerprint_image is not None and enrollment.fingerprint_features is not None:
            match_result, liveness_result = self._recognizers["fingerprint"].verify(
                fingerprint_image,
                enrollment.fingerprint_features,
                fingerprint_liveness_images,
            )
            modality_results["fingerprint"] = {
                "match_score": match_result.match_score / 100.0,
                "is_match": match_result.is_match,
                "matched_minutiae": match_result.matched_minutiae,
            }
            if liveness_result:
                liveness_results["fingerprint"] = {
                    "is_live": liveness_result.is_live,
                    "confidence": liveness_result.confidence,
                }
            quality = enrollment.fingerprint_features.overall_quality
            scores.append((match_result.match_score / 100.0, quality))

        if "voice" in self._modalities and voice_sample is not None and enrollment.voice_features is not None:
            match_result, liveness_result = self._recognizers["voice"].verify(
                voice_sample,
                enrollment.voice_features,
                voice_liveness_samples,
            )
            modality_results["voice"] = {
                "similarity_score": match_result.similarity_score,
                "is_match": match_result.is_match,
                "embedding_distance": match_result.embedding_distance,
            }
            if liveness_result:
                liveness_results["voice"] = {
                    "is_live": liveness_result.is_live,
                    "confidence": liveness_result.confidence,
                }
            quality = enrollment.voice_features.quality_score
            scores.append((match_result.similarity_score, quality))

        is_verified, confidence = self._fuse_results(scores, modality_results, liveness_results)

        return BiometricVerificationResult(
            identity=claimed_identity,
            is_verified=is_verified,
            confidence=confidence,
            modality_results=modality_results,
            liveness_results=liveness_results,
            fusion_method=self._fusion_strategy.name,
        )

    def detect_anomaly(
        self,
        iris_image: np.ndarray | None = None,
        fingerprint_image: np.ndarray | None = None,
        voice_sample: np.ndarray | None = None,
    ) -> BiometricAnomalyResult:
        """
        Detect anomalies in biometric samples without identity verification.

        Checks for presentation attacks, poor quality samples, and unusual patterns.

        Args:
            iris_image: Iris image to analyze
            fingerprint_image: Fingerprint image to analyze
            voice_sample: Voice sample to analyze

        Returns:
            BiometricAnomalyResult with anomaly assessment
        """
        modality_scores: dict[str, float] = {}
        liveness_scores: dict[str, float] = {}
        anomaly_details: dict[str, Any] = {}

        if "iris" in self._modalities and iris_image is not None:
            features = self._recognizers["iris"].extract_features(iris_image)
            quality_score = features.quality_score
            modality_scores["iris"] = quality_score

            liveness_detector = IrisLivenessDetector()
            liveness = liveness_detector.detect([iris_image])
            liveness_scores["iris"] = liveness.confidence
            anomaly_details["iris"] = {
                "quality": quality_score,
                "liveness": liveness.confidence,
                "pupil_response": liveness.pupil_response,
            }

        if "fingerprint" in self._modalities and fingerprint_image is not None:
            features = self._recognizers["fingerprint"].extract_features(fingerprint_image)
            quality_score = features.overall_quality
            modality_scores["fingerprint"] = quality_score

            liveness_detector = FingerprintLivenessDetector()
            liveness = liveness_detector.detect([fingerprint_image])
            liveness_scores["fingerprint"] = liveness.confidence
            anomaly_details["fingerprint"] = {
                "quality": quality_score,
                "liveness": liveness.confidence,
                "minutiae_count": len(features.minutiae),
            }

        if "voice" in self._modalities and voice_sample is not None:
            features = self._recognizers["voice"].extract_features(voice_sample)
            quality_score = features.quality_score
            modality_scores["voice"] = quality_score

            liveness_detector = VoiceLivenessDetector()
            liveness = liveness_detector.detect([voice_sample])
            liveness_scores["voice"] = liveness.confidence
            anomaly_details["voice"] = {
                "quality": quality_score,
                "liveness": liveness.confidence,
                "duration": features.duration,
            }

        combined_quality = np.mean(list(modality_scores.values())) if modality_scores else 0.0
        combined_liveness = np.mean(list(liveness_scores.values())) if liveness_scores else 0.0
        anomaly_score = 1.0 - (0.5 * combined_quality + 0.5 * combined_liveness)

        is_anomaly = anomaly_score > self._anomaly_threshold
        anomaly_type = None
        if is_anomaly:
            if combined_liveness < 0.5:
                anomaly_type = "presentation_attack"
            elif combined_quality < 0.5:
                anomaly_type = "poor_quality"
            else:
                anomaly_type = "suspicious_pattern"

        return BiometricAnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            anomaly_type=anomaly_type,
            modality_scores=modality_scores,
            liveness_scores=liveness_scores,
            details=anomaly_details,
        )

    def _fuse_results(
        self,
        scores: list[tuple[float, float]],
        modality_results: dict[str, Any],
        liveness_results: dict[str, Any],
    ) -> tuple[bool, float]:
        """Fuse results from multiple modalities."""
        if not scores:
            return False, 0.0

        if self._liveness_required:
            for modality, result in liveness_results.items():
                if not result.get("is_live", True):
                    return False, 0.0

        if self._fusion_strategy == FusionStrategy.SCORE_LEVEL:
            avg_score = np.mean([s[0] for s in scores])
            is_verified = avg_score >= 0.5
            return is_verified, float(avg_score)

        elif self._fusion_strategy == FusionStrategy.DECISION_LEVEL:
            matches = sum(1 for _, result in modality_results.items() if result.get("is_match", False))
            total = len(modality_results)
            is_verified = matches > total / 2
            confidence = matches / total if total > 0 else 0.0
            return is_verified, confidence

        else:
            total_weight = sum(q for _, q in scores)
            if total_weight == 0:
                return False, 0.0

            weighted_score = sum(s * q for s, q in scores) / total_weight
            is_verified = weighted_score >= 0.5
            return is_verified, float(weighted_score)


__all__ = [
    # Main classes
    "BiometricAnomalyDetector",
    "BiometricEnrollment",
    "BiometricVerificationResult",
    "BiometricAnomalyResult",
    "BiometricModality",
    "FusionStrategy",
    # Iris recognition
    "IrisRecognizer",
    "IrisFeatures",
    "IrisMatchResult",
    "IrisLivenessResult",
    "IrisSegmenter",
    "IrisNormalizer",
    "IrisEncoder",
    "IrisMatcher",
    "IrisLivenessDetector",
    "GaborFilter",
    # Fingerprint recognition
    "FingerprintRecognizer",
    "FingerprintFeatures",
    "FingerprintMatchResult",
    "FingerprintLivenessResult",
    "Minutia",
    "MinutiaeType",
    "Singularity",
    "SingularityType",
    "MinutiaeExtractor",
    "FingerprintMatcher",
    "FingerprintLivenessDetector",
    "OrientationFieldEstimator",
    "RidgeFrequencyEstimator",
    "GaborEnhancer",
    # Voice recognition
    "VoiceRecognizer",
    "VoiceFeatures",
    "VoiceMatchResult",
    "VoiceLivenessResult",
    "AudioPreprocessor",
    "MFCCExtractor",
    "PitchExtractor",
    "EnergyExtractor",
    "SpeakerEmbedding",
    "VoiceMatcher",
    "VoiceLivenessDetector",
    "VoiceActivityDetector",
]
