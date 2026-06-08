# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Biometric anomaly detection model."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment, unused-ignore]
    TORCH_AVAILABLE = False

try:
    from deepface import DeepFace

    DEEPFACE_AVAILABLE = True
except Exception:
    # ``except Exception`` (not just ``(ImportError, ValueError)``)
    # because the deepface -> retinaface -> tensorflow import chain
    # can raise ``OSError`` (missing CUDA runtime), ``RuntimeError``
    # (tf version mismatch), or ``AttributeError`` (stale transitive
    # deps) in addition to the known ``ValueError`` from the
    # tf-keras check.  ``BaseException`` is deliberately not caught.
    DeepFace = None
    DEEPFACE_AVAILABLE = False


class HarmonicDecomposer:
    """Simple harmonic decomposition using FFT for biometric feature analysis."""

    def decompose(self, signal: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Decompose signal into harmonic components using FFT."""
        if signal.ndim == 1:
            signal = signal.reshape(1, -1)
        fft_result = np.fft.fft(signal, axis=1)
        return np.abs(fft_result)


class FourierAnalyzer:
    """Fourier analysis for frequency-domain biometric features."""

    def analyze(self, data: np.ndarray[Any, Any]) -> dict[str, np.ndarray[Any, Any]]:
        """Analyze frequency components of biometric data."""
        if data.ndim == 1:
            data = data.reshape(1, -1)
        fft_result = np.fft.fft(data, axis=1)
        power_spectrum = np.abs(fft_result) ** 2
        return {
            "frequencies": np.fft.fftfreq(data.shape[1]),
            "power_spectrum": power_spectrum,
            "phase": np.angle(fft_result),
        }


class BiometricAnomalyModel:
    """Biometric anomaly detection for facial recognition and analysis."""

    def __init__(self, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Initialize the instance."""
        self.config = config or {}
        self.model_name = self.config.get("model_name", "Facenet")
        self.use_harmonic_features = self.config.get("use_harmonic_features", True)
        self.harmonic_decomposer = HarmonicDecomposer()
        self.fourier_analyzer = FourierAnalyzer()
        self.target_embedding_size = 128

    @staticmethod
    def _is_image_input(data: object) -> bool:
        """Return True only for inputs DeepFace can actually process.

        DeepFace accepts an image path or an array shaped like an image
        (H x W x C, or a batch N x H x W x C, with C in {1, 3, 4}). The
        fusion pipeline routes 2-D tabular feature matrices through every
        model, including this one; without this guard those tabular arrays
        reach ``DeepFace.represent`` / ``DeepFace.analyze``, triggering a
        ~92 MB model download and a stream of "DeepFace ... failed"
        warnings before falling back. Restricting DeepFace to genuine
        image input keeps the tabular detect path on the harmonic
        fallback silently.
        """
        if isinstance(data, str):
            return True
        if isinstance(data, np.ndarray):
            if data.ndim == 3 and data.shape[-1] in (1, 3, 4):
                return True
            if data.ndim == 4 and data.shape[-1] in (1, 3, 4):
                return True
        return False

    def _extract_harmonic_features(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Extract features using harmonic decomposition and Fourier analysis."""
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if data.ndim == 3:
            data = data.reshape(data.shape[0], -1)
        elif data.ndim == 1:
            data = data.reshape(1, -1)

        batch_size = data.shape[0]

        if not self.use_harmonic_features or self.config.get("use_harmonic_features") is False:
            return np.zeros((batch_size, self.target_embedding_size), dtype=np.float32)

        harmonic = self.harmonic_decomposer.decompose(data)
        fourier_result = self.fourier_analyzer.analyze(data)

        power_spectrum = fourier_result["power_spectrum"]
        phase = fourier_result["phase"]

        features_per_sample = []
        for i in range(batch_size):
            harmonic_feats = (
                harmonic[i, :32]
                if harmonic.shape[1] >= 32
                else np.pad(harmonic[i], (0, 32 - harmonic.shape[1]))
            )
            power_feats = (
                power_spectrum[i, :32]
                if power_spectrum.shape[1] >= 32
                else np.pad(power_spectrum[i], (0, 32 - power_spectrum.shape[1]))
            )
            phase_feats = (
                phase[i, :32]
                if phase.shape[1] >= 32
                else np.pad(phase[i], (0, 32 - phase.shape[1]))
            )

            sample_features = np.concatenate(
                [
                    harmonic_feats[:32],
                    power_feats[:32],
                    phase_feats[:32],
                    np.array(
                        [
                            np.mean(harmonic[i]),
                            np.std(harmonic[i]),
                            np.max(harmonic[i]),
                            np.min(harmonic[i]),
                        ]
                    ),
                    np.array(
                        [
                            np.mean(power_spectrum[i]),
                            np.std(power_spectrum[i]),
                            np.max(power_spectrum[i]),
                            np.min(power_spectrum[i]),
                        ]
                    ),
                    np.zeros(24),
                ]
            )
            features_per_sample.append(sample_features[: self.target_embedding_size])

        return np.array(features_per_sample, dtype=np.float32)

    def _normalize_embedding_size(
        self, embedding: np.ndarray[Any, Any] | torch.Tensor
    ) -> np.ndarray[Any, Any] | torch.Tensor:
        """Normalize embedding to target size (128 features)."""
        if TORCH_AVAILABLE and isinstance(embedding, torch.Tensor):
            is_torch = True
            embedding_np = embedding.detach().cpu().numpy()
        else:
            is_torch = False
            embedding_np = np.array(embedding)

        if embedding_np.ndim == 1:
            embedding_np = embedding_np.reshape(1, -1)

        batch_size, current_size = embedding_np.shape

        if current_size == self.target_embedding_size:
            normalized = embedding_np
        elif current_size > self.target_embedding_size:
            normalized = embedding_np[:, : self.target_embedding_size]
        else:
            padding = np.zeros(
                (batch_size, self.target_embedding_size - current_size), dtype=embedding_np.dtype
            )
            normalized = np.concatenate([embedding_np, padding], axis=1)

        if is_torch and TORCH_AVAILABLE:
            return torch.from_numpy(normalized)
        return normalized

    def extract_features(
        self, data: np.ndarray[Any, Any] | dict[str, Any]
    ) -> np.ndarray[Any, Any] | torch.Tensor:
        """Extract biometric features from image data."""
        if isinstance(data, dict):
            data = data["reference"] if "reference" in data else np.array(next(iter(data.values())))

        image_input = self._is_image_input(data)

        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if DeepFace is not None and image_input:
            try:
                result = DeepFace.represent(
                    data, model_name=self.model_name, enforce_detection=False
                )
                if isinstance(result, list) and len(result) > 0:
                    embeddings = [r["embedding"] for r in result]
                    features = np.array(embeddings, dtype=np.float32)
                else:
                    features = self._extract_harmonic_features(data)
            except Exception as e:
                logger.debug("DeepFace feature extraction failed, using harmonic fallback: %s", e)
                features = self._extract_harmonic_features(data)
        else:
            features = self._extract_harmonic_features(data)

        features = self._normalize_embedding_size(features)  # type: ignore[assignment, unused-ignore]

        if TORCH_AVAILABLE:
            if isinstance(features, np.ndarray):
                return torch.from_numpy(features)
            return features
        return features

    def predict(self, data: np.ndarray[Any, Any] | dict[str, Any]) -> dict[str, Any]:
        """Predict biometric anomalies and quality scores."""
        if data is None:
            return {
                "model_type": "biometric",
                "anomaly_scores": np.array([0.5], dtype=np.float32),
                "age": 25,
                "gender_confidence": 0.5,
                "emotion_confidence": 0.5,
                "error": "No input data provided",
            }

        if isinstance(data, dict):
            if "reference" in data:
                data = data["reference"]
            elif len(data) > 0:
                data = next(iter(data.values()))
            else:
                data = np.array([])

        image_input = self._is_image_input(data)

        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if DeepFace is not None and image_input:
            try:
                result = DeepFace.analyze(
                    data, actions=["age", "gender", "emotion"], enforce_detection=False
                )

                if isinstance(result, list) and len(result) > 0:
                    result = result[0]

                age = result.get("age", 25)
                gender = result.get("gender", {})
                emotion = result.get("emotion", {})

                gender_confidence = max(gender.values()) if gender else 0.5
                emotion_confidence = max(emotion.values()) if emotion else 0.5

                anomaly_score = 1.0 - (gender_confidence + emotion_confidence) / 2.0

                return {
                    "model_type": "biometric",
                    "anomaly_scores": np.array([anomaly_score], dtype=np.float32),
                    "age": age,
                    "gender_confidence": float(gender_confidence),
                    "emotion_confidence": float(emotion_confidence),
                }
            except Exception as e:
                logger.warning("DeepFace biometric analysis failed: %s", e)
                return {
                    "model_type": "biometric",
                    "anomaly_scores": np.array([0.5], dtype=np.float32),
                    "error": f"Analysis failed: {e}",
                }
        else:
            features = self.extract_features(data)
            if TORCH_AVAILABLE and isinstance(features, torch.Tensor):
                features = features.detach().cpu().numpy()

            anomaly_score = np.mean(np.abs(features - 0.5))  # type: ignore[assignment, unused-ignore]

            return {
                "model_type": "biometric",
                "anomaly_scores": np.array([anomaly_score], dtype=np.float32),
                "age": 25,
                "gender_confidence": 0.5,
                "emotion_confidence": 0.5,
            }
