"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""
from __future__ import annotations

"""Biometric anomaly detection model."""

from typing import Any

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from deepface import DeepFace

    DEEPFACE_AVAILABLE = True
except (ImportError, ValueError):
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

    def __init__(self, config: dict[str, Any] | None = None, **kwargs) -> None:
        self.config = config or {}
        self.model_name = self.config.get("model_name", "Facenet")
        self.use_harmonic_features = self.config.get("use_harmonic_features", True)
        self.harmonic_decomposer = HarmonicDecomposer()
        self.fourier_analyzer = FourierAnalyzer()
        self.target_embedding_size = 128

    def _extract_harmonic_features(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Extract features using harmonic decomposition and Fourier analysis."""
        if not isinstance(data, np.ndarray[Any, Any]):
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

        if not isinstance(data, np.ndarray[Any, Any]):
            data = np.array(data)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if DeepFace is not None:
            try:
                result = DeepFace.represent(
                    data, model_name=self.model_name, enforce_detection=False
                )
                if isinstance(result, list) and len(result) > 0:
                    embeddings = [r["embedding"] for r in result]
                    features = np.array(embeddings, dtype=np.float32)
                else:
                    features = self._extract_harmonic_features(data)
            except Exception:
                features = self._extract_harmonic_features(data)
        else:
            features = self._extract_harmonic_features(data)

        features = self._normalize_embedding_size(features)

        if TORCH_AVAILABLE:
            if isinstance(features, np.ndarray[Any, Any]):
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

        if not isinstance(data, np.ndarray[Any, Any]):
            data = np.array(data)

        if DeepFace is not None:
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
            except Exception:
                return {
                    "model_type": "biometric",
                    "anomaly_scores": np.array([0.5], dtype=np.float32),
                    "error": "Analysis failed",
                }
        else:
            features = self.extract_features(data)
            if TORCH_AVAILABLE and isinstance(features, torch.Tensor):
                features = features.detach().cpu().numpy()

            anomaly_score = np.mean(np.abs(features - 0.5))

            return {
                "model_type": "biometric",
                "anomaly_scores": np.array([anomaly_score], dtype=np.float32),
                "age": 25,
                "gender_confidence": 0.5,
                "emotion_confidence": 0.5,
            }
