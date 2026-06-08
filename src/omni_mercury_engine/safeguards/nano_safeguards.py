# Copyright (C) 2025 Steel Security Advisors LLC
"""Nano-Safeguards for Micro-Anomaly Detection.

Implements the N term from the Lyapunov stability framework for detecting
micro-scale anomalies that may be missed by standard detection methods.

Key Features:
- Hierarchical micro-pattern scanning at multiple scales
- Threshold-based alerts (convergence < 0.01)
- Dimensional downsampling for subtle pattern detection
- Integration with fusion network for cross-domain correlation
- 3R mechanism integration (Recursion-Resonance-Refactoring)

Mathematical Foundation:
The nano-safeguard implements dimensional downsampling to detect micro-anomalies:
    N(x) = PCA_reconstruct(PCA_project(x, k)) - x
    micro_score = ||N(x)||_2 / ||x||_2

Where k is the target dimension for downsampling (typically 1-3).

References:
- PROTECTION_OVERVIEW.md: Nano-Safeguards (N Term) specification
- Lyapunov stability framework for convergence guarantees
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
else:
    try:
        import torch
        from torch import nn

        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False

from scipy.fft import fft

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.exceptions import DetectorException

logger = logging.getLogger(__name__)


@dataclass
class NanoSafeguardResult:
    """Results from nano-safeguard micro-anomaly detection."""

    micro_anomaly_detected: bool
    confidence: float
    alert_level: str

    convergence_score: float = 0.0
    dimensional_residual: float = 0.0
    hierarchical_scores: list[float] = field(default_factory=list)

    bit_level_anomalies: int = 0
    molecular_entropy: float = 0.0
    quantum_checksum: float = 0.0

    resonance_score: float = 0.0
    recursion_depth_reached: int = 0
    refactoring_suggestions: list[str] = field(default_factory=list)

    threshold_violations: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)


if TYPE_CHECKING or TORCH_AVAILABLE:

    class HierarchicalMicroScanner(nn.Module):
        """Hierarchical micro-pattern scanner using multi-scale convolutions.

        Implements the Recursion Engine component of 3R for multi-scale pattern detection at
        progressively finer granularities.
        """

        def __init__(self, input_dim: int = 64, num_scales: int = 4) -> None:
            """Initialize the instance."""
            super().__init__()
            self.num_scales = num_scales

            self.scale_encoders = nn.ModuleList()
            for scale in range(num_scales):
                kernel_size = 2 ** (num_scales - scale - 1) + 1
                padding = kernel_size // 2
                self.scale_encoders.append(
                    nn.Sequential(
                        nn.Conv1d(1, 16, kernel_size=kernel_size, padding=padding),
                        nn.BatchNorm1d(16),
                        nn.ReLU(),
                        nn.Conv1d(16, 8, kernel_size=3, padding=1),
                        nn.AdaptiveAvgPool1d(input_dim // (2**scale) or 1),
                    )
                )

            self.fusion = nn.Sequential(
                nn.Linear(8 * num_scales, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
            """Multi-scale hierarchical scanning.

            Args:
                x: Input tensor (batch, features)

            Returns:
                Tuple of (micro_anomaly_score, scale_features)
            """
            if x.dim() == 1:
                x = x.unsqueeze(0)
            if x.dim() == 2:
                x = x.unsqueeze(1)

            scale_features = []
            for encoder in self.scale_encoders:
                feat = encoder(x)
                pooled = feat.mean(dim=-1)
                scale_features.append(pooled)

            combined = torch.cat(scale_features, dim=-1)
            score = self.fusion(combined)

            return score, scale_features

else:

    class HierarchicalMicroScanner:
        """Stub: HierarchicalMicroScanner requires PyTorch."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initialize the instance."""
            raise ImportError(
                "HierarchicalMicroScanner requires PyTorch. Install with: pip install torch"
            )


class ResonanceAnalyzer:
    """FFT-based resonance analysis for frequency-domain micro-anomalies.

    Implements the Resonance Engine component of 3R for detecting harmonic anomalies and periodic
    micro-patterns.
    """

    def __init__(self, fundamental_freq: float = 7.83) -> None:
        """Initialize the instance."""
        self.fundamental_freq = fundamental_freq
        self.harmonic_count = 8

    def analyze(self, signal: np.ndarray[Any, Any]) -> dict[str, float]:
        """Analyze signal for resonance anomalies.

        Args:
            signal: Input signal array

        Returns:
            Dictionary of resonance metrics
        """
        if len(signal) < 8:
            return {"resonance_score": 0.0, "harmonic_ratio": 0.0}

        signal_flat = signal.flatten()

        fft_result = fft(signal_flat)
        power_spectrum = np.abs(fft_result) ** 2

        frequencies = np.fft.fftfreq(len(signal_flat))

        harmonic_powers = []
        for n in range(1, self.harmonic_count + 1):
            harmonic_idx = int(n * self.fundamental_freq * len(signal_flat) / 100)
            if 0 <= harmonic_idx < len(power_spectrum):
                harmonic_powers.append(power_spectrum[harmonic_idx])

        total_power = np.sum(power_spectrum) + 1e-10
        harmonic_power = np.sum(harmonic_powers) if harmonic_powers else 0.0

        harmonic_ratio = harmonic_power / total_power

        spectral_entropy = self._compute_spectral_entropy(power_spectrum)

        resonance_score = 1.0 - min(harmonic_ratio * 2.0, 1.0)

        return {
            "resonance_score": float(resonance_score),
            "harmonic_ratio": float(harmonic_ratio),
            "spectral_entropy": float(spectral_entropy),
            "dominant_frequency": (
                float(frequencies[np.argmax(power_spectrum[1:]) + 1])
                if len(power_spectrum) > 1
                else 0.0
            ),
        }

    @staticmethod
    def _compute_spectral_entropy(power_spectrum: np.ndarray[Any, Any]) -> float:
        """Compute spectral entropy for frequency distribution analysis."""
        normalized = power_spectrum / (np.sum(power_spectrum) + 1e-10)
        entropy = -np.sum(normalized * np.log2(normalized + 1e-10))
        return float(entropy / np.log2(len(power_spectrum) + 1))  # type: ignore[no-any-return, unused-ignore]


class NanoSafeguardDetector(BaseDetector):
    """Nano-Safeguard Detector for Micro-Anomaly Detection.

    Implements the N term from the Lyapunov stability framework,
    providing hierarchical micro-pattern scanning with threshold-based
    alerts and integration with the fusion network.

    Key Features:
    - Hierarchical micro-pattern scanning at multiple scales
    - Threshold-based alerts (convergence < 0.01)
    - Dimensional downsampling for subtle pattern detection
    - 3R mechanism integration (Recursion-Resonance-Refactoring)
    - Molecular-level hash integrity checking
    - Quantum-inspired checksum validation
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the instance."""
        super().__init__(config)

        self.convergence_threshold = self.config.get("convergence_threshold", 0.01)
        self.micro_threshold = self.config.get("micro_threshold", 0.05)
        self.num_scales = self.config.get("num_scales", 4)
        self.target_dim = self.config.get("target_dim", 3)
        self.enable_resonance = self.config.get("enable_resonance", True)
        self.enable_molecular = self.config.get("enable_molecular", True)

        self.hierarchical_scanner = HierarchicalMicroScanner(
            input_dim=64, num_scales=self.num_scales
        )
        self.resonance_analyzer = ResonanceAnalyzer()

        self.baseline_stats: dict[str, float] = {}
        self.memory_buffer: list[np.ndarray[Any, Any]] = []
        self.max_memory = self.config.get("max_memory", 100)

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> NanoSafeguardDetector:
        """Fit nano-safeguard to normal data patterns.

        Args:
            data: Normal data for baseline establishment

        Returns:
            Self for method chaining
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        self.baseline_stats = {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "median": float(np.median(data)),
        }

        self._is_fitted = True
        logger.info("NanoSafeguardDetector fitted with baseline statistics")
        return self

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Detect micro-anomalies using nano-safeguard protocols.

        Args:
            data: Input data for micro-anomaly detection

        Returns:
            Detection results dictionary
        """
        if not self._is_fitted:
            raise DetectorException("Detector must be fitted before detection")

        if isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy()
        else:
            data_np = data

        result = self.detect_micro_anomalies(data_np)

        return {
            "is_anomaly": result.micro_anomaly_detected,
            "scores": result.confidence,
            "convergence_score": result.convergence_score,
            "dimensional_residual": result.dimensional_residual,
            "hierarchical_scores": result.hierarchical_scores,
            "resonance_score": result.resonance_score,
            "alert_level": result.alert_level,
            "threshold_violations": result.threshold_violations,
            "detector_type": "nano_safeguard",
        }

    def detect_micro_anomalies(self, data: np.ndarray[Any, Any]) -> NanoSafeguardResult:
        """Comprehensive micro-anomaly detection.

        Args:
            data: Input data array

        Returns:
            NanoSafeguardResult with detailed analysis
        """
        result = NanoSafeguardResult(
            micro_anomaly_detected=False,
            confidence=0.0,
            alert_level="normal",
        )

        convergence = self._compute_convergence(data)
        result.convergence_score = convergence

        if convergence < self.convergence_threshold:
            result.threshold_violations.append(
                f"Convergence {convergence:.4f} < threshold {self.convergence_threshold}"
            )

        dimensional_residual = self._dimensional_downsampling_detection(data)
        result.dimensional_residual = dimensional_residual

        hierarchical_scores = self._hierarchical_scan(data)
        result.hierarchical_scores = hierarchical_scores

        if self.enable_resonance:
            resonance_result = self.resonance_analyzer.analyze(data)
            result.resonance_score = resonance_result["resonance_score"]

        if self.enable_molecular:
            molecular_result = self._molecular_analysis(data)
            result.molecular_entropy = molecular_result["entropy"]
            result.quantum_checksum = molecular_result["checksum"]
            result.bit_level_anomalies = molecular_result["bit_anomalies"]

        combined_score = self._compute_combined_score(result)
        result.confidence = combined_score

        result.micro_anomaly_detected = combined_score > self.micro_threshold
        result.alert_level = self._determine_alert_level(combined_score)

        result.recommended_actions = self._generate_recommendations(result)

        self._update_memory(data)

        return result

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> torch.Tensor:
        """Extract nano-safeguard features for ML fusion.

        Args:
            data: Input data

        Returns:
            Feature tensor for fusion network
        """
        if isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy()
        else:
            data_np = data

        if not self._is_fitted:
            self.fit(data_np)

        convergence = self._compute_convergence(data_np)
        dimensional = self._dimensional_downsampling_detection(data_np)
        hierarchical = self._hierarchical_scan(data_np)

        resonance_result = self.resonance_analyzer.analyze(data_np)
        molecular_result = self._molecular_analysis(data_np)

        features = [
            convergence,
            dimensional,
            *hierarchical,
            resonance_result["resonance_score"],
            resonance_result["harmonic_ratio"],
            resonance_result["spectral_entropy"],
            molecular_result["entropy"],
            molecular_result["checksum"],
            float(molecular_result["bit_anomalies"]) / 100.0,
        ]

        while len(features) < 20:
            features.append(0.0)

        return torch.tensor(features[:20], dtype=torch.float32)

    def _compute_convergence(self, data: np.ndarray[Any, Any]) -> float:
        """Compute convergence score relative to baseline."""
        if not self.baseline_stats:
            return 1.0

        current_mean = np.mean(data)
        current_std = np.std(data)

        mean_diff = abs(current_mean - self.baseline_stats["mean"])
        std_diff = abs(current_std - self.baseline_stats["std"])

        baseline_range = self.baseline_stats["max"] - self.baseline_stats["min"] + 1e-10

        convergence = 1.0 - min((mean_diff + std_diff) / baseline_range, 1.0)

        return float(convergence)

    def _dimensional_downsampling_detection(self, data: np.ndarray[Any, Any]) -> float:
        """Dimensional downsampling for micro-anomaly detection.

        Implements the N term enhancement from PROTECTION_OVERVIEW.md.
        """
        data_2d = data.reshape(-1, 1) if data.ndim == 1 else data

        if data_2d.shape[0] < self.target_dim or data_2d.shape[1] < 2:
            return 0.0

        try:
            from omni_mercury_engine.ml.mercury_ml import PCA

            target_dim = min(self.target_dim, data_2d.shape[1] - 1, data_2d.shape[0] - 1)
            if target_dim < 1:
                return 0.0

            pca = PCA(n_components=target_dim)
            downsampled = pca.fit_transform(data_2d)
            reconstructed = pca.inverse_transform(downsampled)

            residuals = np.abs(data_2d - reconstructed)

            threshold = np.percentile(residuals.flatten(), 95)
            micro_anomaly_count = np.sum(residuals > threshold)
            total_elements = residuals.size

            micro_rate = micro_anomaly_count / total_elements

            explained_variance = np.sum(pca.explained_variance_ratio_)
            unexplained_score = 1.0 - explained_variance

            final_score = micro_rate * 0.6 + unexplained_score * 0.4

            return float(min(final_score, 1.0))

        except Exception as e:
            logger.debug(f"Dimensional downsampling failed: {e}")
            return 0.0

    def _hierarchical_scan(self, data: np.ndarray[Any, Any]) -> list[float]:
        """Perform hierarchical multi-scale scanning."""
        scores = []

        for scale in range(self.num_scales):
            window_size = 2 ** (self.num_scales - scale)
            scale_score = self._scan_at_scale(data, window_size)
            scores.append(scale_score)

        return scores

    def _scan_at_scale(self, data: np.ndarray[Any, Any], window_size: int) -> float:
        """Scan data at a specific scale."""
        data_flat = data.flatten()

        if len(data_flat) < window_size:
            return 0.0

        variances = []
        for i in range(0, len(data_flat) - window_size + 1, max(1, window_size // 2)):
            window = data_flat[i : i + window_size]
            variances.append(np.var(window))

        if not variances:
            return 0.0

        variance_array = np.array(variances)
        variance_changes = np.abs(np.diff(variance_array)) if len(variance_array) > 1 else [0.0]

        score = np.mean(variance_changes) / (np.std(variance_array) + 1e-10)

        return float(min(score, 1.0))

    def _molecular_analysis(self, data: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Molecular-level analysis for nano-scale integrity."""
        data_bytes = data.tobytes()

        hash_obj = hashlib.sha3_256(data_bytes)
        hash_bytes = hash_obj.digest()
        byte_values = np.frombuffer(hash_bytes, dtype=np.uint8)
        _, counts = np.unique(byte_values, return_counts=True)
        probabilities = counts / len(byte_values)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        normalized_entropy = entropy / 8.0

        current = data_bytes
        checksum = 0.0
        for _ in range(4):
            hash_obj = hashlib.sha3_256(current)
            current = hash_obj.digest()
            byte_sum = sum(current)
            checksum += byte_sum / (256.0 * len(current))
        checksum /= 4.0

        if len(data_bytes) > 1:
            byte_array = np.frombuffer(data_bytes, dtype=np.uint8)
            transitions = np.abs(np.diff(byte_array.astype(int)))
            bit_anomalies = int(np.sum(transitions > 200))
        else:
            bit_anomalies = 0

        return {
            "entropy": float(normalized_entropy),
            "checksum": float(checksum),
            "bit_anomalies": bit_anomalies,
        }

    def _compute_combined_score(self, result: NanoSafeguardResult) -> float:
        """Compute combined micro-anomaly score."""
        weights = {
            "convergence": 0.25,
            "dimensional": 0.25,
            "hierarchical": 0.20,
            "resonance": 0.15,
            "molecular": 0.15,
        }

        convergence_anomaly = 1.0 - result.convergence_score

        hierarchical_avg = (
            np.mean(result.hierarchical_scores) if result.hierarchical_scores else 0.0
        )

        molecular_score = (
            result.molecular_entropy * 0.4
            + (1.0 - result.quantum_checksum) * 0.3
            + min(result.bit_level_anomalies / 100.0, 1.0) * 0.3
        )

        combined = (
            weights["convergence"] * convergence_anomaly
            + weights["dimensional"] * result.dimensional_residual
            + weights["hierarchical"] * hierarchical_avg
            + weights["resonance"] * result.resonance_score
            + weights["molecular"] * molecular_score
        )

        return float(min(combined, 1.0))

    def _determine_alert_level(self, score: float) -> str:
        """Determine alert level based on combined score."""
        if score > 0.8:
            return "critical"
        elif score > 0.6:
            return "high"
        elif score > 0.4:
            return "moderate"
        elif score > 0.2:
            return "low"
        else:
            return "normal"

    def _generate_recommendations(self, result: NanoSafeguardResult) -> list[str]:
        """Generate recommended actions based on detection results."""
        recommendations = []

        if result.convergence_score < self.convergence_threshold:
            recommendations.append(
                "Investigate convergence deviation - system may be drifting from baseline"
            )

        if result.dimensional_residual > 0.5:
            recommendations.append(
                "High dimensional residual detected - check for subtle pattern changes"
            )

        if result.resonance_score > 0.7:
            recommendations.append(
                "Resonance anomaly detected - investigate frequency-domain patterns"
            )

        if result.bit_level_anomalies > 50:
            recommendations.append("Significant bit-level anomalies - verify data integrity")

        if result.micro_anomaly_detected:
            recommendations.append(
                "Micro-anomaly confirmed - escalate to fusion network for cross-domain analysis"
            )

        return recommendations

    def _update_memory(self, data: np.ndarray[Any, Any]) -> None:
        """Update memory buffer for temporal analysis."""
        self.memory_buffer.append(data.copy())
        if len(self.memory_buffer) > self.max_memory:
            self.memory_buffer.pop(0)
