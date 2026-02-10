"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisors LLC

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

"""
Emergent Life Detector - SETI-like Anomaly Detection for Non-Human Intelligence

Novel constructions for detecting emergent and non-human life through:
- SETI-like anomaly detection in cosmic signals (ResonanceEngine for non-natural patterns)
- Bio-signal pattern recognition in space/environmental data
- Multiverse contact protocol exploration for communication strategies

⚠️ SIMULATION-BASED: Uses simulated SETI signals and biosignatures. Real-world validation required.
Consult SETI researchers before acting on detections.

Research sources:
- SETI Institute signal analysis methodologies
- NASA biosignature detection frameworks
- ESA exobiology research
- Breakthrough Listen technosignature search

"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import fft

from omni_mercury_engine.core.three_r_mechanism import ResonanceEngine
from omni_mercury_engine.models.multiverse import MultiverseOmniEngine
from omni_mercury_engine.space.space_exploration_analyzer import SpaceExplorationAnalyzer


@dataclass
class LifeDetectionResult:
    """Result from emergent life detection analysis."""

    life_signal_detected: bool
    confidence: float
    signal_type: str
    anomaly_score: float
    bio_signature_patterns: list[str] = field(default_factory=list)
    seti_technosignatures: list[dict[str, Any]] = field(default_factory=list)
    contact_protocols: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class SETICosmicSignalAnalyzer:
    """
    SETI-like cosmic signal anomaly detection using resonance analysis.

    Detects non-natural patterns in cosmic signals that may indicate
    technological signatures of non-human intelligence.
    """

    def __init__(self, threshold_std: float = 4.0) -> None:
        self.resonance = ResonanceEngine(sampling_rate=1.0)
        self.space_analyzer = SpaceExplorationAnalyzer()
        self.threshold_std = threshold_std
        self.logger = logging.getLogger(__name__)

        self.natural_patterns = {
            "pulsar": {"period_range": (0.001, 10.0), "regularity": "high"},
            "solar_burst": {"duration_ms": (10, 1000), "broadband": True},
            "cosmic_noise": {"distribution": "gaussian", "correlation": "low"},
        }

    def detect_seti_anomaly(
        self,
        signal_data: np.ndarray[Any, Any],
        context: dict[str, Any] | None = None,
        threshold_std: float | None = None,
    ) -> dict[str, Any]:
        """
        Detect SETI-like anomalies in cosmic signals.

        Args:
            signal_data: Time-series signal data from telescope/receiver
            context: Optional context (frequency band, source coordinates, etc.)
            threshold_std: Optional override for anomaly detection threshold

        Returns:
            SETI anomaly detection results with technosignature assessment
        """
        context = context or {}
        threshold = threshold_std if threshold_std is not None else self.threshold_std

        frequencies, magnitudes = self.resonance.compute_resonance_spectrum(signal_data)

        anomalies = self.resonance.detect_resonance_anomalies(signal_data, threshold_std=threshold)

        narrow_band_peaks = self._detect_narrow_band_signals(frequencies, magnitudes)

        repeating_patterns = self._detect_repeating_patterns(signal_data)

        modulation_detected = self._detect_modulation(signal_data)

        seti_confidence = self._calculate_seti_confidence(
            narrow_band_peaks, repeating_patterns, modulation_detected, anomalies
        )

        technosignatures = []
        if narrow_band_peaks:
            technosignatures.append(
                {
                    "type": "narrow_band_signal",
                    "frequencies_hz": narrow_band_peaks,
                    "significance": "high",
                }
            )
        if repeating_patterns["detected"]:
            technosignatures.append(
                {
                    "type": "repeating_pattern",
                    "period_sec": repeating_patterns["period"],
                    "significance": "medium",
                }
            )
        if modulation_detected["detected"]:
            technosignatures.append(
                {
                    "type": "modulation",
                    "modulation_type": modulation_detected["type"],
                    "significance": "high",
                }
            )

        return {
            "seti_anomaly_detected": seti_confidence > 0.5,
            "seti_confidence": seti_confidence,
            "technosignatures": technosignatures,
            "resonance_anomalies": anomalies["num_anomalies"],
            "narrow_band_signals": len(narrow_band_peaks),
            "recommendations": self._generate_seti_recommendations(
                seti_confidence, technosignatures
            ),
        }

    def _detect_narrow_band_signals(
        self, frequencies: np.ndarray[Any, Any], magnitudes: np.ndarray[Any, Any]
    ) -> list[float]:
        """Detect narrow-band signals (key SETI technosignature)."""
        narrow_band_peaks = []

        for i in range(1, len(magnitudes) - 1):
            if magnitudes[i] > magnitudes[i - 1] and magnitudes[i] > magnitudes[i + 1]:
                half_max = magnitudes[i] / 2.0
                left_idx = i
                while left_idx > 0 and magnitudes[left_idx] > half_max:
                    left_idx -= 1
                right_idx = i
                while right_idx < len(magnitudes) - 1 and magnitudes[right_idx] > half_max:
                    right_idx += 1

                bandwidth = frequencies[right_idx] - frequencies[left_idx]
                if bandwidth < 1.0:
                    narrow_band_peaks.append(float(frequencies[i]))

        return narrow_band_peaks

    def _detect_repeating_patterns(self, signal_data: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Detect repeating patterns in signal."""
        if len(signal_data) < 10:
            return {"detected": False, "period": 0.0}

        autocorr = np.correlate(signal_data, signal_data, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]

        peaks = []
        for i in range(1, len(autocorr) - 1):
            if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
                if autocorr[i] > np.mean(autocorr) + 2 * np.std(autocorr):
                    peaks.append(i)

        if peaks:
            return {"detected": True, "period": float(peaks[0])}

        return {"detected": False, "period": 0.0}

    def _detect_modulation(self, signal_data: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Detect modulation patterns (AM, FM, etc.)."""
        envelope = np.abs(signal_data)
        envelope_var = np.var(envelope)

        if envelope_var > np.var(signal_data) * 0.1:
            return {
                "detected": True,
                "type": "amplitude_modulation",
                "variance": float(envelope_var),
            }

        return {"detected": False, "type": "none"}

    def _calculate_seti_confidence(
        self,
        narrow_band_peaks: list[Any],
        repeating_patterns: dict[str, Any],
        modulation: dict[str, Any],
        resonance_anomalies: dict[str, Any],
    ) -> float:
        """Calculate overall SETI confidence score."""
        confidence = 0.0

        if narrow_band_peaks:
            confidence += 0.4

        if repeating_patterns["detected"]:
            confidence += 0.3

        if modulation["detected"]:
            confidence += 0.2

        if resonance_anomalies["num_anomalies"] > 0:
            confidence += 0.1

        return min(confidence, 1.0)

    def _generate_seti_recommendations(
        self, confidence: float, technosignatures: list[dict[str, Any]]
    ) -> list[str]:
        """Generate SETI analysis recommendations."""
        recs = []

        if confidence > 0.7:
            recs.append("HIGH CONFIDENCE: Potential technosignature detected")
            recs.append("Immediate follow-up observations recommended")
            recs.append("Verify signal persistence and characteristics")
            recs.append("Rule out terrestrial interference and natural sources")
        elif confidence > 0.4:
            recs.append("MODERATE: Interesting signal characteristics detected")
            recs.append("Additional observation time recommended")
        else:
            recs.append("LOW: Likely natural cosmic signal")
            recs.append("Continue routine monitoring")

        for sig in technosignatures:
            recs.append(f"Detected {sig['type']} with {sig['significance']} significance")

        return recs


class BioSignalPatternRecognizer:
    """
    Bio-signal pattern recognition for detecting life indicators.

    Analyzes environmental/space data for biosignature patterns
    (e.g., atmospheric gas ratios, periodic biological rhythms).
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        self.biosignatures = {
            "oxygen_methane_disequilibrium": {"o2_min": 0.1, "ch4_min": 0.001},
            "seasonal_variation": {"period_days": (90, 400), "amplitude_min": 0.05},
            "circadian_rhythm": {"period_hours": (20, 28), "regularity": "high"},
        }

    def detect_biosignatures(
        self, environmental_data: np.ndarray[Any, Any], data_type: str = "atmospheric"
    ) -> dict[str, Any]:
        """
        Detect biosignature patterns in environmental/space data.

        Args:
            environmental_data: Time-series environmental measurements
            data_type: Type of data ('atmospheric', 'surface', 'spectral')

        Returns:
            Biosignature detection results
        """
        biosig_detected = []

        periodicity = self._detect_periodicity(environmental_data)
        if periodicity["detected"]:
            if 20 * 3600 < periodicity["period_sec"] < 28 * 3600:
                biosig_detected.append("circadian_like_rhythm")
            elif 90 * 86400 < periodicity["period_sec"] < 400 * 86400:
                biosig_detected.append("seasonal_like_variation")

        if environmental_data.ndim > 1 and environmental_data.shape[1] >= 2:
            disequilibrium = self._detect_chemical_disequilibrium(environmental_data)
            if disequilibrium:
                biosig_detected.append("chemical_disequilibrium")

        confidence = len(biosig_detected) / 3.0

        return {
            "biosignatures_detected": len(biosig_detected) > 0,
            "biosignature_types": biosig_detected,
            "confidence": confidence,
            "periodicity": periodicity,
            "recommendations": self._generate_biosig_recommendations(biosig_detected, confidence),
        }

    def _detect_periodicity(self, data: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Detect periodic patterns in data."""
        if len(data) < 10:
            return {"detected": False, "period_sec": 0.0}

        fft_result = np.abs(fft.fft(data.flatten()))
        frequencies = fft.fftfreq(len(data.flatten()))

        positive_freqs = frequencies[frequencies > 0]
        positive_fft = fft_result[frequencies > 0]

        if len(positive_fft) > 0:
            dominant_freq_idx = np.argmax(positive_fft)
            dominant_freq = positive_freqs[dominant_freq_idx]

            if dominant_freq > 1e-8:
                period_sec = 1.0 / dominant_freq
                return {
                    "detected": True,
                    "period_sec": float(period_sec),
                    "frequency_hz": float(dominant_freq),
                }

        return {"detected": False, "period_sec": 0.0}

    def _detect_chemical_disequilibrium(self, data: np.ndarray[Any, Any]) -> bool:
        """Detect chemical disequilibrium (biosignature indicator)."""
        if data.shape[1] < 2:
            return False

        ratio = np.mean(data[:, 0]) / (np.mean(data[:, 1]) + 1e-8)

        return bool(ratio > 10.0 or ratio < 0.1)

    def _generate_biosig_recommendations(
        self, biosignatures: list[str], confidence: float
    ) -> list[str]:
        """Generate biosignature analysis recommendations."""
        recs = []

        if confidence > 0.6:
            recs.append("SIGNIFICANT: Multiple biosignature patterns detected")
            recs.append("Recommend detailed spectroscopic analysis")
            recs.append("Consider follow-up with specialized instruments")
        elif confidence > 0.3:
            recs.append("INTERESTING: Some biosignature indicators present")
            recs.append("Additional data collection recommended")
        else:
            recs.append("Insufficient biosignature evidence")
            recs.append("Continue baseline monitoring")

        for biosig in biosignatures:
            recs.append(f"Detected: {biosig}")

        return recs


class MultiverseContactProtocolExplorer:
    """
    Multiverse-based exploration of contact protocols.

    Uses multiverse optimization to explore potential communication
    strategies for establishing contact with non-human intelligence.
    """

    def __init__(self, num_universes: int = 30) -> None:
        self.multiverse = MultiverseOmniEngine(
            num_universes=num_universes, state_dim=128, convergence_threshold=0.9
        )
        self.logger = logging.getLogger(__name__)

    def explore_contact_protocols(self, signal_characteristics: dict[str, Any]) -> dict[str, Any]:
        """
        Explore potential contact/communication protocols using multiverse.

        Args:
            signal_characteristics: Characteristics of detected signal

        Returns:
            Optimal contact protocol strategies
        """

        def protocol_fitness(protocol_vector: np.ndarray[Any, Any]) -> float:
            info_transfer = np.mean(protocol_vector[:32])
            error_correction = np.std(protocol_vector[32:64])
            universality = -np.var(protocol_vector[64:96])

            return float(info_transfer + error_correction + universality)

        converged = self.multiverse.converge_multiverse(protocol_fitness)

        sorted_universes = sorted(
            self.multiverse.universes.values(), key=lambda u: u.fitness, reverse=True
        )

        protocol_candidates = []
        for i, universe in enumerate(sorted_universes[:5]):
            protocol_candidates.append(
                {
                    "protocol_id": f"PROTOCOL-{i+1}",
                    "fitness": float(universe.fitness),
                    "characteristics": {
                        "information_density": float(np.mean(universe.state_vector[:32])),
                        "error_resilience": float(np.std(universe.state_vector[32:64])),
                        "universality_score": float(-np.var(universe.state_vector[64:96])),
                    },
                }
            )

        return {
            "optimal_protocols": protocol_candidates,
            "protocols_explored": len(self.multiverse.universes),
            "convergence_achieved": converged.fitness > 0.9,
            "recommendations": self._generate_protocol_recommendations(protocol_candidates),
        }

    def _generate_protocol_recommendations(self, protocols: list[dict[str, Any]]) -> list[str]:
        """Generate contact protocol recommendations."""
        recs = []

        if protocols:
            best = protocols[0]
            recs.append(f"Optimal protocol: {best['protocol_id']}")
            recs.append(
                f"Information density: {best['characteristics']['information_density']:.3f}"
            )
            recs.append(f"Error resilience: {best['characteristics']['error_resilience']:.3f}")
            recs.append("Recommend implementing multi-protocol approach for redundancy")
        else:
            recs.append("No optimal protocol identified")
            recs.append("Continue protocol exploration with refined parameters")

        return recs


class EmergentLifeDetector:
    """
    Unified emergent life detector integrating SETI, biosignatures,
    and contact protocol exploration.
    """

    def __init__(
        self,
        enable_seti: bool = True,
        enable_biosignatures: bool = True,
        enable_contact_protocols: bool = True,
    ):
        self.enable_seti = enable_seti
        self.enable_biosignatures = enable_biosignatures
        self.enable_contact_protocols = enable_contact_protocols

        self.seti_analyzer = SETICosmicSignalAnalyzer() if enable_seti else None
        self.biosig_recognizer = BioSignalPatternRecognizer() if enable_biosignatures else None
        self.protocol_explorer = (
            MultiverseContactProtocolExplorer() if enable_contact_protocols else None
        )

        self.logger = logging.getLogger(__name__)

    def detect_emergent_life(
        self, data: np.ndarray[Any, Any], analysis_type: str, context: dict[str, Any] | None = None
    ) -> LifeDetectionResult:
        """
        Comprehensive emergent life detection.

        Args:
            data: Signal or environmental data
            analysis_type: 'seti', 'biosignatures', or 'comprehensive'
            context: Optional context information

        Returns:
            Life detection results
        """
        result = LifeDetectionResult(
            life_signal_detected=False, confidence=0.0, signal_type="unknown", anomaly_score=0.0
        )

        context = context or {}

        if self.enable_seti and analysis_type in ["seti", "comprehensive"]:
            seti = self.seti_analyzer.detect_seti_anomaly(data, context)  # type: ignore[union-attr]
            if seti is not None and seti.get("seti_anomaly_detected"):
                result.life_signal_detected = True
                result.signal_type = "technosignature"
                result.confidence = max(result.confidence, seti.get("seti_confidence", 0.0))
                result.seti_technosignatures = seti.get("technosignatures", [])
                result.recommendations.extend(seti.get("recommendations", []))

        if self.enable_biosignatures and analysis_type in ["biosignatures", "comprehensive"]:
            biosig = self.biosig_recognizer.detect_biosignatures(  # type: ignore[union-attr]
                data, context.get("data_type", "atmospheric")
            )
            if biosig is not None and biosig.get("biosignatures_detected"):
                result.life_signal_detected = True
                result.signal_type = "biosignature"
                result.confidence = max(result.confidence, biosig.get("confidence", 0.0))
                result.bio_signature_patterns = biosig.get("biosignature_types", [])
                result.recommendations.extend(biosig.get("recommendations", []))

        if (
            self.enable_contact_protocols
            and result.life_signal_detected
            and result.signal_type == "technosignature"
        ):
            protocols = self.protocol_explorer.explore_contact_protocols(  # type: ignore[union-attr]
                {"confidence": result.confidence, "technosignatures": result.seti_technosignatures}
            )
            if protocols is not None:
                result.contact_protocols = [
                    p["protocol_id"] for p in protocols.get("optimal_protocols", [])
                ]
                result.recommendations.extend(protocols.get("recommendations", []))

        result.anomaly_score = result.confidence

        return result
