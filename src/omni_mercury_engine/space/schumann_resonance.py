# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Schumann Resonance Detector Module.

Earth-ionosphere waveguide electromagnetic field anomaly detection for environmental
and planetary monitoring. Detects anomalies in Schumann resonances (fundamental ~7.83 Hz
and harmonics at 14.3, 20.8, 27.3, 33.8 Hz) for early warning of:
- Seismic activity precursors
- Ionospheric disturbances
- Solar storm impacts
- Climate pattern changes
- Global electromagnetic field shifts

Key Features:
- ELF (Extremely Low Frequency) spectrum analysis
- Multi-harmonic anomaly detection
- Temporal correlation with geophysical events
- Amplitude and frequency deviation tracking
- Neurosymbolic correlation with ancient knowledge
- Golden ratio optimization for resonance detection
- O(n log n) complexity via FFT

Scientific Background:
- Schumann Resonances: Standing electromagnetic waves in Earth-ionosphere cavity
- Fundamental frequency: ~7.83 Hz (varies ±0.5 Hz)
- Caused by global lightning activity (~50 flashes/second)
- Amplitude: 0.1-2 picoTesla
- Absent on Moon (no ionosphere)

Research Sources:
- NASA ionosphere research
- NOAA Space Weather Prediction Center
- Academic seismology studies on electromagnetic precursors
- Geophysical research on Schumann resonances

⚠️ SIMULATION-BASED: For research/development. Correlations with seismic/climate
events require extensive validation. Not a replacement for established monitoring systems.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.fft import fft, fftfreq

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment, unused-ignore]
    nn = None  # type: ignore[assignment, unused-ignore]
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

_NNBase: type = nn.Module if TORCH_AVAILABLE else object  # type: ignore[assignment, unused-ignore]


@dataclass
class SchumannAnomalyResult:
    """Result from Schumann resonance anomaly detection."""

    anomaly_detected: bool
    anomaly_type: str
    confidence: float
    risk_score: float

    fundamental_freq: float
    fundamental_deviation: float
    harmonic_deviations: list[float] = field(default_factory=list)

    amplitude_anomaly: bool = False
    frequency_anomaly: bool = False
    power_spectrum_shift: bool = False

    correlated_events: list[str] = field(default_factory=list)
    temporal_pattern: dict[str, Any] | None = None

    recommendations: list[str] = field(default_factory=list)
    ancient_correlation: dict[str, Any] | None = None


class SchumannHarmonicAnalyzer(_NNBase):  # type: ignore[misc, unused-ignore]
    """Neural network for Schumann harmonic pattern analysis.

    Uses 1D CNN + LSTM for temporal ELF spectrum analysis with golden ratio optimized filter banks.
    """

    def __init__(self, spectrum_size: int = 512) -> None:
        """Initialize the instance."""
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for SchumannHarmonicAnalyzer. "
                "Install it with: pip install torch"
            )
        super().__init__()

        phi = 1.618

        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(1, int(32 * phi), kernel_size=7, padding=3),
            nn.BatchNorm1d(int(32 * phi)),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(int(32 * phi), int(64 * phi), kernel_size=5, padding=2),
            nn.BatchNorm1d(int(64 * phi)),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(int(64 * phi), int(128 * phi / 2), kernel_size=3, padding=1),
            nn.BatchNorm1d(int(128 * phi / 2)),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64),
        )

        self.lstm = nn.LSTM(
            input_size=int(128 * phi / 2),
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        self.anomaly_classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 4)
        )

        self.confidence_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def _features(
        self, spectrum: torch.Tensor, temporal_sequence: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Shared CNN/LSTM feature extractor (64-dim per sample)."""
        cnn_features = self.cnn_encoder(spectrum).transpose(1, 2)
        if temporal_sequence is not None:
            lstm_out, _ = self.lstm(temporal_sequence)
            features: torch.Tensor = lstm_out[:, -1, :]
        else:
            features = cnn_features.mean(dim=2)
        return features

    def forward(
        self, spectrum: torch.Tensor, temporal_sequence: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through harmonic analyzer.

        Args:
            spectrum: Power spectrum [batch, 1, freq_bins]
            temporal_sequence: Optional temporal spectrum sequence

        Returns:
            Tuple of (anomaly_logits, confidence)
        """
        features = self._features(spectrum, temporal_sequence)
        anomaly_logits = self.anomaly_classifier(features)
        confidence = self.confidence_head(features)

        return anomaly_logits, confidence

    def confidence_logits(
        self, spectrum: torch.Tensor, temporal_sequence: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Pre-sigmoid confidence logit, for numerically-stable training.

        ``confidence_head`` is ``Sequential(Linear, Sigmoid)``; this returns the
        Linear output *before* the Sigmoid so callers can train with
        ``BCEWithLogitsLoss`` (the correct objective) instead of ``BCELoss`` on a
        clamped sigmoid. Inference (``forward``) is unchanged:
        ``sigmoid(confidence_logits(x)) == forward(x)[1]`` exactly, and no
        parameter names change (checkpoints stay loadable). See WS-C diagnosis in
        ``docs/SCHUMANN_PREREGISTRATION.md``: the historical seed-instability was a
        full-batch optimisation artifact, not this objective, but logit-space
        training is the correct recipe regardless.
        """
        features = self._features(spectrum, temporal_sequence)
        logit: torch.Tensor = self.confidence_head[0](features)  # Linear, pre-Sigmoid
        return logit


class SchumannResonanceDetector:
    """Schumann Resonance Anomaly Detector.

    Monitors Earth-ionosphere electromagnetic cavity resonances for anomalies that may correlate
    with seismic, climate, or space weather events.
    """

    def __init__(
        self,
        sampling_rate: float = 100.0,
        enable_ancient_correlation: bool = True,
        golden_ratio_thresholds: bool = True,
    ):
        """Initialize Schumann resonance detector.

        Args:
            sampling_rate: ELF data sampling rate (Hz)
            enable_ancient_correlation: Correlate with ancient solar/lunar cycles
            golden_ratio_thresholds: Use φ-optimized detection thresholds
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for SchumannResonanceDetector. "
                "Install it with: pip install torch"
            )
        self.logger = logging.getLogger(__name__)
        self.sampling_rate = sampling_rate
        self.enable_ancient_correlation = enable_ancient_correlation
        self.golden_ratio = 1.618 if golden_ratio_thresholds else 1.0

        self.schumann_frequencies = [7.83, 14.3, 20.8, 27.3, 33.8]

        self.harmonic_analyzer = SchumannHarmonicAnalyzer(spectrum_size=512)
        # Anti-theater guard: the CNN-LSTM analyser ships with random weights and
        # no labelled Schumann anomaly corpus exists to train it. Until trained
        # weights are loaded via load_neural_weights(), its softmax/confidence
        # outputs are random noise, so detect_resonance_anomaly() must not derive
        # anomaly_type / confidence / risk_score from them. It falls back to the
        # deterministic FFT-physics assessment instead (see _physics_assessment).
        self._neural_trained = False
        self._warned_untrained = False

        self.ancient_knowledge = self._initialize_ancient_correlations()

        self.logger.info(f"Schumann Resonance Detector initialized (fs={sampling_rate}Hz)")

    @property
    def fundamental_freq(self) -> float:
        """Return the fundamental Schumann resonance frequency (~7.83 Hz).

        The fundamental frequency of the Earth-ionosphere cavity resonance.
        This is the first mode of the Schumann resonances.

        Returns:
            Fundamental frequency in Hz (approximately 7.83 Hz)
        """
        return self.schumann_frequencies[0]

    def _initialize_ancient_correlations(self) -> dict[str, Any]:
        """Initialize ancient knowledge correlations.

        Ancient civilizations observed natural cycles that correlate with electromagnetic phenomena.
        This establishes symbolic connections for neurosymbolic reasoning.
        """
        return {
            "solar_cycles": {
                "sunspot_cycle": 11.0,
                "hale_cycle": 22.0,
                "gleissberg_cycle": 88.0,
                "note": "Solar activity affects ionosphere, modulates Schumann resonances",
            },
            "lunar_cycles": {
                "synodic_month": 29.53,
                "draconic_month": 27.21,
                "note": "Lunar position affects Earth's magnetosphere",
            },
            "ancient_observations": {
                "egyptian_sirius_cycle": 365.25,
                "mayan_tzolkin": 260.0,
                "note": "Ancient calendars tracked celestial/terrestrial harmonics",
            },
            "resonance_ratios": {
                "golden_ratio": 1.618,
                "schumann_harmonic_ratio": 14.3 / 7.83,
                "note": "Natural frequency relationships",
            },
        }

    def detect_resonance_anomaly(
        self,
        elf_signal: np.ndarray[Any, Any],
        temporal_history: list[np.ndarray[Any, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SchumannAnomalyResult:
        """Detect anomalies in Schumann resonance patterns.

        Args:
            elf_signal: ELF electromagnetic field measurements (time series)
            temporal_history: Optional historical ELF measurements
            metadata: Optional metadata (location, date, equipment)

        Returns:
            Schumann resonance anomaly result
        """
        power_spectrum, frequencies = self._compute_power_spectrum(elf_signal)

        fundamental_freq, _fundamental_power = self._detect_fundamental(power_spectrum, frequencies)

        fundamental_deviation = abs(fundamental_freq - 7.83)

        harmonic_deviations = self._analyze_harmonics(power_spectrum, frequencies)

        amplitude_anomaly = self._detect_amplitude_anomaly(power_spectrum, frequencies)

        # Fixed fundamental-frequency deviation threshold (φ scaling removed).
        frequency_anomaly = fundamental_deviation > 0.5

        power_shift = self._detect_spectrum_shift(power_spectrum, frequencies)

        anomaly_detected = any([amplitude_anomaly, frequency_anomaly, power_shift])

        if self._neural_trained:
            # Trained weights present: use the learned CNN-LSTM classifier.
            spectrum_tensor = (
                torch.tensor(power_spectrum[:512], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            )
            temporal_tensor = None
            if temporal_history and len(temporal_history) > 0:
                temporal_tensor = self._process_temporal_history(temporal_history)

            self.harmonic_analyzer.eval()
            with torch.no_grad():
                anomaly_logits, confidence = self.harmonic_analyzer(
                    spectrum_tensor, temporal_tensor
                )
            anomaly_probs = torch.softmax(anomaly_logits[0], dim=0)
            anomaly_class = int(torch.argmax(anomaly_probs).item())
            confidence_score = float(confidence[0].item())
            anomaly_types = ["normal", "amplitude", "frequency", "combined"]
            anomaly_type = anomaly_types[anomaly_class]
        else:
            # Untrained network -> do NOT present random outputs as signal.
            # Derive the type and confidence from the deterministic FFT physics.
            if not self._warned_untrained:
                self.logger.warning(
                    "SchumannHarmonicAnalyzer is untrained (random weights); using the "
                    "deterministic FFT-physics assessment for anomaly_type/confidence. "
                    "Load trained weights via load_neural_weights() to enable the CNN-LSTM."
                )
                self._warned_untrained = True
            anomaly_type, confidence_score = self._physics_assessment(
                amplitude_anomaly, frequency_anomaly, power_shift, fundamental_deviation
            )

        # Raw confidence scaled only by the physics-derived deviation factor;
        # the former φ-scaled ``omni_seismic_precursor_detection`` multiplier
        # was removed (a reported score must not be multiplied by an unlearned
        # constant).
        risk_score = confidence_score * (1 + fundamental_deviation)

        correlated_events = self._correlate_with_events(
            fundamental_deviation, harmonic_deviations, amplitude_anomaly
        )

        temporal_pattern = (
            self._analyze_temporal_pattern(temporal_history) if temporal_history else None
        )

        recommendations = self._generate_recommendations(
            anomaly_type, risk_score, correlated_events
        )

        ancient_correlation = None
        if self.enable_ancient_correlation:
            ancient_correlation = self._correlate_ancient_patterns(
                fundamental_freq, temporal_pattern, metadata
            )

        result = SchumannAnomalyResult(
            anomaly_detected=anomaly_detected,
            anomaly_type=anomaly_type,
            confidence=confidence_score,
            risk_score=risk_score,
            fundamental_freq=fundamental_freq,
            fundamental_deviation=fundamental_deviation,
            harmonic_deviations=harmonic_deviations,
            amplitude_anomaly=amplitude_anomaly,
            frequency_anomaly=frequency_anomaly,
            power_spectrum_shift=power_shift,
            correlated_events=correlated_events,
            temporal_pattern=temporal_pattern,
            recommendations=recommendations,
            ancient_correlation=ancient_correlation,
        )

        self.logger.info(
            f"Schumann anomaly: {anomaly_type} "
            f"(f={fundamental_freq:.2f}Hz, risk={risk_score:.3f})"
        )

        return result

    @staticmethod
    def _physics_assessment(
        amplitude_anomaly: bool,
        frequency_anomaly: bool,
        power_shift: bool,
        fundamental_deviation: float,
    ) -> tuple[str, float]:
        """Deterministic anomaly type and confidence from the FFT physics.

        Used when the neural analyser is untrained, so the reported
        ``anomaly_type``/``confidence`` reflect real spectral evidence rather
        than random-weight network output. The type follows which independent
        spectral tests fired; the confidence combines how many fired (agreement)
        with the fundamental-frequency deviation magnitude, bounded to ``[0, 1]``.

        Args:
            amplitude_anomaly: Amplitude test fired.
            frequency_anomaly: Fundamental-frequency test fired.
            power_shift: Spectral power-shift test fired.
            fundamental_deviation: |f0 - 7.83| in Hz.

        Returns:
            ``(anomaly_type, confidence)`` with type in
            ``{normal, amplitude, frequency, combined}``.
        """
        if amplitude_anomaly and frequency_anomaly:
            anomaly_type = "combined"
        elif frequency_anomaly:
            anomaly_type = "frequency"
        elif amplitude_anomaly or power_shift:
            anomaly_type = "amplitude"
        else:
            anomaly_type = "normal"

        n_flags = int(amplitude_anomaly) + int(frequency_anomaly) + int(power_shift)
        evidence = n_flags / 3.0
        dev_term = min(1.0, fundamental_deviation / 2.0)  # 2 Hz off -> saturate
        confidence = float(min(1.0, 0.6 * evidence + 0.4 * dev_term))
        return anomaly_type, confidence

    def load_neural_weights(self, state_dict: dict[str, Any] | str) -> None:
        """Load trained weights for the CNN-LSTM analyser and enable it.

        Once trained weights exist (a labelled Schumann corpus is required to
        produce them honestly), this activates the learned classifier path in
        :meth:`detect_resonance_anomaly`.

        Args:
            state_dict: An in-memory ``state_dict`` or a path to a saved one.
        """
        loaded: Any = state_dict
        if isinstance(state_dict, str):
            loaded = torch.load(state_dict, map_location="cpu", weights_only=True)
        self.harmonic_analyzer.load_state_dict(loaded)
        self.harmonic_analyzer.eval()
        self._neural_trained = True
        self.logger.info("Schumann CNN-LSTM weights loaded; learned classifier enabled.")

    def _compute_power_spectrum(
        self, elf_signal: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Compute power spectrum using FFT (O(n log n) complexity)."""
        n = len(elf_signal)

        yf = fft(elf_signal)
        power = np.abs(yf[: n // 2]) ** 2
        xf = fftfreq(n, 1.0 / self.sampling_rate)[: n // 2]

        power = power / np.max(power)

        return power, xf

    def _detect_fundamental(
        self, power_spectrum: np.ndarray[Any, Any], frequencies: np.ndarray[Any, Any]
    ) -> tuple[float, float]:
        """Detect fundamental Schumann resonance frequency."""
        search_range = (frequencies >= 6.0) & (frequencies <= 10.0)

        search_spectrum = power_spectrum[search_range]
        search_freqs = frequencies[search_range]

        if len(search_spectrum) == 0:
            return 7.83, 0.0

        peak_idx = np.argmax(search_spectrum)
        fundamental_freq = float(search_freqs[peak_idx])
        fundamental_power = float(search_spectrum[peak_idx])

        return fundamental_freq, fundamental_power

    def _analyze_harmonics(
        self, power_spectrum: np.ndarray[Any, Any], frequencies: np.ndarray[Any, Any]
    ) -> list[float]:
        """Analyze deviations in harmonic frequencies."""
        deviations = []

        for harmonic_freq in self.schumann_frequencies[1:]:
            search_range = (frequencies >= harmonic_freq - 2.0) & (
                frequencies <= harmonic_freq + 2.0
            )

            if np.any(search_range):
                search_spectrum = power_spectrum[search_range]
                search_freqs = frequencies[search_range]

                if len(search_spectrum) > 0:
                    peak_idx = np.argmax(search_spectrum)
                    detected_freq = float(search_freqs[peak_idx])
                    deviation = abs(detected_freq - harmonic_freq)
                    deviations.append(deviation)

        return deviations

    def _detect_amplitude_anomaly(
        self, power_spectrum: np.ndarray[Any, Any], frequencies: np.ndarray[Any, Any]
    ) -> bool:
        """Detect amplitude anomalies in Schumann resonances."""
        schumann_band = (frequencies >= 5.0) & (frequencies <= 40.0)
        schumann_power = power_spectrum[schumann_band]

        if len(schumann_power) == 0:
            return False

        mean_power = np.mean(schumann_power)
        std_power = np.std(schumann_power)

        # Fixed sigma-multiple amplitude threshold (φ scaling removed); flag a
        # spectral peak that exceeds the band mean by more than this many
        # standard deviations.
        threshold = 1.5 * std_power

        max_power = np.max(schumann_power)

        return bool(max_power > (mean_power + threshold))

    def _detect_spectrum_shift(
        self, power_spectrum: np.ndarray[Any, Any], frequencies: np.ndarray[Any, Any]
    ) -> bool:
        """Detect significant shifts in power spectrum distribution."""
        low_band = (frequencies >= 5.0) & (frequencies <= 15.0)
        high_band = (frequencies >= 15.0) & (frequencies <= 40.0)

        low_power = np.sum(power_spectrum[low_band])
        high_power = np.sum(power_spectrum[high_band])

        if low_power == 0:
            return False

        ratio = high_power / low_power

        expected_ratio = 0.3

        # Fixed band-power ratio deviation threshold (φ scaling removed).
        return bool(abs(ratio - expected_ratio) > 0.3)

    def _process_temporal_history(
        self, temporal_history: list[np.ndarray[Any, Any]]
    ) -> torch.Tensor:
        """Process temporal history of spectra."""
        sequence_length = min(len(temporal_history), 10)

        temporal_spectra = np.zeros((1, sequence_length, 103), dtype=np.float32)

        for i, hist_signal in enumerate(temporal_history[-sequence_length:]):
            power, _freqs = self._compute_power_spectrum(hist_signal)
            temporal_spectra[0, i, :] = power[:103]

        return torch.tensor(temporal_spectra, dtype=torch.float32)

    def _correlate_with_events(
        self,
        fundamental_deviation: float,
        harmonic_deviations: list[float],
        amplitude_anomaly: bool,
    ) -> list[str]:
        """Correlate anomalies with potential geophysical events."""
        events = []

        if fundamental_deviation > 0.5:
            events.append("Potential ionospheric disturbance")

            if fundamental_deviation > 1.0:
                events.append("Possible seismic precursor (electromagnetic)")

        if harmonic_deviations and np.mean(harmonic_deviations) > 1.0:
            events.append("Harmonic structure perturbation")
            events.append("Consider space weather monitoring")

        if amplitude_anomaly:
            events.append("Elevated electromagnetic activity")
            events.append("Increased lightning or ionospheric modification")

        return events[:6]

    def _analyze_temporal_pattern(
        self, temporal_history: list[np.ndarray[Any, Any]]
    ) -> dict[str, Any]:
        """Analyze temporal evolution of resonance patterns."""
        if not temporal_history or len(temporal_history) < 2:
            return {}

        fundamental_series = []

        for hist_signal in temporal_history:
            power, freqs = self._compute_power_spectrum(hist_signal)
            fund_freq, _ = self._detect_fundamental(power, freqs)
            fundamental_series.append(fund_freq)

        return {
            "trend": (
                "increasing" if fundamental_series[-1] > fundamental_series[0] else "decreasing"
            ),
            "mean_freq": float(np.mean(fundamental_series)),
            "std_freq": float(np.std(fundamental_series)),
            "measurements": len(fundamental_series),
        }

    def _generate_recommendations(
        self, anomaly_type: str, risk_score: float, correlated_events: list[str]
    ) -> list[str]:
        """Generate monitoring recommendations."""
        recommendations = []

        if risk_score > 0.8:
            recommendations.append("HIGH PRIORITY: Significant electromagnetic anomaly")
            recommendations.append("Cross-correlate with seismic monitoring networks")
            recommendations.append("Alert geophysical research teams")
        elif risk_score > 0.6:
            recommendations.append("Elevated monitoring recommended")
            recommendations.append("Compare with NOAA space weather data")
        else:
            recommendations.append("Continue routine monitoring")

        if anomaly_type == "frequency":
            recommendations.append("Investigate ionospheric conditions")
        elif anomaly_type == "amplitude":
            recommendations.append("Analyze global lightning activity")
        elif anomaly_type == "combined":
            recommendations.append("Multi-factor analysis required")

        return recommendations[:6]

    def _correlate_ancient_patterns(
        self,
        fundamental_freq: float,
        temporal_pattern: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Correlate with ancient astronomical/geophysical cycles."""
        correlations: dict[str, list[str]] = {
            "detected_cycles": [],
            "symbolic_significance": [],
            "harmonic_relationships": [],
        }

        ratio_to_golden = fundamental_freq / self.golden_ratio
        correlations["harmonic_relationships"].append(f"Frequency/φ ratio: {ratio_to_golden:.3f}")

        if temporal_pattern and "measurements" in temporal_pattern:
            days = temporal_pattern["measurements"]

            lunar_correlation = abs(days - 27.21) < 5.0 or abs(days - 29.53) < 5.0
            if lunar_correlation:
                correlations["detected_cycles"].append("Lunar cycle correlation")
                correlations["symbolic_significance"].append(
                    "Ancient lunar observations: Electromagnetic-gravitational coupling"
                )

        if fundamental_freq > 7.83:
            correlations["symbolic_significance"].append(
                "Elevated resonance: Potential increased solar activity (ancient solar tracking)"
            )

        return correlations

    def extract_features(self, data: np.ndarray[Any, Any]) -> torch.Tensor:
        """Extract features for ML fusion integration."""
        power, freqs = self._compute_power_spectrum(data)

        features = []
        features.append(self._detect_fundamental(power, freqs)[0] / 10.0)

        harmonic_devs = self._analyze_harmonics(power, freqs)
        features.extend(harmonic_devs[:4] if len(harmonic_devs) >= 4 else [0.0] * 4)

        schumann_band = (freqs >= 5.0) & (freqs <= 40.0)
        features.append(np.mean(power[schumann_band]))

        features_array = np.array(features[:8], dtype=np.float32)
        return torch.tensor(features_array, dtype=torch.float32).unsqueeze(0)

    def predict(self, data: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Predict for engine integration."""
        result = self.detect_resonance_anomaly(data)

        return {
            "anomaly_scores": np.array([result.risk_score], dtype=np.float32),
            "anomaly_type": result.anomaly_type,
            "confidence": result.confidence,
            "fundamental_freq": result.fundamental_freq,
        }
