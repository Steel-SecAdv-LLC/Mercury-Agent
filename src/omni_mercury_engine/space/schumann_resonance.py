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
- Cyclic geophysical correlation (lunar tidal / solar cycle, exploratory)
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
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.fft import fft, fftfreq

from omni_mercury_engine.data_sources.live_ingestion import (
    LiveDataError,
    fetch_live_datapoints,
    require_live_client,
)

if TYPE_CHECKING:
    from omni_mercury_engine.data_sources.geomagnetic import BGSELFStationSource
    from omni_mercury_engine.data_sources.live_ingestion import LiveFetch
from omni_mercury_engine.detectors.hazard_diagnostics import HazardDiagnostics

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
    cycle_correlation: dict[str, Any] | None = None

    # Live-ingestion provenance (populated only by detect_live()).
    source_id: str | None = None
    data_provenance: str | None = None
    live_context: dict[str, Any] | None = None
    # Populated only when the detector was built with keep_diagnostics=True.
    diagnostics: HazardDiagnostics | None = None


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
        enable_cycle_correlation: bool = True,
        golden_ratio_thresholds: bool = True,
        data_source: BGSELFStationSource | None = None,
        keep_diagnostics: bool = False,
    ):
        """Initialize Schumann resonance detector.

        Live-ingestion pattern (uniform across hazard detectors): pass an
        optional BGS ELF client via ``data_source`` (dependency injection;
        default None = fully offline). :meth:`fetch_live_data` exposes a
        provenance-checked fetch and :meth:`detect_live` runs the detector's
        own FFT pipeline over the ELF record carried by the fetched
        DataPoint, stamping ``source_id`` / ``data_provenance`` /
        ``live_context`` on the result. Because the BGS client labels its
        no-instrument output ``metadata["simulated"]=True``, consuming it
        without caller-supplied ``raw_samples`` requires an explicit
        ``allow_simulated=True``.

        Args:
            sampling_rate: ELF data sampling rate (Hz)
            enable_cycle_correlation: Correlate with cyclic geophysical periods
                (lunar tidal / solar cycle); exploratory context only
            golden_ratio_thresholds: Use φ-optimized detection thresholds
            data_source: Optional BGS ELF station client for the live path.
            keep_diagnostics: When True, each anomaly result carries the
                one-sided, max-normalized ELF power spectrum the detection ran
                on (see
                :class:`~omni_mercury_engine.detectors.hazard_diagnostics.HazardDiagnostics`).
                Default False keeps memory behavior unchanged.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for SchumannResonanceDetector. "
                "Install it with: pip install torch"
            )
        self.logger = logging.getLogger(__name__)
        self.sampling_rate = sampling_rate
        self.enable_cycle_correlation = enable_cycle_correlation
        self.golden_ratio = 1.618 if golden_ratio_thresholds else 1.0
        self._elf_source = data_source
        self.keep_diagnostics = keep_diagnostics

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
        # Ratified operating point for the learned path's anomaly_detected
        # decision (validation-selected threshold carried by the training
        # pipeline's checkpoint payload -- see schumann_harmonics.py
        # _select_operating_point, mirroring the solar-storm/tsunami policy).
        # None (untrained, bare state_dicts, or pre-convention checkpoints)
        # keeps the historical behavior: anomaly_detected stays the
        # deterministic physics flags on every path.
        self._operating_point: dict[str, float] | None = None

        self.geophysical_cycles = self._initialize_cycle_correlations()

        self.omni_resonance_scalars = {
            "omni_electromagnetic_harmony": 1.46 * self.golden_ratio,
            "omni_ionospheric_coherence": 1.42 * self.golden_ratio,
            "omni_planetary_resonance": 1.44 * self.golden_ratio,
            "omni_seismic_precursor_detection": 1.48 * self.golden_ratio,
            "omni_space_weather_correlation": 1.40 * self.golden_ratio,
            "omni_frequency_stability": 1.38 * self.golden_ratio,
            "omni_amplitude_sensitivity": 1.43 * self.golden_ratio,
            "omni_geophysical_cycle_alignment": 1.37 * self.golden_ratio,
        }

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

    def _initialize_cycle_correlations(self) -> dict[str, Any]:
        """Initialize cyclic geophysical correlation references (EXPLORATORY).

        Reference periods for lunar tidal and solar-cycle correlation checks,
        used as symbolic context for neurosymbolic reasoning.

        Evidence status: tidal (lunar) triggering of seismicity has been
        reported but the effect is small and contested for forecasting
        (Cochran, Vidale & Tanaka 2004, Science 306:1164-1166; Ide, Yabe &
        Tanaka 2016, Nature Geoscience 9:834-837). Solar-cycle modulation of
        geomagnetic activity is well established, but neither correlation is
        a validated earthquake/eruption predictor. These correlations are
        tracked as exploratory context only and do not alter the detection
        computation.
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
            "calendar_cycles": {
                "egyptian_sirius_cycle": 365.25,
                "mayan_tzolkin": 260.0,
                "note": "Historical calendar periods (Sothic/tropical year, Tzolk'in) "
                "kept as long-period references",
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

        fundamental_freq, fundamental_power = self._detect_fundamental(power_spectrum, frequencies)

        fundamental_deviation = abs(fundamental_freq - 7.83)

        harmonic_deviations = self._analyze_harmonics(power_spectrum, frequencies)

        amplitude_anomaly = self._detect_amplitude_anomaly(power_spectrum, frequencies)

        frequency_anomaly = fundamental_deviation > (0.5 * self.golden_ratio)

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
            if self._operating_point is not None:
                # Ratified deployed rule for the learned path: the DECISION is
                # confidence > tau with the checkpoint's validation-selected
                # threshold. Only anomaly_detected changes -- the confidence
                # estimate and anomaly_type stay exactly what the network
                # emitted (the merit gate evaluated this precise rule).
                anomaly_detected = confidence_score > self._operating_point["detection_threshold"]
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

        risk_score = (
            confidence_score
            * self.omni_resonance_scalars["omni_seismic_precursor_detection"]
            * (1 + fundamental_deviation)
        )

        correlated_events = self._correlate_with_events(
            fundamental_deviation, harmonic_deviations, amplitude_anomaly
        )

        temporal_pattern = (
            self._analyze_temporal_pattern(temporal_history) if temporal_history else None
        )

        recommendations = self._generate_recommendations(
            anomaly_type, risk_score, correlated_events
        )

        cycle_correlation = None
        if self.enable_cycle_correlation:
            cycle_correlation = self._correlate_cycle_patterns(
                fundamental_freq, temporal_pattern, metadata
            )

        diagnostics: HazardDiagnostics | None = None
        if self.keep_diagnostics:
            # Capture the harmonic power spectrum the detection ALREADY computed
            # (one-sided, max-normalized) -- no recomputation.
            diagnostics = HazardDiagnostics(
                hazard="schumann",
                arrays={
                    "frequencies_hz": np.asarray(frequencies, dtype=float),
                    "power_spectrum": np.asarray(power_spectrum, dtype=float),
                },
                context={
                    "sampling_rate_hz": float(self.sampling_rate),
                    "fundamental_freq_hz": float(fundamental_freq),
                    "fundamental_power": float(fundamental_power),
                    "schumann_harmonics_hz": [float(f) for f in self.schumann_frequencies],
                },
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
            cycle_correlation=cycle_correlation,
            diagnostics=diagnostics,
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

    def load_neural_weights(self, state_dict: dict[str, Any] | str | None = None) -> None:
        """Load trained weights for the CNN-LSTM analyser and enable it.

        Once trained weights exist (a labelled Schumann corpus is required to
        produce them transparently), this activates the learned classifier path in
        :meth:`detect_resonance_anomaly`.

        Historically this hook loaded a bare ``state_dict`` (in memory or from
        a path) directly into ``harmonic_analyzer`` with no dict-key wrapping,
        unlike other hooks; that behavior is preserved. The training pipeline
        (``ml/hazard_training/schumann_harmonics.py``) ships a *wrapped*
        payload ``{"harmonic_analyzer": state_dict, "feature_spec": ...}``, so
        an explicit-path load now accepts both shapes, and calling with no
        argument loads the shipped ``schumann_sierra_nevada`` checkpoint
        (provenance verified and logged by ``load_shipped_checkpoint``).

        A wrapped payload may additionally carry a ratified
        ``operating_point`` (the validation-selected decision threshold for
        the learned path -- part of the deployed rule the merit gate
        evaluated). It is validated BEFORE any state mutates and applied to
        the ``anomaly_detected`` decision only; bare state_dicts and
        pre-convention payloads keep the historical physics-flag decision.

        Args:
            state_dict: An in-memory ``state_dict``, a path to a saved one
                (bare or wrapped payload), or None to load the shipped
                ``schumann_sierra_nevada`` checkpoint.

        Raises:
            FileNotFoundError: ``state_dict`` is None and no checkpoint has
                been shipped.
            RuntimeError: The checkpoint is corrupt, fails its provenance
                sha256 pin, or does not match the analyser architecture.
            ValueError: The payload carries an ``operating_point`` whose
                detection threshold is not a probability in (0, 1) -- a
                nonsensical decision rule must refuse, not load.
        """
        loaded: Any
        operating_point: dict[str, float] | None = None
        if state_dict is None:
            from omni_mercury_engine.models.checkpoint_paths import load_shipped_checkpoint

            payload, _provenance = load_shipped_checkpoint("schumann_sierra_nevada")
            loaded = payload["harmonic_analyzer"]
            operating_point = self._validated_operating_point(payload.get("operating_point"))
        else:
            loaded = state_dict
            if isinstance(state_dict, str):
                loaded = torch.load(state_dict, map_location="cpu", weights_only=True)
            if isinstance(loaded, dict) and "harmonic_analyzer" in loaded:
                # Wrapped training-pipeline payload; a bare analyser state_dict
                # only ever carries parameter names like "cnn_encoder.0.weight".
                operating_point = self._validated_operating_point(loaded.get("operating_point"))
                loaded = loaded["harmonic_analyzer"]
        self.harmonic_analyzer.load_state_dict(loaded)
        self.harmonic_analyzer.eval()
        self._operating_point = operating_point
        self._neural_trained = True
        self.logger.info(
            "Schumann CNN-LSTM weights loaded; learned classifier enabled%s.",
            (
                f" (decision operating point tau=" f"{operating_point['detection_threshold']:.4f})"
                if operating_point is not None
                else " (no operating point; physics flags keep the decision)"
            ),
        )

    @staticmethod
    def _validated_operating_point(op: Any) -> dict[str, float] | None:
        """Validate a payload's ``operating_point`` before any state mutates.

        Args:
            op: The payload's ``operating_point`` entry (or None).

        Returns:
            ``{"detection_threshold": tau}`` or None when absent.

        Raises:
            ValueError: ``tau`` is not a finite probability in (0, 1).
        """
        if op is None:
            return None
        tau = float(op["detection_threshold"])
        if not np.isfinite(tau) or not (0.0 < tau < 1.0):
            raise ValueError(
                f"checkpoint operating point detection threshold {tau} is not a "
                "probability; refusing a nonsensical decision rule"
            )
        return {"detection_threshold": tau}

    def _compute_power_spectrum(
        self, elf_signal: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Compute power spectrum using FFT (O(n log n) complexity)."""
        n = len(elf_signal)

        yf = fft(elf_signal)
        power = np.abs(yf[: n // 2]) ** 2
        xf = fftfreq(n, 1.0 / self.sampling_rate)[: n // 2]

        # Normalise by the spectral peak, but guard the degenerate window: a
        # flat or all-zero ELF segment has ``max(power) == 0`` (and an
        # ``n < 2`` window has an empty ``power``), so an unguarded
        # ``power / np.max(power)`` produces NaN/Inf that then poisons the
        # downstream fundamental-frequency search (peak-picking over NaNs).
        # Leave such a spectrum at zero -- a truthful "no resonance power"
        # result -- instead of propagating non-finite values.
        peak = float(np.max(power)) if power.size else 0.0
        if peak > 0.0 and np.isfinite(peak):
            power = power / peak

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

        threshold = self.golden_ratio * std_power

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

        return bool(abs(ratio - expected_ratio) > (0.2 * self.golden_ratio))

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

    def _correlate_cycle_patterns(
        self,
        fundamental_freq: float,
        temporal_pattern: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Correlate with cyclic geophysical periods (lunar tidal / solar cycle).

        EXPLORATORY: the returned correlation is context only -- it does not
        feed the anomaly decision or risk score. Tidal triggering of
        seismicity is a small, contested effect (Cochran et al. 2004;
        Ide et al. 2016) and solar-cycle modulation of geomagnetic activity,
        while established, is not a validated event predictor.
        """
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
                    "Measurement span near lunar tidal period (draconic 27.21 d / "
                    "synodic 29.53 d); tidal triggering of seismicity is small and "
                    "contested (exploratory)"
                )

        if fundamental_freq > 7.83:
            correlations["symbolic_significance"].append(
                "Elevated resonance: consistent with increased solar activity "
                "(solar-cycle modulation of geomagnetic activity; exploratory)"
            )

        return correlations

    def fetch_live_data(
        self,
        *,
        allow_simulated: bool = False,
        raw_samples: np.ndarray[Any, Any] | None = None,
    ) -> LiveFetch:
        """Fetch Schumann-resonance data points through the injected client.

        Args:
            allow_simulated: Explicit opt-in required when the client has no
                instrument record and therefore emits labelled-simulated data.
            raw_samples: Optional raw ELF record from the caller's instrument;
                when supplied the client runs its real Welch DSP over it and
                the fetch is live (never cached -- each record is distinct).

        Returns:
            Provenance-checked LiveFetch.

        Raises:
            LiveDataError: No client injected, or the fetch failed.
            SimulatedDataError: Simulated data without the explicit opt-in.
        """
        client = require_live_client(self._elf_source, "SchumannResonanceDetector", "BGS ELF")
        return fetch_live_datapoints(
            client,
            allow_simulated=allow_simulated,
            raw_samples=raw_samples,
            sampling_rate_hz=self.sampling_rate,
            # Never serve a cached point for a caller-supplied record; and a
            # simulated point is regenerated deterministically anyway.
            use_cache=raw_samples is None,
        )

    def detect_live(
        self,
        raw_samples: np.ndarray[Any, Any] | None = None,
        *,
        allow_simulated: bool = False,
        temporal_history: list[np.ndarray[Any, Any]] | None = None,
    ) -> SchumannAnomalyResult:
        """Run anomaly detection on a live/instrument ELF record.

        Maps the fetched DataPoint onto the existing
        :meth:`detect_resonance_anomaly` input contract: the client's data
        point carries the exact ELF record its Welch powers were computed
        from (``data["elf_record"]``), and this detector runs its own FFT
        pipeline over that record. No waveform is ever invented -- with no
        instrument record the client's record is labelled simulated and this
        method refuses it unless ``allow_simulated=True``.

        Args:
            raw_samples: Optional raw ELF record from the caller's instrument.
            allow_simulated: Explicit opt-in for the labelled-simulated path.
            temporal_history: Optional historical ELF records.

        Returns:
            SchumannAnomalyResult with ``source_id`` / ``data_provenance`` /
            ``live_context`` populated.

        Raises:
            LiveDataError: No client injected, the fetch failed, or the data
                point carried no ELF record.
            SimulatedDataError: Simulated data without the explicit opt-in.
        """
        fetch = self.fetch_live_data(allow_simulated=allow_simulated, raw_samples=raw_samples)
        if not fetch.data_points:
            raise LiveDataError(f"{fetch.source_id}: fetch returned no data points")
        point = max(fetch.data_points, key=lambda dp: dp.timestamp)

        record = point.data.get("elf_record")
        if record is None:
            raise LiveDataError(
                f"{fetch.source_id}: data point carries no ELF record; refusing to "
                f"invent a waveform from summary powers."
            )
        signal = np.asarray(record, dtype=float)

        result = self.detect_resonance_anomaly(
            signal, temporal_history=temporal_history, metadata=dict(point.metadata)
        )
        result.source_id = fetch.source_id
        result.data_provenance = fetch.data_provenance
        result.live_context = {
            "station": point.data.get("station"),
            "welch_power_spectrum": point.data.get("power_spectrum"),
            "sampling_rate_hz": point.data.get("sampling_rate_hz"),
            "n_samples": point.data.get("n_samples"),
            "record_provenance": point.metadata.get("data_provenance"),
        }
        return result

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


def create_omni_resonance_scalars() -> dict[str, float]:
    """Create doctorate-level Schumann resonance scalars.

    Returns:
        Dictionary of omni-resonance scalars with golden ratio optimization
    """
    phi = 1.618

    return {
        "omni_electromagnetic_harmony": 1.46 * phi,
        "omni_ionospheric_coherence": 1.42 * phi,
        "omni_planetary_resonance": 1.44 * phi,
        "omni_seismic_precursor_detection": 1.48 * phi,
        "omni_space_weather_correlation": 1.40 * phi,
        "omni_frequency_stability": 1.38 * phi,
        "omni_amplitude_sensitivity": 1.43 * phi,
        "omni_geophysical_cycle_alignment": 1.37 * phi,
        "omni_waveguide_propagation": 1.39 * phi,
        "omni_solar_modulation": 1.41 * phi,
    }
