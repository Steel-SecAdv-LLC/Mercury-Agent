# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""AnomalyMathArrest: 21-probe mathematically-independent equation ensemble.

Replaces IsolationForest with transparent, auditable anomaly detection.
Every detection traces to a specific mathematical violation in one or more
of the 21 probe equations.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.detectors.math_arrest.base_probe import (
    MIN_SAMPLES,
    BaseEquationProbe,
    ProbeResult,
)
from omni_mercury_engine.detectors.math_arrest.domain_affinity import (
    get_affinity_order,
)
from omni_mercury_engine.detectors.math_arrest.fusion import (
    MIN_SAMPLES_FOR_DECORRELATION,
    CorrelationAwareDecorrelator,
    PhiWeightedFusion,
)
from omni_mercury_engine.detectors.math_arrest.probes.additive import (
    AdditiveProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.boltzmann_coupling import (
    BoltzmannCouplingProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.catalan import (
    CatalanOptimizedProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.energy_minimization import (
    EnergyMinimizationProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.ethical import (
    EthicalConstrainedProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.exponential import (
    ExponentialDecayProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.fractal_similarity import (
    FractalSelfSimilarityProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.harmonic import (
    HarmonicOscillatorProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.helix import (
    HelixMultiplicativeProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.iqr_robust import (
    IQRRobustProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.lyapunov_chaos import (
    LyapunovChaosProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.modified_zscore import (
    ModifiedZScoreProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.momentum import (
    MomentumProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.quantum_annealing import (
    QuantumAnnealingProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.quantum_superposition import (
    QuantumSuperpositionProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.r3_recursion import (
    R3RecursionResonanceProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.svd_projection import (
    SVDProjectionProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.topology_homology import (
    TopologyHomologyProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.variance_adapted import (
    VarianceAdaptedProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.wave_propagation import (
    WavePropagationProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.zeta_harmonic import (
    ZetaHarmonicProbe,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Probe registry: canonical name → class
# ---------------------------------------------------------------------------
_PROBE_REGISTRY: dict[str, type[BaseEquationProbe]] = {
    "AdditiveProbe": AdditiveProbe,
    "HarmonicOscillatorProbe": HarmonicOscillatorProbe,
    "MomentumProbe": MomentumProbe,
    "VarianceAdaptedProbe": VarianceAdaptedProbe,
    "EthicalConstrainedProbe": EthicalConstrainedProbe,
    "CatalanOptimizedProbe": CatalanOptimizedProbe,
    "ExponentialDecayProbe": ExponentialDecayProbe,
    "HelixMultiplicativeProbe": HelixMultiplicativeProbe,
    "R3RecursionResonanceProbe": R3RecursionResonanceProbe,
    "SVDProjectionProbe": SVDProjectionProbe,
    "LyapunovChaosProbe": LyapunovChaosProbe,
    "TopologyHomologyProbe": TopologyHomologyProbe,
    "FractalSelfSimilarityProbe": FractalSelfSimilarityProbe,
    "ZetaHarmonicProbe": ZetaHarmonicProbe,
    "WavePropagationProbe": WavePropagationProbe,
    "QuantumSuperpositionProbe": QuantumSuperpositionProbe,
    "EnergyMinimizationProbe": EnergyMinimizationProbe,
    "QuantumAnnealingProbe": QuantumAnnealingProbe,
    "BoltzmannCouplingProbe": BoltzmannCouplingProbe,
    "IQRRobustProbe": IQRRobustProbe,
    "ModifiedZScoreProbe": ModifiedZScoreProbe,
}

_ALL_PROBE_NAMES: list[str] = list(_PROBE_REGISTRY.keys())

# ---------------------------------------------------------------------------
# Probe presets: curated subsets for common use-cases
# ---------------------------------------------------------------------------
PROBE_PRESETS: dict[str, list[str]] = {
    "all": _ALL_PROBE_NAMES,
    "robust": [
        "AdditiveProbe",
        "VarianceAdaptedProbe",
        "EthicalConstrainedProbe",
        "IQRRobustProbe",
        "ModifiedZScoreProbe",
    ],
    "frequency": [
        "HarmonicOscillatorProbe",
        "WavePropagationProbe",
        "ZetaHarmonicProbe",
        "QuantumSuperpositionProbe",
        "FractalSelfSimilarityProbe",
    ],
    "chaos": [
        "LyapunovChaosProbe",
        "R3RecursionResonanceProbe",
        "BoltzmannCouplingProbe",
        "QuantumAnnealingProbe",
        "EnergyMinimizationProbe",
    ],
    "minimal": [
        "AdditiveProbe",
        "ModifiedZScoreProbe",
        "IQRRobustProbe",
    ],
    "forensic": _ALL_PROBE_NAMES,
}


class AnomalyMathArrest:
    """21-probe Anomaly Math Arrest.

    A mathematically-independent equation ensemble that replaces
    IsolationForest with transparent, auditable anomaly detection.
    Every detection traces to a specific mathematical violation in
    one or more of the 21 probe equations.

    Args:
        domain: Domain hint for affinity-based probe reordering.
        threshold: Decision threshold for :meth:`predict`.
        probes: Probe selection — a preset name (``"all"``,
            ``"robust"``, ``"frequency"``, ``"chaos"``, ``"minimal"``,
            ``"forensic"``), a list of probe class names, or a list
            of :class:`BaseEquationProbe` instances.  Defaults to
            ``"all"`` (all 21 probes).
    """

    def __init__(
        self,
        domain: str = "default",
        threshold: float = 0.5,
        probes: str | list[str] | list[BaseEquationProbe] | None = None,
    ) -> None:
        self._domain = domain
        self.threshold = threshold
        self._probes: list[BaseEquationProbe] = self._resolve_probes(probes)
        self._fusion = PhiWeightedFusion(n_probes=len(self._probes))
        self._decorrelator = CorrelationAwareDecorrelator()
        self._is_fitted: bool = False
        self._fit_qualities: dict[str, float] = {}

    @staticmethod
    def _resolve_probes(
        spec: str | list[str] | list[BaseEquationProbe] | None,
    ) -> list[BaseEquationProbe]:
        """Resolve a probe specification to a list of probe instances."""
        if spec is None:
            spec = "all"

        # Preset name
        if isinstance(spec, str):
            if spec not in PROBE_PRESETS:
                raise ValueError(
                    f"Unknown probe preset {spec!r}. Available: {sorted(PROBE_PRESETS.keys())}"
                )
            names = PROBE_PRESETS[spec]
            return [_PROBE_REGISTRY[n]() for n in names]

        # List of probe instances
        if spec and isinstance(spec[0], BaseEquationProbe):
            return list(spec)  # type: ignore[arg-type]

        # List of class name strings
        result: list[BaseEquationProbe] = []
        for name in spec:
            if name not in _PROBE_REGISTRY:
                raise ValueError(
                    f"Unknown probe name {name!r}. Available: {sorted(_PROBE_REGISTRY.keys())}"
                )
            result.append(_PROBE_REGISTRY[name]())  # type: ignore[index]
        return result

    def fit(self, data: npt.NDArray[np.float64]) -> AnomalyMathArrest:
        """Fit all probes to training data.

        Automatically calls :meth:`calibrate_decorrelator` at the end
        if ``n_samples >= MIN_SAMPLES_FOR_DECORRELATION``.

        Args:
            data: Training data, shape ``(n_samples,)`` or
                ``(n_samples, n_features)``.

        Returns:
            ``self`` (fluent interface).

        Raises:
            ValueError: If *data* is empty or has fewer than
                ``MIN_SAMPLES`` samples.
        """
        if data.size == 0:
            raise ValueError("Input data is empty.")
        n = data.shape[0]
        if n < MIN_SAMPLES:
            raise ValueError(f"AnomalyMathArrest requires at least {MIN_SAMPLES} samples, got {n}.")

        for probe in self._probes:
            try:
                probe.fit_trajectory(data)
            except (ValueError, RuntimeError) as exc:
                logger.warning(
                    "Probe %s failed to fit: %s",
                    type(probe).__name__,
                    exc,
                )

        self._is_fitted = True

        # Read fit quality directly from each probe (set during fit_trajectory)
        for probe in self._probes:
            if probe.is_fitted:
                self._fit_qualities[type(probe).__name__] = probe._fit_quality

        if n >= MIN_SAMPLES_FOR_DECORRELATION:
            self.calibrate_decorrelator(data)
        else:
            logger.debug(
                "Decorrelator skipped: %d samples < %d minimum.",
                n,
                MIN_SAMPLES_FOR_DECORRELATION,
            )

        return self

    def detect(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute per-sample anomaly scores in ``[0, 1]``.

        Probes that failed to fit are silently skipped. If all probes
        fail, returns a zero array (detection fails open).

        Args:
            data: Evaluation data.

        Returns:
            Array of shape ``(n_samples,)`` with scores in ``[0, 1]``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if not self._is_fitted:
            raise RuntimeError("AnomalyMathArrest has not been fitted. Call fit() first.")

        n_samples = data.shape[0]
        results: list[ProbeResult] = []

        for probe in self._probes:
            if not probe.is_fitted:
                continue
            try:
                result = probe.deviation_score(data)
                results.append(result)
            except (RuntimeError, ValueError) as exc:
                logger.warning(
                    "Probe %s failed during detection: %s",
                    type(probe).__name__,
                    exc,
                )

        if not results:
            return np.zeros(n_samples, dtype=np.float64)

        probe_names = [r.probe_name for r in results]
        affinity_order = get_affinity_order(self._domain, probe_names)

        return self._fusion.fuse(
            results,
            affinity_order=affinity_order,
            decorrelator=self._decorrelator,
        )

    def predict(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
        """Binary classification: 0=normal, 1=anomaly.

        Args:
            data: Evaluation data.

        Returns:
            Array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        scores = self.detect(data)
        return (scores >= self.threshold).astype(np.int32)

    def calibrate_threshold(
        self,
        data: npt.NDArray[np.float64],
        labels: npt.NDArray[np.int32],
        metric: str = "f1",
    ) -> float:
        """Grid search threshold on validation data.

        Args:
            data: Validation data.
            labels: Binary labels (0=normal, 1=anomaly).
            metric: One of ``"f1"``, ``"precision"``, ``"recall"``.

        Returns:
            Optimal threshold found.
        """
        scores = self.detect(data)
        best_threshold = self.threshold
        best_metric_val = -1.0

        for candidate in np.linspace(0.01, 0.99, 99):
            preds = (scores >= candidate).astype(np.int32)
            tp = int(np.sum((preds == 1) & (labels == 1)))
            fp = int(np.sum((preds == 1) & (labels == 0)))
            fn = int(np.sum((preds == 0) & (labels == 1)))

            if metric == "precision":
                val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            elif metric == "recall":
                val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            else:
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                val = (
                    2.0 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

            if val > best_metric_val:
                best_metric_val = val
                best_threshold = float(candidate)

        self.threshold = best_threshold
        return best_threshold

    def calibrate_decorrelator(
        self,
        data: npt.NDArray[np.float64],
    ) -> dict[str, float]:
        """Run all fitted probes on data and compute weight multipliers.

        Args:
            data: Calibration data.

        Returns:
            Weight multipliers dict (probe_name to float in ``(0, 1]``).
        """
        n = data.shape[0]
        if n < MIN_SAMPLES_FOR_DECORRELATION:
            logger.debug(
                "Decorrelator calibration skipped: %d samples < %d minimum.",
                n,
                MIN_SAMPLES_FOR_DECORRELATION,
            )
            return {}

        results: list[ProbeResult] = []
        for probe in self._probes:
            if not probe.is_fitted:
                continue
            try:
                result = probe.deviation_score(data)
                results.append(result)
            except (RuntimeError, ValueError):
                continue

        if not results:
            return {}

        min_len = min(len(r.deviation_scores) for r in results)
        score_matrix = np.column_stack([r.deviation_scores[:min_len] for r in results])
        probe_names = [r.probe_name for r in results]
        fit_qualities = {r.probe_name: r.trajectory_fit_quality for r in results}

        return self._decorrelator.calibrate(score_matrix, probe_names, fit_qualities)

    def get_probe_diagnostics(self) -> list[dict[str, Any]]:
        """Per-probe fit quality and status for full transparency."""
        diagnostics: list[dict[str, Any]] = []
        for probe in self._probes:
            diagnostics.append(
                {
                    "probe_class": type(probe).__name__,
                    "is_fitted": probe.is_fitted,
                    "min_samples": probe._min_samples,
                }
            )
        return diagnostics

    def get_correlation_report(self) -> dict[str, Any]:
        """Return correlation audit results.

        Returns:
            Dict with ``redundant_pairs``, ``weight_multipliers``, and
            ``effective_probe_count``.
        """
        return {
            "redundant_pairs": self._decorrelator.redundant_pairs,
            "weight_multipliers": self._decorrelator.weight_multipliers,
            "effective_probe_count": self._decorrelator.effective_probe_count,
        }

    def get_geometry_report(self, data: npt.NDArray[np.float64]) -> list[dict[str, Any]]:
        """Return per-probe anomaly geometry labels and scores.

        Each entry describes one probe's contribution to the
        ensemble signal, including the anomaly geometry label,
        mean deviation, and confidence.

        Args:
            data: Evaluation data.

        Returns:
            List of dicts with keys ``probe_name``,
            ``anomaly_geometry``, ``mean_deviation``,
            ``max_deviation``, and ``confidence``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if not self._is_fitted:
            raise RuntimeError("AnomalyMathArrest has not been fitted. Call fit() first.")

        report: list[dict[str, Any]] = []
        for probe in self._probes:
            if not probe.is_fitted:
                continue
            try:
                result = probe.deviation_score(data)
                report.append(
                    {
                        "probe_name": result.probe_name,
                        "anomaly_geometry": result.anomaly_geometry,
                        "mean_deviation": float(np.mean(result.deviation_scores)),
                        "max_deviation": float(np.max(result.deviation_scores)),
                        "confidence": result.confidence,
                    }
                )
            except (RuntimeError, ValueError):
                continue
        return report

    def score_window(
        self,
        data: npt.NDArray[np.float64],
        window_size: int = 10,
    ) -> npt.NDArray[np.float64]:
        """Compute windowed anomaly scores using a rolling mean.

        Smooths per-sample anomaly scores with a centered moving
        average of width *window_size*, making transient spikes
        easier to interpret in operational dashboards.

        Args:
            data: Evaluation data.
            window_size: Number of samples in the rolling window.
                Must be >= 1.

        Returns:
            Smoothed score array of shape ``(n_samples,)``, values
            in ``[0, 1]``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
            ValueError: If *window_size* < 1.
        """
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}.")
        raw_scores = self.detect(data)
        if window_size == 1:
            return raw_scores

        n = len(raw_scores)
        kernel = np.ones(window_size, dtype=np.float64) / window_size
        # Full convolution then trim to original length
        convolved = np.convolve(raw_scores, kernel, mode="same")
        return np.clip(convolved[:n], 0.0, 1.0)

    @property
    def ensemble_confidence(self) -> float:
        """Mean fit quality of all fitted probes."""
        if not self._fit_qualities:
            return 0.0
        values = list(self._fit_qualities.values())
        return float(np.mean(values))

    @property
    def active_probe_count(self) -> int:
        """Number of probes that successfully fit."""
        return sum(1 for p in self._probes if p.is_fitted)
