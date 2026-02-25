# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""AnomalyMathRevolver: 21-probe mathematically-independent equation ensemble.

Replaces IsolationForest with transparent, auditable anomaly detection.
Every detection traces to a specific mathematical violation in one or more
of the 21 probe equations.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.detectors.math_revolver.base_probe import (
    MIN_SAMPLES,
    BaseEquationProbe,
    ProbeResult,
)
from omni_mercury_engine.detectors.math_revolver.domain_affinity import (
    get_affinity_order,
)
from omni_mercury_engine.detectors.math_revolver.fusion import (
    MIN_SAMPLES_FOR_DECORRELATION,
    CorrelationAwareDecorrelator,
    PhiWeightedFusion,
)
from omni_mercury_engine.detectors.math_revolver.probes.additive import (
    AdditiveProbe,
)
from omni_mercury_engine.detectors.math_revolver.probes.boltzmann_coupling import (
    BoltzmannCouplingProbe,
)
from omni_mercury_engine.detectors.math_revolver.probes.catalan import (
    CatalanOptimizedProbe,
)
from omni_mercury_engine.detectors.math_revolver.probes.energy_minimization import (
    EnergyMinimizationProbe,
)
from omni_mercury_engine.detectors.math_revolver.probes.ethical import (
    EthicalConstrainedProbe,
)
from omni_mercury_engine.detectors.math_revolver.probes.exponential import (
    ExponentialDecayProbe,
)
from omni_mercury_engine.detectors.math_revolver.probes.fractal_similarity import (
    FractalSelfSimilarityProbe,
)
from omni_mercury_engine.detectors.math_revolver.probes.harmonic import (
    HarmonicOscillatorProbe,
)
from omni_mercury_engine.detectors.math_revolver.probes.helix import (
    HelixMultiplicativeProbe,
)
from omni_mercury_engine.detectors.math_revolver.probes.iqr_robust import (
    IQRRobustProbe,
)
from omni_mercury_engine.detectors.math_revolver.probes.lyapunov_chaos import (
    LyapunovChaosProbe,
)
from omni_mercury_engine.detectors.math_revolver.probes.modified_zscore import (
    ModifiedZScoreProbe,
)
from omni_mercury_engine.detectors.math_revolver.probes.momentum import (
    MomentumProbe,
)
from omni_mercury_engine.detectors.math_revolver.probes.quantum_annealing import (
    QuantumAnnealingProbe,
)
from omni_mercury_engine.detectors.math_revolver.probes.quantum_superposition import (
    QuantumSuperpositionProbe,
)
from omni_mercury_engine.detectors.math_revolver.probes.r3_recursion import (
    R3RecursionResonanceProbe,
)
from omni_mercury_engine.detectors.math_revolver.probes.svd_projection import (
    SVDProjectionProbe,
)
from omni_mercury_engine.detectors.math_revolver.probes.topology_homology import (
    TopologyHomologyProbe,
)
from omni_mercury_engine.detectors.math_revolver.probes.variance_adapted import (
    VarianceAdaptedProbe,
)
from omni_mercury_engine.detectors.math_revolver.probes.wave_propagation import (
    WavePropagationProbe,
)
from omni_mercury_engine.detectors.math_revolver.probes.zeta_harmonic import (
    ZetaHarmonicProbe,
)

logger = logging.getLogger(__name__)


class AnomalyMathRevolver:
    """21-probe Anomaly Math Revolver.

    A mathematically-independent equation ensemble that replaces
    IsolationForest with transparent, auditable anomaly detection.
    Every detection traces to a specific mathematical violation in
    one or more of the 21 probe equations.

    Args:
        domain: Domain hint for affinity-based probe reordering.
        threshold: Decision threshold for :meth:`predict`.
    """

    def __init__(
        self,
        domain: str = "default",
        threshold: float = 0.5,
    ) -> None:
        self._domain = domain
        self.threshold = threshold
        self._probes: list[BaseEquationProbe] = self._init_probes()
        self._fusion = PhiWeightedFusion(n_probes=len(self._probes))
        self._decorrelator = CorrelationAwareDecorrelator()
        self._is_fitted: bool = False
        self._fit_qualities: dict[str, float] = {}

    @staticmethod
    def _init_probes() -> list[BaseEquationProbe]:
        """Instantiate all 21 probes in canonical order."""
        return [
            AdditiveProbe(),                   # 1
            HarmonicOscillatorProbe(),         # 2
            MomentumProbe(),                   # 3
            VarianceAdaptedProbe(),            # 4
            EthicalConstrainedProbe(),         # 5
            CatalanOptimizedProbe(),           # 6
            ExponentialDecayProbe(),           # 7
            HelixMultiplicativeProbe(),        # 8
            R3RecursionResonanceProbe(),       # 9
            SVDProjectionProbe(),              # 10
            LyapunovChaosProbe(),              # 11
            TopologyHomologyProbe(),           # 12
            FractalSelfSimilarityProbe(),      # 13
            ZetaHarmonicProbe(),               # 14
            WavePropagationProbe(),            # 15
            QuantumSuperpositionProbe(),       # 16
            EnergyMinimizationProbe(),         # 17
            QuantumAnnealingProbe(),           # 18
            BoltzmannCouplingProbe(),          # 19
            IQRRobustProbe(),                  # 20
            ModifiedZScoreProbe(),             # 21
        ]

    def fit(self, data: npt.NDArray[np.float64]) -> AnomalyMathRevolver:
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
            raise ValueError(
                f"AnomalyMathRevolver requires at least {MIN_SAMPLES} "
                f"samples, got {n}."
            )

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

        # Collect fit qualities from fitted probes
        for probe in self._probes:
            if probe.is_fitted:
                try:
                    result = probe.deviation_score(data)
                    self._fit_qualities[result.probe_name] = (
                        result.trajectory_fit_quality
                    )
                except (RuntimeError, ValueError):
                    pass

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
            raise RuntimeError(
                "AnomalyMathRevolver has not been fitted. Call fit() first."
            )

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
        score_matrix = np.column_stack(
            [r.deviation_scores[:min_len] for r in results]
        )
        probe_names = [r.probe_name for r in results]
        fit_qualities = {
            r.probe_name: r.trajectory_fit_quality for r in results
        }

        return self._decorrelator.calibrate(
            score_matrix, probe_names, fit_qualities
        )

    def get_probe_diagnostics(self) -> list[dict[str, Any]]:
        """Per-probe fit quality and status for full transparency."""
        diagnostics: list[dict[str, Any]] = []
        for probe in self._probes:
            diagnostics.append({
                "probe_class": type(probe).__name__,
                "is_fitted": probe.is_fitted,
                "min_samples": probe._min_samples,
            })
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
