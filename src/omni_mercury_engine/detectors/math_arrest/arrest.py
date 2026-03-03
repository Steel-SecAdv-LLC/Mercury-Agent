# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""AnomalyMathArrest: 17-probe mathematically-independent equation ensemble.

Replaces IsolationForest with transparent, auditable anomaly detection.
Every detection traces to a specific mathematical violation in one or more
of the 17 probe equations.
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
from omni_mercury_engine.detectors.math_arrest.probes.additive_harmonic import (
    AdditiveHarmonicProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.annealed_zscore import (
    AnnealedZScoreProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.boltzmann_coupling import (
    BoltzmannCouplingProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.catalan_decay import (
    CatalanDecayProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.energy_minimization import (
    EnergyMinimizationProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.ethical_iqr import (
    EthicalIQRProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.fractal_similarity import (
    FractalSelfSimilarityProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.helix import (
    HelixMultiplicativeProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.lyapunov_chaos import (
    LyapunovChaosProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.momentum import (
    MomentumProbe,
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
    "AdditiveHarmonicProbe": AdditiveHarmonicProbe,
    "MomentumProbe": MomentumProbe,
    "VarianceAdaptedProbe": VarianceAdaptedProbe,
    "EthicalIQRProbe": EthicalIQRProbe,
    "CatalanDecayProbe": CatalanDecayProbe,
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
    "AnnealedZScoreProbe": AnnealedZScoreProbe,
    "BoltzmannCouplingProbe": BoltzmannCouplingProbe,
}

_ALL_PROBE_NAMES: list[str] = list(_PROBE_REGISTRY.keys())

# ---------------------------------------------------------------------------
# Probe presets: curated subsets for common use-cases
# ---------------------------------------------------------------------------
PROBE_PRESETS: dict[str, list[str]] = {
    "all": _ALL_PROBE_NAMES,
    "robust": [
        "AdditiveHarmonicProbe",
        "VarianceAdaptedProbe",
        "EthicalIQRProbe",
        "AnnealedZScoreProbe",
    ],
    "frequency": [
        "AdditiveHarmonicProbe",
        "WavePropagationProbe",
        "ZetaHarmonicProbe",
        "QuantumSuperpositionProbe",
        "FractalSelfSimilarityProbe",
    ],
    "chaos": [
        "LyapunovChaosProbe",
        "R3RecursionResonanceProbe",
        "BoltzmannCouplingProbe",
        "AnnealedZScoreProbe",
        "EnergyMinimizationProbe",
    ],
    "minimal": [
        "AdditiveHarmonicProbe",
        "AnnealedZScoreProbe",
        "EthicalIQRProbe",
    ],
    "forensic": _ALL_PROBE_NAMES,
    # Domain-semantic presets (top-down prior, complements geometry routing)
    "infrastructure": [
        "AdditiveHarmonicProbe",
        "WavePropagationProbe",
        "ZetaHarmonicProbe",
        "MomentumProbe",
        "EthicalIQRProbe",
        "LyapunovChaosProbe",
    ],
    "medical": [
        "AdditiveHarmonicProbe",
        "WavePropagationProbe",
        "VarianceAdaptedProbe",
        "BoltzmannCouplingProbe",
        "EthicalIQRProbe",
        "R3RecursionResonanceProbe",
    ],
    "humanitarian": [
        "EthicalIQRProbe",
        "AnnealedZScoreProbe",
        "SVDProjectionProbe",
        "AdditiveHarmonicProbe",
        "TopologyHomologyProbe",
    ],
    "security": [
        "R3RecursionResonanceProbe",
        "BoltzmannCouplingProbe",
        "LyapunovChaosProbe",
        "EthicalIQRProbe",
        "SVDProjectionProbe",
        "AnnealedZScoreProbe",
    ],
    "environmental": [
        "WavePropagationProbe",
        "AdditiveHarmonicProbe",
        "FractalSelfSimilarityProbe",
        "LyapunovChaosProbe",
        "EnergyMinimizationProbe",
        "EthicalIQRProbe",
    ],
    "financial": [
        "EthicalIQRProbe",
        "AnnealedZScoreProbe",
        "SVDProjectionProbe",
        "CatalanDecayProbe",
        "VarianceAdaptedProbe",
    ],
    "tabular": [
        "EthicalIQRProbe",
        "AnnealedZScoreProbe",
        "VarianceAdaptedProbe",
        "SVDProjectionProbe",
        "AdditiveHarmonicProbe",
        "BoltzmannCouplingProbe",
    ],
}


class AnomalyMathArrest:
    """17-probe Anomaly Math Arrest.

    A mathematically-independent equation ensemble that replaces
    IsolationForest with transparent, auditable anomaly detection.
    Every detection traces to a specific mathematical violation in
    one or more of the 17 probe equations.

    Args:
        domain: Domain hint for affinity-based probe reordering.
        threshold: Decision threshold for :meth:`predict`.
        probes: Probe selection — a preset name (``"all"``,
            ``"robust"``, ``"frequency"``, ``"chaos"``, ``"minimal"``,
            ``"forensic"``), a list of probe class names, or a list
            of :class:`BaseEquationProbe` instances.  Defaults to
            ``"all"`` (all 17 probes).
    """

    def __init__(
        self,
        domain: str = "default",
        threshold: float = 0.5,
        probes: str | list[str] | list[BaseEquationProbe] | None = None,
        *,
        geometry_routing: bool = False,
    ) -> None:
        self._domain = domain
        self.threshold = threshold
        self._user_probe_spec = probes  # raw value from caller
        self._probes: list[BaseEquationProbe] = self._resolve_probes(probes)
        self._fusion = PhiWeightedFusion(n_probes=len(self._probes))
        self._decorrelator = CorrelationAwareDecorrelator()
        self._is_fitted: bool = False
        self._fit_qualities: dict[str, float] = {}
        self._geometry_routing: bool = geometry_routing
        self._detected_geometries: list[str] = []
        # Cache for decorrelation results — see calibrate_decorrelator() docstring.
        self._decorrelation_cache: dict[str, float] | None = None

    @property
    def detected_geometries(self) -> list[str]:
        """Geometry types detected during fit (empty before fit)."""
        return list(self._detected_geometries)

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

        # Invalidate decorrelation cache on re-fit so calibrate_decorrelator()
        # recomputes from scratch with the new probe state.
        self._decorrelation_cache = None

        # Option B: geometry-routing probe selection
        if self._geometry_routing and self._user_probe_spec is None:
            from omni_mercury_engine.detectors.math_arrest.geometry_classifier import (
                classify_geometry,
                probes_for_geometries,
            )

            geometries = classify_geometry(data)
            self._detected_geometries = list(geometries)
            probe_spec = probes_for_geometries(geometries)
            if probe_spec != ["all"]:
                # Replace probe list with geometry-selected subset
                self._probes = [
                    _PROBE_REGISTRY[name]() for name in probe_spec if name in _PROBE_REGISTRY
                ]
                self._fusion = PhiWeightedFusion(n_probes=len(self._probes))
                logger.info(
                    "Geometry routing: %s -> %d probes selected",
                    geometries,
                    len(self._probes),
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
        is_multivariate = data.ndim == 2 and data.shape[1] > 1
        results: list[ProbeResult] = []

        for probe in self._probes:
            if not probe.is_fitted:
                continue
            try:
                # Option A: use per-feature analysis for multivariate data.
                # Falls back to deviation_score for univariate or on failure.
                if is_multivariate:
                    pf_scores = probe.per_feature_scores(data)
                    result = ProbeResult(
                        probe_name=type(probe).__name__,
                        deviation_scores=pf_scores,
                        confidence=probe._fit_quality,
                        trajectory_fit_quality=probe._fit_quality,
                        anomaly_geometry="per_feature_max",
                    )
                else:
                    result = probe.deviation_score(data)
            except Exception as exc:
                logger.debug(
                    "Probe %s per_feature_scores failed: %s — falling back",
                    type(probe).__name__,
                    exc,
                )
                try:
                    result = probe.deviation_score(data)
                except Exception as exc2:
                    logger.warning(
                        "Probe %s deviation_score also failed: %s",
                        type(probe).__name__,
                        exc2,
                    )
                    continue
            results.append(result)

        if not results:
            return np.zeros(n_samples, dtype=np.float64)

        probe_names = [r.probe_name for r in results]
        affinity_order = get_affinity_order(self._domain, probe_names)

        return self._fusion.fuse(
            results,
            affinity_order=affinity_order,
            decorrelator=self._decorrelator,
        )

    def detect_per_probe(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return per-probe anomaly scores without fusion.

        Returns:
            Array of shape ``(n_samples, n_active_probes)`` with each
            column being one probe's raw deviation scores in [0, 1].
            Probes that failed to score return a column of 0.5 (uncertain).
        """
        if not self._is_fitted:
            raise RuntimeError("AnomalyMathArrest has not been fitted. Call fit() first.")
        n_samples = data.shape[0]
        n_probes = len(self._probes)
        score_matrix = np.full((n_samples, n_probes), 0.5, dtype=np.float64)

        for col_idx, probe in enumerate(self._probes):
            if not probe.is_fitted:
                continue
            try:
                result = probe.deviation_score(data)
                probe_scores = np.clip(result.deviation_scores[:n_samples], 0.0, 1.0)
                score_matrix[: len(probe_scores), col_idx] = probe_scores
            except Exception:
                pass  # leave column at 0.5 (uncertain)

        return score_matrix

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

        Results are cached on the instance after the first successful
        computation.  Subsequent calls within the same fitted state
        return the cached result without recomputing.  The cache is
        automatically invalidated when :meth:`fit` is called again.

        Caching is required because ``calibrate_decorrelator`` runs
        every fitted probe on *data* to build a score matrix and then
        computes an O(n_probes²) pairwise correlation matrix.  Without
        caching, benchmark CV loops and any code path that calls this
        method more than once per fitted instance would redundantly
        repeat the entire computation and re-emit "Redundant probe
        pair" warnings.

        Args:
            data: Calibration data.

        Returns:
            Weight multipliers dict (probe_name to float in ``(0, 1]``).
        """
        # Return cached result when available (invalidated on re-fit).
        if self._decorrelation_cache is not None:
            return dict(self._decorrelation_cache)

        n = data.shape[0]
        if n < MIN_SAMPLES_FOR_DECORRELATION:
            logger.debug(
                "Decorrelator calibration skipped: %d samples < %d minimum.",
                n,
                MIN_SAMPLES_FOR_DECORRELATION,
            )
            return {}

        # Option A: build score matrix using per-feature scores so that
        # redundancy detection reflects actual probe diversity, not the
        # artificial correlation from column-mean collapse.
        use_per_feature = data.ndim == 2 and data.shape[1] > 1

        results: list[ProbeResult] = []
        for probe in self._probes:
            if not probe.is_fitted:
                continue
            try:
                if use_per_feature:
                    pf = probe.per_feature_scores(data)
                    results.append(
                        ProbeResult(
                            probe_name=type(probe).__name__,
                            deviation_scores=pf,
                            confidence=probe._fit_quality,
                            trajectory_fit_quality=probe._fit_quality,
                            anomaly_geometry="per_feature_max",
                        )
                    )
                else:
                    result = probe.deviation_score(data)
                    results.append(result)
            except (RuntimeError, ValueError):
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

        result = self._decorrelator.calibrate(score_matrix, probe_names, fit_qualities)
        self._decorrelation_cache = dict(result)
        return result

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

    # Alias for backward compatibility
    redundancy_report = get_correlation_report

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
