# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AnomalyMathArrest: 21-probe mathematically-independent equation ensemble.

Transparent, auditable anomaly detection in which every score traces to a
specific mathematical violation in one or more of the 21 probe equations.
Each probe targets a distinct anomaly geometry — trend and level shifts,
periodicity, curvature and jerk, spectral and information-geometric
structure, and chaos onset — and the probe outputs are aggregated into a
single calibrated score with per-probe attribution retained for
explainability.
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
    # The measured redundancy reduction. On the recorded corpora the fit-time
    # probe correlations form two clusters at mean |r| >= 0.80 — {additive,
    # harmonic_oscillator, r3_recursion_resonance, quantum_annealing,
    # modified_zscore} and {catalan_optimized, exponential_decay} — and keeping
    # only each cluster's strongest member (by mean held-out AUROC) measures
    # *better* than the full ensemble through the shipped fusion: mean AUROC
    # 0.7584 vs 0.7494 on the four NAB files, 0.6859 vs 0.6763 on the five
    # ADBench tables, mean rank 1.21 vs 1.79 over the 19 evaluations
    # (``research/math_arrest/harness_probe_redundancy.py``; the matrices are
    # in ``artifacts/math_arrest_probe_correlation.json`` and the selection is
    # pinned by ``tests/detectors/test_probe_redundancy.py``). The default
    # stays ``"all"``: the 21-column per-probe feature surface is a released
    # contract (``MathArrestDetector.extract_features`` feeds fusion
    # checkpoints sized to it), and the shipped decorrelator already
    # down-weights these clusters inside the full ensemble.
    "decorrelated": [
        "AdditiveProbe",
        "MomentumProbe",
        "VarianceAdaptedProbe",
        "EthicalConstrainedProbe",
        "CatalanOptimizedProbe",
        "HelixMultiplicativeProbe",
        "SVDProjectionProbe",
        "LyapunovChaosProbe",
        "TopologyHomologyProbe",
        "FractalSelfSimilarityProbe",
        "ZetaHarmonicProbe",
        "WavePropagationProbe",
        "QuantumSuperpositionProbe",
        "EnergyMinimizationProbe",
        "BoltzmannCouplingProbe",
        "IQRRobustProbe",
    ],
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
        weighting: How the active probes' scores are combined — see
            :class:`~omni_mercury_engine.detectors.math_arrest.fusion.PhiWeightedFusion`
            for the measured comparison. ``"uniform"`` (default, the plain
            mean) measures 0.7581 mean AUC on the four recorded NAB files and
            0.6826 on five ADBench tables, against 0.6519 / 0.5423 for the
            released ``"phi_affinity"`` prior.

    Modality: the probes are **series** instruments (Hankel trajectories,
    chaos/harmonic/wave equations), and ``BaseEquationProbe._to_1d`` reduces a
    2-D input to its per-row mean, so matrix input is scored on that row
    summary and the per-feature structure is not seen. The measured numbers
    above reflect exactly that behaviour on each corpus.
    """

    def __init__(
        self,
        domain: str = "default",
        threshold: float = 0.5,
        probes: str | list[str] | list[BaseEquationProbe] | None = None,
        weighting: str = "uniform",
    ) -> None:
        """Initialize the instance."""
        self._domain = domain
        self.threshold = threshold
        self._probes: list[BaseEquationProbe] = self._resolve_probes(probes)
        self._fusion = PhiWeightedFusion(n_probes=len(self._probes), weighting=weighting)
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

        # List of probe instances — verify every element is a
        # ``BaseEquationProbe`` (the union type ``list[str] |
        # list[BaseEquationProbe]`` permits a homogeneous list of either
        # type, but a caller-constructed mixed list would still satisfy
        # ``isinstance(spec[0], BaseEquationProbe)`` while slipping a
        # ``str`` element past mypy's invariant ``list`` narrowing).
        # Validate explicitly so the contract holds at runtime and the
        # element type is provably ``BaseEquationProbe`` for the type
        # checker as well.
        if spec and isinstance(spec[0], BaseEquationProbe):
            instances: list[BaseEquationProbe] = []
            for item in spec:
                if not isinstance(item, BaseEquationProbe):
                    raise TypeError(
                        "Heterogeneous probe list passed to _resolve_probes: "
                        "first element is BaseEquationProbe but element "
                        f"of type {type(item).__name__} found later in the list"
                    )
                instances.append(item)
            return instances

        # List of class name strings — the ``BaseEquationProbe`` instance
        # branch above already returned, so every element here is expected
        # to be a ``str``. Validate explicitly with ``raise TypeError``
        # rather than ``assert`` so the contract is enforced even when
        # Python is run with ``-O`` / ``PYTHONOPTIMIZE`` (asserts get
        # stripped). The check also serves as type narrowing for mypy.
        result: list[BaseEquationProbe] = []
        for name in spec:
            if not isinstance(name, str):
                raise TypeError(
                    f"Mixed probe spec lists are not supported (got {type(name).__name__})"
                )
            if name not in _PROBE_REGISTRY:
                raise ValueError(
                    f"Unknown probe name {name!r}. Available: {sorted(_PROBE_REGISTRY.keys())}"
                )
            result.append(_PROBE_REGISTRY[name]())
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
                continue
            # Record the fit-time score scale, so a published score means the
            # same thing on every later call. Without this each call divided by
            # the scored array's own 99th percentile, which left the alarm rate
            # a function of the caller's batch size rather than of the data
            # (measured: 0.052 whole-series to 0.461 at 32-point batches against
            # a cut promising alpha = 0.05). See
            # ``BaseEquationProbe._normalize_scores``.
            try:
                probe.calibrate_normalizer(data)
            except (ValueError, RuntimeError) as exc:
                logger.warning(
                    "Probe %s failed to calibrate its normalizer: %s",
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

    def per_probe_scores(self, data: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        """Each fitted probe's own deviation scores, by probe name.

        The breakdown behind :meth:`detect`'s fused score: every alarm traces
        to the probes that raised it, which is the auditability this ensemble
        exists for. A probe that failed to fit, or that raises while scoring,
        is omitted rather than contributing a fabricated column.

        Args:
            data: Evaluation data.

        Returns:
            ``{probe_name: scores}``; empty when nothing fitted.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if not self._is_fitted:
            raise RuntimeError("AnomalyMathArrest has not been fitted. Call fit() first.")
        out: dict[str, npt.NDArray[np.float64]] = {}
        for probe in self._probes:
            if not probe.is_fitted:
                continue
            try:
                result = probe.deviation_score(data)
            except (RuntimeError, ValueError) as exc:
                logger.warning(
                    "Probe %s failed during per-probe scoring: %s", type(probe).__name__, exc
                )
                continue
            out[result.probe_name] = np.asarray(result.deviation_scores, dtype=np.float64)
        return out

    def scored_results(self, data: npt.NDArray[np.float64]) -> list[ProbeResult]:
        """Every fitted probe's :class:`ProbeResult` on ``data``, one pass.

        The shared scoring step behind :meth:`detect`,
        :meth:`per_probe_scores` and :meth:`probe_attributions`: callers that
        need more than one of the fused score, the per-probe columns and the
        attributions can score the probes once and derive the rest
        (:meth:`fuse_results` and
        :meth:`~omni_mercury_engine.detectors.math_arrest.fusion.PhiWeightedFusion.leave_one_out_attribution`),
        instead of paying a full 21-probe pass per view. A probe that failed
        to fit, or that raises while scoring, is omitted and logged.

        Args:
            data: Evaluation data.

        Returns:
            One result per probe that fitted and scored.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if not self._is_fitted:
            raise RuntimeError("AnomalyMathArrest has not been fitted. Call fit() first.")
        results: list[ProbeResult] = []
        for probe in self._probes:
            if not probe.is_fitted:
                continue
            try:
                results.append(probe.deviation_score(data))
            except (RuntimeError, ValueError) as exc:
                logger.warning("Probe %s failed during scoring: %s", type(probe).__name__, exc)
        return results

    def fuse_results(self, results: list[ProbeResult]) -> npt.NDArray[np.float64]:
        """The exact :meth:`detect` fusion of already-scored probe results.

        Args:
            results: Output of :meth:`scored_results`.

        Returns:
            Fused scores in ``[0, 1]`` (empty array for no results).
        """
        if not results:
            return np.array([], dtype=np.float64)
        order = get_affinity_order(self._domain, [r.probe_name for r in results])
        return self._fusion.fuse(results, affinity_order=order, decorrelator=self._decorrelator)

    def probe_attributions(
        self, data: npt.NDArray[np.float64]
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Per-sample leave-one-probe-out attribution of the fused score.

        For every sample, each probe's attribution is the change in the fused
        score when that probe is ablated (its weight zeroed, the rest
        renormalised — under the default ``uniform`` weighting this is
        exactly the fusion of the remaining probes). Computed against the
        same effective weights, affinity order and decorrelator state that
        :meth:`detect` applies, so every attribution explains the score that
        was actually served.

        Args:
            data: Evaluation data.

        Returns:
            ``{probe_name: (n_samples,) attribution}``; empty when nothing
            fitted.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        results = self.scored_results(data)
        if not results:
            return {}
        order = get_affinity_order(self._domain, [r.probe_name for r in results])
        ordered, weights = self._fusion.effective_weights(
            results, affinity_order=order, decorrelator=self._decorrelator
        )
        return self._fusion.leave_one_out_attribution(ordered, weights)

    def refine_contrastive(
        self,
        data: npt.NDArray[np.float64],
        labels: npt.NDArray[np.int32],
        *,
        eta: float = 0.5,
        max_pairs: int = 20_000,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Probe-local re-weighting from the fused score's discordant pairs.

        On a labelled calibration set, the discordant pairs are the (anomaly,
        normal) pairs the fused score orders the wrong way. Each probe is
        scored by the fraction of those pairs it orders *correctly*; probes
        that repair disagreements get their decorrelator multiplier scaled up
        by ``1 + eta * (2r - 1)`` and probes that cause them get scaled down
        — a small, probe-local adjustment through the ensemble's existing
        weighting seam, never a change to any probe's mathematics.

        Opt-in and supervised: nothing changes unless this is called, and the
        caller owns the labelled data (a held-out calibration fold, never the
        evaluation set).

        Args:
            data: Labelled calibration rows/series.
            labels: Binary ground truth, 1 marking anomalies.
            eta: Adjustment size; ``0`` is a no-op.
            max_pairs: Pair-sample budget for the discordance estimate.
            seed: Seed for the pair sampling.

        Returns:
            ``{"discordant_before", "discordant_after", "repair_rates",
            "multipliers"}`` — the measured disagreement rate on the sampled
            pairs before and after the adjustment, per-probe repair rates and
            the updated multipliers.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
            ValueError: On single-class labels or a negative ``eta``.
        """
        if not self._is_fitted:
            raise RuntimeError("AnomalyMathArrest has not been fitted. Call fit() first.")
        if eta < 0.0:
            raise ValueError(f"eta must be >= 0, got {eta}")
        y = np.asarray(labels).reshape(-1).astype(np.int64)
        results = self.scored_results(data)
        if not results:
            return {
                "discordant_before": float("nan"),
                "discordant_after": float("nan"),
                "repair_rates": {},
                "multipliers": self._decorrelator.weight_multipliers,
            }
        width = min(len(y), *(len(r.deviation_scores) for r in results))
        matrix = np.column_stack([r.deviation_scores[:width] for r in results])
        names = [r.probe_name for r in results]
        y = y[:width]
        positives = np.where(y == 1)[0]
        negatives = np.where(y == 0)[0]
        if positives.size == 0 or negatives.size == 0:
            raise ValueError("contrastive refinement needs both classes in the labels")

        rng = np.random.default_rng(seed)
        count = min(max_pairs, positives.size * negatives.size)
        pair_pos = rng.choice(positives, size=count)
        pair_neg = rng.choice(negatives, size=count)

        def discordance() -> tuple[float, npt.NDArray[np.bool_]]:
            fused = self.fuse_results(results)[:width]
            discordant = fused[pair_pos] < fused[pair_neg]
            return float(discordant.mean()), discordant

        before, discordant = discordance()
        repair_rates: dict[str, float] = {}
        factors: dict[str, float] = {}
        if discordant.any() and not self._decorrelator.is_calibrated:
            # A fit on fewer than MIN_SAMPLES_FOR_DECORRELATION rows skipped
            # the calibration; the labelled calibration fold supplies it.
            self.calibrate_decorrelator(data)
        if discordant.any() and self._decorrelator.is_calibrated:
            pos_rows = pair_pos[discordant]
            neg_rows = pair_neg[discordant]
            for j, name in enumerate(names):
                rate = float((matrix[pos_rows, j] > matrix[neg_rows, j]).mean())
                repair_rates[name] = rate
                # Floored so an all-wrong probe is demoted, never zeroed:
                # a zero multiplier would amount to deleting the probe.
                factors[name] = max(1.0 + eta * (2.0 * rate - 1.0), 0.05)
            self._decorrelator.scale_multipliers(factors)
        after, _ = discordance()
        return {
            "discordant_before": before,
            "discordant_after": after,
            "repair_rates": repair_rates,
            "multipliers": self._decorrelator.weight_multipliers,
        }

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
