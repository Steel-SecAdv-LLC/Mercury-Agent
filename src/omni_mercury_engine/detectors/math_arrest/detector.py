# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``BaseDetector`` seam for the 21-probe math-arrest instrument.

``AnomalyMathArrest.detect`` returns a bare score array, which satisfies no
detector contract. This adapter supplies one without changing the instrument.

Measured NAB AUROC 0.7581 (``tests/detectors/test_recorded_math_arrest.py``).
A manifest entry, deliberately *not* in the engine's default set: that needs a
marginal-gain ablation. See ``docs/DORMANCY_LEDGER.md`` §2b.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.core.base import (
    BaseDetector,
    _deserialized_state,
    _serializable_state,
)
from omni_mercury_engine.core.vector_calibration import (
    VectorConformalBinary,
    VectorIsotonicCalibrator,
)
from omni_mercury_engine.detectors._calibration import (
    batch_evidence,
    bound_finite_config,
    conformal_cut,
    finite_features,
    finite_scores,
)
from omni_mercury_engine.detectors.math_arrest.arrest import AnomalyMathArrest
from omni_mercury_engine.detectors.provenance import DecisionSource

if TYPE_CHECKING:
    import torch

__all__ = ["MathArrestDetector"]


class MathArrestDetector(BaseDetector):
    """The 21-probe equation ensemble as a first-class Mercury detector.

    The 21 per-probe deviation scores are served as the fusion feature group,
    so every alarm traces to the probes that raised it.
    """

    #: Share of the fit series the installed operating point leaves above it.
    CALIBRATION_ALPHA = 0.05

    decision_source = DecisionSource(
        kind="experimental",
        description=(
            "In-house 21-equation ensemble (Hankel/SVD trajectory, chaos, "
            "harmonic, wave, robust-location and energy probes) fused by the "
            "plain mean of the probes that fitted, the measured-best of its "
            "three label-free weightings. The alarm fires at the one-class "
            "split-conformal quantile of the fit series' own fused scores "
            "(alpha 0.05), so the false-alarm rate on data exchangeable with "
            "the fit series is bounded by alpha rather than resting on a fixed "
            "cut; an explicitly configured threshold is honoured exactly. No "
            "published threshold exists for the fused score. Measured mean "
            "AUROC 0.7581 over the four recorded NAB files and 0.6826 over "
            "five ADBench tables; the probes are series equations, and a 2-D "
            "input is reduced to its per-row mean before scoring."
        ),
    )

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        domain: str = "default",
        probes: str | list[str] | None = None,
        weighting: str = "uniform",
    ) -> None:
        """Initialize the adapter.

        Args:
            config: Optional ``BaseDetector`` config (``threshold``,
                ``auto_calibrate``, ...). A configured ``threshold`` is the
                caller's decision and survives :meth:`fit`.
            domain: Domain hint passed to the instrument's affinity ordering
                (it only changes the score under the ``phi_affinity``
                weighting).
            probes: Probe selection — a preset name or a list of probe class
                names. Defaults to all 21.
            weighting: Probe-score weighting; see
                :class:`~omni_mercury_engine.detectors.math_arrest.fusion.PhiWeightedFusion`
                for the measured comparison.
        """
        super().__init__(config)
        self._arrest = AnomalyMathArrest(
            domain=domain, threshold=self.threshold, probes=probes, weighting=weighting
        )
        self._threshold_is_configured = isinstance(config, dict) and "threshold" in config
        #: Sorted fused scores of the fit series: the reference the operating
        #: point and the conformal p-values are taken from.
        self._reference_scores: np.ndarray[Any, Any] | None = None
        #: Opt-in supervised vector calibration over the (probes + fused)
        #: score matrix; ``None`` until :meth:`fit_vector_calibration`.
        self._vector_calibrator: VectorIsotonicCalibrator | None = None
        self._vector_conformal: VectorConformalBinary | None = None

    def is_fitted(self) -> bool:
        """Return ``True`` once the instrument's probes have been fitted."""
        return self._is_fitted

    def _series(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Coerce input to a finite float64 array the probes accept."""
        detach = getattr(data, "detach", None)
        if callable(detach):
            data = detach().cpu().numpy()
        arr = bound_finite_config(self, np.asarray(data, dtype=np.float64))
        return arr if arr.ndim == 2 else arr.reshape(-1)

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> MathArrestDetector:
        """Fit every probe, then cut the operating point on the fit series' scores.

        Args:
            data: Training series (predominantly normal). A 2-D input is
                reduced to its per-row mean by the probes themselves.

        Returns:
            ``self``.
        """
        series = self._series(data)
        self._arrest.fit(series)
        self._is_fitted = True
        fused = np.asarray(self._arrest.detect(series), dtype=np.float64).reshape(-1)
        fused = fused[np.isfinite(fused)]
        self._reference_scores = np.sort(fused) if fused.size else None
        if not self._threshold_is_configured and not self._auto_calibrate:
            cut = conformal_cut(self._reference_scores, self.CALIBRATION_ALPHA)
            if cut is not None:
                self.threshold = cut
                self._arrest.threshold = cut
        return self

    def _learned_state(self) -> dict[str, Any]:
        """Hand the checkpoint the instrument's fitted state, probe by probe.

        The shared packer refuses to serialize the ``AnomalyMathArrest`` object
        itself, so this names what is actually learned.

        Returns:
            All probes' state, the decorrelator calibration, the fit-quality
            map and the conformal reference.
        """
        learned = {
            "probes": {type(probe).__name__: dict(vars(probe)) for probe in self._arrest._probes},
            "fit_qualities": dict(self._arrest._fit_qualities),
            "decorrelator": dict(vars(self._arrest._decorrelator)),
            "reference_scores": self._reference_scores,
            "arrest_threshold": float(self._arrest.threshold),
            "arrest_is_fitted": bool(self._arrest._is_fitted),
            "vector_calibrator": (
                None if self._vector_calibrator is None else self._vector_calibrator.to_state()
            ),
            "vector_conformal": (
                None if self._vector_conformal is None else self._vector_conformal.to_state()
            ),
        }
        # Stored verbatim, so it must pack its own numpy: the weights_only
        # unpickler refuses numpy's reconstruction globals.
        return dict(_serializable_state(learned, owner=type(self).__name__, attr="learned"))

    def _restore_learned_state(self, state: dict[str, Any]) -> None:
        """Write a checkpoint's state back onto the live probes.

        Args:
            state: Payload from :meth:`_learned_state`.
        """
        state = _deserialized_state(state)
        probes = state.get("probes", {})
        for probe in self._arrest._probes:
            saved = probes.get(type(probe).__name__)
            if saved:
                vars(probe).update(saved)
        self._arrest._fit_qualities = dict(state.get("fit_qualities", {}))
        vars(self._arrest._decorrelator).update(state.get("decorrelator", {}))
        reference = state.get("reference_scores")
        self._reference_scores = None if reference is None else np.asarray(reference)
        self._arrest.threshold = float(state.get("arrest_threshold", self.threshold))
        self._arrest._is_fitted = bool(state.get("arrest_is_fitted", False))
        self._is_fitted = self._arrest._is_fitted
        calibrator_state = state.get("vector_calibrator")
        conformal_state = state.get("vector_conformal")
        if calibrator_state is not None:
            self._vector_calibrator = VectorIsotonicCalibrator.from_state(calibrator_state)
            if conformal_state is not None:
                self._vector_conformal = VectorConformalBinary.from_state(
                    conformal_state, self._vector_calibrator
                )

    def _vector_matrix(self, series: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """The (probes + fused) score matrix the vector calibration consumes.

        One column per fitted probe (the instrument's own breakdown) plus the
        fused score as the final column — the full vector of Mercury's probe
        and detector scores, never an external feature.
        """
        results = self._arrest.scored_results(series)
        fused = np.asarray(self._arrest.fuse_results(results), dtype=np.float64).reshape(-1)
        return self._matrix_from(results, fused, n_rows=len(series))

    @staticmethod
    def _matrix_from(
        results: list[Any], fused: np.ndarray[Any, Any], n_rows: int
    ) -> np.ndarray[Any, Any]:
        """Assemble the calibration matrix from an existing scoring pass."""
        if fused.size == 0:
            fused = np.zeros(n_rows, dtype=np.float64)
        if not results:
            return fused.reshape(-1, 1)
        width = min(len(fused), *(len(r.deviation_scores) for r in results))
        stacked = [np.asarray(r.deviation_scores[:width], dtype=np.float64) for r in results]
        stacked.append(fused[:width])
        return np.column_stack(stacked)

    def fit_vector_calibration(
        self,
        data: np.ndarray[Any, Any] | torch.Tensor,
        labels: np.ndarray[Any, Any],
        *,
        alpha: float = 0.1,
        seed: int = 0,
    ) -> MathArrestDetector:
        """Fit the opt-in vector calibration on a labelled calibration set.

        Two instruments over the (probes + fused) score matrix, trained on
        Mercury's own outputs only:

        * a :class:`VectorIsotonicCalibrator` mapping the weighted probe
          index to a calibrated anomaly probability, and
        * a :class:`VectorConformalBinary` producing {normal, anomaly}
          prediction sets with the finite-sample ``1 − alpha`` per-class
          coverage guarantee.

        The labelled set is split in two internally (stratified, seeded):
        the first half fits the calibrator, the second supplies the conformal
        nonconformity scores — reusing the calibrator's own fitting fold for
        conformal calibration would break the exchangeability the guarantee
        rests on. Once fitted, :meth:`detect` applies the calibrated
        probability in the decision path and serves the prediction sets.

        Args:
            data: Labelled calibration rows/series (never the fit series).
            labels: Binary ground truth, 1 marking anomalies.
            alpha: Conformal miscoverage level.
            seed: Seed for the internal stratified split.

        Returns:
            ``self``.

        Raises:
            RuntimeError: If the detector has not been fitted.
            ValueError: If a class is missing from either internal fold.
        """
        if not self._is_fitted:
            raise RuntimeError("MathArrestDetector has not been fitted. Call fit() first.")
        series = self._series(data)
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        matrix = self._vector_matrix(series)
        if len(y) < len(matrix):
            raise ValueError(f"labels ({len(y)}) shorter than scored rows ({len(matrix)})")
        y = y[: len(matrix)]

        rng = np.random.RandomState(seed)
        first = np.zeros(len(y), dtype=bool)
        for cls in (0, 1):
            rows = np.where(y == cls)[0]
            rng.shuffle(rows)
            first[rows[: len(rows) // 2]] = True
        if not (first.any() and (~first).any()):
            raise ValueError("calibration set too small to split into two labelled folds")

        calibrator = VectorIsotonicCalibrator().fit(matrix[first], y[first])
        conformal = VectorConformalBinary(alpha=alpha).fit(matrix[~first], y[~first], calibrator)
        self._vector_calibrator = calibrator
        self._vector_conformal = conformal
        return self

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Per-sample fusion features: one column per probe that fitted.

        Args:
            data: Input series.

        Returns:
            ``(n_samples, n_probes)`` float32 per-probe deviation scores — the
            interpretable breakdown behind the fused score.
        """
        series = self._series(data)
        columns = self._arrest.per_probe_scores(series)
        if not columns:
            n = len(series)
            return np.zeros((n, 0), dtype=np.float32)
        width = min(len(c) for c in columns.values())
        matrix = np.column_stack(
            [np.asarray(c[:width], dtype=np.float64) for c in columns.values()]
        )
        return finite_features(matrix, detector=self.name).astype(np.float32)

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Per-sample fused anomaly scores in ``[0, 1]``.

        The verdict is the fused score against the fit-time conformal cut (or
        the caller's threshold). ``probe_names`` and ``probe_scores`` carry the
        per-probe breakdown, and the conformal ``p_values`` / ``fdr_flags`` /
        ``batch_anomaly`` fields are present once a fit reference exists.
        """
        series = self._series(data)
        # One 21-probe scoring pass serves the fused score, the per-probe
        # columns and the attributions (it used to be one pass per view).
        results = self._arrest.scored_results(series)
        fused = np.asarray(self._arrest.fuse_results(results), dtype=np.float64).reshape(-1)
        if fused.size == 0:
            fused = np.zeros(len(series), dtype=np.float64)
        scores = finite_scores(fused, detector=self.name).astype(np.float32)
        columns = {r.probe_name: r.deviation_scores for r in results}
        attributions: dict[str, Any] = {}
        if results:
            from omni_mercury_engine.detectors.math_arrest.domain_affinity import (
                get_affinity_order,
            )

            order = get_affinity_order(self._arrest._domain, [r.probe_name for r in results])
            ordered, weights = self._arrest._fusion.effective_weights(
                results, affinity_order=order, decorrelator=self._arrest._decorrelator
            )
            attributions = self._arrest._fusion.leave_one_out_attribution(ordered, weights)
        result: dict[str, Any] = {
            "anomaly_score": float(scores.max()) if scores.size else 0.0,
            "scores": scores,
            "is_anomaly": scores > self.threshold,
            "confidence": scores,
            "threshold": float(self.threshold),
            "probe_names": list(columns),
            "probe_scores": {k: np.asarray(v, dtype=np.float32) for k, v in columns.items()},
            # Per-sample leave-one-probe-out attribution of the fused score:
            # which mathematical instruments drove each decision, persisted
            # alongside the score itself.
            "probe_attributions": {
                k: np.asarray(v, dtype=np.float32) for k, v in attributions.items()
            },
            # 21 probes are not 21 dimensions: 11.64-16.00 measured on the
            # recorded NAB files, so the probe count is not the evidence count.
            "effective_probe_count": float(self._arrest._decorrelator.effective_probe_count),
            "redundant_probe_pairs": len(self._arrest._decorrelator.redundant_pairs),
        }
        result.update(batch_evidence(self._reference_scores, scores, self.CALIBRATION_ALPHA))
        if self._vector_calibrator is not None and self._vector_conformal is not None:
            # Opt-in supervised vector calibration: the calibrated probability
            # is the decision score (item: the isotonic model is *applied*, not
            # merely fitted), and the conformal prediction sets carry the
            # finite-sample per-class coverage guarantee. The raw fused scores
            # above are untouched, so the uncalibrated surface stays readable.
            matrix = self._matrix_from(results, fused, n_rows=len(series))
            calibrated = self._vector_calibrator.calibrate(matrix)
            width = len(calibrated)
            result["calibrated_scores"] = calibrated.astype(np.float32)
            result["calibrated_threshold"] = 0.5
            result["is_anomaly"] = np.asarray(calibrated > 0.5)
            if width < len(scores):
                # Shorter probe columns (a probe that failed on this batch):
                # fall back to the raw verdict for the uncovered tail.
                tail = scores[width:] > self.threshold
                result["is_anomaly"] = np.concatenate([calibrated > 0.5, tail])
            result["conformal_p_values"] = self._vector_conformal.p_values(matrix)
            result["conformal_sets"] = self._vector_conformal.predict_sets(matrix)
            result["conformal_alpha"] = float(self._vector_conformal.alpha)
        return result
