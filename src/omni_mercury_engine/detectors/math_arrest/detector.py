# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``BaseDetector`` seam for the 21-probe math-arrest instrument.

:class:`~omni_mercury_engine.detectors.math_arrest.arrest.AnomalyMathArrest`
is a 21-equation ensemble whose ``detect`` returns a bare score array, so it
could not satisfy the detector contract the engine, registry, served routes,
CLI and console catalog all consume — it was reachable only by importing it
directly (recorded in ``docs/DORMANCY_LEDGER.md`` §2b). This adapter gives it
the contract without changing the instrument: the fused score becomes
``scores``, the per-probe matrix becomes the fusion feature group, and the
alarm fires at a fit-time one-class split-conformal cut rather than a bare
0.5 on a batch-relative score.

Measured (``tests/detectors/test_recorded_math_arrest.py``,
``tests/detectors/test_recorded_nab_series.py``): mean AUROC **0.7498** over
the four recorded NAB files at the default ``uniform`` weighting, which places
the instrument above every series detector pinned on that corpus except
``lstm_autoencoder`` (0.829). It is a manifest entry so it can be *reached*;
it is **not** in the engine's default set, which this repository only grants
on a measured marginal gain over the live ensemble.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.core.base import (
    BaseDetector,
    _deserialized_state,
    _serializable_state,
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

    Wraps :class:`AnomalyMathArrest` (all 21 probes, ``uniform`` weighting by
    default) and serves its fused score through the standard detector result,
    with the 21 per-probe deviation scores available as an interpretable
    fusion feature group: every alarm traces to the probes that raised it.
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
            "AUROC 0.7498 over the four recorded NAB files and 0.6846 over "
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

        The shared packer refuses to serialize the :class:`AnomalyMathArrest`
        object itself (correctly: half-saving a learned object is worse than
        failing), so this names what is actually learned. Every probe's own
        attributes are ``bool``/``int``/``float``/``str``/``ndarray``, which
        the packer handles, as are the decorrelator's multipliers and the
        fit-quality map, so the whole instrument round-trips exactly.

        Returns:
            The fitted state of all probes, the decorrelator calibration, the
            fit-quality map and the conformal reference.
        """
        learned = {
            "probes": {type(probe).__name__: dict(vars(probe)) for probe in self._arrest._probes},
            "fit_qualities": dict(self._arrest._fit_qualities),
            "decorrelator": dict(vars(self._arrest._decorrelator)),
            "reference_scores": self._reference_scores,
            "arrest_threshold": float(self._arrest.threshold),
            "arrest_is_fitted": bool(self._arrest._is_fitted),
        }
        # The override's payload is stored verbatim, so it packs its own numpy
        # with the same helper the default path uses: the restricted
        # (``weights_only``) unpickler refuses numpy's reconstruction globals,
        # and a raw ndarray here would make the checkpoint unloadable.
        return dict(_serializable_state(learned, owner=type(self).__name__, attr="learned"))

    def _restore_learned_state(self, state: dict[str, Any]) -> None:
        """Write a checkpoint's state back onto the live probes.

        Args:
            state: Payload from :meth:`_learned_state`, with numpy already
                unpacked by the shared loader.
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
        fused = np.asarray(self._arrest.detect(series), dtype=np.float64).reshape(-1)
        scores = finite_scores(fused, detector=self.name).astype(np.float32)
        columns = self._arrest.per_probe_scores(series) if self._is_fitted else {}
        result: dict[str, Any] = {
            "anomaly_score": float(scores.max()) if scores.size else 0.0,
            "scores": scores,
            "is_anomaly": scores > self.threshold,
            "confidence": scores,
            "threshold": float(self.threshold),
            "probe_names": list(columns),
            "probe_scores": {k: np.asarray(v, dtype=np.float32) for k, v in columns.items()},
            # The honest width of the ensemble. A 21-probe count is not 21
            # independent dimensions: on the four recorded NAB files the
            # decorrelator measures 11.64-16.00 effective dimensions and
            # down-weights the duplicates. Surfacing it keeps the probe count
            # from reading as the evidence count.
            "effective_probe_count": float(self._arrest._decorrelator.effective_probe_count),
            "redundant_probe_pairs": len(self._arrest._decorrelator.redundant_pairs),
        }
        result.update(batch_evidence(self._reference_scores, scores, self.CALIBRATION_ALPHA))
        return result
