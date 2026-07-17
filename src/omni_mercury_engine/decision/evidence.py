# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Normalised evidence extracted from a detection result.

:class:`Evidence` is the single, typed view the decision layer reasons over.
It is built by :meth:`Evidence.from_detection` from the dict that
``OmniMercuryEngine.detect_with_fusion`` returns, reading the *real* keys that
pipeline already threads through:

* ``anomaly_prob`` / ``is_anomaly`` / ``threshold_used`` / ``severity`` -- the
  calibrated fusion verdict.
* ``conformal`` -- the distribution-free certificate
  (``set_size`` in ``{0, 1, 2}``, ``prediction_set``, ``coverage``).  Its
  presence is what makes a decision *calibrated* (coverage-guaranteed).
* ``gosnn_metadata`` -- the hard ethical gate verdict
  (``ethical_gate_passed``, ``sigma_immutable_score`` / ``..._threshold``),
  plus -- only when a merit-gated detection head shipped with the fusion
  checkpoint -- the ``detection`` block (the GOSNN fused-state
  ``anomaly_prob`` and its validation-selected demotion thresholds).
* ``symbolic_consistency`` -- the neuro-symbolic agreement (the LTN constraint
  ``satisfaction`` in ``[0, 1]``).
* ``drift_detection`` -- distribution-shift status (``is_drift`` / ``severity``).

Extraction is defensive: every field is optional, missing keys collapse to
``None`` (a transparent "signal absent this run"), and nothing here raises on a
sparse or partial result.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _as_float(value: Any) -> float | None:
    """Best-effort scalar coercion; ``None`` when not a finite scalar."""
    if value is None or isinstance(value, bool):
        return None
    if hasattr(value, "item"):  # numpy scalar / 0-d array / torch scalar
        try:
            value = value.item()
        except (ValueError, TypeError):
            return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    # Reject NaN / inf -- an unusable signal is an absent signal, not a number.
    if not math.isfinite(out):
        return None
    return out


def _as_int(value: Any) -> int | None:
    """Best-effort integer coercion; ``None`` when not an integer-valued scalar."""
    out = _as_float(value)
    if out is None:
        return None
    as_int = int(out)
    # Reject non-integral values (e.g. 1.5) -- a fractional set size or label is
    # malformed, not a number we should silently truncate.
    return as_int if as_int == out else None


def _as_int_tuple(values: Any) -> tuple[int, ...] | None:
    """Coerce a sequence to a tuple of ints; ``None`` if it is not a clean one.

    A single non-integer element invalidates the whole set, so a malformed label
    set is reported as absent rather than partially coerced.
    """
    if not isinstance(values, (list, tuple)):
        return None
    coerced: list[int] = []
    for item in values:
        as_int = _as_int(item)
        if as_int is None:
            return None
        coerced.append(as_int)
    return tuple(coerced)


@dataclass(frozen=True)
class Evidence:
    """A typed, normalised snapshot of one detection result.

    Attributes:
        anomaly_prob: Calibrated ``P(anomaly)`` in ``[0, 1]``.
        is_anomaly: The detector's raw threshold verdict (carried for context).
        threshold: The decision threshold the verdict used.
        severity: Anomaly severity in ``[0, 1]`` (0.0 when absent).
        conformal_set_size: Conformal label-set size in ``{0, 1, 2}``, or
            ``None`` when no conformal certificate was attached this run.
        conformal_labels: The conformal label set (e.g. ``(1,)`` / ``(0, 1)``),
            or ``None``.
        coverage: The certificate's distribution-free coverage level, or
            ``None``.
        ethical_gate_passed: Hard ethical-gate verdict -- ``True`` / ``False``,
            or ``None`` when the gate did not run (e.g. a testing bypass).
        ethical_score: σ_Immutable score, or ``None``.
        ethical_threshold: σ_Immutable threshold, or ``None``.
        symbolic_satisfaction: Neuro-symbolic constraint satisfaction in
            ``[0, 1]`` (higher = neural and symbolic paths agree), or ``None``.
        drift_detected: Whether distribution drift was flagged this run.
        drift_severity: Drift severity name (e.g. ``"HIGH"``), or ``None``.
        gosnn_anomaly_prob: The GOSNN fused-state detection head's
            ``P(anomaly)`` in ``[0, 1]`` (``gosnn_metadata["detection"]``), or
            ``None`` when no merit-gated head shipped this build.  Feeds the
            decider's disagreement demotion overlay -- abstention-only, never
            a grounding signal.
        gosnn_demote_act_below: Validation-selected threshold shipped with the
            head: a grounded positive whose ``gosnn_anomaly_prob`` is at or
            below it is demoted to a deferral.  ``None`` disables that side.
        gosnn_demote_clear_above: Same for grounded negatives at or above it.
        domain: Optional domain hint.
    """

    anomaly_prob: float
    is_anomaly: bool
    threshold: float
    severity: float
    conformal_set_size: int | None = None
    conformal_labels: tuple[int, ...] | None = None
    coverage: float | None = None
    ethical_gate_passed: bool | None = None
    ethical_score: float | None = None
    ethical_threshold: float | None = None
    symbolic_satisfaction: float | None = None
    drift_detected: bool = False
    drift_severity: str | None = None
    gosnn_anomaly_prob: float | None = None
    gosnn_demote_act_below: float | None = None
    gosnn_demote_clear_above: float | None = None
    domain: str | None = None

    @property
    def calibrated(self) -> bool:
        """Whether a distribution-free coverage certificate backs this result."""
        return self.conformal_set_size is not None

    @classmethod
    def from_detection(
        cls,
        result: Mapping[str, Any],
        *,
        domain: str | None = None,
    ) -> Evidence:
        """Build an :class:`Evidence` from a ``detect_with_fusion`` result dict.

        Args:
            result: The detection result mapping.  Only ``anomaly_prob`` is
                effectively required; everything else degrades gracefully.
            domain: Optional domain hint (falls back to ``result['domain']``).

        Returns:
            A normalised :class:`Evidence`.
        """
        anomaly_prob = _as_float(result.get("anomaly_prob")) or 0.0
        threshold = _as_float(result.get("threshold_used"))
        if threshold is None:
            threshold = _as_float(result.get("threshold"))
        if threshold is None:
            threshold = 0.5
        severity = _as_float(result.get("severity")) or 0.0
        is_anomaly = bool(result.get("is_anomaly", anomaly_prob > threshold))

        conformal_set_size: int | None = None
        conformal_labels: tuple[int, ...] | None = None
        coverage: float | None = None
        conformal = result.get("conformal")
        if isinstance(conformal, Mapping):
            # The label set is the source of truth: a certificate counts as
            # *present* only when it carries a clean prediction set (the empty
            # set is valid -- it is the atypical-point case).  ``set_size`` is
            # derived from it rather than trusted independently.  A bare or
            # malformed ``set_size`` is therefore treated as "no certificate",
            # so a partial result can never masquerade as a calibrated singleton
            # and push the gate into guessing a label the certificate never made.
            labels = _as_int_tuple(conformal.get("prediction_set"))
            if labels is not None:
                conformal_labels = labels
                conformal_set_size = len(labels)
                # Coverage is provenance *of the certificate*; only record it when
                # the certificate is actually present, so a malformed section can
                # never yield the contradictory `calibrated=False, coverage=0.9`.
                coverage = _as_float(conformal.get("coverage"))

        ethical_passed: bool | None = None
        ethical_score: float | None = None
        ethical_threshold: float | None = None
        gosnn_anomaly_prob: float | None = None
        gosnn_demote_act_below: float | None = None
        gosnn_demote_clear_above: float | None = None
        gosnn = result.get("gosnn_metadata")
        if isinstance(gosnn, Mapping):
            # A hard safety gate: accept only a genuine boolean.  Anything else
            # -- a truthy string like "False", a number, a malformed value -- is
            # treated as *absent* (the gate "did not run for this decision")
            # rather than coerced via ``bool()``, which would fail *open* by
            # turning "False" into True on a refused boundary.
            raw_passed = gosnn.get("ethical_gate_passed")
            ethical_passed = raw_passed if isinstance(raw_passed, bool) else None
            ethical_score = _as_float(gosnn.get("sigma_immutable_score"))
            ethical_threshold = _as_float(gosnn.get("sigma_immutable_threshold"))

            # The merit-gated detection head's corroboration signal.  The
            # probability is the anchor: a malformed or out-of-range value
            # drops the whole block (signal absent), so the demotion overlay
            # can never fire on garbage.  Thresholds are only meaningful
            # alongside a valid probability; each side of the overlay is
            # independently disabled by an absent/malformed/out-of-range
            # threshold (defence in depth behind the load-time validation --
            # a threshold outside [0, 1] would make its side either inert or
            # unconditional, and neither is a decision this layer may take
            # on malformed provenance).
            detection = gosnn.get("detection")
            if isinstance(detection, Mapping):
                prob = _as_float(detection.get("anomaly_prob"))
                if prob is not None and 0.0 <= prob <= 1.0:
                    gosnn_anomaly_prob = prob
                    act_below = _as_float(detection.get("demote_act_below"))
                    clear_above = _as_float(detection.get("demote_clear_above"))
                    if act_below is not None and 0.0 <= act_below <= 1.0:
                        gosnn_demote_act_below = act_below
                    if clear_above is not None and 0.0 <= clear_above <= 1.0:
                        gosnn_demote_clear_above = clear_above

        symbolic_satisfaction: float | None = None
        symbolic = result.get("symbolic_consistency")
        if isinstance(symbolic, Mapping):
            symbolic_satisfaction = _as_float(symbolic.get("satisfaction"))

        drift_detected = False
        drift_severity: str | None = None
        drift = result.get("drift_detection")
        if isinstance(drift, Mapping):
            drift_detected = bool(drift.get("is_drift", False))
            raw_sev = drift.get("severity")
            drift_severity = None if raw_sev is None else str(raw_sev)

        return cls(
            anomaly_prob=anomaly_prob,
            is_anomaly=is_anomaly,
            threshold=threshold,
            severity=severity,
            conformal_set_size=conformal_set_size,
            conformal_labels=conformal_labels,
            coverage=coverage,
            ethical_gate_passed=ethical_passed,
            ethical_score=ethical_score,
            ethical_threshold=ethical_threshold,
            symbolic_satisfaction=symbolic_satisfaction,
            drift_detected=drift_detected,
            drift_severity=drift_severity,
            gosnn_anomaly_prob=gosnn_anomaly_prob,
            gosnn_demote_act_below=gosnn_demote_act_below,
            gosnn_demote_clear_above=gosnn_demote_clear_above,
            domain=domain if domain is not None else result.get("domain"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe provenance view of the evidence used."""
        return {
            "anomaly_prob": self.anomaly_prob,
            "is_anomaly": self.is_anomaly,
            "threshold": self.threshold,
            "severity": self.severity,
            "calibrated": self.calibrated,
            "conformal_set_size": self.conformal_set_size,
            "conformal_labels": (
                list(self.conformal_labels) if self.conformal_labels is not None else None
            ),
            "coverage": self.coverage,
            "ethical_gate_passed": self.ethical_gate_passed,
            "ethical_score": self.ethical_score,
            "ethical_threshold": self.ethical_threshold,
            "symbolic_satisfaction": self.symbolic_satisfaction,
            "drift_detected": self.drift_detected,
            "drift_severity": self.drift_severity,
            "gosnn_anomaly_prob": self.gosnn_anomaly_prob,
            "gosnn_demote_act_below": self.gosnn_demote_act_below,
            "gosnn_demote_clear_above": self.gosnn_demote_clear_above,
            "domain": self.domain,
        }


__all__ = ["Evidence"]
