# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
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
* ``skipped_detectors`` -- the evidence sources that did not report this run
  (timed out, refused, errored), each flagged with whether the fusion network
  was trained on it.  A trained source that is missing means the probability
  was computed from partial evidence.  The full per-entry provenance is kept
  as ``skipped_detail``.
* ``detector_uncertainties`` / ``out_of_support_rows`` / ``model_provenance``
  -- per-detector decision uncertainty, the extrapolation-floored rows and
  the calibrated model's identity (checkpoint, temperature, conformal
  coverage), carried verbatim for record provenance.
* ``detector_importance`` / ``explanation`` / ``gdpr_report`` -- reduced to a
  bounded ``contributions`` summary (top-16 weights, 6 dp); the raw full
  explanation never rides the evidence.

Extraction is defensive: every field is optional, missing keys collapse to
``None`` (a transparent "signal absent this run"), and nothing here raises on a
sparse or partial result.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

#: Hard cap on each ``contributions`` map: enough to name every detector and
#: the strongest features, never the raw full explanation.
_MAX_CONTRIBUTION_ENTRIES = 16


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


def _bounded_contributions(raw: Any) -> dict[str, float] | None:
    """Bound a ``name -> weight`` mapping to a compact, stable summary.

    Keeps at most :data:`_MAX_CONTRIBUTION_ENTRIES` entries -- the strongest by
    absolute weight, ties broken by name for determinism -- with each weight
    rounded to 6 decimal places.  Non-finite or non-numeric weights are
    dropped; an empty or malformed mapping reads as absent (``None``), never
    as a fabricated zero-attribution.
    """
    if not isinstance(raw, Mapping):
        return None
    entries: list[tuple[str, float]] = []
    for name, value in raw.items():
        as_float = _as_float(value)
        if as_float is not None:
            entries.append((str(name), as_float))
    if not entries:
        return None
    entries.sort(key=lambda item: (-abs(item[1]), item[0]))
    return {name: round(value, 6) for name, value in entries[:_MAX_CONTRIBUTION_ENTRIES]}


def _ig_summary(explanation: Any) -> dict[str, float] | None:
    """Bounded attribution summary of an IG ``explanation`` payload.

    Reads the ``feature_importances`` list the explainer's ``to_dict`` emits
    (``{"feature", "importance", ...}`` items) and reduces it to a top-k
    ``feature -> importance`` map via :func:`_bounded_contributions`.  The raw
    explanation (narratives, counterfactuals, faithfulness scores) never rides
    the record -- it stays at the detection result's top level only.
    """
    if not isinstance(explanation, Mapping):
        return None
    importances = explanation.get("feature_importances")
    if not isinstance(importances, (list, tuple)):
        return None
    collected: dict[str, float] = {}
    for item in importances:
        if not isinstance(item, Mapping):
            continue
        name = item.get("feature")
        value = _as_float(item.get("importance"))
        if name is not None and value is not None:
            collected[str(name)] = value
    return _bounded_contributions(collected)


def _cognitive_summary(analysis: Any) -> dict[str, Any] | None:
    """Bounded self-critique summary of a ``cognitive_analysis`` payload.

    Carries the orchestrator's critique of the detection — reliability,
    confidence, the uncertainties together with the flags that say whether
    each was measured or is a placeholder — its discovery signals (curiosity
    novelty when the component scored this observation; causal factors, at
    most five, truncated), and its historical analogs (case ids and
    recommended actions, at most five each, truncated), never the full
    analysis: the reasoning chain, knowledge updates and raw feedback
    signals stay at the detection result's top level.
    """
    if not isinstance(analysis, Mapping):
        return None

    def strings(value: Any, limit: int) -> list[str]:
        if not isinstance(value, (list, tuple)):
            return []
        return [str(item)[:160] for item in value if str(item).strip()][:limit]

    def flag(key: str) -> bool | None:
        value = analysis.get(key)
        return bool(value) if isinstance(value, bool) else None

    return {
        "is_reliable": (bool(analysis["is_reliable"]) if "is_reliable" in analysis else None),
        "confidence": _as_float(analysis.get("confidence")),
        "confidence_calibrated": flag("confidence_calibrated"),
        "epistemic_uncertainty": _as_float(analysis.get("epistemic_uncertainty")),
        "epistemic_measured": flag("epistemic_measured"),
        "aleatoric_uncertainty": _as_float(analysis.get("aleatoric_uncertainty")),
        "aleatoric_measured": flag("aleatoric_measured"),
        "ethical_permissible": (
            bool(analysis["ethical_permissible"]) if "ethical_permissible" in analysis else None
        ),
        # Discovery signals: present only when the component produced them
        # (the orchestrator's to_dict omits the novelty keys until the
        # curiosity engine scored this observation), so a record without the
        # component is unchanged.
        "novelty_score": _as_float(analysis.get("novelty_score")),
        "is_novel": (bool(analysis["is_novel"]) if "is_novel" in analysis else None),
        "novelty_challenge": _novelty_challenge(analysis),
        "causal_factors": strings(analysis.get("causal_factors"), 5),
        "hypothesis": _chain_hypothesis(analysis.get("reasoning_chain")),
        "similar_cases": strings(analysis.get("similar_cases"), 5),
        "recommendations": strings(analysis.get("recommendations"), 5),
    }


def _chain_hypothesis(chain: Any) -> dict[str, Any] | None:
    """The multi-hop reasoner's derived conclusion, bounded, or ``None``.

    The reasoner's chain ends in the proposition it derived (abductive
    best-explaining-hypothesis rules included); the record carries that final
    conclusion with its confidence and the chain depth, so the proposal is
    reconstructable and challengeable from the record without the full chain
    (which stays at the detection result's top level).

    The ``falsifier`` names the chain's weakest step — the lowest-confidence
    link every later conclusion depends on.  A reviewer challenges the
    hypothesis there: re-test that step, and if it fails, the conclusion
    falls with it.  That makes the hypothesis a *proposal with a stated
    attack surface*, not a take-it-or-leave-it string.
    """
    if not isinstance(chain, (list, tuple)) or not chain:
        return None
    steps = [step for step in chain if isinstance(step, Mapping)]
    last = chain[-1]
    if not isinstance(last, Mapping):
        return None
    conclusion = last.get("conclusion")
    if not isinstance(conclusion, str) or not conclusion.strip():
        return None
    weakest: Mapping[str, Any] | None = None
    weakest_conf: float | None = None
    for step in steps:
        conf = _as_float(step.get("confidence"))
        if conf is None or not isinstance(step.get("conclusion"), str):
            continue
        if weakest_conf is None or conf < weakest_conf:
            weakest, weakest_conf = step, conf
    falsifier: dict[str, Any] | None = None
    if weakest is not None:
        falsifier = {
            "weakest_step": str(weakest.get("conclusion", ""))[:240],
            "rule": str(weakest.get("rule", ""))[:80],
            "confidence": weakest_conf,
            "falsifiable_by": (
                "re-test the weakest step: every later conclusion in the "
                "chain depends on it, so the hypothesis falls if it fails"
            ),
        }
    return {
        "conclusion": conclusion[:240],
        "confidence": _as_float(last.get("confidence")),
        "rule": str(last.get("rule", ""))[:80],
        "chain_depth": len(chain),
        "falsifier": falsifier,
    }


def _novelty_challenge(analysis: Mapping[str, Any]) -> dict[str, Any] | None:
    """The challenge grounds of a novelty verdict, or ``None`` without one.

    Carries the measured standardized distance behind the score and the
    threshold the ``is_novel`` verdict was cut at, plus the recipe that
    falsifies it, so the verdict can be re-measured rather than taken.
    """
    if _as_float(analysis.get("novelty_score")) is None:
        return None
    distance = _as_float(analysis.get("novelty_distance"))
    threshold = _as_float(analysis.get("novelty_threshold"))
    if distance is None and threshold is None:
        return None
    return {
        "standardized_distance": distance,
        "novelty_threshold": threshold,
        "falsifiable_by": (
            "re-score read-only (CuriosityEngine.novelty()) against the "
            "fitted distribution as it grows; is_novel falls when the score "
            "drops below the threshold"
        ),
    }


def _shapley_summary(report: Any) -> dict[str, float] | None:
    """Bounded Shapley summary of a GDPR ``gdpr_report`` payload.

    Reads ``explanation.feature_contributions`` from the report's ``to_dict``
    form and reduces it to the same top-k map shape as the IG summary.  The
    full report (rights narrative, counterfactual actions) stays at the
    detection result's top level only.
    """
    if not isinstance(report, Mapping):
        return None
    explanation = report.get("explanation")
    if not isinstance(explanation, Mapping):
        return None
    return _bounded_contributions(explanation.get("feature_contributions"))


@dataclass(frozen=True)
class Evidence:
    """A typed, normalised snapshot of one detection result.

    Attributes:
        anomaly_prob: Calibrated ``P(anomaly)`` in ``[0, 1]``, or ``None`` when
            the result carried no finite probability (missing, ``None``, NaN,
            infinite or unparseable).  ``None`` is "signal absent this run",
            which the decider answers with an abstention -- it is never
            collapsed to ``0.0``, because a fabricated zero would ground a
            confident CLEAR from no evidence (fail-open).  A genuine ``0.0``
            stays a real probability.
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
        skipped_detectors: Names of the evidence sources the fusion network
            was trained on that did not report this run (a detector that
            timed out, was refused by a saturated executor, declined the
            input or failed).  Non-empty means the probability rests on
            partial evidence, and the decider will not ground a verdict on
            it.  Empty when every trained source reported, or the result did
            not carry the ``skipped_detectors`` section.
        skipped_detail: Full provenance of every skip entry the result
            carried (trained or not): ``name -> {reason, detail,
            trained_group}`` as ``detect_with_fusion`` reported it.  Empty
            when nothing was skipped.  Additive beside
            :attr:`skipped_detectors`, which keeps its released
            names-of-trained-sources meaning unchanged.
        detector_uncertainties: Per-detector decision uncertainty
            (``name -> value``) from the result's ``detector_uncertainties``
            key; empty when absent.  Non-finite values are dropped.
        out_of_support_rows: Rows outside the fusion training envelope whose
            probabilities were floored at the detector consensus (the
            result's ``out_of_support_rows``); empty when absent or clean.
        model_provenance: Identity of the calibrated model that produced the
            probability (checkpoint name/sha256, temperature, conformal
            coverage, trained groups) -- the result's ``model_provenance``
            mapping, or ``None`` when the result carried none.
        cognitive: Bounded self-critique summary of the orchestrator's
            ``cognitive_analysis`` (reliability, calibrated confidence, the
            measured uncertainties, historical analogs and recommended
            actions, truncated), or ``None`` when the orchestrator did not
            run.
        contributions: Bounded causal-contribution summary under a stable
            key: ``{"detector_importance", "integrated_gradients",
            "shapley"}``, each a top-k (at most 16 entries) ``name -> weight``
            map with weights rounded to 6 dp, or ``None`` when that source
            was absent this run.  Never the raw full explanation.  ``None``
            on evidence built without :meth:`from_detection`.
        domain: Optional domain hint.
    """

    anomaly_prob: float | None
    is_anomaly: bool
    threshold: float
    severity: float
    conformal_set_size: int | None = None
    conformal_labels: tuple[int, ...] | None = None
    coverage: float | None = None
    conformal_regime: str | None = None
    conformal_rows: int | None = None
    ethical_gate_passed: bool | None = None
    ethical_score: float | None = None
    ethical_threshold: float | None = None
    ethics_report: Mapping[str, Any] | None = None
    symbolic_satisfaction: float | None = None
    drift_detected: bool = False
    drift_severity: str | None = None
    gosnn_anomaly_prob: float | None = None
    gosnn_demote_act_below: float | None = None
    gosnn_demote_clear_above: float | None = None
    domain: str | None = None
    skipped_detectors: tuple[str, ...] = ()
    skipped_detail: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    detector_uncertainties: Mapping[str, float] = field(default_factory=dict)
    out_of_support_rows: tuple[int, ...] = ()
    model_provenance: Mapping[str, Any] | None = None
    contributions: Mapping[str, Any] | None = None
    cognitive: Mapping[str, Any] | None = None

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
            result: The detection result mapping.  Every field is optional; a
                missing or non-finite ``anomaly_prob`` is preserved as ``None``
                and the decider abstains on it rather than guessing.
            domain: Optional domain hint (falls back to ``result['domain']``).

        Returns:
            A normalised :class:`Evidence`.
        """
        # The probability is a gate signal: absent/NaN/inf stays ``None`` (an
        # abstention driver downstream), never the falsy-``or`` 0.0 that used
        # to turn "no evidence" into the strongest possible CLEAR evidence.
        # A genuine 0.0 (also falsy) is preserved exactly by the ``is None``
        # distinction.
        anomaly_prob = _as_float(result.get("anomaly_prob"))
        threshold = _as_float(result.get("threshold_used"))
        if threshold is None:
            threshold = _as_float(result.get("threshold"))
        if threshold is None:
            threshold = 0.5
        severity = _as_float(result.get("severity")) or 0.0
        default_flag = anomaly_prob is not None and anomaly_prob > threshold
        is_anomaly = bool(result.get("is_anomaly", default_flag))

        conformal_set_size: int | None = None
        conformal_labels: tuple[int, ...] | None = None
        coverage: float | None = None
        conformal_regime: str | None = None
        conformal_rows: int | None = None
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
                regime = conformal.get("regime")
                if isinstance(regime, str) and regime:
                    conformal_regime = regime
                rows = conformal.get("n_rows")
                if isinstance(rows, int) and not isinstance(rows, bool) and rows > 0:
                    conformal_rows = rows

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

        # Evidence sources that did not report.  A source counts against the
        # verdict when the fusion network was trained on it; an entry that
        # does not say (malformed, or a bare list of names) is read as
        # trained -- the fail-closed reading of an unclear record.  The full
        # per-entry provenance ({reason, detail, trained_group}) is carried
        # beside the names so a reloaded record can say *why* evidence was
        # missing, not only that it was.
        skipped_detectors: list[str] = []
        skipped_detail: dict[str, dict[str, Any]] = {}
        skipped = result.get("skipped_detectors")
        if isinstance(skipped, Mapping):
            for name, info in skipped.items():
                skipped_detail[str(name)] = (
                    {str(key): value for key, value in info.items()}
                    if isinstance(info, Mapping)
                    else {}
                )
                trained = info.get("trained_group", True) if isinstance(info, Mapping) else True
                if trained is not False:
                    skipped_detectors.append(str(name))
        elif isinstance(skipped, (list, tuple, set, frozenset)):
            skipped_detectors.extend(str(name) for name in skipped)
            skipped_detail = {str(name): {} for name in skipped}

        detector_uncertainties: dict[str, float] = {}
        raw_uncertainties = result.get("detector_uncertainties")
        if isinstance(raw_uncertainties, Mapping):
            for name, value in raw_uncertainties.items():
                uncertainty = _as_float(value)
                if uncertainty is not None:
                    detector_uncertainties[str(name)] = uncertainty

        raw_provenance = result.get("model_provenance")
        model_provenance = (
            {str(key): value for key, value in raw_provenance.items()}
            if isinstance(raw_provenance, Mapping)
            else None
        )

        # Bounded causal-contribution summary (always three keys; a source
        # absent this run reads None).  ``detector_importance`` rides every
        # fusion result; the IG and Shapley summaries only exist when the
        # caller opted into ``explain`` / ``gdpr_report``.
        contributions: dict[str, Any] = {
            "detector_importance": _bounded_contributions(result.get("detector_importance")),
            "integrated_gradients": _ig_summary(result.get("explanation")),
            "shapley": _shapley_summary(result.get("gdpr_report")),
        }

        return cls(
            anomaly_prob=anomaly_prob,
            is_anomaly=is_anomaly,
            threshold=threshold,
            severity=severity,
            conformal_set_size=conformal_set_size,
            conformal_labels=conformal_labels,
            coverage=coverage,
            conformal_regime=conformal_regime,
            conformal_rows=conformal_rows,
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
            skipped_detectors=tuple(sorted(skipped_detectors)),
            skipped_detail=skipped_detail,
            detector_uncertainties=detector_uncertainties,
            out_of_support_rows=_as_int_tuple(result.get("out_of_support_rows")) or (),
            model_provenance=model_provenance,
            contributions=contributions,
            cognitive=_cognitive_summary(result.get("cognitive_analysis")),
        )

    @classmethod
    def from_ethical_block(
        cls,
        ethics: Mapping[str, Any],
        *,
        domain: str | None = None,
    ) -> Evidence:
        """Build the evidence of a decision the ethical gate refused.

        A blocked boundary produces no probability, no certificate and no
        detector output — the gate raised before any of them existed.  What it
        does produce is the structured refusal report (check, score,
        threshold, ruleset version, details), and that report is the whole
        evidence: the decider's fail-closed Stage 1 turns it into an
        UNDECIDABLE/HOLD record whose ``signals.ethics_report`` carries the
        full report, so a refusal is reconstructable from its record exactly
        like a served verdict.

        Args:
            ethics: The bounded, JSON-safe refusal report (the ``ethics``
                block of the structured 403 payload).
            domain: Optional domain hint.

        Returns:
            Evidence with ``ethical_gate_passed=False`` and the report
            attached.
        """
        return cls(
            anomaly_prob=None,
            is_anomaly=False,
            threshold=0.5,
            severity=0.0,
            ethical_gate_passed=False,
            ethical_score=_as_float(ethics.get("score")),
            ethical_threshold=_as_float(ethics.get("threshold")),
            ethics_report=dict(ethics),
            domain=domain,
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
            "conformal_regime": self.conformal_regime,
            "conformal_rows": self.conformal_rows,
            "ethical_gate_passed": self.ethical_gate_passed,
            "ethical_score": self.ethical_score,
            "ethical_threshold": self.ethical_threshold,
            "ethics_report": (dict(self.ethics_report) if self.ethics_report is not None else None),
            "symbolic_satisfaction": self.symbolic_satisfaction,
            "drift_detected": self.drift_detected,
            "drift_severity": self.drift_severity,
            "gosnn_anomaly_prob": self.gosnn_anomaly_prob,
            "gosnn_demote_act_below": self.gosnn_demote_act_below,
            "gosnn_demote_clear_above": self.gosnn_demote_clear_above,
            "skipped_detectors": list(self.skipped_detectors),
            "skipped_detail": {name: dict(info) for name, info in self.skipped_detail.items()},
            "detector_uncertainties": dict(self.detector_uncertainties),
            "out_of_support_rows": list(self.out_of_support_rows),
            "model_provenance": (
                dict(self.model_provenance) if self.model_provenance is not None else None
            ),
            "contributions": dict(self.contributions) if self.contributions is not None else None,
            "cognitive": dict(self.cognitive) if self.cognitive is not None else None,
            "domain": self.domain,
        }


__all__ = ["Evidence"]
