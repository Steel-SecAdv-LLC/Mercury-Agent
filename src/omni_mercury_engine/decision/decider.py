# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The calibration-grounded abstention gate -- the closed loop's brain.

:class:`DecisionAbstentionResponder` turns a detection certificate into a
:class:`~omni_mercury_engine.decision.record.DecisionRecord`: it classifies the
event into the three-state transparency contract, projects that onto an operational
disposition, and attaches a bounded, non-destructive response.  It is a pure,
deterministic function of the evidence -- same certificate in, same record out
-- which is what makes the whole loop verifiable.

The classifier is **abstention-first and fail-closed**, applied in priority
order:

1. **Ethical fail-closed.** An explicit ethical-gate failure forces
   ``UNDECIDABLE`` / ``HOLD`` -- no score can override a refused boundary.
2. **The calibrated certificate decides.** When a conformal label set is
   present it is authoritative:

   * singleton ``{1}`` / ``{0}`` -> ``GROUNDED`` (the coverage level is the
     transparent confidence);
   * ``{0, 1}`` -> ``UNAVAILABLE`` (a *calibrated* don't-know -- both labels
     are admissible at the target coverage);
   * ``{}`` -> ``UNDECIDABLE`` (an atypical point no class explains; fail-closed
     by default).
3. **Uncalibrated fallback.** With no certificate, a calibrated probability
   inside the threshold's indecision band is ``UNAVAILABLE``; outside it the
   threshold side grounds the label, but the record is flagged
   ``calibrated=False`` (a decision without a coverage guarantee).
4. **Demotion overlays.** A grounded verdict is demoted to ``UNAVAILABLE`` when
   the neuro-symbolic paths disagree or sufficiently severe drift means the
   calibration may no longer hold.  Overlays only ever weaken a verdict toward
   abstention, never the reverse.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.decision.evidence import Evidence
from omni_mercury_engine.decision.policy import DecisionPolicy
from omni_mercury_engine.decision.record import DecisionRecord
from omni_mercury_engine.decision.response import ResponsePlan, ResponsePolicy
from omni_mercury_engine.decision.states import Disposition
from omni_mercury_engine.verifiers.three_state import ThreeState

if TYPE_CHECKING:
    from collections.abc import Mapping


class _Verdict:
    """Mutable scratch verdict the rule stages refine before it is frozen."""

    __slots__ = ("confidence", "disposition", "label", "reasons", "resolvable_by_input", "state")

    def __init__(self) -> None:
        """Start neutral: abstain (UNAVAILABLE / DEFER) until a stage decides."""
        self.state: ThreeState = ThreeState.UNAVAILABLE
        self.disposition: Disposition = Disposition.DEFER
        self.label: int | None = None
        self.confidence: float | None = None
        self.reasons: list[str] = []
        self.resolvable_by_input: bool = False


class DecisionAbstentionResponder:
    """Decide-or-abstain over a detection certificate, then recommend a response.

    Args:
        policy: The abstention thresholds.  Defaults to the conservative,
            fail-closed :class:`DecisionPolicy`.
        response_policy: The disposition -> response mapping.  Defaults to the
            generic non-destructive :class:`ResponsePolicy`.
    """

    def __init__(
        self,
        policy: DecisionPolicy | None = None,
        response_policy: ResponsePolicy | None = None,
        confidence_calibrator: Any | None = None,
    ) -> None:
        """Initialize the responder.

        Args:
            policy: The abstention thresholds.
            response_policy: The disposition -> response mapping.
            confidence_calibrator: Optional fitted
                :class:`~omni_mercury_engine.core.confidence.CalibratedConfidence`
                (or any object with ``is_calibrated`` and ``transform_one``).
                When attached *and* calibrated, the uncalibrated threshold-band
                fallback reports a real calibrated probability instead of the
                ``0.5 + |margin|`` heuristic. The responder stays a pure function
                of (evidence, policy, calibrator): same inputs -> same record.
        """
        self.policy = policy or DecisionPolicy()
        self.response_policy = response_policy or ResponsePolicy()
        self.confidence_calibrator = confidence_calibrator

    # -- public API ---------------------------------------------------------

    def decide(
        self,
        detection_result: Mapping[str, Any],
        *,
        domain: str | None = None,
    ) -> DecisionRecord:
        """Classify a detection result and attach a bounded response.

        Args:
            detection_result: A ``detect_with_fusion``-style result mapping.
            domain: Optional domain hint (falls back to ``result['domain']``).

        Returns:
            A :class:`DecisionRecord` carrying the verdict, confidence, response
            and reasoning trail.
        """
        evidence = Evidence.from_detection(detection_result, domain=domain)
        return self.decide_from_evidence(evidence)

    def decide_from_evidence(self, evidence: Evidence) -> DecisionRecord:
        """Classify pre-normalised :class:`Evidence` (the pure core)."""
        verdict = _Verdict()
        self._classify(evidence, verdict)
        caveats = self._caveats(evidence)

        abstained = verdict.state is not ThreeState.GROUNDED
        response: ResponsePlan = self.response_policy.plan(
            verdict.disposition,
            severity=evidence.severity,
            domain=evidence.domain,
            resolvable_by_input=verdict.resolvable_by_input,
        )

        signals = evidence.to_dict()
        signals["policy"] = self.policy.to_dict()

        return DecisionRecord(
            state=verdict.state,
            disposition=verdict.disposition,
            decision_label=verdict.label,
            abstained=abstained,
            anomaly_prob=evidence.anomaly_prob,
            threshold=evidence.threshold,
            decision_confidence=verdict.confidence,
            coverage=evidence.coverage,
            calibrated=evidence.calibrated,
            severity=evidence.severity,
            response=response,
            reasons=tuple(verdict.reasons),
            caveats=caveats,
            signals=signals,
            domain=evidence.domain,
        )

    # -- classification stages ---------------------------------------------

    def _classify(self, ev: Evidence, v: _Verdict) -> None:
        """Run the priority-ordered rule stages, mutating ``v`` in place."""
        # Stage 1 -- ethical fail-closed (highest priority).
        if self.policy.fail_closed_on_ethical_block and ev.ethical_gate_passed is False:
            v.state = ThreeState.UNDECIDABLE
            v.disposition = Disposition.HOLD
            v.label = None
            v.confidence = None
            v.reasons.append(
                "ethical gate refused the decision boundary (fail-closed): no "
                "score can override a blocked boundary"
            )
            return

        # Stage 2/3 -- ground the base verdict from the certificate or the band.
        if ev.conformal_set_size is not None:
            self._from_conformal(ev, v)
        else:
            self._from_threshold_band(ev, v)

        # Stage 4 -- demotion overlays (only weaken a grounded verdict).
        if v.state is ThreeState.GROUNDED:
            self._apply_demotions(ev, v)

    def _from_conformal(self, ev: Evidence, v: _Verdict) -> None:
        """Ground (or abstain) from the conformal label set -- authoritative."""
        size = ev.conformal_set_size
        cov = ev.coverage
        cov_str = f"{cov:.0%}" if cov is not None else "target"

        if size == 1:
            label = ev.conformal_labels[0] if ev.conformal_labels else int(ev.is_anomaly)
            v.state = ThreeState.GROUNDED
            v.label = int(label)
            v.disposition = Disposition.ACT if label == 1 else Disposition.CLEAR
            v.confidence = cov
            v.reasons.append(
                f"conformal singleton {{{label}}} at {cov_str} coverage " "(calibrated decision)"
            )
            # ``require_calibrated_for_act`` is already satisfied here -- a
            # conformal singleton *is* the coverage certificate -- so no extra
            # gate is needed on this path (it gates the uncalibrated path only).
            return

        if size == 2:
            v.state = ThreeState.UNAVAILABLE
            v.disposition = Disposition.DEFER
            v.label = None
            v.confidence = None
            # A calibrated ambiguity is genuine -- more of the same data need not
            # break the tie -- so route it to a human, not a data request.
            v.resolvable_by_input = False
            v.reasons.append(
                f"conformal set {{0, 1}} at {cov_str} coverage: both labels "
                "admissible -- a calibrated don't-know"
            )
            return

        # size == 0 (or any other empty/degenerate set) -- atypical point.
        if self.policy.fail_closed_on_atypical:
            v.state = ThreeState.UNDECIDABLE
            v.disposition = Disposition.HOLD
            v.reasons.append(
                "conformal set empty: the point is atypical and no class "
                "explains it at the target coverage -- fail-closed"
            )
        else:
            v.state = ThreeState.UNAVAILABLE
            v.disposition = Disposition.DEFER
            v.resolvable_by_input = True
            v.reasons.append("conformal set empty: atypical point; deferring for review")
        v.label = None
        v.confidence = None

    def _from_threshold_band(self, ev: Evidence, v: _Verdict) -> None:
        """Uncalibrated fallback: ground by the threshold, abstain in the band."""
        margin = self.policy.indecision_margin
        distance = ev.anomaly_prob - ev.threshold

        # Inclusive band (``threshold +/- margin``): a probability sitting exactly
        # on the boundary is an uncalibrated don't-know, so it abstains too --
        # the fail-closed reading of the band the docstring describes.
        if abs(distance) <= margin:
            v.state = ThreeState.UNAVAILABLE
            v.disposition = Disposition.DEFER
            v.label = None
            v.confidence = None
            v.resolvable_by_input = True
            v.reasons.append(
                f"probability {ev.anomaly_prob:.3f} within +/-{margin:g} of "
                f"threshold {ev.threshold:.3f} and no coverage certificate -- "
                "an uncalibrated don't-know"
            )
            return

        # Outside the band: the threshold side grounds the label, but without a
        # certificate the confidence is a margin heuristic, not a guarantee.
        label = 1 if distance >= margin else 0
        if label == 1 and self.policy.require_calibrated_for_act:
            v.state = ThreeState.UNAVAILABLE
            v.disposition = Disposition.DEFER
            v.label = None
            v.confidence = None
            v.resolvable_by_input = True
            v.reasons.append(
                "positive call requires a calibrated coverage certificate "
                "(require_calibrated_for_act) and none was fit this run"
            )
            return

        v.state = ThreeState.GROUNDED
        v.label = label
        v.disposition = Disposition.ACT if label == 1 else Disposition.CLEAR
        cal = self.confidence_calibrator
        calibrated_ok = False
        if cal is not None and getattr(cal, "is_calibrated", False):
            # A fitted score->probability calibrator gives a real calibrated
            # confidence even without a conformal coverage certificate.
            # transform_one() reports P(anomaly); confidence is in the chosen
            # *verdict* (symmetric like the margin-heuristic fallback below
            # and the conformal-coverage path above), so a CLEAR call's
            # confidence is P(not anomaly) = 1 - P(anomaly), not P(anomaly)
            # itself -- otherwise a confidently-correct CLEAR (e.g.
            # P(anomaly)=0.03) would be misreported as near-zero confidence.
            # The calibrator is typed ``Any`` and the contract only requires a
            # ``transform_one`` method, so a custom/misbehaving one could raise,
            # or return a non-numeric / out-of-range / non-finite value. Any
            # such misbehaviour must NOT crash the decider and must NOT yield a
            # degenerate confidence: on failure we fall through to the margin
            # heuristic and record the fallback, rather than fabricating a
            # calibrated number.
            try:
                p_raw = float(cal.transform_one(ev.anomaly_prob))
            except Exception:  # any misbehaving calibrator -> margin fallback
                p_raw = float("nan")
            if math.isfinite(p_raw):
                p_anomaly = min(1.0, max(0.0, p_raw))
                v.confidence = p_anomaly if label == 1 else 1.0 - p_anomaly
                v.reasons.append(
                    f"probability {ev.anomaly_prob:.3f} past threshold "
                    f"{ev.threshold:.3f}; confidence {v.confidence:.3f} from the fitted "
                    "score calibrator (calibrated, but no distribution-free coverage "
                    "certificate)"
                )
                calibrated_ok = True
            else:
                v.reasons.append(
                    "confidence calibrator misbehaved (raised or returned a "
                    "non-finite value); falling back to the uncalibrated margin heuristic"
                )
        if not calibrated_ok:
            # Margin confidence in [0.5, 1.0]: how far past the threshold we are.
            # Reached when no calibrator is fitted, or a fitted one misbehaved.
            v.confidence = min(1.0, 0.5 + abs(distance))
            v.reasons.append(
                f"probability {ev.anomaly_prob:.3f} is {abs(distance):.3f} past "
                f"threshold {ev.threshold:.3f} (uncalibrated margin heuristic: no "
                "calibrator fitted and no coverage guarantee)"
            )

    def _apply_demotions(self, ev: Evidence, v: _Verdict) -> None:
        """Weaken a grounded verdict to a deferral on disagreement / drift."""
        if (
            ev.symbolic_satisfaction is not None
            and ev.symbolic_satisfaction < self.policy.symbolic_agreement_floor
        ):
            # Neural-vs-symbolic conflict needs human adjudication, not more data.
            self._demote(
                v,
                f"neuro-symbolic satisfaction {ev.symbolic_satisfaction:.2f} below "
                f"floor {self.policy.symbolic_agreement_floor:g}: the neural verdict "
                "and the symbolic rules disagree",
                resolvable_by_input=False,
            )
            return

        if self.policy.drift_is_deferring(ev.drift_severity):
            # Drift is resolved by recalibrating on fresh data -- a data request.
            self._demote(
                v,
                f"distribution drift ({ev.drift_severity}) detected: the "
                "calibration may no longer hold",
                resolvable_by_input=True,
            )

    @staticmethod
    def _demote(v: _Verdict, reason: str, *, resolvable_by_input: bool) -> None:
        """Demote a grounded verdict to a deferral (toward abstention only)."""
        v.state = ThreeState.UNAVAILABLE
        v.disposition = Disposition.DEFER
        v.label = None
        v.confidence = None
        v.resolvable_by_input = resolvable_by_input
        v.reasons.append(reason)

    @staticmethod
    def _caveats(ev: Evidence) -> tuple[str, ...]:
        """Non-blocking transparency notes that do not change the verdict."""
        caveats: list[str] = []
        if ev.ethical_gate_passed is None:
            caveats.append(
                "ethical-gate verdict is absent from this result; it is not part "
                "of this decision"
            )
        if not ev.calibrated:
            caveats.append(
                "no conformal coverage certificate was attached; confidence is "
                "a margin heuristic, not a distribution-free guarantee"
            )
        return tuple(caveats)


__all__ = ["DecisionAbstentionResponder"]
