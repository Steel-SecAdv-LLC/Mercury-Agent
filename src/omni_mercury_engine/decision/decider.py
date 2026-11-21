# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
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
   a detector the fusion network was trained on did not report (the
   probability rests on partial evidence), the neuro-symbolic paths disagree,
   sufficiently severe drift means the
   calibration may no longer hold, or the GOSNN fused-state detection head
   (when its merit-gated checkpoint shipped one) strongly disagrees with the
   grounded label at its validation-selected thresholds.  Overlays only ever
   weaken a verdict toward abstention, never the reverse.  Every applicable
   overlay leaves its reason on the record (the state is demoted once), and
   the deferral is resolvable by re-input only when all of them are.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from omni_mercury_engine.decision import hazard_evidence
from omni_mercury_engine.decision.evidence import Evidence
from omni_mercury_engine.decision.policy import DecisionPolicy
from omni_mercury_engine.decision.record import DecisionRecord
from omni_mercury_engine.decision.response import ResponsePlan, ResponsePolicy
from omni_mercury_engine.decision.states import Disposition
from omni_mercury_engine.verifiers.three_state import ThreeState


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
            # ``None`` when no probability existed.  The record used to carry
            # a 0.0 placeholder here, which ``explain()`` printed as
            # ``p=0.000`` and the CAP bridge sent out as "Score 0.00" -- a
            # fabricated all-clear number on a decision that abstained
            # precisely because there was no number.
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

        # Stage 2 -- no usable probability: abstain, fail-closed.  ``None``
        # means the result carried no finite ``anomaly_prob`` (missing, NaN,
        # inf, unparseable).  That is "decidable in principle, but the input
        # was not produced this run" -- exactly ``UNAVAILABLE`` in the
        # three-state taxonomy -- so it defers for a scored re-run rather than
        # grounding anything.  This outranks even a conformal label set: a
        # certificate is derived from the calibrated probability upstream, so
        # a result carrying one without the probability is malformed, and a
        # malformed result may only ever abstain.
        #
        # Which abstention depends on the certificate.  A NaN score and an
        # EMPTY conformal set arrive together in normal operation: the
        # classifier's ``nan <= q`` is false for both labels, so a fitted
        # certificate answers a NaN score with the empty set.  That is the
        # atypical-point case -- no class explains the point -- and it holds
        # fail-closed exactly as an empty set does when the probability is
        # present; deferring it as a resolvable "please re-run" would be the
        # more permissive answer to the less trustworthy input.
        prob = ev.anomaly_prob
        if prob is None:
            if ev.conformal_set_size == 0 and self.policy.fail_closed_on_atypical:
                v.state = ThreeState.UNDECIDABLE
                v.disposition = Disposition.HOLD
                v.label = None
                v.confidence = None
                v.reasons.append(
                    "anomaly probability is absent or non-finite and the "
                    "conformal set is empty: the point is atypical and no "
                    "class explains it at the target coverage -- fail-closed"
                )
                return
            v.state = ThreeState.UNAVAILABLE
            v.disposition = Disposition.DEFER
            v.label = None
            v.confidence = None
            v.resolvable_by_input = True
            v.reasons.append(
                "anomaly probability is absent or non-finite: no usable "
                "evidence to ground a verdict -- abstaining fail-closed; "
                "re-run with a scored result"
            )
            return

        # Stage 2/3 -- ground the base verdict from the certificate or the band.
        if ev.conformal_set_size is not None:
            self._from_conformal(ev, v)
        else:
            self._from_threshold_band(ev, v, prob)

        # Stage 4 -- demotion overlays (only weaken a grounded verdict).
        if v.state is ThreeState.GROUNDED:
            self._apply_demotions(ev, v)

        # Stage 5 -- the critique's other direction: a reliable, *calibrated*
        # critique at or above the policy floor may dispute a resolvable
        # abstention, on the record.  The challenge is stated in ``reasons``
        # and changes nothing: a critique can challenge but not overturn a
        # fail-closed gate (an UNDECIDABLE hold — ethical block, atypical
        # point — is never challenged), and promotion would need measured
        # evidence of its own.  This closes the "one-way demotions only"
        # asymmetry without trading away fail-closed semantics.
        if (
            v.state is ThreeState.UNAVAILABLE
            and v.disposition is Disposition.DEFER
            and isinstance(ev.cognitive, Mapping)
        ):
            cog = ev.cognitive
            conf = cog.get("confidence")
            if (
                cog.get("is_reliable") is True
                and cog.get("confidence_calibrated") is True
                and isinstance(conf, (int, float))
                and not isinstance(conf, bool)
                and float(conf) >= self.policy.critique_support_confidence_floor
            ):
                v.reasons.append(
                    f"challenge on the record: the self-critique disputes this "
                    f"abstention (reliable, calibrated confidence {float(conf):.2f} "
                    f">= floor {self.policy.critique_support_confidence_floor:g}); "
                    "the abstention stands — a critique may challenge but not "
                    "overturn a fail-closed gate"
                )

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

    def _from_threshold_band(self, ev: Evidence, v: _Verdict, prob: float) -> None:
        """Uncalibrated fallback: ground by the threshold, abstain in the band.

        ``prob`` is the evidence's ``anomaly_prob`` already proven present by
        ``_classify`` (an absent probability abstains before this stage runs).
        """
        margin = self.policy.indecision_margin
        distance = prob - ev.threshold

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
                f"probability {prob:.3f} within +/-{margin:g} of "
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
                p_raw = float(cal.transform_one(prob))
            except Exception:  # any misbehaving calibrator -> margin fallback
                p_raw = float("nan")
            if math.isfinite(p_raw):
                p_anomaly = min(1.0, max(0.0, p_raw))
                v.confidence = p_anomaly if label == 1 else 1.0 - p_anomaly
                v.reasons.append(
                    f"probability {prob:.3f} past threshold "
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
                f"probability {prob:.3f} is {abs(distance):.3f} past "
                f"threshold {ev.threshold:.3f} (uncalibrated margin heuristic: no "
                "calibrator fitted and no coverage guarantee)"
            )

    def _apply_demotions(self, ev: Evidence, v: _Verdict) -> None:
        """Weaken a grounded verdict to a deferral on partial evidence / disagreement / drift.

        Every applicable overlay is collected before the verdict is touched,
        so a record demoted for several reasons at once names all of them --
        the gate used to stop at the first match, and a simultaneous symbolic
        conflict or drift silently vanished from the audit trail.  The state
        is demoted exactly once, and the deferral is resolvable by re-input
        only when *every* matched overlay is (a single human-adjudication
        reason among them routes the whole deferral to a human).
        """
        demotions: list[tuple[str, bool]] = []

        if ev.skipped_detectors:
            # A detector the fusion network was trained on did not report
            # (timed out, refused by a saturated executor, declined, failed):
            # the probability -- and any certificate derived from it -- was
            # computed from part of the evidence the calibration assumed.
            # A re-run with every detector reporting resolves it.
            demotions.append(
                (
                    "partial evidence: trained detector(s) did not report this run "
                    f"({', '.join(ev.skipped_detectors)}); a verdict on part of the "
                    "evidence is not grounded",
                    True,
                )
            )

        if (
            ev.symbolic_satisfaction is not None
            and ev.symbolic_satisfaction < self.policy.symbolic_agreement_floor
        ):
            # Neural-vs-symbolic conflict needs human adjudication, not more data.
            demotions.append(
                (
                    f"neuro-symbolic satisfaction {ev.symbolic_satisfaction:.2f} below "
                    f"floor {self.policy.symbolic_agreement_floor:g}: the neural verdict "
                    "and the symbolic rules disagree",
                    False,
                )
            )

        if ev.drift_detected and self.policy.drift_is_deferring(ev.drift_severity):
            # Drift is resolved by recalibrating on fresh data -- a data request.
            demotions.append(
                (
                    f"distribution drift ({ev.drift_severity}) detected: the "
                    "calibration may no longer hold",
                    True,
                )
            )

        # GOSNN disagreement overlay: the merit-gated fused-state detection
        # head corroborates (or contests) the grounded label.  Strong
        # disagreement at the thresholds validated during training demotes to
        # a deferral -- like the symbolic conflict above, this is a
        # model-vs-model contradiction for a human to adjudicate, not a
        # data request.  The signal is absent (None) unless a shipped,
        # detection-metric-gated head produced it, so the overlay is inert on
        # observability-only builds.  Inclusive comparisons: a probability
        # exactly at a demotion threshold defers (fail-closed reading).
        # ``v.label`` is still the grounded label here: no demotion has been
        # applied yet, so the head is compared against the verdict it contests.
        if self.policy.defer_on_gosnn_disagreement and ev.gosnn_anomaly_prob is not None:
            if (
                v.label == 1
                and ev.gosnn_demote_act_below is not None
                and ev.gosnn_anomaly_prob <= ev.gosnn_demote_act_below
            ):
                demotions.append(
                    (
                        f"GOSNN fused-state detection head estimates P(anomaly)="
                        f"{ev.gosnn_anomaly_prob:.3f} <= {ev.gosnn_demote_act_below:g} "
                        "against a grounded positive: the cross-detector fusion and "
                        "the serving verdict disagree",
                        False,
                    )
                )
            elif (
                v.label == 0
                and ev.gosnn_demote_clear_above is not None
                and ev.gosnn_anomaly_prob >= ev.gosnn_demote_clear_above
            ):
                demotions.append(
                    (
                        f"GOSNN fused-state detection head estimates P(anomaly)="
                        f"{ev.gosnn_anomaly_prob:.3f} >= {ev.gosnn_demote_clear_above:g} "
                        "against a grounded negative: the cross-detector fusion and "
                        "the serving verdict disagree",
                        False,
                    )
                )

        # Cognitive self-critique overlay (dispute direction): the engine's
        # own critic found this detection unreliable.  It counts only when
        # the critique is *measured* — at least one transparency flag says a
        # real quantity (ensemble/MC variance, heteroscedastic residuals, a
        # fitted confidence calibrator) stood behind the reliability verdict.
        # A critique of placeholder uncertainties is surfaced in the record's
        # signals but may not move a verdict.  Model-vs-critic contradiction
        # is for a human, not a data request.
        cog = ev.cognitive
        if (
            self.policy.defer_on_unreliable_critique
            and isinstance(cog, Mapping)
            and cog.get("is_reliable") is False
            and self._critique_measured(cog)
        ):
            demotions.append(
                (
                    "the engine's own self-critique found this detection "
                    "unreliable on measured grounds: a verdict its critic "
                    "disputes needs a human",
                    False,
                )
            )

        if demotions:
            self._demote(
                v,
                [reason for reason, _ in demotions],
                resolvable_by_input=all(resolvable for _, resolvable in demotions),
            )

    @staticmethod
    def _critique_measured(cog: Mapping[str, Any]) -> bool:
        """Whether the cognitive critique rests on at least one measured quantity."""
        return any(
            cog.get(key) is True
            for key in ("epistemic_measured", "aleatoric_measured", "confidence_calibrated")
        )

    @staticmethod
    def _demote(v: _Verdict, reasons: list[str], *, resolvable_by_input: bool) -> None:
        """Demote a grounded verdict to a deferral (toward abstention only).

        Applied once per verdict with every matched overlay's reason, so the
        state transition happens a single time however many overlays fired.
        """
        v.state = ThreeState.UNAVAILABLE
        v.disposition = Disposition.DEFER
        v.label = None
        v.confidence = None
        v.resolvable_by_input = resolvable_by_input
        v.reasons.extend(reasons)

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
        elif (
            ev.conformal_regime == "per_sample"
            and ev.conformal_rows is not None
            and ev.conformal_rows > 1
        ):
            caveats.append(
                f"the conformal certificate is per-sample and this verdict is the "
                f"batch-max row of {ev.conformal_rows}; selecting the max is not "
                f"exchangeable with the per-sample fit, so the stated coverage is "
                f"not a batch-level guarantee"
            )
        elif (
            ev.conformal_regime == "batch_max"
            and ev.conformal_rows is not None
            and ev.conformal_rows > 1
        ):
            caveats.append(
                f"the conformal certificate is batch-level: it covers whether this "
                f"batch of {ev.conformal_rows} rows contains an anomaly (through its "
                f"most anomalous row), not which other rows are anomalous"
            )
        # What the certificate was calibrated on and measured to cover, when
        # the model's own provenance says: a "90%" is a statement about that
        # calibration family, and the record should not let it read wider.
        scope = (ev.model_provenance or {}).get("certificate_scope")
        if ev.calibrated and isinstance(scope, str) and scope:
            caveats.append(f"certificate scope: {scope}")
        # What the harm-uplift gate upstream of this decision read, when the
        # result says: a gate that passed an upload it could only partly read
        # has not cleared the part it did not read, and the record says which.
        if ev.harm_gate_scope:
            caveats.append(f"harm gate scope: {ev.harm_gate_scope}")
        # What decisions about this hazard have been worth on real events
        # held out, where that has been measured: a "90%" reads the same for
        # every hazard, and has not held the same for every hazard.
        measured = hazard_evidence.caveat(ev.domain)
        if measured is not None:
            caveats.append(measured)
        return tuple(caveats)


__all__ = ["DecisionAbstentionResponder"]
