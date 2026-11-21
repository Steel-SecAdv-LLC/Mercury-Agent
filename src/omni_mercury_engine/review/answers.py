# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The three answers a reviewer needs, read from one review result.

A flag on an uploaded artifact is not enough to act on.  Whoever reads it
needs to know **why this segment**, **how sure** the system is, and **what
record** they hold.  :func:`three_answers` states each in plain language and
lists, beside it, the fields of the result it was read from -- nothing in the
sentences is computed here that the record does not already carry.

The shape is the one ``mercury-agent demo`` uses for a numeric window
(:mod:`omni_mercury_engine.demo`): ``{"plain", "read_from"}`` per answer.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any

from omni_mercury_engine.review.decode import DOCUMENT, IMAGE

__all__ = ["HEADINGS", "KEYS", "render", "three_answers"]

HEADINGS = ("1. WHY THIS SEGMENT", "2. HOW SURE", "3. THE RECORD")
KEYS = ("why_this_segment", "how_sure", "the_record")


def _percent(value: Any) -> str:
    return f"{float(value):.0%}" if isinstance(value, (int, float)) else "an unstated"


def _where(kind: str, segment: dict[str, Any]) -> str:
    """Where a segment is, in the artifact's own units."""
    if kind == DOCUMENT:
        first, last = segment["characters"]
        return f'characters {first}-{last} ("{str(segment.get("excerpt", ""))[:80]}...")'
    if kind == IMAGE:
        x, y, width, height = segment["box"]
        return f"the {width} x {height} pixel tile at ({x}, {y})"
    start, stop = segment["seconds"]
    return f"{start:.2f}-{stop:.2f} s"


def _why(result: dict[str, Any]) -> dict[str, Any]:
    described = result["review"]
    kind, total = described["kind"], described["segments"]
    top = described.get("top_segment")
    flagged = [segment["index"] for segment in described.get("flagged_segments", [])]

    sentences = []
    if top is None:
        sentences.append(f"No segment of this {kind} carries a verdict.")
    else:
        apart = ", ".join(
            f"{feature['name']} ({feature['robust_z']:+.1f})" for feature in top["top_features"]
        )
        sentences.append(
            f"Segment {top['index']} of {total} carries the verdict: {_where(kind, top)}, with "
            f"an anomaly probability of {top['probability']:.3f}. What sets it apart from the "
            f"rest of this {kind}, in robust standard deviations from the {kind}'s own median: "
            f"{apart}."
        )
    sentences.append(
        f"{described['flagged_count']} of {total} segments are flagged"
        + (f" (most anomalous first: {flagged})." if flagged else ".")
    )
    sentences.append(
        f"The comparison is internal: a flagged segment is unlike the rest of this {kind}. That "
        "is not a finding that it was altered."
    )
    return {
        "plain": " ".join(sentences),
        "read_from": {
            "review.top_segment": top,
            "review.flagged_count": described["flagged_count"],
            "review.segments": total,
            "review.flagged_segments[].index": flagged,
            "review.basis": described.get("basis"),
        },
    }


def _how_sure(result: dict[str, Any]) -> dict[str, Any]:
    decision, described = result["decision"], result["review"]
    signals = decision["signals"]
    kind = described["kind"]
    conformal = (signals.get("model_provenance") or {}).get("conformal") or {}
    labels = list(signals.get("conformal_labels") or [])
    family = result.get("drift_detection") or {}
    shown = "{" + ", ".join(str(label) for label in labels) + "}"
    target = _percent(decision.get("coverage"))

    sentences = [
        f"The most anomalous segment's probability is {decision['anomaly_prob']:.3f} against a "
        f"threshold of {decision['threshold']:.3f}."
    ]
    if signals.get("conformal_set_size") is None:
        sentences.append("This model carries no conformal certificate, so no coverage is claimed.")
    else:
        meaning = {
            0: f"neither label is consistent with this {kind}",
            1: f"exactly one label is consistent with this {kind}",
        }.get(len(labels), "the two labels cannot be told apart for it: a calibrated don't-know")
        sentences.append(
            f"The conformal prediction set is {shown}. The certificate is batch-level: it was "
            f"calibrated on {conformal.get('n_calibration')} whole reference artifacts, each "
            f"represented by its most anomalous segment, to {target} coverage "
            f"({_percent(conformal.get('coverage_achieved'))} achieved). So it speaks to whether "
            f"this {kind} contains an anomalous segment, and {meaning}."
        )
    p_value = family.get("p_value")
    if family.get("status") == "tested" and isinstance(p_value, (int, float)):
        sentences.append(
            f"But measured against the artifacts that calibration was done on, this {kind} is "
            f"not one of that family (p = {p_value:.3f}), so the {target} coverage is not claimed "
            "for it."
            if family.get("is_drift")
            else f"Measured against the artifacts that calibration was done on, this {kind} is "
            f"not distinguishable from that family (p = {p_value:.3f}), so the coverage applies."
        )
    scope = described.get("certificate_scope")
    if scope:
        sentences.append(f"What the certificate has been measured to cover: {scope}")
    stance = "abstains" if decision["abstained"] else "decides"
    confidence = decision.get("decision_confidence")
    sentences.append(
        f"Mercury therefore {stance}: state '{decision['state']}', "
        + (f"confidence {confidence:.2f}." if confidence is not None else "no confidence stated.")
    )
    return {
        "plain": " ".join(sentences),
        "read_from": {
            "decision.anomaly_prob": decision["anomaly_prob"],
            "decision.threshold": decision["threshold"],
            "decision.signals.conformal_labels": labels,
            "decision.coverage": decision.get("coverage"),
            "decision.signals.conformal_regime": signals.get("conformal_regime"),
            "decision.signals.model_provenance.conformal": conformal,
            "drift_detection.status": family.get("status"),
            "drift_detection.reference_source": family.get("reference_source"),
            "drift_detection.p_value": p_value,
            "decision.signals.drift_detected": signals.get("drift_detected"),
            "review.certificate_scope": scope,
            "decision.state": decision["state"],
            "decision.abstained": decision["abstained"],
            "decision.decision_confidence": confidence,
        },
    }


def _the_record(result: dict[str, Any]) -> dict[str, Any]:
    decision, described = result["decision"], result["review"]
    signals = decision["signals"]
    provenance = signals.get("model_provenance") or {}
    response = decision.get("response") or {}
    harm_gate = described.get("harm_gate") or {}
    gate = signals.get("ethical_gate_passed")
    score, floor = signals.get("ethical_score"), signals.get("ethical_threshold")

    human = ", which needs a human" if response.get("requires_human") else ""
    sentences = [
        f"The decision is '{decision['disposition']}' and the response is "
        f"'{response.get('action')}'{human}. The record gives its reasons: "
        + "; ".join(str(reason) for reason in decision.get("reasons", []))
        + "."
    ]
    sentences.append(
        f"It names the model that decided ({provenance.get('checkpoint')}, sha256 "
        f"{str(provenance.get('checkpoint_sha256'))[:16]}...) and the upload it decided about "
        f"({described.get('format')}, {described.get('bytes')} bytes, sha256 "
        f"{str(described.get('sha256'))[:16]}...)."
    )
    sentences.append(
        f"Before anything was measured the harm-uplift gate read {harm_gate.get('read')} "
        f"({harm_gate.get('characters_read')} characters) and permitted the review. It did "
        f"not inspect {harm_gate.get('not_inspected')}, and the record says so"
        + (
            f"; the engine's ethical gate {'passed' if gate else 'BLOCKED'} (score {score:.3f} "
            f"against a floor of {floor:.3f})."
            if isinstance(score, (int, float)) and isinstance(floor, (int, float))
            else "."
        )
    )
    sentences.append(
        "To contest it, review the same bytes again: the SHA-256 identifies them, and the same "
        "bytes with the same checkpoint give the same record."
    )
    return {
        "plain": " ".join(sentences),
        "read_from": {
            "decision.disposition": decision["disposition"],
            "decision.response.action": response.get("action"),
            "decision.response.requires_human": response.get("requires_human"),
            "decision.reasons": list(decision.get("reasons", [])),
            "decision.signals.model_provenance.checkpoint": provenance.get("checkpoint"),
            "decision.signals.model_provenance.checkpoint_sha256": provenance.get(
                "checkpoint_sha256"
            ),
            "review.sha256": described.get("sha256"),
            "review.bytes": described.get("bytes"),
            "review.harm_gate": harm_gate,
            "decision.signals.ethical_gate_passed": gate,
            "decision.signals.ethical_score": score,
            "decision.signals.ethical_threshold": floor,
        },
    }


def three_answers(result: dict[str, Any]) -> dict[str, Any]:
    """The three answers, in plain language, each with the fields it was read from.

    Args:
        result: A review result (:func:`omni_mercury_engine.review.review_bytes`).

    Returns:
        ``{"why_this_segment", "how_sure", "the_record"}``; each holds
        ``plain`` (the sentences) and ``read_from`` (result path -> value).
    """
    return {
        "why_this_segment": _why(result),
        "how_sure": _how_sure(result),
        "the_record": _the_record(result),
    }


def render(result: dict[str, Any], *, rerun: str | None = None) -> str:
    """A review as text: the three answers, then the complete decision record.

    Args:
        result: A review result carrying ``review.answers``.
        rerun: The command that reviews the same file again, printed last.

    Returns:
        The text.
    """
    from omni_mercury_engine.demo import _compact

    described = result["review"]
    answers = described.get("answers") or three_answers(result)
    name = described.get("filename") or "upload"
    lines = [
        f"Mercury Agent review: {name} ({described['kind']}, {described['format']}, "
        f"{described['bytes']} bytes, {described['segments']} segments)",
    ]
    for heading, key in zip(HEADINGS, KEYS, strict=True):
        plain = textwrap.fill(
            answers[key]["plain"], width=92, initial_indent="   ", subsequent_indent="   "
        )
        lines += ["", heading, plain, "   read from the result:"]
        lines += [
            f"     {path} = {json.dumps(value, default=str, ensure_ascii=False)}"
            for path, value in answers[key]["read_from"].items()
        ]
    lines += [
        "",
        "THE COMPLETE DECISION RECORD (every answer above was read from this result)",
        _compact(result["decision"]),
    ]
    if rerun:
        lines += ["", "Review the same file again:", f"  {rerun}"]
    return "\n".join(lines)
