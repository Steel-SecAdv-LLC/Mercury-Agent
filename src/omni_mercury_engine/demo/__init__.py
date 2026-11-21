# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The ten-minute demonstration: one decision, three answers, one challenge.

The issue Mercury resolves is auditable early warning on high-dimensional or
multi-source data, where an anomaly score alone is not enough to act on.
Whoever must act needs three more things: why this row, how sure the system
is (with a stated coverage), and a record they can reconstruct and contest.

:func:`run` shows all three on a single real decision.  It scores one
committed window with the bundled flagship model, in the posture
``POST /api/v1/detect/flagship`` and ``mercury-agent detect -d fusion`` serve
it, states the three answers in plain language beside the complete decision
record they were read from, challenges the decision (replays its recorded
findings against the data, and shows a tampered copy failing), and ends with
the command that re-runs the challenge.  ``mercury-agent demo`` is this
function.

The window (``auditable_window.csv`` beside this file, 80 rows x 16 columns)
is built from the data the bundled checkpoint was trained on, so every answer
can be checked against what was put in (:data:`PLANTED`):

* its ordinary rows come from one of that data's four regimes;
* column X0 was made to lead column X1 by one step;
* rows 40-43 are four of that data's labelled anomalies.

The recipe that builds it is ``examples/auditable_decision.py --rebuild-window``;
``tests/test_demo_three_answers.py`` pins the committed file to the recipe.

Nothing here is a second implementation: the answers are read out of the
decision record the engine wrote, and the challenge is
:func:`omni_mercury_engine.discovery.replay_record`.
"""

from __future__ import annotations

import copy
import json
import textwrap
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "COLUMNS",
    "HEADINGS",
    "PLANTED",
    "ROWS",
    "WINDOW_PATH",
    "challenge",
    "decide",
    "load_window",
    "render",
    "run",
    "three_answers",
]

#: The committed window, shipped inside the package.
WINDOW_PATH = Path(__file__).resolve().parent / "auditable_window.csv"
ROWS, COLUMNS = 80, 16

#: What was put into the window, so the answers can be checked against it.
PLANTED: dict[str, Any] = {
    "driver": "X0",
    "target": "X1",
    "lag": 1,
    "coefficient": 0.7,
    "anomaly_rows": [40, 41, 42, 43],
    "regimes_in_window": 1,
    "regimes_in_calibration": 4,
}

#: The three answers, in the order they are asked.
HEADINGS = ("1. WHY THIS ROW", "2. HOW SURE", "3. THE RECORD")


def load_window() -> np.ndarray[Any, Any]:
    """The committed 80 x 16 window."""
    window: np.ndarray[Any, Any] = np.loadtxt(WINDOW_PATH, delimiter=",")
    return window


def decide(window: np.ndarray[Any, Any]) -> dict[str, Any]:
    """Score ``window`` with the bundled flagship model in the served posture.

    The bundled checkpoint is loaded by path, not through the serving default:
    an operator's own installed model has another width and another
    calibration, and the demonstration's window belongs to this one.

    Returns:
        The detection result, with its ``decision`` record.
    """
    from omni_mercury_engine.engine import OmniMercuryEngine, bundled_fusion_checkpoint_path

    engine = OmniMercuryEngine(mode="fusion", require_explicit_fit=False)
    engine.load_model(str(bundled_fusion_checkpoint_path()))
    engine.enable_decision_layer()
    engine.enable_cognitive_analysis(enable_curiosity=True)
    result: dict[str, Any] = engine.detect_with_fusion(window)
    return result


def _percent(value: Any) -> str:
    return f"{float(value):.0%}" if isinstance(value, (int, float)) else "an unstated"


def _why(result: dict[str, Any]) -> dict[str, Any]:
    decision = result["decision"]
    signals = decision["signals"]
    cognitive = signals.get("cognitive") or {}
    hypothesis = cognitive.get("hypothesis") or {}
    falsifier = hypothesis.get("falsifier") or {}
    importance = (signals.get("contributions") or {}).get("detector_importance") or {}
    leading = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True)[:3])
    flagged = [index for index, flag in enumerate(result.get("is_anomaly_rows", [])) if flag]
    factors = list(cognitive.get("causal_factors") or [])

    sentences = [
        f"Row {result.get('top_row')} carries the verdict; {len(flagged)} row(s) are flagged: {flagged}."
    ]
    if leading:
        weights = ", ".join(f"{name} ({value:.3f})" for name, value in leading.items())
        sentences.append(f"The fusion model weighed these feature groups most: {weights}.")
    if factors:
        sentences.append(
            f"Inside the window one column moves before another: {factors[0]}. "
            "The test behind it is corrected for every lag and pair it tried."
        )
    else:
        sentences.append("No lead-lag relation between columns survived correction.")
    if hypothesis.get("conclusion"):
        sentences.append(
            f"From the findings that hold, the record derives: {hypothesis['conclusion']} "
            "It is a candidate, stated with no confidence number, and it says how it fails: "
            f"its weakest premise is \"{falsifier.get('weakest_step')}\", and the conclusion is "
            f"withdrawn if {falsifier.get('withdrawn_if')}."
        )
    else:
        sentences.append("No hypothesis was derived: fewer than two measured findings hold.")
    return {
        "plain": " ".join(sentences),
        "read_from": {
            "top_row": result.get("top_row"),
            "is_anomaly_rows (flagged)": flagged,
            "decision.signals.contributions.detector_importance": leading,
            "decision.signals.cognitive.causal_factors": factors,
            "decision.signals.cognitive.hypothesis.conclusion": hypothesis.get("conclusion"),
            "decision.signals.cognitive.hypothesis.falsifier.weakest_step": falsifier.get(
                "weakest_step"
            ),
            "decision.signals.cognitive.hypothesis.falsifier.withdrawn_if": falsifier.get(
                "withdrawn_if"
            ),
        },
    }


def _how_sure(result: dict[str, Any]) -> dict[str, Any]:
    decision = result["decision"]
    signals = decision["signals"]
    conformal = (signals.get("model_provenance") or {}).get("conformal") or {}
    labels = list(signals.get("conformal_labels") or [])
    drift = signals.get("drift_severity") if signals.get("drift_detected") else None
    shown = "{" + ", ".join(str(label) for label in labels) + "}"
    target = _percent(decision.get("coverage"))

    sentences = [
        f"The probability of an anomaly is {decision['anomaly_prob']:.3f} against a threshold "
        f"of {decision['threshold']:.3f}."
    ]
    if signals.get("conformal_set_size") is None:
        sentences.append("This model carries no conformal certificate, so no coverage is claimed.")
    else:
        meaning = {
            0: "no label is consistent with this row: it is unlike anything the model was calibrated on",
            1: "exactly one label is consistent with this row",
        }.get(len(labels), "the labels cannot be told apart for this row")
        sentences.append(
            f"The conformal prediction set is {shown}: at the {target} coverage the model was "
            f"calibrated to ({conformal.get('n_calibration')} held-out points, "
            f"{_percent(conformal.get('coverage_achieved'))} achieved), {meaning}."
        )
    if drift:
        sentences.append(
            f"But the window's ordinary rows do not look like the data that calibration was done "
            f"on (drift: {drift}), so the {target} coverage is not claimed for this window."
        )
    else:
        sentences.append("No drift against the model's reference was detected.")
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
            "decision.signals.model_provenance.conformal": conformal,
            "decision.calibrated": decision.get("calibrated"),
            "decision.signals.drift_severity": drift,
            "decision.state": decision["state"],
            "decision.abstained": decision["abstained"],
            "decision.decision_confidence": confidence,
        },
    }


def _the_record(result: dict[str, Any]) -> dict[str, Any]:
    decision = result["decision"]
    signals = decision["signals"]
    provenance = signals.get("model_provenance") or {}
    response = decision.get("response") or {}
    artifacts = sorted((signals.get("cognitive") or {}).get("discovery") or {})
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
        f"It names the model that decided ({provenance.get('checkpoint')}, "
        f"sha256 {str(provenance.get('checkpoint_sha256'))[:16]}...)"
        + (
            f" and the ethical gate's outcome ({'passed' if gate else 'BLOCKED'}: "
            f"score {score:.3f} against a floor of {floor:.3f})."
            if isinstance(score, (int, float)) and isinstance(floor, (int, float))
            else " and the ethical gate's outcome."
        )
    )
    sentences.append(
        f"It carries {len(artifacts)} finding(s) that replay from the record alone "
        f"({', '.join(artifacts) or 'none'}): anyone holding it can re-run them and contest it."
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
            "decision.signals.ethical_gate_passed": gate,
            "decision.signals.ethical_score": score,
            "decision.signals.ethical_threshold": floor,
            "decision.signals.cognitive.discovery (replayable)": artifacts,
        },
    }


def three_answers(result: dict[str, Any]) -> dict[str, Any]:
    """The three answers, in plain language, each with the fields it was read from.

    Args:
        result: A flagship detection result carrying its ``decision`` record.

    Returns:
        ``{"why_this_row", "how_sure", "the_record"}``; each holds ``plain``
        (the sentences) and ``read_from`` (record path -> value).
    """
    return {
        "why_this_row": _why(result),
        "how_sure": _how_sure(result),
        "the_record": _the_record(result),
    }


def challenge(result: dict[str, Any], window: np.ndarray[Any, Any]) -> dict[str, Any]:
    """Replay the decision's findings against the data, then a tampered copy.

    Returns:
        ``{"replay", "holds", "tampered_replay", "tampered_fails"}``.
    """
    from omni_mercury_engine.discovery import FAILS, HOLDS, replay_record

    replayed = replay_record(result, data=window)
    tampered = copy.deepcopy(result)
    carriers = (
        (tampered.get("decision") or {}).get("signals", {}).get("cognitive") or {},
        tampered.get("cognitive_analysis") or {},
    )
    for carrier in carriers:
        causal = (carrier.get("discovery") or {}).get("causal")
        if causal is not None:
            causal["claim"]["edges"] = []
    tampered_replay = replay_record(tampered, data=window)
    return {
        "replay": replayed,
        "holds": bool(replayed) and all(report["status"] == HOLDS for report in replayed),
        "tampered_replay": tampered_replay,
        "tampered_fails": any(report["status"] == FAILS for report in tampered_replay),
    }


def _compact(value: Any, indent: int = 0) -> str:
    """JSON in which a list of plain values stays on one line: complete, and readable."""
    pad, inner = "  " * indent, "  " * (indent + 1)
    if isinstance(value, dict):
        if not value:
            return "{}"
        rows = [f"{inner}{json.dumps(str(k))}: {_compact(v, indent + 1)}" for k, v in value.items()]
        return "{\n" + ",\n".join(rows) + f"\n{pad}}}"
    if isinstance(value, (list, tuple)):
        if all(not isinstance(item, (dict, list, tuple)) for item in value):
            return json.dumps(list(value), default=str, ensure_ascii=False)
        rows = [f"{inner}{json.dumps(item, default=str, ensure_ascii=False)}" for item in value]
        return "[\n" + ",\n".join(rows) + f"\n{pad}]"
    return json.dumps(value, default=str, ensure_ascii=False)


def replay_command(kept: Path) -> str:
    """The exact command that re-runs the stored challenge recipe."""
    return f"mercury-agent replay {kept / 'decision.json'} --data {kept / 'window.csv'}"


def render(
    result: dict[str, Any], answers: dict[str, Any], outcome: dict[str, Any], kept: Path
) -> str:
    """The demonstration as text: three answers, the record, the challenge, the command.

    Args:
        result: The detection result.
        answers: :func:`three_answers` of it.
        outcome: :func:`challenge` of it.
        kept: Directory holding ``decision.json`` and ``window.csv``.

    Returns:
        The text; its last line is :func:`replay_command`.
    """
    lines = [
        "Mercury Agent: one decision, three answers",
        "",
        "A score says a row is unusual. Before anyone acts on it they need to know why this",
        "row, how sure the system is, and what record they can reconstruct and contest.",
        textwrap.fill(
            f"Window: {ROWS} rows x {COLUMNS} columns, built from the bundled model's own "
            f"training data. Planted, so the answers can be checked: {PLANTED['driver']} leads "
            f"{PLANTED['target']} by {PLANTED['lag']} step; rows "
            f"{PLANTED['anomaly_rows'][0]}-{PLANTED['anomaly_rows'][-1]} are labelled "
            f"anomalies; the ordinary rows are {PLANTED['regimes_in_window']} regime of the "
            f"{PLANTED['regimes_in_calibration']} the model was calibrated on.",
            width=92,
        ),
    ]
    for heading, key in zip(HEADINGS, ("why_this_row", "how_sure", "the_record"), strict=True):
        plain = textwrap.fill(
            answers[key]["plain"], width=92, initial_indent="   ", subsequent_indent="   "
        )
        lines += ["", heading, plain, "   read from the record:"]
        lines += [
            f"     {path} = {json.dumps(value, default=str, ensure_ascii=False)}"
            for path, value in answers[key]["read_from"].items()
        ]
    lines += [
        "",
        "THE COMPLETE DECISION RECORD (every answer above was read from this)",
        _compact(result["decision"]),
        "",
        "CHALLENGE THIS DECISION (its recorded findings, replayed against the data)",
    ]
    lines += [
        f"   {report['kind']:<11}{report['status']:<8}{report['detail']}"
        for report in outcome["replay"]
    ]
    lines.append(
        "   -> the decision's findings HOLD"
        if outcome["holds"]
        else "   -> the decision's findings DO NOT HOLD"
    )
    lines.append("   and a copy with the lead-lag finding removed:")
    lines += [
        f"   {report['kind']:<11}{report['status']:<8}{report['detail']}"
        for report in outcome["tampered_replay"]
    ]
    lines.append(
        "   -> the tampered copy FAILS"
        if outcome["tampered_fails"]
        else "   -> the tampered copy WAS NOT CAUGHT"
    )
    lines += [
        "",
        f"Kept: {kept / 'decision.json'} (the whole result) and {kept / 'window.csv'}.",
        "Re-run the stored challenge recipe yourself:",
        f"  {replay_command(kept)}",
    ]
    return "\n".join(lines)


def run(out: Path | None = None, *, as_json: bool = False) -> tuple[int, str]:
    """Run the demonstration.

    Args:
        out: Where to keep ``decision.json`` and ``window.csv`` (default
            ``./mercury-demo``).
        as_json: Return the answers, the record and the challenge as JSON.

    Returns:
        ``(exit_status, text)``.  The status is 0 only when the decision's
        findings hold on replay and the tampered copy fails.
    """
    kept = Path(out) if out is not None else Path("mercury-demo")
    window = load_window()
    result = decide(window)
    answers, outcome = three_answers(result), challenge(result, window)
    kept.mkdir(parents=True, exist_ok=True)
    (kept / "decision.json").write_text(json.dumps(result), encoding="utf-8")
    np.savetxt(kept / "window.csv", window, delimiter=",", fmt="%.4f")
    status = 0 if outcome["holds"] and outcome["tampered_fails"] else 1
    if as_json:
        payload = {
            "answers": answers,
            "decision": result["decision"],
            "challenge": outcome,
            "replay_command": replay_command(kept),
        }
        return status, json.dumps(payload, indent=2, default=str)
    return status, render(result, answers, outcome, kept)
