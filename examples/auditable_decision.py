# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One decision, three answers, and a challenge you can run yourself.

An anomaly score says that a row is unusual.  Before anyone acts on it they
need three more things: why this row, how sure the system is (with a stated
coverage), and a record they can reconstruct and contest.  This example runs
the shipped flagship model, in the same posture ``POST /api/v1/detect/flagship``
and ``mercury-agent detect -d fusion`` serve it, on one committed window and
prints those three answers from the decision record.  Then it challenges the
decision: it replays the record's discovery artifacts against the data, and
shows that a tampered copy fails.

The window (``examples/data/auditable_window.csv``, 80 rows x 16 columns) is
built from the data the shipped checkpoint was trained on
(``scripts/train_default_fusion.build_dataset``), so the answers can be
checked against what was put in:

* its ordinary rows come from one of that data's four regimes;
* column X0 was made to lead column X1 by one step (coefficient 0.7);
* rows 40-43 are four of that data's labelled anomalies.

What comes back is worth reading in that light.  The four anomalies are
flagged and the planted driver is recovered.  The conformal set is the
singleton {1} at 90% coverage.  And the decision still *defers*: the ordinary
rows are one regime of the four the model was calibrated on, the drift test
sees that, and the coverage is not claimed for a window the calibration does
not describe.  A score of 1.000 alone would have hidden all of it.

Usage::

    python examples/auditable_decision.py              # print the three answers and challenge them
    python examples/auditable_decision.py --out DIR    # also keep decision.json and window.csv there
    python examples/auditable_decision.py --json       # the same, machine-readable

    # The same decision and challenge with the CLI alone:
    mercury-agent detect -i examples/data/auditable_window.csv -d fusion -o decision.json
    mercury-agent replay decision.json --data examples/data/auditable_window.csv

Exit status 0 when the decision's artifacts hold on replay and the tampered
copy fails; 1 otherwise.  Needs the full agent (``pip install -e ".[pqc,ml]"``).
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
WINDOW = Path(__file__).resolve().parent / "data" / "auditable_window.csv"

#: How the window was made (see :func:`build_window`).
ROWS, DRIVER_COEFFICIENT, ANOMALY_ROWS, SEED = 80, 0.7, (40, 41, 42, 43), 0


def build_window() -> np.ndarray[Any, Any]:
    """Rebuild the committed window from the shipped checkpoint's training data.

    Returns:
        The 80 x 16 window, rounded to four decimals as committed.
    """
    sys.path.insert(0, str(REPO))
    from scripts.train_default_fusion import build_dataset

    features, labels = build_dataset()
    normals = features[labels == 0].astype(float)
    anomalies = features[labels == 1].astype(float)
    # One regime: the ordinary rows near the first one (the regimes sit about
    # 22 apart in this space and are about 6 across).
    regime = normals[np.linalg.norm(normals - normals[0], axis=1) < 8.0]
    rng = np.random.default_rng(SEED)
    window = regime[rng.choice(len(regime), size=ROWS, replace=False)].copy()
    window[1:, 1] += DRIVER_COEFFICIENT * (window[:-1, 0] - regime[:, 0].mean())
    window[list(ANOMALY_ROWS)] = anomalies[
        rng.choice(len(anomalies), size=len(ANOMALY_ROWS), replace=False)
    ]
    rounded: np.ndarray[Any, Any] = np.round(window, 4)
    return rounded


def decide(window: np.ndarray[Any, Any]) -> dict[str, Any]:
    """Run the shipped flagship model on ``window`` in the served posture.

    Returns:
        The detection result, with its ``decision`` record.
    """
    from omni_mercury_engine.engine import OmniMercuryEngine

    engine = OmniMercuryEngine(mode="fusion", require_explicit_fit=False)
    if not engine.load_default_fusion_checkpoint():
        raise SystemExit("the shipped fusion checkpoint is not installed")
    engine.enable_decision_layer()
    engine.enable_cognitive_analysis(enable_curiosity=True)
    result: dict[str, Any] = engine.detect_with_fusion(window)
    return result


def three_answers(result: dict[str, Any]) -> dict[str, Any]:
    """Read the three answers out of a detection result's decision record.

    Returns:
        ``{"why_this_row", "how_sure", "the_record"}``, each a mapping of the
        record's own fields.
    """
    decision = result["decision"]
    signals = decision["signals"]
    cognitive = signals.get("cognitive") or {}
    hypothesis = cognitive.get("hypothesis") or {}
    falsifier = hypothesis.get("falsifier") or {}
    provenance = signals.get("model_provenance") or {}
    conformal = provenance.get("conformal") or {}
    importance = (signals.get("contributions") or {}).get("detector_importance") or {}
    return {
        "why_this_row": {
            "top_row": result.get("top_row"),
            "rows_flagged": [i for i, flag in enumerate(result.get("is_anomaly_rows", [])) if flag],
            "detector_importance": dict(
                sorted(importance.items(), key=lambda item: item[1], reverse=True)[:5]
            ),
            "causal_factors": cognitive.get("causal_factors") or [],
            "hypothesis": hypothesis.get("conclusion"),
            "weakest_premise": falsifier.get("weakest_step"),
            "withdrawn_if": falsifier.get("withdrawn_if"),
        },
        "how_sure": {
            "anomaly_prob": decision["anomaly_prob"],
            "threshold": decision["threshold"],
            "conformal_set": list(signals.get("conformal_labels") or []),
            "coverage_target": decision["coverage"],
            "coverage_achieved": conformal.get("coverage_achieved"),
            "n_calibration": conformal.get("n_calibration"),
            "calibrated": decision["calibrated"],
            "drift": signals.get("drift_severity") if signals.get("drift_detected") else None,
            "state": decision["state"],
            "abstained": decision["abstained"],
            "decision_confidence": decision["decision_confidence"],
        },
        "the_record": {
            "disposition": decision["disposition"],
            "response": decision["response"]["action"],
            "requires_human": decision["response"]["requires_human"],
            "reasons": decision["reasons"],
            "checkpoint": provenance.get("checkpoint"),
            "checkpoint_sha256": provenance.get("checkpoint_sha256"),
            "ethical_gate_passed": signals.get("ethical_gate_passed"),
            "ethical_score": signals.get("ethical_score"),
            "ethical_threshold": signals.get("ethical_threshold"),
            "artifacts": sorted(cognitive.get("discovery") or {}),
        },
    }


def challenge(result: dict[str, Any], window: np.ndarray[Any, Any]) -> dict[str, Any]:
    """Replay the decision's artifacts against the data, then a tampered copy.

    Returns:
        ``{"replay", "holds", "tampered_replay", "tampered_fails"}``.
    """
    from omni_mercury_engine.discovery import FAILS, HOLDS, replay_record

    replayed = replay_record(result, data=window)
    tampered = copy.deepcopy(result)
    for carrier in (tampered["decision"]["signals"]["cognitive"], tampered["cognitive_analysis"]):
        carrier["discovery"]["causal"]["claim"]["edges"] = []
    tampered_replay = replay_record(tampered, data=window)
    return {
        "replay": replayed,
        "holds": bool(replayed) and all(report["status"] == HOLDS for report in replayed),
        "tampered_replay": tampered_replay,
        "tampered_fails": any(report["status"] == FAILS for report in tampered_replay),
    }


def _print(answers: dict[str, Any], outcome: dict[str, Any], kept: Path | None) -> None:
    why, sure, record = answers["why_this_row"], answers["how_sure"], answers["the_record"]
    weights = ", ".join(f"{name} {value:.3f}" for name, value in why["detector_importance"].items())
    print("Mercury Agent: one decision, three answers")
    print(f"window: {WINDOW.relative_to(REPO)} ({ROWS} rows x 16 columns)")
    print()
    print("1. WHY THIS ROW")
    print(f"   row {why['top_row']} carries the verdict; rows flagged: {why['rows_flagged']}")
    print(f"   detectors, by fusion weight: {weights}")
    for factor in why["causal_factors"][:3]:
        print(f"   lead-lag: {factor}")
    if why["hypothesis"]:
        print(f"   hypothesis: {why['hypothesis']}")
        print(f"   weakest premise: {why['weakest_premise']}")
        print(f"   withdrawn if: {why['withdrawn_if']}")
    print()
    print("2. HOW SURE")
    print(f"   P(anomaly) {sure['anomaly_prob']:.3f} against threshold {sure['threshold']:.3f}")
    labels = "{" + ", ".join(str(label) for label in sure["conformal_set"]) + "}"
    print(
        f"   conformal set {labels} at {sure['coverage_target']:.0%} target coverage, calibrated "
        f"on {sure['n_calibration']} held-out points ({sure['coverage_achieved']:.1%} achieved)"
    )
    print(f"   drift against the model's reference: {sure['drift'] or 'none detected'}")
    print(
        f"   state {sure['state']}, abstained {sure['abstained']}, "
        f"confidence {sure['decision_confidence']}"
    )
    print()
    print("3. THE RECORD")
    human = " (needs a human)" if record["requires_human"] else ""
    print(f"   disposition {record['disposition']}; response {record['response']}{human}")
    for reason in record["reasons"]:
        print(f"   because: {reason}")
    print(f"   model {record['checkpoint']} sha256 {str(record['checkpoint_sha256'])[:16]}...")
    gate = "passed" if record["ethical_gate_passed"] else "BLOCKED"
    print(
        f"   ethical gate {gate}: score {record['ethical_score']:.3f} against "
        f"threshold {record['ethical_threshold']:.3f}"
    )
    print(f"   replayable artifacts: {', '.join(record['artifacts']) or 'none'}")
    print()
    print("CHALLENGE THIS DECISION (replayed against the data)")
    for report in outcome["replay"]:
        print(f"   {report['kind']:<11}{report['status']:<8}{report['detail']}")
    print(f"   the decision's findings {'HOLD' if outcome['holds'] else 'DO NOT HOLD'}")
    print("   a copy with the lead-lag finding removed:")
    for report in outcome["tampered_replay"]:
        print(f"   {report['kind']:<11}{report['status']:<8}{report['detail']}")
    if kept is not None:
        print()
        print("Run the challenge yourself:")
        print(f"   mercury-agent replay {kept / 'decision.json'} --data {kept / 'window.csv'}")


def main(argv: list[str] | None = None) -> int:
    """Print the three answers for the committed window and challenge them."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument(
        "--out", type=Path, help="keep decision.json and window.csv in this directory"
    )
    parser.add_argument(
        "--json", action="store_true", help="print the answers and the challenge as JSON"
    )
    parser.add_argument(
        "--rebuild-window",
        action="store_true",
        help="rebuild examples/data/auditable_window.csv from the training data and exit",
    )
    args = parser.parse_args(argv)
    if args.rebuild_window:
        WINDOW.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(WINDOW, build_window(), delimiter=",", fmt="%.4f")
        print(f"wrote {WINDOW.relative_to(REPO)}")
        return 0

    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)
    window = np.loadtxt(WINDOW, delimiter=",")
    result = decide(window)
    answers, outcome = three_answers(result), challenge(result, window)
    kept = args.out
    if kept is None and not args.json:
        kept = Path(tempfile.mkdtemp(prefix="mercury-decision-"))
    if kept is not None:
        kept.mkdir(parents=True, exist_ok=True)
        (kept / "decision.json").write_text(json.dumps(result), encoding="utf-8")
        np.savetxt(kept / "window.csv", window, delimiter=",", fmt="%.4f")
    if args.json:
        print(json.dumps({"answers": answers, "challenge": outcome}, indent=2, default=str))
    else:
        _print(answers, outcome, kept)
    return 0 if outcome["holds"] and outcome["tampered_fails"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
