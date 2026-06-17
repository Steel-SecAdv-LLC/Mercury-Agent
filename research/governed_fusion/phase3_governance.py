# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 3 gated Reflexion, drift recalibration, and dormant-revival routing.

Phase 3 does not give Reflexion, drift monitors, or dormant-module revival
permission to mutate Mercury directly.  It turns those surfaces into candidate
evidence and routes every proposed change through the Phase 2 governed
promotion gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast

from research.governed_fusion.promotion_gate import (
    GateDecision,
    GateResult,
    evaluate_candidate,
)

JsonMap = dict[str, object]

_DEFAULT_MANIFEST = Path(__file__).resolve().with_name("manifest.json")
_DEFAULT_LEDGER = Path(__file__).resolve().with_name("ablation_ledger.json")
_DEFAULT_STORE_DIR = Path("artifacts/phase3_governance")
_TRIGGERING_DRIFT = frozenset({"high", "critical"})


class Phase3Surface(StrEnum):
    """Governed Phase 3 surfaces."""

    REFLEXION_THRESHOLD = "reflexion_threshold"
    DRIFT_RECALIBRATION = "drift_recalibration"
    DORMANT_REVIVAL = "dormant_revival"


class Phase3Action(StrEnum):
    """Action emitted by Phase 3 governance."""

    MAINTAIN = "maintain"
    QUEUE_REFLEXION_CANDIDATE = "queue_reflexion_candidate"
    QUEUE_RECALIBRATION_CANDIDATE = "queue_recalibration_candidate"
    QUEUE_DORMANT_REVIVAL_CANDIDATE = "queue_dormant_revival_candidate"
    REJECT = "reject"
    ROLLBACK = "rollback"


class ThresholdRecommender(Protocol):
    """Structural protocol for the existing AnomalyReflexion seam."""

    def get_threshold_recommendation(self) -> Mapping[str, object]:
        """Return the threshold recommendation to be gated."""
        ...


@dataclass(frozen=True)
class Phase3Decision:
    """JSON-serialisable Phase 3 routing decision."""

    schema_version: int
    phase: str
    surface: str
    action: str
    candidate_id: str | None
    gate_decision: str | None
    reasons: list[str]
    evidence: JsonMap
    gate_result: JsonMap | None
    decided_at: str


def _load_json(path: Path) -> JsonMap:
    with path.open(encoding="utf-8") as fh:
        return cast("JsonMap", json.load(fh))


def _as_mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _optional_mapping(value: object | None, path: str) -> Mapping[str, object] | None:
    if value is None:
        return None
    return _as_mapping(value, path)


def _as_str(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _as_bool(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be boolean")
    return value


def _decision(
    surface: Phase3Surface,
    action: Phase3Action,
    *,
    candidate_id: str | None = None,
    gate_result: GateResult | None = None,
    reasons: Iterable[str] = (),
    evidence: Mapping[str, object] | None = None,
) -> Phase3Decision:
    return Phase3Decision(
        schema_version=1,
        phase="phase3",
        surface=surface.value,
        action=action.value,
        candidate_id=candidate_id,
        gate_decision=gate_result.decision if gate_result else None,
        reasons=list(reasons),
        evidence=dict(evidence or {}),
        gate_result=asdict(gate_result) if gate_result else None,
        decided_at=datetime.now(UTC).isoformat(),
    )


def _action_from_gate(gate_result: GateResult, promote_action: Phase3Action) -> Phase3Action:
    if gate_result.decision == GateDecision.PROMOTE.value:
        return promote_action
    if gate_result.decision == GateDecision.ROLLBACK.value:
        return Phase3Action.ROLLBACK
    return Phase3Action.REJECT


def _gate_candidate(
    candidate_record: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    ledger: Mapping[str, object],
) -> GateResult:
    return evaluate_candidate(candidate_record, manifest=manifest, ledger=ledger)


def route_reflexion_threshold(
    recommendation: Mapping[str, object],
    *,
    candidate_record: Mapping[str, object] | None,
    manifest: Mapping[str, object],
    ledger: Mapping[str, object],
) -> Phase3Decision:
    """Route an AnomalyReflexion threshold recommendation through Phase 2."""

    surface = Phase3Surface.REFLEXION_THRESHOLD
    recommendation_value = _as_str(recommendation.get("recommendation"), "recommendation")
    evidence = {"recommendation": dict(recommendation)}
    if recommendation_value == "maintain":
        return _decision(
            surface,
            Phase3Action.MAINTAIN,
            reasons=["Reflexion recommendation is maintain; no candidate routed."],
            evidence=evidence,
        )
    if candidate_record is None:
        return _decision(
            surface,
            Phase3Action.REJECT,
            reasons=["Reflexion threshold change requires candidate evidence."],
            evidence=evidence,
        )
    gate_result = _gate_candidate(candidate_record, manifest=manifest, ledger=ledger)
    return _decision(
        surface,
        _action_from_gate(gate_result, Phase3Action.QUEUE_REFLEXION_CANDIDATE),
        candidate_id=gate_result.candidate_id,
        gate_result=gate_result,
        reasons=gate_result.reasons,
        evidence=evidence,
    )


def route_reflexion_executor(
    reflexion: ThresholdRecommender,
    *,
    candidate_record: Mapping[str, object] | None,
    manifest: Mapping[str, object],
    ledger: Mapping[str, object],
) -> Phase3Decision:
    """Read `AnomalyReflexion.get_threshold_recommendation()` and route it."""

    recommendation = _as_mapping(
        reflexion.get_threshold_recommendation(), "reflexion.get_threshold_recommendation()"
    )
    return route_reflexion_threshold(
        recommendation,
        candidate_record=candidate_record,
        manifest=manifest,
        ledger=ledger,
    )


def _triggering_drift_results(
    drift_results: Iterable[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    triggered: list[Mapping[str, object]] = []
    for idx, result in enumerate(drift_results):
        is_drift = _as_bool(result.get("is_drift"), f"drift_results[{idx}].is_drift")
        severity = _as_str(result.get("severity"), f"drift_results[{idx}].severity")
        if is_drift and severity in _TRIGGERING_DRIFT:
            triggered.append(result)
    return triggered


def route_drift_recalibration(
    drift_results: Iterable[Mapping[str, object]],
    *,
    candidate_record: Mapping[str, object] | None,
    manifest: Mapping[str, object],
    ledger: Mapping[str, object],
) -> Phase3Decision:
    """Route drift-triggered recalibration only after high/critical drift."""

    surface = Phase3Surface.DRIFT_RECALIBRATION
    drift_list = [dict(result) for result in drift_results]
    evidence = {"drift_results": drift_list, "trigger_severities": sorted(_TRIGGERING_DRIFT)}
    triggered = _triggering_drift_results(drift_list)
    if not triggered:
        return _decision(
            surface,
            Phase3Action.MAINTAIN,
            reasons=["No high/critical drift trigger; recalibration is not routed."],
            evidence=evidence,
        )
    if candidate_record is None:
        return _decision(
            surface,
            Phase3Action.REJECT,
            reasons=["High/critical drift requires recalibration candidate evidence."],
            evidence=evidence,
        )
    gate_result = _gate_candidate(candidate_record, manifest=manifest, ledger=ledger)
    return _decision(
        surface,
        _action_from_gate(gate_result, Phase3Action.QUEUE_RECALIBRATION_CANDIDATE),
        candidate_id=gate_result.candidate_id,
        gate_result=gate_result,
        reasons=gate_result.reasons,
        evidence=evidence | {"triggered_drift": [dict(result) for result in triggered]},
    )


def route_dormant_revival_candidate(
    candidate_name: str,
    verdict: Mapping[str, object],
    *,
    candidate_record: Mapping[str, object] | None,
    manifest: Mapping[str, object],
    ledger: Mapping[str, object],
) -> Phase3Decision:
    """Route a dormant-module revival candidate through Phase 2."""

    surface = Phase3Surface.DORMANT_REVIVAL
    carries_signal = _as_bool(verdict.get("carries_signal"), "verdict.carries_signal")
    evidence = {"candidate": candidate_name, "verdict": dict(verdict)}
    if not carries_signal:
        return _decision(
            surface,
            Phase3Action.MAINTAIN,
            candidate_id=candidate_name,
            reasons=["Dormant candidate did not clear the pre-registered signal bar."],
            evidence=evidence,
        )
    if candidate_record is None:
        return _decision(
            surface,
            Phase3Action.REJECT,
            candidate_id=candidate_name,
            reasons=["Dormant revival requires promotion-gate candidate evidence."],
            evidence=evidence,
        )
    gate_result = _gate_candidate(candidate_record, manifest=manifest, ledger=ledger)
    return _decision(
        surface,
        _action_from_gate(gate_result, Phase3Action.QUEUE_DORMANT_REVIVAL_CANDIDATE),
        candidate_id=gate_result.candidate_id,
        gate_result=gate_result,
        reasons=gate_result.reasons,
        evidence=evidence,
    )


def evaluate_phase3_report(
    report: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    ledger: Mapping[str, object],
) -> list[Phase3Decision]:
    """Evaluate a composite Phase 3 report JSON."""

    decisions: list[Phase3Decision] = []
    reflexion = _optional_mapping(report.get("reflexion_threshold"), "reflexion_threshold")
    if reflexion is not None:
        decisions.append(
            route_reflexion_threshold(
                _as_mapping(reflexion.get("recommendation"), "reflexion_threshold.recommendation"),
                candidate_record=_optional_mapping(
                    reflexion.get("candidate_record"), "reflexion_threshold.candidate_record"
                ),
                manifest=manifest,
                ledger=ledger,
            )
        )
    drift = _optional_mapping(report.get("drift_recalibration"), "drift_recalibration")
    if drift is not None:
        raw_results = drift.get("drift_results")
        if not isinstance(raw_results, list):
            raise ValueError("drift_recalibration.drift_results must be a list")
        decisions.append(
            route_drift_recalibration(
                [_as_mapping(item, "drift_result") for item in raw_results],
                candidate_record=_optional_mapping(
                    drift.get("candidate_record"), "drift_recalibration.candidate_record"
                ),
                manifest=manifest,
                ledger=ledger,
            )
        )
    dormant = _optional_mapping(report.get("dormant_revival"), "dormant_revival")
    if dormant is not None:
        candidates = _as_mapping(dormant.get("candidates"), "dormant_revival.candidates")
        records = (
            _optional_mapping(dormant.get("candidate_records"), "dormant_revival.records") or {}
        )
        for name, verdict in sorted(candidates.items()):
            decisions.append(
                route_dormant_revival_candidate(
                    name,
                    _as_mapping(verdict, f"dormant_revival.candidates.{name}"),
                    candidate_record=_optional_mapping(
                        records.get(name), f"candidate_records.{name}"
                    ),
                    manifest=manifest,
                    ledger=ledger,
                )
            )
    if not decisions:
        raise ValueError("Phase 3 report contains no recognized sections")
    return decisions


class Phase3DecisionStore:
    """Append-only store for Phase 3 routing decisions."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def write_many(self, decisions: Iterable[Phase3Decision]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        record_path = self.root / f"{stamp}-phase3-decisions.json"
        payload = [asdict(decision) for decision in decisions]
        with record_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, allow_nan=False)
            fh.write("\n")
        with (self.root / "index.jsonl").open("a", encoding="utf-8") as fh:
            for decision in payload:
                row = {
                    "surface": decision["surface"],
                    "action": decision["action"],
                    "candidate_id": decision["candidate_id"],
                    "record": str(record_path),
                    "decided_at": decision["decided_at"],
                }
                json.dump(row, fh, sort_keys=True, allow_nan=False)
                fh.write("\n")
        return record_path


def dormant_revival_report_to_section(
    report: Mapping[str, object],
    *,
    candidate_records: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Convert a ``benchmarks.dormant_module_revival`` report into a routable section.

    The revival benchmark nests its measured verdicts under
    ``verdicts.candidates``; Phase 3 routing consumes them under the report's
    ``dormant_revival.candidates`` key, with optional per-candidate
    promotion-gate ``candidate_records``. This closes the measurement→routing
    loop: the recurring workflow measures revival on real labels, then routes
    every verdict through the governed gate instead of merely uploading it.

    Args:
        report: The parsed ``dormant_module_revival.json`` benchmark report.
        candidate_records: Optional per-candidate held-out-replay records for
            candidates that have promotion-gate evidence.

    Returns:
        A ``dormant_revival`` report section suitable for
        :func:`evaluate_phase3_report`.
    """
    verdicts = _as_mapping(report.get("verdicts"), "dormant_revival_report.verdicts")
    candidates = _as_mapping(
        verdicts.get("candidates"), "dormant_revival_report.verdicts.candidates"
    )
    section: dict[str, object] = {"candidates": dict(candidates)}
    if candidate_records:
        section["candidate_records"] = dict(candidate_records)
    return section


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Composite Phase 3 evidence JSON.")
    parser.add_argument(
        "--dormant-revival",
        type=Path,
        help="A benchmarks.dormant_module_revival report to route through the gate.",
    )
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--ledger", type=Path, default=_DEFAULT_LEDGER)
    parser.add_argument("--store-dir", type=Path, default=_DEFAULT_STORE_DIR)
    parser.add_argument("--out", type=Path, help="Write decisions JSON instead of using store-dir.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any routed decision rejects or rolls back.",
    )
    return parser


def _error_decision(reasons: list[str]) -> Phase3Decision:
    """Build a fail-closed REJECT decision for an unevaluable Phase 3 report."""
    return Phase3Decision(
        schema_version=1,
        phase="phase3",
        surface="error",
        action=Phase3Action.REJECT.value,
        candidate_id=None,
        gate_decision=None,
        reasons=reasons,
        evidence={"error": "phase3 report could not be evaluated"},
        gate_result=None,
        decided_at=datetime.now(UTC).isoformat(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    errored = False
    try:
        report: JsonMap = {}
        if args.report is not None:
            report.update(_load_json(args.report))
        if args.dormant_revival is not None:
            report["dormant_revival"] = dormant_revival_report_to_section(
                _load_json(args.dormant_revival)
            )
        if not report:
            parser.error("provide --report and/or --dormant-revival")
        decisions = evaluate_phase3_report(
            report,
            manifest=_load_json(args.manifest),
            ledger=_load_json(args.ledger),
        )
    except ValueError as exc:
        # Malformed evidence is a fail-closed reject record, never a bare crash.
        decisions = [_error_decision([f"phase3 report rejected: {exc}"])]
        errored = True
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as fh:
            json.dump(
                [asdict(decision) for decision in decisions],
                fh,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            fh.write("\n")
        output_path = args.out
    else:
        output_path = Phase3DecisionStore(args.store_dir).write_many(decisions)
    print(f"phase3-governance record={output_path}")
    for decision in decisions:
        print(f"  - {decision.surface}: {decision.action}")
        for reason in decision.reasons:
            print(f"      {reason}")
    if errored:
        return 1
    if args.check:
        blocked = {
            Phase3Action.REJECT.value,
            Phase3Action.ROLLBACK.value,
        }
        if any(decision.action in blocked for decision in decisions):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
