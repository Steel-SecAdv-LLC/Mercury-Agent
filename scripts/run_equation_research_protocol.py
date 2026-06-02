#!/usr/bin/env python3
"""Run the Mercury equation research protocol against benchmark artifacts.

The protocol operationalizes eight governance requirements:
1. Freeze the mathematical baseline via immutable reference surfaces.
2. Track unknown-discovery, humanitarian, and security outcomes.
3. Enforce a formal hypothesis matrix before optimization promotion.
4. Evaluate ablation/sweep deltas against measurable bars.
5. Gate on uncertainty, calibration, and boundary-stress checks.
6. Require humanitarian-first and security-first lane consistency.
7. Promote advances only when all hard criteria pass.
8. Publish promoted advances as theorem-to-test artifacts.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Comparison:
    field: str
    op: str
    # Config-driven threshold: scalar for ordering/equality ops, ``list`` for
    # ``in``/``not_in``, or ``None`` when a gate omits a value.
    value: float | int | bool | str | list[Any] | None


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required: pip install pyyaml>=6.0") from exc
    data = yaml.safe_load(path.read_text())
    return data if isinstance(data, dict) else {}


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    return data if isinstance(data, dict) else {}


def _resolve_path(payload: dict[str, Any], dotted: str) -> Any:
    cur: Any = payload
    for token in dotted.split("."):
        if isinstance(cur, dict) and token in cur:
            cur = cur[token]
            continue
        return None
    return cur


def _as_float(value: Any) -> float | None:
    try:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        return float(value)
    except (TypeError, ValueError):
        return None


def _compare(actual: Any, comparison: Comparison) -> bool:
    expected = comparison.value
    op = comparison.op

    if op in {">", ">=", "<", "<="}:
        actual_num = _as_float(actual)
        expected_num = _as_float(expected)
        if actual_num is None or expected_num is None:
            return False
        if op == ">":
            return actual_num > expected_num
        if op == ">=":
            return actual_num >= expected_num
        if op == "<":
            return actual_num < expected_num
        return actual_num <= expected_num

    if op == "==":
        return bool(actual == expected)
    if op == "!=":
        return bool(actual != expected)
    if op == "in":
        return isinstance(expected, list) and actual in expected
    if op == "not_in":
        return isinstance(expected, list) and actual not in expected
    return False


def _validate_immutable_surfaces(
    root: Path, surfaces: list[dict[str, Any]]
) -> tuple[bool, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    all_ok = True
    for surface in surfaces:
        name = str(surface.get("name", "unnamed"))
        rel = str(surface.get("path", "")).strip()
        exists = bool(rel) and (root / rel).exists()
        all_ok = all_ok and exists
        checks.append({"name": name, "path": rel, "exists": exists})
    return all_ok, checks


def _track_metric(domain_summary: dict[str, Any], domains: list[str], metric: str) -> float | None:
    total_weight = 0
    weighted = 0.0
    for domain in domains:
        domain_payload = domain_summary.get(domain)
        if not isinstance(domain_payload, dict):
            continue
        stats = domain_payload.get("stats")
        if not isinstance(stats, dict):
            continue
        metric_value = _as_float(stats.get(metric))
        weight = int(domain_payload.get("n_measured", 0))
        if metric_value is None or weight <= 0:
            continue
        weighted += metric_value * weight
        total_weight += weight
    if total_weight == 0:
        return None
    return weighted / total_weight


def _evaluate_tracks(
    benchmark: dict[str, Any], track_cfg: list[dict[str, Any]]
) -> tuple[bool, dict[str, Any]]:
    domain_summary = benchmark.get("domain_summary")
    if not isinstance(domain_summary, dict):
        return False, {"error": "benchmark domain_summary missing"}

    out: dict[str, Any] = {}
    all_ok = True
    for track in track_cfg:
        track_id = str(track.get("id", "track"))
        domains = track.get("domains", [])
        targets = track.get("targets", {})
        if not isinstance(domains, list) or not isinstance(targets, dict):
            out[track_id] = {"passed": False, "reason": "invalid config"}
            all_ok = False
            continue

        track_pass = True
        metrics: dict[str, Any] = {}
        for metric, min_value in targets.items():
            measured = _track_metric(domain_summary, [str(d) for d in domains], str(metric))
            passed = measured is not None and float(measured) >= float(min_value)
            metrics[str(metric)] = {
                "measured": measured,
                "min_required": float(min_value),
                "passed": passed,
            }
            track_pass = track_pass and passed

        out[track_id] = {
            "name": track.get("name", track_id),
            "domains": domains,
            "metrics": metrics,
            "passed": track_pass,
        }
        all_ok = all_ok and track_pass
    return all_ok, out


def _evaluate_matrix(
    source: dict[str, Any], matrix_cfg: list[dict[str, Any]]
) -> tuple[bool, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    all_ok = True
    for row in matrix_cfg:
        component = str(row.get("component", "unknown"))
        hypothesis = str(row.get("hypothesis", ""))
        checks_cfg = row.get("acceptance", [])
        checks: list[dict[str, Any]] = []
        row_ok = True
        if not isinstance(checks_cfg, list):
            checks_cfg = []
        for check in checks_cfg:
            comp = Comparison(
                field=str(check.get("field", "")),
                op=str(check.get("op", "==")),
                value=check.get("value"),
            )
            actual = _resolve_path(source, comp.field)
            passed = _compare(actual, comp)
            checks.append(
                {
                    "field": comp.field,
                    "op": comp.op,
                    "expected": comp.value,
                    "actual": actual,
                    "passed": passed,
                }
            )
            row_ok = row_ok and passed

        rows.append(
            {
                "component": component,
                "hypothesis": hypothesis,
                "passed": row_ok,
                "checks": checks,
                "fail_condition": row.get("fail_condition", ""),
            }
        )
        all_ok = all_ok and row_ok
    return all_ok, rows


def _evaluate_gate_block(
    source: dict[str, Any], gate_cfg: list[dict[str, Any]]
) -> tuple[bool, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    all_ok = True
    for gate in gate_cfg:
        comp = Comparison(
            field=str(gate.get("field", "")),
            op=str(gate.get("op", "==")),
            value=gate.get("value"),
        )
        actual = _resolve_path(source, comp.field)
        passed = _compare(actual, comp)
        rows.append(
            {
                "name": gate.get("name", comp.field),
                "field": comp.field,
                "op": comp.op,
                "expected": comp.value,
                "actual": actual,
                "passed": passed,
            }
        )
        all_ok = all_ok and passed
    return all_ok, rows


def _evaluate_lane_consistency(
    gate_source: dict[str, Any], lane_cfg: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    humanitarian = _resolve_path(gate_source, str(lane_cfg.get("humanitarian_lane", "")))
    security = _resolve_path(gate_source, str(lane_cfg.get("security_lane", "")))
    metrics = lane_cfg.get("consistency_metrics", {})

    if (
        not isinstance(humanitarian, dict)
        or not isinstance(security, dict)
        or not isinstance(metrics, dict)
    ):
        return False, {"passed": False, "reason": "lane payload missing"}

    checks: dict[str, Any] = {}
    all_ok = True
    for metric, tolerance in metrics.items():
        h_val = _as_float(humanitarian.get(metric))
        s_val = _as_float(security.get(metric))
        if h_val is None or s_val is None:
            passed = False
            delta = None
        else:
            delta = abs(h_val - s_val)
            passed = delta <= float(tolerance)
        checks[metric] = {
            "humanitarian": h_val,
            "security": s_val,
            "abs_delta": delta,
            "max_delta": float(tolerance),
            "passed": passed,
        }
        all_ok = all_ok and passed

    return all_ok, {"passed": all_ok, "checks": checks}


def _validate_ai_equation_library(library_cfg: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    required_raw = library_cfg.get("required_fields", ["id", "equation", "purpose", "source"])
    required = [str(field) for field in required_raw] if isinstance(required_raw, list) else []
    sections = {
        "known_reference_equations": library_cfg.get("known_reference_equations", []),
        "in_house_equations": library_cfg.get("in_house_equations", []),
    }

    section_results: dict[str, Any] = {}
    all_ok = True
    for section, equations in sections.items():
        rows = equations if isinstance(equations, list) else []
        missing: list[dict[str, Any]] = []
        for idx, item in enumerate(rows):
            if not isinstance(item, dict):
                missing.append({"index": idx, "missing_fields": required})
                continue
            absent = [field for field in required if item.get(field) in (None, "")]
            if absent:
                missing.append({"id": item.get("id", f"equation_{idx}"), "missing_fields": absent})

        section_ok = len(rows) > 0 and not missing
        section_results[section] = {
            "passed": section_ok,
            "count": len(rows),
            "missing": missing,
            "items": rows,
        }
        all_ok = all_ok and section_ok

    return all_ok, {"passed": all_ok, "required_fields": required, "sections": section_results}


def _publish_theorem_artifacts(
    out_dir: Path | None,
    promoted: list[dict[str, Any]],
    required_fields: list[str],
) -> list[str]:
    if out_dir is None:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    for idx, item in enumerate(promoted, start=1):
        slug = str(item.get("id") or f"advance_{idx}")
        artifact = {key: item.get(key) for key in required_fields}
        artifact["id"] = slug
        path = out_dir / f"{slug}.json"
        path.write_text(json.dumps(artifact, indent=2, sort_keys=True))
        created.append(str(path))
    return created


def _run_commands(commands: list[str], cwd: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for command in commands:
        argv = shlex.split(command)
        proc = subprocess.run(  # noqa: S603 - commands are repository-authored protocol entries
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
        runs.append(
            {
                "command": command,
                "rc": int(proc.returncode),
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        )
    return runs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate equation research protocol artifacts.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/equation_research_protocol.yaml"),
        help="Protocol YAML config path.",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("benchmarks/mercury_benchmark_results.json"),
        help="Benchmark JSON used for track evaluation.",
    )
    parser.add_argument(
        "--ablation",
        type=Path,
        default=Path("artifacts/equation_protocol_ablation.json"),
        help="Ablation/sweep metrics JSON used by the hypothesis matrix.",
    )
    parser.add_argument(
        "--gates",
        type=Path,
        default=Path("artifacts/equation_protocol_gates.json"),
        help="Gate and lane metrics JSON (uncertainty/calibration/stress).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/equation_research_protocol_result.json"),
        help="Where to write the protocol report JSON.",
    )
    parser.add_argument(
        "--theorem-out",
        type=Path,
        default=None,
        help="Optional directory for promoted theorem-to-test JSON artifacts.",
    )
    parser.add_argument(
        "--execute-plan",
        action="store_true",
        help="Execute optimization commands listed in the protocol config.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]

    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 2
    if not args.benchmark.exists():
        print(f"ERROR: benchmark not found: {args.benchmark}", file=sys.stderr)
        return 2

    protocol = _load_yaml(args.config)
    benchmark = _load_json(args.benchmark)
    ablation = _load_json(args.ablation if args.ablation.exists() else None)
    gates = _load_json(args.gates if args.gates.exists() else None)

    control_cfg = protocol.get("control_model", {})
    surfaces_cfg = control_cfg.get("immutable_reference_surfaces", [])
    surfaces_ok, surface_checks = _validate_immutable_surfaces(
        repo_root,
        surfaces_cfg if isinstance(surfaces_cfg, list) else [],
    )

    track_cfg = protocol.get("outcome_tracks", [])
    tracks_ok, track_results = _evaluate_tracks(
        benchmark,
        track_cfg if isinstance(track_cfg, list) else [],
    )

    matrix_cfg = protocol.get("hypothesis_matrix", [])
    matrix_ok, matrix_rows = _evaluate_matrix(
        ablation,
        matrix_cfg if isinstance(matrix_cfg, list) else [],
    )

    first_class_cfg = protocol.get("first_class_gates", [])
    first_class_ok, first_class_rows = _evaluate_gate_block(
        gates,
        first_class_cfg if isinstance(first_class_cfg, list) else [],
    )

    lane_cfg = protocol.get("lane_consistency", {})
    lane_ok, lane_result = _evaluate_lane_consistency(
        gates,
        lane_cfg if isinstance(lane_cfg, dict) else {},
    )

    ai_library_cfg = protocol.get("ai_equation_library", {})
    ai_library_ok, ai_library_result = _validate_ai_equation_library(
        ai_library_cfg if isinstance(ai_library_cfg, dict) else {}
    )

    decision_justification = protocol.get("decision_justification", [])
    decision_rows = decision_justification if isinstance(decision_justification, list) else []

    command_cfg = protocol.get("optimization_plan", {})
    commands = command_cfg.get("commands", []) if isinstance(command_cfg, dict) else []
    command_runs: list[dict[str, Any]] = []
    if args.execute_plan and isinstance(commands, list):
        command_runs = _run_commands([str(c) for c in commands], cwd=repo_root)

    advances = ablation.get("advances", []) if isinstance(ablation.get("advances"), list) else []
    promoted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in advances:
        if not isinstance(item, dict):
            continue
        empirical = bool(item.get("empirical_gain"))
        ethical = bool(item.get("ethical_gates_preserved", first_class_ok))
        stability = bool(item.get("stability_preserved", matrix_ok))
        originality = bool(item.get("originality_preserved"))
        passed = empirical and ethical and stability and originality and lane_ok and tracks_ok
        verdict = {
            "id": item.get("id", "advance"),
            "passed": passed,
            "empirical_gain": empirical,
            "ethical_gates_preserved": ethical,
            "stability_preserved": stability,
            "originality_preserved": originality,
        }
        if passed:
            promoted.append(item)
        else:
            rejected.append(verdict)

    theorem_cfg = protocol.get("theorem_artifact", {})
    required_fields = (
        theorem_cfg.get("required_fields", []) if isinstance(theorem_cfg, dict) else []
    )
    theorem_required = [str(field) for field in required_fields if field]

    missing_fields: list[dict[str, Any]] = []
    for item in promoted:
        missing = [field for field in theorem_required if item.get(field) in (None, "")]
        if missing:
            missing_fields.append({"id": item.get("id", "advance"), "missing_fields": missing})

    theorem_artifacts: list[str] = []
    promoted_clean = [
        item for item in promoted if not any(m["id"] == item.get("id") for m in missing_fields)
    ]
    theorem_artifacts = _publish_theorem_artifacts(
        args.theorem_out, promoted_clean, theorem_required
    )

    protocol_passed = (
        surfaces_ok
        and tracks_ok
        and matrix_ok
        and first_class_ok
        and lane_ok
        and ai_library_ok
        and not missing_fields
        and len(promoted_clean) > 0
    )

    report = {
        "config": str(args.config),
        "benchmark": str(args.benchmark),
        "ablation": str(args.ablation),
        "gates": str(args.gates),
        "protocol_passed": protocol_passed,
        "control_model": {
            "frozen": surfaces_ok,
            "surfaces": surface_checks,
        },
        "outcome_tracks": {
            "passed": tracks_ok,
            "tracks": track_results,
        },
        "hypothesis_matrix": {
            "passed": matrix_ok,
            "rows": matrix_rows,
        },
        "first_class_gates": {
            "passed": first_class_ok,
            "checks": first_class_rows,
        },
        "lane_consistency": lane_result,
        "ai_equation_library": ai_library_result,
        "decision_justification": decision_rows,
        "optimization_plan": {
            "commands": commands,
            "executed": args.execute_plan,
            "runs": command_runs,
        },
        "promotion": {
            "promoted_count": len(promoted_clean),
            "rejected_count": len(rejected),
            "rejected": rejected,
            "missing_theorem_fields": missing_fields,
        },
        "theorem_artifacts": theorem_artifacts,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))

    return 0 if protocol_passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
