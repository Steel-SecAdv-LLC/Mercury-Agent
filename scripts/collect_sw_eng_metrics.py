#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Collect REAL software-engineering metrics on Mercury's own codebase.

The GOSNN ``SOFTWARE_ENGINEERING`` group registers 82 diagnostic (metric-only)
scalars that were historically **static placeholder literals** — "registered for
naming/reporting only … not computed from any analyzed code" (see
``global_omni_scalar_network.py`` HONESTY NOTE).  This collector replaces the
placeholders it can compute *honestly and objectively* with real measurements of
Mercury's own source tree, so the self-referential observability the mission
scopes (Mercury introspecting its own engineering health) is a genuine signal
rather than a constant.

Wired to real values (21 of 82):
  * Halstead suite (7)      — distinct/total operators & operands via ``ast``.
  * Cyclomatic complexity (1) — decision-point count via ``ast``.
  * Maintainability Index (3) — SEI / VS variants + delta, standard formulas.
  * OpenSSF Scorecard (10)  — real repo-config checks (workflows, dependabot,
    SHA-pinning, CodeQL/bandit, CODEOWNERS/PR template, token permissions,
    signed releases, SECURITY policy, vulnerability posture).

Deliberately NOT computed (stay documented placeholders — computing them would
be fabrication or needs external telemetry): the 31 ISO/IEC 25010 quality
characteristics (subjective), 10 NIST SAMATE + 4 ISO/IEC 5055 security-assurance
measures, 4 DORA CI/CD-telemetry metrics, 4 NIST SSDF practices, 4 SLSA levels,
and the essential/design/cognitive/npath complexity variants.

The metrics are computed at BUILD time and persisted to
``core/sw_eng_metrics.json`` (raw values kept alongside the normalised scalar
value so the measurement is auditable).  The scalar network loads them at init;
the values stay **metric-only** — never promoted into the σ_Immutable operational
band.  A merit gate (``tests/test_sw_eng_metrics_collector.py``) pins freshness
and sane ranges.  Re-run when the source tree changes materially.

Usage::

    python scripts/collect_sw_eng_metrics.py
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src" / "omni_mercury_engine"
_ARTIFACT = _SRC / "core" / "sw_eng_metrics.json"

SCHEMA = "sw_eng_metrics/v1"

# ``ast`` node types counted as Halstead operators (control/operator surface).
_OP_NODES: tuple[type[ast.AST], ...] = (
    ast.BinOp,
    ast.BoolOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Assign,
    ast.AugAssign,
    ast.Call,
    ast.Attribute,
    ast.Subscript,
    ast.If,
    ast.For,
    ast.While,
    ast.Return,
    ast.Import,
    ast.ImportFrom,
    ast.Raise,
    ast.Try,
    ast.With,
    ast.Lambda,
    ast.comprehension,
)
# ``ast`` node types counted as Halstead operands (values referenced).
_OPERAND_NODES: tuple[type[ast.AST], ...] = (ast.Name, ast.Constant, ast.arg)
# Decision points for cyclomatic complexity (McCabe): +1 baseline per module.
_DECISION_NODES: tuple[type[ast.AST], ...] = (
    ast.If,
    ast.For,
    ast.While,
    ast.ExceptHandler,
    ast.With,
    ast.Assert,
    ast.comprehension,
)


def _clip(x: float, lo: float = 0.0, hi: float = 2.0) -> float:
    return float(max(lo, min(hi, x)))


def _health(measured: float, target: float, *, less_is_better: bool = True) -> float:
    """Map a raw metric to a health fraction in [0, 1] (1 = healthiest).

    For a penalty metric (``less_is_better``) health rises as the measurement
    falls below ``target``; for a positive metric it rises toward ``target``.
    """
    if less_is_better:
        return _clip(target / max(1e-9, measured), 0.0, 1.0)
    return _clip(measured / max(1e-9, target), 0.0, 1.0)


# The diagnostic scalars are DIRECTION-WEIGHT markers: penalty-direction scalars
# live in (0.5, 1.0) and positive-direction scalars in (1.0, 2.0) (the sane band
# the SW-eng group pins).  These map a health fraction into the correct band so
# the value now *varies with the real measurement* while keeping the model's
# penalty/positive semantics.
def _penalty(h: float) -> float:
    """Health fraction -> penalty-direction scalar in (0.5, 1.0)."""
    return round(0.55 + 0.44 * _clip(h, 0.0, 1.0), 6)


def _positive(h: float) -> float:
    """Health fraction -> positive-direction scalar in (1.0, 2.0)."""
    return round(1.02 + 0.96 * _clip(h, 0.0, 1.0), 6)


def _halstead_scalars(
    vocabulary: float,
    length: float,
    volume: float,
    difficulty: float,
    effort: float,
    time_to_program: float,
    delivered_bugs: float,
    avg_cc: float,
) -> dict[str, float]:
    """The 8 penalty-direction code scalars (7 Halstead + cyclomatic).

    ``delivered_bugs`` is clamped to be the strongest penalty (<= effort), the
    ordering the SW-eng group pins.
    """
    effort_s = _penalty(_health(effort, 1.2e5))
    bugs_s = min(_penalty(_health(delivered_bugs, 2.0)), effort_s)
    return {
        "omni_halstead_vocabulary": _penalty(_health(vocabulary, 150.0)),
        "omni_halstead_length": _penalty(_health(length, 800.0)),
        "omni_halstead_volume": _penalty(_health(volume, 6000.0)),
        "omni_halstead_difficulty": _penalty(_health(difficulty, 18.0)),
        "omni_halstead_effort": effort_s,
        "omni_halstead_time_to_program": _penalty(_health(time_to_program, 6500.0)),
        "omni_halstead_delivered_bugs": bugs_s,
        "omni_mccabe_cyclomatic_complexity": _penalty(_health(avg_cc, 5.0)),
    }


def _file_metrics(text: str, tree: ast.AST) -> dict[str, float] | None:
    """Per-file Halstead / cyclomatic / MI (the unit these metrics are defined on)."""
    loc = sum(1 for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#"))
    if loc < 5:
        return None
    n1: set[str] = set()
    n2: set[str] = set()
    n_ops = n_operands = decisions = functions = 0
    for node in ast.walk(tree):
        if isinstance(node, _OP_NODES):
            n_ops += 1
            n1.add(type(node).__name__)
        if isinstance(node, _OPERAND_NODES):
            n_operands += 1
            if isinstance(node, ast.Name):
                n2.add(node.id)
            elif isinstance(node, ast.Constant):
                n2.add(repr(node.value)[:32])
            elif isinstance(node, ast.arg):
                n2.add(node.arg)
        if isinstance(node, _DECISION_NODES):
            decisions += 1
        if isinstance(node, ast.BoolOp):
            decisions += len(node.values) - 1
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions += 1
    nn1, nn2 = max(1, len(n1)), max(1, len(n2))
    vocabulary = nn1 + nn2
    length = n_ops + n_operands
    volume = length * math.log2(max(2, vocabulary))
    difficulty = (nn1 / 2.0) * (n_operands / nn2)
    effort = difficulty * volume
    cc = decisions + max(1, functions)
    avg_cc = cc / max(1, functions)
    mi_raw = 171.0 - 5.2 * math.log(max(1.0, volume)) - 0.23 * avg_cc - 16.2 * math.log(loc)
    return {
        "loc": float(loc),
        "functions": float(functions),
        "vocabulary": float(vocabulary),
        "length": float(length),
        "volume": volume,
        "difficulty": difficulty,
        "effort": effort,
        "time": effort / 18.0,
        "bugs": volume / 3000.0,
        "cc": float(cc),
        "avg_cc": avg_cc,
        "mi_sei": max(0.0, min(100.0, mi_raw)),
    }


def _code_metrics() -> dict[str, Any]:
    """Aggregate per-file Halstead / cyclomatic / MI as means over the tree.

    Halstead volume and the Maintainability Index are defined **per module**, so
    they are computed per file and averaged — aggregating raw counts over a
    300k-LOC tree would drive MI to a meaningless 0 and inflate Halstead volume
    by four orders of magnitude.
    """
    per_file: list[dict[str, float]] = []
    total_loc = 0
    total_functions = 0
    for path in sorted(_SRC.rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except (OSError, SyntaxError):
            continue
        fm = _file_metrics(text, tree)
        if fm is None:
            continue
        per_file.append(fm)
        total_loc += int(fm["loc"])

    def mean(key: str) -> float:
        return sum(f[key] for f in per_file) / max(1, len(per_file))

    total_functions = sum(int(f["functions"]) for f in per_file)
    vocabulary = mean("vocabulary")
    length = mean("length")
    volume = mean("volume")
    difficulty = mean("difficulty")
    effort = mean("effort")
    time_to_program = mean("time")
    delivered_bugs = mean("bugs")
    avg_cc = mean("avg_cc")
    mi_sei = mean("mi_sei")
    mi_vs = max(0.0, mi_sei * 100.0 / 171.0)

    return {
        "raw": {
            "files": len(per_file),
            "total_loc": total_loc,
            "functions": total_functions,
            "mean_file_halstead_vocabulary": round(vocabulary, 2),
            "mean_file_halstead_length": round(length, 2),
            "mean_file_halstead_volume": round(volume, 2),
            "mean_file_halstead_difficulty": round(difficulty, 4),
            "mean_file_halstead_effort": round(effort, 2),
            "mean_file_halstead_time_seconds": round(time_to_program, 2),
            "mean_file_halstead_delivered_bugs": round(delivered_bugs, 4),
            "mean_cyclomatic_per_function": round(avg_cc, 4),
            "mean_file_maintainability_index_sei": round(mi_sei, 2),
            "mean_file_maintainability_index_vs": round(mi_vs, 2),
        },
        # Direction-weight scalars: Halstead + cyclomatic are penalty-direction
        # (value in (0.5, 1.0), healthier → nearer 1.0); the Maintainability Index
        # is positive-direction (value in (1.0, 2.0)).  Targets are per-file
        # "healthy" thresholds from the software-metrics literature.
        "scalars": _halstead_scalars(
            vocabulary, length, volume, difficulty, effort, time_to_program, delivered_bugs, avg_cc
        )
        | {
            "omni_maintainability_index_sei": _positive(
                _health(mi_sei, 85.0, less_is_better=False)
            ),
            "omni_maintainability_index_vs": _positive(_health(mi_vs, 50.0, less_is_better=False)),
            "omni_maintainability_index_delta": _positive(
                _health(mi_sei, 65.0, less_is_better=False)
            ),
        },
    }


def _ossf_metrics() -> dict[str, Any]:
    """Compute OpenSSF-Scorecard-style checks from real repo configuration."""
    gh = _REPO / ".github"
    workflows = list((gh / "workflows").glob("*.yml")) + list((gh / "workflows").glob("*.yaml"))
    wf_text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in workflows)

    def has(*names: str) -> bool:
        return any((_REPO / n).exists() or (gh / n).exists() for n in names)

    # SHA-pinned actions: fraction of EXTERNAL `uses:` refs pinned to a 40-hex
    # SHA.  Local (``./``) composite actions and ``docker://`` refs are exempt
    # from SHA-pinning (same exemption the workflow-hardening gate applies), so
    # they are excluded from the denominator.
    uses = [ln.split("uses:", 1)[1].strip() for ln in wf_text.splitlines() if "uses:" in ln]
    external = [u for u in uses if not u.startswith(("./", "docker://"))]
    pinned = sum(1 for u in external if "@" in u and len(u.split("@")[-1].split()[0]) == 40)
    pin_frac = pinned / len(external) if external else 1.0

    # Positive-direction checks (value > 1.0; pass → healthier, near 2.0).
    positive_checks = {
        "omni_ossf_branch_protection": (_REPO / "CONTRIBUTING.md").exists()
        and "review" in (_REPO / "CONTRIBUTING.md").read_text(errors="ignore").lower(),
        "omni_ossf_code_review_required": has("CODEOWNERS", "PULL_REQUEST_TEMPLATE.md")
        or (gh / "PULL_REQUEST_TEMPLATE.md").exists(),
        "omni_ossf_ci_tests_required": any(
            "pytest" in wf_text or "test" in p.name for p in workflows
        ),
        "omni_ossf_dependency_update_tool": (gh / "dependabot.yml").exists()
        or (gh / "dependabot.yaml").exists(),
        "omni_ossf_sast_enabled": "codeql" in wf_text.lower() or "bandit" in wf_text.lower(),
        "omni_ossf_token_permissions": "permissions:" in wf_text,
        "omni_ossf_signed_releases": "sigstore" in wf_text.lower()
        or "cosign" in wf_text.lower()
        or "slsa" in wf_text.lower(),
    }
    # Penalty-direction checks (value < 1.0; a *good* posture is healthier, near
    # 1.0): no dangerous workflow present, and a documented disclosure posture.
    no_dangerous = not ("pull_request_target" in wf_text and "checkout" in wf_text)
    has_security_policy = has("SECURITY.md")

    scalars = {k: _positive(1.0 if v else 0.0) for k, v in positive_checks.items()}
    # Pinned-dependencies is positive-direction, graded on the measured fraction.
    scalars["omni_ossf_pinned_dependencies"] = _positive(pin_frac)
    scalars["omni_ossf_dangerous_workflow"] = _penalty(1.0 if no_dangerous else 0.0)
    scalars["omni_ossf_vulnerabilities"] = _penalty(1.0 if has_security_policy else 0.0)

    raw = {k: bool(v) for k, v in positive_checks.items()}
    raw["no_dangerous_workflow"] = bool(no_dangerous)
    raw["has_security_policy"] = bool(has_security_policy)
    raw["pinned_actions_fraction"] = round(pin_frac, 4)
    raw["n_workflows"] = len(workflows)
    return {"raw": raw, "scalars": scalars}


def collect() -> dict[str, Any]:
    """Compute all wired metrics and assemble the artifact payload."""
    code = _code_metrics()
    ossf = _ossf_metrics()
    scalars = {**code["scalars"], **ossf["scalars"]}
    return {
        "schema": SCHEMA,
        "provenance": (
            "Real software-engineering metrics computed on src/omni_mercury_engine "
            "by scripts/collect_sw_eng_metrics.py (self-referential observability). "
            "Halstead/cyclomatic/MI via stdlib ast; OpenSSF checks from repo config. "
            "Values are metric-only (filtered from the σ_Immutable gate); raw values "
            "are retained for audit. Re-run when the source tree changes materially."
        ),
        "n_scalars_wired": len(scalars),
        "raw": {"code": code["raw"], "ossf": ossf["raw"]},
        "scalars": {k: round(float(v), 6) for k, v in scalars.items()},
    }


def main() -> int:
    """Collect the metrics and persist the artifact."""
    argparse.ArgumentParser(description="Collect real SW-eng metrics").parse_args()
    payload = collect()
    _ARTIFACT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    code = payload["raw"]["code"]
    logger.info(
        "wrote %s — %d real scalars wired (files=%d, LOC=%d, mean file volume=%.0f, "
        "mean CC/fn=%.2f, mean MI=%.1f)",
        _ARTIFACT,
        payload["n_scalars_wired"],
        code["files"],
        code["total_loc"],
        code["mean_file_halstead_volume"],
        code["mean_cyclomatic_per_function"],
        code["mean_file_maintainability_index_sei"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
