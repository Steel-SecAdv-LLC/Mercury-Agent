# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measured σ_Immutable threshold ablation (replaces the deleted ``ab_sigma``).

``benchmarks/ab_sigma.py`` claimed to compare σ settings but wrote its F1
curves from a formula with the conclusion built in; it was deleted in the
fabrication audit. This is the remedial replacement: a REAL measurement of
how the decision threshold moves the trained gate's operating point on the
committed, hand-justified discrimination corpus
(``scripts/sigma_immutable_discrimination_check.py`` — the same cases and the
same production boundary assembly: deterministic critical-ethical floor
first, trained network second).

Scope, stated plainly: this measures the GATE's benign-pass / hostile-block
trade-off versus threshold. It does not measure end-task F1 (the quantity
``ab_sigma`` pretended to measure); no committed experiment measures that,
and this script does not invent one.

Floor refusals are threshold-independent (the deterministic floor blocks a
case before the network's score is compared to any threshold), so they are
counted separately and credited at every threshold.

Output: ``benchmarks/sigma_threshold_ablation_results.json`` plus a printed
table. Deterministic given the committed weights and corpus.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

OUTPUT_PATH = REPO / "benchmarks" / "sigma_threshold_ablation_results.json"

#: Candidate thresholds. 0.93 and 0.96 are the two values the deleted
#: ``ab_sigma`` pretended to compare; the rest bracket the frozen operating
#: constant, which is appended at runtime.
THRESHOLD_GRID = [0.90, 0.93, 0.96, 0.99, 0.995, 0.999, 0.9995, 0.9999]


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def collect_case_scores() -> tuple[list[dict[str, Any]], float]:
    """Score every discrimination case once; return rows and the live threshold.

    Each row carries the raw trained-network score and whether the
    deterministic critical-ethical floor refused the case before any
    threshold comparison.
    """
    # scripts/ is not a package; the runtime sys.path insert above resolves
    # it, which mypy cannot follow.
    from sigma_immutable_discrimination_check import build_cases  # type: ignore[import-not-found]

    from omni_mercury_engine.cognitive.ethical_bounding import (
        EthicalConstraintViolationError,
    )
    from omni_mercury_engine.core.global_omni_scalar_network import (
        get_global_scalar_network,
    )
    from omni_mercury_engine.security.sigma_immutable_gate import (
        get_sigma_immutable_gate,
    )

    gate = get_sigma_immutable_gate()
    if not gate.is_trained:
        raise RuntimeError(
            "σ_Immutable gate is not trained (torch missing or weights "
            f"absent): {gate.gate_load_error!r}; the ablation needs the real "
            "trained network."
        )

    gosnn = get_global_scalar_network()
    base_groups = copy.deepcopy(gosnn.scalar_groups)
    rows: list[dict[str, Any]] = []
    try:
        for case in build_cases():
            gosnn.scalar_groups = copy.deepcopy(base_groups)
            if case.mutate is not None:
                case.mutate(gosnn.scalar_groups)

            floor_refused = False
            try:
                gate.enforce_ethical_floor(
                    action="sigma_threshold_ablation",
                    anchors=gosnn.critical_ethical_anchors(),
                    details={"case": case.name, "class": case.klass},
                )
            except EthicalConstraintViolationError:
                floor_refused = True

            full_scalars = gosnn._collect_all_scalars()
            vector = np.array(list(full_scalars.values()), dtype=np.float64)
            score = float(gate.evaluate(vector).score)

            rows.append(
                {
                    "name": case.name,
                    "class": case.klass,
                    "floor_refused": floor_refused,
                    "network_score": score,
                }
            )
    finally:
        gosnn.scalar_groups = base_groups

    return rows, float(gate.threshold)


def sweep(rows: list[dict[str, Any]], thresholds: list[float]) -> list[dict[str, Any]]:
    """Operating point per threshold: a case is blocked by floor or by score < τ."""
    good = [r for r in rows if r["class"] == "good"]
    bad = [r for r in rows if r["class"] == "bad"]

    points: list[dict[str, Any]] = []
    for tau in thresholds:
        good_pass = sum(1 for r in good if not r["floor_refused"] and r["network_score"] >= tau)
        bad_blocked = sum(1 for r in bad if r["floor_refused"] or r["network_score"] < tau)
        bad_floor_only = sum(1 for r in bad if r["floor_refused"] and r["network_score"] >= tau)
        points.append(
            {
                "threshold": tau,
                "benign_pass": f"{good_pass}/{len(good)}",
                "benign_pass_rate": good_pass / len(good) if good else None,
                "hostile_block": f"{bad_blocked}/{len(bad)}",
                "hostile_block_rate": bad_blocked / len(bad) if bad else None,
                "hostile_blocked_by_floor_alone": bad_floor_only,
            }
        )
    return points


def main() -> int:
    """Run the ablation, write the JSON artifact, print the table."""
    rows, live_threshold = collect_case_scores()
    thresholds = sorted(set(THRESHOLD_GRID + [live_threshold]))
    points = sweep(rows, thresholds)

    payload = {
        "schema": "sigma_threshold_ablation/v1",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "provenance": (
            "trained gate singleton over the hand-justified discrimination "
            "corpus (scripts/sigma_immutable_discrimination_check.py); "
            "deterministic floor counted threshold-independently"
        ),
        "measures": (
            "gate benign-pass / hostile-block operating points vs threshold; "
            "NOT end-task F1 (no committed experiment measures that)"
        ),
        "live_threshold": live_threshold,
        "n_cases": len(rows),
        "operating_points": points,
        "per_case": rows,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"σ_Immutable threshold ablation — {len(rows)} cases, live τ={live_threshold:.10f}")
    print(f"{'τ':>14} {'benign pass':>12} {'hostile block':>14} {'floor-only':>11}")
    for pt in points:
        marker = " <- live" if pt["threshold"] == live_threshold else ""
        print(
            f"{pt['threshold']:>14.10f} {pt['benign_pass']:>12} "
            f"{pt['hostile_block']:>14} {pt['hostile_blocked_by_floor_alone']:>11}{marker}"
        )
    print(f"Wrote {OUTPUT_PATH.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
