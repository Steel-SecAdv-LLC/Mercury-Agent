#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""CI entry point for the adversarial co-training red-team harness.

Runs :func:`omni_mercury_engine.intel.red_team.run_red_team` against the shipped
weapons/mass-casualty gate, appends surviving bypasses to ``corpus/pending``, and
emits a run summary. The gate requires the AMA/PQC backend, so run this in the
``ci/red-team`` lane (which builds AMA).

Everything is deterministic (fixed seeds + mutation registry), so the
surviving-bypass rate is a stable, pin-able number:

    PYTHONPATH=src python benchmarks/red_team_harness.py            # run + append + print
    PYTHONPATH=src python benchmarks/red_team_harness.py --update   # (re)pin the baseline
    PYTHONPATH=src python benchmarks/red_team_harness.py --check    # no-weakening gate (exit 1)
    PYTHONPATH=src python benchmarks/red_team_harness.py --no-append # run without writing pending

``--check`` fails when the survival rate rises above the pinned floor
(``benchmarks/red_team_baseline.json``, kept ``<=`` the value-metric baseline in
:data:`omni_mercury_engine.intel.value_metrics.VALUE_METRICS`), i.e. a change that
*weakens* the gate against obfuscation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from omni_mercury_engine.intel.red_team import (
    RedTeamConfig,
    append_survivors,
    run_red_team,
)
from omni_mercury_engine.intel.value_metrics import VALUE_METRICS

BASELINE_PATH = _REPO / "benchmarks" / "red_team_baseline.json"
ARTIFACT_PATH = _REPO / "artifacts" / "red_team" / "run_summary.json"
#: Float-comparison epsilon only. The survival rate is ``survivors/candidates`` --
#: a set-cardinality ratio that is fully deterministic and order-independent for a
#: fixed config + gate, so there is NO benign drift for a slack margin to absorb.
#: The gate therefore fails on any rise above the pinned floor (a real weakening),
#: and separately never permits a rate above the declared value-metric ceiling.
_FLOAT_EPS = 1e-9


def _run() -> tuple[Any, dict[str, Any]]:
    cfg = RedTeamConfig.load()
    result = run_red_team(cfg)
    summary = result.summary()
    return result, summary


def _write_artifact(summary: dict[str, Any], survivors: list[dict[str, Any]]) -> None:
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(
        json.dumps({"summary": summary, "survivors": survivors}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--update", action="store_true", help="(re)pin the survival-rate baseline")
    group.add_argument("--check", action="store_true", help="fail on a raised survival rate")
    parser.add_argument(
        "--no-append", action="store_true", help="do not write survivors to corpus/pending"
    )
    args = parser.parse_args(argv)

    result, summary = _run()
    survivor_rows = [c.as_pending_row() for c in result.survivors]

    appended = 0
    if not args.no_append:
        appended = append_survivors(result.survivors)
    summary["appended_to_pending"] = appended
    _write_artifact(summary, survivor_rows)

    declared = VALUE_METRICS["adversarial_co_training"].baseline
    rate = result.survival_rate

    if args.update:
        BASELINE_PATH.write_text(
            json.dumps(
                {
                    "survival_rate": round(rate, 6),
                    "n_candidates": summary["n_candidates"],
                    "n_survivors": summary["n_survivors"],
                    # Pinned so a seed the gate USED to block becoming ALLOW (which
                    # drops it from the denominator and can silently lower the
                    # survival rate) is caught as a seed-level weakening in --check.
                    "n_skipped_seeds": summary["n_skipped_seeds"],
                    "harness_version": summary["harness_version"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"pinned red-team baseline: survival_rate={rate:.4f} ({summary['n_survivors']} bypasses)"
        )
        return 0

    if args.check:
        if not BASELINE_PATH.is_file():
            print(f"missing baseline {BASELINE_PATH.name}; run --update to pin it", file=sys.stderr)
            return 1
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        floor = baseline["survival_rate"]
        problems = []
        # Compare at the precision the floor is pinned to (6 decimals, as --update
        # writes): the true rate is a full-precision ratio (e.g. 140/420 =
        # 0.333333...) while the pinned floor is rounded, so a raw ``>`` would trip
        # on the rounding delta, not a real weakening.
        rate_q = round(rate, 6)
        # The pinned floor must itself stay within the declared value-metric ceiling.
        if floor > declared + _FLOAT_EPS:
            print(
                f"pinned floor {floor:.4f} exceeds value-metric baseline {declared:.4f}; "
                "re-declare the value metric before pinning higher",
                file=sys.stderr,
            )
            return 1
        # No weakening: the rate may not rise above the pinned floor (deterministic,
        # so no slack beyond the pinning precision), and may never exceed the
        # declared value-metric ceiling.
        if rate_q > floor + _FLOAT_EPS:
            problems.append(
                f"survival rate {rate_q:.6f} rose above pinned floor {floor:.6f} "
                "(gate weakened against obfuscation)"
            )
        if rate_q > declared + _FLOAT_EPS:
            problems.append(
                f"survival rate {rate_q:.6f} exceeds declared value-metric ceiling {declared:.4f}"
            )
        # Seed-level weakening: a seed the gate previously blocked becoming ALLOW
        # is dropped from the denominator (n_skipped_seeds rises / n_candidates
        # falls) and can lower the survival rate while the gate is *more* broken.
        # Guard both counts so that failure mode cannot pass green.
        pinned_skipped = baseline.get("n_skipped_seeds")
        if pinned_skipped is not None and summary["n_skipped_seeds"] > pinned_skipped:
            problems.append(
                f"n_skipped_seeds rose {pinned_skipped} -> {summary['n_skipped_seeds']}: a seed "
                "the gate used to block is now ALLOWed (seed-level weakening)"
            )
        pinned_candidates = baseline.get("n_candidates")
        if pinned_candidates is not None and summary["n_candidates"] < pinned_candidates:
            problems.append(
                f"n_candidates fell {pinned_candidates} -> {summary['n_candidates']}: fewer "
                "attackable seeds (a blocked seed became ALLOW, shrinking the denominator)"
            )
        if problems:
            print("RED-TEAM REGRESSION:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 1
        print(
            f"OK: survival rate {rate:.4f} at/below pinned floor {floor:.4f}; "
            f"{summary['n_survivors']} bypass(es), {summary['n_skipped_seeds']} skipped seed(s), "
            f"{appended} newly appended"
        )
        return 0

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
