#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""CI entry point for the adversarial co-training red-team harness.

Runs :func:`omni_mercury_engine.intel.red_team.run_red_team` against the shipped
weapons/mass-casualty gate, appends surviving bypasses to ``corpus/pending``, and
emits a run summary. The gate requires the AMA/PQC backend, so run this in the
``ci/red-team`` lane (which builds AMA).

Everything is deterministic (fixed seeds + mutation registry), so both rates
below are stable, pin-able numbers:

    PYTHONPATH=src python benchmarks/red_team_harness.py            # run + append + print
    PYTHONPATH=src python benchmarks/red_team_harness.py --update   # (re)pin the baseline
    PYTHONPATH=src python benchmarks/red_team_harness.py --check    # no-weakening gate (exit 1)
    PYTHONPATH=src python benchmarks/red_team_harness.py --no-append # run without writing pending

**What ``--check`` gates on, and why it changed.** The gating quantity is
:func:`~omni_mercury_engine.intel.red_team.measure_fixed_universe_bypass` --
the bypass rate over a candidate universe fixed by the *config*, which is the
metric ``VALUE_METRICS["adversarial_co_training"]`` has declared since
2026-08-04 (``fixed_universe_gate_bypass_rate``).

It previously gated on :attr:`RedTeamResult.survival_rate`, which is not a
sound no-weakening guard: ``run_red_team`` skips a seed the gate does not
block, so the denominator *shrinks as the gate weakens and grows as it
strengthens*, and a strictly stronger gate can fail the floor. That is not
hypothetical -- it is recorded in ``red_team.measure_fixed_universe_bypass``'s
own docstring. The two were also being compared across the type boundary: the
declared ceiling had already moved to the fixed-universe metric while the
harness still measured survival rate against it, so the check was comparing
two different quantities and passing only because 0.438 happens to be below
0.56.

``survival_rate`` is still computed, still printed, and still pinned -- it
describes one run usefully. It is no longer what decides the lane.
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
    measure_fixed_universe_bypass,
    run_red_team,
)
from omni_mercury_engine.intel.value_metrics import VALUE_METRICS

BASELINE_PATH = _REPO / "benchmarks" / "red_team_baseline.json"
ARTIFACT_PATH = _REPO / "artifacts" / "red_team" / "run_summary.json"
#: Float-comparison epsilon only. Both rates are set-cardinality ratios that are
#: fully deterministic and order-independent for a fixed config + gate, so there
#: is NO benign drift for a slack margin to absorb. The gate therefore fails on
#: any rise above the pinned floor (a real weakening), and separately never
#: permits a rate above the declared value-metric ceiling.
_FLOAT_EPS = 1e-9


def _run() -> tuple[Any, dict[str, Any]]:
    cfg = RedTeamConfig.load()
    result = run_red_team(cfg)
    summary = result.summary()
    # The gating measurement. Scored over the same config, so one invocation of
    # the harness reports both the descriptive per-run view and the sound
    # no-weakening quantity rather than leaving the latter to a separate tool.
    summary["fixed_universe"] = measure_fixed_universe_bypass(cfg)
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
    fixed = summary["fixed_universe"]
    fixed_rate = float(fixed["bypass_rate"])

    if args.update:
        BASELINE_PATH.write_text(
            json.dumps(
                {
                    # The gating quantity: denominator fixed by the config, so it
                    # is monotone in gate strength.
                    "fixed_universe_bypass_rate": round(fixed_rate, 6),
                    "fixed_universe_candidates": fixed["n_candidates"],
                    "fixed_universe_bypassed": fixed["n_bypassed"],
                    # Descriptive, retained: survival_rate characterises one run,
                    # but its denominator moves with gate strength so it is not
                    # gated on. See the module docstring.
                    "survival_rate": round(rate, 6),
                    "n_candidates": summary["n_candidates"],
                    "n_survivors": summary["n_survivors"],
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
            f"pinned red-team baseline: fixed_universe_bypass_rate={fixed_rate:.4f} "
            f"({fixed['n_bypassed']}/{fixed['n_candidates']} bypass); "
            f"survival_rate={rate:.4f} (descriptive)"
        )
        return 0

    if args.check:
        if not BASELINE_PATH.is_file():
            print(f"missing baseline {BASELINE_PATH.name}; run --update to pin it", file=sys.stderr)
            return 1
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        if "fixed_universe_bypass_rate" not in baseline:
            print(
                f"{BASELINE_PATH.name} predates the fixed-universe metric and pins only the "
                "survival rate, which is not a sound no-weakening guard; run --update to "
                "re-pin before gating on it",
                file=sys.stderr,
            )
            return 1
        floor = baseline["fixed_universe_bypass_rate"]
        problems = []
        # Compare at the precision the floor is pinned to (6 decimals, as --update
        # writes): the true rate is a full-precision ratio while the pinned floor is
        # rounded, so a raw ``>`` would trip on the rounding delta, not a real
        # weakening.
        rate_q = round(fixed_rate, 6)
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
                f"fixed-universe bypass rate {rate_q:.6f} rose above pinned floor "
                f"{floor:.6f} (gate weakened against obfuscation)"
            )
        if rate_q > declared + _FLOAT_EPS:
            problems.append(
                f"fixed-universe bypass rate {rate_q:.6f} exceeds declared value-metric "
                f"ceiling {declared:.4f}"
            )
        # The universe is a property of the config alone, so it must not move
        # unless the config did. A shrinking universe would mean the harness is
        # scoring fewer candidates -- the exact denominator drift this metric
        # exists to remove.
        pinned_universe = baseline.get("fixed_universe_candidates")
        if pinned_universe is not None and fixed["n_candidates"] != pinned_universe:
            problems.append(
                f"fixed universe changed size {pinned_universe} -> {fixed['n_candidates']}: "
                "the candidate set is config-derived, so re-pin deliberately with --update "
                "when the config changes and never let it drift silently"
            )
        if problems:
            print("RED-TEAM REGRESSION:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 1
        print(
            f"OK: fixed-universe bypass rate {fixed_rate:.4f} at/below pinned floor "
            f"{floor:.4f} ({fixed['n_bypassed']}/{fixed['n_candidates']} bypass, "
            f"{fixed['n_blocked']} blocked); survival_rate {rate:.4f} (descriptive); "
            f"{summary['n_skipped_seeds']} skipped seed(s), {appended} newly appended"
        )
        return 0

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
