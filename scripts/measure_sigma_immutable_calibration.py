#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure σ_Immutable gate calibration + sweep threshold/temperature (read-only).

Loads the frozen σ_Immutable :class:`EthicalGate`, scores a held-out labelled
integrity corpus, and reports ECE / MCE / Brier / reliability plus AUROC and
gate confusion across a temperature grid and a threshold grid. It proves the
frozen operational constant ``0.9999216794967651`` is untouched and never
repoints any operational constant -- every swept value is advisory.

Usage::

    python scripts/measure_sigma_immutable_calibration.py            # write artifact
    python scripts/measure_sigma_immutable_calibration.py --check     # assert invariants
    python scripts/measure_sigma_immutable_calibration.py --out PATH  # custom output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "benchmarks" / "sigma_immutable_calibration_results.json"


def _summarise(report: dict[str, Any]) -> str:
    """Return a compact human-readable summary."""
    inv = report["frozen_constant_invariant"]
    op = report["operational_point"]
    ts = report["temperature_sweep"]
    rec = report["recommended_advisory"]
    lines = [
        "σ_Immutable calibration (read-only measurement)",
        f"  frozen-constant invariant holds: {inv['invariant_holds']} "
        f"(operational={inv['operational_score']:.16f})",
        f"  operational (T=1, thr=0.93): AUROC={op['auroc']:.3f} "
        f"ECE={op['ece']:.3f} MCE={op['mce']:.3f} Brier={op['brier']:.3f} "
        f"sens={op['sensitivity']:.3f} spec={op['specificity']:.3f}",
        f"  best temperature by ECE: T={ts['best_temperature_by_ece']}",
        f"  advisory optimum: T={rec['temperature']} thr={rec['threshold']:.3f} "
        f"ECE={rec['metrics']['ece']:.3f} balanced_acc="
        f"{rec['metrics']['balanced_accuracy']:.3f}",
    ]
    return "\n".join(lines)


def _check(report: dict[str, Any]) -> list[str]:
    """Return invariant violations (empty => all hold)."""
    problems: list[str] = []
    inv = report["frozen_constant_invariant"]
    if not inv["invariant_holds"]:
        problems.append(
            f"frozen constant moved: operational={inv['operational_score']!r} "
            f"vs {inv['frozen_constant']!r}"
        )
    op = report["operational_point"]
    # The trained gate must discriminate intact from tampered clearly.
    if op["auroc"] < 0.6:
        problems.append(f"operational AUROC {op['auroc']:.3f} below 0.6 (gate not discriminating)")
    # A temperature that lowers ECE must exist (calibration head-room is measurable).
    temps = report["temperature_sweep"]["points"]
    best_ece = min(p["ece"] for p in temps)
    if best_ece > op["ece"] + 1e-9:
        problems.append("temperature sweep found no ECE at or below the operational point")
    return problems


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    try:
        from omni_mercury_engine.security.sigma_calibration import build_report
    except Exception as exc:  # pragma: no cover - import environment guard
        print(f"σ_Immutable calibration unavailable: {exc}", file=sys.stderr)
        return 2

    try:
        report = build_report(seed=args.seed)
    except RuntimeError as exc:  # trained gate / weights missing
        print(f"σ_Immutable calibration cannot run: {exc}", file=sys.stderr)
        return 2

    print(_summarise(report))

    if args.check:
        problems = _check(report)
        if problems:
            print("\nINVARIANT VIOLATIONS:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 1
        print("\nAll σ_Immutable calibration invariants hold.")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"\nWrote {args.out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
