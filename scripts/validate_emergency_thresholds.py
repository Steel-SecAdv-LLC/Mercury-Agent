#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Validate medical emergency-routing thresholds against seeded outcome cohorts.

Measures the operating characteristics (sensitivity / specificity / PPV / NPV /
Youden-J / F2) of Mercury's literature-anchored emergency cutoffs -- NIHSS
``stroke_risk >= 0.6``, troponin I ``> 0.4 ng/mL``, NEWS2 ``>= 7`` -- on
reproducible synthetic outcome cohorts with documented DGPs, sweeps the full
grid, and reports where an outcome-optimal operating point would sit.

The operational thresholds are NOT changed (see the harness docstring): this is
measurement + advisory. Metrics hold under the stated synthetic DGP and do not
establish real-world clinical performance.

Usage::

    python scripts/validate_emergency_thresholds.py            # write artifact
    python scripts/validate_emergency_thresholds.py --check     # assert invariants
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "benchmarks" / "emergency_threshold_validation.json"
SEED = 20260719


def _sigmoid(x: Any) -> Any:
    import numpy as np

    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _nihss_cohort(n: int = 4000, seed: int = SEED) -> dict[str, Any]:
    """NIHSS stroke-risk cohort vs an independent emergency-intervention DGP."""
    import numpy as np

    from omni_mercury_engine.medical.emergency_thresholds import nihss_stroke_risk

    rng = np.random.RandomState(seed)
    # Mixture: most patients low NIHSS, a tail of moderate-severe strokes.
    minor = rng.randint(0, 5, size=n)
    severe = rng.randint(5, 43, size=n)
    is_severe = rng.uniform(size=n) < 0.35
    nihss = np.where(is_severe, severe, minor)
    # Emergency intervention (LVO/thrombectomy-eligible or tPA-warranted) rises
    # steeply with NIHSS; independent logistic + noise (not the risk-band map).
    logit = -3.0 + 0.30 * nihss + rng.normal(0.0, 0.7, size=n)
    outcomes = (rng.uniform(size=n) < _sigmoid(logit)).astype(float)
    scores = np.array([nihss_stroke_risk(int(v)) for v in nihss], dtype=float)
    return {
        "instrument": "NIHSS_stroke_risk",
        "literature_anchor": "stroke_risk>=0.6 == NIHSS>=5 (moderate stroke); AHA/ASA tPA/LVO triage",
        "current_threshold": 0.6,
        "scores": scores,
        "outcomes": outcomes,
        "grid": [0.0, 0.3, 0.6, 0.8, 1.0],
        "dgp_doc": (
            "NIHSS ~ mixture(minor U[0,4], severe U[5,42], 35% severe); "
            "emergency ~ Bernoulli(sigmoid(-3.0 + 0.30*NIHSS + N(0,0.7))); "
            "score = production NIHSS->stroke_risk band"
        ),
    }


def _troponin_cohort(n: int = 4000, seed: int = SEED + 1) -> dict[str, Any]:
    """Troponin I cohort vs an acute-MI DGP (log-normal separation)."""
    import numpy as np

    rng = np.random.RandomState(seed)
    is_mi = rng.uniform(size=n) < 0.3
    # Non-MI: mostly < URL (0.04), some non-MI elevation up to ~0.3.
    non_mi = np.exp(rng.normal(np.log(0.015), 0.9, size=n))
    # MI: elevated, median ~2 ng/mL, wide.
    mi = np.exp(rng.normal(np.log(2.0), 1.0, size=n))
    trop = np.where(is_mi, mi, non_mi)
    outcomes = is_mi.astype(float)
    grid = [
        round(float(x), 4)
        for x in np.unique(
            np.concatenate(
                [
                    np.array([0.02, 0.04, 0.06, 0.1, 0.2, 0.4, 0.6, 1.0, 2.0]),
                ]
            )
        )
    ]
    return {
        "instrument": "troponin_i_ng_ml",
        "literature_anchor": "troponin I>0.4 ng/mL acute-MI trigger (~10x the 0.04 URL)",
        "current_threshold": 0.4,
        "scores": trop.astype(float),
        "outcomes": outcomes,
        "grid": grid,
        "dgp_doc": (
            "MI prevalence 0.3; non-MI troponin ~ LogNormal(log 0.015, 0.9); "
            "MI troponin ~ LogNormal(log 2.0, 1.0); outcome = acute MI"
        ),
    }


def _news2_cohort(n: int = 4000, seed: int = SEED + 2) -> dict[str, Any]:
    """NEWS2 aggregate cohort vs a 24h-deterioration DGP from vitals."""
    import numpy as np

    from omni_mercury_engine.medical.emergency_thresholds import news2_score

    rng = np.random.RandomState(seed)
    scores = np.empty(n, dtype=float)
    severity = rng.beta(1.6, 4.0, size=n)  # latent acuity in [0,1], right-skewed
    for i in range(n):
        s = severity[i]
        vitals = {
            "respiratory_rate": int(np.clip(rng.normal(16 + 14 * s, 3), 6, 40)),
            "spo2": int(np.clip(rng.normal(98 - 12 * s, 2), 80, 100)),
            "on_oxygen": bool(rng.uniform() < 0.15 + 0.5 * s),
            "temperature_c": float(np.clip(rng.normal(36.8 + 1.6 * s, 0.5), 34.0, 41.0)),
            "systolic_bp": int(np.clip(rng.normal(124 - 34 * s, 12), 70, 210)),
            "heart_rate": int(np.clip(rng.normal(78 + 48 * s, 12), 38, 170)),
            "consciousness": "A" if rng.uniform() > 0.6 * s else "V",
        }
        scores[i] = float(news2_score(vitals))
    logit = -3.2 + 6.5 * severity + rng.normal(0.0, 0.6, size=n)
    outcomes = (rng.uniform(size=n) < _sigmoid(logit)).astype(float)
    return {
        "instrument": "NEWS2_aggregate",
        "literature_anchor": "NEWS2>=7 high-risk / urgent-response trigger (RCP NEWS2 2017)",
        "current_threshold": 7.0,
        "scores": scores,
        "outcomes": outcomes,
        "grid": [float(t) for t in range(0, 18)],
        "dgp_doc": (
            "latent acuity ~ Beta(1.6,4.0) drives vitals derangement; "
            "score = NEWS2 aggregate of those vitals; "
            "deterioration ~ Bernoulli(sigmoid(-3.2 + 6.5*acuity + N(0,0.6)))"
        ),
    }


def build_report() -> dict[str, Any]:
    """Validate every emergency threshold and assemble the report."""
    from omni_mercury_engine.medical.emergency_thresholds import validate_threshold

    cohorts = [_nihss_cohort(), _troponin_cohort(), _news2_cohort()]
    reports = []
    for c in cohorts:
        report = validate_threshold(
            instrument=c["instrument"],
            literature_anchor=c["literature_anchor"],
            scores=c["scores"],
            outcomes=c["outcomes"],
            current_threshold=c["current_threshold"],
            grid=c["grid"],
            dgp_doc=c["dgp_doc"],
        )
        reports.append(report.to_dict())
    return {"schema_version": "1.0", "master_seed": SEED, "thresholds": reports}


def _summarise(report: dict[str, Any]) -> str:
    """Return a compact human-readable summary."""
    lines = [f"emergency threshold validation (seed {report['master_seed']})", ""]
    for t in report["thresholds"]:
        cur = t["current"]
        f2 = t["recommended_f2"]
        yj = t["recommended_youden"]
        lines.append(
            f"- {t['instrument']:<20} current thr={t['current_threshold']}: "
            f"sens={cur['sensitivity']:.3f} spec={cur['specificity']:.3f} "
            f"F2={cur['f2']:.3f} J={cur['youden_j']:.3f}"
        )
        lines.append(
            f"  {'':<20} advisory F2-opt thr={f2['threshold']} "
            f"(sens={f2['sensitivity']:.3f} spec={f2['specificity']:.3f}); "
            f"Youden-opt thr={yj['threshold']}"
        )
    return "\n".join(lines)


def _check(report: dict[str, Any]) -> list[str]:
    """Return invariant violations (empty => all hold)."""
    problems: list[str] = []
    for t in report["thresholds"]:
        name = t["instrument"]
        # Each instrument must discriminate: its Youden-optimal J must be positive.
        if t["recommended_youden"]["youden_j"] <= 0.1:
            problems.append(f"{name}: no discriminating threshold (max J <= 0.1)")
        # Monotonicity: sensitivity is non-increasing as the threshold rises.
        sens = [p["sensitivity"] for p in t["sweep"]]
        if any(sens[i] < sens[i + 1] - 1e-9 for i in range(len(sens) - 1)):
            problems.append(f"{name}: sensitivity not monotonic in threshold")
    return problems


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    report = build_report()
    print(_summarise(report))

    if args.check:
        problems = _check(report)
        if problems:
            print("\nINVARIANT VIOLATIONS:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 1
        print("\nAll emergency-threshold invariants hold.")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
