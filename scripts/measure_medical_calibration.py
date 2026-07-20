#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure calibration + clinical metrics for medical scores (reproducible harness).

Runs Mercury's clinical measurement layer end to end and writes a reproducible
JSON artifact:

1. builds seeded reference cohorts (synthetic self-validation cohorts always; the
   real Framingham instrument cohort when ``torch`` is importable);
2. for every cohort, sweeps all wired calibrators
   (conformal / Venn-Abers / Bayesian / isotonic / Platt / Beta / temperature),
   measuring AUROC, sensitivity, specificity, PPV, NPV, Brier, ECE, MCE,
   reliability and distribution-free conformal coverage before and after
   calibration;
3. applies the metric-based :class:`ClinicalSignalGate` to each raw score.

Usage::

    python scripts/measure_medical_calibration.py            # write artifact
    python scripts/measure_medical_calibration.py --check     # verify invariants
    python scripts/measure_medical_calibration.py --out PATH  # custom output

The ``--check`` mode recomputes the measurements and asserts the *invariants*
(fitted calibrators never worsen ECE beyond tolerance; conformal coverage meets
target within sampling slack; the gate proves signal on the signal-bearing
cohorts and refuses the noise cohort). It does not require bit-identical floats,
so it stays stable across numpy point releases.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "benchmarks" / "medical_calibration_report.json"
SCHEMA_VERSION = "1.0"
SEED = 20260719
COVERAGE_TARGET = 0.9
COVERAGE_SLACK = 0.06


def _build_cohorts() -> list[Any]:
    """Build the reference cohorts available in this environment."""
    from omni_mercury_engine.medical.reference_cohorts import (
        synthetic_calibrated_cohort,
    )

    cohorts: list[Any] = [
        synthetic_calibrated_cohort(n=6000, seed=SEED, miscalibration=1.0),
        synthetic_calibrated_cohort(n=6000, seed=SEED + 1, miscalibration=2.2),
    ]
    try:  # real instrument cohort needs the torch-backed cardiology module
        import torch  # noqa: F401

        from omni_mercury_engine.medical.reference_cohorts import framingham_cvd_cohort

        cohorts.append(framingham_cvd_cohort(n=1500, seed=SEED))
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[info] Framingham cohort skipped (torch unavailable): {exc}")
    return cohorts


def _noise_cohort() -> Any:
    """A no-signal cohort the gate must refuse (untrained-net stand-in)."""
    import numpy as np

    from omni_mercury_engine.medical.reference_cohorts import ReferenceCohort

    rng = np.random.RandomState(SEED + 99)
    n = 3000
    labels = (rng.uniform(size=n) < 0.4).astype(float)
    scores = rng.uniform(size=n)
    return ReferenceCohort(
        name="noise_control",
        description="Score independent of outcome; the gate must refuse it",
        dgp_doc="label ~ Bernoulli(0.4); score ~ Uniform(0,1) independent of label",
        scores=scores,
        labels=labels,
        seed=SEED + 99,
    )


def _measure_cohort(cohort: Any) -> dict[str, Any]:
    """Split, sweep calibrators, and gate one cohort (raw and best-calibrated)."""
    from omni_mercury_engine.medical.clinical_calibration import (
        compare_calibrators,
        fit_calibrator,
    )
    from omni_mercury_engine.medical.clinical_metrics import evaluate_clinical_scores
    from omni_mercury_engine.medical.clinical_signal_gate import ClinicalSignalGate
    from omni_mercury_engine.medical.reference_cohorts import split_cohort

    cal, test = split_cohort(cohort, calibration_fraction=0.5, seed=cohort.seed)
    sweep = compare_calibrators(
        cal.scores,
        cal.labels,
        test.scores,
        test.labels,
        coverage=COVERAGE_TARGET,
        seed=cohort.seed,
    )
    gate = ClinicalSignalGate()
    raw_report = evaluate_clinical_scores(test.labels, test.scores, seed=cohort.seed)
    raw_verdict = gate.evaluate(raw_report)

    # Gate the best-calibrated score too: this demonstrates the end-to-end value
    # -- a discriminating-but-miscalibrated score the gate refuses on ECE becomes
    # trustworthy once the chosen calibrator is applied.
    best_method = sweep["best_method"]
    best_cal = fit_calibrator(best_method, cal.scores, cal.labels, seed=cohort.seed or 42)
    cal_scores_test = best_cal.transform(test.scores)
    cal_report = evaluate_clinical_scores(test.labels, cal_scores_test, seed=cohort.seed)
    cal_verdict = gate.evaluate(cal_report)

    return {
        "name": cohort.name,
        "description": cohort.description,
        "dgp": cohort.dgp_doc,
        "seed": cohort.seed,
        "prevalence": float(cohort.meta.get("prevalence", float("nan"))) if cohort.meta else None,
        "n_calibration": len(cal.scores),
        "n_test": len(test.scores),
        "raw_score_metrics": raw_report.to_dict(),
        "signal_gate": raw_verdict.to_dict(),
        "best_calibrated_method": best_method,
        "best_calibrated_metrics": cal_report.to_dict(),
        "signal_gate_calibrated": cal_verdict.to_dict(),
        "calibration_sweep": sweep,
    }


def build_report() -> dict[str, Any]:
    """Compute the full measurement report for all cohorts."""
    import numpy as np

    cohorts = _build_cohorts()
    results = [_measure_cohort(c) for c in cohorts]
    results.append(_measure_cohort(_noise_cohort()))
    return {
        "schema_version": SCHEMA_VERSION,
        "numpy_version": np.__version__,
        "coverage_target": COVERAGE_TARGET,
        "master_seed": SEED,
        "cohorts": results,
        "note": (
            "Synthetic cohorts validate the measurement/calibration wiring and "
            "characterise internal reliability under a documented DGP; they do "
            "not establish real-world clinical accuracy, which requires governed "
            "datasets (e.g. MIMIC-III) and independent clinical validation."
        ),
    }


def _summarise(report: dict[str, Any]) -> str:
    """Return a compact human-readable summary of the report."""
    lines = [
        f"medical calibration report (schema {report['schema_version']}, "
        f"numpy {report['numpy_version']}, seed {report['master_seed']})",
        "",
    ]
    for c in report["cohorts"]:
        raw = c["raw_score_metrics"]
        sweep = c["calibration_sweep"]
        gate = c["signal_gate"]
        best = sweep["comparisons"][sweep["best_method"]]
        cov = best["coverage"]
        cov_str = (
            f"{cov.get('empirical_coverage', float('nan')):.3f}" if cov.get("available") else "n/a"
        )
        lines.append(
            f"- {c['name']:<22} AUROC={raw['auroc']:.3f} "
            f"[{raw['auroc_ci_low']:.3f},{raw['auroc_ci_high']:.3f}] "
            f"sens={raw['sensitivity']:.3f} spec={raw['specificity']:.3f} "
            f"brier={raw['brier']:.3f}"
        )
        cal_gate = c["signal_gate_calibrated"]
        lines.append(
            f"  {'':<22} ECE {raw['ece']:.3f} -> {sweep['best_ece']:.3f} "
            f"via {sweep['best_method']}; coverage@{report['coverage_target']}="
            f"{cov_str}; signal_proven raw={gate['proven']} "
            f"calibrated={cal_gate['proven']}"
        )
    return "\n".join(lines)


def _check_invariants(report: dict[str, Any]) -> list[str]:
    """Return a list of invariant violations (empty => all invariants hold)."""
    problems: list[str] = []
    for c in report["cohorts"]:
        name = c["name"]
        sweep = c["calibration_sweep"]
        baseline = sweep["baseline_ece"]
        for method, comp in sweep["comparisons"].items():
            if comp["fitted"]:
                after = comp["report_calibrated"]["ece"]
                # A fitted calibrator must not badly worsen ECE on the test split.
                if after > baseline + 0.05:
                    problems.append(
                        f"{name}/{method}: fitted calibrator worsened ECE "
                        f"{baseline:.3f} -> {after:.3f}"
                    )
                cov = comp["coverage"]
                if cov.get("available") and cov["empirical_coverage"] < (
                    report["coverage_target"] - COVERAGE_SLACK
                ):
                    problems.append(
                        f"{name}/{method}: conformal coverage "
                        f"{cov['empirical_coverage']:.3f} below target"
                    )

        raw = c["raw_score_metrics"]
        raw_proven = c["signal_gate"]["proven"]
        cal_proven = c["signal_gate_calibrated"]["proven"]
        has_signal = raw["auroc_ci_low"] > 0.5

        # Pure noise must never clear the gate, raw or calibrated (calibration
        # cannot manufacture discrimination it does not have).
        if name == "noise_control":
            if raw_proven or cal_proven:
                problems.append("noise_control: gate proved signal on pure noise")
            continue

        # A genuinely discriminating score must (a) calibrate to good reliability
        # and (b) clear the gate once calibrated -- the end-to-end guarantee.
        if has_signal:
            if sweep["best_ece"] >= 0.06:
                problems.append(f"{name}: best calibrated ECE {sweep['best_ece']:.3f} not < 0.06")
            if not cal_proven:
                problems.append(f"{name}: signal-bearing score not proven after calibration")
    return problems


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Recompute and assert invariants instead of writing the artifact.",
    )
    args = parser.parse_args(argv)

    report = build_report()
    print(_summarise(report))

    if args.check:
        problems = _check_invariants(report)
        if problems:
            print("\nINVARIANT VIOLATIONS:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 1
        print("\nAll medical-calibration invariants hold.")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
