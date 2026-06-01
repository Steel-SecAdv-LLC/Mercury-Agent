"""
Mercury Agent - Copyright (C) 2025 Steel Security Advisors LLC
Licensed under GNU GPL v3

Statistical confirmation of the sub-threshold neuro-symbolic sweep results.

PR #265 ran same-cell sweeps and reported *point* deltas plus a "seed
agreement" heuristic against a pre-registered +0.002 bar:

  * salience rule graph: consensus_salience +0.0022 vs consensus +0.0009
    (seed-agree 0.81) -> KEEP consensus,
  * crisp t-norm semantics: godel +0.0039 (best of three) vs product +0.0025
    -> KEEP product.

Both calls were made on point estimates.  This module adds the *formal
inference* those calls deserve, computed directly from the committed sweep
artifacts (no re-run required):

  * paired difference per cell (A_auc - B_auc), same dataset/seed/fraction,
  * paired t-test (two-sided p-value),
  * Wilcoxon signed-rank test (scipy, optional),
  * exact sign test (binomial),
  * bootstrap 95% confidence interval (seeded, reproducible),
  * one-sample tests of each series' delta-over-neural vs 0.

A claim is only "confirmed" when the paired mean both clears the
pre-registered bar AND the 95% CI excludes 0 AND the paired t-test is
significant at alpha=0.05.  Anything else is reported as within the noise
floor — which is the honest reading of #265's KEEP decisions.

Dependencies: numpy only (scipy used opportunistically for Wilcoxon).

Usage::

    python -m benchmarks.statistical_significance
    python -m benchmarks.statistical_significance \\
        --rulegraph artifacts/symbolic_rulegraph_sweep.json \\
        --semantics artifacts/symbolic_semantics_sweep.json \\
        --out artifacts/statistical_significance.json --seed 0 --boot 10000
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ALPHA = 0.05
PREREGISTERED_BAR = 0.002  # the +0.002 mean-ΔAUC dominance bar used in #265


def _t_sf(t_abs: float, df: int) -> float:
    """Two-sided p-value for a t statistic. Uses scipy if present, else a
    numerically-stable approximation via the regularized incomplete beta."""
    try:
        # ``scipy.*`` is a declared ignore_missing_imports boundary in
        # pyproject.toml, so no inline ``# type: ignore`` is needed (and one
        # would be flagged unused when scipy is installed, e.g. the [ml] lane).
        from scipy import stats

        return float(2.0 * stats.t.sf(t_abs, df))
    except Exception:
        # Student-t survival via incomplete beta: P(|T|>t) = I_x(df/2, 1/2),
        # x = df/(df+t^2).  math.lgamma-based continued-fraction betainc.
        x = df / (df + t_abs * t_abs)
        return float(min(1.0, _betainc(df / 2.0, 0.5, x)))


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a
    # Lentz continued fraction
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 200):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= d * c
        if abs(1.0 - d * c) < 1e-12:
            break
    return front * (f - 1.0)


def _sign_test_p(n_pos: int, n: int) -> float:
    """Exact two-sided sign test p-value (binomial, p=0.5)."""
    if n == 0:
        return 1.0
    k = max(n_pos, n - n_pos)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) * (0.5**n)
    return float(min(1.0, 2.0 * tail))


def _bootstrap_ci(d: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(d)
    means = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def paired_stats(
    a: np.ndarray, b: np.ndarray, *, n_boot: int, seed: int, bar: float
) -> dict[str, Any]:
    """Paired inference on d = a - b (same cells)."""
    d = a - b
    n = len(d)
    mean = float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 0 and sd > 0 else 0.0
    t_stat = mean / se if se > 0 else 0.0
    p_t = _t_sf(abs(t_stat), n - 1) if n > 1 and se > 0 else 1.0
    ci_lo, ci_hi = _bootstrap_ci(d, n_boot, seed)
    n_pos = int(np.sum(d > 0))
    p_sign = _sign_test_p(n_pos, n)
    p_wilcoxon: float | None = None
    try:
        from scipy import stats

        if n > 0 and np.any(d != 0):
            p_wilcoxon = float(
                stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue
            )
    except Exception:
        p_wilcoxon = None

    clears_bar = mean > bar
    ci_excludes_zero = ci_lo > 0.0
    significant = bool((p_t < ALPHA) and ci_excludes_zero)
    confirmed = bool(clears_bar and significant)
    return {
        "n_cells": n,
        "mean_diff": mean,
        "std_diff": sd,
        "t_stat": t_stat,
        "p_value_ttest": p_t,
        "p_value_wilcoxon": p_wilcoxon,
        "p_value_sign_test": p_sign,
        "n_positive": n_pos,
        "frac_positive": float(n_pos / n) if n else 0.0,
        "bootstrap_ci95": [ci_lo, ci_hi],
        "preregistered_bar": bar,
        "clears_bar": clears_bar,
        "ci_excludes_zero": ci_excludes_zero,
        "significant_at_0.05": significant,
        "confirmed": confirmed,
    }


def one_sample_over_neural(
    series: np.ndarray, neural: np.ndarray, *, n_boot: int, seed: int
) -> dict[str, Any]:
    """Is the series' delta over the neural baseline distinguishable from 0?"""
    return paired_stats(series, neural, n_boot=n_boot, seed=seed, bar=0.0)


def _series_matrix(cells: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([float(c[key]) for c in cells], dtype=float)


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm–Bonferroni step-down adjusted p-values (family-wise error control).

    Given ``m`` raw p-values, returns adjusted p-values such that rejecting
    every hypothesis whose adjusted value is ``< alpha`` controls the
    family-wise error rate at ``alpha`` — strictly more powerful than plain
    Bonferroni while making no independence assumption.  The transform is the
    cumulative-max of ``(m - rank) * p`` over p-values sorted ascending, each
    clamped to ``[0, 1]``; ties and ordering are handled by argsort/inverse.
    """
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        scaled = min(1.0, (m - rank) * pvals[idx])
        running = max(running, scaled)  # enforce monotonicity (step-down)
        adjusted[idx] = running
    return adjusted


def _apply_family_correction(comparisons: dict[str, dict[str, Any]]) -> None:
    """Annotate each comparison in a reported family with Holm-adjusted
    inference, in place.  The raw single-test fields are left untouched; the
    family-wise verdict (``confirmed_familywise``) is what survives multiple
    testing and is the defensible basis for any KEEP/REPLACE decision."""
    names = sorted(comparisons)  # deterministic family order
    raw_p = [float(comparisons[n]["p_value_ttest"]) for n in names]
    adj_p = holm_bonferroni(raw_p)
    for name, p_holm in zip(names, adj_p):
        c = comparisons[name]
        sig_fw = bool(p_holm < ALPHA and c["ci_excludes_zero"])
        c["p_value_ttest_holm"] = p_holm
        c["significant_familywise"] = sig_fw
        c["confirmed_familywise"] = bool(c["clears_bar"] and sig_fw)


def analyze(
    rulegraph_path: Path, semantics_path: Path, *, n_boot: int, seed: int
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "method": {
            "alpha": ALPHA,
            "preregistered_bar": PREREGISTERED_BAR,
            "bootstrap_resamples": n_boot,
            "bootstrap_seed": seed,
            "confirm_rule": "mean_diff > bar AND bootstrap CI95 lower > 0 AND paired t-test p < alpha",
            "multiple_comparison_correction": "holm-bonferroni",
            "familywise_confirm_rule": (
                "mean_diff > bar AND bootstrap CI95 lower > 0 AND "
                "Holm-adjusted paired t-test p < alpha (across the reported family)"
            ),
        }
    }

    rg = json.loads(rulegraph_path.read_text())["cells"]
    rg_sal = _series_matrix(rg, "consensus_salience")
    rg_con = _series_matrix(rg, "consensus")
    rg_neu = _series_matrix(rg, "neural")
    out["rulegraph_salience_vs_consensus"] = paired_stats(
        rg_sal, rg_con, n_boot=n_boot, seed=seed, bar=PREREGISTERED_BAR
    )
    out["rulegraph_consensus_vs_neural"] = one_sample_over_neural(
        rg_con, rg_neu, n_boot=n_boot, seed=seed
    )
    out["rulegraph_salience_vs_neural"] = one_sample_over_neural(
        rg_sal, rg_neu, n_boot=n_boot, seed=seed
    )

    sem = json.loads(semantics_path.read_text())["cells"]
    sem_god = _series_matrix(sem, "godel")
    sem_prod = _series_matrix(sem, "product")
    sem_neu = _series_matrix(sem, "neural")
    out["semantics_godel_vs_product"] = paired_stats(
        sem_god, sem_prod, n_boot=n_boot, seed=seed, bar=PREREGISTERED_BAR
    )
    out["semantics_godel_vs_neural"] = one_sample_over_neural(
        sem_god, sem_neu, n_boot=n_boot, seed=seed
    )
    out["semantics_product_vs_neural"] = one_sample_over_neural(
        sem_prod, sem_neu, n_boot=n_boot, seed=seed
    )

    # Family-wise correction over the full reported family of paired t-tests.
    # Every comparison is annotated in place with Holm-adjusted inference;
    # ``confirmed_familywise`` is the multiple-testing-defensible verdict.
    comparisons = {k: v for k, v in out.items() if isinstance(v, dict) and "p_value_ttest" in v}
    _apply_family_correction(comparisons)

    sal = out["rulegraph_salience_vs_consensus"]
    god = out["semantics_godel_vs_product"]
    # Any alternative whose advantage survives both the single-test rule and
    # the family-wise correction would force revisiting a default.
    fw_winners = sorted(n for n, c in comparisons.items() if c["confirmed_familywise"])
    # Claims that look significant uncorrected but dissolve under correction —
    # surfaced explicitly so they are never quoted as confirmed findings.
    uncorrected_only = sorted(
        n for n, c in comparisons.items() if c["confirmed"] and not c["confirmed_familywise"]
    )
    out["conclusion"] = {
        "salience_beats_consensus": sal["confirmed_familywise"],
        "godel_beats_product": god["confirmed_familywise"],
        "familywise_confirmed": fw_winners,
        "confirmed_uncorrected_only": uncorrected_only,
        "verdict": (
            "Neither sub-threshold winner is confirmed by paired inference: "
            f"salience-vs-consensus mean Δ={sal['mean_diff']:+.4f} "
            f"(CI95 [{sal['bootstrap_ci95'][0]:+.4f}, {sal['bootstrap_ci95'][1]:+.4f}], "
            f"p={sal['p_value_ttest']:.3f}, Holm p={sal['p_value_ttest_holm']:.3f}); "
            f"godel-vs-product mean Δ={god['mean_diff']:+.4f} "
            f"(CI95 [{god['bootstrap_ci95'][0]:+.4f}, {god['bootstrap_ci95'][1]:+.4f}], "
            f"p={god['p_value_ttest']:.3f}, Holm p={god['p_value_ttest_holm']:.3f}). "
            "This statistically corroborates PR #265's KEEP-default decisions."
            + (
                f" Note: {', '.join(uncorrected_only)} clear the single-test bar "
                "but do NOT survive Holm-Bonferroni across the 6-test family, so "
                "they are not reported as confirmed."
                if uncorrected_only
                else ""
            )
            if not fw_winners
            else (
                f"Family-wise confirmed alternative(s): {', '.join(fw_winners)}; "
                "revisit the corresponding default."
            )
        ),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rulegraph", default="artifacts/symbolic_rulegraph_sweep.json")
    parser.add_argument("--semantics", default="artifacts/symbolic_semantics_sweep.json")
    parser.add_argument("--out", default="artifacts/statistical_significance.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--boot", type=int, default=10000)
    args = parser.parse_args(argv)

    rg, sem = Path(args.rulegraph), Path(args.semantics)
    if not rg.exists() or not sem.exists():
        print(
            "ERROR: sweep artifacts missing. Regenerate them first:\n"
            "  python -m benchmarks.symbolic_rulegraph_sweep\n"
            "  python -m benchmarks.symbolic_semantics_sweep"
        )
        return 2

    result = analyze(rg, sem, n_boot=args.boot, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))

    def _line(name: str, s: dict[str, Any]) -> None:
        ci = s["bootstrap_ci95"]
        print(
            f"  {name:<34} n={s['n_cells']:>2}  meanΔ={s['mean_diff']:+.4f}  "
            f"CI95=[{ci[0]:+.4f},{ci[1]:+.4f}]  p_t={s['p_value_ttest']:.3f}  "
            f"p_holm={s['p_value_ttest_holm']:.3f}  "
            f"confirmed_fw={s['confirmed_familywise']}"
        )

    print("Statistical significance — neuro-symbolic sub-threshold winners")
    print("-" * 96)
    _line("salience vs consensus (bar+.002)", result["rulegraph_salience_vs_consensus"])
    _line("godel vs product (bar+.002)", result["semantics_godel_vs_product"])
    _line("consensus vs neural", result["rulegraph_consensus_vs_neural"])
    _line("salience vs neural", result["rulegraph_salience_vs_neural"])
    _line("product vs neural", result["semantics_product_vs_neural"])
    _line("godel vs neural", result["semantics_godel_vs_neural"])
    print("-" * 96)
    print(f"VERDICT: {result['conclusion']['verdict']}")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
