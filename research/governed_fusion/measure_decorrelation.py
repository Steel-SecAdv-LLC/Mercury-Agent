# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Decorrelated-stream fusion protocol (FINDINGS.md) — executed, not deferred.

``FINDINGS.md`` closed *fusion-beats-best-single* as a conclusive negative on the
reachable suite: every weighting of the three committed component streams
(resonance, kinematic, info_geometry) underperforms the single best stream
(reliability-clipped 0.840 < best-single 0.876).  It logged **one** untried
hypothesis as a precise, kill-criteria'd protocol rather than a vague TODO: that
the dilution is caused by *redundancy* among the three streams, and that adding a
genuinely **decorrelated** stream could let a learned stacker beat best-single.

This script runs that protocol end-to-end on the live reachable suite (the same
``(X, y)`` + cached default-detector scores every other ``measure_*.py`` uses):

1. **Diagnose redundancy.**  Per event, the pairwise Spearman rank correlation of
   the three component score vectors; report mean ``|rho_bar|``.
   *Pre-registered prediction:* ``|rho_bar| >~ 0.6`` accounts for the dilution.
2. **Add one decorrelated stream.**  The protocol-named primary candidate is a
   *temporal/sequence* detector over the same windows — a learned multi-lag
   autoregressive innovation residual, a genuinely different inductive bias from
   the three (global spectrum / single-step kinematics / global Mahalanobis).
   A *k*-NN local-density stream is measured alongside as a pre-declared
   sensitivity check so the verdict is not hostage to one detector choice.
   Re-measure ``|rho_bar|`` of each new stream against the existing three.
3. **Learned stacking on the enlarged pool.**  Per event a seeded class-stratified
   50/50 calibration/eval split; a logistic stacker fit on the calibration scores
   only; AUROC on the held-out eval split (the existing harness, no peeking).
4. **Decision rule (paired, pre-registered).**
   * **SHIP** only if the stacked AUROC beats best-single by a paired-bootstrap
     mean ``Delta >= +0.01`` with the 95% CI lower bound ``> 0`` across the live
     suite, **and** no per-domain regression worse than ``-0.01``.
   * **KILL** (record the negative with committed numbers exactly as Item 3 did)
     if, after adding the decorrelated stream, either ``|rho_bar| >= 0.5`` **or**
     the stacked gap's 95% CI upper bound is ``< +0.01``.

The verdict is whatever the numbers say.  Nothing here touches the runtime
``detect()`` path: this is a research measurement that writes one committed
artifact (``results/decorrelation_results.json``); a stream is only ever a
candidate for promotion if it clears the SHIP rule above.

Run::

    source research/governed_fusion/gf_env.sh
    python research/governed_fusion/measure_decorrelation.py
"""

from __future__ import annotations

import json
import os
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import ConstantInputWarning, spearmanr

from omni_mercury_engine.ml.mercury_ml import LogisticRegression, NearestNeighbors
from research.governed_fusion.measure_baseline import load_scores
from research.governed_fusion.measure_conformal import _split
from research.governed_fusion.metrics import _safe_auc
from research.governed_fusion.suite import build_suite, stratified_subsample

if TYPE_CHECKING:
    from research.governed_fusion.score_cache import EventScores
    from research.governed_fusion.suite import EventData

CAP = 6000
SEED = 42
AR_LAGS = 3
RIDGE = 1.0
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 12345
SHIP_DELTA = 0.01
KILL_RHO = 0.5
DOMAIN_REGRESSION = -0.01
COMPONENTS = ("resonance", "kinematic", "info_geometry")
_OUT_DIR = os.environ.get("GF_RESULTS_DIR", os.environ.get("GF_CACHE_DIR", "/home/user/gf_cache"))


def _robust_standardize(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Column-wise robust standardization: ``(x - median) / (1.4826 * MAD)``.

    Robust scaling keeps the temporal / density geometry stable when a handful of
    anomalous rows would otherwise dominate a mean/std scaler.  Columns with a
    zero MAD (constant features) collapse to zero, contributing no spurious
    signal.
    """
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0)
    scale = 1.4826 * mad
    scale = np.where(scale > 1e-12, scale, 1.0)
    return np.asarray((x - med) / scale, dtype=np.float64)


def temporal_innovation_score(x: np.ndarray[Any, Any], lags: int = AR_LAGS) -> np.ndarray[Any, Any]:
    """Decorrelated stream #1 — learned autoregressive innovation residual.

    Predicts each standardized row from its ``lags`` predecessors with a single
    ridge-regularized least-squares fit over the whole event, then scores each
    row by the L2 norm of its prediction residual ("how surprising is this row
    given recent context").  A *learned* multi-lag predictor is a different
    inductive bias from the fixed single-step finite differences of the kinematic
    stream, the global FFT profile of the resonance stream, and the global
    Mahalanobis distance of the info-geometry stream.  Higher = more anomalous.
    """
    z = _robust_standardize(x)
    n, d = z.shape
    if n <= lags + 1:
        return np.asarray(np.linalg.norm(z, axis=1), dtype=np.float64)
    design = np.concatenate([z[lags - k - 1 : n - k - 1] for k in range(lags)], axis=1)
    design = np.column_stack([design, np.ones(n - lags)])
    target = z[lags:]
    gram = design.T @ design + RIDGE * np.eye(design.shape[1])
    weights = np.linalg.solve(gram, design.T @ target)
    residual = target - design @ weights
    score = np.empty(n, dtype=np.float64)
    score[lags:] = np.linalg.norm(residual, axis=1)
    score[:lags] = np.median(score[lags:]) if n > lags else 0.0
    return np.asarray(score, dtype=np.float64)


def knn_density_score(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Decorrelated stream #2 (sensitivity) — k-NN local-density outlier score.

    Distance to the ``k``-th nearest neighbour in robustly-standardized feature
    space: a *local density* inductive bias (anomalies sit in sparse regions),
    distinct from the global centroid distance of the info-geometry stream.
    Higher = sparser neighbourhood = more anomalous.
    """
    z = _robust_standardize(x)
    n = z.shape[0]
    k = int(min(max(5, n // 50), 50))
    k = min(k, max(1, n - 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(z)
    dist, _ = nn.kneighbors(z)
    return np.asarray(dist[:, -1], dtype=np.float64)


def _mean_abs_pairwise_spearman(mat: np.ndarray[Any, Any]) -> float:
    """Mean ``|rho|`` over all unordered column pairs of ``mat`` (n, k)."""
    k = mat.shape[1]
    vals: list[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        for i in range(k):
            for j in range(i + 1, k):
                rho = spearmanr(mat[:, i], mat[:, j]).statistic
                if not np.isnan(rho):  # constant-column guard
                    vals.append(abs(float(rho)))
    return float(np.mean(vals)) if vals else float("nan")


def _mean_abs_spearman_against(new: np.ndarray[Any, Any], base: np.ndarray[Any, Any]) -> float:
    """Mean ``|rho|`` of one new stream against each column of ``base`` (n, k)."""
    vals: list[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        for j in range(base.shape[1]):
            rho = spearmanr(new, base[:, j]).statistic
            if not np.isnan(rho):
                vals.append(abs(float(rho)))
    return float(np.mean(vals)) if vals else float("nan")


def _stack_auroc(
    streams: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
    cal_idx: np.ndarray[Any, Any],
    ev_idx: np.ndarray[Any, Any],
) -> float:
    """Logistic stacker fit on calibration scores only; AUROC on eval split."""
    x_cal, x_ev = streams[cal_idx], streams[ev_idx]
    y_cal, y_ev = y[cal_idx], y[ev_idx]
    mu = x_cal.mean(axis=0)
    sd = x_cal.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    x_cal = (x_cal - mu) / sd
    x_ev = (x_ev - mu) / sd
    clf = LogisticRegression(max_iter=2000).fit(x_cal, y_cal)
    proba = np.asarray(clf.predict_proba(x_ev))[:, 1]
    return _safe_auc(y_ev, proba)


def _aligned_features(ev: EventData, es: EventScores) -> np.ndarray[Any, Any]:
    """Reproduce the exact subsample the cached scores were computed on.

    ``score_cache.event_scores`` fits the detector on
    ``stratified_subsample(ev.X, ev.y, CAP, seed=SEED)``; reproducing it with the
    same seed yields the identical row set and order, so a new stream computed on
    it aligns element-for-element with the cached component scores.
    """
    x_sub, y_sub = stratified_subsample(ev.X, ev.y, CAP, seed=SEED)
    if not np.array_equal(np.asarray(y_sub, dtype=int).reshape(-1), es.y):
        raise RuntimeError(
            f"{ev.domain}/{ev.event_id}: subsample labels do not match cached scores; "
            "the cache and the suite have diverged."
        )
    return np.asarray(x_sub, dtype=np.float64)


def _bootstrap_ci(
    deltas: list[float], resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float, float]:
    """Paired percentile bootstrap over per-event deltas: (mean, ci_low, ci_high)."""
    arr = np.asarray(deltas, dtype=np.float64)
    rng = np.random.RandomState(seed)
    n = arr.size
    means = np.empty(resamples, dtype=np.float64)
    for b in range(resamples):
        means[b] = arr[rng.randint(0, n, size=n)].mean()
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _domain_mean(rows: list[dict[str, Any]], key: str) -> float:
    xs = [r[key] for r in rows if isinstance(r.get(key), float) and not np.isnan(r[key])]
    return float(np.mean(xs)) if xs else float("nan")


def _measure_group(kind: str) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Run the full protocol over one event class (``real`` or ``reconstructed``)."""
    events = {(e.domain, e.event_id): e for e in build_suite(kind=kind)}
    scored = load_scores(kind=kind)

    rows: list[dict[str, Any]] = []
    by_dom: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for es in scored:
        ev = events[(es.domain, es.event_id)]
        cal_idx, ev_idx = _split(es.y, seed=SEED)
        y_cal, y_ev = es.y[cal_idx], es.y[ev_idx]
        if np.sum(y_cal == 1) < 1 or np.sum(y_ev == 1) < 1:
            continue

        comp = es.components()  # (n, 3): resonance, kinematic, info_geometry
        feats = _aligned_features(ev, es)
        temporal = temporal_innovation_score(feats)
        knn = knn_density_score(feats)

        pool4_t = np.column_stack([comp, temporal])
        pool4_k = np.column_stack([comp, knn])
        pool5 = np.column_stack([comp, temporal, knn])

        comp_ev = comp[ev_idx]
        singles = [_safe_auc(y_ev, comp_ev[:, j]) for j in range(3)]
        best_single = float(np.nanmax(singles))

        row: dict[str, Any] = {
            "domain": es.domain,
            "event": es.event_id,
            "rho_bar_3": _mean_abs_pairwise_spearman(comp),
            "rho_bar_4_temporal": _mean_abs_pairwise_spearman(pool4_t),
            "rho_temporal_vs_3": _mean_abs_spearman_against(temporal, comp),
            "rho_knn_vs_3": _mean_abs_spearman_against(knn, comp),
            "single_temporal": _safe_auc(y_ev, temporal[ev_idx]),
            "single_knn": _safe_auc(y_ev, knn[ev_idx]),
            "best_single": best_single,
            "best_single_name": COMPONENTS[int(np.nanargmax(singles))],
            "stack3": _stack_auroc(comp, es.y, cal_idx, ev_idx),
            "stack4_temporal": _stack_auroc(pool4_t, es.y, cal_idx, ev_idx),
            "stack4_knn": _stack_auroc(pool4_k, es.y, cal_idx, ev_idx),
            "stack5": _stack_auroc(pool5, es.y, cal_idx, ev_idx),
        }
        row["delta_temporal"] = float(row["stack4_temporal"]) - best_single
        rows.append(row)
        by_dom[es.domain].append(row)

    return rows, by_dom


def main() -> None:
    """Execute the decorrelated-stream protocol and write the committed verdict.

    The **live headline suite** is the decision basis (every FINDINGS figure is
    computed on it).  The **reconstructed-from-live** group is replicated and
    reported separately as labelled corroboration / added power -- never folded
    into the headline verdict, exactly as ``measure_baseline.py`` keeps them apart.
    """
    real_rows, real_by_dom = _measure_group("real")
    recon_rows, recon_by_dom = _measure_group("reconstructed")
    real_summary = _report("LIVE HEADLINE SUITE", real_rows, real_by_dom, headline=True)
    recon_summary = _report(
        "RECONSTRUCTED-FROM-LIVE (labelled; not in headline)",
        recon_rows,
        recon_by_dom,
        headline=False,
    )
    combined = _combined_power_check(real_rows, recon_rows)
    _write_artifact(real_summary, recon_summary, combined)


def _report(
    title: str,
    rows: list[dict[str, Any]],
    by_dom: dict[str, list[dict[str, Any]]],
    *,
    headline: bool,
) -> dict[str, Any]:
    """Print the protocol tables, decide SHIP/KILL, and return the summary dict."""
    metric_keys = (
        "rho_bar_3",
        "rho_temporal_vs_3",
        "best_single",
        "stack3",
        "stack4_temporal",
        "stack4_knn",
    )
    print(f"\n==== DECORRELATED-STREAM FUSION PROTOCOL -- {title} ({len(rows)} events) ====")
    header = f"{'domain':<16}" + "".join(f"{k.replace('stack', 'stk_'):>18}" for k in metric_keys)
    print(header)
    for dom in sorted(by_dom):
        vals = "".join(f"{_domain_mean(by_dom[dom], k):>18.3f}" for k in metric_keys)
        print(f"{dom:<16}{vals}")
    print("-" * len(header))
    overall = {k: _domain_mean(rows, k) for k in metric_keys}
    print(f"{'OVERALL (mean)':<16}" + "".join(f"{overall[k]:>18.3f}" for k in metric_keys))

    rho_bar_3 = _domain_mean(rows, "rho_bar_3")
    rho_bar_4 = _domain_mean(rows, "rho_bar_4_temporal")
    rho_t_vs_3 = _domain_mean(rows, "rho_temporal_vs_3")
    rho_k_vs_3 = _domain_mean(rows, "rho_knn_vs_3")
    best_single = _domain_mean(rows, "best_single")
    stack4_t = _domain_mean(rows, "stack4_temporal")
    stack4_k = _domain_mean(rows, "stack4_knn")

    deltas = [r["delta_temporal"] for r in rows]
    mean_delta, ci_low, ci_high = _bootstrap_ci(deltas)
    regressions = {
        dom: round(
            _domain_mean(by_dom[dom], "stack4_temporal") - _domain_mean(by_dom[dom], "best_single"),
            3,
        )
        for dom in sorted(by_dom)
        if _domain_mean(by_dom[dom], "stack4_temporal")
        < _domain_mean(by_dom[dom], "best_single") + DOMAIN_REGRESSION
    }

    ship = mean_delta >= SHIP_DELTA and ci_low > 0.0 and not regressions
    kill = (rho_bar_4 >= KILL_RHO) or (ci_high < SHIP_DELTA)
    if ship:
        verdict = "SHIP: decorrelated stacking beats best-single (paired bootstrap clears the bar)"
    elif kill:
        verdict = "KILL: decorrelation does not let fusion beat best-single on this suite"
    else:
        verdict = "INCONCLUSIVE: neither SHIP nor KILL criteria met"

    print("\n--- Step 1: redundancy diagnosis ---")
    print(
        f"mean |rho_bar| (3 committed streams)      = {rho_bar_3:.3f}  (pre-reg prediction: >~0.6)"
    )
    print("\n--- Step 2: decorrelated streams vs the 3 ---")
    print(f"temporal AR-innovation  mean |rho| vs 3   = {rho_t_vs_3:.3f}")
    print(f"kNN local-density       mean |rho| vs 3   = {rho_k_vs_3:.3f}")
    print(f"mean |rho_bar| (4 streams incl temporal)  = {rho_bar_4:.3f}  (KILL if >= {KILL_RHO})")
    print("\n--- Step 3/4: stacked fusion vs best-single ---")
    print(f"best-single                 = {best_single:.4f}")
    print(f"stack4 (temporal)           = {stack4_t:.4f}")
    print(f"stack4 (kNN, sensitivity)   = {stack4_k:.4f}")
    print(
        f"paired Delta (stack4_temporal - best_single): mean {mean_delta:+.4f}  "
        f"95% CI [{ci_low:+.4f}, {ci_high:+.4f}]  (SHIP needs mean>=+{SHIP_DELTA}, CI_low>0)"
    )
    if regressions:
        print(f"per-domain regressions (< {DOMAIN_REGRESSION}): {regressions}")
    print(f"\nVERDICT ({'headline' if headline else 'corroboration'}) -> {verdict}")

    return {
        "n_events": len(rows),
        "cap": CAP,
        "headline": headline,
        "step1_redundancy": {"mean_abs_pairwise_spearman_3": rho_bar_3},
        "step2_decorrelation": {
            "temporal_vs_3_mean_abs_spearman": rho_t_vs_3,
            "knn_vs_3_mean_abs_spearman": rho_k_vs_3,
            "mean_abs_pairwise_spearman_4_temporal": rho_bar_4,
        },
        "step3_stacking": {
            "best_single": best_single,
            "stack3": _domain_mean(rows, "stack3"),
            "stack4_temporal": stack4_t,
            "stack4_knn": stack4_k,
            "stack5": _domain_mean(rows, "stack5"),
        },
        "step4_decision": {
            "paired_delta_mean": mean_delta,
            "paired_delta_ci95": [ci_low, ci_high],
            "ship_delta_threshold": SHIP_DELTA,
            "kill_rho_threshold": KILL_RHO,
            "per_domain_regressions": regressions,
            "verdict": verdict,
        },
        "per_event": rows,
        "per_domain": {
            dom: {k: _domain_mean(by_dom[dom], k) for k in metric_keys} for dom in sorted(by_dom)
        },
    }


def _combined_power_check(
    real_rows: list[dict[str, Any]], recon_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Pool every reachable + reconstructed event into one paired bootstrap.

    A power-sensitivity check on ``stack4_temporal - best_single`` over the
    combined set.

    The live headline suite (20 events) lacks the power to *exclude* a small
    positive (its 95% CI upper bound sits just above the +0.01 ship line), so the
    literal protocol lands it ``INCONCLUSIVE``.  Pooling all 27 reachable +
    reconstructed events tightens the interval and resolves whether that residual
    inconclusiveness hides a real effect.  This is explicitly a labelled
    sensitivity check -- it is **not** the headline mean (which keeps the two
    groups separate per the suite's provenance discipline).
    """
    deltas = [r["delta_temporal"] for r in (*real_rows, *recon_rows)]
    mean_delta, ci_low, ci_high = _bootstrap_ci(deltas)
    kill = ci_high < SHIP_DELTA
    verdict = (
        "KILL (combined power): even pooling all reachable + reconstructed evidence, "
        "the 95% CI upper bound cannot reach the +0.01 ship line"
        if kill
        else "still underpowered: combined CI upper bound remains >= +0.01"
    )
    print("\n==== COMBINED POWER SENSITIVITY (live + reconstructed; not headline) ====")
    print(f"n_events = {len(deltas)}")
    print(
        f"paired Delta (stack4_temporal - best_single): mean {mean_delta:+.4f}  "
        f"95% CI [{ci_low:+.4f}, {ci_high:+.4f}]"
    )
    print(f"VERDICT (combined power) -> {verdict}")
    return {
        "n_events": len(deltas),
        "paired_delta_mean": mean_delta,
        "paired_delta_ci95": [ci_low, ci_high],
        "ship_delta_threshold": SHIP_DELTA,
        "verdict": verdict,
    }


def _write_artifact(
    real_summary: dict[str, Any],
    recon_summary: dict[str, Any],
    combined: dict[str, Any],
) -> None:
    """Write the committed decorrelation artifact (headline + corroboration)."""
    out = {
        "protocol": "FINDINGS.md decorrelated-stream fusion",
        "cap": CAP,
        "ship_delta_threshold": SHIP_DELTA,
        "kill_rho_threshold": KILL_RHO,
        "live_headline": real_summary,
        "reconstructed_corroboration": recon_summary,
        "combined_power_sensitivity": combined,
    }
    os.makedirs(_OUT_DIR, exist_ok=True)
    out_path = os.path.join(_OUT_DIR, "decorrelation_results.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
        fh.write("\n")  # trailing newline: byte-stable + end-of-file-fixer clean
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
