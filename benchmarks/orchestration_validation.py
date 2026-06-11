# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Orchestration validation: does the revived planning / coordination / reflexion / chain-of-thought tier carry *real* signal on the engine's own task?

This is the measurement harness the Dormancy & Salvage Ledger (rows 10-11)
was waiting for. The reasoning/planning/coordination tier was retained as
"reference only" because it does not speak the anomaly-AUC metric — control
and meta machinery must be measured against the *right* metric, on a *real*
task, or not at all. The real task now exists: the
``MultiAgentOrchestrator`` (``agentic/orchestration.py``) runs these modules
over the live detector ensemble on real ADBench labels. Four pre-registered
questions, one per revived module, each on its own honest metric:

1. **Coordination** (``multi_agent_coordination``): per-sample
   confidence-weighted consensus across the five real detectors. Bar:
   mean held-out consensus AUC >= mean member AUC - 0.005 (coordination must
   not destroy member signal; we explicitly do NOT claim it beats the
   trained ``OmniFusionModel`` — that comparison belongs to the
   ensemble-marginal ablation discipline and is reported as context only).

2. **Reflexion** (``reflexion``): sequential-batch episodes with real label
   feedback; the critic's threshold recommendations adapt the operating
   point. Paired arms on identical batches and identical fitted agents:
   FIXED (threshold pinned at the 0.5 default) vs ADAPTIVE (reflexion
   applied between batches). Bar: mean paired delta in balanced accuracy
   over post-adaptation batches >= -0.002 (never hurts beyond the noise
   floor), and where reflexion actually acted, the mean delta > 0.

3. **Planning** (``hierarchical_planning``): the planner must *drive* every
   episode — Bar: 100% of episodes execute the full planned stage sequence
   to goal completion, and the TD value of the initial pipeline state is
   non-decreasing across episodes with final > first (real value learning
   from real stage rewards).

4. **Trace fidelity** (``chain_of_thought``): every sampled decision's
   reasoning trace must state the same determination the pipeline issued,
   and quote the real consensus score. Bar: fidelity rate == 1.0 and
   numeric-quote rate == 1.0.

Ablation integrity (non-negotiable): real held-out ADBench labels only; if
the data cannot be downloaded the run reports that and exits non-zero — it
never fabricates a pass.

Usage::

    python -m benchmarks.orchestration_validation
    python -m benchmarks.orchestration_validation \\
        --datasets cardio thyroid breastw WBC Pima --seeds 0 1 2 \\
        --out artifacts/orchestration_validation.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import warnings
from itertools import pairwise
from typing import Any

import numpy as np

DEFAULT_DATASETS = ["cardio", "thyroid", "breastw", "WBC", "Pima"]
DEFAULT_SEEDS = [0, 1, 2]
N_FEEDBACK_BATCHES = 4
N_FIDELITY_SAMPLES = 40

# Pre-registered bars (see module docstring).
_COORDINATION_TOLERANCE = 0.005
_REFLEXION_NOISE_FLOOR = 0.002


def _load_dataset(name: str) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    from omni_mercury_engine.datasets.adbench import ADBenchLoader
    from omni_mercury_engine.datasets.base import DatasetConfig

    loader = ADBenchLoader(DatasetConfig(name="adbench", preprocessing={"dataset": name}))
    loader.download()
    data = loader.load()
    return np.asarray(data[0], dtype=np.float32), np.asarray(data[1]).astype(int).ravel()


def _stratified(
    y: np.ndarray[Any, Any], frac: float, rng: np.random.RandomState
) -> np.ndarray[Any, Any]:
    keep: list[int] = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        keep.extend(idx[: max(1, int(len(idx) * frac))].tolist())
    return np.array(sorted(keep))


def _stratified_batches(
    y: np.ndarray[Any, Any], n_batches: int, rng: np.random.RandomState
) -> list[np.ndarray[Any, Any]]:
    """Round-robin each class across batches so every batch sees both."""
    assignments: list[list[int]] = [[] for _ in range(n_batches)]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for j, sample in enumerate(idx):
            assignments[j % n_batches].append(int(sample))
    return [np.array(sorted(b)) for b in assignments if len(b) > 0]


def _balanced_accuracy(pred: np.ndarray[Any, Any], truth: np.ndarray[Any, Any]) -> float:
    tp = float(np.sum(pred & truth))
    tn = float(np.sum(~pred & ~truth))
    fp = float(np.sum(pred & ~truth))
    fn = float(np.sum(~pred & truth))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return (tpr + tnr) / 2.0


_SCORE_QUOTE = re.compile(r"score: ([0-9]*\.?[0-9]+)")


def run_dataset_seed(name: str, seed: int) -> dict[str, Any] | None:
    """One full validation pass for (dataset, seed). None if unmeasurable."""
    from omni_mercury_engine.agentic.orchestration import (
        MultiAgentOrchestrator,
        OrchestrationError,
    )
    from omni_mercury_engine.ml.mercury_ml import roc_auc_score

    # Pin the global RNGs per run: some live detectors carry stochastic
    # components that follow the *global* seed (e.g. DimensionalAnalyzer's
    # autoencoder lane), so without this the grid is honest but not
    # bit-reproducible run-to-run.
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass

    X, y = _load_dataset(name)
    rng = np.random.RandomState(seed)
    te = _stratified(y, 0.3, rng)
    mask = np.zeros(len(y), dtype=bool)
    mask[te] = True
    tr = np.where(~mask)[0]
    if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
        return None
    mu = X[tr].mean(0)
    sd = X[tr].std(0)
    sd[sd < 1e-8] = 1.0
    X_tr, X_te = (X[tr] - mu) / sd, (X[te] - mu) / sd
    y_te = y[te].astype(bool)

    orch = MultiAgentOrchestrator(seed=seed).fit(np.asarray(X_tr, dtype=np.float64))

    plan_episodes_ok = 0
    plan_episodes_total = 0
    goal_values: list[float] = []

    def _track_plan(episode: Any) -> None:
        nonlocal plan_episodes_ok, plan_episodes_total
        plan_episodes_total += 1
        completed = episode.plan.goal_status == "completed" and (
            episode.plan.executed_actions
            == ["score_agents", "form_consensus", "issue_decisions"]
        )
        plan_episodes_ok += int(completed)
        goal_values.append(float(episode.plan.goal_value))

    # ---- Q1: coordination on the full held-out split --------------------
    episode = orch.detect(np.asarray(X_te, dtype=np.float64))
    _track_plan(episode)
    batch = episode.coordination
    consensus_auc = float(roc_auc_score(y[te], batch.consensus_scores))
    member_aucs = {
        agent: float(roc_auc_score(y[te], scores))
        for agent, scores in batch.per_agent_scores.items()
    }

    # ---- Q4: trace fidelity on boundary-adjacent + extreme samples ------
    order = np.argsort(np.abs(batch.consensus_scores - episode.threshold))
    sample_indices = list(order[: N_FIDELITY_SAMPLES - 2])
    sample_indices += [int(np.argmax(batch.consensus_scores)), int(np.argmin(batch.consensus_scores))]
    fidelity_ok = 0
    numeric_ok = 0
    fidelity_total = 0
    for index in sample_indices:
        fidelity_total += 1
        try:
            trace = orch.explain(episode, int(index))
        except OrchestrationError:
            continue  # counted as a fidelity failure
        fidelity_ok += 1
        quotes = _SCORE_QUOTE.findall(str(trace["conclusion"]))
        actual = round(float(batch.consensus_scores[index]), 2)
        if quotes and abs(float(quotes[-1]) - actual) < 0.005:
            numeric_ok += 1

    # ---- Q2: paired FIXED vs ADAPTIVE arms over sequential batches ------
    batches = _stratified_batches(y[te], N_FEEDBACK_BATCHES, np.random.RandomState(seed + 1))
    usable = [b for b in batches if len(np.unique(y[te][b])) == 2]
    fixed_acc: list[float] = []
    adaptive_acc: list[float] = []
    recommendations: list[str] = []
    if len(usable) >= 2:
        # FIXED arm: threshold pinned at the untouched default; no critic.
        orch.set_operating_threshold(0.5)
        for k, b in enumerate(usable):
            ep = orch.detect(np.asarray(X_te[b], dtype=np.float64))
            _track_plan(ep)
            if k >= 1:
                decided = ~ep.coordination.abstained
                fixed_acc.append(
                    _balanced_accuracy(ep.coordination.decisions[decided], y_te[b][decided])
                )
        # ADAPTIVE arm: same agents, fresh critic, reflexion applied.
        orch.set_operating_threshold(0.5)
        orch.reset_reflexion()
        for k, b in enumerate(usable):
            ep = orch.run_episode(
                np.asarray(X_te[b], dtype=np.float64), y_te[b], apply_reflection=True
            )
            _track_plan(ep)
            if ep.reflection is not None:
                recommendations.append(ep.reflection.recommendation)
            if k >= 1:
                decided = ~ep.coordination.abstained
                adaptive_acc.append(
                    _balanced_accuracy(ep.coordination.decisions[decided], y_te[b][decided])
                )

    values_non_decreasing = all(b >= a - 1e-12 for a, b in pairwise(goal_values))

    return {
        "dataset": name,
        "seed": seed,
        "n_train": len(tr),
        "n_test": len(te),
        "consensus_auc": consensus_auc,
        "member_aucs": member_aucs,
        "mean_member_auc": float(np.mean(list(member_aucs.values()))),
        "best_member_auc": float(np.max(list(member_aucs.values()))),
        "fixed_balanced_acc": float(np.mean(fixed_acc)) if fixed_acc else float("nan"),
        "adaptive_balanced_acc": float(np.mean(adaptive_acc)) if adaptive_acc else float("nan"),
        "reflexion_acted": any(r != "maintain" for r in recommendations),
        "reflexion_recommendations": recommendations,
        "plan_episodes_ok": plan_episodes_ok,
        "plan_episodes_total": plan_episodes_total,
        "goal_value_first": goal_values[0] if goal_values else float("nan"),
        "goal_value_final": goal_values[-1] if goal_values else float("nan"),
        "goal_values_non_decreasing": bool(values_non_decreasing),
        "fidelity_ok": fidelity_ok,
        "fidelity_total": fidelity_total,
        "numeric_quote_ok": numeric_ok,
    }


def derive_verdict(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the four pre-registered bars to the measured results."""
    consensus = [r["consensus_auc"] for r in results]
    mean_members = [r["mean_member_auc"] for r in results]
    coordination_passed = bool(
        np.mean(consensus) >= np.mean(mean_members) - _COORDINATION_TOLERANCE
    )

    paired = [
        (r["adaptive_balanced_acc"], r["fixed_balanced_acc"], r["reflexion_acted"])
        for r in results
        if not (np.isnan(r["adaptive_balanced_acc"]) or np.isnan(r["fixed_balanced_acc"]))
    ]
    deltas_all = [a - f for a, f, _ in paired]
    deltas_acted = [a - f for a, f, acted in paired if acted]
    reflexion_delta_all = float(np.mean(deltas_all)) if deltas_all else float("nan")
    reflexion_delta_acted = float(np.mean(deltas_acted)) if deltas_acted else float("nan")
    reflexion_passed = bool(
        deltas_all
        and reflexion_delta_all >= -_REFLEXION_NOISE_FLOOR
        and (not deltas_acted or reflexion_delta_acted > 0.0)
    )

    executability = float(
        sum(r["plan_episodes_ok"] for r in results)
        / max(1, sum(r["plan_episodes_total"] for r in results))
    )
    value_learning = all(
        r["goal_values_non_decreasing"] and r["goal_value_final"] > r["goal_value_first"]
        for r in results
    )
    planning_passed = bool(executability == 1.0 and value_learning)

    fidelity_rate = float(
        sum(r["fidelity_ok"] for r in results) / max(1, sum(r["fidelity_total"] for r in results))
    )
    numeric_rate = float(
        sum(r["numeric_quote_ok"] for r in results)
        / max(1, sum(r["fidelity_total"] for r in results))
    )
    fidelity_passed = bool(fidelity_rate == 1.0 and numeric_rate == 1.0)

    passed = coordination_passed and reflexion_passed and planning_passed and fidelity_passed
    return {
        "coordination": {
            "mean_consensus_auc": float(np.mean(consensus)),
            "mean_member_auc": float(np.mean(mean_members)),
            "mean_best_member_auc": float(np.mean([r["best_member_auc"] for r in results])),
            "passed": coordination_passed,
        },
        "reflexion": {
            "paired_runs": len(paired),
            "delta_all": reflexion_delta_all,
            "delta_when_acted": reflexion_delta_acted,
            "acted_runs": len(deltas_acted),
            "passed": reflexion_passed,
        },
        "planning": {
            "executability": executability,
            "value_learning": bool(value_learning),
            "passed": planning_passed,
        },
        "trace_fidelity": {
            "fidelity_rate": fidelity_rate,
            "numeric_quote_rate": numeric_rate,
            "passed": fidelity_passed,
        },
        "passed": passed,
        "verdict": (
            "VALIDATED -- the revived planning/coordination/reflexion/chain-of-thought "
            "tier runs the live ensemble with preserved signal, real adaptive feedback, "
            "fully-executed plans, and decision-faithful traces"
            if passed
            else "NOT VALIDATED -- one or more pre-registered bars failed; see sub-verdicts"
        ),
    }


def main() -> int:
    warnings.filterwarnings("ignore")
    logging.getLogger("omni_mercury_engine").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--out", default="artifacts/orchestration_validation.json", type=str)
    args = parser.parse_args()

    print("Orchestration validation (revived cognitive tier on real ADBench labels)")
    print(f"datasets={args.datasets}  seeds={args.seeds}")
    print("-" * 88)

    results: list[dict[str, Any]] = []
    failed_cells: list[str] = []
    for name in args.datasets:
        for seed in args.seeds:
            try:
                row = run_dataset_seed(name, seed)
            except Exception as exc:
                failed_cells.append(f"{name}/seed{seed}")
                print(f"  {name} seed={seed}: FAILED ({type(exc).__name__}: {exc})")
                continue
            if row is None:
                print(f"  {name} seed={seed}: skipped (degenerate split)")
                continue
            results.append(row)
            print(
                f"  {name:<9} seed={seed}  consensus_AUC={row['consensus_auc']:.3f} "
                f"members(mean/best)={row['mean_member_auc']:.3f}/{row['best_member_auc']:.3f}  "
                f"fixed/adaptive bal_acc={row['fixed_balanced_acc']:.3f}/"
                f"{row['adaptive_balanced_acc']:.3f}  "
                f"plans={row['plan_episodes_ok']}/{row['plan_episodes_total']}  "
                f"fidelity={row['fidelity_ok']}/{row['fidelity_total']}"
            )

    if not results:
        print("INTEGRITY FAILURE: nothing could be measured (network unavailable?).")
        return 1

    verdict = derive_verdict(results)
    print("-" * 88)
    for key in ("coordination", "reflexion", "planning", "trace_fidelity"):
        sub = verdict[key]
        status = "PASS" if sub["passed"] else "FAIL"
        detail = {k: v for k, v in sub.items() if k != "passed"}
        print(f"  {key:<15} {status}  {detail}")
    print(f"VERDICT: {verdict['verdict']}")

    from pathlib import Path

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "results": results,
        "verdict": verdict,
        "complete": not failed_cells,
        "failed_cells": failed_cells,
    }
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"report -> {out}")
    if failed_cells:
        # Fail-closed on a partial grid: bars measured on a subset must not
        # read as a clean pass, however the subset scored.
        print(f"INTEGRITY FAILURE: {len(failed_cells)} grid cell(s) unmeasured: {failed_cells}")
        return 1
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
