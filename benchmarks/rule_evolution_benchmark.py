# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Genetic rule evolution vs the hand-written consensus graph, on held-out test F1.

Protocol (pre-registered; the dataset list below is fixed *before* any
test-split evaluation and is not amended afterwards):

1. For each dataset: three-way split (``split_three_way``, stratified,
   train 40% / val 20% / test 40%).  The engine's base detectors are fit on
   the **train split only**; the per-sample detector-score matrices
   (``OmniMercuryEngine._extract_consensus_scores`` -- the exact channels the
   symbolic layer reasons over) are extracted for train/val/test over the
   channel intersection shared by all datasets.
2. Evolution sees train (unsupervised channel statistics) and val (fitness =
   mean val F1 across all datasets, threshold fit on val only) -- the test
   split is structurally out of reach (``FitnessDataset`` has no test fields).
   The hand-written consensus graph is a seed individual, so the search can
   only be selected over it by genuinely out-scoring it on validation data.
3. The test split is touched **exactly once**, at the end: for both the
   baseline (``consensus_rule_graph``) and the evolved champion, samples are
   scored through the *same deployed path*
   (``SymbolicConstraintModule.score_samples``), the operating threshold is
   fit on val, and F1 is reported on test (plus threshold-free AUC-ROC for
   context).
4. The evolved graph is committed as a schema-versioned artifact loadable via
   ``resolve_rule_graph("evolved:<path>")``; the benchmark re-loads it and
   asserts the served scores reproduce the in-memory result (serve-path
   check).

Usage::

    PYTHONPATH=src python benchmarks/rule_evolution_benchmark.py \
        --population 40 --generations 30 \
        --out benchmarks/rule_evolution_results.json \
        --graph-out benchmarks/evolved_rule_graph.json
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# Pre-registered evaluation datasets (real ADBench tabular anomaly benchmarks
# with binary ground-truth labels).  Fixed before any test-split evaluation:
# the three symbolic_rulegraph_sweep defaults plus Pima for a fourth,
# higher-contamination regime.  Do not edit after seeing test results.
PREREGISTERED_DATASETS = ["cardio", "thyroid", "WBC", "Pima"]

VAL_FRAC = 0.2
TEST_FRAC = 0.4


def _git_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
                cwd=Path(__file__).resolve().parent,
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def _load_dataset(name: str) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    from omni_mercury_engine.datasets.adbench import ADBenchLoader
    from omni_mercury_engine.datasets.base import DatasetConfig

    loader = ADBenchLoader(DatasetConfig(name="adbench", preprocessing={"dataset": name}))
    loader.download()
    features, labels = loader.load()
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels).astype(int).ravel(),
    )


def main() -> int:
    warnings.filterwarnings("ignore")
    logging.getLogger("omni_mercury_engine").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=PREREGISTERED_DATASETS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--penalty", type=float, default=1e-4)
    parser.add_argument("--out", default="benchmarks/rule_evolution_results.json")
    parser.add_argument("--graph-out", default="benchmarks/evolved_rule_graph.json")
    args = parser.parse_args()

    import torch

    from omni_mercury_engine.engine import OmniMercuryEngine
    from omni_mercury_engine.evaluation.metrics import (
        compute_auc_roc,
        compute_f1,
        fit_threshold,
        split_three_way,
    )
    from omni_mercury_engine.ml.rule_evolution import (
        ChannelStats,
        EvolvedRuleSearch,
        FitnessDataset,
        RuleFitnessEvaluator,
        genome_from_rule_graph,
        save_evolved_rule_graph,
    )
    from omni_mercury_engine.ml.symbolic_constraint import (
        SymbolicConstraintModule,
        consensus_rule_graph,
        resolve_rule_graph,
    )

    total_t0 = time.time()
    print("Rule-evolution benchmark: consensus baseline vs evolved rule graph")
    print(
        f"datasets={args.datasets} seed={args.seed} population={args.population} "
        f"generations={args.generations} patience={args.patience}"
    )
    print("-" * 88)

    # -- Phase 1: real detector-score channels, detectors fit on train only ----
    prepared: list[dict[str, Any]] = []
    for name in args.datasets:
        t0 = time.time()
        try:
            features, labels = _load_dataset(name)
        except Exception as exc:
            print(f"  {name:<10} SKIP (load failed: {exc})")
            continue
        train_idx, val_idx, test_idx = split_three_way(
            len(labels),
            labels,
            val_frac=VAL_FRAC,
            test_frac=TEST_FRAC,
            random_state=args.seed,
        )
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        engine = OmniMercuryEngine(mode="fusion", device="cpu")
        engine._extract_fusion_features(features[train_idx], fit_detectors=True)
        _, channels = engine._extract_consensus_scores(features[train_idx], return_channels=True)
        prepared.append(
            {
                "name": name,
                "engine": engine,
                "X": features,
                "y": labels,
                "train": train_idx,
                "val": val_idx,
                "test": test_idx,
                "channels": channels,
                "wall_extract_s": time.time() - t0,
            }
        )
        print(
            f"  {name:<10} n={len(labels):<6} channels={len(channels):<3} "
            f"train/val/test={len(train_idx)}/{len(val_idx)}/{len(test_idx)} "
            f"({time.time() - t0:.1f}s)"
        )

    if len(prepared) < 3:
        print("INTEGRITY FAILURE: fewer than 3 datasets prepared (network unavailable?).")
        return 1

    common_channels = sorted(set.intersection(*(set(d["channels"]) for d in prepared)))
    if len(common_channels) < 2:
        print(f"INTEGRITY FAILURE: common channel set too small: {common_channels}")
        return 1
    print(f"common channels ({len(common_channels)}): {common_channels}")

    for entry in prepared:
        matrices = {}
        for split in ("train", "val", "test"):
            scores, got = entry["engine"]._extract_consensus_scores(
                entry["X"][entry[split]], channels=common_channels, return_channels=True
            )
            if got != common_channels:
                print(
                    f"INTEGRITY FAILURE: {entry['name']} {split} split lost channels "
                    f"({got} != {common_channels})"
                )
                return 1
            matrices[split] = scores.numpy()
        entry["scores"] = matrices

    # -- Phase 2: evolution on train (stats) + val (fitness) only --------------
    stats = ChannelStats.from_train_scores([e["scores"]["train"] for e in prepared])
    fitness_datasets = [
        FitnessDataset(
            name=e["name"],
            scores_train=e["scores"]["train"],
            y_train=e["y"][e["train"]],
            scores_val=e["scores"]["val"],
            y_val=e["y"][e["val"]],
        )
        for e in prepared
    ]
    evaluator = RuleFitnessEvaluator(fitness_datasets, complexity_penalty=args.penalty)
    baseline_genome = genome_from_rule_graph(consensus_rule_graph())
    baseline_val = evaluator.evaluate(baseline_genome)
    print(
        f"baseline (consensus) val fitness={baseline_val.fitness:.4f} "
        f"mean_val_f1={baseline_val.mean_val_f1:.4f}"
    )

    t0 = time.time()
    search = EvolvedRuleSearch(
        evaluator,
        stats,
        population_size=args.population,
        generations=args.generations,
        patience=args.patience,
        seed=args.seed,
        seed_genomes=[baseline_genome],
    )
    result = search.run()
    wall_evolution = time.time() - t0
    print(
        f"evolution: {result.generations_run} generations "
        f"(early_stop={result.stopped_early}) in {wall_evolution:.1f}s; "
        f"best val fitness={result.best_report.fitness:.4f} "
        f"mean_val_f1={result.best_report.mean_val_f1:.4f} "
        f"complexity={result.best_report.complexity}"
    )
    evolved_graph = result.best_genome.to_rule_graph(name="evolved_consensus_v1")
    for rule in evolved_graph.rules:
        print(f"    {rule.name}: {rule.antecedent} -> {rule.consequent}")

    # -- Persist the evolved artifact (provenance included) --------------------
    history_json = [
        {
            "generation": record.generation,
            "best_fitness": record.best_fitness,
            "mean_fitness": record.mean_fitness,
            "best_val_f1": record.best_val_f1,
        }
        for record in result.history
    ]
    provenance = {
        "datasets": [e["name"] for e in prepared],
        "seed": args.seed,
        "commit": _git_commit(),
        "created_utc": datetime.now(UTC).isoformat(),
        "population": args.population,
        "generations_budget": args.generations,
        "generations_run": result.generations_run,
        "patience": args.patience,
        "complexity_penalty": args.penalty,
        "val_fitness_history": history_json,
        "baseline_val_fitness": baseline_val.fitness,
        "best_val_fitness": result.best_report.fitness,
        "channel_semantics": "OmniMercuryEngine._extract_consensus_scores channels",
    }
    graph_path = Path(args.graph_out)
    save_evolved_rule_graph(
        graph_path,
        result.best_genome,
        graph_name="evolved_consensus_v1",
        num_channels=len(common_channels),
        channel_names=common_channels,
        provenance=provenance,
    )
    print(f"evolved graph artifact -> {graph_path}")

    # Serve-path check: the committed artifact must reproduce the in-memory
    # graph through the deployment seam.
    served_graph = resolve_rule_graph(f"evolved:{graph_path}")
    if served_graph != evolved_graph:
        print("INTEGRITY FAILURE: artifact round-trip changed the evolved graph.")
        return 1

    # -- Phase 3: held-out TEST evaluation (touched exactly once) --------------
    num_channels = len(common_channels)
    arms = {"baseline_consensus": consensus_rule_graph(), "evolved": served_graph}
    rows: list[dict[str, Any]] = []
    print("-" * 88)
    print(f"  {'dataset':<12} {'baseline F1':>12} {'evolved F1':>12} {'dF1':>8}   AUC b/e")
    for entry in prepared:
        row: dict[str, Any] = {"dataset": entry["name"]}
        for arm, graph in arms.items():
            module = SymbolicConstraintModule(num_detectors=num_channels, rule_graph=graph)
            s_val = module.score_samples(
                torch.as_tensor(entry["scores"]["val"], dtype=torch.float32)
            ).numpy()
            s_test = module.score_samples(
                torch.as_tensor(entry["scores"]["test"], dtype=torch.float32)
            ).numpy()
            y_val = entry["y"][entry["val"]]
            y_test = entry["y"][entry["test"]]
            threshold = fit_threshold(y_val, s_val)
            row[f"{arm}_f1"] = float(compute_f1(y_test, (s_test >= threshold).astype(int)))
            row[f"{arm}_auc"] = float(compute_auc_roc(y_test, s_test))
            row[f"{arm}_threshold"] = float(threshold)
        row["delta_f1"] = row["evolved_f1"] - row["baseline_consensus_f1"]
        rows.append(row)
        print(
            f"  {row['dataset']:<12} {row['baseline_consensus_f1']:>12.4f} "
            f"{row['evolved_f1']:>12.4f} {row['delta_f1']:>+8.4f}   "
            f"{row['baseline_consensus_auc']:.3f}/{row['evolved_auc']:.3f}"
        )

    mean_baseline = float(np.mean([r["baseline_consensus_f1"] for r in rows]))
    mean_evolved = float(np.mean([r["evolved_f1"] for r in rows]))
    margin = mean_evolved - mean_baseline
    evolved_wins = bool(margin > 0.0)
    verdict = (
        f"EVOLVED graph beats the consensus baseline on held-out test F1 "
        f"(mean {mean_evolved:.4f} vs {mean_baseline:.4f}, margin {margin:+.4f})"
        if evolved_wins
        else (
            f"evolved graph does NOT beat the consensus baseline on held-out test F1 "
            f"(mean {mean_evolved:.4f} vs {mean_baseline:.4f}, margin {margin:+.4f})"
        )
    )

    report = {
        "preregistered_datasets": PREREGISTERED_DATASETS,
        "datasets_evaluated": [e["name"] for e in prepared],
        "protocol": {
            "split": {"val_frac": VAL_FRAC, "test_frac": TEST_FRAC, "stratified": True},
            "fitness": "mean val F1 across datasets, threshold fit on val only",
            "scoring_path": "SymbolicConstraintModule.score_samples (fitness == deployment)",
            "test_policy": "test split scored exactly once, after the search finished",
        },
        "budget": {
            "population": args.population,
            "generations_budget": args.generations,
            "generations_run": result.generations_run,
            "patience": args.patience,
            "early_stop": result.stopped_early,
        },
        "seed": args.seed,
        "complexity_penalty": args.penalty,
        "common_channels": common_channels,
        "baseline_val_fitness": baseline_val.fitness,
        "evolved_val_fitness": result.best_report.fitness,
        "val_fitness_history": history_json,
        "evolved_rules": [
            {"name": r.name, "antecedent": r.antecedent, "consequent": r.consequent}
            for r in evolved_graph.rules
        ],
        "test_results": rows,
        "mean_test_f1": {"baseline_consensus": mean_baseline, "evolved": mean_evolved},
        "margin": margin,
        "evolved_beats_baseline": evolved_wins,
        "verdict": verdict,
        "wall_times_s": {
            "extraction_per_dataset": {e["name"]: round(e["wall_extract_s"], 2) for e in prepared},
            "evolution": round(wall_evolution, 2),
            "total": round(time.time() - total_t0, 2),
        },
        "provenance": {
            "commit": _git_commit(),
            "created_utc": datetime.now(UTC).isoformat(),
            "graph_artifact": str(graph_path),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print("-" * 88)
    print(f"  mean test F1: baseline={mean_baseline:.4f}  evolved={mean_evolved:.4f}")
    print(f"VERDICT: {verdict}")
    print(f"report -> {out_path}")
    print(f"total wall time: {time.time() - total_t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
