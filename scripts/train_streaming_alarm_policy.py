#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Train, evaluate and ship the streaming worker's learned alarm policy.

Data: the five real NAB categories (47 files, 110 labelled windows), from a
NAB checkout or downloaded at a pinned commit; every file's SHA-256 is
recorded. Score: the NAB standard profile (``evaluation.nab_scoring``, held
equal to NAB's own scorer).

Evaluation is leave-one-category-out: each category is scored by a policy
trained on the other four, for several PPO seeds. Baselines, on the same
folds: the streaming worker's EMA z-score rule, and a threshold plus
refractory period on each single feature, grid-tuned on the training folds
(reported both with the feature chosen by training score, which is fair, and
with the best feature chosen on the held-out folds, which flatters it). The
PPO settings and the candidate gate were chosen on the realAWSCloudwatch
fold, so the report also gives the four other folds alone.

The shipped policy is then trained on all 47 files (seed 0), exported as a
NumPy archive with a provenance sidecar pinning its SHA-256 and recording the
evaluation above.

Usage::

    python scripts/train_streaming_alarm_policy.py --download --ship
    python scripts/train_streaming_alarm_policy.py --nab-root ~/NAB --seeds 0 --steps 200000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omni_mercury_engine.evaluation.nab_scoring import (
    NAB_REAL_CATEGORIES,
    load_nab_corpus,
    score_corpus,
)
from omni_mercury_engine.ml.alarm_policy import (
    FEATURE_NAMES,
    SHIPPED_POLICY,
    default_training_config,
    export_alarm_policy,
    nab_features_and_tables,
    policy_from_trainer,
    run_alarm_policy,
    train_alarm_policy,
)

NAB_COMMIT = "ea702d75cc2258d9d7dd35ca8e5e2539d71f3140"
GATE = 0.962  # rank extremeness: 5% of points, 107 of 110 windows reachable
DEV_FOLD = "realAWSCloudwatch"
RESULTS = ROOT / "benchmarks" / "streaming_alarm_policy_results.json"


def download_nab(dest: Path) -> None:
    """Fetch the real NAB files and labels at the pinned commit."""
    from omni_mercury_engine.datasets.timeseries import NABLoader

    base = f"https://raw.githubusercontent.com/numenta/NAB/{NAB_COMMIT}/"
    paths = ["labels/combined_windows.json"]
    paths += [f"data/{cat}/{f}" for cat, files in NABLoader.NAB_FILES.items() for f in files]
    for rel in paths:
        out = dest / rel
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        url = base + rel
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https host
            out.write_bytes(response.read())


def corpus_hashes(root: Path) -> dict[str, str]:
    """SHA-256 of every NAB file read."""
    files = [root / "labels" / "combined_windows.json"]
    files += sorted((root / "data").glob("real*/*.csv"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def deployed_rule_alarms(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """The streaming worker's EMA z-score rule, one value per message."""
    from omni_mercury_engine.infrastructure.streaming import StreamingAnomalyPipeline

    pipeline = StreamingAnomalyPipeline("in", "out", detector=lambda d: d)
    return np.array([pipeline.zscore_detector({"value": float(v)})["is_anomaly"] for v in values])


def threshold_alarms(
    score: np.ndarray[Any, Any], threshold: float, refractory: int
) -> np.ndarray[Any, Any]:
    """Alarm above ``threshold``, then stay silent for ``refractory`` points."""
    alarms = np.zeros(score.size, dtype=bool)
    last = -(10**9)
    for t in np.flatnonzero(score > threshold):
        if t - last > refractory:
            alarms[t] = True
            last = int(t)
    return alarms


def tune_threshold(
    scores: dict[str, np.ndarray[Any, Any]], tables: dict[str, Any], names: list[str]
) -> tuple[float, float, int]:
    """Best (training NAB score, threshold, refractory) on a quantile grid."""
    pool = np.concatenate([scores[n] for n in names])
    best = (-np.inf, 0.0, 0)
    for q in np.linspace(0.90, 0.9999, 60):
        threshold = float(np.quantile(pool, q))
        for refractory in (0, 10, 50, 100, 200, 500):
            alarms = {n: threshold_alarms(scores[n], threshold, refractory) for n in names}
            score = score_corpus(alarms, {n: tables[n] for n in names})["score"]
            if score > best[0]:
                best = (score, threshold, refractory)
    return best


def _ppo_fold(job: tuple[str, int, int, str]) -> tuple[str, int, dict[str, list[int]], float]:
    """Train on every category but ``held_out``; return held-out alarms (worker process)."""
    import torch

    torch.set_num_threads(1)
    nab_root, seed, steps, held_out = job
    streams = load_nab_corpus(nab_root)
    features, tables = nab_features_and_tables(streams)
    train = {n: f for n, f in features.items() if not n.startswith(held_out)}
    started = time.perf_counter()
    trainer = train_alarm_policy(
        train,
        {n: tables[n] for n in train},
        gate=GATE,
        total_timesteps=steps,
        config=default_training_config(seed),
    )
    policy = policy_from_trainer(trainer, GATE)
    alarms = {
        n: np.flatnonzero(run_alarm_policy(policy, f)).tolist()
        for n, f in features.items()
        if n.startswith(held_out)
    }
    return held_out, seed, alarms, time.perf_counter() - started


def _as_alarms(
    indices: dict[str, list[int]], features: dict[str, np.ndarray[Any, Any]]
) -> dict[str, np.ndarray[Any, Any]]:
    out = {}
    for name, idx in indices.items():
        alarms = np.zeros(features[name].shape[0], dtype=bool)
        alarms[idx] = True
        out[name] = alarms
    return out


def _summary(alarms: dict[str, np.ndarray[Any, Any]], tables: dict[str, Any]) -> dict[str, Any]:
    clean = [n for n in alarms if not n.startswith(DEV_FOLD)]
    every = score_corpus(alarms, {n: tables[n] for n in alarms})
    four = score_corpus({n: alarms[n] for n in clean}, {n: tables[n] for n in clean})
    per_fold = {}
    for cat in NAB_REAL_CATEGORIES:
        names = [n for n in alarms if n.startswith(cat)]
        per_fold[cat] = round(
            score_corpus({n: alarms[n] for n in names}, {n: tables[n] for n in names})["score"], 2
        )
    return {
        "score_all_folds": round(every["score"], 2),
        "score_without_dev_fold": round(four["score"], 2),
        "per_fold": per_fold,
        "windows_detected": every["detected"],
        "windows_missed": every["missed"],
        "false_alarms": every["false_alarms"],
    }


def main() -> None:
    """Evaluate, then (optionally) train and ship the production policy."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--nab-root", type=Path, default=ROOT / "cache" / "nab_corpus")
    parser.add_argument("--download", action="store_true", help="fetch NAB at the pinned commit")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--ship", action="store_true", help="train on all files and write the policy"
    )
    args = parser.parse_args()
    if args.download:
        download_nab(args.nab_root)
    streams = load_nab_corpus(args.nab_root)
    features, tables = nab_features_and_tables(streams)
    names = [s.name for s in streams]
    print(f"{len(streams)} NAB files, {sum(f.shape[0] for f in features.values())} points")

    baselines: dict[str, dict[str, np.ndarray[Any, Any]]] = {"deployed_zscore_rule": {}}
    for s in streams:
        baselines["deployed_zscore_rule"][s.name] = deployed_rule_alarms(s.values)
    fair: dict[str, np.ndarray[Any, Any]] = {}
    tuned: dict[str, Any] = {}
    for cat in NAB_REAL_CATEGORIES:
        train = [n for n in names if not n.startswith(cat)]
        test = [n for n in names if n.startswith(cat)]
        best_train, best_feature = -np.inf, ""
        for j, feature in enumerate(FEATURE_NAMES):
            column = {n: features[n][:, j] for n in names}
            train_score, threshold, refractory = tune_threshold(column, tables, train)
            held = {n: threshold_alarms(column[n], threshold, refractory) for n in test}
            baselines.setdefault(f"threshold_{feature}", {}).update(held)
            tuned.setdefault(cat, {})[feature] = {
                "train_score": round(train_score, 2),
                "threshold": threshold,
                "refractory": refractory,
            }
            if train_score > best_train:
                best_train, best_feature = train_score, feature
        fair.update({n: baselines[f"threshold_{best_feature}"][n] for n in test})
        tuned[cat]["chosen_by_training_score"] = best_feature
        print(f"  baselines for {cat}: chose {best_feature}")
    baselines["threshold_feature_chosen_by_training_score"] = fair
    summaries = {k: _summary(v, tables) for k, v in baselines.items()}
    best_on_test = max(
        (k for k in summaries if k.startswith("threshold_") and "chosen" not in k),
        key=lambda k: summaries[k]["score_all_folds"],
    )

    jobs = [
        (str(args.nab_root), seed, args.steps, cat)
        for seed in args.seeds
        for cat in NAB_REAL_CATEGORIES
    ]
    ppo: dict[int, dict[str, list[int]]] = {seed: {} for seed in args.seeds}
    train_seconds: list[float] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for held_out, seed, alarms, seconds in pool.map(_ppo_fold, jobs):
            ppo[seed].update(alarms)
            train_seconds.append(seconds)
            print(f"  PPO seed {seed}, {held_out} held out: {seconds:.0f}s")
    ppo_summaries = {seed: _summary(_as_alarms(a, features), tables) for seed, a in ppo.items()}

    report: dict[str, Any] = {
        "method": (
            "leave-one-category-out over the five real NAB categories; NAB standard profile "
            "(evaluation.nab_scoring); PPO candidate triage, gate rank_extremeness > "
            f"{GATE}; {args.steps} decisions per training run; baselines tuned on the "
            "training folds only"
        ),
        "dev_fold": DEV_FOLD,
        "dev_fold_note": "PPO settings and the gate were chosen on this fold; "
        "score_without_dev_fold excludes it",
        "nab_commit": NAB_COMMIT,
        "nab_sha256": corpus_hashes(args.nab_root),
        "n_files": len(streams),
        "n_points": int(sum(f.shape[0] for f in features.values())),
        "ppo_config": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in default_training_config().to_dict().items()
            if k != "seed"
        },
        "ppo_by_seed": {str(k): v for k, v in ppo_summaries.items()},
        "ppo_mean_score_all_folds": round(
            float(np.mean([v["score_all_folds"] for v in ppo_summaries.values()])), 2
        ),
        "ppo_mean_score_without_dev_fold": round(
            float(np.mean([v["score_without_dev_fold"] for v in ppo_summaries.values()])), 2
        ),
        "baselines": summaries,
        "best_single_feature_chosen_on_held_out_folds": best_on_test,
        "tuned_thresholds": tuned,
        "ppo_train_seconds_per_run": round(float(np.mean(train_seconds)), 1),
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "timestamp": datetime.now(UTC).isoformat(),
    }
    RESULTS.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: report[k] for k in ("ppo_mean_score_all_folds", "ppo_mean_score_without_dev_fold")}
        )
    )
    print(json.dumps({k: v["score_all_folds"] for k, v in summaries.items()}, indent=1))

    if args.ship:
        import torch

        torch.set_num_threads(args.workers)
        trainer = train_alarm_policy(
            features,
            tables,
            gate=GATE,
            total_timesteps=args.steps,
            config=default_training_config(0),
        )
        sidecar = export_alarm_policy(
            trainer,
            SHIPPED_POLICY,
            gate=GATE,
            provenance={
                "description": "Streaming alarm policy: PPO candidate triage on NAB (standard profile)",
                "trainer": "scripts/train_streaming_alarm_policy.py",
                "data": f"NAB real categories at commit {NAB_COMMIT}, all {len(streams)} files",
                "nab_sha256": report["nab_sha256"],
                "training_steps": args.steps,
                "seed": 0,
                "ppo_config": report["ppo_config"],
                "evaluation": "benchmarks/streaming_alarm_policy_results.json",
                "leave_one_category_out_mean_score": report["ppo_mean_score_all_folds"],
                "created_utc": datetime.now(UTC).isoformat(),
            },
        )
        print(f"shipped {SHIPPED_POLICY} and {sidecar}")


if __name__ == "__main__":
    main()
