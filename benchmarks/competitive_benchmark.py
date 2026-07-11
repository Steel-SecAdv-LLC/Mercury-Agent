# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Competitive head-to-head benchmark: Mercury vs PyOD on the ADBench suite.

Measures Mercury's tier detector (:class:`MercuryAnomalyDetector`, unsupervised
``.fit`` path) and fusion engine (:meth:`OmniMercuryEngine.fit_fusion` /
:meth:`score_fusion`, label-free consensus path) against real PyOD baselines
(IForest, ECOD, COPOD, LOF, KNN, HBOS) on the standard ADBench suite: the 47
Classical tabular datasets plus 10 CV/NLP embedding datasets (57 total).
Losses are reported with the same prominence as wins -- this harness measures
position vs the competition, it is not a highlight reel.

Protocol (identical for every method, per dataset):

1. Load raw ``(X, y)`` (sha256 of the NPZ recorded for provenance).
2. Stratified cap to ``2 * MAX_SAMPLES`` rows (seed 42), mirroring
   ``mercury_benchmark._benchmark_single``.
3. Contamination-free split: fit on a normal-only train half (seed 42),
   test = remaining normals + all anomalies (capped to ``MAX_SAMPLES``).
4. **De-leak shuffle**: test rows are shuffled with a fixed seed before
   scoring. ``MercuryAnomalyDetector.detect()`` applies a Conv1d finite
   difference ACROSS the batch, so a label-sorted batch would leak label
   information through evaluation order (see the long comment in
   ``mercury_benchmark._benchmark_single``). The same shuffled rows go to
   every method, so no method sees ordering information another doesn't.
5. ``StandardScaler`` fit on train only; transform applied to both splits.
6. Score with every method; report ROC-AUC + Average Precision.

Fairness rules: library defaults everywhere (the only non-default is a fixed
``random_state`` on stochastic detectors), no per-dataset hyperparameter
tuning for ANY method, identical preprocessing, and the comparison is
unsupervised-fair -- Mercury fusion runs its label-free consensus path
(``y=None``) because the PyOD baselines get no labels either. PyOD's deep
baselines (AutoEncoder) are excluded from the default set: under the
defaults-only CPU budget they measure the budget cap, not the method (see
``omni_mercury_engine.comparison.pyod_integration.DEFAULT_BASELINES``).

Any dataset that fails to load or evaluate is recorded with a named reason in
the output JSON -- no silent drops.

Usage::

    python benchmarks/competitive_benchmark.py                 # full 57-dataset run
    python benchmarks/competitive_benchmark.py --quick         # documented 10-set CI subset
    python benchmarks/competitive_benchmark.py --datasets cardio WBC nlp:20news_0
    python benchmarks/competitive_benchmark.py --skip-fusion   # tier + PyOD only

Output:
    benchmarks/competitive_results.json (full run) /
    benchmarks/competitive_results_quick.json (--quick)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing as _mp
import os
import queue as _queue
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))

import mercury_benchmark as mb
import numpy as np

from omni_mercury_engine.comparison.pyod_integration import (
    DEFAULT_BASELINES,
    pyod_version,
    run_pyod_baselines,
)
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector
from omni_mercury_engine.ml.mercury_ml import (
    StandardScaler,
    average_precision_score,
    roc_auc_score,
)

SEED = 42
FUSION_EPOCHS = 50  # fit_fusion default; early stopping usually stops well before
OUTPUT_PATH = _HERE / "competitive_results.json"
QUICK_OUTPUT_PATH = _HERE / "competitive_results_quick.json"

#: Mercury methods measured head-to-head (registered order = report order).
MERCURY_METHODS: tuple[str, ...] = ("mercury_tier", "mercury_fusion")

#: PyOD baseline names, in report order.
PYOD_METHODS: tuple[str, ...] = tuple(a.value for a in DEFAULT_BASELINES)

#: Per-(dataset, method) wall-clock cap in seconds. A single cell exceeding this
#: budget is recorded as a transparent deferral -- never a silent drop, and
#: never a hang that stalls the rest of the suite on a large shared box.
#: Override via the ``MERCURY_METHOD_TIMEOUT`` environment variable.
METHOD_TIMEOUT_SECONDS: int = int(os.environ.get("MERCURY_METHOD_TIMEOUT", "300"))


def _run_cell(fn: Any, timeout_s: int = METHOD_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Run one (dataset, method) cell under a hard wall-clock cap.

    The cell runs in a forked child so the cap is a real wall-clock kill, not a
    cooperative signal that a long native call (e.g. an O(n^2) neighbour search)
    could ignore. The fork inherits the parent's already-imported modules, so a
    child re-runs no imports and does not re-pay the PQC import gate.

    Returns:
        ``fn()``'s result dict with a ``wall_seconds`` key added on success; on
        overrun ``{"deferred": "exceeded {N}s wall budget", "wall_seconds": ...}``;
        on a child crash ``{"error": ..., "wall_seconds": ...}``.
    """
    ctx = _mp.get_context("fork")
    result_q: Any = ctx.Queue()

    def _target() -> None:
        try:
            result_q.put(("ok", fn()))
        except Exception as exc:  # pragma: no cover - reported to parent, never raised
            result_q.put(("err", f"{type(exc).__name__}: {exc}"))

    t0 = time.perf_counter()
    proc = ctx.Process(target=_target, daemon=True)
    proc.start()
    try:
        status, payload = result_q.get(timeout=timeout_s)
    except _queue.Empty:
        wall = round(time.perf_counter() - t0, 1)
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            return {"deferred": f"exceeded {timeout_s}s wall budget", "wall_seconds": wall}
        proc.join()
        return {"error": f"cell died (exit={proc.exitcode}) before returning", "wall_seconds": wall}
    proc.join(5)
    wall = round(time.perf_counter() - t0, 1)
    if status == "ok":
        result = payload if isinstance(payload, dict) else {"value": payload}
        result.setdefault("wall_seconds", wall)
        return result
    return {"error": payload, "wall_seconds": wall}


# ---------------------------------------------------------------------------
# Dataset catalog: 47 ADBench Classical + 10 CV/NLP embedding datasets = 57
# ---------------------------------------------------------------------------

#: ADBench ships its CV/NLP corpora as *embedding* NPZs with one variant per
#: "normal class" (``CIFAR10_0`` .. ``CIFAR10_9``). Following common practice
#: for the "57-dataset" ADBench suite we take ONE representative variant per
#: source corpus: the ``_0`` variant where per-class variants exist
#: (CIFAR10 / FashionMNIST / SVHN / 20news / agnews), the first-alphabetical
#: variant for the corruption/object-keyed corpora that have no ``_0``
#: (``MNIST-C_brightness``, ``MVTec-AD_bottle`` -- verified against the
#: upstream repository listing), and the single shipped file for the binary
#: sentiment corpora (``amazon`` / ``imdb`` / ``yelp``, which have no
#: variants at all).
EMBEDDING_DATASETS: tuple[tuple[str, str], ...] = (
    ("CV_by_ResNet18", "CIFAR10_0"),
    ("CV_by_ResNet18", "FashionMNIST_0"),
    ("CV_by_ResNet18", "SVHN_0"),
    ("CV_by_ResNet18", "MNIST-C_brightness"),
    ("CV_by_ResNet18", "MVTec-AD_bottle"),
    ("NLP_by_BERT", "20news_0"),
    ("NLP_by_BERT", "agnews_0"),
    ("NLP_by_BERT", "amazon"),
    ("NLP_by_BERT", "imdb"),
    ("NLP_by_BERT", "yelp"),
)

_ADBENCH_DATASETS_BASE = "https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/"

#: The documented ``--quick`` subset: the 8 genuine-label Classical guard
#: datasets already pinned by ``anomaly_regression_guard.GUARD_DATASETS``
#: (small, fast, spanning strong- and weak-signal regimes) plus one CV and one
#: NLP embedding set so the quick lane exercises every dataset family.
QUICK_DATASETS: tuple[str, ...] = (
    "breastw",
    "cardio",
    "Ionosphere",
    "WBC",
    "Lymphography",
    "Pima",
    "glass",
    "pendigits",
    "cv:MVTec-AD_bottle",
    "nlp:20news_0",
)


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _data_root() -> Path:
    """Resolve the dataset cache root (honours ``MERCURY_DATA_DIR``)."""
    import os

    return Path(os.environ.get("MERCURY_DATA_DIR", "./data"))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_classical(name: str) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Load an ADBench Classical dataset via the shared loader.

    Returns:
        ``(X, y, npz_sha256, source_url)``.
    """
    from omni_mercury_engine.datasets.adbench import ADBenchLoader
    from omni_mercury_engine.datasets.base import DatasetConfig

    loader = ADBenchLoader(DatasetConfig(name="adbench", preprocessing={"dataset": name}))
    loader.download()
    npz_path = loader.data_path / loader.npz_filename
    sha = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    X, y = loader._load_raw()
    return X, (y > 0).astype(int), sha, loader.npz_url


def _load_embedding(group: str, name: str) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Load (fetching + caching on first use) an ADBench CV/NLP embedding NPZ.

    Uses the same allowlisted ``raw.githubusercontent.com`` transport
    (:func:`http_get_with_retry`) as the Classical loader; files cache under
    ``<MERCURY_DATA_DIR>/adbench_embeddings/<group>/``.

    Returns:
        ``(X, y, npz_sha256, source_url)``.
    """
    from omni_mercury_engine.datasets.base import http_get_with_retry

    url = f"{_ADBENCH_DATASETS_BASE}{group}/{name}.npz"
    target = _data_root() / "adbench_embeddings" / group / f"{name}.npz"
    if not target.exists():
        content = http_get_with_retry(url, timeout=120)
        data = np.load(io.BytesIO(content), allow_pickle=False)
        if "X" not in data or "y" not in data:
            raise ValueError(f"NPZ missing X/y keys, found: {list(data.keys())}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    raw = target.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    data = np.load(io.BytesIO(raw), allow_pickle=False)
    X = data["X"].astype(np.float64)
    y = (data["y"].astype(np.int64).ravel() > 0).astype(int)
    return X, y, sha, url


def build_dataset_catalog(
    only: list[str] | None = None, quick: bool = False
) -> list[dict[str, Any]]:
    """Build the evaluation catalog (name, family, loader thunk).

    Args:
        only: Optional dataset names (Classical name, or ``cv:<name>`` /
            ``nlp:<name>`` for embedding sets).
        quick: Use the documented :data:`QUICK_DATASETS` subset.

    Returns:
        List of ``{"name", "family", "load"}`` entries in a fixed order.
    """
    from omni_mercury_engine.datasets.adbench import ADBENCH_CATALOG

    selected = list(only) if only else (list(QUICK_DATASETS) if quick else None)

    catalog: list[dict[str, Any]] = []
    for _idx, cname in sorted(ADBENCH_CATALOG.items()):
        catalog.append(
            {
                "name": cname,
                "family": "adbench_classical",
                "load": (lambda n=cname: _load_classical(n)),
            }
        )
    for group, ename in EMBEDDING_DATASETS:
        prefix = "cv" if group.startswith("CV") else "nlp"
        catalog.append(
            {
                "name": f"{prefix}:{ename}",
                "family": f"adbench_{prefix}_embedding",
                "load": (lambda g=group, n=ename: _load_embedding(g, n)),
            }
        )
    if selected is None:
        return catalog
    wanted = {s.lower() for s in selected}
    picked = [c for c in catalog if c["name"].lower() in wanted]
    missing = wanted - {c["name"].lower() for c in picked}
    if missing:
        raise ValueError(f"Unknown dataset selection(s): {sorted(missing)}")
    return picked


# ---------------------------------------------------------------------------
# Shared protocol: split + de-leak shuffle + scale (mirrors _benchmark_single)
# ---------------------------------------------------------------------------


def prepare_split(
    X_full: np.ndarray, y_full: np.ndarray, *, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | str:
    """Produce the shared (X_train, X_test, y_test) protocol arrays.

    Mirrors ``mercury_benchmark._benchmark_single`` exactly: stratified cap,
    normal-only train half, remaining-normals + all-anomalies test, test cap,
    fixed-seed de-leak shuffle, NaN/Inf scrub, train-fitted StandardScaler.
    Every compared method consumes these exact arrays.

    Returns:
        ``(X_train, X_test, y_test)`` on success, or a string skip-reason.
    """
    if X_full.ndim == 1:
        X_full = X_full.reshape(-1, 1)
    if len(np.unique(y_full)) < 2:
        return f"Only one class present (labels={np.unique(y_full).tolist()})"

    X_full, y_full = mb._cap_stratified(X_full, y_full, mb.MAX_SAMPLES * 2)

    normal_mask = y_full == 0
    X_normal = X_full[normal_mask]
    n_train = min(mb.MAX_SAMPLES, len(X_normal) // 2)
    if n_train < 5:
        return f"Too few normal samples for training ({n_train})"

    rng = np.random.RandomState(seed)
    train_idx = rng.choice(len(X_normal), n_train, replace=False)
    X_train = X_normal[train_idx]

    test_normal_mask = np.ones(len(X_normal), dtype=bool)
    test_normal_mask[train_idx] = False
    X_test = np.vstack([X_normal[test_normal_mask], X_full[~normal_mask]])
    y_test = np.concatenate(
        [
            np.zeros(int(test_normal_mask.sum()), dtype=int),
            np.ones(int((~normal_mask).sum()), dtype=int),
        ]
    )
    X_test, y_test = mb._cap_stratified(X_test, y_test, mb.MAX_SAMPLES)

    # De-leak shuffle (see module docstring / _benchmark_single): evaluation
    # order must carry no label information, for EVERY method identically.
    perm = np.random.RandomState(seed).permutation(len(X_test))
    X_test, y_test = X_test[perm], y_test[perm]

    if len(np.unique(y_test)) < 2:
        return "Test split single-class after capping"

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10).astype(np.float64)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10).astype(np.float64)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_test


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------


def _metrics(y_test: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """ROC-AUC + Average Precision (NaN-safe)."""
    try:
        auc = float(roc_auc_score(y_test, scores))
    except ValueError:
        auc = float("nan")
    try:
        ap = float(average_precision_score(y_test, scores))
    except ValueError:
        ap = float("nan")
    return {"roc_auc": auc, "average_precision": ap}


def _run_mercury_tier(
    X_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> dict[str, Any]:
    """Mercury tier: unsupervised MercuryAnomalyDetector fit/detect."""
    t0 = time.perf_counter()
    detector = MercuryAnomalyDetector()
    detector.fit(X_train)
    # Same runtime-only domain marker mercury_benchmark plants (category
    # "adbench"), so the tier runs the identical eval path (see
    # _benchmark_single).
    detector._benchmark_domain = "adbench"  # type: ignore[attr-defined]
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    scores = np.asarray(detector.detect(X_test)["scores"], dtype=np.float64).ravel()
    score_s = time.perf_counter() - t0
    return {**_metrics(y_test, scores), "fit_seconds": fit_s, "score_seconds": score_s}


def _run_mercury_fusion(
    X_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, *, seed: int = SEED
) -> dict[str, Any]:
    """Mercury fusion: label-free fit_fusion (consensus path) + score_fusion.

    Unsupervised-fair: ``y=None`` -- the fusion model trains on
    detector-consensus pseudo-labels derived from the same unlabelled train
    rows every PyOD baseline sees. No ground-truth labels are used anywhere.
    """
    import torch

    from omni_mercury_engine.engine import OmniMercuryEngine

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    fit_info = engine.fit_fusion(
        X_train.astype(np.float32), None, epochs=FUSION_EPOCHS, batch_size=64
    )
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    scores = np.asarray(engine.score_fusion(X_test.astype(np.float32)), dtype=np.float64).ravel()
    score_s = time.perf_counter() - t0
    return {
        **_metrics(y_test, scores),
        "fit_seconds": fit_s,
        "score_seconds": score_s,
        "epochs_trained": int(fit_info.get("epochs_trained", -1)),
    }


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------


def evaluate_dataset(
    entry: dict[str, Any], *, skip_fusion: bool = False, seed: int = SEED
) -> dict[str, Any]:
    """Run every method on one dataset under the shared protocol."""
    name = entry["name"]
    row: dict[str, Any] = {"name": name, "family": entry["family"]}
    try:
        X, y, sha, url = entry["load"]()
    except Exception as exc:
        row["error"] = f"load failed: {type(exc).__name__}: {exc}"
        print(f"  [{name}] SKIP: {row['error'][:100]}")
        return row
    row.update(
        n_samples=len(X),
        n_features=int(X.shape[1] if X.ndim > 1 else 1),
        anomaly_ratio=float(np.mean(y)),
        npz_sha256=sha,
        source_url=url,
    )

    split = prepare_split(X, y, seed=seed)
    if isinstance(split, str):
        row["error"] = split
        print(f"  [{name}] SKIP: {split}")
        return row
    X_train, X_test, y_test = split
    row.update(n_train=len(X_train), n_test=len(X_test))

    methods: dict[str, dict[str, Any]] = {}

    # Every (dataset, method) cell runs under a hard per-cell wall-clock cap in
    # a forked child (METHOD_TIMEOUT_SECONDS): an overrun is a recorded deferral,
    # never a silent drop and never a hang that stalls the rest of the suite.
    methods["mercury_tier"] = _run_cell(lambda: _run_mercury_tier(X_train, X_test, y_test))
    if not skip_fusion:
        methods["mercury_fusion"] = _run_cell(
            lambda: _run_mercury_fusion(X_train, X_test, y_test, seed=seed)
        )

    # Each PyOD baseline gets its OWN capped cell so one slow detector (LOF/KNN
    # on the largest sets) defers alone instead of taking the whole row with it.
    for algo in DEFAULT_BASELINES:
        cell = _run_cell(
            lambda a=algo: run_pyod_baselines(X_train, X_test, algorithms=[a], seed=seed)[a.value]
        )
        if "scores" in cell:
            methods[algo.value] = {
                **_metrics(y_test, cell["scores"]),
                "fit_seconds": cell["fit_seconds"],
                "score_seconds": cell["score_seconds"],
                "wall_seconds": cell.get("wall_seconds"),
            }
        else:  # error or deferral -- recorded verbatim, never dropped
            methods[algo.value] = cell

    row["methods"] = methods
    aucs = {
        m: r["roc_auc"] for m, r in methods.items() if "roc_auc" in r and not np.isnan(r["roc_auc"])
    }
    deferred = [m for m, r in methods.items() if "deferred" in r]
    best = max(aucs, key=aucs.__getitem__) if aucs else None
    row["best_method"] = best
    parts = "  ".join(f"{m}={aucs.get(m, float('nan')):.3f}" for m in aucs)
    tail = f"  DEFERRED={deferred}" if deferred else ""
    print(f"  [{name}] best={best}  {parts}{tail}")
    return row


# ---------------------------------------------------------------------------
# Aggregation: means, medians, ranks, wins/losses
# ---------------------------------------------------------------------------


def summarize(per_dataset: list[dict[str, Any]], method_order: list[str]) -> dict[str, Any]:
    """Aggregate per-method stats + rank stats + Mercury wins/losses.

    Mean/median are computed over each method's own successful datasets.
    Rank statistics (mean rank, wins/losses) use only *complete* datasets --
    those where every method in ``method_order`` produced a finite AUC -- so
    ranks are always comparing identical work.
    """
    measured = [r for r in per_dataset if "methods" in r]

    per_method: dict[str, dict[str, Any]] = {}
    for m in method_order:
        aucs = []
        aps = []
        errors = []
        for r in measured:
            res = r["methods"].get(m)
            if res is None:
                continue
            if "error" in res:
                errors.append({"dataset": r["name"], "error": res["error"]})
            elif not np.isnan(res["roc_auc"]):
                aucs.append(res["roc_auc"])
                aps.append(res["average_precision"])
        per_method[m] = {
            "n_datasets": len(aucs),
            "mean_auc": float(np.mean(aucs)) if aucs else None,
            "median_auc": float(np.median(aucs)) if aucs else None,
            "std_auc": float(np.std(aucs)) if aucs else None,
            "mean_ap": float(np.mean(aps)) if aps else None,
            "median_ap": float(np.median(aps)) if aps else None,
            "errors": errors,
        }

    # Complete rows for rank stats.
    def _complete(r: dict[str, Any]) -> bool:
        return all(
            m in r["methods"]
            and "roc_auc" in r["methods"][m]
            and not np.isnan(r["methods"][m]["roc_auc"])
            for m in method_order
        )

    complete = [r for r in measured if _complete(r)]
    rank_sum = dict.fromkeys(method_order, 0.0)
    for r in complete:
        aucs = np.array([r["methods"][m]["roc_auc"] for m in method_order])
        # rank 1 = best AUC; average ranks on ties
        order = (-aucs).argsort(kind="mergesort")
        ranks = np.empty(len(method_order))
        ranks[order] = np.arange(1, len(method_order) + 1)
        # tie-average
        for val in np.unique(aucs):
            tie = aucs == val
            if tie.sum() > 1:
                ranks[tie] = ranks[tie].mean()
        for i, m in enumerate(method_order):
            rank_sum[m] += float(ranks[i])
    mean_rank = {m: (rank_sum[m] / len(complete)) if complete else None for m in method_order}
    for m in method_order:
        per_method[m]["mean_rank"] = mean_rank[m]

    # Mercury wins/losses vs each baseline (per Mercury method), with margins.
    head_to_head: dict[str, Any] = {}
    for mm in MERCURY_METHODS:
        if mm not in method_order:
            continue
        vs: dict[str, Any] = {}
        for pm in method_order:
            if pm in MERCURY_METHODS:
                continue
            rows = [
                r
                for r in measured
                if all(
                    k in r["methods"]
                    and "roc_auc" in r["methods"][k]
                    and not np.isnan(r["methods"][k]["roc_auc"])
                    for k in (mm, pm)
                )
            ]
            wins = losses = ties = 0
            loss_list: list[dict[str, Any]] = []
            for r in rows:
                d = r["methods"][mm]["roc_auc"] - r["methods"][pm]["roc_auc"]
                if d > 0:
                    wins += 1
                elif d < 0:
                    losses += 1
                    loss_list.append({"dataset": r["name"], "auc_margin": float(-d)})
                else:
                    ties += 1
            loss_list.sort(key=lambda e: -e["auc_margin"])
            vs[pm] = {
                "n_compared": len(rows),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "worst_losses": loss_list[:5],
            }
        head_to_head[mm] = vs

    # Every dataset where ANY baseline beats the Mercury method, by margin
    # (feeds the "Where Mercury loses" section of the report).
    losses_any: dict[str, list[dict[str, Any]]] = {}
    for mm in MERCURY_METHODS:
        if mm not in method_order:
            continue
        rows_out: list[dict[str, Any]] = []
        for r in measured:
            res_m = r["methods"].get(mm, {})
            if "roc_auc" not in res_m or np.isnan(res_m.get("roc_auc", float("nan"))):
                continue
            beat_by = {}
            for pm in method_order:
                if pm in MERCURY_METHODS:
                    continue
                res_p = r["methods"].get(pm, {})
                if "roc_auc" in res_p and not np.isnan(res_p["roc_auc"]):
                    if res_p["roc_auc"] > res_m["roc_auc"]:
                        beat_by[pm] = float(res_p["roc_auc"] - res_m["roc_auc"])
            if beat_by:
                worst = max(beat_by, key=beat_by.__getitem__)
                rows_out.append(
                    {
                        "dataset": r["name"],
                        "mercury_auc": float(res_m["roc_auc"]),
                        "n_baselines_beating": len(beat_by),
                        "worst_baseline": worst,
                        "worst_margin": beat_by[worst],
                        "beaten_by": beat_by,
                    }
                )
        rows_out.sort(key=lambda e: -e["worst_margin"])
        losses_any[mm] = rows_out

    # Every (dataset, method) cell the wall-clock guard deferred, recorded with
    # its measured wall time -- surfaced so the report's "Deferred cells"
    # subsection is exact and no timed-out cell is silently absent.
    deferred_cells = [
        {"dataset": r["name"], "method": m, "wall_seconds": res.get("wall_seconds")}
        for r in measured
        for m, res in r["methods"].items()
        if isinstance(res, dict) and "deferred" in res
    ]

    return {
        "n_datasets_attempted": len(per_dataset),
        "n_datasets_measured": len(measured),
        "n_datasets_complete_for_ranks": len(complete),
        "per_method": per_method,
        "head_to_head": head_to_head,
        "datasets_where_mercury_loses": losses_any,
        "deferred_cells": deferred_cells,
        "skipped": [
            {"dataset": r["name"], "reason": r["error"]} for r in per_dataset if "error" in r
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    *,
    quick: bool = False,
    datasets: list[str] | None = None,
    skip_fusion: bool = False,
    seed: int = SEED,
) -> dict[str, Any]:
    """Run the full competitive benchmark and return the results dict."""
    catalog = build_dataset_catalog(only=datasets, quick=quick)
    method_order = [m for m in MERCURY_METHODS if not (skip_fusion and m == "mercury_fusion")]
    method_order += list(PYOD_METHODS)

    # Warm the PyOD model imports in the PARENT once, so every forked per-cell
    # child inherits them (copy-on-write) instead of re-importing pyod + numba
    # from disk on each of the ~6*N baseline cells.
    from omni_mercury_engine.comparison.pyod_integration import build_pyod_detector

    for _algo in DEFAULT_BASELINES:
        try:
            build_pyod_detector(_algo)
        except Exception:  # noqa: BLE001 - warm-up only; real errors surface per cell
            pass

    mode = "quick" if quick else ("custom" if datasets else "full")
    print("=" * 78)
    print("Mercury Competitive Benchmark -- Mercury tier + fusion vs PyOD baselines")
    print(f"  datasets={len(catalog)} ({mode})  methods={method_order}  seed={seed}")
    print("=" * 78)

    t_start = time.perf_counter()
    per_dataset = [evaluate_dataset(entry, skip_fusion=skip_fusion, seed=seed) for entry in catalog]
    runtime_s = time.perf_counter() - t_start

    summary = summarize(per_dataset, method_order)

    # torch is optional here: --skip-fusion must run on a [benchmark]-only
    # install (no ML stack), so its version is recorded when importable and
    # null otherwise rather than hard-importing it just to stamp a version.
    def _version(mod: str) -> str | None:
        try:
            return str(__import__(mod).__version__)
        except Exception:
            return None

    results = {
        "metadata": {
            "benchmark": "competitive_benchmark (Mercury vs PyOD, ADBench 47 Classical + 10 CV/NLP)",
            "protocol": (
                "per dataset: stratified cap to 2*MAX_SAMPLES (seed 42); normal-only train "
                "half; test = remaining normals + all anomalies capped to MAX_SAMPLES; "
                "fixed-seed de-leak shuffle of test rows (batch-order leakage guard, applied "
                "to every method identically); StandardScaler fit on train only; library "
                "defaults for every method; no per-dataset tuning; unsupervised-fair "
                "(no method sees labels)"
            ),
            "mercury_fusion_protocol": (
                "fit_fusion(X_train, y=None): label-free detector-consensus path; "
                "semi-supervised mode NOT used because PyOD baselines get no labels"
            ),
            "deep_baselines_note": (
                "PyOD deep baselines (AutoEncoder) excluded: under the defaults-only, "
                "CPU-only, no-tuning budget they measure the budget cap rather than the "
                "method; see pyod_integration.DEFAULT_BASELINES"
            ),
            "embedding_variant_note": (
                "CV/NLP corpora use one representative ADBench variant per source: _0 where "
                "per-class variants exist; MNIST-C_brightness / MVTec-AD_bottle (no _0 "
                "variant exists upstream); amazon/imdb/yelp ship a single file"
            ),
            "mode": mode,
            "seed": seed,
            "max_samples": mb.MAX_SAMPLES,
            "fusion_epochs": FUSION_EPOCHS,
            "git_commit": _git_commit(),
            "command_line": " ".join(sys.argv),
            "timestamp": datetime.now(UTC).isoformat(),
            "runtime_seconds": round(runtime_s, 1),
            "versions": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "torch": _version("torch"),
                "pyod": pyod_version(),
                "scikit_learn": _version("sklearn"),
            },
            "dataset_source": "https://github.com/Minqi824/ADBench (MIT)",
            "methods": method_order,
        },
        "summary": summary,
        "per_dataset": per_dataset,
    }
    return results


def _print_summary(results: dict[str, Any]) -> None:
    """Print the aggregate table."""
    summary = results["summary"]
    print("\n" + "=" * 78)
    print(
        f"{'method':<18} {'n':>3} {'mean AUC':>9} {'med AUC':>9} "
        f"{'mean AP':>9} {'med AP':>9} {'mean rank':>10}"
    )
    print("-" * 78)
    for m in results["metadata"]["methods"]:
        s = summary["per_method"][m]
        if s["mean_auc"] is None:
            print(f"{m:<18} {s['n_datasets']:>3}  (no successful datasets)")
            continue
        rank = f"{s['mean_rank']:.2f}" if s["mean_rank"] is not None else "-"
        print(
            f"{m:<18} {s['n_datasets']:>3} {s['mean_auc']:>9.4f} {s['median_auc']:>9.4f} "
            f"{s['mean_ap']:>9.4f} {s['median_ap']:>9.4f} {rank:>10}"
        )
    print("-" * 78)
    for mm, vs in summary["head_to_head"].items():
        for pm, rec in vs.items():
            print(
                f"{mm} vs {pm}: {rec['wins']}W/{rec['losses']}L/{rec['ties']}T "
                f"of {rec['n_compared']}"
            )
    if summary["skipped"]:
        print(f"skipped ({len(summary['skipped'])}):")
        for s in summary["skipped"]:
            print(f"  {s['dataset']}: {s['reason']}")


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="documented 10-dataset CI subset (QUICK_DATASETS); writes *_quick.json",
    )
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="explicit dataset names (Classical name, or cv:<name> / nlp:<name>)",
    )
    ap.add_argument("--skip-fusion", action="store_true", help="skip the (slow) fusion engine lane")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()

    results = run(
        quick=args.quick,
        datasets=args.datasets,
        skip_fusion=args.skip_fusion,
        seed=args.seed,
    )
    _print_summary(results)

    out = args.output or (QUICK_OUTPUT_PATH if args.quick else OUTPUT_PATH)
    out.write_text(json.dumps(results, indent=2, sort_keys=False) + "\n")
    print(f"\nresults written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
