# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Paired sweep of the kNN density stream's fusion weight (re-runnable).

The statistical ensemble fuses its kNN density stream at a fixed 50/50
weight (``fused_z = 0.5 * (z_live + z_knn)`` in
``StatisticalAnomalyDetector._fuse_knn_stream``). That constant was never
itself measured — the stream's original validation compared stream-on
against stream-off. This harness measures the constant: it evaluates
``fused_z = (1 - w) * z_live + w * z_knn`` for ``w`` in a fixed grid over
every ground-truth ADBench set the committed headline uses, through the
exact benchmark eval path (``mercury_benchmark._benchmark_single``,
seed-42 split), and writes paired per-dataset AUCs plus the analysis to
``benchmarks/knn_weight_sweep_results.json``.

Experimental controls baked in:

* **Patch fidelity.** The weighted fusion at ``w = 0.5`` must reproduce the
  unpatched production detector bit-for-bit on every dataset, or the run
  aborts — the patch provably changes nothing but ``w``.
* **Split-half validation.** The grid winner is re-selected on each half of
  the datasets and confirmed on the other half, so a winner that only
  exists by selection on the full set is visible as such.

Finding (2026-10-14, 47 sets, committed in the results JSON): w = 0.6/0.7
beat the shipped 0.5 by a mean +0.0024/+0.0039 AUC (Wilcoxon p = 0.0098 /
0.0232; split-half confirms the direction both ways), but the median delta
is ~0 and the per-dataset optimum is bimodal (24 sets prefer 0.7, 17 prefer
0.3) with no fit-time predictor among n_train / n_features / anomaly_ratio
(|Spearman| <= 0.20). Under the repo's bar for touching the ensemble (a
multi-seed protocol, as used when the stream itself landed), this
single-protocol, near-zero-median result does not justify moving the
constant; it is recorded so a future multi-seed pass starts from here.

Run (needs the native AMA backend on LD_LIBRARY_PATH and the ADBench cache
under ``data/adbench/``):

    python benchmarks/knn_weight_sweep.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))

import mercury_benchmark as mb

from omni_mercury_engine.datasets.adbench import ADBenchLoader
from omni_mercury_engine.datasets.base import DatasetConfig
from omni_mercury_engine.detectors.statistical import (
    StatisticalAnomalyDetector,
    logger as _stat_logger,
)

GRID: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)
SHIPPED_W = 0.5
OUTPUT_PATH = _HERE / "knn_weight_sweep_results.json"

_ORIGINAL_FUSE = StatisticalAnomalyDetector._fuse_knn_stream
_ACTIVE_W = {"w": SHIPPED_W}


def _fuse_weighted(
    self: StatisticalAnomalyDetector,
    data: np.ndarray[Any, Any],
    combined_scores: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any] | None]:
    """Production fusion with ``0.5`` generalised to the active grid weight."""
    if self._knn_tree is None or self._knn_ref_stats is None or len(data) == 0:
        return combined_scores, None
    try:
        live = self._knn_live_reference()
        if live is None:
            return combined_scores, None
        knn = self._knn_distances(data)
    except Exception as exc:
        _stat_logger.warning("kNN density stream skipped for this batch: %s", exc)
        return combined_scores, None
    live_med, live_scale = live
    knn_med, knn_scale = self._knn_ref_stats
    z_live = (combined_scores - live_med) / live_scale
    z_knn = (knn - knn_med) / knn_scale
    w = _ACTIVE_W["w"]
    fused = live_med + live_scale * ((1.0 - w) * z_live + w * z_knn)
    return self._saturate_upper(fused), knn


def _headline_adbench_indices() -> list[int]:
    """ADBench indices the committed headline scores as ground truth."""
    committed = json.loads((_HERE / "mercury_benchmark_results.json").read_text())
    return sorted(
        int(r["name"].split("-")[1])
        for r in committed["per_dataset"]
        if r["category"] == "adbench"
        and r.get("label_source") == "ground_truth"
        and r.get("ensemble_auc") is not None
    )


def _evaluate(index: int, cache: dict[int, tuple[Any, Any]]) -> float:
    """Ensemble AUC for one dataset via the exact benchmark eval path."""
    if index not in cache:
        cfg = DatasetConfig(name=f"ADBench-{index:02d}", preprocessing={"dataset": str(index)})
        loader = ADBenchLoader(cfg)
        loader.download()
        x, y = loader._load_raw()
        cache[index] = (x, (y > 0).astype(int))
    x, y = cache[index]
    result = mb._benchmark_single(
        {
            "name": f"ADBench-{index:02d}",
            "category": "adbench",
            "X": x,
            "y": y,
            "label_source": "ground_truth",
        }
    )
    if result.get("error"):
        raise RuntimeError(f"ADBench-{index:02d} failed: {result['error']}")
    return float(result["ensemble_auc"])


def _spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation without a scipy dependency."""
    rank_a = {v: r for r, v in enumerate(sorted(a))}
    rank_b = {v: r for r, v in enumerate(sorted(b))}
    xa = [rank_a[x] for x in a]
    xb = [rank_b[x] for x in b]
    ma, mb_ = statistics.mean(xa), statistics.mean(xb)
    cov = sum((x - ma) * (y - mb_) for x, y in zip(xa, xb))
    var = math.sqrt(sum((x - ma) ** 2 for x in xa) * sum((y - mb_) ** 2 for y in xb))
    return cov / var if var else 0.0


def run_sweep() -> dict[str, Any]:
    """Run the full paired sweep and return the results document."""
    indices = _headline_adbench_indices()
    cache: dict[int, tuple[Any, Any]] = {}

    control = {i: _evaluate(i, cache) for i in indices}

    per_weight: dict[float, dict[int, float]] = {}
    StatisticalAnomalyDetector._fuse_knn_stream = _fuse_weighted  # type: ignore[method-assign]
    try:
        for w in GRID:
            _ACTIVE_W["w"] = w
            per_weight[w] = {i: _evaluate(i, cache) for i in indices}
    finally:
        StatisticalAnomalyDetector._fuse_knn_stream = _ORIGINAL_FUSE  # type: ignore[method-assign]

    mismatches = [i for i in indices if abs(per_weight[SHIPPED_W][i] - control[i]) > 1e-12]
    if mismatches:
        raise RuntimeError(
            f"patch fidelity broken: w={SHIPPED_W} diverges from the "
            f"unpatched detector on {mismatches} — sweep invalid"
        )

    base = per_weight[SHIPPED_W]
    comparisons = {}
    for w in GRID:
        if w == SHIPPED_W:
            continue
        deltas = [per_weight[w][i] - base[i] for i in indices]
        comparisons[str(w)] = {
            "mean_delta": statistics.mean(deltas),
            "median_delta": statistics.median(deltas),
            "wins": sum(1 for d in deltas if d > 1e-9),
            "losses": sum(1 for d in deltas if d < -1e-9),
            "worst_delta": min(deltas),
            "best_delta": max(deltas),
        }

    halves = (indices[::2], indices[1::2])
    split_half = []
    for select, confirm in (halves, halves[::-1]):
        winner = max(GRID, key=lambda w: statistics.mean(per_weight[w][i] for i in select))
        split_half.append(
            {
                "selected_w": winner,
                "confirm_mean_delta": statistics.mean(
                    per_weight[winner][i] - base[i] for i in confirm
                ),
                "confirm_wins": sum(1 for i in confirm if per_weight[winner][i] > base[i] + 1e-9),
                "confirm_losses": sum(1 for i in confirm if per_weight[winner][i] < base[i] - 1e-9),
            }
        )

    argmax = Counter(max(GRID, key=lambda w: per_weight[w][i]) for i in indices)

    committed = json.loads((_HERE / "mercury_benchmark_results.json").read_text())
    meta = {r["name"]: r for r in committed["per_dataset"]}
    preference = [per_weight[0.7][i] - per_weight[0.3][i] for i in indices]
    predictors = {}
    for field in ("n_train", "n_features", "anomaly_ratio"):
        values = [float(meta[f"ADBench-{i:02d}"][field]) for i in indices]
        predictors[field] = _spearman(preference, values)

    return {
        "protocol": {
            "eval_path": "mercury_benchmark._benchmark_single (seed-42 split)",
            "fusion": "fused_z = (1 - w) * z_live + w * z_knn",
            "datasets": len(indices),
            "grid": list(GRID),
            "shipped_w": SHIPPED_W,
            "patch_fidelity_mismatches": 0,
        },
        "per_dataset_auc": {
            f"w={w}": {f"ADBench-{i:02d}": per_weight[w][i] for i in indices} for w in GRID
        },
        "comparisons_vs_shipped": comparisons,
        "split_half_validation": split_half,
        "per_dataset_argmax_w": {str(k): v for k, v in sorted(argmax.items())},
        "fit_time_predictors_spearman_vs_w07_minus_w03": predictors,
        "disposition": (
            "constant unchanged at 0.5: single-protocol evidence with ~0 median "
            "delta does not meet the multi-seed bar used when the stream landed; "
            "re-run under additional split seeds before moving it"
        ),
    }


def main() -> int:
    """Run the sweep, print the analysis, write the results JSON."""
    doc = run_sweep()
    for w, row in doc["comparisons_vs_shipped"].items():
        print(
            f"w={w}: mean {row['mean_delta']:+.4f} median {row['median_delta']:+.4f} "
            f"W/L {row['wins']}/{row['losses']} worst {row['worst_delta']:+.4f}"
        )
    for half in doc["split_half_validation"]:
        print(
            f"split-half: selected w={half['selected_w']} -> confirm "
            f"{half['confirm_mean_delta']:+.4f} "
            f"({half['confirm_wins']}W/{half['confirm_losses']}L)"
        )
    print("argmax(w) dispersion:", doc["per_dataset_argmax_w"])
    OUTPUT_PATH.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
