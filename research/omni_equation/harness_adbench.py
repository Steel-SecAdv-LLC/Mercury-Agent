# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reproducible transductive ADBench harness for the MercuryAnomalyDetector fusion.

Pins a fixed 18-set ADBench ("Classical", Minqi824) subset and measures the real
Mercury detector under the standard transductive anomaly-detection protocol: fit
the unlabeled ensemble on X, score X, AUROC vs y (Mann-Whitney, via mercury_ml).
It exists so the fusion-hardening headline in ``docs/BENCHMARKS.md`` is a
committed, re-runnable measurement rather than an ad-hoc number:

    LD_LIBRARY_PATH=<ama-build-lib> python research/omni_equation/harness_adbench.py

Real data is mandatory. ADBench downloads from the trusted raw.githubusercontent
mirror; a dataset that fails to download raises rather than silently degrading,
and any set that resolves to a single label is reported as an error and excluded
from the mean (never scored as a meaningless AUROC). Results are deterministic:
the detector's self-supervised weighting uses a fixed seed and the protocol adds
no randomness of its own.

Pass ``--baseline adbench_base_e118e1f.json`` to diff this run against the
committed pre-hardening base run and emit the per-dataset win/tie/loss ledger
(``adbench_base_vs_current.json``) that backs the headline. The diff itself is
the engine-free ``_compare.py`` so it can be unit-tested without the AMA backend.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

# Runnable from a fresh checkout without an editable install: put the repo's
# src/ on sys.path before importing the engine (mirrors benchmarks/*.py). The
# script's own directory is added too so the sibling, engine-free ``_compare``
# module resolves whether the harness is run as a script or imported.
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _compare import compare_to_baseline

from omni_mercury_engine.datasets.adbench import ADBenchLoader
from omni_mercury_engine.datasets.base import DatasetConfig
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector
from omni_mercury_engine.ml.mercury_ml import StandardScaler, roc_auc_score

# Fixed, reproducible composition: 18 ADBench "Classical" tabular sets spanning
# the size (80-11k), dimensionality (6-64) and contamination (1.9-36%) range, and
# including every set named in the fusion-hardening analysis (wine, glass,
# PageBlocks, cardio, Ionosphere, Waveform, optdigits). Names use the catalog's
# canonical casing (see datasets/adbench.py ADBENCH_CATALOG).
DATASETS: tuple[str, ...] = (
    "breastw",
    "cardio",
    "Cardiotocography",
    "glass",
    "Hepatitis",
    "Ionosphere",
    "Lymphography",
    "mammography",
    "optdigits",
    "PageBlocks",
    "Pima",
    "Stamps",
    "thyroid",
    "vertebral",
    "Waveform",
    "WBC",
    "wine",
    "WPBC",
)


def _auroc(y: np.ndarray[Any, Any], s: np.ndarray[Any, Any]) -> float:
    """Mann-Whitney AUROC via mercury_ml (no sklearn)."""
    return float(roc_auc_score(y, s))


def run_one(name: str, data_dir: str) -> dict[str, Any]:
    """Fetch one ADBench set (real data) and score it transductively."""
    t0 = time.time()
    loader = ADBenchLoader(DatasetConfig(name=name, data_dir=data_dir, max_samples=None))
    X_raw, y_raw = loader.load()
    X = np.asarray(X_raw, dtype=float)
    y = np.asarray(y_raw).astype(int).reshape(-1)

    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return {"dataset": name, "error": f"single-class after load (n_pos={n_pos})"}

    X_std = StandardScaler().fit_transform(X)
    det = MercuryAnomalyDetector()
    det.fit(X_std)
    scores = np.asarray(det.detect(X_std)["scores"], dtype=float).reshape(-1)

    return {
        "dataset": name,
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),
        "anom_rate": round(float(y.mean()), 4),
        "data_type": str(det._data_type.value),
        "weight_source": det._weight_source,
        "auroc": round(_auroc(y, scores), 4),
        "secs": round(time.time() - t0, 1),
    }


def main() -> None:
    """Run the fixed ADBench suite and persist per-dataset + mean AUROC."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="./data/adbench",
        help="Directory for the cached ADBench .npz files (downloaded if absent).",
    )
    parser.add_argument(
        "--out",
        default="research/omni_equation/adbench_results.json",
        help="Where to write the JSON results.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help=(
            "Optional baseline results JSON (same schema as --out) to diff this "
            "run against, emitting a per-dataset win/tie/loss comparison. Use the "
            "committed base run: --baseline research/omni_equation/adbench_base_e118e1f.json"
        ),
    )
    parser.add_argument(
        "--compare-out",
        default="research/omni_equation/adbench_base_vs_current.json",
        help="Where to write the base-vs-current comparison (only with --baseline).",
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for name in DATASETS:
        try:
            row = run_one(name, args.data_dir)
        except Exception as exc:
            row = {"dataset": name, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
        rows.append(row)
        print(json.dumps(row))

    scored = [r for r in rows if "auroc" in r]
    mean_auroc = round(float(np.mean([r["auroc"] for r in scored])), 4) if scored else float("nan")
    summary = {
        "n_datasets": len(DATASETS),
        "n_scored": len(scored),
        "mean_auroc": mean_auroc,
        "protocol": "transductive (fit unlabeled on X, score X, AUROC vs y); full sets",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "results": rows}, indent=2) + "\n")

    print("\n==== ADBench transductive suite (real Mercury detector) ====")
    for r in scored:
        print(
            f"{r['dataset']:18} {r['data_type']:8} n={r['n']:>5} d={r['d']:>3} "
            f"anom={r['anom_rate']:.4f}  AUROC={r['auroc']:.4f}  [{r['weight_source']}]"
        )
    failed = [r for r in rows if "auroc" not in r]
    for r in failed:
        print(f"{r['dataset']:18} ERROR: {r.get('error')}")
    print(f"\nMean AUROC ({len(scored)}/{len(DATASETS)} scored): {mean_auroc}")

    if args.baseline:
        baseline_doc = json.loads(Path(args.baseline).read_text())
        baseline_rows = baseline_doc.get("results", baseline_doc)
        comparison = compare_to_baseline(rows, baseline_rows)
        baseline_commit = baseline_doc.get("summary", {}).get("commit")
        if baseline_commit:
            comparison["summary"]["baseline_commit"] = baseline_commit
        compare_path = Path(args.compare_out)
        compare_path.parent.mkdir(parents=True, exist_ok=True)
        compare_path.write_text(json.dumps(comparison, indent=2) + "\n")

        cs = comparison["summary"]
        print("\n==== base vs current (per-set win/tie/loss) ====")
        for r in comparison["per_set"]:
            gate = (
                f'{r["baseline_data_type"]}->{r["data_type"]}'
                if r["baseline_data_type"] != r["data_type"]
                else str(r["data_type"])
            )
            print(
                f'{r["dataset"]:18} {r["baseline_auroc"]:.4f} -> {r["auroc"]:.4f}  '
                f'{r["delta"]:+.4f}  {r["verdict"]:4}  [{gate}]'
            )
        print(
            f'\nMean AUROC {cs["mean_baseline"]} -> {cs["mean_current"]} '
            f'({cs["mean_delta"]:+.4f}); {cs["wins"]} W / {cs["ties"]} tie / {cs["losses"]} L'
        )


if __name__ == "__main__":
    main()
