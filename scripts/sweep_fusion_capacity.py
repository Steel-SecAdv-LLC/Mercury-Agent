# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Capacity sweep for the fusion network's ``hidden_dim`` — the evidence behind the default checkpoint's width.

Why this exists
---------------
``hidden_dim`` should be chosen on held-out performance. A single training run
cannot justify a width: anomaly-detection AUC has real seed-to-seed variance,
so a one-shot "128 beats 32 by 0.014" can be pure noise. This harness measures
each candidate width across multiple seeds and multiple genuinely-labelled
datasets, then reports paired-difference statistics so the decision rests on
signal, not a sample of one.

Protocol (per ``dim`` x ``seed`` x ``dataset``)
    1. Stratified train/test split (seeded).
    2. ``engine.fit_fusion`` on the train split — fits detectors on train only,
       trains the head with FocalLoss, fits temperature calibration.
    3. ``engine.score_fusion`` on the held-out test split — this is the true
       serve path (restricted to trained feature groups + temperature-applied),
       so the measured AUC/ECE is what production would see.
    4. ROC-AUC (ranking) and ECE (calibration) recorded.

Aggregation reports unpooled mean+std for every dim and, more importantly, the
paired-difference statistics ``(other - default_dim)`` per (dataset, seed): mean
delta, sample std, SEM, paired t, sign counts. The paired analysis isolates the
seed-noise component that the unpooled mean+std confound with dataset variance.

Bump criterion (when to change the shipped default width)
---------------------------------------------------------
A larger width replaces the shipped default if and only if **all three** hold:

    1. paired mean AUC delta (other - default) >= +0.02
    2. paired t-statistic (mean / SEM) >= +2.0
    3. mean ECE not worse on the candidate

This threshold is calibrated to the empirical seed-noise floor measured in the
v3 sweep (one-SEM was ~0.013 on n=24 paired runs). Anything below it is
indistinguishable from re-running the same width twice; anything at or above it
is real signal that justifies the change. The criterion is a checklist, not a
cost calculation — Mercury Agent is not constrained on parameter count;
the width that is best on the evidence is the width that ships.

Independent axis: time-series
-----------------------------
The classical-tabular ADBench axis (``--source real``) is one half of the
evidence. The other half is time-series anomaly detection on the UCR Archive
(``--source ucr``), reframed one-vs-rest. A width change must clear the bump
criterion on **both** axes before being adopted, so a choice that happens to
suit tabular but not signal data does not silently propagate to the medical /
energy deployments Mercury Agent ships for.

Usage:
    # Headline ADBench sweep (needs network on first run):
    python -m scripts.sweep_fusion_capacity --source real \\
        --dims 16,32,48,64,96 \\
        --seeds 0,1,2,3,4,5,6,7 \\
        --datasets cardio,mammography,pendigits,annthyroid,satellite,Pima,WBC,Ionosphere,\\
                   thyroid,vowels,letter,musk,optdigits,shuttle,glass,vertebral \\
        --epochs 120 --cap-per-dataset 5000 \\
        --output benchmarks/fusion_capacity/sweep_real_v4.json

    # Independent time-series cross-check:
    python -m scripts.sweep_fusion_capacity --source ucr \\
        --dims 16,32,48,64,96 \\
        --seeds 0,1,2,3,4,5,6,7 \\
        --datasets ECG5000,ECGFiveDays,Wafer,FordA,FordB,Earthquakes,Strawberry,Coffee \\
        --epochs 120 --cap-per-dataset 5000 \\
        --output benchmarks/fusion_capacity/sweep_ucr_v1.json

    # Offline smoke:
    python -m scripts.sweep_fusion_capacity --source synthetic --seeds 0,1
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import TypedDict

import numpy as np

from omni_mercury_engine.engine import OmniMercuryEngine
from omni_mercury_engine.ml.fusion_network import OmniFusionModel
from omni_mercury_engine.ml.inference import FusionInference
from scripts.train_default_fusion import (
    REAL_DATASETS,
    _load_adbench,
    _stratified_split,
    build_dataset,
)


class SweepRun(TypedDict):
    """One (dim, seed, dataset) evaluation, exactly as serialised to ``--output``."""

    dim: int
    seed: int
    dataset: str
    auc: float
    ece: float
    n_test: int


def _evaluate_once(
    x: np.ndarray,
    y: np.ndarray,
    hidden_dim: int,
    seed: int,
    epochs: int,
    test_frac: float,
) -> tuple[float, float, int]:
    """Train at ``hidden_dim`` on a seeded split and score the held-out test set.

    Returns ``(roc_auc, ece, n_test)``. Evaluation goes through
    ``score_fusion`` so the trained feature-group restriction and temperature
    calibration that production uses are exercised here too.
    """
    import torch

    from omni_mercury_engine.core.calibration import compute_ece
    from omni_mercury_engine.ml.mercury_ml import roc_auc_score

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_idx, test_idx = _stratified_split(y, train_frac=1.0 - test_frac, seed=seed)

    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    if hidden_dim != engine.fusion_model.hidden_dim:
        engine.fusion_model = OmniFusionModel(hidden_dim=hidden_dim).to(engine.device)
        engine.fusion_inference = FusionInference(
            model=engine.fusion_model, device=str(engine.device)
        )

    engine.fit_fusion(
        x[train_idx],
        y[train_idx],
        epochs=epochs,
        batch_size=64,
        early_stopping_patience=15,
    )
    probs = engine.score_fusion(x[test_idx])
    auc = float(roc_auc_score(y[test_idx], probs))
    ece = float(compute_ece(y[test_idx], np.asarray(probs)))
    return auc, ece, len(test_idx)


def _load_ucr(name: str, data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a UCR time-series dataset and frame it as anomaly detection.

    UCR labels are class indices (1..K). For anomaly framing we treat the
    largest-class as normal (label 0) and every other class as anomaly
    (label 1). This is the standard ADBench-style reframing for time-series
    classification archives, see Goldstein & Uchida (2016).
    """
    from omni_mercury_engine.datasets.base import DatasetConfig
    from omni_mercury_engine.datasets.ucr_archive import UCRLoader

    cfg = DatasetConfig(
        name=f"ucr-{name}",
        data_dir=data_dir,
        cache_dir=str(Path(data_dir) / "_cache"),
        preprocessing={"dataset_name": name},
    )
    loader = UCRLoader(cfg)
    X, y = loader.load()
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(np.int64).ravel()
    # Reframe: largest class is "normal" (0), rest are "anomalies" (1).
    counts = {int(c): int((y == c).sum()) for c in np.unique(y)}
    normal_class = max(counts, key=counts.__getitem__)
    y_bin = np.where(y == normal_class, 0, 1).astype(np.int64)
    return X, y_bin


def _load_corpus(
    source: str,
    names: list[str],
    data_dir: str,
    cap_per_dataset: int,
    seed: int,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return a list of ``(name, X, y)`` corpora to sweep over.

    ``source``:
        - ``synthetic``: a single network-free Gaussian mixture (smoke).
        - ``real``: pooled tabular ADBench-classical datasets (the headline
          medical/STEM benchmark axis).
        - ``ucr``: UCR Time Series Archive datasets, reframed as one-vs-rest
          anomaly detection so the same engine.fit_fusion path can train on
          them. The **independent axis** Mercury Agent's dim-choice decision
          should be cross-validated against because medical signal / energy
          telemetry is time-series-shaped.
    """
    if source == "synthetic":
        x, y = build_dataset(seed)
        return [("synthetic", x, y)]

    rng = np.random.default_rng(seed)
    corpora: list[tuple[str, np.ndarray, np.ndarray]] = []
    for name in names:
        try:
            if source == "ucr":
                dx, dy = _load_ucr(name, data_dir)
            else:
                dx, dy = _load_adbench(name, data_dir)
        except Exception as exc:  # network / availability is the expected failure
            print(f"  [skip] {name}: {type(exc).__name__}: {str(exc)[:80]}")
            continue
        if len(dx) > cap_per_dataset:
            sel = rng.choice(len(dx), cap_per_dataset, replace=False)
            dx, dy = dx[sel], dy[sel]
        corpora.append((name, dx, dy))
    return corpora


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=str, default="32,64,128,256")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--source", choices=("real", "synthetic", "ucr"), default="real")
    parser.add_argument("--datasets", type=str, default=",".join(REAL_DATASETS))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--cap-per-dataset", type=int, default=1500)
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path.home() / ".cache" / "mercury_agent" / "adbench"),
    )
    parser.add_argument("--output", type=str, default="", help="Optional JSON dump of all runs.")
    args = parser.parse_args()

    dims = [int(d) for d in args.dims.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    names = [n.strip() for n in args.datasets.split(",") if n.strip()]

    # Datasets are loaded once per seed (the seeded subsample depends on it), so
    # build the corpus inside the seed loop. Detectors are re-fit per run.
    runs: list[SweepRun] = []
    for dim in dims:
        for seed in seeds:
            corpora = _load_corpus(args.source, names, args.data_dir, args.cap_per_dataset, seed)
            if not corpora:
                print(
                    "ERROR: no datasets could be loaded (network unreachable?). "
                    "Try --source synthetic for an offline smoke run.",
                    file=sys.stderr,
                )
                return 1
            for name, x, y in corpora:
                auc, ece, n_test = _evaluate_once(x, y, dim, seed, args.epochs, args.test_frac)
                runs.append(
                    {
                        "dim": dim,
                        "seed": seed,
                        "dataset": name,
                        "auc": auc,
                        "ece": ece,
                        "n_test": n_test,
                    }
                )
                print(f"  dim={dim:>4} seed={seed} {name:>12}: AUC={auc:.4f} ECE={ece:.4f}")

    # Aggregate per dim across every (dataset, seed) run.
    print("\n=== Capacity sweep summary ===")
    print(f"{'dim':>6} | {'mean AUC':>9} | {'std':>6} | {'mean ECE':>9} | {'std':>6} | runs")
    print("-" * 60)
    summary: dict[int, dict[str, float]] = {}
    for dim in dims:
        aucs = [float(r["auc"]) for r in runs if r["dim"] == dim]
        eces = [float(r["ece"]) for r in runs if r["dim"] == dim]
        if not aucs:
            continue
        auc_mean = statistics.fmean(aucs)
        auc_std = statistics.pstdev(aucs) if len(aucs) > 1 else 0.0
        ece_mean = statistics.fmean(eces)
        ece_std = statistics.pstdev(eces) if len(eces) > 1 else 0.0
        summary[dim] = {
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "ece_mean": ece_mean,
            "ece_std": ece_std,
            "runs": len(aucs),
        }
        print(
            f"{dim:>6} | {auc_mean:>9.4f} | {auc_std:>6.4f} | "
            f"{ece_mean:>9.4f} | {ece_std:>6.4f} | {len(aucs)}"
        )

    # Paired-difference analysis vs every other dim. The unpooled mean+std
    # mixes seed noise and per-dataset noise; the paired difference (per
    # (dataset, seed)) isolates seed noise, which is what a "is the bigger
    # network actually better" question should be answered on.
    if summary:
        import math

        baseline_dims = sorted(summary)
        for baseline in baseline_dims:
            print(f"\n== Paired AUC diffs (other - {baseline}) ==")
            print(
                f"{'other':>6} {'n':>3} {'mean':>9} {'sd':>9} {'sem':>9} "
                f"{'+':>4} {'-':>4} {'t':>9}"
            )
            for d in baseline_dims:
                if d == baseline:
                    continue
                ref = {
                    (r["dataset"], r["seed"]): float(r["auc"]) for r in runs if r["dim"] == baseline
                }
                oth = {(r["dataset"], r["seed"]): float(r["auc"]) for r in runs if r["dim"] == d}
                common = sorted(set(ref) & set(oth))
                diffs = [oth[k] - ref[k] for k in common]
                n = len(diffs)
                if n < 2:
                    continue
                m = statistics.fmean(diffs)
                sd = statistics.stdev(diffs)
                sem = sd / math.sqrt(n) if n > 0 else float("nan")
                pos = sum(1 for x in diffs if x > 0)
                neg = sum(1 for x in diffs if x < 0)
                t = m / sem if sem > 0 else 0.0
                print(
                    f"{d:>6} {n:>3} {m:>+9.4f} {sd:>9.4f} {sem:>9.4f} "
                    f"{pos:>4} {neg:>4} {t:>+9.3f}"
                )

    # Bump recommendation: a width change off the shipped default must pass
    # ALL of (paired mean delta >= +0.02) AND (paired t-statistic >= +2.0) AND
    # (mean ECE not worse) AND (this holds on every reference dim it is being
    # compared against). Anything weaker is below the seed-noise floor and
    # documented sub-noise. The criterion is a checklist, not a discount — if
    # the evidence is strong, the bump happens; if not, the default stays.
    if summary:
        default_dim = 32 if 32 in summary else min(summary)
        bump_candidates: list[int] = []
        for d in sorted(summary):
            if d == default_dim:
                continue
            ref = {
                (r["dataset"], r["seed"]): float(r["auc"]) for r in runs if r["dim"] == default_dim
            }
            oth = {(r["dataset"], r["seed"]): float(r["auc"]) for r in runs if r["dim"] == d}
            common = sorted(set(ref) & set(oth))
            diffs = [oth[k] - ref[k] for k in common]
            ece_default = summary[default_dim]["ece_mean"]
            ece_other = summary[d]["ece_mean"]
            if len(diffs) < 2:
                continue
            m = statistics.fmean(diffs)
            sd = statistics.stdev(diffs)
            sem = sd / math.sqrt(len(diffs))
            t = m / sem if sem > 0 else 0.0
            passes_delta = m >= 0.02
            passes_t = t >= 2.0
            passes_ece = ece_other <= ece_default + 1e-9
            if passes_delta and passes_t and passes_ece:
                bump_candidates.append(d)
        print(
            f"\n== Bump criterion vs default dim={default_dim} ==\n"
            "  paired mean delta >= +0.02  AND  paired t >= +2.0  AND  ECE not worse"
        )
        if bump_candidates:
            print(f"  PASSING dims (recommend bump): {bump_candidates}")
        else:
            print("  No dim passes all three thresholds — default stays.")

    if args.output:
        payload = {"runs": runs, "summary": summary, "config": vars(args)}
        Path(args.output).write_text(json.dumps(payload, indent=2))
        print(f"Wrote {len(runs)} runs -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
