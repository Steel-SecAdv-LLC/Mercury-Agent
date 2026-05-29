"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.

Capacity sweep for the fusion network's ``hidden_dim`` — the evidence behind the
default checkpoint's width.

Why this exists
---------------
``hidden_dim`` should be chosen on held-out performance, not on whatever keeps an
artifact under a pre-commit large-file cap. A single training run cannot justify
a width: anomaly-detection AUC has real seed-to-seed variance, so a one-shot
"128 beats 32 by 0.014" can be pure noise. This harness measures each candidate
width **across multiple seeds and multiple genuinely-labelled datasets**, then
reports mean +/- std so the decision rests on a signal, not a sample of one.

Protocol (per ``dim`` x ``seed`` x ``dataset``)
    1. Stratified train/test split (seeded).
    2. ``engine.fit_fusion`` on the train split — fits detectors on train only,
       trains the head with FocalLoss, fits temperature calibration.
    3. ``engine.score_fusion`` on the held-out test split — this is the true
       serve path (restricted to trained feature groups + temperature-applied),
       so the measured AUC/ECE is what production would see.
    4. ROC-AUC (ranking) and ECE (calibration) recorded.

Aggregation pools every (dataset, seed) run for a width into mean/std, and a
**parsimony recommendation** picks the smallest width whose mean AUC is within
one standard deviation of the best width's mean — i.e. do not pay for capacity
that is not measurably better. The recommendation is advisory; the full per-run
table and an optional JSON dump are emitted for inspection.

Usage:
    # Real evidence (needs network for ADBench on first run):
    python -m scripts.sweep_fusion_capacity --dims 32,64,128,256 --seeds 0,1,2,3,4
    # Offline smoke (synthetic saturates ~1.0 AUC; proves the harness runs):
    python -m scripts.sweep_fusion_capacity --source synthetic --seeds 0,1
    python -m scripts.sweep_fusion_capacity --output sweep_results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

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


def _load_corpus(
    source: str, names: list[str], data_dir: str, cap_per_dataset: int, seed: int
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return a list of ``(name, X, y)`` corpora to sweep over."""
    if source == "synthetic":
        x, y = build_dataset(seed)
        return [("synthetic", x, y)]

    rng = np.random.default_rng(seed)
    corpora: list[tuple[str, np.ndarray, np.ndarray]] = []
    for name in names:
        try:
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
    parser.add_argument("--source", choices=("real", "synthetic"), default="real")
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
    runs: list[dict[str, object]] = []
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

    # Parsimony recommendation: smallest dim whose mean AUC is within one std of
    # the best dim's mean. "Within noise of the best" -> do not buy capacity that
    # is not measurably better. Advisory only.
    if summary:
        best_dim = max(summary, key=lambda d: summary[d]["auc_mean"])
        best_mean = summary[best_dim]["auc_mean"]
        best_std = summary[best_dim]["auc_std"]
        within = [d for d in sorted(summary) if summary[d]["auc_mean"] >= best_mean - best_std]
        recommended = within[0] if within else best_dim
        print(
            f"\nBest mean AUC: dim={best_dim} ({best_mean:.4f} +/- {best_std:.4f}).\n"
            f"Parsimony pick (smallest within 1 std of best): dim={recommended}.\n"
            "Advisory only — inspect per-dataset ECE and runs before changing the default."
        )

    if args.output:
        payload = {"runs": runs, "summary": summary, "config": vars(args)}
        Path(args.output).write_text(json.dumps(payload, indent=2))
        print(f"Wrote {len(runs)} runs -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
