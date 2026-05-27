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

Train and persist the shipped default fusion checkpoint.

The checkpoint produced here gives ``mercury-agent detect -d fusion`` (and any
engine consumer that calls
:meth:`OmniMercuryEngine.load_default_fusion_checkpoint`) a *trained*, calibrated
fusion network out of the box instead of the random-init network that would
otherwise ship.

The network is trained through the real :meth:`OmniMercuryEngine.fit_fusion`
path, which extracts the **exact feature set ``detect_with_fusion`` uses at
inference** (base-detector features plus every domain-model feature that
survives the input). The engine records those trained feature groups in the
checkpoint and restricts inference to them, so the shipped weights transfer to
the headline fusion path rather than merely loading. Training also applies
FocalLoss and post-hoc temperature calibration, so the shipped probabilities are
calibrated, not just well-ranked.

It is a generic tabular baseline trained on a reproducible synthetic mixture of
normal Gaussian clusters and injected anomalies; it is not a substitute for
fitting on real domain data. Operators should fine-tune with
``mercury-agent train`` on their own corpus.

Reproducibility: dataset and training are fully seeded. The compact
(``hidden_dim=32``) network keeps the artifact small.

Usage:
    python -m scripts.train_default_fusion
    python -m scripts.train_default_fusion --epochs 120 --output /tmp/f.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from omni_mercury_engine.engine import OmniMercuryEngine, default_fusion_checkpoint_path
from omni_mercury_engine.ml.fusion_network import OmniFusionModel
from omni_mercury_engine.ml.inference import FusionInference

SEED = 20260526
HIDDEN_DIM = 32
N_FEATURES = 16


def build_dataset(seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Build a reproducible labelled tabular anomaly dataset.

    Normal points are drawn from a small number of Gaussian clusters (a
    multi-modal "normal manifold"); anomalies are a mix of global outliers
    (displaced far from every cluster) and scale anomalies (inflated
    variance), the regime tabular detectors are expected to catch.
    """
    rng = np.random.default_rng(seed)

    n_normal = 1500
    n_anom = 170
    n_clusters = 4

    centers = rng.normal(0.0, 4.0, size=(n_clusters, N_FEATURES))
    assignments = rng.integers(0, n_clusters, size=n_normal)
    normal = centers[assignments] + rng.normal(0.0, 1.0, size=(n_normal, N_FEATURES))

    n_global = n_anom // 2
    n_scale = n_anom - n_global
    global_out = rng.normal(0.0, 1.0, size=(n_global, N_FEATURES)) + rng.choice(
        [-9.0, 9.0], size=(n_global, N_FEATURES)
    )
    base_centers = centers[rng.integers(0, n_clusters, size=n_scale)]
    scale_out = base_centers + rng.normal(0.0, 6.0, size=(n_scale, N_FEATURES))

    x = np.vstack([normal, global_out, scale_out]).astype(np.float32)
    y = np.concatenate([np.zeros(n_normal), np.ones(n_global + n_scale)]).astype(np.int64)

    perm = rng.permutation(len(x))
    return x[perm], y[perm]


def _stratified_split(y: np.ndarray, train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * train_frac))
        train_idx += idx[:cut].tolist()
        test_idx += idx[cut:].tolist()
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the shipped default fusion checkpoint.")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--output",
        type=str,
        default=str(default_fusion_checkpoint_path()),
        help="Checkpoint output path (defaults to the in-tree shipped path).",
    )
    args = parser.parse_args()

    import torch

    from omni_mercury_engine.ml.mercury_ml import roc_auc_score

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x, y = build_dataset(args.seed)
    print(
        f"Dataset: {x.shape[0]} samples, {x.shape[1]} features, "
        f"{int(y.sum())} anomalies ({y.mean():.1%})."
    )

    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    # Compact network keeps the shipped artifact small. ``load_model`` rebuilds
    # the engine's fusion model from the checkpoint's ``hidden_dim``/``feature_dims``,
    # so a smaller default here is fully self-describing on load.
    if args.hidden_dim != engine.fusion_model.hidden_dim:
        engine.fusion_model = OmniFusionModel(hidden_dim=args.hidden_dim).to(engine.device)
        engine.fusion_inference = FusionInference(
            model=engine.fusion_model, device=str(engine.device)
        )

    train_idx, test_idx = _stratified_split(y, train_frac=0.7, seed=args.seed)

    print("Training via fit_fusion (full inference feature pipeline + FocalLoss + calibration)...")
    metrics = engine.fit_fusion(
        x[train_idx],
        y[train_idx],
        epochs=args.epochs,
        batch_size=64,
        early_stopping_patience=15,
    )

    auc = float(roc_auc_score(y[test_idx], engine.score_fusion(x[test_idx])))
    print(
        f"  trained: best_loss={metrics['best_loss']:.4f}, "
        f"T={metrics.get('temperature')}, held-out AUC={auc:.4f}, "
        f"ECE {metrics.get('ece_before')}->{metrics.get('ece_after')}, "
        f"groups={engine._fusion_feature_groups}"
    )

    out = args.output
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    engine.save_model(out)
    size_kb = Path(out).stat().st_size / 1024
    print(f"Saved default fusion checkpoint -> {out} ({size_kb:.1f} KB)")

    # Sanity: a fresh engine loads it, reports trained, and restores calibration.
    fresh = OmniMercuryEngine(mode="fusion", device="cpu")
    fresh.load_model(out)
    assert fresh._fusion_trained, "fresh engine should report trained after load"
    assert fresh._fusion_calibrator is not None, "calibrator should be restored on load"
    print("Verified: fresh engine loads the checkpoint, reports trained, restores calibrator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
