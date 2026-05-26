"""
Train and ship the default fusion checkpoint (Issue #2).

Produces the versioned checkpoint loaded by ``detect``/``serve`` so a fresh
install scores with a trained, calibrated fusion network out of the box, with
no training step required of the user.

The checkpoint is trained via the real raw path (``engine.fit_fusion``) on a
genuinely-labelled ADBench dataset (ground-truth labels), so it inherits the
FocalLoss training and post-hoc temperature calibration. The trained fusion
head operates on detector *feature* groups (fixed-dimensional regardless of
input dimensionality), so it applies to arbitrary detection input; users
should still retrain on their own domain via ``mercury-agent train`` for best
results.

Usage::

    python -m scripts.train_default_fusion                # default dataset
    python -m scripts.train_default_fusion --dataset cardio --epochs 80
    python -m scripts.train_default_fusion --output /tmp/ckpt.pt

Requires network access to download the ADBench dataset (cached after first
run).

Mercury Agent - Copyright (C) 2025 Steel Security Advisors LLC
Licensed under GNU GPL v3
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("omni_mercury_engine").setLevel(logging.WARNING)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="cardio",
        help="Genuinely-labelled ADBench dataset to train on (default: cardio)",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: the packaged default checkpoint location)",
    )
    args = parser.parse_args()

    import torch

    from omni_mercury_engine.datasets.adbench import ADBenchLoader
    from omni_mercury_engine.datasets.base import DatasetConfig
    from omni_mercury_engine.engine import DEFAULT_FUSION_CHECKPOINT, OmniMercuryEngine
    from omni_mercury_engine.ml.mercury_ml import roc_auc_score

    output = args.output or str(DEFAULT_FUSION_CHECKPOINT)

    print(f"Training default fusion checkpoint on ADBench '{args.dataset}' ...")
    loader = ADBenchLoader(DatasetConfig(name="adbench", preprocessing={"dataset": args.dataset}))
    loader.download()
    data = loader.load()
    X = np.asarray(data[0], dtype=np.float32)
    y = np.asarray(data[1]).astype(int).ravel()

    # Stratified 70/30 split for an honest held-out AUC report.
    rng = np.random.RandomState(args.seed)
    tr, te = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * 0.7))
        tr += idx[:cut].tolist()
        te += idx[cut:].tolist()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    metrics = engine.fit_fusion(
        X[tr], y[tr], epochs=args.epochs, batch_size=64, early_stopping_patience=15
    )

    auc = roc_auc_score(y[te], engine.score_fusion(X[te]))
    print(
        f"  trained: best_loss={metrics['best_loss']:.4f}, "
        f"T={metrics.get('temperature')}, held-out AUC={auc:.4f}, "
        f"ECE {metrics.get('ece_before')}->{metrics.get('ece_after')}"
    )

    from pathlib import Path

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    engine.save_model(output)
    print(f"Saved default fusion checkpoint to {output}")

    # Sanity: a fresh engine loads it and reports trained.
    fresh = OmniMercuryEngine(mode="fusion", device="cpu")
    fresh.load_model(output)
    assert fresh._fusion_trained, "fresh engine should report trained after load"
    print("Verified: fresh engine loads the checkpoint and reports trained.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
