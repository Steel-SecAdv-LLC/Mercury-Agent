# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Train and persist the shipped default fusion checkpoint.

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

Two data sources (``--source``):

* ``synthetic`` (default) — a reproducible, network-free mixture of normal
  Gaussian clusters and injected anomalies. This is what ships as the
  **committed, deterministic** default checkpoint: same seed → same weights,
  hash-pinnable, works offline, and auditable for supply-chain purposes.
* ``real`` — an **opt-in** pooled prior over genuinely-labelled ADBench
  datasets (``engine.fit_fusion_pooled``). It needs network access on first
  run and is **not committed** (its weights depend on downloaded data and so
  are not bit-reproducible across machines). Operators who want real-data
  weights out of the box run this script with
  ``--source real``; everyone else gets the deterministic synthetic default
  and can still fine-tune via ``mercury-agent train`` on their own corpus.

Either way the checkpoint is calibrated (temperature scaling) and records its
provenance (source, datasets, seed) so a shipped artifact is self-describing.

Reproducibility: the synthetic path is fully seeded. ``hidden_dim`` defaults to
64 — raised from 32 after production-axis analysis showed the live benchmark
suite operates at up to 1555 features and 620K samples per dataset, making
the 32-dim bottleneck a 48:1 compression on real inputs. dim=64 provides 2×
encoder capacity at 3.2× param cost (0.71 MB fp32) and is the evidence-backed
transitional default. See ``benchmarks/fusion_capacity/README.md`` for the
full cost/stability/OOD analysis.

Usage:
    python -m scripts.train_default_fusion                      # synthetic default
    python -m scripts.train_default_fusion --source real        # opt-in ADBench prior
    python -m scripts.train_default_fusion --epochs 150 --output /tmp/f.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from omni_mercury_engine.engine import OmniMercuryEngine, default_fusion_checkpoint_path
from omni_mercury_engine.ml.fusion_network import OmniFusionModel
from omni_mercury_engine.ml.inference import FusionInference

SEED = 20260526
HIDDEN_DIM = 64
N_FEATURES = 16

# Genuinely-labelled ADBench datasets (external ground-truth labels) spanning a
# range of sizes, dimensionalities and anomaly ratios. Used only for the opt-in
# ``--source real`` prior; never the self-labelled (threshold-derived) loaders,
# whose AUC is inflated by label leakage and would teach the prior nothing
# honest.
REAL_DATASETS = (
    "cardio",
    "mammography",
    "pendigits",
    "annthyroid",
    "satellite",
    "Pima",
    "WBC",
    "Ionosphere",
)


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


def _load_adbench(name: str, data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one genuinely-labelled ADBench dataset (z-scored).

    Resolves through the registry name ``adbench-<name>`` so the corrected
    :class:`ADBenchLoader` loads the right NPZ (the loader historically
    collapsed every name to ``fraud``; see datasets/adbench.py). Returns
    ``(X, y)`` with float32 features and binary int labels.
    """
    from omni_mercury_engine.datasets.adbench import ADBenchLoader
    from omni_mercury_engine.datasets.base import DatasetConfig

    cfg = DatasetConfig(
        name=f"adbench-{name.lower()}",
        data_dir=data_dir,
        cache_dir=str(Path(data_dir) / "_cache"),
        download=True,
    )
    loader = ADBenchLoader(cfg)
    loader.download()
    raw_x, raw_y = loader._load_raw()
    x = loader.preprocess(raw_x).astype(np.float32)
    y = (np.asarray(raw_y).ravel() > 0).astype(np.int64)
    return x, y


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
    parser.add_argument(
        "--source",
        choices=("synthetic", "real"),
        default="synthetic",
        help=(
            "Training data. 'synthetic' (default) is the fully-seeded, "
            "network-free Gaussian-mixture prior that ships as the committed, "
            "deterministic checkpoint. 'real' is an opt-in pooled prior over "
            "genuinely-labelled ADBench datasets (needs network on first run) "
            "and is intentionally NOT committed."
        ),
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(REAL_DATASETS),
        help="Comma-separated ADBench dataset names for --source real.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path.home() / ".cache" / "mercury_agent" / "adbench"),
        help="Where real ADBench datasets are downloaded / cached (--source real).",
    )
    parser.add_argument(
        "--cap-per-dataset",
        type=int,
        default=1500,
        help="Max samples kept per ADBench dataset (seeded subsample) for --source real.",
    )
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

    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    # ``load_model`` rebuilds the engine's fusion model from the checkpoint's
    # ``hidden_dim``/``feature_dims``, so any width here is self-describing on load.
    if args.hidden_dim != engine.fusion_model.hidden_dim:
        engine.fusion_model = OmniFusionModel(hidden_dim=args.hidden_dim).to(engine.device)
        engine.fusion_inference = FusionInference(
            model=engine.fusion_model, device=str(engine.device)
        )

    if args.source == "synthetic":
        x, y = build_dataset(args.seed)
        print(
            f"Synthetic dataset: {x.shape[0]} samples, {x.shape[1]} features, "
            f"{int(y.sum())} anomalies ({y.mean():.1%})."
        )
        train_idx, test_idx = _stratified_split(y, train_frac=0.7, seed=args.seed)
        print(
            "Training via fit_fusion (full inference feature pipeline + FocalLoss + "
            "calibration + adaptive neuro-symbolic co-training)..."
        )
        metrics = engine.fit_fusion(
            x[train_idx],
            y[train_idx],
            epochs=args.epochs,
            batch_size=64,
            early_stopping_patience=15,
            # Evidence-backed default (benchmarks/neurosymbolic_ablation.py): the
            # label-scarcity schedule dominates neural-only, so the shipped model
            # is co-trained, decaying to the neural path when labels are abundant.
            symbolic_weight="adaptive",
        )
        auc = float(roc_auc_score(y[test_idx], engine.score_fusion(x[test_idx])))
        print(
            f"  trained: best_loss={metrics['best_loss']:.4f}, "
            f"T={metrics.get('temperature')}, held-out AUC={auc:.4f}, "
            f"ECE {metrics.get('ece_before')}->{metrics.get('ece_after')}, "
            f"symbolic_λ={metrics.get('symbolic_weight_resolved')}, "
            f"groups={engine._fusion_feature_groups}"
        )
        provenance: dict[str, object] = {
            "source": "synthetic_gaussian_mixture",
            "datasets": [],
            "seed": int(args.seed),
            "hidden_dim": int(args.hidden_dim),
        }
    else:  # real: opt-in pooled ADBench prior
        names = [n.strip() for n in args.datasets.split(",") if n.strip()]
        print(
            f"Opt-in REAL prior: pooling genuinely-labelled ADBench {names} (cache: {args.data_dir})."
        )
        print(
            "NOTE: this regenerates the working-tree checkpoint with non-committed "
            "real-data weights; the committed deterministic default stays in git."
        )
        rng = np.random.default_rng(args.seed)
        datasets: list[tuple[np.ndarray, np.ndarray]] = []
        used: list[str] = []
        for name in names:
            try:
                dx, dy = _load_adbench(name, args.data_dir)
            except Exception as exc:  # network / availability is the expected failure
                print(f"  [skip] {name}: {type(exc).__name__}: {str(exc)[:80]}")
                continue
            if len(dx) > args.cap_per_dataset:
                sel = rng.choice(len(dx), args.cap_per_dataset, replace=False)
                dx, dy = dx[sel], dy[sel]
            print(f"  [{name}] {len(dx)} samples x {dx.shape[1]} feats, {int(dy.sum())} anomalies")
            datasets.append((dx, dy))
            used.append(name)
        if not datasets:
            print(
                "ERROR: no real datasets could be loaded (network unreachable?). "
                "Use --source synthetic for the offline deterministic default.",
                file=sys.stderr,
            )
            return 1
        metrics = engine.fit_fusion_pooled(
            datasets,
            epochs=args.epochs,
            batch_size=64,
            early_stopping_patience=15,
        )
        print(
            f"  trained on {metrics['pooled_datasets']} datasets / "
            f"{metrics['pooled_samples']} pooled samples: best_loss={metrics['best_loss']:.4f}, "
            f"T={metrics.get('temperature')}, "
            f"val_ECE {metrics.get('ece_before')}->{metrics.get('ece_after')}, "
            f"groups={metrics['pooled_groups']}"
        )
        provenance = {
            "source": "adbench",
            "datasets": used,
            "seed": int(args.seed),
            "hidden_dim": int(args.hidden_dim),
            "cap_per_dataset": int(args.cap_per_dataset),
        }

    engine._fusion_provenance = provenance

    out = args.output
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    engine.save_model(out)
    size_kb = Path(out).stat().st_size / 1024
    print(f"Saved default fusion checkpoint -> {out} ({size_kb:.1f} KB)")

    # Sanity: a fresh engine loads it, reports trained, and restores calibration
    # + provenance — exactly the contract detect_with_fusion relies on.
    fresh = OmniMercuryEngine(mode="fusion", device="cpu")
    fresh.load_model(out)
    assert fresh._fusion_trained, "fresh engine should report trained after load"
    assert fresh._fusion_calibrator is not None, "calibrator should be restored on load"
    assert fresh._fusion_provenance == provenance, "provenance should round-trip via the checkpoint"
    print(
        "Verified: fresh engine loads the checkpoint, reports trained, restores "
        f"calibrator + provenance (source={fresh._fusion_provenance['source']})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
