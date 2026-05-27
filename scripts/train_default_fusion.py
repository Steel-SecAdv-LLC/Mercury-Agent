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
"""

from __future__ import annotations

"""Train and persist the shipped default fusion checkpoint.

The checkpoint produced here gives ``mercury-agent detect -d fusion`` (and any
engine consumer that calls :meth:`OmniMercuryEngine.load_default_fusion_checkpoint`)
a *trained, calibrated* fusion network out of the box instead of the random-init
network the engine would otherwise carry.

Data source (``--source``)
--------------------------
* ``real`` (default): pooled, genuinely-labelled real datasets from ADBench
  (Han et al., NeurIPS 2022 — external ground-truth labels, not thresholded
  from the scored signal). Each raw dataset is run through the **exact
  inference feature pipeline** ``detect_with_fusion`` uses, then pooled so the
  network learns a general tabular-anomaly prior. Requires network access on
  first run (datasets are cached under the loader's ``data_dir``).
* ``synthetic``: a reproducible, network-free Gaussian-mixture fallback for
  fully-offline regeneration. Lower-fidelity; use only when real data is
  unreachable.

Capacity (``hidden_dim``)
-------------------------
Defaults to **128** — ``OmniFusionModel``'s own designed default, and the
empirical capacity knee on real ADBench held-out AUC (32 underfits the harder
datasets; 256 overfits). The previous default of 32 was chosen only to keep the
artifact under a generic pre-commit large-file limit; capacity is now chosen on
the merits and the file-size concern is handled separately (the artifact is
generated at build time and not committed — see ``Makefile`` target
``checkpoint`` and ``.github/workflows/build-checkpoint.yml``).

Calibration
-----------
After training, a post-hoc temperature scalar (Guo et al., 2017) is fit on a
held-out split and stored in the checkpoint. Temperature scaling is monotonic,
so it improves ECE without changing AUC ranking.

Reproducibility: dataset selection, sampling, and training are fully seeded.

Usage:
    python -m scripts.train_default_fusion                  # real data, hidden_dim=128
    python -m scripts.train_default_fusion --source synthetic
    python -m scripts.train_default_fusion --epochs 150 --cap-per-dataset 2000
    python -m scripts.train_default_fusion --output /tmp/f.pt
"""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from omni_mercury_engine.engine import OmniMercuryEngine, default_fusion_checkpoint_path
from omni_mercury_engine.ml.fusion_network import OmniFusionModel
from omni_mercury_engine.ml.inference import FusionInference

if TYPE_CHECKING:
    import torch

SEED = 20260526
HIDDEN_DIM = 128
N_FEATURES = 16

# Genuinely-labelled ADBench datasets spanning sizes / dimensionalities /
# anomaly ratios. Only externally-labelled (ground-truth) sources are used —
# never the self-labelled (threshold-derived) loaders, whose AUC is inflated by
# label leakage and would teach the prior nothing honest.
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
    """Build a reproducible labelled tabular anomaly dataset (synthetic fallback).

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
    """Load one genuinely-labelled ADBench dataset, z-scored."""
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
    X, y = loader._load_raw()
    X = loader.preprocess(X)
    y = (np.asarray(y).ravel() > 0).astype(np.int64)
    return X.astype(np.float32), y


def extract_inference_features(engine: OmniMercuryEngine, x: np.ndarray) -> dict[str, torch.Tensor]:
    """Extract the full inference feature set per sample, stacked to (N, dim).

    Mirrors what ``detect_with_fusion`` feeds the fusion network for a single
    sample, then stacks across samples so the network trains on exactly that
    distribution. Models that aggregate over a batch (returning one row for
    many inputs) are handled correctly because each sample is extracted on its
    own (batch size 1).
    """
    import torch

    for detector in engine.detectors.values():
        if not detector.is_fitted():
            detector.fit(x)

    per_key: dict[str, list[np.ndarray]] = {}
    n = len(x)
    for i in range(n):
        feats, _ = engine._extract_features_parallel(x[i : i + 1])
        for key, value in feats.items():
            arr = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
            per_key.setdefault(key, []).append(arr.reshape(-1).astype(np.float32))
        if (i + 1) % 250 == 0:
            print(f"  extracted features for {i + 1}/{n} samples")

    # Keep only keys present for every sample so the stacked tensors align.
    keys = [k for k in per_key if len(per_key[k]) == n]
    return {k: torch.tensor(np.stack(per_key[k]), dtype=torch.float32) for k in keys}


def build_real_pool(
    datasets: tuple[str, ...],
    cap_per_dataset: int,
    data_dir: str,
    seed: int = SEED,
) -> tuple[dict[str, torch.Tensor], np.ndarray]:
    """Pool the inference feature pipeline across real ADBench datasets.

    Detectors are fit per dataset (a fresh engine per source), features are
    extracted per sample, then concatenated on the feature groups shared by
    every dataset so the stacked tensors align.
    """
    import torch

    rng = np.random.default_rng(seed)
    per_ds: list[tuple[dict[str, torch.Tensor], np.ndarray]] = []
    shared: set[str] | None = None

    for name in datasets:
        try:
            X, y = _load_adbench(name, data_dir)
        except Exception as exc:  # network / availability
            print(f"  [skip] {name}: {type(exc).__name__}: {str(exc)[:80]}")
            continue
        if len(X) > cap_per_dataset:
            sel = rng.choice(len(X), cap_per_dataset, replace=False)
            X, y = X[sel], y[sel]
        engine = OmniMercuryEngine(mode="fusion")  # fresh detectors per dataset
        print(f"  [{name}] {len(X)} samples x {X.shape[1]} feats", flush=True)
        feats = extract_inference_features(engine, X)
        per_ds.append((feats, y))
        shared = set(feats) if shared is None else (shared & set(feats))

    if not per_ds:
        raise RuntimeError(
            "No real datasets could be loaded (network unreachable?). "
            "Re-run with --source synthetic for an offline fallback."
        )

    keys = sorted(shared or set())
    pooled = {k: torch.cat([feats[k] for feats, _ in per_ds]) for k in keys}
    labels = np.concatenate([y for _, y in per_ds])
    return pooled, labels


def train(
    model: OmniFusionModel,
    features: dict[str, torch.Tensor],
    y: np.ndarray,
    epochs: int,
    seed: int = SEED,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Supervised BCE training of the fusion network on pre-extracted features.

    Returns ``(best_val_loss, val_probs, val_labels)`` — the validation
    predictions are reused to fit post-hoc temperature calibration.
    """
    import torch

    g = torch.Generator().manual_seed(seed)
    n = len(y)
    labels = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    n_val = max(1, int(n * 0.2))
    perm = torch.randperm(n, generator=g)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)

    best_val = float("inf")
    best_state = None
    patience, since_improve = 15, 0
    batch_size = 64

    for epoch in range(epochs):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), generator=g)]
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            batch = {k: v[idx] for k, v in features.items()}
            optimizer.zero_grad()
            out = model(batch)
            loss = torch.nn.functional.binary_cross_entropy(out["anomaly_probs"], labels[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vbatch = {k: v[val_idx] for k, v in features.items()}
            vout = model(vbatch)["anomaly_probs"]
            vloss = torch.nn.functional.binary_cross_entropy(vout, labels[val_idx]).item()
        scheduler.step(vloss)

        if vloss < best_val:
            best_val, since_improve = vloss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            since_improve += 1
            if since_improve >= patience:
                print(f"  early stop at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_probs = model({k: v[val_idx] for k, v in features.items()})["anomaly_probs"]
        val_probs = val_probs.cpu().numpy().reshape(-1)
    return best_val, val_probs, y[val_idx.numpy()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("real", "synthetic"), default="real")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--cap-per-dataset", type=int, default=1500)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path.home() / ".cache" / "mercury_agent" / "adbench"),
        help="Where real ADBench datasets are downloaded / cached.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(default_fusion_checkpoint_path()),
        help="Checkpoint output path (defaults to the in-tree shipped path).",
    )
    args = parser.parse_args()

    import torch

    from omni_mercury_engine.core.calibration import compute_ece
    from omni_mercury_engine.core.calibration import TemperatureScaling

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    engine = OmniMercuryEngine(mode="fusion")

    if args.source == "real":
        print(f"Building pooled real ADBench training set: {list(REAL_DATASETS)}")
        features, y = build_real_pool(REAL_DATASETS, args.cap_per_dataset, args.data_dir, seed=SEED)
        provenance_source = "adbench"
        datasets_used = list(REAL_DATASETS)
    else:
        print("Building synthetic Gaussian-mixture training set (offline fallback).")
        x, y = build_dataset()
        features = extract_inference_features(engine, x)
        provenance_source = "synthetic_gaussian_mixture"
        datasets_used = []

    feature_keys = sorted(features)
    print(
        f"Pooled training set: {len(y)} samples, {int(y.sum())} anomalies "
        f"({y.mean():.1%}); {len(feature_keys)} feature groups: {feature_keys}"
    )

    feature_dims = {k: features[k].shape[1] for k in feature_keys}
    model = OmniFusionModel(feature_dims=feature_dims, hidden_dim=args.hidden_dim).to(engine.device)
    best_val, val_probs, val_labels = train(model, features, y, epochs=args.epochs)
    print(f"Training complete: best_val_loss={best_val:.4f}")

    # Post-hoc temperature calibration (Guo et al. 2017) on the val split.
    calibrator = TemperatureScaling().fit(val_probs, val_labels)
    ece_before = compute_ece(val_labels, val_probs)
    ece_after = compute_ece(val_labels, calibrator.calibrate(val_probs))
    print(
        f"Calibration: T={calibrator.temperature:.3f}  "
        f"val_ECE {ece_before:.4f} -> {ece_after:.4f}"
    )

    engine.fusion_model = model
    engine.fusion_inference = FusionInference(model=model, device=str(engine.device))
    engine._fusion_trained = True

    out = args.output
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    engine.save_model(out)

    # Enrich the checkpoint with calibration + provenance (additive keys; the
    # current load path ignores unknown keys, so this stays backward compatible).
    checkpoint = torch.load(out, map_location="cpu", weights_only=True)
    checkpoint["temperature"] = float(calibrator.temperature)
    checkpoint["feature_keys"] = feature_keys
    checkpoint["provenance"] = {
        "source": provenance_source,
        "datasets": datasets_used,
        "hidden_dim": int(args.hidden_dim),
        "seed": SEED,
        "val_ece_before": float(ece_before),
        "val_ece_after": float(ece_after),
    }
    torch.save(checkpoint, out)

    print(f"Saved default fusion checkpoint -> {out} ({Path(out).stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
