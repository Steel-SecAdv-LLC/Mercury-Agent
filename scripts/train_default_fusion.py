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
a *trained* fusion network out of the box instead of the random-init network the
previous default shipped.

Critically, the network is trained on the **exact feature pipeline that
``detect_with_fusion`` uses at inference** — the concatenation of all base
detector features and every domain-model feature that survives tabular input
(extracted via ``OmniMercuryEngine._extract_features_parallel``). Training on
detector features alone (as ``fit_fusion`` does) would leave the model facing a
different, wider feature distribution at inference, so its trained weights would
not transfer. Matching the pipeline is what makes the shipped checkpoint
genuinely improve the headline fusion path rather than merely loading.

It is a generic tabular baseline trained on a reproducible synthetic mixture of
normal Gaussian clusters and injected anomalies; it is not a substitute for
fitting on real domain data. Operators should fine-tune with
``mercury-agent train`` on their own corpus.

Reproducibility: dataset and training are fully seeded. The compact
(``hidden_dim=32``) network keeps the artifact under the repository's 1 MB
large-file limit.

Usage:
    python -m scripts.train_default_fusion
    python -m scripts.train_default_fusion --epochs 120 --output /tmp/f.pt
"""

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


def extract_inference_features(
    engine: OmniMercuryEngine, x: np.ndarray
) -> dict[str, "object"]:
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


def train(
    model: OmniFusionModel,
    features: dict[str, "object"],
    y: np.ndarray,
    epochs: int,
    seed: int = SEED,
) -> float:
    """Supervised BCE training of the fusion network on pre-extracted features."""
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
            vloss = torch.nn.functional.binary_cross_entropy(
                model(vbatch)["anomaly_probs"], labels[val_idx]
            ).item()
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
    return best_val


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument(
        "--output",
        type=str,
        default=str(default_fusion_checkpoint_path()),
        help="Checkpoint output path (defaults to the in-tree shipped path).",
    )
    args = parser.parse_args()

    import torch

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    x, y = build_dataset()
    print(f"Dataset: {x.shape[0]} samples, {x.shape[1]} features, "
          f"{int(y.sum())} anomalies ({y.mean():.1%}).")

    engine = OmniMercuryEngine(mode="fusion")
    print("Extracting full inference feature pipeline (detector + model features)...")
    features = extract_inference_features(engine, x)
    print(f"Feature groups ({len(features)}): {sorted(features)}")

    model = OmniFusionModel(hidden_dim=HIDDEN_DIM).to(engine.device)
    best_val = train(model, features, y, epochs=args.epochs)
    print(f"Training complete: best_val_loss={best_val:.4f}")

    # Install the trained model on the engine and persist via the engine so
    # the checkpoint carries feature_dims + the dynamic-projection registry.
    engine.fusion_model = model
    engine.fusion_inference = FusionInference(model=model, device=str(engine.device))
    engine._fusion_trained = True

    out = args.output
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    engine.save_model(out)
    print(f"Saved default fusion checkpoint -> {out} ({Path(out).stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
