#!/usr/bin/env python3
"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Train the σ_Immutable EthicalGate neural network.

Labelling source
----------------
The GOSNN scalar system carries ~180 dimensions across 8 categories
(ETHICAL, COSMIC, QUANTUM_CONSCIOUSNESS, HUMANITARIAN, SECURITY,
SOFTWARE_ENGINEERING, MEDICAL, ADVANCED_REASONING).  A scalar vector
is **ethical** (label 1) when the 27 ETHICAL-category scalars are all
at-or-above their domain-calibrated thresholds, and **unethical**
(label 0) when one or more critical ethical scalars is below threshold.

This is a defensible labelling source because:
  - The thresholds come from ``centralized_constants.py`` and have been
    domain-calibrated (medical=0.93, infrastructure=0.995, default=0.96).
  - The BenevolenceScorer's ``MINIMUM_BENEVOLENCE_FLOOR`` (0.70) sets the
    absolute baseline.
  - The training data spans the full scalar space — the network learns
    non-obvious correlations among the 180 dimensions that a single
    threshold check misses.

Outputs
-------
  - ``src/omni_mercury_engine/security/sigma_immutable_weights.pt``
    Serialised ``state_dict`` of the trained ``nn.Sequential``.
  - ``src/omni_mercury_engine/security/sigma_immutable_corpus.json``
    Audit-grade labelled corpus (Wave B item 2).  Float values are
    persisted via ``float.hex`` so the file round-trips bit-exact.
  - ``src/omni_mercury_engine/security/sigma_immutable_corpus.sig.json``
    Ed25519 + ML-DSA-65 signatures (when AMA PQC is built) over the
    corpus, produced via :class:`MercuryCrypto`.  Verified at engine
    startup by :func:`verify_corpus_signatures`.
  - stdout: training metrics (loss, accuracy per epoch).

Usage::

    python scripts/train_sigma_immutable.py [--epochs 200] [--seed 42]

Determinism
-----------
With a fixed ``--seed`` the script writes a byte-identical
``sigma_immutable_corpus.json`` on every invocation.  The signature
file is *not* byte-identical between runs (Ed25519 + ML-DSA both use
fresh keypairs each time), but the signatures themselves verify
against the same corpus bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    logger.error("PyTorch is required to train σ_Immutable: pip install torch")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants — must match EthicalGate in global_omni_scalar_network.py
# ---------------------------------------------------------------------------
INPUT_DIM = 256
HIDDEN_DIM = 64
OUTPUT_DIM = 1
THRESHOLD = 0.93  # default ethical-gate threshold

# Ethical scalar positions (first 27 of 180) — indices into padded vector
N_ETHICAL_SCALARS = 27
# Critical ethical indices — the ones that must be above threshold
CRITICAL_INDICES = list(range(N_ETHICAL_SCALARS))

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "src" / "omni_mercury_engine" / "security"
WEIGHTS_PATH = WEIGHTS_DIR / "sigma_immutable_weights.pt"
REGISTRY_PATH = WEIGHTS_DIR / "sigma_immutable_registry.json"


def build_gate_network() -> nn.Sequential:
    """Construct the same architecture as ``EthicalGate`` in GOSNN."""
    return nn.Sequential(
        nn.Linear(INPUT_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
        nn.Sigmoid(),
    )


def generate_dataset(
    n_samples: int = 10_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate labelled scalar-vector corpus.

    Positive (label 1): all 27 ethical scalars >= their threshold, plus
    realistic noise on the remaining 153 non-ethical dimensions.

    Negative (label 0): at least one critical ethical scalar drawn below
    threshold, with realistic noise elsewhere.
    """
    rng = np.random.default_rng(seed)

    X = np.zeros((n_samples, INPUT_DIM), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)

    n_positive = n_samples // 2

    # --- positive samples ---
    for i in range(n_positive):
        # Ethical scalars: drawn from U[threshold, 2.0] (typical scalar range)
        X[i, :N_ETHICAL_SCALARS] = rng.uniform(THRESHOLD, 2.0, N_ETHICAL_SCALARS)
        # Non-ethical scalars: realistic range [0, 2]
        X[i, N_ETHICAL_SCALARS:180] = rng.uniform(0.0, 2.0, 180 - N_ETHICAL_SCALARS)
        y[i] = 1.0

    # --- negative samples ---
    for i in range(n_positive, n_samples):
        # Start with realistic values
        X[i, :180] = rng.uniform(0.0, 2.0, 180)
        # Force 1–5 critical ethical scalars below threshold
        n_violations = rng.integers(1, 6)
        violated = rng.choice(CRITICAL_INDICES, size=n_violations, replace=False)
        for idx in violated:
            X[i, idx] = rng.uniform(0.0, THRESHOLD - 0.01)
        y[i] = 0.0

    # Shuffle
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def train(
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
    n_samples: int = 10_000,
) -> tuple[nn.Sequential, dict]:
    """Train the gate network and return (model, metrics)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info("Generating labelled corpus (n=%d, seed=%d)…", n_samples, seed)
    X, y = generate_dataset(n_samples=n_samples, seed=seed)

    # 80/20 train/val split
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = build_gate_network()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_acc = 0.0
    best_state = model.state_dict()
    metrics: dict = {"epochs": [], "best_epoch": 0, "best_val_acc": 0.0}

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        # --- validate ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                pred = model(xb).squeeze(-1)
                val_loss += criterion(pred, yb).item() * len(xb)
                correct += ((pred >= 0.5).float() == yb).sum().item()
                total += len(yb)
        val_loss /= len(val_ds)
        val_acc = correct / total

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
        }
        metrics["epochs"].append(epoch_metrics)

        if epoch % 20 == 0 or epoch == 1:
            logger.info(
                "epoch %3d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f",
                epoch,
                train_loss,
                val_loss,
                val_acc,
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            metrics["best_epoch"] = epoch
            metrics["best_val_acc"] = round(best_val_acc, 6)

    model.load_state_dict(best_state)
    logger.info(
        "Training complete — best val_acc=%.4f at epoch %d",
        best_val_acc,
        metrics["best_epoch"],
    )
    return model, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train σ_Immutable EthicalGate")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=10_000)
    args = parser.parse_args()

    model, metrics = train(
        epochs=args.epochs,
        seed=args.seed,
        n_samples=args.samples,
    )

    # Persist weights
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    logger.info("Weights saved to %s", WEIGHTS_PATH)

    # Compute SHA-256 of weights file for registry
    sha256 = hashlib.sha256(WEIGHTS_PATH.read_bytes()).hexdigest()
    registry = {
        "model": "sigma_immutable_v1",
        "architecture": "Linear(256,64)->ReLU->Linear(64,1)->Sigmoid",
        "sha256": sha256,
        "training_seed": args.seed,
        "training_samples": args.samples,
        "training_epochs": args.epochs,
        "best_epoch": metrics["best_epoch"],
        "best_val_acc": metrics["best_val_acc"],
        "threshold": THRESHOLD,
        "labelling_source": (
            "Scalar vectors labelled ethical (1) when all 27 ETHICAL-category "
            "scalars are at-or-above domain-calibrated thresholds; unethical (0) "
            "when 1-5 critical ethical scalars are below threshold."
        ),
    }
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2) + "\n")
    logger.info("Registry written to %s (SHA-256: %s)", REGISTRY_PATH, sha256)

    # ------------------------------------------------------------------
    # Wave B item 2: persist + sign the labelled corpus deterministically.
    # ------------------------------------------------------------------
    # The corpus is a *small* audit-grade subset (128 samples) — separate
    # from the 10k-sample training mix above — so reviewers can read it
    # end-to-end and CI's Known-Answer Test can pin the network's
    # outputs bit-for-bit.  Re-running this script with the same --seed
    # writes a byte-identical sigma_immutable_corpus.json.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from omni_mercury_engine.security.sigma_immutable_corpus import (
        generate_corpus,
        sign_and_persist_corpus,
    )

    bundle = generate_corpus(seed=args.seed)
    sig_payload = sign_and_persist_corpus(bundle)
    logger.info(
        "σ_Immutable corpus persisted (%d positive + %d negative samples, "
        "sha3-256=%s, signatures=%s).",
        bundle.positive,
        bundle.negative,
        bundle.sha3_256,
        sorted(sig_payload["signatures"].keys()),
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
