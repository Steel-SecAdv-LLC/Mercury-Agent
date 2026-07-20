#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Train the σ_Immutable EthicalGate neural network.

Labelling source (harvested config-integrity corpus)
----------------------------------------------------
The σ_Immutable gate is a **config-integrity / tamper check**: on a healthy
system the 127 operational governance scalars are bit-constant, and the gate's
job is to recognise that intact configuration and refuse when it is corrupted.
Training therefore uses the **real** harvested intact vector
(``scripts/harvest_sigma_baseline.py`` → ``sigma_immutable_baseline.json``),
not synthetic ``U[0,2]`` noise:

  - **Positives (intact):** the exact harvested baseline plus intact
    variations that hold the 24 critical ethical *anchors* at-or-above
    threshold, while the narrative-tuning scalars and the non-ethical
    operational band vary across their real ranges.  Labelling matches the
    deterministic floor's real 24-anchor contract — not the earlier
    all-27-ethical rule, which mislabelled the real (narrative-low) config.
  - **Negatives (tampered):** real corruptions of an intact draw — anchor
    collapse (1..5 anchors below threshold), plus gross-outlier leaks (the
    F10 timestamp class) and band corruption.

Because the exact baseline is a training positive, the network passes the
real production vector **by construction** — a ``main()`` DoS guard asserts
``score(baseline) >= threshold`` and refuses to ship otherwise.

Why the learned score stays advisory, not authoritative
-------------------------------------------------------
The deterministic critical-ethical floor (``SigmaImmutableGate.enforce_ethical_floor``)
is composed BEFORE the learned score at every boundary and remains the
authoritative gate — a collapsed anchor is a categorical fail-closed refusal no
learned score can override.  The harvested corpus makes the *advisory* network
agree with the floor on real anchor collapse and recognise the real intact
config, closing the synthetic-data gap tracked in ``docs/DORMANCY_LEDGER.md``
without changing the gate's ratified constant-score / config-integrity posture.

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
from typing import Any

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
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Generate the labelled scalar-vector corpus from the harvested baseline.

    Positive (label 1, *intact*): the real harvested configuration plus
    intact variations holding every critical ethical **anchor** at-or-above
    threshold (the deterministic floor's contract); the exact baseline is
    always present so the trained network passes the real production vector
    by construction (no DoS).

    Negative (label 0, *tampered*): real corruptions — anchor collapse plus a
    minority of gross-outlier / band-corruption mutations.

    The construction is shared with the signed audit corpus via
    :func:`build_integrity_samples`, so the trainer and the corpus can never
    label differently.  Falls back to the earlier synthetic scheme only if
    the baseline artifact is absent (with a warning).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    try:
        from omni_mercury_engine.security.sigma_immutable_corpus import (
            build_integrity_samples,
            load_baseline,
        )

        baseline = load_baseline()
    except (FileNotFoundError, ValueError) as exc:  # pragma: no cover - build guard
        logger.error(
            "σ baseline unavailable (%s); run scripts/harvest_sigma_baseline.py first.", exc
        )
        raise SystemExit(1) from exc

    n_positive = n_samples // 2
    X, y = build_integrity_samples(
        baseline,
        seed=seed,
        n_positive=n_positive,
        n_negative=n_samples - n_positive,
        threshold=THRESHOLD,
    )
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def train(
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
    n_samples: int = 10_000,
) -> tuple[nn.Sequential, dict[str, Any]]:
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
    metrics: dict[str, Any] = {"epochs": [], "best_epoch": 0, "best_val_acc": 0.0}

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
    # Defaults reproduce the shipped weights: the harvested anchor-faithful
    # corpus needs more data/epochs than the old synthetic scheme for the
    # shallow gate to crisply isolate the 24 anchor positions from the 3
    # narrative dims in the same band (so a generic all-anchors-high vector
    # passes while the narrative-low real baseline also passes robustly).
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=40_000)
    args = parser.parse_args()

    model, metrics = train(
        epochs=args.epochs,
        seed=args.seed,
        n_samples=args.samples,
    )

    # ------------------------------------------------------------------
    # DoS guard: the trained network MUST pass the real harvested intact
    # config vector, or every production detect_with_fusion call would
    # fail-closed.  This is the invariant the harvested corpus exists to
    # guarantee; assert it before shipping the weights.
    # ------------------------------------------------------------------
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from omni_mercury_engine.security.sigma_immutable_corpus import load_baseline

    baseline = load_baseline()
    baseline_vec = np.zeros(INPUT_DIM, dtype=np.float32)
    baseline_vec[: len(baseline.values)] = baseline.values.astype(np.float32)
    model.eval()
    with torch.no_grad():
        baseline_score = float(model(torch.from_numpy(baseline_vec)).squeeze(-1).item())
    logger.info("DoS guard — trained score on the real intact baseline: %.6f", baseline_score)
    if baseline_score < THRESHOLD:
        logger.error(
            "REFUSING to ship: retrained σ network scores the real intact config %.6f < %.2f — "
            "this would DoS production. Adjust the corpus/epochs and retry.",
            baseline_score,
            THRESHOLD,
        )
        return 1

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
            "Harvested config-integrity corpus: positives are the real intact "
            "operational vector (scripts/harvest_sigma_baseline.py) plus intact "
            "variations holding the 24 critical ethical anchors at-or-above "
            "threshold; negatives are real tamper mutations (anchor collapse, "
            "gross-outlier leak, band corruption). The exact baseline is a "
            "training positive, so the network passes the real production "
            "configuration by construction (DoS guard enforced in main())."
        ),
        "baseline_sigma_score": round(baseline_score, 10),
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
