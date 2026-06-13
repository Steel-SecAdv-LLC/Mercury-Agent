# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Schumann sub-net evaluation (WS-C) -- pre-registered, opt-in, instrumented.

Runs exactly the protocol in ``docs/SCHUMANN_PREREGISTRATION.md``:

1. Build **real** weak labels from public-domain NOAA SWPC catalogs
   (``space/schumann_labeling.py``): driver-coincident windows (Kp>=5 storms,
   M/X flares + documented lag) vs geomagnetically-quiet windows.
2. Because no openly-licensed **real** ELF corpus could be cleared + ingested in
   this environment, exercise the encoder on a **clearly-labelled synthetic,
   physically-grounded** ELF spectrum generator (Schumann peaks at 7.83 / 14.3 /
   20.8 / 27.3 / 33.8 Hz; driver windows perturb amplitude/Q/centre-frequency
   per documented flare/storm SR responses). **Synthetic -> cannot lift
   quarantine; never presented as real.**
3. Train the actual quarantined ``SchumannHarmonicAnalyzer`` (CNN encoder +
   confidence head) on the weak labels and report ROC-AUC, 3 seeds, temporal
   split.

The verdict is QUARANTINE by construction here (synthetic signal). Swapping in a
real hash-pinned ELF corpus is the only thing that can lift it.

Usage::

    python benchmarks/schumann_eval.py --n 600 --epochs 30 --out artifacts/schumann_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

import torch
from torch import nn

from omni_mercury_engine.ml.mercury_ml import roc_auc_score
from omni_mercury_engine.space.schumann_labeling import (
    fetch_catalogs,
    label_noise_disclosure,
)
from omni_mercury_engine.space.schumann_resonance import SchumannHarmonicAnalyzer

SPECTRUM_SIZE = 512
FREQ_MAX_HZ = 50.0
SCHUMANN_PEAKS_HZ = (7.83, 14.3, 20.8, 27.3, 33.8)
_AUC_LIFT_BAR = 0.70  # only meaningful on REAL ELF (see pre-registration)


def _synth_spectrum(label: int, rng: np.random.RandomState) -> np.ndarray[Any, Any]:
    """Synthetic-but-physically-grounded ELF power spectrum. NOT real data.

    Schumann peaks on a 1/f background. Driver-coincident windows (label==1)
    perturb the cavity the way documented flare/storm responses do: raised
    fundamental amplitude, slight upward centre-frequency shift, lower Q
    (broadening), and added transient power -- all within physical ranges.
    """
    freqs = np.linspace(0.1, FREQ_MAX_HZ, SPECTRUM_SIZE)
    spec = 1.0 / freqs  # 1/f background
    amp_boost = 1.0 + (0.6 * rng.rand() if label else 0.0)
    f_shift = 0.15 * rng.rand() if label else 0.0  # Hz upward shift on events
    width = 0.6 * (1.0 + (0.5 * rng.rand() if label else 0.0))  # Q broadening
    for i, f0 in enumerate(SCHUMANN_PEAKS_HZ):
        peak_amp = (amp_boost if i == 0 else 1.0) * (1.0 / (i + 1))
        spec += peak_amp * np.exp(-((freqs - (f0 + f_shift)) ** 2) / (2 * width**2))
    if label:
        spec += 0.2 * rng.rand() * np.exp(-((freqs - 4.0) ** 2) / (2 * 1.0**2))  # transient
    spec += 0.05 * rng.rand(SPECTRUM_SIZE)  # instrument noise
    out: np.ndarray[Any, Any] = (spec / spec.max()).astype(np.float32)
    return out


def _sample_times(catalog: Any, n: int, rng: np.random.RandomState) -> list[datetime]:
    """Uniformly sample times across the catalog span (mix of pos/neg)."""
    if not catalog.windows:
        base = datetime.utcnow()
        return [base + timedelta(minutes=int(m)) for m in rng.randint(0, 7 * 24 * 60, n)]
    t0 = min(w.start for w in catalog.windows)
    t1 = max(w.end for w in catalog.windows) + timedelta(hours=12)
    span = (t1 - t0).total_seconds()
    return [t0 + timedelta(seconds=float(rng.rand() * span)) for _ in range(n)]


class _SchumannBinary(nn.Module):
    """The quarantined analyzer's CNN encoder + a trainable binary head."""

    def __init__(self) -> None:
        super().__init__()
        self.analyzer = SchumannHarmonicAnalyzer(spectrum_size=SPECTRUM_SIZE)

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        _logits, confidence = self.analyzer(spectrum)
        return confidence.squeeze(-1)  # type: ignore[no-any-return]


def _split_indices(
    labels: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], str]:
    """Temporal earliest-70% split; seeded-stratified fallback if a fold is
    single-class (the available NOAA window's events are temporally clustered,
    a degeneracy registered in the pre-registration)."""
    n = len(labels)
    cut = int(n * 0.7)
    tr, te = np.arange(cut), np.arange(cut, n)
    if len(np.unique(labels[te])) >= 2 and len(np.unique(labels[tr])) >= 2:
        return tr, te, "temporal"
    rng = np.random.RandomState(12345)
    tr_l, te_l = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        c = max(1, round(len(idx) * 0.7))
        tr_l.extend(idx[:c].tolist())
        te_l.extend(idx[c:].tolist())
    return np.array(sorted(tr_l)), np.array(sorted(te_l)), "stratified_fallback_temporal_degenerate"


def run_seed(
    specs: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    train_idx: np.ndarray[Any, Any],
    test_idx: np.ndarray[Any, Any],
    epochs: int,
    seed: int,
    *,
    batch_size: int = 64,
    objective: str = "logits",
    regime: str = "minibatch",
) -> float:
    """Train the quarantined encoder for one seed; return test ROC-AUC.

    The defaults are the **stable recipe** established by the WS-C root-cause
    diagnosis (``benchmarks/schumann_diagnostic.py``): mini-batch SGD with a
    logit-space objective. The historical per-seed collapse (AUC ``[.97,1,.23]``)
    was **not** an ill-posed objective or a bad init -- it was a *full-batch*
    optimisation artifact (only ~``epochs`` gradient updates total, too few for
    some seeds to escape a sign-inverted basin). ``regime``/``objective`` are
    exposed so the diagnostic can reproduce the collapse and the fix; production
    use should keep the stable defaults.

    * ``objective="logits"`` -> ``BCEWithLogitsLoss`` on
      ``analyzer.confidence_logits`` (numerically correct);
      ``"sigmoid"`` -> the historical ``BCELoss`` on a clamped sigmoid.
    * ``regime="minibatch"`` -> shuffled mini-batches (the fix);
      ``"full_batch"`` -> one update/epoch (the historical, unstable recipe).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_tr = torch.from_numpy(specs[train_idx]).unsqueeze(1)
    y_tr = torch.from_numpy(labels[train_idx].astype(np.float32))
    x_te = torch.from_numpy(specs[test_idx]).unsqueeze(1)
    model = _SchumannBinary()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Class weighting for ~10% positives (pos_weight = n_neg/n_pos) -- standard
    # imbalance handling, not tuning to a result.
    n_pos = float(y_tr.sum().item())
    pos_w = torch.tensor((len(y_tr) - n_pos) / max(n_pos, 1.0))
    logit_loss = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    def _step(xb: torch.Tensor, yb: torch.Tensor) -> None:
        opt.zero_grad()
        if objective == "logits":
            logits = model.analyzer.confidence_logits(xb).squeeze(-1)
            loss = logit_loss(logits, yb)
        else:  # historical "sigmoid" objective
            w = torch.where(yb > 0, pos_w, torch.tensor(1.0))
            loss = nn.BCELoss(weight=w)(model(xb).clamp(1e-6, 1 - 1e-6), yb)
        loss.backward()
        opt.step()

    model.train()
    if regime == "minibatch":
        order = np.arange(len(y_tr))
        shuffler = np.random.RandomState(seed)
        for _ in range(epochs):
            shuffler.shuffle(order)
            for s in range(0, len(order), batch_size):
                b = order[s : s + batch_size]
                _step(x_tr[b], y_tr[b])
    else:  # full_batch (historical, unstable)
        for _ in range(epochs):
            _step(x_tr, y_tr)
    model.eval()
    with torch.no_grad():
        scores = model(x_te).numpy()
    return float(roc_auc_score(labels[test_idx], scores))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default="artifacts/schumann_eval.json")
    args = ap.parse_args()

    catalog = fetch_catalogs()  # real, public-domain NOAA labels + provenance
    rng = np.random.RandomState(0)
    times = sorted(_sample_times(catalog, args.n, rng))
    labels = np.array([catalog.label(t) for t in times], dtype=int)
    # Deterministic synthetic spectrum per sample (seeded by index).
    specs = np.stack(
        [_synth_spectrum(int(lb), np.random.RandomState(i)) for i, lb in enumerate(labels)]
    )

    pos_frac = float(labels.mean())
    train_idx, test_idx, split_used = _split_indices(labels)
    aucs = [run_seed(specs, labels, train_idx, test_idx, args.epochs, s) for s in args.seeds]
    mean_auc = float(np.mean(aucs))

    artifact = {
        "metadata": {
            "purpose": "WS-C Schumann sub-net pre-registered evaluation",
            "label_provenance": catalog.provenance,
            "label_noise": label_noise_disclosure(),
            "signal": "SYNTHETIC physically-grounded ELF (NOT real) -- cannot lift quarantine",
            "schumann_peaks_hz": list(SCHUMANN_PEAKS_HZ),
            "split": split_used,
            "training_recipe": (
                "minibatch SGD + BCEWithLogitsLoss (WS-C stable recipe; the prior "
                "seed-collapse was a full-batch optimisation artifact -- see "
                "benchmarks/schumann_diagnostic.py)"
            ),
            "metric": "ROC-AUC (mercury_ml, no sklearn)",
            "seeds": args.seeds,
            "n_samples": args.n,
            "positive_fraction": pos_frac,
            "lift_bar_real_elf": _AUC_LIFT_BAR,
        },
        "per_seed_auc": aucs,
        "mean_auc": mean_auc,
        "verdict": (
            "QUARANTINE -- signal is synthetic; the pipeline (real NOAA labels -> encoder "
            "-> metric) is validated but the sub-net stays off-by-default until an openly-"
            f"licensed real ELF corpus clears mean AUC >= {_AUC_LIFT_BAR}. "
            f"(synthetic mean AUC={mean_auc:.4f}, pipeline-plumbing only)"
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"positive_fraction={pos_frac:.3f}  per_seed_auc={[round(a,4) for a in aucs]}")
    print(f"mean_auc(synthetic)={mean_auc:.4f}")
    print("VERDICT: QUARANTINE (synthetic signal; real ELF corpus required to lift)")
    print(f"artifact: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
