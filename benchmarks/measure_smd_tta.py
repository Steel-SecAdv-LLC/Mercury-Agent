# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reproduce the PR #302 multi-scale-TTA measurement on real SMD telemetry.

Validates the opt-in ``multiscale_tta`` time-dilation augmentation on the real
Server Machine Dataset (SMD ``machine-1-1``), windowed for tractable compute:

  * ``default-off byte-identity`` — TTA off reproduces the baseline scores
    exactly (Invariant I2).
  * ``clean gain`` — mean / max pooling vs the un-augmented baseline.
  * ``global rate-drift recovery`` — how much AUROC lost to a whole-series
    sampling-rate stretch/compress each pooling mode recovers.

Data is fetched on demand by :class:`SMDLoader` from the OmniAnomaly source
(github.com/NetManAIOps/OmniAnomaly, Su et al., KDD 2019; MIT-licensed),
cached under ``./data/smd/`` (git-ignored). No third-party data is vendored.

Requires the native AMA Cryptography backend at import time (same precondition
as the test suite); the detector path performs no cryptographic operations.

Usage::

    python benchmarks/measure_smd_tta.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MERCURY_DATA_DIR", str(ROOT / "data"))
os.environ.setdefault("MERCURY_CACHE_DIR", str(ROOT / "cache"))
sys.path.insert(0, str(ROOT / "src"))

from omni_mercury_engine.datasets.base import DatasetConfig
from omni_mercury_engine.datasets.timeseries import SMDLoader
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector as MAD

# Window indices matching the PR #302 measurement (kept fixed for determinism).
_TRAIN_N = 6000
_TEST_SLICE = slice(13000, 21000)


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U), no sklearn."""
    y = np.asarray(y).astype(int).reshape(-1)
    s = np.asarray(s, float).reshape(-1)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def global_drift(X: np.ndarray, factor: float) -> np.ndarray:
    """Global sampling-rate drift: stretch/compress the whole series by ``factor``."""
    T = X.shape[0]
    mid = max(8, round(T * factor))
    g0 = np.linspace(0, 1, T)
    g1 = np.linspace(0, 1, mid)
    Xm = np.column_stack([np.interp(g1, g0, X[:, j]) for j in range(X.shape[1])])
    return np.column_stack([np.interp(g0, g1, Xm[:, j]) for j in range(X.shape[1])])


def _scores(det: MAD, X: np.ndarray) -> np.ndarray:
    return np.asarray(det.detect(X)["scores"], float)


def _load_smd_machine_1_1() -> tuple:
    """Fetch (if needed) and load the SMD machine-1-1 train/test/label arrays."""
    cfg = DatasetConfig(name="smd", preprocessing={"machines": ["machine-1-1"]})
    loader = SMDLoader(cfg)
    loader.download()  # fetch+cache from OmniAnomaly; returns early if cached
    md = loader.data_path / "machine-1-1"
    tr = np.load(md / "train.npy")[:_TRAIN_N]
    te = np.load(md / "test.npy")[_TEST_SLICE]
    y = np.load(md / "test_label.npy").astype(int)[_TEST_SLICE]
    return tr, te, y


def main() -> int:
    tr, te, y = _load_smd_machine_1_1()
    mu = tr.mean(0)
    sd = np.where(tr.std(0) < 1e-8, 1e-8, tr.std(0))
    Xtr = (tr - mu) / sd
    Xte = (te - mu) / sd
    print(f"SMD window: train{Xtr.shape} test{Xte.shape} anom={y.mean():.4f}")

    base = MAD().fit(Xtr)
    dm = MAD({"multiscale_tta": True, "multiscale_tta_pool": "mean"}).fit(Xtr)
    dx = MAD({"multiscale_tta": True, "multiscale_tta_pool": "max"}).fit(Xtr)
    off = MAD({"multiscale_tta": False}).fit(Xtr)
    print(
        f"data_type={base._data_type.value}  default-off byte-identical:"
        f" {np.array_equal(_scores(off, Xte), _scores(base, Xte))}"
    )

    clean = auroc(y, _scores(base, Xte))
    am = auroc(y, _scores(dm, Xte))
    ax = auroc(y, _scores(dx, Xte))
    print(
        f"\n[clean]  base={clean:.4f}  TTA-mean={am:.4f} ({(am - clean) * 100:+.2f})  "
        f"TTA-max={ax:.4f} ({(ax - clean) * 100:+.2f})"
    )

    print("\n[global rate drift recovery]")
    for fac in (0.85, 1.15):
        Xd = global_drift(Xte, fac)
        c = auroc(y, _scores(base, Xd))
        m = auroc(y, _scores(dm, Xd))
        xx = auroc(y, _scores(dx, Xd))
        lost = clean - c
        if lost <= 1e-9:
            # The base detector did not degrade under this drift on this window
            # (global resampling can incidentally smooth a borderline series),
            # so a "% recovery" ratio is undefined. Report the raw TTA deltas vs
            # the drifted base instead of a meaningless fraction.
            print(
                f"  drift x{fac}: base->{c:.4f} (no degradation, {lost * 100:+.2f}pt vs clean)  "
                f"TTA-mean={m:.4f} ({(m - c) * 100:+.2f})  "
                f"TTA-max={xx:.4f} ({(xx - c) * 100:+.2f})"
            )
            continue
        rm = m - c
        rx = xx - c
        print(
            f"  drift x{fac}: base->{c:.4f} (lost {lost * 100:+.2f}pt)  "
            f"TTA-mean={m:.4f} (recover {rm * 100:+.2f}, {rm / lost * 100:.0f}%)  "
            f"TTA-max={xx:.4f} (recover {rx * 100:+.2f}, {rx / lost * 100:.0f}%)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
