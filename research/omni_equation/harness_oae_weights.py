# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Does the OAE's golden-ratio weighting beat uniform or learned weights?

R(x), H(omega), O(theta) are reproduced verbatim from the served path
(api/routes/detection.py, /three_r) with its defaults: recursion_depth=5,
harmonic_bands=8. Each produces ONE scalar per window, so the unit of
measurement is a window, and labels come from NAB's own anomaly windows.

The eta^p gate is omitted deliberately: it is a single scalar per call, so it
is a monotone transform of the batch and cannot change within-batch ranking,
hence cannot change AUROC. The question here is purely the weights.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.recorded_data import auroc, nab_corpus

PHI = 1.618033988749895
RECURSION_DEPTH = 5
HARMONIC_BANDS = 8
WINDOW = 128
STRIDE = 16


# ---- served producers, copied verbatim from api/routes/detection.py ----
def compute_recursion_score(x: np.ndarray, depth: int) -> float:
    if depth == 0 or len(x) < 2:
        return float(np.std(x) / (np.mean(np.abs(x)) + 1e-8))
    mid = len(x) // 2
    left_score = compute_recursion_score(x[:mid], depth - 1)
    right_score = compute_recursion_score(x[mid:], depth - 1)
    return float(0.5 * (left_score + right_score) + 0.5 * np.std(x) / (np.mean(np.abs(x)) + 1e-8))


def streams(data: np.ndarray) -> tuple[float, float, float]:
    r = float(np.clip(compute_recursion_score(data, RECURSION_DEPTH), 0.0, 1.0))

    magnitudes = np.abs(np.fft.fft(data))
    n_bands = min(HARMONIC_BANDS, len(magnitudes) // 2)
    band_size = len(magnitudes) // (2 * n_bands) if n_bands > 0 else 1
    band_energies = [
        float(np.sum(magnitudes[i * band_size : (i + 1) * band_size] ** 2)) for i in range(n_bands)
    ]
    total = sum(band_energies) + 1e-8
    band_ratios = [e / total for e in band_energies]
    expected = [1.0 / (PHI**i) for i in range(n_bands)]
    esum = sum(expected)
    expected = [e / esum for e in expected]
    deviation = sum(abs(a - e) for a, e in zip(band_ratios, expected))
    # Normalised by the maximum L1 distance between probability vectors, as the
    # served route now does. Set OAE_LEGACY_H=1 to reproduce the pre-fix
    # saturated form (mean AUROC 0.511, constant on realTraffic/speed_7578).
    import os as _os

    if _os.environ.get("OAE_LEGACY_H") == "1":
        h = float(np.clip(1.0 - deviation, 0.0, 1.0))
    else:
        h = float(np.clip(1.0 - deviation / 2.0, 0.0, 1.0))

    signal_variance = np.var(data)
    noise_estimate = np.var(np.diff(data)) / 2
    snr = signal_variance / (noise_estimate + 1e-8)
    o = float(np.clip(1.0 / (1.0 + np.exp(-np.log10(snr + 1))), 0.0, 1.0))
    return r, h, o


# ---- weightings ----
PHI_SUM = PHI + 2.0
W_PHI = np.array([PHI / PHI_SUM, 1.0 / PHI_SUM, 1.0 / PHI_SUM])
W_UNIFORM = np.array([1 / 3, 1 / 3, 1 / 3])


def learn_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative weights summing to 1, fitted by grid search on train AUROC.

    A grid rather than a regression: AUROC is rank-based, so least squares on
    the labels optimises the wrong objective. Simplex grid at 0.05 resolution.
    """
    best, best_w = -1.0, W_UNIFORM
    grid = np.arange(0.0, 1.0001, 0.05)
    for a in grid:
        for b in grid:
            if a + b > 1.0 + 1e-9:
                continue
            c = 1.0 - a - b
            w = np.array([a, b, c])
            s = X @ w
            if np.ptp(s) == 0:
                continue
            val = auroc(s[y], s[~y])
            if val > best:
                best, best_w = val, w
    return best_w


def main() -> None:
    per_file: dict[str, dict[str, Any]] = {}
    for series in nab_corpus():
        values, positive = series.values, series.positive
        rows, labels = [], []
        for start in range(0, len(values) - WINDOW + 1, STRIDE):
            win = values[start : start + WINDOW]
            rows.append(streams(win))
            labels.append(bool(positive[start : start + WINDOW].any()))
        X = np.asarray(rows, dtype=float)
        y = np.asarray(labels, dtype=bool)
        if y.all() or not y.any():
            continue

        half = len(y) // 2
        Xtr, ytr, Xte, yte = X[:half], y[:half], X[half:], y[half:]
        if not (ytr.any() and not ytr.all() and yte.any() and not yte.all()):
            # fall back to scoring every window when a half is single-class
            Xtr, ytr, Xte, yte = X, y, X, y

        w_learned = learn_weights(Xtr, ytr)
        out: dict[str, Any] = {
            "phi": auroc((Xte @ W_PHI)[yte], (Xte @ W_PHI)[~yte]),
            "uniform": auroc((Xte @ W_UNIFORM)[yte], (Xte @ W_UNIFORM)[~yte]),
            "learned": auroc((Xte @ w_learned)[yte], (Xte @ w_learned)[~yte]),
            "R_alone": auroc(Xte[yte, 0], Xte[~yte, 0]),
            "H_alone": auroc(Xte[yte, 1], Xte[~yte, 1]),
            "O_alone": auroc(Xte[yte, 2], Xte[~yte, 2]),
            "_w_learned": w_learned,
            "_n": len(yte),
            "_pos": int(yte.sum()),
        }
        per_file[series.name] = out

    keys = ["phi", "uniform", "learned", "R_alone", "H_alone", "O_alone"]
    print(f"{'file':38s} " + " ".join(f"{k:>9s}" for k in keys) + "   n/pos")
    for name, out in per_file.items():
        line = " ".join(f"{out[k]:9.4f}" for k in keys)
        print(f"{name[:38]:38s} {line}   {out['_n']}/{out['_pos']}")
    print()
    print(
        f"{'MEAN':38s} "
        + " ".join(f"{np.mean([o[k] for o in per_file.values()]):9.4f}" for k in keys)
    )
    print()
    print("learned weights per file (w_R, w_H, w_O):")
    for name, out in per_file.items():
        w = out["_w_learned"]
        print(f"  {name[:38]:38s} ({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f})")
    print(f"\nphi weights for reference: ({W_PHI[0]:.4f}, {W_PHI[1]:.4f}, {W_PHI[2]:.4f})")


if __name__ == "__main__":
    main()
