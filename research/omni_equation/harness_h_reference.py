# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Is the golden-ratio band profile the right reference for H(omega)?

H(omega) scores a window by the L1 distance between its 8-band spectral energy
ratios and a fixed expected profile ``1/PHI^i``. That profile had never been
compared against an alternative. ``/three_r`` is stateless -- one array in, one
score out, with no fit step -- so a *learned* profile would require an API
change; every candidate here is drop-in and needs no fitted state.

The metric is window AUROC, as in ``docs/MATH_SPEC.md`` 2.1.6 and 5.1a. It is a
ranking metric and does not charge for a false alarm; the route has no
calibrated threshold (the served verdict is a bare ``> 0.5``), so there is no
alarm rate to score. That is why this measures a reference choice and not an
operating point -- compare
``research/math_arrest/harness_decorrelation.py``, where the two metrics
disagree and the ranking one is the wrong guide.

Reproduces both tables in MATH_SPEC 2.1.6. Run from the repository root::

    python research/omni_equation/harness_h_reference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.recorded_data import auroc, nab_corpus

PHI = 1.618033988749895
BANDS = 8
WINDOW, STRIDE = 128, 16
PROFILES = ("phi", "flat", "pink", "steep")


def band_ratios(data: np.ndarray) -> np.ndarray:
    """The served banding: energy share per band, summing to 1."""
    magnitudes = np.abs(np.fft.fft(data))
    n_bands = min(BANDS, len(magnitudes) // 2)
    size = len(magnitudes) // (2 * n_bands) if n_bands > 0 else 1
    energies = np.array(
        [float(np.sum(magnitudes[i * size : (i + 1) * size] ** 2)) for i in range(n_bands)]
    )
    return energies / (energies.sum() + 1e-8)


def profile(kind: str, n: int) -> np.ndarray:
    """A fixed expected profile, normalised to a probability vector."""
    if kind == "phi":
        weights = np.array([1.0 / (PHI**i) for i in range(n)])
    elif kind == "flat":
        weights = np.ones(n)
    elif kind == "pink":
        weights = 1.0 / (np.arange(n) + 1.0)
    elif kind == "steep":
        weights = 1.0 / (np.arange(n) + 1.0) ** 2
    else:
        raise ValueError(f"unknown profile: {kind}")
    return weights / weights.sum()


def resonance(data: np.ndarray, kind: str) -> float:
    """The served H(omega) with the reference profile swapped."""
    ratios = band_ratios(data)
    deviation = float(np.sum(np.abs(ratios - profile(kind, len(ratios)))))
    return float(np.clip(1.0 - deviation / 2.0, 0.0, 1.0))


def flatness(data: np.ndarray) -> float:
    """Spectral flatness: geometric over arithmetic mean of band energies."""
    ratios = band_ratios(data) + 1e-12
    return float(np.exp(np.mean(np.log(ratios))) / np.mean(ratios))


def _windows(series: np.ndarray, positive: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    windows, labels = [], []
    for start in range(0, len(series) - WINDOW + 1, STRIDE):
        windows.append(series[start : start + WINDOW])
        labels.append(bool(positive[start : start + WINDOW].any()))
    return windows, np.asarray(labels, dtype=bool)


def main() -> None:
    """Print the profile comparison and the structural reason behind it."""
    scores: dict[str, list[float]] = {k: [] for k in (*PROFILES, "flatness")}
    structure: list[tuple[str, float, int, float]] = []

    for series in nab_corpus():
        windows, labels = _windows(series.values, series.positive)
        if labels.all() or not labels.any():
            continue
        for kind in PROFILES:
            values = np.asarray([resonance(w, kind) for w in windows])
            scores[kind].append(
                auroc(values[labels], values[~labels]) if np.ptp(values) > 0 else 0.5
            )
        values = np.asarray([flatness(w) for w in windows])
        scores["flatness"].append(
            auroc(values[labels], values[~labels]) if np.ptp(values) > 0 else 0.5
        )

        # Why three files score identically: the L1 distance telescopes to
        # 2*(r0 - e0) whenever the sign pattern of r - e is fixed, leaving the
        # profile as an additive constant that cannot reorder windows.
        ratios = np.array([band_ratios(w) for w in windows])
        deviations = np.array([np.sum(np.abs(r - profile("phi", len(r)))) for r in ratios])
        patterns = {tuple(np.sign(r - profile("phi", len(r))).astype(int)) for r in ratios}
        correlation = (
            abs(float(np.corrcoef(deviations, ratios[:, 0])[0, 1]))
            if np.std(ratios[:, 0]) > 0
            else float("nan")
        )
        structure.append((series.name, correlation, len(patterns), float(ratios[:, 0].mean())))

    print("Mean window AUROC by expected profile\n")
    for kind, values in scores.items():
        per_file = " ".join(f"{v:8.4f}" for v in values)
        print(f"  {kind:10s} {per_file}   mean {np.mean(values):.4f}")

    print("\nWhy most files are insensitive to the profile\n")
    for name, correlation, patterns, mean_r0 in structure:
        print(
            f"  {name[:40]:40s} |corr(delta, r0)| {correlation:.4f}"
            f"   sign patterns {patterns:3d}   mean r0 {mean_r0:.3f}"
        )
    print(
        "\nWhere |corr| is 1.0000 the deviation is linear in r0 and the profile is a\n"
        "constant offset, so every profile ranks the windows identically. H(omega) is\n"
        "then a measure of lowest-band spectral concentration, not of golden-ratio\n"
        "resonance; PHI is kept because it measured best, not because it is required."
    )


if __name__ == "__main__":
    main()
