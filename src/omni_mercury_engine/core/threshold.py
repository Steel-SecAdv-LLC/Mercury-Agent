# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Label-free threshold selection via histogram variance maximization.

Otsu's method (Otsu, 1979 — IEEE Trans. Sys. Man. Cyb.) finds the score
threshold that maximizes between-class variance across two populations
(normal / anomalous) without requiring any ground-truth labels.

This replaces the contamination-rate percentile approach used by the
operational F1 benchmark, which requires knowing the anomaly ratio in
advance — information that is unavailable in deployment.

References:
    Otsu, N. (1979). A threshold selection method from gray-level
    histograms. IEEE Transactions on Systems, Man, and Cybernetics,
    9(1), 62–66. https://doi.org/10.1109/TSMC.1979.4310076
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Number of histogram bins for Otsu computation.
# 256 bins gives <0.4% resolution error on [0, 1] scores.
_OTSU_BINS: int = 256

# Minimum samples required for Otsu to be meaningful.
_OTSU_MIN_SAMPLES: int = 30

# Fallback threshold when Otsu degenerates (single-class or insufficient data).
_OTSU_FALLBACK: float = 0.5


def otsu_threshold(scores: npt.NDArray[np.float64]) -> float:
    """Find the score threshold that maximizes between-class variance.

    Operates on the empirical histogram of ``scores``, scanning all bin
    edges to find the split that maximizes:

        σ²_between = w₀ * w₁ * (μ₀ - μ₁)²

    where w₀, w₁ are the population weights and μ₀, μ₁ are the class
    means at each candidate threshold.

    Args:
        scores: 1-D anomaly scores in [0, 1]. Must have dtype castable
            to float64.

    Returns:
        Threshold in [0, 1]. Returns ``_OTSU_FALLBACK`` (0.5) when:
        - fewer than ``_OTSU_MIN_SAMPLES`` samples are provided,
        - all scores are identical (zero variance),
        - the histogram has only a single occupied bin.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n = len(scores)

    if n < _OTSU_MIN_SAMPLES:
        return _OTSU_FALLBACK

    scores_clipped = np.clip(scores, 0.0, 1.0)

    if float(np.std(scores_clipped)) < 1e-10:
        # Constant scores — degenerate: no separable threshold
        return _OTSU_FALLBACK

    hist, bin_edges = np.histogram(scores_clipped, bins=_OTSU_BINS, range=(0.0, 1.0))
    hist_f = hist.astype(np.float64)
    total = hist_f.sum()

    if total < 1.0:
        return _OTSU_FALLBACK

    hist_norm = hist_f / total  # normalized to probability mass

    # Cumulative sums for vectorized between-class variance computation
    w0 = np.cumsum(hist_norm)  # weight of class 0 (below threshold)
    w1 = 1.0 - w0  # weight of class 1 (above threshold)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    cumsum_mu = np.cumsum(hist_norm * bin_centers)
    total_mu = cumsum_mu[-1]

    # Safe division: avoid divide-by-zero warnings in vectorized code
    w0_safe = np.where(w0 > 1e-10, w0, 1.0)
    w1_safe = np.where(w1 > 1e-10, w1, 1.0)
    mu0 = np.where(w0 > 1e-10, cumsum_mu / w0_safe, 0.0)
    mu1 = np.where(w1 > 1e-10, (total_mu - cumsum_mu) / w1_safe, 0.0)

    # Between-class variance (Otsu criterion)
    sigma_between = w0 * w1 * (mu0 - mu1) ** 2

    # Find first bin index with maximum between-class variance
    best_idx = int(np.argmax(sigma_between))

    # Guard: if best variance is near zero, Otsu found no meaningful split
    if float(sigma_between[best_idx]) < 1e-10:
        return _OTSU_FALLBACK

    return float(bin_edges[best_idx + 1])  # threshold = upper edge of best bin


def adaptive_threshold(
    scores: npt.NDArray[np.float64],
    contamination_hint: float | None = None,
    prefer_recall: bool = False,
) -> tuple[float, str]:
    """Select the best available threshold from a cascade of methods.

    Cascade order (first successful method wins):
    1. Otsu (no labels required, preferred when n >= 30)
    2. MAD: median + 3.0 * MAD  (robust to outliers in the score distribution)
    3. Contamination percentile: requires a contamination_hint
    4. Fallback: 0.5

    Args:
        scores: Anomaly scores in [0, 1].
        contamination_hint: Optional estimated anomaly fraction in (0, 1).
            Used as fallback only if Otsu and MAD both fail.
        prefer_recall: If True, lower the threshold by 0.05 (bias toward
            sensitivity — appropriate for life-safety applications).

    Returns:
        Tuple of (threshold, method_name) where method_name identifies
        which strategy was applied.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()

    # Method 1: Otsu
    thr = otsu_threshold(scores)
    method = "otsu"

    # Otsu returned fallback (0.5) — try MAD
    otsu_degenerate = abs(thr - _OTSU_FALLBACK) < 1e-10
    if otsu_degenerate:
        median = float(np.median(scores))
        mad = float(np.median(np.abs(scores - median)))
        if mad > 1e-10:
            thr = min(float(np.clip(median + 3.0 * mad, 0.0, 1.0)), 0.95)
            method = "mad"

    # Both Otsu and MAD degenerate — use contamination percentile if available
    if otsu_degenerate and method != "mad":
        if contamination_hint is not None and 0.0 < contamination_hint < 1.0:
            thr = float(np.percentile(scores, 100.0 * (1.0 - contamination_hint)))
            method = "contamination_percentile"
        else:
            method = "fallback_0.5"

    if prefer_recall:
        thr = max(float(thr) - 0.05, 0.01)

    return float(np.clip(thr, 0.0, 1.0)), method
