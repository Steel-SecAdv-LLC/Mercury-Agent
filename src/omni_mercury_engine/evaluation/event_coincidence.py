# Copyright (C) 2025 Steel Security Advisors LLC
"""Pre-registered event-coincidence null-test (WS-D harvest).

WS-D (parapsychology/GCP) built real, reusable scientific-integrity machinery:
honest ingestion with explicit reachability, a **pre-registration** that fixes
every analytic degree of freedom before seeing data, a **null test**, and a
**multiple-comparison correction**. The psi question was a faithful null and is
closed -- but the machinery is exactly what a free, life-safety anomaly system
needs whenever it asks *"do my detector's flags coincide with real events more
than chance?"* That question is endemic to Mercury's mission (space-weather,
seismic, environmental hazard) and is the classic home of post-hoc-flexibility
self-deception -- the precise failure mode the GCP literature is infamous for.

This module generalises the GCP pattern into a domain-agnostic tool:

* :class:`PreregisteredCoincidenceTest` -- fix the statistic, permutation count,
  alpha, and correction **before** running (commit it with the analysis);
* :func:`permutation_coincidence_test` -- a **circular time-shift permutation**
  null (the gold standard for autocorrelated streams: it preserves the score's
  own temporal structure while destroying only its alignment to the fixed event
  windows, so a significant result cannot be an autocorrelation artifact);
* :func:`benjamini_hochberg` / :func:`bonferroni` -- multiple-comparison control.

It makes **no** domain claim. A clean null is a valid, first-class result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PreregisteredCoincidenceTest:
    """A pre-registered coincidence protocol. Fix every field before analysis."""

    name: str
    statistic: str = "mean_diff"  # "mean_diff" | "mean_in"
    n_permutations: int = 2000
    alpha: float = 0.05
    correction: str = "bonferroni"  # "bonferroni" | "bh_fdr" | "none"
    seed: int = 0
    min_circular_shift: int = 1  # avoid the (near-)identity shift

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable protocol."""
        return {
            "name": self.name,
            "statistic": self.statistic,
            "n_permutations": self.n_permutations,
            "alpha": self.alpha,
            "correction": self.correction,
            "seed": self.seed,
        }


@dataclass
class CoincidenceResult:
    """Result from one event-coincidence permutation test."""

    observed: float
    p_value: float
    n_in_window: int
    n_total: int
    n_permutations: int
    statistic: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result."""
        return {
            "observed": self.observed,
            "p_value": self.p_value,
            "n_in_window": self.n_in_window,
            "n_total": self.n_total,
            "n_permutations": self.n_permutations,
            "statistic": self.statistic,
        }


def _coincidence_statistic(
    scores: np.ndarray[Any, Any], in_window: np.ndarray[Any, Any], kind: str
) -> float:
    """Effect size: how much higher the score is inside event windows."""
    inside = scores[in_window]
    if inside.size == 0:
        return 0.0
    if kind == "mean_in":
        return float(inside.mean())
    outside = scores[~in_window]
    out_mean = float(outside.mean()) if outside.size else 0.0
    return float(inside.mean() - out_mean)  # "mean_diff" (default)


def permutation_coincidence_test(
    scores: np.ndarray[Any, Any],
    in_window: np.ndarray[Any, Any],
    *,
    statistic: str = "mean_diff",
    n_permutations: int = 2000,
    seed: int = 0,
    min_circular_shift: int = 1,
) -> CoincidenceResult:
    """Test whether ``scores`` are elevated inside the fixed event mask.

    The null circularly rolls ``scores`` by a random offset, preserving its
    autocorrelation while breaking alignment to the events. The one-sided
    p-value (with +1 smoothing) is the fraction of null statistics ≥ the
    observed -- i.e. the probability of seeing this much coincidence if the score
    stream were unrelated to the events. **No effect ⇒ p ≈ uniform ⇒ not
    significant; that is the expected, valid outcome under a null.**
    """
    scores = np.asarray(scores, dtype=float)
    in_window = np.asarray(in_window, dtype=bool)
    if scores.shape != in_window.shape:
        raise ValueError("scores and in_window must have the same shape")
    n = scores.size
    n_in = int(in_window.sum())
    if n == 0 or n_in == 0 or n_in == n:
        # Degenerate: no events, or every/no sample in-window -> undefined test.
        return CoincidenceResult(0.0, 1.0, n_in, n, 0, statistic)

    observed = _coincidence_statistic(scores, in_window, statistic)
    rng = np.random.RandomState(seed)
    # Random circular shifts in [min_circular_shift, n - min_circular_shift].
    hi = max(min_circular_shift + 1, n - min_circular_shift)
    shifts = rng.randint(min_circular_shift, hi, size=n_permutations)
    ge = 1  # +1 smoothing (observed counts as one draw)
    for s in shifts:
        null_stat = _coincidence_statistic(np.roll(scores, int(s)), in_window, statistic)
        if null_stat >= observed:
            ge += 1
    p = ge / (n_permutations + 1)
    return CoincidenceResult(observed, float(p), n_in, n, n_permutations, statistic)


def windows_to_mask(
    timestamps: np.ndarray[Any, Any],
    windows: list[tuple[float, float]],
) -> np.ndarray[Any, Any]:
    """Boolean mask: True where ``timestamps`` fall inside any ``(start, end)``."""
    ts = np.asarray(timestamps, dtype=float)
    mask = np.zeros(ts.shape, dtype=bool)
    for start, end in windows:
        mask |= (ts >= start) & (ts <= end)
    return mask


def bonferroni(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """Bonferroni: reject p_i if p_i <= alpha / m."""
    m = len(pvalues)
    if m == 0:
        return []
    thr = alpha / m
    return [p <= thr for p in pvalues]


def benjamini_hochberg(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR control. Returns a reject mask in input order."""
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    reject = [False] * m
    max_k = -1
    for rank, idx in enumerate(order, start=1):
        if pvalues[idx] <= alpha * rank / m:
            max_k = rank
    if max_k >= 0:
        for rank, idx in enumerate(order, start=1):
            if rank <= max_k:
                reject[idx] = True
    return reject


def correct(pvalues: list[float], method: str, alpha: float = 0.05) -> list[bool]:
    """Apply the pre-registered multiple-comparison correction."""
    if method == "bonferroni":
        return bonferroni(pvalues, alpha)
    if method == "bh_fdr":
        return benjamini_hochberg(pvalues, alpha)
    if method == "none":
        return [p <= alpha for p in pvalues]
    raise ValueError(f"unknown correction method: {method!r}")


@dataclass
class CoincidenceReport:
    """A full pre-registered run: per-event results + corrected verdict."""

    protocol: dict[str, Any]
    results: list[CoincidenceResult] = field(default_factory=list)
    reject: list[bool] = field(default_factory=list)

    @property
    def any_significant(self) -> bool:
        """Return whether any corrected comparison is significant."""
        return any(self.reject)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return {
            "protocol": self.protocol,
            "results": [r.as_dict() for r in self.results],
            "reject_after_correction": self.reject,
            "any_significant": self.any_significant,
        }


def run_preregistered(
    test: PreregisteredCoincidenceTest,
    score_streams: list[np.ndarray[Any, Any]],
    masks: list[np.ndarray[Any, Any]],
) -> CoincidenceReport:
    """Run the pre-registered test over multiple score/mask pairs.

    Applies the registered multiple-comparison correction across all pairs.
    """
    if len(score_streams) != len(masks):
        raise ValueError("score_streams and masks must align")
    results = [
        permutation_coincidence_test(
            s,
            m,
            statistic=test.statistic,
            n_permutations=test.n_permutations,
            seed=test.seed,
            min_circular_shift=test.min_circular_shift,
        )
        for s, m in zip(score_streams, masks)
    ]
    reject = correct([r.p_value for r in results], test.correction, test.alpha)
    return CoincidenceReport(protocol=test.as_dict(), results=results, reject=reject)


__all__ = [
    "CoincidenceReport",
    "CoincidenceResult",
    "PreregisteredCoincidenceTest",
    "benjamini_hochberg",
    "bonferroni",
    "correct",
    "permutation_coincidence_test",
    "run_preregistered",
    "windows_to_mask",
]
