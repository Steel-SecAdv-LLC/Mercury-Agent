# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multiple-comparison control shared by live detection and evaluation.

Benjamini-Hochberg false-discovery-rate control (Benjamini & Hochberg 1995,
"Controlling the False Discovery Rate", J. R. Statist. Soc. B 57(1):289-300)
and the Bonferroni family-wise bound.  These lived in
``evaluation.event_coincidence`` and were therefore reachable only from the
evaluation harness; the live drift detectors
(:mod:`omni_mercury_engine.detectors.drift`) now consume them from here, and
the evaluation module re-exports them unchanged.
"""

from __future__ import annotations


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
    """Apply the named multiple-comparison correction.

    Args:
        pvalues: Raw p-values in test order.
        method: ``"bonferroni"``, ``"bh_fdr"`` or ``"none"``.
        alpha: Significance (Bonferroni/none) or FDR (BH) level.

    Returns:
        Reject mask in input order.

    Raises:
        ValueError: On an unknown method name.
    """
    if method == "bonferroni":
        return bonferroni(pvalues, alpha)
    if method == "bh_fdr":
        return benjamini_hochberg(pvalues, alpha)
    if method == "none":
        return [p <= alpha for p in pvalues]
    raise ValueError(f"unknown correction method: {method!r}")
