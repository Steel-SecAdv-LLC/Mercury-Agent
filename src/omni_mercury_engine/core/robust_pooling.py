# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded-influence reliability-weighted log-odds pooling (NumPy-only).

Mercury's own implementation of robust opinion pooling for combining the three
ensemble component scores (resonance / kinematic / info-geometry).  The point of
the construction (Item 3) is that a single diluting component — e.g. the
kinematic stream at AUROC ~0.66 — cannot drag the fused score down:

* ``compute_reliability_weights`` gives each component a weight proportional to
  its measured reliability-in-regime (per-component AUROC above chance), so a
  weak component self-down-weights (#38).
* ``clipped_logodds`` pools in log-odds space but clips each component's signed
  deviation from the robust centre to ``+/- c`` before the weighted sum, so the
  influence of any one component on the pooled log-odds is bounded (the
  ``2*c*sum(w)`` cap) — an extreme outlier opinion cannot dominate.
* ``trimmed_logodds`` instead discards the ``t`` most deviant components per row
  (breakdown ``t/k``) and pools the rest.

Attribution: the bounded-influence / trimmed pooling framing is a *blueprint*
adapted from FINDOYOU robust-pooling work; this code and the numbers Mercury
reports with it are Mercury's own.  Default-off everywhere it is wired.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_EPS = 1e-6


def _logit(p: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    q = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(q / (1.0 - q))


def _sigmoid(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _weighted_median(
    values: np.ndarray[Any, Any], weights: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Row-wise weighted median of ``values`` (n, k) with column weights (k,)."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    order = np.argsort(v, axis=1)
    v_sorted = np.take_along_axis(v, order, axis=1)
    w_sorted = np.take_along_axis(np.broadcast_to(w, v.shape), order, axis=1)
    cum = np.cumsum(w_sorted, axis=1)
    half = 0.5 * cum[:, -1:]
    idx = np.argmax(cum >= half, axis=1)
    return v_sorted[np.arange(v.shape[0]), idx]


def normalize_weights(weights: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Clip negatives and renormalize to sum 1 (uniform if all non-positive)."""
    w = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    s = w.sum()
    if s <= 0:
        return np.full(w.shape, 1.0 / max(len(w), 1))
    return w / s


def compute_reliability_weights(
    component_scores: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    *,
    floor: float = 0.0,
) -> np.ndarray[Any, Any]:
    """Reliability-in-regime weights ∝ ``max(AUROC_j - 0.5, 0)`` per component.

    Measured on a labelled calibration split.  Components that do not separate
    better than chance get zero (or ``floor``) weight, so the kinematic diluter
    self-down-weights relative to info-geometry.  An inverted component (AUROC <
    0.5) contributes nothing rather than fighting the consensus.
    """
    scores = np.asarray(component_scores, dtype=np.float64)
    y = np.asarray(labels, dtype=int).reshape(-1)
    k = scores.shape[1]
    rel = np.zeros(k, dtype=np.float64)
    pos = y == 1
    neg = y == 0
    if not (np.any(pos) and np.any(neg)):
        return np.full(k, 1.0 / k)
    for j in range(k):
        s = scores[:, j]
        # AUROC via the Mann-Whitney U probability P(s_pos > s_neg).
        sp, sn = s[pos], s[neg]
        gt = (sp[:, None] > sn[None, :]).mean()
        eq = (sp[:, None] == sn[None, :]).mean()
        auc = float(gt + 0.5 * eq)
        rel[j] = max(auc - 0.5, floor)
    if rel.sum() <= 0:
        return np.full(k, 1.0 / k)
    return normalize_weights(rel)


def clipped_logodds(
    probs: np.ndarray[Any, Any],
    weights: np.ndarray[Any, Any],
    *,
    c: float = 2.0,
) -> np.ndarray[Any, Any]:
    """Bounded-influence weighted log-odds pool.

    Pools ``logit(p_j)`` around the robust (weighted-median) centre, clipping
    each component's signed deviation to ``[-c, c]`` so no single component can
    move the pooled log-odds by more than its weighted share of ``c`` (total
    influence bounded by ``2*c*sum(w)``).
    """
    p = np.atleast_2d(np.asarray(probs, dtype=np.float64))
    w = normalize_weights(weights)
    logits = _logit(p)
    centre = _weighted_median(logits, w)[:, None]
    clipped = np.clip(logits - centre, -c, c)
    pooled = centre[:, 0] + clipped @ w
    return _sigmoid(pooled)


def trimmed_logodds(
    probs: np.ndarray[Any, Any],
    weights: np.ndarray[Any, Any],
    *,
    t: int = 1,
) -> np.ndarray[Any, Any]:
    """Trimmed weighted log-odds pool (breakdown ``t/k``).

    Per row, drops the ``t`` components whose log-odds deviate most from the
    weighted-median centre, then pools the remaining components with
    renormalized weights.
    """
    p = np.atleast_2d(np.asarray(probs, dtype=np.float64))
    w = normalize_weights(weights)
    n, k = p.shape
    t = int(np.clip(t, 0, k - 1))
    logits = _logit(p)
    centre = _weighted_median(logits, w)[:, None]
    dev = np.abs(logits - centre)
    # Keep the (k - t) least-deviant components per row.
    keep_idx = np.argsort(dev, axis=1)[:, : k - t]
    kept_logits = np.take_along_axis(logits, keep_idx, axis=1)
    kept_w = np.take_along_axis(np.broadcast_to(w, p.shape), keep_idx, axis=1)
    kept_w = kept_w / np.clip(kept_w.sum(axis=1, keepdims=True), _EPS, None)
    pooled = np.sum(kept_logits * kept_w, axis=1)
    return _sigmoid(pooled)


def reliability_pool(
    probs: np.ndarray[Any, Any],
    weights: np.ndarray[Any, Any],
    *,
    mode: str = "clipped",
    c: float = 2.0,
    t: int = 1,
) -> np.ndarray[Any, Any]:
    """Dispatch to the clipped (default) or trimmed bounded-influence pool."""
    if mode == "trimmed":
        return trimmed_logodds(probs, weights, t=t)
    return clipped_logodds(probs, weights, c=c)
