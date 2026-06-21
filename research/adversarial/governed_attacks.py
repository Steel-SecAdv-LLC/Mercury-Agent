# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Adversarial-survivability probes for the governed fusion substrate.

Test/research-only: every function consumes a score function and never changes
Mercury runtime decisions.  Two things are measured:

1. The **controlled-channel floor curve.**  A fixed-budget attacker is allowed
   to perturb only a chosen subset of ``m`` channels (the on-manifold,
   half-channel evasion that drives the proven ~0.5 floor).  ``floor_curve``
   sweeps ``m`` over the feature dims and reports worst-case AUROC vs ``m``.

2. The **cubic-moment escape.**  A detector that uses only mean+covariance
   (Gaussian / Mahalanobis) has that floor.  The polynomial-lift detective
   ``D_phi`` over ``phi(z) = [z, z^2 - 1, z^3]`` (a Gaussian manifold fit in the
   lifted moment space) escapes it: it equals the floor on Gaussian data and
   beats it only when the anomaly carries 3rd-moment (skew) structure that
   mean+covariance cannot see.

Attribution: the floor/escape framing is a *blueprint* adapted from FINDOYOU
floor-theory work; the implementation and every number reported here are
Mercury's own, measured on Mercury's real fused anomaly score.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from omni_mercury_engine.ml.mercury_ml import roc_auc_score

ScoreFn = Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]]

_ATTACKS = ("condmean", "bpda", "nes", "transfer")


def _auc(y: np.ndarray[Any, Any], scores: np.ndarray[Any, Any]) -> float:
    labels = np.asarray(y, dtype=int).reshape(-1)
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size != arr.size or np.unique(labels).size != 2 or np.unique(arr).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, arr))


def _channel_mask(controlled: np.ndarray[Any, Any] | None, k: int) -> np.ndarray[Any, Any]:
    """Boolean ``(k,)`` mask of attacker-controlled channels (all if None)."""
    if controlled is None:
        return np.ones(k, dtype=bool)
    mask = np.zeros(k, dtype=bool)
    mask[np.asarray(controlled, dtype=int)] = True
    return mask


def _apply_controlled(
    adv: np.ndarray[Any, Any], x0: np.ndarray[Any, Any], mask: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Keep ``adv`` only on controlled channels; everything else stays at x0."""
    out = x0.copy()
    out[:, mask] = adv[:, mask]
    return out


def most_informative_channels(
    x: np.ndarray[Any, Any], y: np.ndarray[Any, Any], m: int
) -> np.ndarray[Any, Any]:
    """The ``m`` channels with the largest standardized normal/anomaly gap.

    Controlling the most discriminative channels is the strongest on-manifold
    budget, so the floor curve traced over these is a genuine worst case.
    """
    arr = np.asarray(x, dtype=np.float64)
    labels = np.asarray(y, dtype=int).reshape(-1)
    if m <= 0:
        return np.empty(0, dtype=int)
    pos, neg = arr[labels == 1], arr[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return np.arange(min(m, arr.shape[1]), dtype=int)
    scale = np.std(arr, axis=0) + 1e-12
    gap = np.abs(np.mean(pos, axis=0) - np.mean(neg, axis=0)) / scale
    return np.argsort(gap)[::-1][:m].astype(int)


def condmean_keyblind(
    x: np.ndarray[Any, Any],
    normal_reference: np.ndarray[Any, Any],
    controlled: np.ndarray[Any, Any] | None = None,
) -> np.ndarray[Any, Any]:
    """Replace controlled channels by the normal conditional means."""
    arr = np.asarray(x, dtype=np.float64).copy()
    ref = np.asarray(normal_reference, dtype=np.float64)
    channels = np.arange(arr.shape[1]) if controlled is None else np.asarray(controlled, dtype=int)
    arr[:, channels] = np.mean(ref[:, channels], axis=0)
    return arr


def attack_condmean_init_refine(
    score_fn: ScoreFn,
    x: np.ndarray[Any, Any],
    normal_reference: np.ndarray[Any, Any],
    *,
    controlled: np.ndarray[Any, Any] | None = None,
    eps: float = 0.5,
    steps: int = 8,
) -> np.ndarray[Any, Any]:
    """Coordinate refinement around the conditional-mean init (controlled only)."""
    base = np.asarray(x, dtype=np.float64)
    mask = _channel_mask(controlled, base.shape[1])
    means = np.mean(np.asarray(normal_reference, dtype=np.float64), axis=0)
    best = _apply_controlled(
        condmean_keyblind(base, normal_reference, np.where(mask)[0]), base, mask
    )
    best_scores = np.asarray(score_fn(best), dtype=np.float64)
    for j in np.where(mask)[0]:
        for frac in np.linspace(0.0, 1.0, steps):
            cand = best.copy()
            target = base[:, j] + frac * (means[j] - base[:, j])
            cand[:, j] = np.clip(target, base[:, j] - eps, base[:, j] + eps)
            scores = np.asarray(score_fn(cand), dtype=np.float64)
            keep = scores < best_scores
            best[keep, j] = cand[keep, j]
            best_scores[keep] = scores[keep]
    return best


def attack_bpda(
    score_fn: ScoreFn,
    x: np.ndarray[Any, Any],
    *,
    controlled: np.ndarray[Any, Any] | None = None,
    eps: float = 0.5,
    step: float = 0.05,
    steps: int = 12,
) -> np.ndarray[Any, Any]:
    """Finite-difference (gradient-based) evasion over controlled channels."""
    x0 = np.asarray(x, dtype=np.float64)
    mask = _channel_mask(controlled, x0.shape[1])
    adv = x0.copy()
    for _ in range(steps):
        grad = np.zeros_like(adv)
        for j in np.where(mask)[0]:
            plus, minus = adv.copy(), adv.copy()
            plus[:, j] += step
            minus[:, j] -= step
            grad[:, j] = (np.asarray(score_fn(plus)) - np.asarray(score_fn(minus))) / (2.0 * step)
        norm = np.linalg.norm(grad, axis=1, keepdims=True)
        direction = grad / np.maximum(norm, 1e-12)
        adv = np.clip(adv - step * direction, x0 - eps, x0 + eps)
        adv = _apply_controlled(adv, x0, mask)
    return adv


def attack_nes(
    score_fn: ScoreFn,
    x: np.ndarray[Any, Any],
    *,
    controlled: np.ndarray[Any, Any] | None = None,
    eps: float = 0.5,
    draws: int = 24,
    seed: int = 0,
) -> np.ndarray[Any, Any]:
    """Gradient-free random-search/NES evasion over controlled channels."""
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x, dtype=np.float64)
    mask = _channel_mask(controlled, x0.shape[1])
    best = x0.copy()
    best_scores = np.asarray(score_fn(best), dtype=np.float64)
    for _ in range(draws):
        noise = np.asarray(rng.normal(size=x0.shape))
        noise[:, ~mask] = 0.0
        nrm = np.maximum(np.linalg.norm(noise, axis=1, keepdims=True), 1e-12)
        cand = _apply_controlled(x0 + eps * noise / nrm, x0, mask)
        scores = np.asarray(score_fn(cand), dtype=np.float64)
        keep = scores < best_scores
        best[keep] = cand[keep]
        best_scores[keep] = scores[keep]
    return best


def attack_transfer(
    score_fn: ScoreFn,
    x: np.ndarray[Any, Any],
    surrogate_scores: np.ndarray[Any, Any],
    *,
    controlled: np.ndarray[Any, Any] | None = None,
    eps: float = 0.5,
) -> np.ndarray[Any, Any]:
    """Transfer attack: move high-surrogate rows' controlled channels to median."""
    arr = np.asarray(x, dtype=np.float64)
    mask = _channel_mask(controlled, arr.shape[1])
    med = np.median(arr, axis=0)
    order = np.argsort(np.asarray(surrogate_scores, dtype=np.float64).reshape(-1))[::-1]
    cand = arr.copy()
    half = order[: max(1, len(order) // 2)]
    target = np.clip(med[None, :], arr[half] - eps, arr[half] + eps)
    cand[np.ix_(half, np.where(mask)[0])] = target[:, mask]
    _ = score_fn(cand)
    return cand


@dataclass(frozen=True)
class AttackResult:
    name: str
    scores: np.ndarray[Any, Any]
    auroc: float


def worst_case_over_attacks(
    score_fn: ScoreFn,
    x: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
    *,
    normal_reference: np.ndarray[Any, Any] | None = None,
    controlled: np.ndarray[Any, Any] | None = None,
    eps: float = 0.5,
    seed: int = 0,
) -> dict[str, Any]:
    """Fixed-budget battery over *controlled* channels; worst-case AUROC report.

    ``gradient_masking_flag`` fires iff the gradient-free NES attack achieves a
    *lower* (stronger) AUROC than the gradient-based BPDA attack — the signature
    of masked gradients — independent of which attack is the global worst.
    ``win_counts`` is the true per-row tally: for each row, the attack that
    drove the score lowest (best evasion) gets the credit.
    """
    arr = np.asarray(x, dtype=np.float64)
    labels = np.asarray(y, dtype=int).reshape(-1)
    normal = (
        arr[labels == 0]
        if normal_reference is None
        else np.asarray(normal_reference, dtype=np.float64)
    )
    if normal.size == 0:
        normal = arr
    clean_scores = np.asarray(score_fn(arr), dtype=np.float64)
    advs = {
        "condmean": attack_condmean_init_refine(
            score_fn, arr, normal, controlled=controlled, eps=eps
        ),
        "bpda": attack_bpda(score_fn, arr, controlled=controlled, eps=eps),
        "nes": attack_nes(score_fn, arr, controlled=controlled, eps=eps, seed=seed),
        "transfer": attack_transfer(score_fn, arr, clean_scores, controlled=controlled, eps=eps),
    }
    per_attack_scores = {
        name: np.asarray(score_fn(adv), dtype=np.float64) for name, adv in advs.items()
    }
    results = {
        name: AttackResult(name, s, _auc(labels, s)) for name, s in per_attack_scores.items()
    }

    # True per-row win counts: the attack that minimised each row's score.
    stack = np.vstack([per_attack_scores[name] for name in _ATTACKS])  # (n_attacks, n)
    winners = np.argmin(stack, axis=0)
    win_counts = {name: int(np.sum(winners == i)) for i, name in enumerate(_ATTACKS)}

    finite = [r for r in results.values() if np.isfinite(r.auroc)]
    worst = min(finite, key=lambda r: r.auroc) if finite else next(iter(results.values()))
    nes_auc = results["nes"].auroc
    bpda_auc = results["bpda"].auroc
    masking = bool(np.isfinite(nes_auc) and np.isfinite(bpda_auc) and nes_auc < bpda_auc)
    return {
        "clean_auroc": _auc(labels, clean_scores),
        "worst_case_auroc": worst.auroc,
        "worst_attack": worst.name,
        "per_attack": {name: {"auroc": r.auroc} for name, r in results.items()},
        "win_counts": win_counts,
        "gradient_masking_flag": masking,
    }


def floor_curve(
    score_fn: ScoreFn,
    x: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
    *,
    normal_reference: np.ndarray[Any, Any] | None = None,
    eps: float = 0.5,
    seed: int = 0,
    ms: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Worst-case AUROC vs controlled-channel budget ``m`` (the floor curve).

    ``m`` sweeps ``{0, floor(k/4), floor(k/2), floor(3k/4)}`` by default; the
    controlled subset at each ``m`` is the most-informative channels.  At
    ``m = 0`` there is no perturbation budget, so this returns the clean AUROC;
    as ``m`` grows the attacker can drive the fused score toward the on-manifold
    half-channel floor.
    """
    arr = np.asarray(x, dtype=np.float64)
    k = arr.shape[1]
    if ms is None:
        ms = sorted({0, k // 4, k // 2, (3 * k) // 4})
    curve: list[dict[str, Any]] = []
    for m in ms:
        if m <= 0:
            clean = _auc(np.asarray(y, int), np.asarray(score_fn(arr), dtype=np.float64))
            curve.append({"m": 0, "worst_case_auroc": clean, "worst_attack": "none"})
            continue
        controlled = most_informative_channels(arr, y, m)
        rep = worst_case_over_attacks(
            score_fn,
            arr,
            y,
            normal_reference=normal_reference,
            controlled=controlled,
            eps=eps,
            seed=seed,
        )
        curve.append(
            {
                "m": int(m),
                "worst_case_auroc": rep["worst_case_auroc"],
                "worst_attack": rep["worst_attack"],
                "gradient_masking_flag": rep["gradient_masking_flag"],
            }
        )
    return curve


# ---------------------------------------------------------------------------
# Cubic-moment escape detector (polynomial-lift D_phi)
# ---------------------------------------------------------------------------


def _whiten(
    normal: np.ndarray[Any, Any], reg: float = 1e-6
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Return ``(mu, W)`` with ``W = Sigma^{-1/2}`` of the normal reference."""
    mu = np.mean(normal, axis=0)
    cov = np.cov(normal.T) + reg * np.eye(normal.shape[1])
    if cov.ndim == 0:
        cov = np.atleast_2d(cov)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-12, None)
    w = (vecs * np.sqrt(1.0 / vals)) @ vecs.T
    return mu, w


def _lift(z: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Polynomial moment lift ``phi(z) = [z, z^2 - 1, z^3]`` (column-stacked)."""
    return np.concatenate([z, z**2 - 1.0, z**3], axis=1)


def gaussian_floor_score(
    x: np.ndarray[Any, Any], normal_reference: np.ndarray[Any, Any], reg: float = 1e-6
) -> np.ndarray[Any, Any]:
    """Mahalanobis (mean+covariance only) score — the floor detector."""
    arr = np.asarray(x, dtype=np.float64)
    normal = np.asarray(normal_reference, dtype=np.float64)
    mu = np.mean(normal, axis=0)
    cov = np.cov(normal.T) + reg * np.eye(arr.shape[1])
    if cov.ndim == 0:
        cov = np.atleast_2d(cov)
    prec = np.linalg.pinv(cov, hermitian=True)
    z = arr - mu
    dist: np.ndarray[Any, Any] = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", z, prec, z), 0.0))
    return dist


def cubic_moment_score(
    x: np.ndarray[Any, Any], normal_reference: np.ndarray[Any, Any], reg: float = 1e-3
) -> np.ndarray[Any, Any]:
    """Polynomial-lift cubic-moment detector ``D_phi``.

    Fits a Gaussian manifold in the lifted moment space ``phi(z)``.  Because the
    lifted covariance is estimated on the (whitened) normal reference, it encodes
    the Gaussian moment relationships, so a *Gaussian* anomaly moves only along
    directions the manifold already expects (``D_phi`` reduces to the floor),
    while genuine skew/kurtosis pushes ``phi`` into under-modelled directions
    (``D_phi`` exceeds the floor).
    """
    arr = np.asarray(x, dtype=np.float64)
    normal = np.asarray(normal_reference, dtype=np.float64)
    mu, w = _whiten(normal)
    phi_n = _lift((normal - mu) @ w.T)
    phi_x = _lift((arr - mu) @ w.T)
    m_phi = np.mean(phi_n, axis=0)
    cov_phi = np.cov(phi_n.T) + reg * np.eye(phi_n.shape[1])
    prec = np.linalg.pinv(cov_phi, hermitian=True)
    d = phi_x - m_phi
    dist: np.ndarray[Any, Any] = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, prec, d), 0.0))
    return dist


def cubic_moment_escape(
    x: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
) -> dict[str, float]:
    """AUC of the cubic-moment detector vs the Gaussian floor (the *escape*).

    ``escape = cubic_auc - floor_auc`` is ~0 on Gaussian data and positive only
    when the anomaly carries 3rd-moment structure mean+covariance cannot see.
    """
    arr = np.asarray(x, dtype=np.float64)
    labels = np.asarray(y, dtype=int).reshape(-1)
    normal = arr[labels == 0] if np.any(labels == 0) else arr
    floor = gaussian_floor_score(arr, normal)
    cubic = cubic_moment_score(arr, normal)
    floor_auc = _auc(labels, floor)
    cubic_auc = _auc(labels, cubic)
    return {
        "floor_auc": floor_auc,
        "cubic_auc": cubic_auc,
        "escape": (
            cubic_auc - floor_auc
            if np.isfinite(cubic_auc) and np.isfinite(floor_auc)
            else float("nan")
        ),
    }
