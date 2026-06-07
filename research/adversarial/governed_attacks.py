"""Adversarial-survivability probes for the governed fusion substrate.

Attribution: adapted from FINDOYOU ``mercury_equation/adaptive_attacks.py`` and
``floor_theory.py``.  The harness is test/research-only: it consumes a score
function and never changes Mercury runtime decisions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from omni_mercury_engine.ml.mercury_ml import roc_auc_score

ScoreFn = Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]]


@dataclass(frozen=True)
class AttackResult:
    name: str
    scores: np.ndarray[Any, Any]
    auroc: float


def _auc(y: np.ndarray[Any, Any], scores: np.ndarray[Any, Any]) -> float:
    labels = np.asarray(y, dtype=int).reshape(-1)
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size != arr.size or np.unique(labels).size != 2 or np.unique(arr).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, arr))


def condmean_keyblind(
    x: np.ndarray[Any, Any],
    normal_reference: np.ndarray[Any, Any],
    controlled: np.ndarray[Any, Any] | None = None,
) -> np.ndarray[Any, Any]:
    """Replace controlled channels by normal conditional means."""

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
    eps: float = 0.5,
    steps: int = 8,
) -> np.ndarray[Any, Any]:
    """Coordinate refinement around the conditional-mean initialization."""

    base = np.asarray(x, dtype=np.float64)
    means = np.mean(np.asarray(normal_reference, dtype=np.float64), axis=0)
    best = condmean_keyblind(base, normal_reference)
    best_scores = np.asarray(score_fn(best), dtype=np.float64)
    for j in range(base.shape[1]):
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
    eps: float = 0.5,
    step: float = 0.05,
    steps: int = 12,
) -> np.ndarray[Any, Any]:
    """Finite-difference exact-gradient attack against the supplied score function."""

    x0 = np.asarray(x, dtype=np.float64)
    adv = x0.copy()
    for _ in range(steps):
        grad = np.zeros_like(adv)
        for j in range(adv.shape[1]):
            plus = adv.copy()
            minus = adv.copy()
            plus[:, j] += step
            minus[:, j] -= step
            grad[:, j] = (np.asarray(score_fn(plus)) - np.asarray(score_fn(minus))) / (2.0 * step)
        norm = np.linalg.norm(grad, axis=1, keepdims=True)
        direction = grad / np.maximum(norm, 1e-12)
        adv = np.clip(adv - step * direction, x0 - eps, x0 + eps)
    return adv


def attack_random_search(
    score_fn: ScoreFn,
    x: np.ndarray[Any, Any],
    *,
    eps: float = 0.5,
    draws: int = 24,
    seed: int = 0,
) -> np.ndarray[Any, Any]:
    """Gradient-free NES/random-search probe for gradient masking."""

    rng = np.random.default_rng(seed)
    x0 = np.asarray(x, dtype=np.float64)
    best = x0.copy()
    best_scores = np.asarray(score_fn(best), dtype=np.float64)
    for _ in range(draws):
        noise = rng.normal(size=x0.shape)
        noise /= np.maximum(np.linalg.norm(noise, axis=1, keepdims=True), 1e-12)
        cand = x0 + eps * noise
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
    eps: float = 0.5,
) -> np.ndarray[Any, Any]:
    """Transfer attack by moving highest-surrogate-score rows toward the batch median."""

    arr = np.asarray(x, dtype=np.float64)
    med = np.median(arr, axis=0)
    order = np.argsort(np.asarray(surrogate_scores, dtype=np.float64).reshape(-1))[::-1]
    cand = arr.copy()
    half = order[: max(1, len(order) // 2)]
    cand[half] = np.clip(med, arr[half] - eps, arr[half] + eps)
    _ = score_fn(cand)
    return cand


def worst_case_over_attacks(
    score_fn: ScoreFn,
    x: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
    *,
    normal_reference: np.ndarray[Any, Any] | None = None,
    eps: float = 0.5,
    seed: int = 0,
) -> dict[str, Any]:
    """Run fixed-budget attacks and report worst-case AUROC plus NES masking check."""

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
    candidates = {
        "clean": arr,
        "condmean": attack_condmean_init_refine(score_fn, arr, normal, eps=eps),
        "bpda": attack_bpda(score_fn, arr, eps=eps),
        "nes": attack_random_search(score_fn, arr, eps=eps, seed=seed),
        "transfer": attack_transfer(score_fn, arr, clean_scores, eps=eps),
    }
    results = []
    for name, adv in candidates.items():
        scores = np.asarray(score_fn(adv), dtype=np.float64)
        results.append(AttackResult(name=name, scores=scores, auroc=_auc(labels, scores)))
    finite = [r for r in results if np.isfinite(r.auroc)]
    worst = min(finite, key=lambda r: r.auroc) if finite else results[0]
    return {
        "clean_auroc": next(r.auroc for r in results if r.name == "clean"),
        "worst_case_auroc": worst.auroc,
        "worst_attack": worst.name,
        "per_attack": {r.name: {"auroc": r.auroc} for r in results},
        "win_counts": {r.name: int(r.name == worst.name) for r in results},
        "gradient_masking_flag": bool(worst.name == "nes" and worst.auroc < results[2].auroc),
    }


def sparse_score_statistic(
    x: np.ndarray[Any, Any], loc: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    centered = np.asarray(x, dtype=np.float64) - np.asarray(loc, dtype=np.float64)
    return np.max(np.abs(centered), axis=1)


def cubic_mahalanobis_floor(
    x: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
) -> dict[str, float]:
    """Moment-matched floor probe for detectors using only mean/covariance."""

    arr = np.asarray(x, dtype=np.float64)
    labels = np.asarray(y, dtype=int).reshape(-1)
    loc = np.mean(arr[labels == 0], axis=0) if np.any(labels == 0) else np.mean(arr, axis=0)
    cov = np.cov((arr - loc).T) + np.eye(arr.shape[1]) * 1e-6
    prec = np.linalg.pinv(cov, hermitian=True)
    centered = arr - loc
    mahal = np.einsum("ij,jk,ik->i", centered, prec, centered)
    cubic = np.cbrt(np.maximum(mahal, 0.0))
    sparse = sparse_score_statistic(arr, loc)
    return {
        "mahalanobis_floor_auc": _auc(labels, cubic),
        "sparse_axis_auc": _auc(labels, sparse),
        "mean_gap": (
            float(np.mean(cubic[labels == 1]) - np.mean(cubic[labels == 0]))
            if np.any(labels == 0) and np.any(labels == 1)
            else float("nan")
        ),
    }
