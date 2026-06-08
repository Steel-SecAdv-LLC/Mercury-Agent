"""Pooled detection metrics for the governed-fusion suite (mercury_ml only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from omni_mercury_engine.ml.mercury_ml import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_auc(y: np.ndarray[Any, Any], s: np.ndarray[Any, Any]) -> float:
    y = np.asarray(y, dtype=int).reshape(-1)
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    if y.size == 0 or np.unique(y).size < 2 or np.unique(s).size < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _safe_auprc(y: np.ndarray[Any, Any], s: np.ndarray[Any, Any]) -> float:
    y = np.asarray(y, dtype=int).reshape(-1)
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def pooled_metrics(
    y: np.ndarray[Any, Any],
    score: np.ndarray[Any, Any],
    pred: np.ndarray[Any, Any],
) -> dict[str, float]:
    """AUROC/AUPRC from continuous scores; F1/P/R from the boolean verdict."""
    y = np.asarray(y, dtype=int).reshape(-1)
    pred = np.asarray(pred, dtype=int).reshape(-1)
    return {
        "n": int(y.size),
        "pos": int(np.sum(y == 1)),
        "auroc": _safe_auc(y, score),
        "auprc": _safe_auprc(y, score),
        "f1": float(f1_score(y, pred)),
        "precision": float(precision_score(y, pred)),
        "recall": float(recall_score(y, pred)),
    }


def fmt_row(name: str, m: dict[str, float]) -> str:
    return (
        f"{name:<22} n={m['n']:>5} pos={m['pos']:>4} "
        f"AUROC={m['auroc']:.3f} AUPRC={m['auprc']:.3f} "
        f"F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f}"
    )
