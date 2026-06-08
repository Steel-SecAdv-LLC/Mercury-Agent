# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

from __future__ import annotations

import argparse
import math
from typing import Any

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.live_dataset_protection_gate/v1"
_HIST_BINS = 16


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.live_dataset_protection_gate",
        description=(
            "Defend the live dataset as Mercury's reference distribution: "
            "verify that any reenactment (synthetic fallback) is faithful to live, "
            "and fail-closed when the reenactment drifts beyond tolerance."
        ),
    )
    parser.add_argument(
        "--live-scores",
        required=True,
        help=(
            "Reference live-dataset detector outputs (.npy, shape (N, K)). "
            "This is the source of truth; the reenactment is judged against it."
        ),
    )
    parser.add_argument(
        "--reenactment-scores",
        "--synthetic-scores",
        dest="reenactment_scores",
        required=True,
        help=(
            "Reenactment detector outputs (.npy, shape (N, K)) — only "
            "acceptable when live ingestion has degraded.  Synthetic origin "
            "is OK only when this gate passes against the live reference. "
            "``--synthetic-scores`` is accepted as a legacy alias."
        ),
    )
    parser.add_argument(
        "--live-labels",
        default=None,
        help="Optional .npy of binary ground-truth labels for the live set (the reference).",
    )
    parser.add_argument(
        "--reenactment-labels",
        "--synthetic-labels",
        dest="reenactment_labels",
        default=None,
        help="Optional .npy of binary ground-truth labels for the reenactment set.",
    )
    parser.add_argument(
        "--column-names",
        default=None,
        help="Comma-separated names for the K score columns.",
    )
    parser.add_argument(
        "--ks-max",
        type=float,
        default=0.20,
        help="Maximum permitted KS two-sample statistic per column.",
    )
    parser.add_argument(
        "--kl-max",
        type=float,
        default=0.50,
        help="Maximum permitted symmetric KL divergence per column (nats).",
    )
    parser.add_argument(
        "--auroc-drop-max",
        type=float,
        default=0.05,
        help="Maximum permitted absolute AUROC delta (live - reenactment).",
    )
    return parser


def _ks_two_sample(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    """Handwritten Kolmogorov–Smirnov two-sample D statistic.

    The KS statistic is the maximum absolute difference between the
    two empirical CDFs evaluated on the union of points.  Computed
    without scipy so the tool ships with the engine.
    """
    a_sorted = np.sort(a.astype(np.float64))
    b_sorted = np.sort(b.astype(np.float64))
    all_pts = np.sort(np.unique(np.concatenate([a_sorted, b_sorted])))
    # ``np.searchsorted(side='right')`` returns the position of each
    # query value in the sorted array, which is exactly the empirical
    # CDF count.
    cdf_a = np.searchsorted(a_sorted, all_pts, side="right") / len(a_sorted)
    cdf_b = np.searchsorted(b_sorted, all_pts, side="right") / len(b_sorted)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _hist(a: npt.NDArray[np.float64], lo: float, hi: float) -> npt.NDArray[np.float64]:
    counts, _ = np.histogram(a, bins=_HIST_BINS, range=(lo, hi))
    # Laplace-smooth so the KL divergence is finite even when a bin is
    # empty in one sample.
    p = (counts + 1).astype(np.float64)
    normalised: npt.NDArray[np.float64] = p / p.sum()
    return normalised


def _symmetric_kl(p: npt.NDArray[np.float64], q: npt.NDArray[np.float64]) -> float:
    """Jensen-style symmetric KL: KL(P||M) + KL(Q||M) where M = (P+Q)/2."""
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return kl_pm + kl_qm


def _auroc(scores: npt.NDArray[np.float64], labels: npt.NDArray[np.float64]) -> float:
    """Handwritten AUROC via the U-statistic identity.

    AUROC = P(score_pos > score_neg) + 0.5 * P(score_pos == score_neg).
    Implemented as a vectorised rank-sum; no scikit-learn dependency.
    """
    labels = labels.astype(np.int64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1)
    # Tied-rank correction.
    combined = np.concatenate([pos, neg])
    for v in np.unique(combined):
        idx = combined == v
        if idx.sum() > 1:
            ranks[idx] = ranks[idx].mean()
    pos_rank_sum = ranks[: pos.size].sum()
    n_pos = pos.size
    n_neg = neg.size
    u = pos_rank_sum - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _load_scores(path: str) -> npt.NDArray[np.float64]:
    raw = np.load(path, allow_pickle=False)
    arr: npt.NDArray[np.float64] = raw.astype(np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _collect(args: argparse.Namespace) -> Certificate:
    live = _load_scores(args.live_scores)
    reenactment = _load_scores(args.reenactment_scores)
    if live.shape[1] != reenactment.shape[1]:
        return Certificate(
            tool="live_dataset_protection_gate",
            schema=_SCHEMA,
            status="fail",
            body={
                "error": (
                    f"column count mismatch: live {live.shape[1]} != "
                    f"reenactment {reenactment.shape[1]} — reenactment must "
                    "match the live schema exactly"
                ),
                "policy": "live_is_reference",
            },
        )
    n_cols = live.shape[1]
    names = (
        args.column_names.split(",") if args.column_names else [f"col{i}" for i in range(n_cols)]
    )
    if len(names) != n_cols:
        return Certificate(
            tool="live_dataset_protection_gate",
            schema=_SCHEMA,
            status="fail",
            body={
                "error": f"--column-names has {len(names)} entries; expected {n_cols}",
                "policy": "live_is_reference",
            },
        )

    live_labels = np.load(args.live_labels, allow_pickle=False) if args.live_labels else None
    reenactment_labels = (
        np.load(args.reenactment_labels, allow_pickle=False) if args.reenactment_labels else None
    )

    per_col: list[dict[str, Any]] = []
    failures: list[str] = []
    for i, name in enumerate(names):
        live_col = live[:, i]
        reenactment_col = reenactment[:, i]
        ks = _ks_two_sample(live_col, reenactment_col)
        lo = float(min(live_col.min(), reenactment_col.min()))
        hi = float(max(live_col.max(), reenactment_col.max()))
        if math.isclose(lo, hi):
            hi = lo + 1.0  # avoid degenerate single-bin histogram
        kl = _symmetric_kl(_hist(live_col, lo, hi), _hist(reenactment_col, lo, hi))
        auroc_live = _auroc(live_col, live_labels) if live_labels is not None else float("nan")
        auroc_reenactment = (
            _auroc(reenactment_col, reenactment_labels)
            if reenactment_labels is not None
            else float("nan")
        )
        auroc_drop = (
            abs(auroc_live - auroc_reenactment)
            if not (math.isnan(auroc_live) or math.isnan(auroc_reenactment))
            else None
        )
        # Live is the reference; the reenactment is judged against it.
        record: dict[str, Any] = {
            "name": name,
            "ks": ks,
            "kl_symmetric": kl,
            "reference_live": {
                "mean": float(live_col.mean()),
                "std": float(live_col.std()),
                "p05": float(np.quantile(live_col, 0.05)),
                "p95": float(np.quantile(live_col, 0.95)),
                "auroc": auroc_live if not math.isnan(auroc_live) else None,
            },
            "reenactment": {
                "mean": float(reenactment_col.mean()),
                "std": float(reenactment_col.std()),
                "p05": float(np.quantile(reenactment_col, 0.05)),
                "p95": float(np.quantile(reenactment_col, 0.95)),
                "auroc": auroc_reenactment if not math.isnan(auroc_reenactment) else None,
            },
            "auroc_drop": auroc_drop,
        }
        per_col.append(record)
        if ks > float(args.ks_max):
            failures.append(
                f"{name}: reenactment KS {ks:.3f} > {args.ks_max} — reenactment drifted from live"
            )
        if kl > float(args.kl_max):
            failures.append(
                f"{name}: reenactment symmetric-KL {kl:.3f} > {args.kl_max} — reenactment drifted from live"
            )
        if auroc_drop is not None and auroc_drop > float(args.auroc_drop_max):
            failures.append(
                f"{name}: |AUROC live - reenactment| {auroc_drop:.3f} > {args.auroc_drop_max} "
                "— reenactment regresses the live discriminative signal"
            )

    body: dict[str, Any] = {
        "policy": "live_is_reference",
        "live_scores": args.live_scores,
        "reenactment_scores": args.reenactment_scores,
        "live_n": int(live.shape[0]),
        "reenactment_n": int(reenactment.shape[0]),
        "columns": per_col,
        "thresholds": {
            "ks_max": float(args.ks_max),
            "kl_max": float(args.kl_max),
            "auroc_drop_max": float(args.auroc_drop_max),
        },
        "failures": failures,
        "resolution_guidance": (
            "Restore live ingestion; if a reenactment is required, refresh it from the "
            "most-recently-collected live corpus and re-run this gate."
        ),
    }
    return Certificate(
        tool="live_dataset_protection_gate",
        schema=_SCHEMA,
        status="fail" if failures else "ok",
        body=body,
        warnings=failures,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
