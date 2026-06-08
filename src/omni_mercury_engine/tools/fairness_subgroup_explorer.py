# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Operator tool: intersectional fairness subgroup explorer.

Extends the existing :mod:`bias_audit_standalone` (which audits a
single sensitive feature) to the full cartesian product of sensitive
features.  Auto-discovers intersectional subgroups and ranks them by
Demographic Parity Difference (DPD) and Equal Opportunity Difference
(EOD).

Inputs are three ``.npy`` files:

* ``--features``: shape (N, K) with K sensitive-feature columns.
  Columns are treated as categorical — non-integer values are
  bucketed to their string form.
* ``--scores``: shape (N,) model scores or binary predictions.
* ``--labels``: shape (N,) ground-truth labels (0/1).

Output: every intersectional subgroup with its size, positive rate
(DP), TPR (EO), DPD, and EOD relative to the global rate.  Fails when
any subgroup with at least ``--min-size`` samples exceeds the
``--dpd-max`` or ``--eod-max`` thresholds.
"""

from __future__ import annotations

import argparse
import itertools
from typing import Any

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.fairness_subgroup_explorer/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.fairness_subgroup_explorer",
        description=("Explore intersectional fairness subgroups and rank by DPD/EOD."),
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument(
        "--feature-names",
        default=None,
        help="Comma-separated names for the K feature columns.",
    )
    parser.add_argument("--min-size", type=int, default=8)
    parser.add_argument("--dpd-max", type=float, default=0.10)
    parser.add_argument("--eod-max", type=float, default=0.10)
    parser.add_argument(
        "--max-cardinality",
        type=int,
        default=12,
        help=(
            "Cap on the number of distinct levels per feature.  Features "
            "with more levels are bucketed to the top --max-cardinality "
            "by count plus an 'other' bucket."
        ),
    )
    return parser


def _bucket(col: npt.NDArray[Any], max_card: int) -> npt.NDArray[np.str_]:
    """Return the column with high-cardinality levels collapsed to 'other'.

    The output is always a string ndarray — the caller groups subgroups
    by string identity (``"other"`` for the collapsed tail, the
    original level name otherwise).  Accepting ``NDArray[Any]`` keeps
    the function callable on numeric *and* object-typed inputs, which
    is what ``np.unique`` returns when sensitive features are mixed
    string / int columns.
    """
    levels, counts = np.unique(col, return_counts=True)
    if len(levels) <= max_card:
        return col.astype(str)
    keep = {str(v) for v in levels[np.argsort(-counts)][:max_card]}
    return np.array(["other" if str(v) not in keep else str(v) for v in col], dtype=np.str_)


def _collect(args: argparse.Namespace) -> Certificate:
    features = np.load(args.features, allow_pickle=False)
    scores = np.load(args.scores, allow_pickle=False).ravel()
    labels = np.load(args.labels, allow_pickle=False).astype(int).ravel()
    if features.ndim != 2:
        return Certificate(
            tool="fairness_subgroup_explorer",
            schema=_SCHEMA,
            status="fail",
            body={"error": f"--features must be 2-D; got shape {features.shape}"},
        )
    if not (features.shape[0] == scores.size == labels.size):
        return Certificate(
            tool="fairness_subgroup_explorer",
            schema=_SCHEMA,
            status="fail",
            body={
                "error": (
                    f"row mismatch: features {features.shape[0]}, "
                    f"scores {scores.size}, labels {labels.size}"
                ),
            },
        )

    # Binarise scores using the global median so the tool works for
    # both probability and binary inputs.
    preds = (scores >= float(np.median(scores))).astype(int)
    K = features.shape[1]
    names = args.feature_names.split(",") if args.feature_names else [f"f{i}" for i in range(K)]
    if len(names) != K:
        return Certificate(
            tool="fairness_subgroup_explorer",
            schema=_SCHEMA,
            status="fail",
            body={"error": f"--feature-names has {len(names)} entries; expected {K}"},
        )

    bucketed = [_bucket(features[:, i], int(args.max_cardinality)) for i in range(K)]
    global_dp = float(preds.mean())
    pos_mask = labels == 1
    global_tpr = float(preds[pos_mask].mean()) if pos_mask.any() else 0.0

    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    # Enumerate the full cartesian product of (level_per_feature) tuples.
    unique_levels = [sorted(set(col.tolist())) for col in bucketed]
    for combo in itertools.product(*unique_levels):
        mask = np.ones_like(labels, dtype=bool)
        for i, level in enumerate(combo):
            mask &= bucketed[i] == level
        n = int(mask.sum())
        if n < int(args.min_size):
            continue
        dp = float(preds[mask].mean())
        sub_pos = mask & pos_mask
        tpr = float(preds[sub_pos].mean()) if sub_pos.any() else 0.0
        dpd = abs(dp - global_dp)
        eod = abs(tpr - global_tpr)
        row = {
            "subgroup": dict(zip(names, [str(v) for v in combo], strict=True)),
            "n": float(n),
            "dp": dp,
            "tpr": tpr,
            "dpd": dpd,
            "eod": eod,
        }
        rows.append(row)
        if dpd > float(args.dpd_max) or eod > float(args.eod_max):
            failed.append(row)

    rows.sort(key=lambda r: (-r["dpd"], -r["eod"]))
    body: dict[str, Any] = {
        "n": int(labels.size),
        "global_dp": global_dp,
        "global_tpr": global_tpr,
        "feature_names": names,
        "thresholds": {
            "dpd_max": float(args.dpd_max),
            "eod_max": float(args.eod_max),
            "min_size": int(args.min_size),
        },
        "subgroups": rows,
        "failed": failed,
    }
    return Certificate(
        tool="fairness_subgroup_explorer",
        schema=_SCHEMA,
        status="fail" if failed else "ok",
        body=body,
        warnings=[
            f"subgroup {r['subgroup']} exceeds DPD/EOD threshold (DPD={r['dpd']:.3f}, EOD={r['eod']:.3f})"
            for r in failed
        ],
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
