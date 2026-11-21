# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""How ``results.json`` states an event's AUROC, and how its means are taken.

An event's AUROC does not exist when its labels hold one class or when the
detector gives every row the same score (the cholera event, once its case
columns are excluded, has no non-zero column left). Such an event used to be
written as a bare ``NaN``, which is not JSON, and its status was only
implied by the number being missing from a mean.

It is now written as ``null`` with ``auroc_status`` saying why, counted in
``events_auroc_defined`` and listed under ``auroc_undefined``, and left out
of every mean: the convention ``benchmarks/domain_benchmark_base.py`` follows
(``auc_undefined``). ``measure_cached.py`` and ``remeasure_recorded.py`` both
summarise through :func:`summarise`, so the two cannot disagree about it.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

#: The per-event metrics a mean is taken over.
METRICS = (
    "auroc",
    "auprc",
    "f1_before",
    "f1_after",
    "prec_before",
    "prec_after",
    "rec_before",
    "rec_after",
)

MEASURED = "measured"
ONE_CLASS = "undefined: one class only"
CONSTANT_SCORE = "undefined: constant score"


def auroc_status(labels: Any, scores: Any) -> str:
    """Whether an AUROC exists for these labels and scores, and if not, why.

    Args:
        labels: Binary ground truth, one per row.
        scores: The detector's score, one per row.

    Returns:
        :data:`MEASURED`, :data:`ONE_CLASS` or :data:`CONSTANT_SCORE`.
    """
    y = np.asarray(labels).astype(int).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    if y.size == 0 or y.min() == y.max():
        return ONE_CLASS
    if np.unique(s).size < 2:
        return CONSTANT_SCORE
    return MEASURED


def is_number(value: Any) -> bool:
    """True for a finite ``int`` or ``float`` (not ``None``, ``NaN`` or ``bool``)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def mean(rows: list[dict[str, Any]], key: str, domain: str | None = None) -> float | None:
    """Mean of ``key`` over the rows that have it, to four places; ``None`` if none do."""
    values = [
        float(row[key])
        for row in rows
        if (domain is None or row["dom"] == domain) and is_number(row.get(key))
    ]
    return round(float(np.mean(values)), 4) if values else None


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The artifact ``results.json`` holds, from its per-event rows.

    Args:
        rows: One dict per event with ``dom``, ``event`` and the metrics; an
            undefined AUROC is ``None`` (or ``NaN``) with ``auroc_status``.

    Returns:
        ``events``, ``events_auroc_defined``, ``auroc_undefined``,
        ``domains``, ``overall``, ``per_domain`` and ``per_event``.
    """
    rows = sorted((json_ready(row) for row in rows), key=lambda row: (row["dom"], row["event"]))
    for row in rows:
        row.setdefault("auroc_status", MEASURED if is_number(row.get("auroc")) else ONE_CLASS)
    domains = sorted({row["dom"] for row in rows})
    per_domain: dict[str, Any] = {}
    for domain in domains:
        mine = [row for row in rows if row["dom"] == domain]
        per_domain[domain] = {
            **{key: mean(rows, key, domain) for key in METRICS},
            "n_events": len(mine),
            "n_auroc_defined": sum(is_number(row.get("auroc")) for row in mine),
        }
    return {
        "events": len(rows),
        "events_auroc_defined": sum(is_number(row.get("auroc")) for row in rows),
        "auroc_undefined": [
            {"dom": row["dom"], "event": row["event"], "reason": row["auroc_status"]}
            for row in rows
            if not is_number(row.get("auroc"))
        ],
        "domains": domains,
        "overall": {key: mean(rows, key) for key in METRICS},
        "per_domain": per_domain,
        "per_event": rows,
    }


def json_ready(value: Any) -> Any:
    """``value`` with every ``NaN`` or infinity replaced by ``None``, numpy types unwrapped."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path | str, artifact: dict[str, Any]) -> None:
    """Write ``artifact`` as strict JSON: a non-finite number is an error, not ``NaN``."""
    text = json.dumps(json_ready(artifact), indent=2, allow_nan=False) + "\n"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
