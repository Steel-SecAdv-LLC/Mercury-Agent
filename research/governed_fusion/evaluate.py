# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-event-mean aggregation for the governed-fusion suite.

A *scorer* maps one event's cached scores to ``(y, score, pred)``.  We compute
each event's metrics independently (every event self-calibrates its own
operating point) and report the macro mean across events — the same
equal-weight-per-event convention as ``research/omni_equation`` — so a single
148k-row event cannot swamp 28 smaller ones.  Per-domain means are reported
alongside the overall mean; regressions are shown, never averaged away.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np

from research.governed_fusion.label_provenance import (
    event_is_external_label,
    label_provenance,
    series_provenance,
)
from research.governed_fusion.metrics import pooled_metrics
from research.governed_fusion.score_cache import EventScores

Scorer = Callable[
    [EventScores], tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]
]

_KEYS = ("auroc", "auprc", "f1", "precision", "recall")


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    xs = [r[key] for r in rows if isinstance(r.get(key), float) and not np.isnan(r[key])]
    return float(np.mean(xs)) if xs else float("nan")


def aggregate(events: list[EventScores], scorer: Scorer) -> dict[str, Any]:
    """Return per-event rows, per-domain / per-provenance means and overall macro mean.

    ``per_provenance`` carries the honest fitness substrate's headline: the
    mean over the ``external_label`` bucket (live events with audited genuine
    labels) reported separately from the leakage-flagged ``self_label``
    bucket and the ``reconstructed`` bucket.  The autonomous loop's promotion
    gate reads ``per_provenance["external_label"]`` only.
    """
    per_event: list[dict[str, Any]] = []
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_provenance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for es in events:
        y, score, pred = scorer(es)
        m: dict[str, Any] = dict(pooled_metrics(y, score, pred))
        m["domain"] = es.domain
        m["event"] = es.event_id
        m["label_provenance"] = label_provenance(es.domain)
        m["series_provenance"] = series_provenance(es.domain, es.event_id)
        m["external_label"] = bool(event_is_external_label(es.domain, es.event_id))
        per_event.append(m)
        by_domain[es.domain].append(m)
        if m["series_provenance"] == "reconstructed":
            bucket = "reconstructed"
        elif m["external_label"]:
            bucket = "external_label"
        else:
            bucket = "self_label"
        by_provenance[bucket].append(m)
    per_domain = {
        dom: {k: _mean(rows, k) for k in _KEYS} | {"n_events": len(rows)}
        for dom, rows in by_domain.items()
    }
    per_provenance = {
        bucket: {k: _mean(rows, k) for k in _KEYS} | {"n_events": len(rows)}
        for bucket, rows in by_provenance.items()
    }
    overall = {k: _mean(per_event, k) for k in _KEYS} | {"n_events": len(per_event)}
    return {
        "per_event": per_event,
        "per_domain": per_domain,
        "per_provenance": per_provenance,
        "overall": overall,
    }


def print_compare(
    name_a: str,
    res_a: dict[str, Any],
    name_b: str,
    res_b: dict[str, Any],
) -> None:
    """Print a per-domain + overall before/after table with deltas."""
    doms = sorted(set(res_a["per_domain"]) | set(res_b["per_domain"]))
    hdr = f"{'domain':<18}" + "".join(f"{k:>9}" for k in _KEYS)
    print(f"\n{'=' * len(hdr)}")
    print(f"{name_a}  ->  {name_b}   (per-event macro mean)")
    print("=" * len(hdr))
    print(f"{'':<18}{name_a:>27} | {name_b}")
    for dom in doms + ["OVERALL"]:
        a = res_a["per_domain"].get(dom) if dom != "OVERALL" else res_a["overall"]
        b = res_b["per_domain"].get(dom) if dom != "OVERALL" else res_b["overall"]
        if a is None or b is None:
            continue
        sep = "-" if dom != "OVERALL" else "="
        print(sep * len(hdr))
        line_a = f"{dom:<18}" + "".join(f"{a[k]:>9.3f}" for k in _KEYS)
        line_b = f"{'  delta':<18}" + "".join(f"{b[k] - a[k]:>+9.3f}" for k in _KEYS)
        print(line_a)
        print(line_b)
