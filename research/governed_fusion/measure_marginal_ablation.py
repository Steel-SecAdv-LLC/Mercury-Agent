# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fusion-marginal ablation ledger (Phase 1 of governed self-improvement).

For every component in the default fusion stack (``resonance``, ``kinematic``,
``info_geo``), compute the leave-one-out marginal lift on **external-label
live events only** -- the honest fitness substrate Phase 2's promotion gate
reads from.  The lift per component is

    delta_AUROC[c] = AUROC(mean(all components)) - AUROC(mean(all - c))

with the analogous deltas for AUPRC and F1.  Means are macro across events
(equal weight per event, never row-pooled), matching the
``evaluate.aggregate`` convention used by every other governed-fusion
measurement.

The ledger is the fitness function the whole autonomous self-improvement
loop will climb.  It is the artefact the future Phase 2 promotion gate
compares against to decide whether a candidate self-change (new threshold,
fusion weight, enabled detector, retrained model) is genuinely better.

The ledger lives at ``research/governed_fusion/ablation_ledger.json``.
Each run appends one record under ``runs``; the last record's per-component
lifts form the baseline that ``test_marginal_ablation_regression`` checks
against the next run.

CI runs::

    source research/governed_fusion/gf_env.sh
    python research/governed_fusion/measure_marginal_ablation.py

When the score cache is absent (typical on a fresh CI runner that has not
yet built ``$GF_CACHE_DIR``), the script writes a ``needs_cache`` record so
the ledger keeps a chronological account of when the suite was last
runnable, and exits ``0`` in informational mode (default) or ``1`` in
``--check`` mode.  The leakage-split discipline (every event correctly
tagged, the external-label bucket non-empty) is enforced offline and
exits non-zero independently of whether the cache is present.

The intentional simplification of using a *mean-of-components* fusion
ablation is the only fusion-style available offline-from-cache today; the
shipped fusion's combined-score is precomputed and we cannot recompute
"fusion without component c" without re-running the model.  When the
real ablation surface (re-train without c) is wired by Phase 3's dormant-
module revival job, the ledger schema is forward-compatible.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any

import numpy as np

from research.governed_fusion.label_provenance import external_label_events
from research.governed_fusion.metrics import _safe_auc, _safe_auprc
from research.governed_fusion.score_cache import EventScores
from research.governed_fusion.suite import build_suite

_LEDGER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ablation_ledger.json")
_DEFAULT_CACHE_DIR = os.environ.get("GF_CACHE_DIR", "/home/user/gf_cache")

# Detectors whose marginal lift the ledger tracks.  Names must match the
# attribute names on ``EventScores``.
_COMPONENTS = ("resonance", "kinematic", "info_geo")


def _f1_from_scores(y: np.ndarray[Any, Any], s: np.ndarray[Any, Any]) -> float:
    """Best-of-threshold F1, mirroring the suite's oracle-F1 convention.

    The honest-fitness ledger reports the *best-achievable* operational F1
    (a calibration-independent, ranking-only summary), the same convention
    the rest of the governed-fusion measurements use.
    """
    y = np.asarray(y, dtype=int).reshape(-1)
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    order = np.argsort(-s)
    ys = y[order]
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    pos = int((y == 1).sum())
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(pos, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(np.nanmax(f1)) if f1.size else float("nan")


def _stack_components(es: EventScores) -> np.ndarray[Any, Any]:
    """Per-row, per-component score matrix in canonical column order."""
    return np.column_stack([getattr(es, c) for c in _COMPONENTS])


def _component_metrics(es: EventScores, include: tuple[int, ...]) -> dict[str, float]:
    """Macro metrics for one event, fusing the components at column indices ``include``.

    The fusion here is a deliberately simple **z-score-normalised mean**
    over the included columns.  It is monotone-equivalent to the unsupervised
    rank fusion the default detector uses for events with a single calibrated
    operating point; for ranking metrics (AUROC, AUPRC, oracle-F1) this is a
    faithful leave-one-out ablation surface against the same components.
    """
    cols = _stack_components(es)[:, list(include)]
    # Column-wise z-normalisation so a high-variance component does not
    # dominate the mean and trivialise the ablation.
    mu = cols.mean(axis=0, keepdims=True)
    sd = cols.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    s = ((cols - mu) / sd).mean(axis=1)
    return {
        "auroc": _safe_auc(es.y, s),
        "auprc": _safe_auprc(es.y, s),
        "f1": _f1_from_scores(es.y, s),
    }


def _macro_mean(metrics_per_event: list[dict[str, float]], key: str) -> float:
    xs = [m[key] for m in metrics_per_event if isinstance(m[key], float) and not np.isnan(m[key])]
    return float(np.mean(xs)) if xs else float("nan")


def compute_marginal_lift(events: list[EventScores]) -> dict[str, Any]:
    """Compute per-component leave-one-out marginal lift on an event list.

    Returns a dict ``{ "full": <metrics>, "leave_one_out": { component:
    {ablated_metrics, delta_metrics} } }`` where each metrics block is
    ``{auroc, auprc, f1, n_events}``.
    """
    full_idx = tuple(range(len(_COMPONENTS)))
    full_per_event = [_component_metrics(es, full_idx) for es in events]
    full = {
        "auroc": _macro_mean(full_per_event, "auroc"),
        "auprc": _macro_mean(full_per_event, "auprc"),
        "f1": _macro_mean(full_per_event, "f1"),
        "n_events": len(events),
    }
    loo: dict[str, dict[str, Any]] = {}
    for j, name in enumerate(_COMPONENTS):
        keep = tuple(i for i in full_idx if i != j)
        without_per_event = [_component_metrics(es, keep) for es in events]
        ablated = {
            "auroc": _macro_mean(without_per_event, "auroc"),
            "auprc": _macro_mean(without_per_event, "auprc"),
            "f1": _macro_mean(without_per_event, "f1"),
            "n_events": len(events),
        }
        loo[name] = {
            "ablated": ablated,
            "delta": {
                "auroc": full["auroc"] - ablated["auroc"],
                "auprc": full["auprc"] - ablated["auprc"],
                "f1": full["f1"] - ablated["f1"],
            },
        }
    return {"full": full, "leave_one_out": loo}


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        return out or None
    except Exception:
        return None


def _has_cache(cache_dir: str) -> bool:
    return os.path.isdir(cache_dir) and any(os.scandir(cache_dir))


def _load_external_label_scores(cache_dir: str) -> list[EventScores]:
    """Load cached scores for the external-label live subset only.

    Refuses to fit live (which would require network) -- this script is the
    measurement step, not the fit step. ``measure_baseline.py`` is the
    designated fit-and-cache entry point.
    """
    # Lazy import: pulling event_scores cold-imports the detector, which is
    # heavy and not needed in the no-cache informational path.
    from research.governed_fusion.score_cache import event_scores

    events = external_label_events(build_suite(kind="real"))
    return [event_scores(ev) for ev in events]


def _empty_subset_record(reason: str) -> dict[str, Any]:
    return {
        "status": reason,
        "external_label_events": 0,
        "components": list(_COMPONENTS),
        "full": None,
        "leave_one_out": None,
    }


def _load_ledger() -> dict[str, Any]:
    if not os.path.exists(_LEDGER_PATH):
        return {
            "schema_version": 1,
            "ledger": "governed-fusion marginal ablation",
            "components": list(_COMPONENTS),
            "honest_fitness_bucket": "external_label",
            "runs": [],
        }
    with open(_LEDGER_PATH) as fh:
        return json.load(fh)


def _write_ledger(ledger: dict[str, Any]) -> None:
    with open(_LEDGER_PATH, "w") as fh:
        json.dump(ledger, fh, indent=2)
        fh.write("\n")


def _append_run(ledger: dict[str, Any], record: dict[str, Any]) -> None:
    runs = ledger.setdefault("runs", [])
    runs.append(record)


def measure(*, cache_dir: str = _DEFAULT_CACHE_DIR) -> dict[str, Any]:
    """Build one ledger record for the current code/data state.

    Returns the record (a dict).  Does not persist; callers decide whether
    to append to the ledger.
    """
    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "components": list(_COMPONENTS),
        "cache_dir": cache_dir,
    }
    if not _has_cache(cache_dir):
        record.update(_empty_subset_record("needs_cache"))
        record["note"] = (
            f"score cache {cache_dir!r} absent; run measure_baseline first to "
            "fit + cache event scores"
        )
        return record

    events = _load_external_label_scores(cache_dir)
    if not events:
        record.update(_empty_subset_record("no_external_label_events"))
        record["note"] = (
            "the manifest contains no live events with audited ground-truth "
            "labels; the honest fitness substrate is empty"
        )
        return record

    lift = compute_marginal_lift(events)
    record.update(
        {
            "status": "ok",
            "external_label_events": len(events),
            "event_keys": [(es.domain, es.event_id) for es in events],
            "full": lift["full"],
            "leave_one_out": lift["leave_one_out"],
        }
    )
    return record


def main(argv: list[str] | None = None) -> int:
    """Append one ablation-ledger record and print a short summary."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if no external-label events are scorable (cache absent or empty subset).",
    )
    ap.add_argument(
        "--cache-dir",
        default=_DEFAULT_CACHE_DIR,
        help=f"directory holding cached event scores (default: {_DEFAULT_CACHE_DIR!r}).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="compute the record but do not persist it to the ledger.",
    )
    args = ap.parse_args(argv)

    record = measure(cache_dir=args.cache_dir)
    status = record["status"]
    print(f"marginal-ablation ledger: status={status!r}")
    if status == "ok":
        print(f"  external_label events: {record['external_label_events']}")
        full = record["full"]
        print(
            f"  full fusion: AUROC={full['auroc']:.4f} AUPRC={full['auprc']:.4f} F1={full['f1']:.4f}"
        )
        print("  leave-one-out marginal lift (full - ablated):")
        for name, block in record["leave_one_out"].items():
            d = block["delta"]
            a = block["ablated"]
            print(
                f"    {name:<10} dAUROC={d['auroc']:+.4f}  dAUPRC={d['auprc']:+.4f}  "
                f"dF1={d['f1']:+.4f}  (ablated AUROC={a['auroc']:.4f})"
            )
    else:
        print(f"  {record.get('note', '')}")

    if not args.dry_run:
        ledger = _load_ledger()
        _append_run(ledger, record)
        _write_ledger(ledger)
        print(f"  appended record to {_LEDGER_PATH}")

    if args.check and status != "ok":
        return 1
    return 0


__all__ = [
    "compute_marginal_lift",
    "measure",
]


if __name__ == "__main__":
    sys.exit(main())
