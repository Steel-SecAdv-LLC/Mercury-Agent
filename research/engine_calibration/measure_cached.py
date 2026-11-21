# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Canonical before/after measurement using cached scores + the REAL detector.

Scores are rank-preserving under the calibration change, so the cached per-event
scores (scores.pkl, one detector pass) are reused and the detector's actual
_adaptive_operating_point() is applied to them. This reproduces the transparent-
benchmark operating-point metrics exactly (verified against the live scripts)
without re-running detect() on the 148k-row NSL-KDD event.

  BEFORE = fixed 0.5 cut   AFTER = MercuryAnomalyDetector._adaptive_operating_point
AUROC/AUPRC are rank metrics (unchanged by the cut). Writes results.json.

``scores.pkl`` is written by ``score_events.py`` (its docstring gives the whole
sequence from an empty workspace). An event whose AUROC does not exist (a
constant score, one class) is written with ``"auroc": null`` and an
``auroc_status`` saying why, and is left out of the means (``_summary.py``).
"""

from __future__ import annotations

import importlib
import json
import os
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
from _summary import MEASURED, auroc_status, summarise, write_json
from _workspace import cache_dir, workspace_file

from omni_mercury_engine.ml.mercury_ml import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

stat = importlib.import_module("omni_mercury_engine.detectors.statistical")
DET = stat.MercuryAnomalyDetector()
S = pickle.load(open(workspace_file("scores.pkl"), "rb"))
doms = sorted({d for d, _ in S})


def _prf(y, pred):
    return (
        f1_score(y, pred, zero_division=0),
        precision_score(y, pred, zero_division=0),
        recall_score(y, pred, zero_division=0),
    )


rows = []
for (dom, eid), r in S.items():
    s = np.asarray(r["scores"], float)
    y = np.asarray(r["y"], int)
    thr, _ = DET._adaptive_operating_point(s)  # real detector logic
    status = auroc_status(y, s)
    bf = _prf(y, (s > 0.5).astype(int))
    af = _prf(y, (s > thr).astype(int))
    rows.append(
        {
            "dom": dom,
            "event": eid,
            "n": int(y.size),
            "n_total": int(r.get("rows", y.size)),
            "pos": int(y.sum()),
            "subsampled": bool(r.get("subsampled", False)),
            "auroc": float(roc_auc_score(y, s)) if status == MEASURED else None,
            "auroc_status": status,
            "auprc": float(average_precision_score(y, s)) if y.sum() else None,
            "f1_before": bf[0],
            "prec_before": bf[1],
            "rec_before": bf[2],
            "f1_after": af[0],
            "prec_after": af[1],
            "rec_after": af[2],
        }
    )


SUMMARY = summarise(rows)


def _cell(value):
    return "  n/a" if value is None else f"{value:5.3f}"


def _not_measured():
    """Domains the cache build fetched nothing for, from its manifest."""
    manifest = Path(cache_dir()) / "manifest.json"
    if not manifest.exists():
        return []
    held = json.loads(manifest.read_text(encoding="utf-8"))
    return sorted(dom for dom in held if dom not in SUMMARY["per_domain"])


hdr = (
    f"{'domain':16s} {'nev':>3s} {'AUROC':>6s} {'AUPRC':>6s} | "
    f"{'F1.b':>5s} {'F1.a':>5s} | {'pr.b':>5s} {'pr.a':>5s} | {'rc.b':>5s} {'rc.a':>5s}"
)
print(
    f"events={SUMMARY['events']} ({SUMMARY['events_auroc_defined']} with an AUROC)  "
    f"domains={len(doms)}  (no event cached: {', '.join(_not_measured()) or 'none'})"
)
print(hdr)
for d in doms:
    row = SUMMARY["per_domain"][d]
    print(
        f"{d:16s} {row['n_events']:3d} {_cell(row['auroc']):>6s} {_cell(row['auprc']):>6s} | "
        f"{_cell(row['f1_before'])} {_cell(row['f1_after'])} | "
        f"{_cell(row['prec_before'])} {_cell(row['prec_after'])} | "
        f"{_cell(row['rec_before'])} {_cell(row['rec_after'])}"
    )
print("-" * len(hdr))
overall = SUMMARY["overall"]
print(
    f"{'OVERALL':16s} {SUMMARY['events']:3d} {_cell(overall['auroc']):>6s} "
    f"{_cell(overall['auprc']):>6s} | "
    f"{_cell(overall['f1_before'])} {_cell(overall['f1_after'])} | "
    f"{_cell(overall['prec_before'])} {_cell(overall['prec_after'])} | "
    f"{_cell(overall['rec_before'])} {_cell(overall['rec_after'])}"
)
for undefined in SUMMARY["auroc_undefined"]:
    print(f"no AUROC: {undefined['dom']}/{undefined['event']} ({undefined['reason']})")

out = os.path.join(os.path.dirname(__file__), "results.json")
write_json(out, SUMMARY)
print(f"\nwrote {out}")
