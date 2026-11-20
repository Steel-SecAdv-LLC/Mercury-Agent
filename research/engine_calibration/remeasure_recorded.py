# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Re-measure the domains whose loaders read a recorded archive directly.

``results.json`` backs the README's per-domain hazard table. Its rows come
from a cache of fetched events (``build_cache.py``, ``score_events.py``, then
``measure_cached.py``). The tsunami and energy loaders changed from
constructed series to recorded ones (NDBC DART, NASA OMNI2, EIA-930, DOE
OE-417), so their rows are measured here from the loaders themselves, with
synthetic stand-ins refused, and spliced into the artifact. FEMA is measured
the same way since its label rule was corrected (individual assistance is
recorded as ``ihProgramDeclared`` on declarations since about 2007). Every
other row is left exactly as committed.

Since ``build_cache.py`` caches ``fema_activity`` as well, this step is a
cross-check and no longer the only source of any row: on the same archives it
writes the same figures the three-step sequence does, from a cache directory
of its own that starts empty.

Usage::

    python research/engine_calibration/remeasure_recorded.py          # rewrite
    python research/engine_calibration/remeasure_recorded.py --dry-run

Needs network access to the archives and to OpenFEMA. Exit code 1 if any event cannot be
read as recorded data; nothing is written in that case.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "results.json"
sys.path.insert(0, str(_HERE))
from _proxy_fallback import install_from_env
from _summary import MEASURED, auroc_status, summarise, write_json

#: Domain -> (loader module, loader class).
RECORDED = {
    "energy": ("energy_loader", "EnergyLoader"),
    "fema": ("fema_loader", "FEMALoader"),
    "fema_activity": ("fema_activity_loader", "FEMAActivityLoader"),
    "tsunami": ("tsunami_loader", "TsunamiLoader"),
}


def _measure_event(loader: Any, domain: str, event_id: str) -> dict[str, Any] | None:
    """Score one event with the shipped detector, exactly as ``measure.py`` does.

    Returns ``None`` for an event whose labels hold a single class: no ranking
    metric exists for it.
    """
    measure = importlib.import_module("measure")
    statistical = importlib.import_module("omni_mercury_engine.detectors.statistical")

    raw = loader.fetch_historical(event_id)
    if loader.is_synthetic(raw):
        raise RuntimeError(f"{domain}/{event_id}: the loader returned a synthetic stand-in")
    features = np.asarray(loader.engineer_scored_features(raw), dtype=float)
    labels = np.asarray(loader.get_ground_truth(event_id)).astype(int).reshape(-1)
    if labels.size == 0 or labels.min() == labels.max():
        return None
    detector = statistical.MercuryAnomalyDetector()
    detector.fit(features, unlabeled_reference=True)  # the event is its own reference
    result = detector.detect(features)
    scores = np.asarray(result["scores"], dtype=float).reshape(-1)
    after = np.asarray(result["is_anomaly"]).astype(int).reshape(-1)
    before = (scores > 0.5).astype(int)
    f1_b, prec_b, rec_b = measure.prf(labels, before)
    f1_a, prec_a, rec_a = measure.prf(labels, after)
    status = auroc_status(labels, scores)
    return {
        "dom": domain,
        "event": event_id,
        "n": int(labels.size),
        "n_total": int(labels.size),
        "pos": int(labels.sum()),
        "subsampled": False,
        "auroc": measure.auc(labels, scores) if status == MEASURED else None,
        "auroc_status": status,
        "auprc": measure.ap(labels, scores),
        "f1_before": f1_b,
        "prec_before": prec_b,
        "rec_before": rec_b,
        "f1_after": f1_a,
        "prec_after": prec_a,
        "rec_after": rec_a,
    }


def remeasure() -> dict[str, Any]:
    """Return ``results.json`` with the recorded domains measured afresh."""
    os.environ["MERCURY_ALLOW_SYNTHETIC"] = "0"
    install_from_env()
    artifact: dict[str, Any] = json.loads(_RESULTS.read_text(encoding="utf-8"))
    rows = [row for row in artifact["per_event"] if row["dom"] not in RECORDED]
    for domain, (module, cls) in RECORDED.items():
        loader_cls = getattr(importlib.import_module(f"omni_mercury_engine.loaders.{module}"), cls)
        with tempfile.TemporaryDirectory() as scratch:
            loader = loader_cls(cache_dir=Path(scratch))
            for event in loader.list_events():
                row = _measure_event(loader, domain, event["event_id"])
                if row is None:
                    print(f"{domain:8s} {event['event_id']:20s} not scorable: one class only")
                    continue
                shown = row["auroc_status"] if row["auroc"] is None else f"{row['auroc']:.4f}"
                print(
                    f"{domain:8s} {row['event']:20s} n={row['n']:5d} pos={row['pos']:4d} "
                    f"auroc={shown}"
                )
                rows.append(row)
    return summarise(rows)


def main(argv: list[str] | None = None) -> int:
    """Re-measure and, unless ``--dry-run``, rewrite ``results.json``."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--dry-run", action="store_true", help="measure and print; write nothing")
    args = parser.parse_args(argv)
    try:
        artifact = remeasure()
    except Exception as exc:
        print(f"not measured: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for domain in RECORDED:
        print(f"{domain}: AUROC {artifact['per_domain'][domain]['auroc']}")
    print(f"overall: AUROC {artifact['overall']['auroc']} over {artifact['events']} events")
    if not args.dry_run:
        write_json(_RESULTS, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
