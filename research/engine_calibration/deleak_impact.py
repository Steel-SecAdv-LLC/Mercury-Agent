# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measure what excluding label-derived features does to per-event AUROC.

Rebuilds ``research/engine_calibration/deleak_impact.json`` from the de-leak
cache (``{domain}__{event}.deleak.npz`` files holding ``X_full``, the loader's
complete engineered matrix; ``X_scored``, the same rows with
``LABEL_DERIVED_FEATURES`` removed by ``engineer_scored_features``; and ``y``,
the ground truth). For every event the real ``MercuryAnomalyDetector`` is fit
and run on each view and the AUROC of its scores against ``y`` is recorded,
exactly as ``measure.py`` scores the canonical cache.

The file this writes used to be produced by hand, so a change to a loader's
exclusion list (the earthquake ``b_value`` channel, 2026-11-01) left no way to
re-measure the figures the README quotes. This script is that way. Domains
whose events cannot be re-fetched are carried over from the previous artifact
unchanged and listed under ``carried_over`` so the provenance stays explicit.

An AUROC that does not exist is not 0.5. When a view's scores are constant
(the cholera event has no non-zero column left once its case columns are
excluded) or the labels hold one class, that view's AUROC is recorded as
``null`` with ``undefined`` saying why. A domain's two means and its delta are
taken over the events where **both** views have an AUROC, so the delta
compares like with like; ``n_events_scored`` and ``undefined_events`` say
which those are. This is the convention of ``results.json`` (``_summary.py``)
and of ``benchmarks/domain_benchmark_base.py``. Before 2026-11-20 this script
scored such an event as 0.5 and averaged it in (pandemic read 0.7741).

``fetch`` writes an archive for every event it can read and goes on past one
it cannot (an event marked not-benchmarkable, a source that is gone), saying
which and why; it used to stop at the first.

Build the cache for a domain with::

    python research/engine_calibration/deleak_impact.py --fetch earthquake
    python research/engine_calibration/deleak_impact.py --fetch all

and re-measure every cached domain with::

    python research/engine_calibration/deleak_impact.py
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import numpy as np
from _proxy_fallback import install_from_env
from _summary import MEASURED, auroc_status
from _workspace import cache_dir

from omni_mercury_engine.loaders.base import SyntheticDataNotDeleakable
from omni_mercury_engine.ml.mercury_ml import roc_auc_score

HERE = Path(__file__).resolve().parent
ARTIFACT = HERE / "deleak_impact.json"
CACHE = Path(cache_dir())

LOADERS = {
    "earthquake": ("earthquake_loader", "EarthquakeLoader"),
    "tsunami": ("tsunami_loader", "TsunamiLoader"),
    # Energy reads recorded history (OMNI2, EIA-930) since 2026-11-16; while
    # its historical frames were synthetic it could not be measured here.
    "energy": ("energy_loader", "EnergyLoader"),
    "tornado": ("tornado_loader", "TornadoLoader"),
    "marine": ("marine_loader", "MarineLoader"),
    "hurricane": ("hurricane_loader", "HurricaneLoader"),
    "volcanic": ("volcanic_loader", "VolcanicLoader"),
    "fema": ("fema_loader", "FEMALoader"),
    "fema_activity": ("fema_activity_loader", "FEMAActivityLoader"),
    "network_security": ("network_security_loader", "NetworkSecurityLoader"),
    "pandemic": ("pandemic_loader", "PandemicLoader"),
}

#: Domains that declare no exclusion; their drift between two runs bounds
#: what a "de-leak delta" must exceed to mean anything.
CONTROL_DOMAINS = ("marine", "network_security", "tsunami")


def fetch(domain: str) -> None:
    """Fetch every readable event of ``domain`` into the de-leak cache."""
    install_from_env()
    mod, cls = LOADERS[domain]
    loader = getattr(importlib.import_module(f"omni_mercury_engine.loaders.{mod}"), cls)()
    for event in loader.list_events():
        eid = event["event_id"]
        archive = CACHE / f"{domain}__{eid}.deleak.npz"
        if event.get("not_benchmarkable"):
            archive.unlink(missing_ok=True)
            print(f"{domain}/{eid}: not benchmarkable (catalogue)")
            continue
        try:
            _fetch_event(loader, domain, eid, archive)
        except Exception as exc:
            # An archive of an event that can no longer be read is not evidence.
            archive.unlink(missing_ok=True)
            print(f"{domain}/{eid}: not fetched, {type(exc).__name__}: {str(exc)[:200]}")


def _fetch_event(loader: Any, domain: str, eid: str, archive: Path) -> None:
    """Write one event's ``X_full`` / ``X_scored`` / ``y`` archive, if it has both classes."""
    started = time.time()
    raw = loader.fetch_historical(eid)
    # A grid event's full view is the grid matrix: the space-weather
    # matrix is not what its scored columns are a subset of.
    is_grid = getattr(loader, "is_grid_frame", None)
    if is_grid is not None and is_grid(raw):
        x_full = np.asarray(loader.engineer_grid_features(raw), float)
    else:
        x_full = np.asarray(loader.engineer_features(raw), float)
    try:
        x_scored = np.asarray(loader.engineer_scored_features(raw), float)
    except SyntheticDataNotDeleakable as exc:
        # A synthetic frame has no label-disjoint view (every channel is
        # built from the label profile); it is kept out of the measurement
        # rather than written as a "de-leaked" archive.
        archive.unlink(missing_ok=True)
        print(f"{domain}/{eid}: quarantined, synthetic frame ({exc})")
        return
    y = np.asarray(loader.get_ground_truth(eid)).astype(int).reshape(-1)
    m = min(len(x_full), len(y))
    if m == 0 or y[:m].min() == y[:m].max():
        archive.unlink(missing_ok=True)
        print(f"{domain}/{eid}: not written, {'no rows' if m == 0 else 'one class only'} (n={m})")
        return
    np.savez_compressed(archive, X_full=x_full[:m], X_scored=x_scored[:m], y=y[:m])
    print(f"{domain}/{eid}: full{x_full.shape} scored{x_scored.shape} {time.time() - started:.1f}s")


def _auroc(x: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> tuple[float | None, str]:
    """AUROC of the detector's scores on ``x``, or ``None`` and why it does not exist."""
    stat = importlib.import_module("omni_mercury_engine.detectors.statistical")
    detector = stat.MercuryAnomalyDetector()
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    detector.fit(x, unlabeled_reference=True)
    scores = np.asarray(detector.detect(x)["scores"], float).reshape(-1)
    status = auroc_status(y, scores)
    if status != MEASURED:
        return None, status
    return round(float(roc_auc_score(y, scores)), 4), status


def measure() -> dict[str, Any]:
    """Score every cached event on both views and assemble the artifact."""
    previous = json.loads(ARTIFACT.read_text(encoding="utf-8")) if ARTIFACT.exists() else {}
    per_event: list[dict[str, Any]] = []
    measured_domains: set[str] = set()
    for path in sorted(CACHE.glob("*.deleak.npz")):
        domain, event = path.name[: -len(".deleak.npz")].split("__", 1)
        data = np.load(path)
        y = data["y"].astype(int)
        if y.min() == y.max():
            continue
        circular, circular_status = _auroc(data["X_full"], y)
        deleaked, deleaked_status = _auroc(data["X_scored"], y)
        row: dict[str, Any] = {
            "domain": domain,
            "event": event,
            "auroc_circular": circular,
            "auroc_deleaked": deleaked,
            "delta": (
                None if circular is None or deleaked is None else round(deleaked - circular, 4)
            ),
        }
        undefined = {
            view: status
            for view, status in (("circular", circular_status), ("deleaked", deleaked_status))
            if status != MEASURED
        }
        if undefined:
            row["undefined"] = undefined
        per_event.append(row)
        measured_domains.add(domain)

    carried: list[str] = []
    for row in previous.get("per_event", []):
        if row["domain"] not in measured_domains:
            per_event.append(row)
            if row["domain"] not in carried:
                carried.append(row["domain"])

    per_domain: dict[str, Any] = {}
    for domain in sorted({r["domain"] for r in per_event}):
        rows = [r for r in per_event if r["domain"] == domain]
        # Both means over the same events: those with an AUROC in both views.
        scored = [r for r in rows if r.get("delta") is not None]
        circ = float(np.mean([r["auroc_circular"] for r in scored])) if scored else None
        dele = float(np.mean([r["auroc_deleaked"] for r in scored])) if scored else None
        mod, cls = LOADERS[domain]
        excluded = list(
            getattr(
                importlib.import_module(f"omni_mercury_engine.loaders.{mod}"), cls
            ).LABEL_DERIVED_FEATURES
        )
        per_domain[domain] = {
            "n_events": len(rows),
            "n_events_scored": len(scored),
            "undefined_events": [r["event"] for r in rows if r.get("delta") is None],
            "auroc_circular": None if circ is None else round(circ, 4),
            "auroc_deleaked": None if dele is None else round(dele, 4),
            "delta": None if circ is None or dele is None else round(dele - circ, 4),
            "excluded_features": excluded,
        }

    # The band is the largest per-domain mean drift among the domains that
    # exclude nothing: the number the README compares every de-leak delta to.
    control = [
        abs(per_domain[d]["delta"])
        for d in CONTROL_DOMAINS
        if d in per_domain and per_domain[d]["delta"] is not None
    ]
    git = shutil.which("git")
    commit = (
        subprocess.run(  # noqa: S603 - fixed argv, resolved executable, no user input
            [git, "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
        if git
        else ""
    )
    return {
        **{k: previous[k] for k in ("schema", "what", "honest_limits") if k in previous},
        "provenance": {
            "commit": commit,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "script": "research/engine_calibration/deleak_impact.py",
            "measured_domains": sorted(measured_domains),
            "carried_over_domains": carried,
            "baseline_artifact": previous.get("provenance", {}).get("baseline_artifact"),
        },
        "no_exclusion_control_band": {
            "domains": list(CONTROL_DOMAINS),
            "max_abs_delta": round(max(control), 4) if control else None,
        },
        "per_domain": per_domain,
        "per_event": per_event,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--fetch", metavar="DOMAIN", help="fetch one domain (or 'all') into the cache"
    )
    parser.add_argument("--check", action="store_true", help="exit 1 if the artifact is stale")
    args = parser.parse_args(argv)
    if args.fetch:
        domains = list(LOADERS) if args.fetch == "all" else [args.fetch]
        for domain in domains:
            try:
                fetch(domain)
            except Exception as exc:
                if args.fetch != "all":
                    raise
                print(f"{domain}: loader not usable, {type(exc).__name__}: {str(exc)[:200]}")
        return 0
    artifact = measure()
    text = json.dumps(artifact, indent=2, allow_nan=False) + "\n"
    if args.check:
        current = ARTIFACT.read_text(encoding="utf-8") if ARTIFACT.exists() else ""
        same = json.loads(current)["per_domain"] == artifact["per_domain"] if current else False
        print(
            "deleak artifact is current" if same else "STALE: deleak_impact.json per_domain differs"
        )
        return 0 if same else 1
    ARTIFACT.write_text(text, encoding="utf-8")
    for domain, row in artifact["per_domain"].items():
        if row["delta"] is None:
            print(f"{domain:18s} n={row['n_events']} no event has an AUROC in both views")
            continue
        print(
            f"{domain:18s} n={row['n_events_scored']} of {row['n_events']} "
            f"circular={row['auroc_circular']:.4f} deleaked={row['auroc_deleaked']:.4f} "
            f"delta={row['delta']:+.4f}"
            + (
                f"  (no AUROC: {', '.join(row['undefined_events'])})"
                if row["undefined_events"]
                else ""
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
