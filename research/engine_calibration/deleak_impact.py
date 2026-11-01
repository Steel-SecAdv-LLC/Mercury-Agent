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

Build the cache for a domain with::

    python research/engine_calibration/deleak_impact.py --fetch earthquake

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
from _workspace import cache_dir

from omni_mercury_engine.ml.mercury_ml import roc_auc_score

HERE = Path(__file__).resolve().parent
ARTIFACT = HERE / "deleak_impact.json"
CACHE = Path(cache_dir())

LOADERS = {
    "earthquake": ("earthquake_loader", "EarthquakeLoader"),
    "tsunami": ("tsunami_loader", "TsunamiLoader"),
    "tornado": ("tornado_loader", "TornadoLoader"),
    "marine": ("marine_loader", "MarineLoader"),
    "hurricane": ("hurricane_loader", "HurricaneLoader"),
    "volcanic": ("volcanic_loader", "VolcanicLoader"),
    "fema": ("fema_loader", "FEMALoader"),
    "network_security": ("network_security_loader", "NetworkSecurityLoader"),
    "pandemic": ("pandemic_loader", "PandemicLoader"),
}

#: Domains that declare no exclusion; their drift between two runs bounds
#: what a "de-leak delta" must exceed to mean anything.
CONTROL_DOMAINS = ("marine", "network_security", "tsunami")


def fetch(domain: str) -> None:
    """Fetch every event of ``domain`` into the de-leak cache."""
    mod, cls = LOADERS[domain]
    loader = getattr(importlib.import_module(f"omni_mercury_engine.loaders.{mod}"), cls)()
    for event in loader.list_events():
        eid = event["event_id"]
        started = time.time()
        raw = loader.fetch_historical(eid)
        x_full = np.asarray(loader.engineer_features(raw), float)
        x_scored = np.asarray(loader.engineer_scored_features(raw), float)
        y = np.asarray(loader.get_ground_truth(eid)).astype(int).reshape(-1)
        m = min(len(x_full), len(y))
        np.savez_compressed(
            CACHE / f"{domain}__{eid}.deleak.npz",
            X_full=x_full[:m],
            X_scored=x_scored[:m],
            y=y[:m],
        )
        print(
            f"{domain}/{eid}: full{x_full.shape} scored{x_scored.shape} {time.time() - started:.1f}s"
        )


def _auroc(x: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> float:
    stat = importlib.import_module("omni_mercury_engine.detectors.statistical")
    detector = stat.MercuryAnomalyDetector()
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    detector.fit(x)
    scores = np.asarray(detector.detect(x)["scores"], float).reshape(-1)
    return float(roc_auc_score(y, scores))


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
        circular = _auroc(data["X_full"], y)
        deleaked = _auroc(data["X_scored"], y)
        per_event.append(
            {
                "domain": domain,
                "event": event,
                "auroc_circular": round(circular, 4),
                "auroc_deleaked": round(deleaked, 4),
                "delta": round(deleaked - circular, 4),
            }
        )
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
        circ = float(np.mean([r["auroc_circular"] for r in rows]))
        dele = float(np.mean([r["auroc_deleaked"] for r in rows]))
        mod, cls = LOADERS[domain]
        excluded = list(
            getattr(
                importlib.import_module(f"omni_mercury_engine.loaders.{mod}"), cls
            ).LABEL_DERIVED_FEATURES
        )
        per_domain[domain] = {
            "n_events": len(rows),
            "auroc_circular": round(circ, 4),
            "auroc_deleaked": round(dele, 4),
            "delta": round(dele - circ, 4),
            "excluded_features": excluded,
        }

    # The band is the largest per-domain mean drift among the domains that
    # exclude nothing: the number the README compares every de-leak delta to.
    control = [abs(per_domain[d]["delta"]) for d in CONTROL_DOMAINS if d in per_domain]
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
    parser.add_argument("--fetch", metavar="DOMAIN", help="fetch one domain into the cache")
    parser.add_argument("--check", action="store_true", help="exit 1 if the artifact is stale")
    args = parser.parse_args(argv)
    if args.fetch:
        fetch(args.fetch)
        return 0
    artifact = measure()
    text = json.dumps(artifact, indent=2) + "\n"
    if args.check:
        current = ARTIFACT.read_text(encoding="utf-8") if ARTIFACT.exists() else ""
        same = json.loads(current)["per_domain"] == artifact["per_domain"] if current else False
        print(
            "deleak artifact is current" if same else "STALE: deleak_impact.json per_domain differs"
        )
        return 0 if same else 1
    ARTIFACT.write_text(text, encoding="utf-8")
    for domain, row in artifact["per_domain"].items():
        print(
            f"{domain:18s} n={row['n_events']} circular={row['auroc_circular']:.4f} deleaked={row['auroc_deleaked']:.4f} delta={row['delta']:+.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
