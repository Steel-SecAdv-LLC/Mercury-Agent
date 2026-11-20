# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fetch real (features, ground_truth) per event ONCE and cache to disk.

Mirrors the transparent-benchmark path: real loaders -> engineer_scored_features
-> get_ground_truth. Lets us iterate the detector offline (the live USGS/NOAA
fetch is the slow/flaky part; the detector itself is fast).

First step of the hazard table (``score_events.py`` and ``measure_cached.py``
follow; ``score_events.py`` gives the whole sequence). ``manifest.json`` says
what became of every catalogue event: cached, one class only, no rows, a
synthetic frame refused, or the error its fetch raised.

``MERCURY_CALIBRATION_DOMAINS=hurricane,flood`` rebuilds only those domains and
leaves the rest of the manifest as it was. ``_proxy_fallback.py`` describes the
opt-in for a host that reaches the sources only through ``HTTPS_PROXY``.
"""

from __future__ import annotations

import importlib
import json
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
from _proxy_fallback import install_from_env
from _workspace import ARCHIVE_SCHEMA, cache_dir, clear_quarantine, quarantine_event

from omni_mercury_engine.loaders.base import SyntheticDataNotDeleakable

CACHE = cache_dir()
install_from_env()

LOADERS = {
    "earthquake": ("earthquake_loader", "EarthquakeLoader"),
    "tsunami": ("tsunami_loader", "TsunamiLoader"),
    "tornado": ("tornado_loader", "TornadoLoader"),
    "marine": ("marine_loader", "MarineLoader"),
    "hurricane": ("hurricane_loader", "HurricaneLoader"),
    "wildfire": ("wildfire_loader", "WildfireLoader"),
    "flood": ("flood_loader", "FloodLoader"),
    "volcanic": ("volcanic_loader", "VolcanicLoader"),
    "landslide": ("landslide_loader", "LandslideLoader"),
    "energy": ("energy_loader", "EnergyLoader"),
    "fema": ("fema_loader", "FEMALoader"),
    # The README's FEMA declaration activity row. It was measured only by
    # remeasure_recorded.py, so a table built from this cache alone lacked it.
    "fema_activity": ("fema_activity_loader", "FEMAActivityLoader"),
    "financial": ("financial_loader", "FinancialLoader"),
    "network_security": ("network_security_loader", "NetworkSecurityLoader"),
    "pandemic": ("pandemic_loader", "PandemicLoader"),
    "sepsis": ("sepsis_loader", "SepsisLoader"),
}

_ONLY = [
    d.strip() for d in os.environ.get("MERCURY_CALIBRATION_DOMAINS", "").split(",") if d.strip()
]
_unknown = [d for d in _ONLY if d not in LOADERS]
if _unknown:
    raise SystemExit(f"MERCURY_CALIBRATION_DOMAINS names no loader: {_unknown}")
_MANIFEST = Path(CACHE) / "manifest.json"
# A partial rebuild keeps what the manifest says of the other domains.
manifest = json.loads(_MANIFEST.read_text(encoding="utf-8")) if _ONLY and _MANIFEST.exists() else {}


def _discard_archive(dom: str, eid: str) -> None:
    """Remove an older archive of an event this build could not cache.

    An event that now yields no rows, or one class, must not go on being
    measured from an archive an earlier build left in the same directory.
    """
    archive = Path(CACHE) / f"{dom}__{eid}.npz"
    if archive.exists():
        archive.unlink()


for dom, (mod, cls) in LOADERS.items():
    if _ONLY and dom not in _ONLY:
        continue
    try:
        L = getattr(importlib.import_module(f"omni_mercury_engine.loaders.{mod}"), cls)()
        events = L.list_events()
    except Exception as e:
        print(f"{dom:18s} LOADER_ERR {type(e).__name__}: {str(e)[:80]}", flush=True)
        manifest[dom] = {"status": "loader_err", "err": str(e)[:120]}
        continue
    recs = []
    skipped = []
    for ev in events:
        eid = ev["event_id"] if isinstance(ev, dict) else str(ev)
        t = time.time()
        if isinstance(ev, dict) and ev.get("not_benchmarkable"):
            # The catalogue says the event has no recorded source; it is not fetched.
            _discard_archive(dom, eid)
            skipped.append({"event": eid, "status": "not benchmarkable", "why": ev.get("reason")})
            print(f"  {dom}/{eid} not benchmarkable (catalogue)", flush=True)
            continue
        try:
            raw = L.fetch_historical(eid)
            # De-leaked view -- see BaseDomainLoader.engineer_scored_features.
            # This cache is what research/engine_calibration/results.json is
            # measured from, and that file backs the README's per-domain
            # hazard table, so the exclusion has to apply here or the table
            # keeps reporting circular numbers.
            try:
                feats = np.asarray(L.engineer_scored_features(raw), float)
            except SyntheticDataNotDeleakable as exc:
                # A rejected rebuild must not leave an older archive of this
                # event selectable: the archive is removed and a marker left,
                # which cache_archives() also honours.
                quarantine_event(dom, eid, str(exc))
                skipped.append({"event": eid, "status": "quarantined: synthetic frame"})
                print(f"  {dom}/{eid} quarantined: synthetic frame", flush=True)
                continue
            y = np.asarray(L.get_ground_truth(eid)).astype(int).reshape(-1)
            m = min(len(feats), len(y))
            feats = feats[:m]
            y = y[:m]
            if len(feats) == 0 or y.min() == y.max():
                _discard_archive(dom, eid)
                status = "no rows" if len(feats) == 0 else "one class only"
                skipped.append({"event": eid, "status": status, "n": int(m)})
                print(f"  {dom}/{eid} skip(novar) n={len(feats)}", flush=True)
                continue
            fn = f"{CACHE}/{dom}__{eid}.npz"
            # The archive states what X holds (the de-leaked columns) and what
            # was excluded; cache_archives() refuses archives without the
            # schema, which is how a pre-exclusion archive stays out.
            np.savez_compressed(
                fn,
                X=feats,
                y=y,
                schema=np.array(ARCHIVE_SCHEMA),
                scored_columns=np.array(list(L.scored_feature_columns()), dtype=str),
                label_derived_excluded=np.array(list(L.LABEL_DERIVED_FEATURES), dtype=str),
            )
            # A validated archive is on disk: an earlier quarantine (a
            # synthetic frame that has since been replaced by real data) no
            # longer applies, so the marker goes and the event is evaluated again.
            if clear_quarantine(dom, eid):
                print(f"  {dom}/{eid} recovered: quarantine lifted", flush=True)
            recs.append(
                {
                    "event": eid,
                    "n": int(m),
                    "pos": int(y.sum()),
                    "feat_dim": int(feats.shape[1] if feats.ndim > 1 else 1),
                    "dt": round(time.time() - t, 1),
                }
            )
            print(
                f"  {dom}/{eid} n={m} pos={int(y.sum())} dim={feats.shape[1] if feats.ndim > 1 else 1} ({time.time()-t:.1f}s)",
                flush=True,
            )
        except Exception as e:
            _discard_archive(dom, eid)
            skipped.append(
                {
                    "event": eid,
                    "status": "fetch error",
                    "why": f"{type(e).__name__}: {str(e)[:300]}",
                }
            )
            print(f"  {dom}/{eid} FETCH_ERR {type(e).__name__}: {str(e)[:70]}", flush=True)
    manifest[dom] = {
        "status": "ok" if recs else "unreachable",
        "events": recs,
        "not_cached": skipped,
    }
    print(f"{dom:18s} -> {len(recs)} events cached", flush=True)

_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
reach = [d for d, v in manifest.items() if v.get("status") == "ok"]
unreach = [d for d, v in manifest.items() if v.get("status") != "ok"]
print("\n==== CACHE SUMMARY ====")
print("reachable :", reach)
print("unreachable:", unreach)
print("total events:", sum(len(v.get("events", [])) for v in manifest.values()))
