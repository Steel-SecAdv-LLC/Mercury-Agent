# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fetch real (features, ground_truth) per event ONCE and cache to disk.

Mirrors the honest-benchmark path: real loaders -> engineer_features ->
get_ground_truth. Lets us iterate the detector offline (the live USGS/NOAA
fetch is the slow/flaky part; the detector itself is fast).
"""

from __future__ import annotations
import warnings, json, importlib, os, time
warnings.filterwarnings("ignore")
import numpy as np

CACHE = "/home/user/eqlab/cache"
os.makedirs(CACHE, exist_ok=True)

LOADERS = {
    "earthquake": ("earthquake_loader", "EarthquakeLoader"),
    "tsunami":    ("tsunami_loader", "TsunamiLoader"),
    "tornado":    ("tornado_loader", "TornadoLoader"),
    "marine":     ("marine_loader", "MarineLoader"),
    "hurricane":  ("hurricane_loader", "HurricaneLoader"),
    "wildfire":   ("wildfire_loader", "WildfireLoader"),
    "flood":      ("flood_loader", "FloodLoader"),
    "volcanic":   ("volcanic_loader", "VolcanicLoader"),
    "landslide":  ("landslide_loader", "LandslideLoader"),
    "energy":     ("energy_loader", "EnergyLoader"),
    "fema":       ("fema_loader", "FEMALoader"),
    "financial":  ("financial_loader", "FinancialLoader"),
    "network_security": ("network_security_loader", "NetworkSecurityLoader"),
    "pandemic":   ("pandemic_loader", "PandemicLoader"),
    "sepsis":     ("sepsis_loader", "SepsisLoader"),
}

manifest = {}
for dom, (mod, cls) in LOADERS.items():
    try:
        L = getattr(importlib.import_module(f"omni_mercury_engine.loaders.{mod}"), cls)()
        events = L.list_events()
    except Exception as e:
        print(f"{dom:18s} LOADER_ERR {type(e).__name__}: {str(e)[:80]}", flush=True)
        manifest[dom] = {"status": "loader_err", "err": str(e)[:120]}
        continue
    recs = []
    for ev in events:
        eid = ev["event_id"] if isinstance(ev, dict) else str(ev)
        t = time.time()
        try:
            raw = L.fetch_historical(eid)
            feats = np.asarray(L.engineer_features(raw), float)
            y = np.asarray(L.get_ground_truth(eid)).astype(int).reshape(-1)
            m = min(len(feats), len(y)); feats = feats[:m]; y = y[:m]
            if len(feats) == 0 or y.min() == y.max():
                print(f"  {dom}/{eid} skip(novar) n={len(feats)}", flush=True); continue
            fn = f"{CACHE}/{dom}__{eid}.npz"
            np.savez_compressed(fn, X=feats, y=y)
            recs.append({"event": eid, "n": int(m), "pos": int(y.sum()),
                         "feat_dim": int(feats.shape[1] if feats.ndim > 1 else 1),
                         "dt": round(time.time()-t, 1)})
            print(f"  {dom}/{eid} n={m} pos={int(y.sum())} dim={feats.shape[1] if feats.ndim>1 else 1} ({time.time()-t:.1f}s)", flush=True)
        except Exception as e:
            print(f"  {dom}/{eid} FETCH_ERR {type(e).__name__}: {str(e)[:70]}", flush=True)
    manifest[dom] = {"status": "ok" if recs else "unreachable", "events": recs}
    print(f"{dom:18s} -> {len(recs)} events cached", flush=True)

json.dump(manifest, open(f"{CACHE}/manifest.json", "w"), indent=2)
reach = [d for d, v in manifest.items() if v.get("status") == "ok"]
unreach = [d for d, v in manifest.items() if v.get("status") != "ok"]
print("\n==== CACHE SUMMARY ====")
print("reachable :", reach)
print("unreachable:", unreach)
print("total events:", sum(len(v.get("events", [])) for v in manifest.values()))
