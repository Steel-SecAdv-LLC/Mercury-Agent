"""Data manifest for the governed-fusion reachable suite.

Fingerprints every event's cached ``(X, y)`` so the suite is auditable without
live API access: a reviewer can confirm exactly which data each FINDINGS figure
was computed on (shape, class balance, SHA-256) and that the seeded stratified
cap used for iteration is reproducible.  The full-event hash anchors the source
data; the ``capped`` hash anchors the actual measurement input (``cap=6000``,
``seed=42`` -- a no-op for every event except the few large ones).

The manifest is split into the **live headline suite** (23 real events / 7
domains) and the **reconstructed-from-live group** (7 events / 3 domains --
tsunami, energy, ebola_2014), so provenance is never ambiguous.

Writes ``research/governed_fusion/manifest.json``.

Run::

    source research/governed_fusion/gf_env.sh
    python research/governed_fusion/build_manifest.py
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import numpy as np

from research.governed_fusion.measure_baseline import CAP
from research.governed_fusion.suite import EventData, build_suite, stratified_subsample

_GF_DIR = os.path.dirname(os.path.abspath(__file__))


def _sha256_xy(X: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> str:
    """Deterministic, shape-tagged SHA-256 over canonical ``(X, y)`` bytes."""
    h = hashlib.sha256()
    xc = np.ascontiguousarray(X, dtype=np.float64)
    yc = np.ascontiguousarray(y, dtype=np.int64).reshape(-1)
    h.update(b"X" + str(xc.shape).encode())
    h.update(xc.tobytes())
    h.update(b"y" + str(yc.shape).encode())
    h.update(yc.tobytes())
    return h.hexdigest()


def _entry(ev: EventData) -> dict[str, Any]:
    """One manifest entry: shape, class balance and full + capped fingerprints."""
    xc, yc = stratified_subsample(ev.X, ev.y, CAP, seed=42)
    return {
        "domain": ev.domain,
        "event_id": ev.event_id,
        "n_rows": int(ev.X.shape[0]),
        "n_features": int(ev.X.shape[1]),
        "n_pos": int(ev.n_pos),
        "sha256_xy": _sha256_xy(ev.X, ev.y),
        "capped": {
            "cap": CAP,
            "seed": 42,
            "n_rows": int(xc.shape[0]),
            "n_pos": int(np.sum(yc == 1)),
            "sha256_xy": _sha256_xy(xc, yc),
        },
    }


def _entries(kind: str) -> list[dict[str, Any]]:
    """Sorted manifest entries for one event class."""
    events = build_suite(kind=kind)
    return [_entry(ev) for ev in sorted(events, key=lambda e: (e.domain, e.event_id))]


def main() -> None:
    """Build and write the live + reconstructed data manifest."""
    real = _entries("real")
    recon = _entries("reconstructed")

    manifest = {
        "suite": "governed-fusion reachable suite",
        "n_real_events": len(real),
        "n_real_domains": len({e["domain"] for e in real}),
        "n_reconstructed_events": len(recon),
        "n_reconstructed_domains": len({e["domain"] for e in recon}),
        "reconstructed_note": (
            "tsunami/energy synthesise the series by design and ebola_2014 has no "
            "live WHO feed; these reconstruct documented real events and are "
            "reported separately, never folded into the live headline mean"
        ),
        "cap": CAP,
        "real": real,
        "reconstructed": recon,
    }
    out_path = os.path.join(_GF_DIR, "manifest.json")
    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(
        f"wrote {out_path}: {len(real)} live events / "
        f"{manifest['n_real_domains']} domains + {len(recon)} reconstructed"
    )
    for label, entries in (("LIVE", real), ("RECON", recon)):
        for e in entries:
            cap_note = (
                f" -> capped {e['capped']['n_rows']}"
                if e["capped"]["n_rows"] != e["n_rows"]
                else ""
            )
            print(
                f"  [{label}] {e['domain']:<18}{e['event_id']:<24} rows={e['n_rows']:>6} "
                f"pos={e['n_pos']:>5} sha={e['sha256_xy'][:12]}{cap_note}"
            )


if __name__ == "__main__":
    main()
