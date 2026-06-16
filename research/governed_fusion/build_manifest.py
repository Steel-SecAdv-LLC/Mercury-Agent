# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
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

Each entry additionally carries the audited ``label_provenance`` (the
producing loader's ``LABEL_SOURCE`` -- ``ground_truth | expert_annotated |
statistical | none``) and ``series_provenance`` (``live | reconstructed``)
so the autonomous fitness loop reads only independently labelled live events.
The single source of truth for these fields is
:mod:`research.governed_fusion.label_provenance`, which pivots from the
loader-side audit (:mod:`omni_mercury_engine.loaders.label_provenance`).
Top-level ``provenance_summary`` enumerates the bucket counts so a
reviewer can confirm at a glance which subset feeds the fitness signal
(today: the two ``network_security`` events).

Writes ``research/governed_fusion/manifest.json``.

Run::

    source research/governed_fusion/gf_env.sh
    python research/governed_fusion/build_manifest.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.governed_fusion.label_provenance import (
    event_is_external_label,
    label_provenance,
    series_provenance,
)
from research.governed_fusion.measure_baseline import CAP
from research.governed_fusion.suite import EventData, build_suite, stratified_subsample

_GF_DIR = Path(__file__).resolve().parent


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
    """One manifest entry: shape, class balance, fingerprints, provenance."""
    xc, yc = stratified_subsample(ev.X, ev.y, CAP, seed=42)
    return {
        "domain": ev.domain,
        "event_id": ev.event_id,
        "n_rows": int(ev.X.shape[0]),
        "n_features": int(ev.X.shape[1]),
        "n_pos": int(ev.n_pos),
        "sha256_xy": _sha256_xy(ev.X, ev.y),
        # Audited provenance, sourced from the loader registry.  ``external``
        # is the single trust bit the autonomous fitness signal reads.
        "label_provenance": label_provenance(ev.domain),
        "series_provenance": series_provenance(ev.domain, ev.event_id),
        "external_label": bool(event_is_external_label(ev.domain, ev.event_id)),
        "capped": {
            "cap": CAP,
            "seed": 42,
            "n_rows": int(xc.shape[0]),
            "n_pos": int(np.sum(yc == 1)),
            "sha256_xy": _sha256_xy(xc, yc),
        },
    }


def _provenance_summary(real: list[dict[str, Any]], recon: list[dict[str, Any]]) -> dict[str, Any]:
    """Bucket counts for the live and reconstructed event classes."""

    def _bucket(entries: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {
            "external_label": {"n_events": 0, "n_rows": 0, "n_pos": 0},
            "self_label": {"n_events": 0, "n_rows": 0, "n_pos": 0},
            "reconstructed": {"n_events": 0, "n_rows": 0, "n_pos": 0},
        }
        for e in entries:
            if e["series_provenance"] == "reconstructed":
                bucket = "reconstructed"
            elif e["external_label"]:
                bucket = "external_label"
            else:
                bucket = "self_label"
            out[bucket]["n_events"] += 1
            out[bucket]["n_rows"] += int(e["n_rows"])
            out[bucket]["n_pos"] += int(e["n_pos"])
        return out

    return {
        "real": _bucket(real),
        "reconstructed": _bucket(recon),
        "transparent_fitness_bucket": "external_label",
        "transparent_fitness_note": (
            "The governed promotion gate / autonomous fitness signal reads only "
            "live events with label_provenance in {ground_truth, expert_annotated}. "
            "Self-labelled events are reported separately as leakage-flagged; "
            "reconstructed-series events are reported separately by design."
        ),
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
        "provenance_summary": _provenance_summary(real, recon),
        "real": real,
        "reconstructed": recon,
    }
    out_path = _GF_DIR / "manifest.json"
    with out_path.open("w") as fh:
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
