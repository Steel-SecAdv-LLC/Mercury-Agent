# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-event label-provenance lookup for the governed-fusion suite.

Phase 1 of governed recursive self-improvement: pivot from the per-loader
provenance registry (``omni_mercury_engine.loaders.label_provenance``) to a
per-event view the suite, manifest, and ablation ledger can read.

The suite's manifest enumerates (domain, event_id) pairs.  This module maps
each pair to the audited ``LABEL_SOURCE`` of the loader that produced it,
plus an :func:`external_label_events` helper the autonomous fitness loop will
use to filter out manufactured / reconstructed events before any promotion
decision is graded.

Two orthogonal axes of "trust" exist for an event:

* **label_provenance** -- where do labels come from?
  ``ground_truth | expert_annotated | statistical | none``.  See
  ``omni_mercury_engine.datasets.metadata.VALID_LABEL_SOURCES``.
* **series_provenance** -- is the row data live or reconstructed-from-stats?
  ``live | reconstructed``.  Tracked by ``suite.is_reconstructed``.

An event is **eligible for the honest fitness signal** iff both axes are
trustworthy: ``label_provenance in GENUINE_LABEL_SOURCES`` *and*
``series_provenance == "live"``.  Today that intersection is exactly the
two ``network_security`` events (``batadal``, ``nsl_kdd``); ``sepsis`` is
the third genuine loader but it has no events in the governed-fusion
manifest yet.  Phase 2's promotion gate reads only this subset.

This module is read-only and offline (no network, no fits).  It is the
single source of truth that ``build_manifest.py`` (writer of
``manifest.json``), the suite's ``external_label_events`` helper, and the
ablation ledger all use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from omni_mercury_engine.datasets.metadata import GENUINE_LABEL_SOURCES
from omni_mercury_engine.loaders.label_provenance import (
    LABEL_PROVENANCE_REGISTRY,
)
from research.governed_fusion.suite import REACHABLE, is_reconstructed

if TYPE_CHECKING:
    from research.governed_fusion.suite import EventData


def _loader_registry_key(domain: str) -> str:
    """Map a manifest ``domain`` to its loader's registry key."""
    if domain not in REACHABLE:
        raise KeyError(f"unknown governed-fusion domain {domain!r}")
    module_name, class_name, _max_ev = REACHABLE[domain]
    return f"{module_name}.{class_name}"


def label_provenance(domain: str) -> str:
    """Return the audited ``LABEL_SOURCE`` of the loader that produces ``domain``.

    One of :data:`~omni_mercury_engine.datasets.metadata.VALID_LABEL_SOURCES`.
    Raises if the domain is not in :data:`~research.governed_fusion.suite.REACHABLE`
    or its loader is missing from the registry -- by design, the suite cannot
    score a domain whose provenance has never been audited.
    """
    key = _loader_registry_key(domain)
    if key not in LABEL_PROVENANCE_REGISTRY:
        raise KeyError(
            f"loader {key!r} for domain {domain!r} is not in "
            "LABEL_PROVENANCE_REGISTRY -- run the loader provenance audit "
            "(``python -m omni_mercury_engine.loaders.label_provenance --check``)."
        )
    src, _just = LABEL_PROVENANCE_REGISTRY[key]
    return src


def series_provenance(domain: str, event_id: str) -> str:
    """Return ``"live"`` or ``"reconstructed"`` for a (domain, event_id) pair."""
    return "reconstructed" if is_reconstructed(domain, event_id) else "live"


def event_is_external_label(domain: str, event_id: str) -> bool:
    """True iff an event is honest for the autonomous fitness signal.

    Honest = labels are genuine ground-truth / expert annotation **and** the
    row data is live (not reconstructed).  These are the only events the
    Phase 2 promotion gate is allowed to grade a self-improvement proposal
    on; everything else is reported separately as leakage-flagged.
    """
    return (
        label_provenance(domain) in GENUINE_LABEL_SOURCES
        and series_provenance(domain, event_id) == "live"
    )


def external_label_events(events: list[EventData]) -> list[EventData]:
    """Filter ``events`` to those eligible for the honest fitness signal."""
    return [ev for ev in events if event_is_external_label(ev.domain, ev.event_id)]


def partition_by_provenance(events: list[EventData]) -> dict[str, list[EventData]]:
    """Split ``events`` into ``external_label`` / ``self_label`` / ``reconstructed``.

    Buckets:

    * ``external_label``  -- ``label in GENUINE_LABEL_SOURCES`` AND ``series == live``.
    * ``self_label``      -- ``label == "statistical"`` AND ``series == live``.
    * ``reconstructed``   -- ``series == reconstructed`` (regardless of label
      provenance: a reconstructed series is never a live signal even when its
      labels are catalog-derived, as with ``tsunami``).
    """
    out: dict[str, list[EventData]] = {
        "external_label": [],
        "self_label": [],
        "reconstructed": [],
    }
    for ev in events:
        if series_provenance(ev.domain, ev.event_id) == "reconstructed":
            out["reconstructed"].append(ev)
            continue
        if label_provenance(ev.domain) in GENUINE_LABEL_SOURCES:
            out["external_label"].append(ev)
        else:
            out["self_label"].append(ev)
    return out


def summary(events: list[EventData]) -> dict[str, dict[str, int]]:
    """Counts per provenance bucket: ``{bucket: {n_events, n_rows, n_pos}}``."""
    buckets = partition_by_provenance(events)
    out: dict[str, dict[str, int]] = {}
    for name, evs in buckets.items():
        out[name] = {
            "n_events": len(evs),
            "n_rows": int(sum(ev.X.shape[0] for ev in evs)),
            "n_pos": int(sum(ev.n_pos for ev in evs)),
        }
    return out


#: Bucket name used everywhere for the honest fitness substrate.
HONEST_BUCKET: Final[str] = "external_label"


__all__ = [
    "HONEST_BUCKET",
    "event_is_external_label",
    "external_label_events",
    "label_provenance",
    "partition_by_provenance",
    "series_provenance",
    "summary",
]
