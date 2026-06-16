# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-event label-provenance integrity for the governed-fusion manifest.

This locks the honest-fitness substrate Phase 2's promotion gate reads from:

* every (domain, event_id) in the manifest classifies cleanly into one of
  ``external_label`` / ``self_label`` / ``reconstructed``;
* the manifest's ``provenance_summary`` agrees with the loader-side audit;
* the ``external_label`` bucket is exactly the two ``network_security``
  events (Phase 1's honest finding) -- not zero (a regression that would
  silently strand the autonomous fitness loop with no honest signal) and
  not silently broader (a regression that would re-admit a leakage-flagged
  event without an audited registry update);
* every manifest entry's ``label_provenance`` matches the loader's
  ``LABEL_SOURCE`` (no drift between the writer and the live audit).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from omni_mercury_engine.datasets.metadata import GENUINE_LABEL_SOURCES
from research.governed_fusion.label_provenance import (
    HONEST_BUCKET,
    event_is_external_label,
    label_provenance,
    series_provenance,
)

_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2] / "research" / "governed_fusion" / "manifest.json"
)


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    with _MANIFEST_PATH.open() as fh:
        return json.load(fh)


def _all_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return list(manifest["real"]) + list(manifest["reconstructed"])


def test_every_entry_has_provenance_fields(manifest: dict[str, Any]) -> None:
    for e in _all_entries(manifest):
        for key in ("label_provenance", "series_provenance", "external_label"):
            assert key in e, f"{e['domain']}/{e['event_id']}: manifest missing {key!r}"


def test_manifest_provenance_matches_loader_audit(manifest: dict[str, Any]) -> None:
    """Manifest entries must agree with the loader-side audited registry.

    A drift here means either the manifest was hand-edited out of sync (which
    is what build_manifest.py prevents), or a loader's ``LABEL_SOURCE`` was
    flipped without re-running the manifest builder. Either way the headline
    is no longer honest.
    """
    for e in _all_entries(manifest):
        d, eid = e["domain"], e["event_id"]
        assert e["label_provenance"] == label_provenance(d), f"{d}/{eid}: label drift"
        assert e["series_provenance"] == series_provenance(d, eid), f"{d}/{eid}: series drift"
        assert e["external_label"] == event_is_external_label(d, eid), f"{d}/{eid}: external drift"


def test_external_label_bucket_is_the_phase_1_finding(manifest: dict[str, Any]) -> None:
    """Phase 1's honest finding: only 2 live events feed the fitness signal.

    Today those are network_security/batadal and network_security/nsl_kdd
    (the only loader in the manifest with audited ``LABEL_SOURCE = "ground_truth"``).
    Any change that broadens or empties this set must be intentional: extend
    the loader registry with an audited justification, then update this test
    explicitly. Do not weaken the assertion silently.
    """
    bucket = manifest["provenance_summary"]["real"][HONEST_BUCKET]
    assert bucket["n_events"] == 2, bucket

    expected = {
        ("network_security", "batadal"),
        ("network_security", "nsl_kdd"),
    }
    actual = {(e["domain"], e["event_id"]) for e in manifest["real"] if e["external_label"]}
    assert actual == expected, sorted(actual)


def test_reconstructed_events_are_never_external_label(manifest: dict[str, Any]) -> None:
    """A reconstructed-series event can never feed the live fitness signal."""
    for e in _all_entries(manifest):
        if e["series_provenance"] == "reconstructed":
            assert not e["external_label"], e


def test_self_label_bucket_is_consistent_with_loader_registry(
    manifest: dict[str, Any],
) -> None:
    """Self-labelled live events must come from loaders whose audited
    provenance is **not** in the genuine set. If a future PR flips a loader to
    ``ground_truth`` without re-running the manifest builder, this drift fires
    here before the headline is touched.
    """
    for e in manifest["real"]:
        if e["series_provenance"] == "live" and not e["external_label"]:
            assert e["label_provenance"] not in GENUINE_LABEL_SOURCES, e


def test_provenance_summary_counts_match_entries(manifest: dict[str, Any]) -> None:
    """The summary block is the audit reviewers read; counts must reconcile."""
    summary = manifest["provenance_summary"]
    for kind in ("real", "reconstructed"):
        from_summary = {
            b: summary[kind][b]["n_events"]
            for b in ("external_label", "self_label", "reconstructed")
        }
        entries = manifest[kind]
        counted = {"external_label": 0, "self_label": 0, "reconstructed": 0}
        for e in entries:
            if e["series_provenance"] == "reconstructed":
                counted["reconstructed"] += 1
            elif e["external_label"]:
                counted["external_label"] += 1
            else:
                counted["self_label"] += 1
        assert from_summary == counted, (kind, from_summary, counted)


def test_honest_fitness_bucket_is_named_external_label(manifest: dict[str, Any]) -> None:
    assert manifest["provenance_summary"]["honest_fitness_bucket"] == HONEST_BUCKET
