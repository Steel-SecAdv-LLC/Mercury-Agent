# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Loader-side label-provenance leak gate (Phase 1 of governed self-improvement).

The ``datasets/`` side of the codebase already has this discipline
(``tests/datasets/test_label_provenance_gate.py``); this gate extends it to
the live-API ``loaders/`` package that the governed-fusion suite consumes.

Without this gate, a new loader could silently inherit the
``LABEL_SOURCE = "ground_truth"`` default, get pulled into the governed-
fusion manifest, and inflate the headline AUROC by being graded on labels
that are a deterministic function of the same signal it scores. The
fitness signal the autonomous self-improvement loop reads from -- the
*external_label* subset of the manifest -- would then be contaminated by
circular events the moment the loop's first promotion proposal is graded.

These assertions are fully offline (no network, no dataset downloads).
"""

from __future__ import annotations

import pytest

from omni_mercury_engine.datasets.metadata import (
    GENUINE_LABEL_SOURCES,
    VALID_LABEL_SOURCES,
)
from omni_mercury_engine.loaders.label_provenance import (
    LABEL_PROVENANCE_REGISTRY,
    ProvenanceFinding,
    audit_label_provenance,
    discover_loaders,
    ground_truth_loader_keys,
    scan_circular_label_construction,
)


def test_no_label_provenance_leaks() -> None:
    """The canonical gate: the live ``loaders/`` tree must be provenance-clean.

    A failure here means a loader either (a) manufactures labels but is
    counted as genuine (circular -> would inflate the governed-fusion
    headline / contaminate the autonomous fitness signal), (b) is a new
    loader whose provenance was never audited, or (c) diverges from the
    committed registry. Fix the loader's ``LABEL_SOURCE`` or add an audited
    registry entry -- do not weaken this test.
    """
    findings = audit_label_provenance()
    assert findings == [], "loader label-provenance leaks:\n" + "\n".join(
        f"  {f}" for f in findings
    )


def test_every_loader_is_registered() -> None:
    loaders = discover_loaders()
    assert loaders, "no loaders discovered -- import failure?"
    unregistered = sorted(set(loaders) - set(LABEL_PROVENANCE_REGISTRY))
    assert not unregistered, f"loaders missing an audited registry entry: {unregistered}"


def test_registry_values_are_valid() -> None:
    for key, (src, just) in LABEL_PROVENANCE_REGISTRY.items():
        assert src in VALID_LABEL_SOURCES, f"{key}: invalid label_source {src!r}"
        assert just.strip(), f"{key}: empty justification"


def test_registry_has_no_stale_entries() -> None:
    loaders = discover_loaders()
    stale = sorted(set(LABEL_PROVENANCE_REGISTRY) - set(loaders))
    assert not stale, f"registry entries with no live loader class: {stale}"


def test_ground_truth_set_is_the_phase_1_finding() -> None:
    """The audit confirms only two ground-truth loaders feed the live suite.

    This is the transparent baseline Phase 1 surfaces: of the 15 concrete live-API
    loaders, only ``network_security`` and ``sepsis`` produce labels
    independent of any scored feature. Every other loader thresholds a
    scored column (statistical) or reconstructs the series (statistical).
    Phase 2's autonomous fitness signal will read only the ground-truth
    subset; a future loader earns inclusion by being declared and audited
    here, never by quietly inheriting the default.
    """
    expected = {
        "network_security_loader.NetworkSecurityLoader",
        "sepsis_loader.SepsisLoader",
    }
    assert ground_truth_loader_keys() == expected


def test_circularity_heuristic_fires_on_explicit_feature_threshold() -> None:
    """The defense-in-depth scanner catches an explicit ``labels = (df[col] > c)``.

    The earthquake loader uses exactly that shape on a feature column, so
    flipping it to ``ground_truth`` to inflate the headline would be caught
    by the AST scanner even before the registry mismatch fires.
    """
    loaders = discover_loaders()
    assert scan_circular_label_construction(loaders["earthquake_loader.EarthquakeLoader"])


def test_circularity_heuristic_does_not_fire_on_genuine_loaders() -> None:
    """False-positive guard: the scanner must not flag the genuine loaders."""
    loaders = discover_loaders()
    for key in (
        "network_security_loader.NetworkSecurityLoader",
        "sepsis_loader.SepsisLoader",
    ):
        assert not scan_circular_label_construction(loaders[key]), key


def test_gate_detects_an_unregistered_loader() -> None:
    """A loader with no registry entry must be flagged ``unregistered``."""

    class _Rogue:
        LABEL_SOURCE = "ground_truth"

    findings = audit_label_provenance(loaders={"mod._Rogue": _Rogue})  # type: ignore[dict-item]
    rogue = [f for f in findings if f.loader == "mod._Rogue"]
    assert rogue and rogue[0].kind == "unregistered"


def test_gate_detects_a_mismatch() -> None:
    """A loader whose declared source disagrees with the audited registry value."""

    class _Flipped:
        LABEL_SOURCE = "ground_truth"  # dishonest flip of a 'statistical' loader

    findings = audit_label_provenance(
        loaders={"earthquake_loader.EarthquakeLoader": _Flipped}  # type: ignore[dict-item]
    )
    kinds = {f.kind for f in findings if f.loader == "earthquake_loader.EarthquakeLoader"}
    assert "mismatch" in kinds


def test_finding_str_is_informative() -> None:
    f = ProvenanceFinding("mod.Foo", "mismatch", "x")
    assert "mod.Foo" in str(f) and "mismatch" in str(f)


@pytest.mark.parametrize("src", sorted(GENUINE_LABEL_SOURCES))
def test_genuine_sources_are_supervised_safe(src: str) -> None:
    from omni_mercury_engine.datasets.metadata import is_supervised_eval_safe

    assert is_supervised_eval_safe(src)
