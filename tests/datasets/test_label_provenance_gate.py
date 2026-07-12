# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Repo-wide label-provenance leak gate (WS-A follow-on).

PR #262 de-leaked the supervised headline by excluding circular,
manufactured-label datasets (``LABEL_SOURCE = "statistical"``). That de-leak is
only as good as each loader's transparency, because the base class silently defaults
``LABEL_SOURCE = "ground_truth"``. This gate promotes the one-off audit into a
permanent, repo-wide check:

* every concrete ``DatasetLoader`` is in the committed provenance registry;
* every loader's declared ``LABEL_SOURCE`` equals its audited value;
* no loader manufactures labels from a feature threshold while being declared
  genuine (the circularity heuristic);
* the registry has no stale or invalid entries.

These assertions are fully offline (no network, no dataset downloads).
"""

from __future__ import annotations

import pytest

from omni_mercury_engine.datasets.label_provenance import (
    LABEL_PROVENANCE_REGISTRY,
    ProvenanceFinding,
    audit_label_provenance,
    discover_loaders,
    scan_circular_label_construction,
)
from omni_mercury_engine.datasets.metadata import (
    GENUINE_LABEL_SOURCES,
    VALID_LABEL_SOURCES,
)


def test_no_label_provenance_leaks() -> None:
    """The canonical gate: the live tree must be provenance-clean.

    A failure here means a dataset loader either (a) manufactures anomaly
    labels but is counted as genuine (circular -> inflates the headline AUC),
    (b) is a new loader whose provenance was never audited, or (c) diverges
    from the committed registry. Fix the loader's ``LABEL_SOURCE`` or add an
    audited registry entry -- do not weaken this test.
    """
    findings = audit_label_provenance()
    assert findings == [], "label-provenance leaks:\n" + "\n".join(f"  {f}" for f in findings)


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


def test_climate_loaders_are_quarantined_as_statistical() -> None:
    """Regression for this round's residual-circularity finding: all four
    climate loaders threshold the same features the detector consumes, so they
    must be excluded from the supervised headline."""
    loaders = discover_loaders()
    for key in (
        "climate.SimonsCMAPLoader",
        "climate.WorldOceanDatabaseLoader",
        "climate.CopernicusSeaLevelLoader",
        "climate.CopernicusERA5Loader",
    ):
        assert loaders[key].LABEL_SOURCE == "statistical", key


def test_circularity_heuristic_fires_on_feature_threshold() -> None:
    """The heuristic must catch the exact circular pattern (feature-threshold
    labels in a real path) and not fire on genuine label-column reads."""
    loaders = discover_loaders()
    # Manufactured-from-feature-threshold -> must fire.
    assert scan_circular_label_construction(loaders["climate.SimonsCMAPLoader"])
    assert scan_circular_label_construction(loaders["climate.CopernicusSeaLevelLoader"])
    # Genuine label-column reads -> must NOT fire (false-positive guard).
    for key in (
        "industrial.SWaTLoader",
        "security.NSLKDDLoader",
        "timeseries.NABLoader",
        "adbench.ADBenchLoader",
        "ucr_archive.UCRLoader",
    ):
        assert not scan_circular_label_construction(loaders[key]), key


def test_gate_detects_an_unregistered_loader() -> None:
    """A loader with no registry entry must be flagged ``unregistered`` -- the
    mechanism that forces every future dataset to declare its provenance.

    Uses injected loaders (no global subclass-registry pollution / test-order
    coupling); ``_Rogue`` is not a real ``DatasetLoader`` subclass on purpose.
    """

    class _Rogue:
        LABEL_SOURCE = "ground_truth"

    findings = audit_label_provenance(loaders={"mod._Rogue": _Rogue})  # type: ignore[dict-item]
    rogue = [f for f in findings if f.loader == "mod._Rogue"]
    assert rogue and rogue[0].kind == "unregistered"


def test_gate_detects_a_mismatch() -> None:
    """A loader whose declared source disagrees with the audited registry value
    (e.g. a manufactured-label loader flipped to ``ground_truth``) is flagged.

    Plain stand-in class (no real ``DatasetLoader`` subclassing) so nothing
    leaks into the global subclass registry / other tests.
    """

    class _Flipped:
        LABEL_SOURCE = "ground_truth"  # dishonest flip of a 'statistical' loader

    findings = audit_label_provenance(
        loaders={"climate.SimonsCMAPLoader": _Flipped}  # type: ignore[dict-item]
    )
    kinds = {f.kind for f in findings if f.loader == "climate.SimonsCMAPLoader"}
    assert "mismatch" in kinds


def test_finding_str_is_informative() -> None:
    f = ProvenanceFinding("mod.Foo", "mismatch", "x")
    assert "mod.Foo" in str(f) and "mismatch" in str(f)


@pytest.mark.parametrize("src", sorted(GENUINE_LABEL_SOURCES))
def test_genuine_sources_are_supervised_safe(src: str) -> None:
    from omni_mercury_engine.datasets.metadata import is_supervised_eval_safe

    assert is_supervised_eval_safe(src)
