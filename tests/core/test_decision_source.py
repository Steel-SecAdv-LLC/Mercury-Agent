# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every manifest detector must declare where its decision rule comes from.

A detector with no :class:`~omni_mercury_engine.detectors.provenance.
DecisionSource` is a detector whose alarm thresholds cannot be traced to a
published value, a calibrated operating point, a trained checkpoint, or an
explicit experimental declaration.  This suite fails the build when any
class reachable from ``DETECTOR_MANIFEST`` declares none, and when a
declared record is structurally dishonest (wrong type, empty description,
missing mandatory evidence for its kind -- the dataclass refuses those at
construction, so here it is enough that the attribute holds real
``DecisionSource`` records).
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST
from omni_mercury_engine.detectors.provenance import DecisionSource


def _resolve(entry: Any) -> type:
    module = importlib.import_module(entry.module_path)
    return getattr(module, entry.class_name)  # type: ignore[no-any-return]


def _flatten(source: Any) -> list[Any]:
    if isinstance(source, tuple):
        return list(source)
    return [source]


@pytest.mark.parametrize(
    "entry", DETECTOR_MANIFEST, ids=[entry.name for entry in DETECTOR_MANIFEST]
)
def test_every_manifest_detector_declares_a_decision_source(entry: Any) -> None:
    """The class (or, for an adapter, what it wraps) records its provenance."""
    cls = _resolve(entry)
    source = getattr(cls, "decision_source", None)
    assert source is not None, (
        f"{entry.module_path}.{entry.class_name} (manifest entry {entry.name!r}) "
        "declares no decision_source: its alarm rule has no recorded origin. "
        "Declare a DecisionSource (published / calibrated / trained / "
        "experimental) on the class."
    )
    records = _flatten(source)
    assert records, f"{entry.name}: decision_source is an empty tuple"
    for record in records:
        assert isinstance(record, DecisionSource), (
            f"{entry.name}: decision_source entries must be DecisionSource "
            f"records, got {type(record).__name__}"
        )


def test_decision_source_kinds_carry_their_evidence() -> None:
    """The dataclass itself refuses records without mandatory evidence."""
    with pytest.raises(ValueError, match="citation"):
        DecisionSource(kind="published", description="x")
    with pytest.raises(ValueError, match="false_alarm_rate"):
        DecisionSource(kind="calibrated", description="x")
    with pytest.raises(ValueError, match="calibration_data"):
        DecisionSource(kind="calibrated", description="x", false_alarm_rate=0.05)
    with pytest.raises(ValueError, match="operating_point"):
        DecisionSource(kind="trained", description="x")
    with pytest.raises(ValueError, match="kind"):
        DecisionSource(kind="vibes", description="x")
    with pytest.raises(ValueError, match="description"):
        DecisionSource(kind="experimental", description="  ")
