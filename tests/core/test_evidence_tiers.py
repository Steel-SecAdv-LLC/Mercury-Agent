# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Test evidence tiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.core import evidence_tiers

if TYPE_CHECKING:
    from omni_mercury_engine.engine import OmniMercuryEngine

BROKEN = {"affective", "chemistry", "parapsychology"}


@pytest.fixture(scope="module")
def engine() -> OmniMercuryEngine:
    from omni_mercury_engine.engine import OmniMercuryEngine

    return OmniMercuryEngine(mode="fusion", require_explicit_fit=False)


def test_the_packaged_tiers_are_the_ledgers() -> None:
    ledger = json.loads(
        (
            Path(__file__).resolve().parents[2] / "artifacts" / "detector_evidence" / "ledger.json"
        ).read_text(encoding="utf-8")
    )

    held = evidence_tiers.tiers()

    assert set(held) == set(ledger["detectors"])
    for name, card in ledger["detectors"].items():
        assert held[name] == {"tier": card["tier"], "basis": card["tier_basis"]}, name
    assert evidence_tiers.measured_broken() == BROKEN
    assert evidence_tiers.tier("temporal") == "T2"
    assert evidence_tiers.tier("no-such-component") is None


def test_tiers_that_cannot_be_read_claim_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evidence_tiers, "_PATH", tmp_path / "missing.json")
    evidence_tiers.tiers.cache_clear()
    try:
        assert evidence_tiers.tiers() == {}
        assert evidence_tiers.measured_broken() == frozenset()
        assert evidence_tiers.tier("temporal") is None
    finally:
        evidence_tiers.tiers.cache_clear()


def test_measured_broken_models_are_constructed_and_contribute_nothing(
    engine: OmniMercuryEngine,
) -> None:
    assert set(engine.models) >= BROKEN  # the released API keeps them
    assert engine.excluded_unproven == frozenset(BROKEN)
    walked = {name for name, _ in engine._models_on_the_default_path()}
    assert walked.isdisjoint(BROKEN)
    assert "quantum" in walked and "neurosymbolic" in walked

    features, scores = engine._extract_model_features(np.random.RandomState(7).rand(24, 6))

    assert set(features).isdisjoint(BROKEN) and set(scores).isdisjoint(BROKEN)
    assert features  # the others still speak


def test_include_unproven_restores_them() -> None:
    from omni_mercury_engine.engine import OmniMercuryEngine

    opted_in = OmniMercuryEngine(mode="fusion", require_explicit_fit=False, include_unproven=True)

    assert opted_in.excluded_unproven == frozenset()
    assert {name for name, _ in opted_in._models_on_the_default_path()} >= BROKEN


def test_a_checkpoint_trained_with_its_group_keeps_that_group_fed(
    engine: OmniMercuryEngine,
) -> None:
    before = engine._fusion_feature_groups
    engine._fusion_feature_groups = ["statistical", "chemistry"]
    try:
        walked = {name for name, _ in engine._models_on_the_default_path()}
    finally:
        engine._fusion_feature_groups = before

    assert "chemistry" in walked  # a trained projection is not starved
    assert walked.isdisjoint(BROKEN - {"chemistry"})
