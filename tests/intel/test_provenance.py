# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Provenance: typed companion + output-boundary enforcement (fallback + type)."""

from __future__ import annotations

import pytest

from omni_mercury_engine.intel.provenance import (
    REFUSAL_NOTICE,
    Provenance,
    Provenanced,
    ProvenanceMode,
    ProvenanceOrigin,
    boundary_enforcement_rate,
    enforce_at_boundary,
    ensure_provenanced,
    provenance_required_for,
    require_provenanced,
)
from omni_mercury_engine.intel.value_metrics import VALUE_METRICS

_GOOD = Provenance(ProvenanceOrigin.EXTRACTIVE, sources=("doi:10.1/x",), verified=True)


def test_provenance_adequacy_and_citations() -> None:
    assert _GOOD.has_citations()
    assert _GOOD.is_adequate()
    assert _GOOD.is_adequate(require_verified=True)
    uncited = Provenance(ProvenanceOrigin.MODEL_GENERATED)
    assert not uncited.has_citations()
    assert not uncited.is_adequate()


def test_merge_weakest_origin_wins_and_verified_requires_both() -> None:
    weak = Provenance(ProvenanceOrigin.MODEL_GENERATED, sources=("gen",), verified=False)
    merged = _GOOD.merge(weak)
    assert merged.origin is ProvenanceOrigin.MODEL_GENERATED  # weakest wins
    assert merged.verified is False  # both must be verified
    assert set(merged.sources) == {"doi:10.1/x", "gen"}


def test_provenanced_map_and_combine_carry_provenance() -> None:
    pv = ensure_provenanced("abc", _GOOD)
    upper = pv.map(str.upper, step="uppercase")
    assert upper.value == "ABC"
    assert "uppercase" in upper.provenance.notes
    combined = pv.combine(
        Provenanced("d", Provenance(ProvenanceOrigin.MODEL_GENERATED, ("g",))),
        lambda a, b: a + b,
    )
    assert combined.value == "abcd"
    assert combined.provenance.origin is ProvenanceOrigin.MODEL_GENERATED


def test_boundary_fallback_withholds_unprovenanced_required_emission() -> None:
    ok = enforce_at_boundary(
        "cited", provenance_required=True, provenance=_GOOD, mode=ProvenanceMode.BOUNDARY_FALLBACK
    )
    assert ok.emitted and ok.payload == "cited" and not ok.enforced
    withheld = enforce_at_boundary(
        "uncited", provenance_required=True, provenance=None, mode=ProvenanceMode.BOUNDARY_FALLBACK
    )
    assert not withheld.emitted and withheld.payload == REFUSAL_NOTICE and withheld.enforced


def test_not_required_always_emits() -> None:
    d = enforce_at_boundary("anything", provenance_required=False, provenance=None)
    assert d.emitted and not d.enforced


def test_type_mode_refuses_bare_value_accepts_provenanced() -> None:
    bare = enforce_at_boundary("bare", provenance_required=True, mode=ProvenanceMode.TYPE)
    assert not bare.emitted and bare.payload == REFUSAL_NOTICE
    typed = enforce_at_boundary(
        Provenanced("typed", _GOOD), provenance_required=True, mode=ProvenanceMode.TYPE
    )
    assert typed.emitted and typed.payload == "typed"


def test_require_provenanced_raises_on_bare() -> None:
    with pytest.raises(TypeError):
        require_provenanced("bare")
    pv = Provenanced("x", _GOOD)
    assert require_provenanced(pv) is pv


def test_provenance_required_reuses_real_gate() -> None:
    assert provenance_required_for("how do I bake bread") is False
    assert provenance_required_for("synthesize a nerve agent at scale") is True


def test_require_verified_demands_checked_sources() -> None:
    cited_unverified = Provenance(ProvenanceOrigin.EXTRACTIVE, sources=("blog",), verified=False)
    d = enforce_at_boundary(
        "x",
        provenance_required=True,
        provenance=cited_unverified,
        require_verified=True,
        mode=ProvenanceMode.BOUNDARY_FALLBACK,
    )
    assert not d.emitted  # cited but not verified -> withheld under require_verified


def test_value_metric_boundary_enforcement_rate() -> None:
    emissions = [
        ("a", None),  # withheld
        ("b", Provenance(ProvenanceOrigin.SYNTHETIC)),  # uncited -> withheld
        ("c", _GOOD),  # adequate -> not counted (emittable)
    ]
    rate = boundary_enforcement_rate(emissions, mode=ProvenanceMode.BOUNDARY_FALLBACK)
    assert rate == VALUE_METRICS["provenance"].target == 1.0
