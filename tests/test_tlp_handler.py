# Copyright (C) 2025 Steel Security Advisors LLC
"""Tests for :mod:`omni_mercury_engine.compliance.tlp_handler`."""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from omni_mercury_engine.compliance import (
    TLPClassification,
    TLPColor,
    TLPHandler,
    TLPValidationError,
    get_tlp_handler,
)


@pytest.fixture()
def handler() -> TLPHandler:
    return TLPHandler()


# ---------------------------------------------------------------------------
# TLPColor enum
# ---------------------------------------------------------------------------


def test_tlp_color_includes_amber_strict() -> None:
    assert {c.value for c in TLPColor} == {
        "CLEAR",
        "GREEN",
        "AMBER",
        "AMBER+STRICT",
        "RED",
    }


def test_tlp_color_labels_are_canonical() -> None:
    assert TLPColor.RED.label == "TLP:RED"
    assert TLPColor.AMBER_STRICT.label == "TLP:AMBER+STRICT"
    assert TLPColor.CLEAR.label == "TLP:CLEAR"


def test_tlp_color_rank_is_monotonic() -> None:
    ranks = [
        c.rank
        for c in (
            TLPColor.CLEAR,
            TLPColor.GREEN,
            TLPColor.AMBER,
            TLPColor.AMBER_STRICT,
            TLPColor.RED,
        )
    ]
    assert ranks == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Handler construction
# ---------------------------------------------------------------------------


def test_handler_default_thresholds(handler: TLPHandler) -> None:
    assert handler.red_threshold == TLPHandler.DEFAULT_RED_THRESHOLD
    assert handler.amber_threshold == TLPHandler.DEFAULT_AMBER_THRESHOLD
    assert handler.green_threshold == TLPHandler.DEFAULT_GREEN_THRESHOLD


def test_handler_rejects_non_monotonic_thresholds() -> None:
    with pytest.raises(TLPValidationError, match=r"monotonic|must satisfy"):
        TLPHandler(red_threshold=0.5, amber_threshold=0.7, green_threshold=0.3)


def test_handler_rejects_out_of_range_thresholds() -> None:
    with pytest.raises(TLPValidationError):
        TLPHandler(red_threshold=1.5, amber_threshold=0.6, green_threshold=0.3)


def test_factory_returns_default_handler() -> None:
    h = get_tlp_handler()
    assert isinstance(h, TLPHandler)
    assert h.red_threshold == TLPHandler.DEFAULT_RED_THRESHOLD


# ---------------------------------------------------------------------------
# classify_anomaly: score-based paths
# ---------------------------------------------------------------------------


def test_classify_red_via_score(handler: TLPHandler) -> None:
    cls = handler.classify_anomaly(0.95, "anomaly", "general")
    assert cls.color is TLPColor.RED
    assert "Critical severity" in cls.reasoning


def test_classify_amber_via_score(handler: TLPHandler) -> None:
    cls = handler.classify_anomaly(0.65, "anomaly", "general")
    assert cls.color is TLPColor.AMBER


def test_classify_green_via_score(handler: TLPHandler) -> None:
    cls = handler.classify_anomaly(0.40, "anomaly", "general")
    assert cls.color is TLPColor.GREEN


def test_classify_clear_via_low_score(handler: TLPHandler) -> None:
    cls = handler.classify_anomaly(0.10, "anomaly", "general")
    assert cls.color is TLPColor.CLEAR


# ---------------------------------------------------------------------------
# classify_anomaly: domain-driven escalation
# ---------------------------------------------------------------------------


def test_critical_domain_escalates_to_red(handler: TLPHandler) -> None:
    cls = handler.classify_anomaly(0.75, "anomaly", "cyber")
    assert cls.color is TLPColor.RED


def test_critical_type_escalates_to_red(handler: TLPHandler) -> None:
    cls = handler.classify_anomaly(0.75, "malware", "general")
    assert cls.color is TLPColor.RED


def test_critical_type_mid_score_classifies_amber(handler: TLPHandler) -> None:
    cls = handler.classify_anomaly(0.55, "intrusion", "general")
    assert cls.color is TLPColor.AMBER


def test_sensitive_type_above_amber_classifies_amber_or_strict(
    handler: TLPHandler,
) -> None:
    cls = handler.classify_anomaly(0.70, "patient_data", "medical")
    assert cls.color is TLPColor.AMBER_STRICT


def test_sensitive_type_above_green_classifies_green(handler: TLPHandler) -> None:
    cls = handler.classify_anomaly(0.40, "vulnerability", "cyber")
    assert cls.color is TLPColor.GREEN


# ---------------------------------------------------------------------------
# classify_anomaly: AMBER+STRICT escalation
# ---------------------------------------------------------------------------


def test_strict_sharing_flag_escalates_amber_to_strict(handler: TLPHandler) -> None:
    cls = handler.classify_anomaly(0.65, "anomaly", "general", context={"strict_sharing": True})
    assert cls.color is TLPColor.AMBER_STRICT


def test_contains_pii_flag_escalates_amber_to_strict(handler: TLPHandler) -> None:
    cls = handler.classify_anomaly(0.65, "anomaly", "general", context={"contains_pii": True})
    assert cls.color is TLPColor.AMBER_STRICT


def test_medical_patient_data_defaults_to_strict(handler: TLPHandler) -> None:
    cls = handler.classify_anomaly(0.65, "patient_data", "medical")
    assert cls.color is TLPColor.AMBER_STRICT


def test_red_classification_is_not_downgraded_by_strict_logic(
    handler: TLPHandler,
) -> None:
    cls = handler.classify_anomaly(0.95, "anomaly", "general", context={"strict_sharing": True})
    assert cls.color is TLPColor.RED


# ---------------------------------------------------------------------------
# classify_anomaly: input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_score", [-0.01, 1.01, math.nan, math.inf, -math.inf])
def test_classify_rejects_invalid_scores(handler: TLPHandler, bad_score: float) -> None:
    with pytest.raises(TLPValidationError):
        handler.classify_anomaly(bad_score, "anomaly", "general")


def test_classify_rejects_empty_type(handler: TLPHandler) -> None:
    with pytest.raises(TLPValidationError, match="anomaly_type"):
        handler.classify_anomaly(0.5, "", "general")


def test_classify_rejects_empty_domain(handler: TLPHandler) -> None:
    with pytest.raises(TLPValidationError, match="domain"):
        handler.classify_anomaly(0.5, "anomaly", "")


# ---------------------------------------------------------------------------
# Confidence calculation
# ---------------------------------------------------------------------------


def test_confidence_higher_for_extreme_scores_and_well_known_domains(
    handler: TLPHandler,
) -> None:
    high = handler.classify_anomaly(0.95, "malware", "cyber")
    mid = handler.classify_anomaly(0.55, "anomaly", "general")
    assert high.confidence >= mid.confidence
    assert 0.0 <= high.confidence <= 1.0
    assert 0.0 <= mid.confidence <= 1.0


def test_confidence_capped_at_one(handler: TLPHandler) -> None:
    cls = handler.classify_anomaly(0.99, "malware", "cyber")
    assert cls.confidence <= 1.0


# ---------------------------------------------------------------------------
# Reasoning / ethical considerations
# ---------------------------------------------------------------------------


def test_reasoning_includes_domain_and_type(handler: TLPHandler) -> None:
    cls = handler.classify_anomaly(0.50, "anomaly", "general")
    assert "Domain: general" in cls.reasoning
    assert "Type: anomaly" in cls.reasoning


def test_red_classification_adds_extra_ethical_considerations(
    handler: TLPHandler,
) -> None:
    cls = handler.classify_anomaly(0.95, "malware", "cyber")
    text = " ".join(cls.ethical_considerations)
    assert "explicit authorization" in text.lower()
    assert "vulnerability" in text.lower() or "ongoing investigations" in text.lower()


def test_medical_domain_adds_hipaa_considerations(handler: TLPHandler) -> None:
    cls = handler.classify_anomaly(0.50, "patient_data", "medical")
    text = " ".join(cls.ethical_considerations)
    assert "hipaa" in text.lower()


# ---------------------------------------------------------------------------
# batch_classify & statistics
# ---------------------------------------------------------------------------


def test_batch_classify_returns_one_result_per_input(handler: TLPHandler) -> None:
    inputs = [
        {"score": 0.95, "type": "malware", "domain": "cyber"},
        {"score": 0.65, "type": "patient_data", "domain": "medical"},
        {"score": 0.40, "type": "anomaly", "domain": "general"},
        {"score": 0.10, "type": "anomaly", "domain": "general"},
    ]
    results = handler.batch_classify(inputs)
    assert len(results) == 4
    colors = [r.color for r in results]
    assert colors == [
        TLPColor.RED,
        TLPColor.AMBER_STRICT,
        TLPColor.GREEN,
        TLPColor.CLEAR,
    ]


def test_batch_classify_handles_missing_fields(handler: TLPHandler) -> None:
    results = handler.batch_classify([{}])
    assert len(results) == 1
    # Default score=0.0 -> CLEAR
    assert results[0].color is TLPColor.CLEAR


def test_get_color_statistics_zero_initialised(handler: TLPHandler) -> None:
    stats = handler.get_color_statistics([])
    assert set(stats.keys()) == {c.value for c in TLPColor}
    assert all(v == 0 for v in stats.values())


def test_get_color_statistics_counts_each_color(handler: TLPHandler) -> None:
    inputs = [
        {"score": 0.95, "type": "malware", "domain": "cyber"},
        {"score": 0.95, "type": "malware", "domain": "cyber"},
        {"score": 0.65, "type": "patient_data", "domain": "medical"},
        {"score": 0.40, "type": "anomaly", "domain": "general"},
    ]
    results = handler.batch_classify(inputs)
    stats = handler.get_color_statistics(results)
    assert stats["RED"] == 2
    assert stats["AMBER+STRICT"] == 1
    assert stats["GREEN"] == 1
    assert stats["AMBER"] == 0
    assert stats["CLEAR"] == 0


# ---------------------------------------------------------------------------
# Watermark + export metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("color", list(TLPColor))
def test_watermark_starts_with_canonical_label(handler: TLPHandler, color: TLPColor) -> None:
    watermark = handler.generate_watermark_text(color)
    assert watermark.startswith(color.label + " - ")


def test_export_metadata_has_required_keys(handler: TLPHandler) -> None:
    cls = handler.classify_anomaly(0.95, "malware", "cyber")
    meta = handler.get_export_metadata(cls)
    expected_keys = {
        "tlp_color",
        "tlp_label",
        "tlp_rank",
        "tlp_confidence",
        "tlp_reasoning",
        "sharing_guidelines",
        "ethical_considerations",
        "watermark",
    }
    assert expected_keys <= set(meta.keys())
    # The export block must be JSON-serialisable in isolation.
    json.dumps(meta)


def test_export_metadata_label_matches_color(handler: TLPHandler) -> None:
    cls = handler.classify_anomaly(0.65, "patient_data", "medical")
    meta = handler.get_export_metadata(cls)
    assert meta["tlp_color"] == cls.color.value
    assert meta["tlp_label"] == cls.color.label
    assert meta["tlp_rank"] == cls.color.rank


# ---------------------------------------------------------------------------
# Dataclass invariants
# ---------------------------------------------------------------------------


def test_classification_is_immutable(handler: TLPHandler) -> None:
    cls = handler.classify_anomaly(0.65, "anomaly", "general")
    assert isinstance(cls, TLPClassification)
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        cls.color = TLPColor.RED  # type: ignore[misc]


def test_ethical_considerations_is_tuple(handler: TLPHandler) -> None:
    cls = handler.classify_anomaly(0.65, "anomaly", "general")
    assert isinstance(cls.ethical_considerations, tuple)
