# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioral tests for ``detectors/cross_domain_frequency.py``.

Covers:
- ``_compute_overlap``: partial overlap, containment, disjoint and
  touching-edge intervals
- ``_extract_bands_from_influence_vector``: dict payloads, attribute-style
  payloads (including the real ``FrequencyInfluenceVector``), unknown band
  labels, unknown domains falling back to environmental band definitions,
  significance flagging at the 0.3 boundary
- ``CrossDomainFrequencyCorrelator.correlate``: empty / single-domain inputs,
  non-overlapping bands, overlap geometry and geometric-mean strength,
  negative-score clamping, significance filtering, alert-level thresholds
  (defaults, custom, partial and empty mappings), pair counting for three
  domains, top-5 truncation in the description
- Safety-language contract: every description contains
  "requires human assessment" (correlation, never prediction)
- ``BandOverlap`` immutability (frozen dataclass)

NOTE: ``cross_domain_frequency`` was unreferenced anywhere else in ``src/``
when this suite was written (verified by repo-wide grep); it has since been
wired into the ``omni_mercury_engine.detectors`` public surface (lazy
export + ``__all__``) as the executed ROADMAP row-19 decision, and
``TestPublicWiring`` below pins that wiring.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from omni_mercury_engine.detectors.cross_domain_frequency import (
    BandOverlap,
    CrossDomainCorrelation,
    CrossDomainFrequencyCorrelator,
    DomainBandInfo,
    _compute_overlap,
    _extract_bands_from_influence_vector,
)
from omni_mercury_engine.detectors.spectral_domain_frequency import (
    FrequencyInfluenceVector,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vector(band_scores: dict[str, float]) -> dict[str, dict[str, float]]:
    """Dict-shaped influence-vector payload accepted by the correlator."""
    return {"band_scores": band_scores}


def _real_vector(band_scores: dict[str, float]) -> FrequencyInfluenceVector:
    """A genuine FrequencyInfluenceVector carrying the given band scores."""
    return FrequencyInfluenceVector(
        influence_multiplier=1.0,
        band_scores=band_scores,
        aggregate_score=0.5,
        aggregate_p_value=0.05,
        spectral_entropy=0.5,
        dominant_frequency=8.0,
        spectral_centroid=8.0,
        change_point_detected=False,
        confidence=0.95,
    )


# ---------------------------------------------------------------------------
# _compute_overlap
# ---------------------------------------------------------------------------


class TestComputeOverlap:
    def test_partial_overlap_returns_intersection(self) -> None:
        assert _compute_overlap(1.0, 10.0, 5.0, 20.0) == (5.0, 10.0)

    def test_containment_returns_inner_interval(self) -> None:
        assert _compute_overlap(0.0, 100.0, 40.0, 60.0) == (40.0, 60.0)

    def test_disjoint_intervals_return_none(self) -> None:
        assert _compute_overlap(0.0, 1.0, 2.0, 3.0) is None

    def test_touching_edges_return_none(self) -> None:
        # Shared boundary only: zero-width interval is not an overlap.
        assert _compute_overlap(0.0, 5.0, 5.0, 10.0) is None

    def test_symmetric_in_argument_order(self) -> None:
        assert _compute_overlap(5.0, 20.0, 1.0, 10.0) == _compute_overlap(1.0, 10.0, 5.0, 20.0)


# ---------------------------------------------------------------------------
# _extract_bands_from_influence_vector
# ---------------------------------------------------------------------------


class TestExtractBands:
    def test_dict_payload_resolves_known_band_hz_ranges(self) -> None:
        bands = _extract_bands_from_influence_vector(
            "environmental", _vector({"schumann_fundamental": 0.9})
        )
        assert len(bands) == 1
        band = bands[0]
        assert isinstance(band, DomainBandInfo)
        assert band.domain == "environmental"
        assert band.band_label == "schumann_fundamental"
        assert band.low_hz == pytest.approx(7.83)
        assert band.high_hz == pytest.approx(8.5)
        assert band.anomaly_score == pytest.approx(0.9)
        assert band.is_significant is True

    def test_real_influence_vector_payload(self) -> None:
        bands = _extract_bands_from_influence_vector(
            "space", _real_vector({"schumann_coupling": 0.7})
        )
        assert len(bands) == 1
        assert bands[0].low_hz == pytest.approx(0.1)
        assert bands[0].high_hz == pytest.approx(8.0)

    def test_unknown_band_label_is_skipped(self) -> None:
        bands = _extract_bands_from_influence_vector(
            "environmental", _vector({"no_such_band": 0.9, "schumann_fundamental": 0.4})
        )
        assert [b.band_label for b in bands] == ["schumann_fundamental"]

    def test_unknown_domain_falls_back_to_environmental_bands(self) -> None:
        # get_domain_frequency_bands falls back to environmental definitions,
        # so environmental labels resolve under any domain name.
        bands = _extract_bands_from_influence_vector(
            "made_up_domain", _vector({"schumann_fundamental": 0.5})
        )
        assert len(bands) == 1
        assert bands[0].domain == "made_up_domain"
        assert bands[0].low_hz == pytest.approx(7.83)

    def test_payload_without_band_scores_yields_no_bands(self) -> None:
        assert _extract_bands_from_influence_vector("environmental", object()) == []
        assert _extract_bands_from_influence_vector("environmental", {"other": 1}) == []

    def test_significance_threshold_is_strictly_above_0_3(self) -> None:
        bands = _extract_bands_from_influence_vector(
            "environmental",
            _vector({"schumann_fundamental": 0.3, "schumann_harmonic_1": 0.31}),
        )
        by_label = {b.band_label: b for b in bands}
        assert by_label["schumann_fundamental"].is_significant is False
        assert by_label["schumann_harmonic_1"].is_significant is True


# ---------------------------------------------------------------------------
# CrossDomainFrequencyCorrelator
# ---------------------------------------------------------------------------


class TestCorrelatorConstruction:
    def test_default_thresholds(self) -> None:
        correlator = CrossDomainFrequencyCorrelator()
        assert correlator.significance_threshold == pytest.approx(0.3)
        assert correlator.alert_thresholds == {"low": 0.2, "medium": 0.4, "high": 0.7}

    def test_custom_thresholds_are_kept(self) -> None:
        custom = {"low": 0.1, "medium": 0.5, "high": 0.9}
        correlator = CrossDomainFrequencyCorrelator(
            significance_threshold=0.6, alert_thresholds=custom
        )
        assert correlator.significance_threshold == pytest.approx(0.6)
        assert correlator.alert_thresholds == custom


class TestCorrelateBasics:
    def test_empty_input_produces_null_result(self) -> None:
        result = CrossDomainFrequencyCorrelator().correlate({})
        assert isinstance(result, CrossDomainCorrelation)
        assert result.correlation_score == 0.0
        assert result.alert_level == "none"
        assert result.overlapping_bands == []
        assert result.domain_pairs_checked == 0
        assert result.significant_overlaps == 0
        assert "requires human assessment" in result.description.lower()

    def test_single_domain_checks_no_pairs(self) -> None:
        result = CrossDomainFrequencyCorrelator().correlate(
            {"environmental": _vector({"schumann_fundamental": 0.99})}
        )
        assert result.domain_pairs_checked == 0
        assert result.overlapping_bands == []
        assert result.alert_level == "none"

    def test_disjoint_bands_yield_no_overlaps(self) -> None:
        # security/ultra_high_rate spans 10 kHz - 100 kHz; medical/alpha_neural
        # spans 8 - 13 Hz. No Hz intersection despite both being anomalous.
        result = CrossDomainFrequencyCorrelator().correlate(
            {
                "security": _vector({"ultra_high_rate": 0.9}),
                "medical": _vector({"alpha_neural": 0.9}),
            }
        )
        assert result.domain_pairs_checked == 1
        assert result.overlapping_bands == []
        assert result.significant_overlaps == 0
        assert result.correlation_score == 0.0
        assert result.alert_level == "none"
        assert "1 domain pair(s)" in result.description
        assert "requires human assessment" in result.description.lower()

    def test_three_domains_check_three_pairs(self) -> None:
        result = CrossDomainFrequencyCorrelator().correlate(
            {
                "environmental": _vector({}),
                "space": _vector({}),
                "medical": _vector({}),
            }
        )
        assert result.domain_pairs_checked == 3


class TestCorrelateOverlaps:
    def test_overlap_geometry_and_geometric_mean_strength(self) -> None:
        # environmental/schumann_fundamental: 7.83 - 8.5 Hz, score 0.9
        # space/schumann_coupling:            0.1  - 8.0 Hz, score 0.8
        # Overlap: 7.83 - 8.0 Hz; strength = sqrt(0.9 * 0.8)
        result = CrossDomainFrequencyCorrelator().correlate(
            {
                "environmental": _vector({"schumann_fundamental": 0.9}),
                "space": _vector({"schumann_coupling": 0.8}),
            }
        )
        assert len(result.overlapping_bands) == 1
        overlap = result.overlapping_bands[0]
        # Domains are iterated in sorted order.
        assert overlap.domain_a == "environmental"
        assert overlap.domain_b == "space"
        assert overlap.band_a == "schumann_fundamental"
        assert overlap.band_b == "schumann_coupling"
        assert overlap.overlap_low_hz == pytest.approx(7.83)
        assert overlap.overlap_high_hz == pytest.approx(8.0)
        assert overlap.score_a == pytest.approx(0.9)
        assert overlap.score_b == pytest.approx(0.8)
        expected_strength = math.sqrt(0.9 * 0.8)
        assert overlap.correlation_strength == pytest.approx(expected_strength)
        # Single significant overlap: the mean equals the strength itself.
        assert result.correlation_score == pytest.approx(expected_strength)
        assert result.significant_overlaps == 1
        assert result.alert_level == "high"

    def test_significant_description_lists_overlap_and_alert(self) -> None:
        result = CrossDomainFrequencyCorrelator().correlate(
            {
                "environmental": _vector({"schumann_fundamental": 0.9}),
                "space": _vector({"schumann_coupling": 0.8}),
            }
        )
        description = result.description
        assert "Detected 1 significant" in description
        assert "alert=high" in description
        assert "environmental/schumann_fundamental" in description
        assert "space/schumann_coupling" in description
        assert "requires human assessment" in description.lower()
        # The module promises correlation-only language, never prediction.
        assert "predicted" not in description.lower()

    def test_weak_overlap_is_recorded_but_not_significant(self) -> None:
        result = CrossDomainFrequencyCorrelator().correlate(
            {
                "environmental": _vector({"schumann_fundamental": 0.2}),
                "space": _vector({"schumann_coupling": 0.2}),
            }
        )
        assert len(result.overlapping_bands) == 1
        assert result.significant_overlaps == 0
        # correlation_score averages *significant* overlaps only.
        assert result.correlation_score == 0.0
        assert result.alert_level == "none"

    def test_negative_scores_are_clamped_to_zero_strength(self) -> None:
        result = CrossDomainFrequencyCorrelator().correlate(
            {
                "environmental": _vector({"schumann_fundamental": -1.0}),
                "space": _vector({"schumann_coupling": 0.9}),
            }
        )
        assert len(result.overlapping_bands) == 1
        strength = result.overlapping_bands[0].correlation_strength
        assert strength == 0.0
        assert not math.isnan(strength)

    def test_custom_significance_threshold_filters_overlaps(self) -> None:
        domain_vectors = {
            "environmental": _vector({"schumann_fundamental": 0.6}),
            "space": _vector({"schumann_coupling": 0.6}),
        }
        default = CrossDomainFrequencyCorrelator().correlate(domain_vectors)
        strict = CrossDomainFrequencyCorrelator(significance_threshold=0.9).correlate(
            domain_vectors
        )
        assert default.significant_overlaps == 1
        assert strict.significant_overlaps == 0
        assert strict.correlation_score == 0.0

    @pytest.mark.parametrize(
        ("score", "expected_level"),
        [
            (0.25, "low"),  # strength 0.25 in [0.2, 0.4)
            (0.5, "medium"),  # strength 0.5 in [0.4, 0.7)
            (0.9, "high"),  # strength 0.9 >= 0.7
        ],
    )
    def test_default_alert_levels(self, score: float, expected_level: str) -> None:
        # With equal scores the geometric mean equals the score, so the
        # resulting correlation_score is the score itself.
        result = CrossDomainFrequencyCorrelator(significance_threshold=0.0).correlate(
            {
                "environmental": _vector({"schumann_fundamental": score}),
                "space": _vector({"schumann_coupling": score}),
            }
        )
        assert result.correlation_score == pytest.approx(score)
        assert result.alert_level == expected_level

    def test_partial_alert_thresholds_use_defensive_default(self) -> None:
        # Levels missing from the mapping default to 1.0 (unreachable), and
        # present levels are honoured in high -> medium -> low order.
        result = CrossDomainFrequencyCorrelator(
            alert_thresholds={"high": 0.1},
        ).correlate(
            {
                "environmental": _vector({"schumann_fundamental": 0.5}),
                "space": _vector({"schumann_coupling": 0.5}),
            }
        )
        assert result.alert_level == "high"

    def test_empty_alert_thresholds_fall_back_to_defaults(self) -> None:
        # ``__init__`` uses ``alert_thresholds or {...}``: an explicitly empty
        # mapping is falsy and therefore silently replaced by the defaults.
        correlator = CrossDomainFrequencyCorrelator(alert_thresholds={})
        assert correlator.alert_thresholds == {"low": 0.2, "medium": 0.4, "high": 0.7}
        result = correlator.correlate(
            {
                "environmental": _vector({"schumann_fundamental": 0.95}),
                "space": _vector({"schumann_coupling": 0.95}),
            }
        )
        assert result.alert_level == "high"

    def test_description_truncates_to_top_five_overlaps(self) -> None:
        # Score every environmental and every medical band highly: the two
        # domains' band grids intersect in far more than five places.
        env_scores = dict.fromkeys(
            [
                "infrasound_geophysical",
                "sub_schumann",
                "schumann_fundamental",
                "schumann_harmonic_1",
                "schumann_harmonics_upper",
                "elf_upper",
                "vlf_environmental",
                "atmospheric_noise",
            ],
            0.9,
        )
        med_scores = dict.fromkeys(
            [
                "vlf_hrv",
                "lf_hrv_sympathetic",
                "hf_hrv_parasympathetic",
                "respiratory_cardiac",
                "theta_neural",
                "alpha_neural",
                "beta_neural",
                "gamma_neural_40hz",
                "high_gamma_motor",
            ],
            0.9,
        )
        result = CrossDomainFrequencyCorrelator().correlate(
            {"environmental": _vector(env_scores), "medical": _vector(med_scores)}
        )
        assert result.significant_overlaps > 5
        assert f"Detected {result.significant_overlaps} significant" in result.description
        # Only the top five overlaps are itemised in the summary.
        assert result.description.count("<->") == 5
        assert "requires human assessment" in result.description.lower()

    def test_mixed_payload_styles_interoperate(self) -> None:
        result = CrossDomainFrequencyCorrelator().correlate(
            {
                "environmental": _real_vector({"schumann_fundamental": 0.9}),
                "space": _vector({"schumann_coupling": 0.8}),
            }
        )
        assert result.significant_overlaps == 1


class TestBandOverlapDataclass:
    def test_band_overlap_is_frozen(self) -> None:
        overlap = BandOverlap(
            domain_a="a",
            domain_b="b",
            band_a="x",
            band_b="y",
            overlap_low_hz=1.0,
            overlap_high_hz=2.0,
            score_a=0.5,
            score_b=0.5,
            correlation_strength=0.5,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            overlap.correlation_strength = 0.9  # type: ignore[misc]

    def test_correlation_result_fields_are_typed(self) -> None:
        result = CrossDomainFrequencyCorrelator().correlate({})
        assert isinstance(result.correlation_score, float)
        assert isinstance(result.alert_level, str)
        assert isinstance(result.overlapping_bands, list)
        assert isinstance(result.domain_pairs_checked, int)
        assert isinstance(result.significant_overlaps, int)
        assert isinstance(result.description, str)


# =============================================================================
# Public wiring (ROADMAP row 19 executed decision)
# =============================================================================


class TestPublicWiring:
    """The module is reachable from the detectors public surface."""

    def test_exported_from_detectors_package(self) -> None:
        import omni_mercury_engine.detectors as detectors_pkg

        for name in (
            "BandOverlap",
            "CrossDomainCorrelation",
            "CrossDomainFrequencyCorrelator",
            "DomainBandInfo",
        ):
            exported = getattr(detectors_pkg, name)
            assert exported.__module__ == "omni_mercury_engine.detectors.cross_domain_frequency"
            assert name in detectors_pkg.__all__
