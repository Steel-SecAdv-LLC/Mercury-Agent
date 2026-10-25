# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for Anti-Terrorism Pattern Recognition module."""

import pytest

from omni_mercury_engine.security.anti_terrorism.pattern_recognition import (
    MOBILIZATION_INDICATORS,
    TerrorismPatternDetector,
    TerrorismThreatResult,
)


class TestTerrorismThreatResult:
    """Tests for TerrorismThreatResult dataclass."""

    def test_init(self) -> None:
        """Test initialization."""
        result = TerrorismThreatResult(
            threat_detected=True,
            radicalization_stage="indoctrination",
            confidence=0.75,
            threat_indicators=["indicator1", "indicator2"],
            recommended_actions=["action1"],
        )
        assert result.threat_detected
        assert result.radicalization_stage == "indoctrination"
        assert result.confidence == 0.75
        assert len(result.threat_indicators) == 2
        assert len(result.recommended_actions) == 1


class TestTerrorismPatternDetector:
    """Tests for TerrorismPatternDetector class."""

    def test_init_default(self) -> None:
        """Test initialization with default config."""
        detector = TerrorismPatternDetector()
        assert detector.config == {}
        assert len(detector.radicalization_stages) == 5

    def test_init_with_config(self) -> None:
        """Test initialization with custom config."""
        config = {"threshold": 0.5}
        detector = TerrorismPatternDetector(config=config)
        assert detector.config == config

    def test_detect_radicalization_no_data(self) -> None:
        """Test detection with no data."""
        detector = TerrorismPatternDetector()
        result = detector.detect_radicalization()
        assert isinstance(result, TerrorismThreatResult)
        assert not result.threat_detected
        assert result.radicalization_stage == "pre_radicalization"

    def test_detect_radicalization_low_threat(self) -> None:
        """An unrecognised indicator sets no stage, whatever the score."""
        detector = TerrorismPatternDetector()
        osint_data = {"threat_score": 0.2, "indicators": ["suspicious_activity"]}
        result = detector.detect_radicalization(osint_data=osint_data)
        assert not result.threat_detected
        assert result.radicalization_stage == "pre_radicalization"

    def test_detect_radicalization_identification_stage(self) -> None:
        """Ideology-type motivation indicators alone are not a threat."""
        detector = TerrorismPatternDetector()
        osint_data = {"threat_score": 0.4, "indicators": ["violent_extremist_content_for_crime"]}
        result = detector.detect_radicalization(osint_data=osint_data)
        assert not result.threat_detected
        assert result.radicalization_stage == "identification"

    def test_detect_radicalization_indoctrination_stage(self) -> None:
        """Intent-type motivation indicators (threats) are."""
        detector = TerrorismPatternDetector()
        osint_data = {"threat_score": 0.6, "indicators": ["threat_against_individuals"]}
        result = detector.detect_radicalization(osint_data=osint_data)
        assert result.threat_detected
        assert result.radicalization_stage == "indoctrination"

    def test_detect_radicalization_action_planning_stage(self) -> None:
        """Test detection at action planning stage."""
        detector = TerrorismPatternDetector()
        osint_data = {"threat_score": 0.8, "indicators": ["target_surveillance"]}
        result = detector.detect_radicalization(osint_data=osint_data)
        assert result.threat_detected
        assert result.radicalization_stage == "action_planning"

    def test_detect_radicalization_imminent_action_stage(self) -> None:
        """Test detection at imminent action stage."""
        detector = TerrorismPatternDetector()
        osint_data = {"threat_score": 0.95, "indicators": ["attack_dry_run_or_target_access"]}
        result = detector.detect_radicalization(osint_data=osint_data)
        assert result.threat_detected
        assert result.radicalization_stage == "imminent_action"

    def test_detect_radicalization_with_comint(self) -> None:
        """Test detection with COMINT data."""
        detector = TerrorismPatternDetector()
        comint_data = {
            "threat_score": 0.7,
            "indicators": ["encrypted_comms", "communicated_violent_intent"],
        }
        result = detector.detect_radicalization(comint_data=comint_data)
        assert result.threat_detected
        assert result.threat_indicators == ["communicated_violent_intent"]
        assert result.unrecognised_indicators == ["encrypted_comms"]
        assert result.matched_indicators == ["communicated_violent_intent (preparation #8)"]

    def test_detect_radicalization_combined_data(self) -> None:
        """Independent reports at the same stage corroborate (noisy-OR)."""
        detector = TerrorismPatternDetector()
        osint_data = {"threat_score": 0.5, "indicators": ["target_surveillance"]}
        comint_data = {"threat_score": 0.8, "indicators": ["weapons_acquisition_for_crime"]}
        result = detector.detect_radicalization(osint_data=osint_data, comint_data=comint_data)
        assert result.threat_detected
        assert result.confidence == pytest.approx(1 - 0.5 * 0.2)
        assert "target_surveillance" in result.threat_indicators
        assert "weapons_acquisition_for_crime" in result.threat_indicators

    def test_classify_radicalization_stage(self) -> None:
        """The most advanced recognised indicator sets the stage."""
        detector = TerrorismPatternDetector()
        classify = detector._classify_radicalization_stage
        assert classify([]) == "pre_radicalization"
        assert classify(["rejecting_nonviolent_voices"]) == "identification"
        assert classify(["violence_accepted_as_necessary"]) == "indoctrination"
        assert classify(["rejecting_nonviolent_voices", "target_surveillance"]) == (
            "action_planning"
        )
        assert classify(["target_surveillance", "unusual_goodbyes"]) == "imminent_action"

    def test_recommend_actions_imminent(self) -> None:
        """Test action recommendations for imminent threat."""
        detector = TerrorismPatternDetector()
        actions = detector._recommend_actions("imminent_action", 0.95)
        assert any("law enforcement" in a.lower() for a in actions)
        assert any("counter-terrorism" in a.lower() for a in actions)

    def test_recommend_actions_planning(self) -> None:
        """Test action recommendations for action planning."""
        detector = TerrorismPatternDetector()
        actions = detector._recommend_actions("action_planning", 0.8)
        assert any("monitoring" in a.lower() for a in actions)
        assert any("federal" in a.lower() for a in actions)

    def test_recommend_actions_indoctrination(self) -> None:
        """Test action recommendations for indoctrination."""
        detector = TerrorismPatternDetector()
        actions = detector._recommend_actions("indoctrination", 0.6)
        assert any("community" in a.lower() for a in actions)
        assert any("counter-narrative" in a.lower() for a in actions)

    def test_recommend_actions_low_threat(self) -> None:
        """Test action recommendations for low threat."""
        detector = TerrorismPatternDetector()
        actions = detector._recommend_actions("pre_radicalization", 0.2)
        assert any("monitoring" in a.lower() for a in actions)

    def test_indicator_limit(self) -> None:
        """Test that indicators are limited to 10."""
        detector = TerrorismPatternDetector()
        osint_data = {
            "threat_score": 0.6,
            "indicators": [f"indicator_{i}" for i in range(20)],
        }
        result = detector.detect_radicalization(osint_data=osint_data)
        assert len(result.threat_indicators) <= 10


class TestBehaviourBasedStaging:
    """The stage follows reported behaviour, not a caller's score.

    The released detector bucketed ``max(threat_score)`` and ignored the
    indicators, so a benign report at 0.95 was ``imminent_action``.
    """

    def test_score_alone_does_not_stage(self) -> None:
        """Fails on the released detector (``imminent_action``)."""
        result = TerrorismPatternDetector().detect_radicalization(
            osint_data={"threat_score": 0.95, "indicators": ["school bake sale"]}
        )
        assert result.radicalization_stage == "pre_radicalization"
        assert not result.threat_detected
        assert result.confidence == 0.0
        assert any("score alone" in a for a in result.recommended_actions)

    def test_behaviour_stages_even_from_a_low_confidence_source(self) -> None:
        """Fails on the released detector (``pre_radicalization`` at score 0)."""
        result = TerrorismPatternDetector().detect_radicalization(
            osint_data={"threat_score": 0.0, "indicators": ["Stated attack details"]}
        )
        assert result.radicalization_stage == "imminent_action"
        assert result.confidence == 0.0

    def test_unscored_report_is_staged_but_not_actioned(self) -> None:
        """A report without a stated source confidence needs corroboration first."""
        result = TerrorismPatternDetector().detect_radicalization(
            osint_data={"indicators": ["unusual_goodbyes"]}
        )
        assert result.radicalization_stage == "imminent_action"
        assert not result.threat_detected
        assert result.recommended_actions[0].startswith("Source reliability not stated")
        assert not any(a.startswith("URGENT") for a in result.recommended_actions)

    @pytest.mark.parametrize(
        ("code", "detected"),
        [
            ("violence_accepted_as_necessary", False),  # 34: may be protected speech
            ("willing_to_die_or_seek_notoriety_through_violence", False),  # 29
            ("threat_against_individuals", True),  # 31: a direct threat
            ("threat_against_physical_target", True),  # 30
            ("intent_to_harm_if_confronted", True),  # 28
        ],
    )
    def test_only_direct_threats_raise_a_motivation_alarm(self, code: str, detected: bool) -> None:
        result = TerrorismPatternDetector().detect_radicalization(
            osint_data={"threat_score": 0.3, "indicators": [code]}
        )
        assert result.radicalization_stage == "indoctrination"
        assert result.threat_detected is detected

    def test_identity_terms_are_never_threat_indicators(self) -> None:
        identity = ["Muslim convert", "attends mosque", "Somali immigrant", "aged 19", "races"]
        result = TerrorismPatternDetector().detect_radicalization(
            osint_data={
                "threat_score": 0.9,
                "indicators": [*identity, "threat_against_individuals"],
            }
        )
        assert result.threat_indicators == ["threat_against_individuals"]
        assert result.excluded_indicators == ["aged 19", "races"]
        assert result.unrecognised_indicators == identity[:3]

    def test_recognised_codes_are_not_crowded_out(self) -> None:
        noise = [f"note {i}" for i in range(15)]
        result = TerrorismPatternDetector().detect_radicalization(
            osint_data={"threat_score": 0.5, "indicators": [*noise, "target_surveillance"]}
        )
        assert result.threat_indicators == ["target_surveillance"]

    def test_numeric_and_bytes_indicators(self) -> None:
        result = TerrorismPatternDetector().detect_radicalization(
            osint_data={"threat_score": 0.5, "indicators": [15.0, b"unusual_goodbyes"]}
        )
        assert result.threat_indicators == ["target_surveillance", "unusual_goodbyes"]

    def test_boolean_score_is_not_a_confidence(self) -> None:
        result = TerrorismPatternDetector().detect_radicalization(
            osint_data={"threat_score": True, "indicators": ["target_surveillance"]}
        )
        assert result.confidence == 0.0 and not result.threat_detected

    def test_protected_characteristics_are_excluded(self) -> None:
        """Identity is never an indicator (booklet caveat)."""
        result = TerrorismPatternDetector().detect_radicalization(
            osint_data={
                "threat_score": 0.9,
                "indicators": ["Religion: Islam", "national_origin", "rejecting_nonviolent_voices"],
            }
        )
        assert result.excluded_indicators == ["Religion: Islam", "national_origin"]
        assert result.threat_indicators == ["rejecting_nonviolent_voices"]
        assert result.radicalization_stage == "identification"
        assert not result.threat_detected

    def test_booklet_numbers_are_accepted(self) -> None:
        result = TerrorismPatternDetector().detect_radicalization(
            comint_data={"threat_score": 0.5, "indicators": [15, "4"]}
        )
        assert result.radicalization_stage == "imminent_action"
        assert result.matched_indicators == [
            "target_surveillance (preparation #15)",
            "attack_dry_run_or_target_access (mobilization #4)",
        ]
        assert result.confidence == pytest.approx(0.5)

    def test_one_report_counts_once(self) -> None:
        result = TerrorismPatternDetector().detect_radicalization(
            osint_data={"threat_score": 0.6, "indicators": ["target_surveillance", 20]}
        )
        assert result.confidence == pytest.approx(0.6)

    def test_indicator_table_matches_the_booklet(self) -> None:
        numbers = sorted(n for n, _, _ in MOBILIZATION_INDICATORS.values())
        assert numbers == list(range(1, 43))
        stages = {code: stage for code, (_, stage, _) in MOBILIZATION_INDICATORS.items()}
        assert [
            n for c, (n, _, _) in MOBILIZATION_INDICATORS.items() if stages[c] == "mobilization"
        ] == list(range(1, 8))
        assert [
            n for c, (n, _, _) in MOBILIZATION_INDICATORS.items() if stages[c] == "preparation"
        ] == list(range(8, 28))

    def test_indicator_types_match_the_booklet(self) -> None:
        intent = {n for n, _, kind in MOBILIZATION_INDICATORS.values() if kind == "intent"}
        assert intent == {2, 3, 5, 8, 28, 29, 30, 31, 34}
        finance = {n for n, _, kind in MOBILIZATION_INDICATORS.values() if kind == "finance"}
        assert finance == {6, 23}
        travel = {n for n, _, kind in MOBILIZATION_INDICATORS.values() if kind == "travel"}
        assert travel == {1, 10, 11, 22}

    def test_invalid_scores_count_as_no_confidence(self) -> None:
        result = TerrorismPatternDetector().detect_radicalization(
            osint_data={"threat_score": float("nan"), "indicators": ["target_surveillance"]}
        )
        assert result.radicalization_stage == "action_planning"
        assert result.confidence == 0.0
