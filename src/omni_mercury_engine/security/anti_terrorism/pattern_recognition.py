# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Anti-Terrorism Pattern Recognition.

Stages reported behaviour against the 42 observable indicators of *US Violent
Extremist Mobilization Indicators* (NCTC, FBI and DHS, 2025 edition; the
numbering matches the 2021 edition). The booklet groups them into three
behaviour stages, mobilization, preparation and motivation, and types them as
finance, ideology, intent, relationship, tactics or travel.

The stage comes only from recognised indicators: a caller's ``threat_score``
sets the confidence, never the stage, because a number is not an observed
behaviour. Only recognised indicator codes are reported as threat indicators;
anything else is returned separately and never scored. Strings naming a legally
protected characteristic are also listed as excluded (a best-effort screen of
category words, not of every identity term), following the booklet's caveat
that action "should never be taken solely on the basis of constitutionally
protected activities" or of such characteristics. Every result is a referral
for trained human review, not a determination.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

from omni_mercury_engine.detectors.provenance import DecisionSource

# code -> (booklet number, booklet stage, indicator type)
MOBILIZATION_INDICATORS: dict[str, tuple[int, str, str]] = {
    "violent_extremist_travel": (1, "mobilization", "travel"),
    "threatening_or_resisting_law_enforcement": (2, "mobilization", "intent"),
    "martyrdom_or_final_statement": (3, "mobilization", "intent"),
    "attack_dry_run_or_target_access": (4, "mobilization", "tactics"),
    "stated_attack_details": (5, "mobilization", "intent"),
    "unusual_asset_disposal": (6, "mobilization", "finance"),
    "unusual_goodbyes": (7, "mobilization", "relationship"),
    "communicated_violent_intent": (8, "preparation", "intent"),
    "explosives_research_or_testing": (9, "preparation", "tactics"),
    "planning_travel_abroad_to_join": (10, "preparation", "travel"),
    "planning_domestic_travel_for_violence": (11, "preparation", "travel"),
    "justification_for_planned_violence": (12, "preparation", "ideology"),
    "suspicious_tactical_equipment_purchase": (13, "preparation", "tactics"),
    "operationally_secure_splinter_group": (14, "preparation", "relationship"),
    "target_surveillance": (15, "preparation", "tactics"),
    "physical_concealment_for_attack": (16, "preparation", "tactics"),
    "online_concealment_for_attack": (17, "preparation", "tactics"),
    "seeking_violent_extremist_contact": (18, "preparation", "relationship"),
    "explosive_precursor_acquisition": (19, "preparation", "tactics"),
    "weapons_acquisition_for_crime": (20, "preparation", "tactics"),
    "weapons_training_for_crime": (21, "preparation", "tactics"),
    "suspicious_travel_activity": (22, "preparation", "travel"),
    "unexplained_resources_with_violent_extremists": (23, "preparation", "finance"),
    "violent_extremist_group_membership": (24, "preparation", "relationship"),
    "target_or_tactic_research": (25, "preparation", "tactics"),
    "exploiting_sensitive_access": (26, "preparation", "tactics"),
    "seeking_technical_expertise_for_violence": (27, "preparation", "tactics"),
    "intent_to_harm_if_confronted": (28, "motivation", "intent"),
    "willing_to_die_or_seek_notoriety_through_violence": (29, "motivation", "intent"),
    "threat_against_physical_target": (30, "motivation", "intent"),
    "threat_against_individuals": (31, "motivation", "intent"),
    "violent_extremist_content_for_crime": (32, "motivation", "ideology"),
    "posing_with_weapons_and_extremist_imagery": (33, "motivation", "ideology"),
    "violence_accepted_as_necessary": (34, "motivation", "intent"),
    "radicalizing_others_to_violence": (35, "motivation", "relationship"),
    "praising_or_studying_past_attacks": (36, "motivation", "ideology"),
    "crisis_narratives_justifying_violence": (37, "motivation", "ideology"),
    "outbursts_while_advocating_violence": (38, "motivation", "relationship"),
    "multiple_violent_ideologies": (39, "motivation", "ideology"),
    "rejecting_nonviolent_voices": (40, "motivation", "ideology"),
    "hardened_speech_after_catalyst": (41, "motivation", "ideology"),
    "isolation_citing_violent_doctrine": (42, "motivation", "relationship"),
}

_CODE_BY_NUMBER = {number: code for code, (number, _, _) in MOBILIZATION_INDICATORS.items()}

# Direct threats of violence against a person, place or law enforcement. Alone
# among motivation indicators these raise ``threat_detected``: the others may be
# lawful or constitutionally protected and "should not serve as the sole basis
# for any investigative activity" (booklet, motivation section).
_DIRECT_THREATS = frozenset(
    {
        "intent_to_harm_if_confronted",
        "threat_against_physical_target",
        "threat_against_individuals",
    }
)

# Category words for the legally protected characteristics in the booklet's caveat.
_PROTECTED = re.compile(
    r"\b(races?|racial|ages?|aged|ethnic\w*|national origins?|nationalit\w*|religio\w*|"
    r"faiths?|sex|sexes|genders?|gender identit\w*|sexual orientations?)\b"
)


@dataclass
class TerrorismThreatResult:
    """Result from terrorism pattern detection."""

    threat_detected: bool
    radicalization_stage: str
    confidence: float
    threat_indicators: list[str]
    recommended_actions: list[str]
    matched_indicators: list[str] = field(default_factory=list)
    excluded_indicators: list[str] = field(default_factory=list)
    unrecognised_indicators: list[str] = field(default_factory=list)


class TerrorismPatternDetector:
    """Terrorism Pattern Detector for CI.

    Stages OSINT and COMINT reports by the most advanced recognised
    mobilization indicator. The legacy stage names map to the booklet:

    - ``pre_radicalization``: no recognised indicator
    - ``identification``: motivation indicators of ideology or relationship
      type only (these may be lawful and are never a basis for action alone)
    - ``indoctrination``: motivation indicators of intent type (threats,
      acceptance of violence)
    - ``action_planning``: preparation indicators (weeks before an attack)
    - ``imminent_action``: mobilization indicators (days or hours before)

    ``threat_detected`` needs a preparation or mobilization indicator, or a
    direct threat (#28, #30, #31), from a report that states a source confidence
    (``threat_score`` > 0); an unscored report is staged but flagged for
    corroboration instead. Indicators are the codes in
    :data:`MOBILIZATION_INDICATORS` (case, spaces and hyphens are ignored) or
    the booklet's numbers 1-42; other strings go to ``unrecognised_indicators``
    and are not scored, so free-text reports must be coded first.
    """

    decision_source = DecisionSource(
        kind="experimental",
        description=(
            "Reports are staged by the most advanced recognised mobilization indicator; the "
            "indicator table and stage grouping follow the NCTC/FBI/DHS booklet cited in the "
            "module, which was not re-read. threat_detected needs a preparation or mobilization "
            "indicator, or a direct threat (#28, #30, #31), from a source with stated reliability; "
            "confidence is a noisy-OR over source reliabilities. The actionable cut and the "
            "combination rule are the detector's own, and every result is a referral for human "
            "review."
        ),
    )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        self.radicalization_stages = [
            "pre_radicalization",
            "identification",
            "indoctrination",
            "action_planning",
            "imminent_action",
        ]

        self.logger.info("TerrorismPatternDetector initialized")

    def detect_radicalization(
        self,
        osint_data: dict[str, Any] | None = None,
        comint_data: dict[str, Any] | None = None,
    ) -> TerrorismThreatResult:
        """Stage the behaviour reported in intelligence data.

        Args:
            osint_data: Open-source intelligence: ``indicators`` (codes) and
                ``threat_score``, the source's confidence in [0, 1]
            comint_data: Communications intelligence, same keys

        Returns:
            TerrorismThreatResult with threat assessment
        """
        unrecognised: list[str] = []
        excluded: list[str] = []
        evidence: list[tuple[str, int, float]] = []
        for source, report in enumerate((osint_data, comint_data)):
            if not report:
                continue
            reliability = _unit_score(report.get("threat_score"))
            for indicator in _as_list(report.get("indicators")):
                code = _indicator_code(indicator)
                if code in MOBILIZATION_INDICATORS:
                    evidence.append((code, source, reliability))
                elif _PROTECTED.search(re.sub(r"[_\-]+", " ", _text(indicator)).lower()):
                    excluded.append(_text(indicator))
                else:
                    unrecognised.append(_text(indicator))

        codes = list(dict.fromkeys(code for code, _, _ in evidence))
        stage = self._classify_radicalization_stage(codes)
        rank = self.radicalization_stages.index(stage)

        # Noisy-OR over the reliabilities of the reports behind the stage.
        at_stage = {src: r for code, src, r in evidence if self._stage_of(code) == stage}
        confidence = 1.0 - math.prod(1.0 - r for r in at_stage.values()) if at_stage else 0.0

        actionable = rank >= self.radicalization_stages.index("action_planning") or any(
            code in _DIRECT_THREATS for code in codes
        )
        threat_detected = actionable and confidence > 0.0

        actions = self._recommend_actions(stage, confidence)
        if actionable and confidence == 0.0:
            actions.insert(
                0, "Source reliability not stated: corroborate the report before any action"
            )
            actions = [a for a in actions if not a.startswith("URGENT")]
        if unrecognised and not evidence:
            actions.append("No recognised behavioural indicator: a score alone is not evidence")
        if excluded:
            self.logger.warning(
                f"Excluded {len(excluded)} indicator(s) naming a protected characteristic"
            )

        result = TerrorismThreatResult(
            threat_detected=threat_detected,
            radicalization_stage=stage,
            confidence=confidence,
            threat_indicators=codes[:10],
            recommended_actions=actions,
            matched_indicators=[
                f"{code} ({MOBILIZATION_INDICATORS[code][1]} #{MOBILIZATION_INDICATORS[code][0]})"
                for code in codes
            ],
            excluded_indicators=excluded,
            unrecognised_indicators=unrecognised,
        )

        if threat_detected:
            self.logger.warning(
                f"Mobilization indicators at stage {stage} (confidence={confidence:.2f}); "
                "refer for human review"
            )

        return result

    def _stage_of(self, code: str) -> str:
        """Legacy stage name of one recognised indicator code."""
        _number, booklet_stage, indicator_type = MOBILIZATION_INDICATORS[code]
        if booklet_stage == "mobilization":
            return "imminent_action"
        if booklet_stage == "preparation":
            return "action_planning"
        return "indoctrination" if indicator_type == "intent" else "identification"

    def _classify_radicalization_stage(self, indicator_codes: list[str]) -> str:
        """Most advanced stage among recognised indicator codes."""
        stages = [self._stage_of(code) for code in indicator_codes]
        if not stages:
            return "pre_radicalization"
        return max(stages, key=self.radicalization_stages.index)

    def _recommend_actions(self, stage: str, threat_score: float) -> list[str]:
        """Recommend actions; each one starts with trained human review."""
        actions = []

        if stage == "imminent_action":
            actions.append(
                "URGENT: Human threat-assessment review now; notify law enforcement "
                "immediately if the behaviour is confirmed"
            )
            actions.append("Refer to counter-terrorism and behavioural threat-assessment teams")
            actions.append("Protect identified targets and the people at risk")

        elif stage == "action_planning":
            actions.append("Refer to a behavioural threat-assessment team for case management")
            actions.append("Coordinate with federal counter-terrorism partners after human review")
            actions.append("Continue monitoring the reported behaviours within legal authority")

        elif stage == "indoctrination":
            actions.append("Assess any expressed intent with a trained threat-assessment team")
            actions.append("Warn and protect any person or place threatened")
            actions.append(
                "Offer off-ramps: community-based prevention and counter-narrative programs"
            )

        elif stage == "identification":
            actions.append(
                "Beliefs and associations alone may be constitutionally protected: "
                "take no action on them alone"
            )
            actions.append("Continue monitoring for preparation or mobilization behaviour")

        else:
            actions.append("Continue monitoring for escalation")

        return actions


def _text(indicator: Any) -> str:
    """An indicator as text; bytes are decoded."""
    if isinstance(indicator, bytes):
        return indicator.decode("utf-8", errors="replace")
    return str(indicator)


def _indicator_code(indicator: Any) -> str:
    """Normalised code of an indicator: a code, a booklet number, or other text."""
    if isinstance(indicator, float) and indicator.is_integer():
        indicator = int(indicator)
    code = re.sub(r"[^a-z0-9]+", "_", _text(indicator).lower()).strip("_")
    if code.isdigit():
        return _CODE_BY_NUMBER.get(int(code), code)
    return code


def _unit_score(value: Any) -> float:
    """A source confidence clipped to [0, 1]; missing or invalid is 0."""
    if isinstance(value, bool):
        return 0.0
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(max(score, 0.0), 1.0) if math.isfinite(score) else 0.0


def _as_list(value: Any) -> list[Any]:
    """Indicators as a list; a single string counts as one indicator."""
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]
