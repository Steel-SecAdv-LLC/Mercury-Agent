"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Personality Engine - Omni-Scalar Shaped Communication

Shapes Mercury's communication style using the Global Omni-Scalar Network.
This creates consistent, principled communication - not performative personality.

Key Scalar Influences:
    - omnibenevolence (0.99): Shapes supportive framing
    - omnitransparency (0.18): Controls reasoning exposure depth
    - omniexplainability (0.9): Influences explanation verbosity
    - omnicompassion (1.30): Affects emotional acknowledgment
    - omnidetermination (1.30): Shapes persistence in communication
    - omnipatience (1.20): Controls response pacing
    - omnivigilance (1.20): Influences alertness expression

Communication is shaped by scalars, not scripted personalities.
The result is consistent, principled voice - genuine, not performed.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CommunicationTone(Enum):
    """Communication tone influenced by scalar configuration."""

    NEUTRAL = "neutral"  # Balanced, factual
    SUPPORTIVE = "supportive"  # Reassuring, empathetic
    DIRECT = "direct"  # Concise, action-oriented
    THOROUGH = "thorough"  # Detailed, comprehensive
    CAUTIOUS = "cautious"  # Careful, uncertainty-aware


class VerbosityLevel(Enum):
    """Verbosity level for explanations."""

    MINIMAL = "minimal"  # Essential information only
    STANDARD = "standard"  # Key details and context
    DETAILED = "detailed"  # Full reasoning exposure
    COMPREHENSIVE = "comprehensive"  # Maximum transparency


@dataclass
class PersonalityProfile:
    """Current personality configuration derived from scalars."""

    tone: CommunicationTone
    verbosity: VerbosityLevel
    reasoning_depth: int  # 1-5 scale
    empathy_level: float  # 0.0-1.0
    persistence_factor: float  # 0.0-1.0
    caution_factor: float  # 0.0-1.0

    # Specific behavioral flags
    acknowledge_uncertainty: bool
    show_reasoning_chain: bool
    include_historical_context: bool
    provide_alternatives: bool
    express_confidence_level: bool

    # Scalar sources (for transparency)
    source_scalars: dict[str, float] = field(default_factory=dict)


@dataclass
class CommunicationModifiers:
    """Modifiers for communication generation."""

    # Opening modifiers
    opening_acknowledgment: str | None = None  # e.g., "I understand this is concerning."

    # Framing modifiers
    confidence_framing: str | None = None  # How to frame confidence
    uncertainty_framing: str | None = None  # How to frame uncertainty

    # Closing modifiers
    support_statement: str | None = None  # e.g., "I'm here to help."
    follow_up_prompt: str | None = None  # e.g., "Would you like more details?"

    # Tone words
    intensity_words: list[str] = field(default_factory=list)
    softening_words: list[str] = field(default_factory=list)


class PersonalityEngine:
    """
    Shapes Communication Using Omni-Scalars.

    This engine derives communication personality from the Global Omni-Scalar
    Network, ensuring consistent, principled voice across all interactions.

    Key Principles:
        1. Personality emerges from scalars, not scripts
        2. Ethical constraints shape honesty, not evasion
        3. Compassion modulates tone, not truth
        4. Transparency scales with complexity
        5. Persistence serves clarity, not annoyance

    Usage:
        engine = PersonalityEngine()

        # Get current personality profile
        profile = engine.get_profile(domain="medical")

        # Get communication modifiers
        modifiers = engine.get_modifiers(
            severity=0.8,
            confidence=0.7,
            anomaly_detected=True
        )

        # Apply to text
        shaped_text = engine.shape_text(
            raw_text="Anomaly detected.",
            profile=profile,
            modifiers=modifiers
        )
    """

    # Scalar thresholds for personality derivation
    HIGH_COMPASSION_THRESHOLD = 1.25
    HIGH_TRANSPARENCY_THRESHOLD = 0.15
    HIGH_EXPLAINABILITY_THRESHOLD = 0.8
    HIGH_DETERMINATION_THRESHOLD = 1.25
    HIGH_VIGILANCE_THRESHOLD = 1.15

    def __init__(self, domain_overrides: dict[str, dict[str, float]] | None = None) -> None:
        """
        Initialize Personality Engine.

        Args:
            domain_overrides: Per-domain scalar overrides
        """
        self._gosnn: Any | None = None  # Lazy-loaded
        self._domain_overrides = domain_overrides or {}

        # Default scalar values (used when GOSNN unavailable)
        self._default_scalars = {
            "omnibenevolence": 0.99,
            "omnitransparency": 0.18,
            "omniexplainability": 0.9,
            "omnicompassion": 1.30,
            "omnidetermination": 1.30,
            "omnipatience": 1.20,
            "omnivigilance": 1.20,
            "omniwisdom": 1.30,
            "omnicourage": 1.30,
            "omnihumility": 1.15,
        }

        self.logger = logging.getLogger(__name__)

    def _get_gosnn(self) -> Any:
        """Lazy-load Global Omni-Scalar Network."""
        if self._gosnn is None:
            try:
                from omni_mercury_engine.core.global_omni_scalar_network import (
                    get_global_scalar_network,
                )

                self._gosnn = get_global_scalar_network()
            except ImportError:
                self.logger.debug("GOSNN not available, using default scalars")
        return self._gosnn

    def _get_scalar(self, name: str, domain: str | None = None) -> float:
        """Get scalar value with domain override support."""
        # Check domain overrides first
        if domain and domain in self._domain_overrides:
            if name in self._domain_overrides[domain]:
                return self._domain_overrides[domain][name]

        # Try GOSNN
        gosnn = self._get_gosnn()
        if gosnn is not None:
            return float(gosnn.get_scalar(name, self._default_scalars.get(name, 1.0)))

        return self._default_scalars.get(name, 1.0)

    def get_profile(self, domain: str | None = None) -> PersonalityProfile:
        """
        Get current personality profile derived from scalars.

        Args:
            domain: Optional domain for context-specific profile

        Returns:
            PersonalityProfile with derived characteristics
        """
        # Gather relevant scalars
        scalars = {
            "omnibenevolence": self._get_scalar("omnibenevolence", domain),
            "omnitransparency": self._get_scalar("omnitransparency", domain),
            "omniexplainability": self._get_scalar("omniexplainability", domain),
            "omnicompassion": self._get_scalar("omnicompassion", domain),
            "omnidetermination": self._get_scalar("omnidetermination", domain),
            "omnipatience": self._get_scalar("omnipatience", domain),
            "omnivigilance": self._get_scalar("omnivigilance", domain),
            "omnihumility": self._get_scalar("omnihumility", domain),
        }

        # Derive tone
        tone = self._derive_tone(scalars)

        # Derive verbosity
        verbosity = self._derive_verbosity(scalars)

        # Calculate derived factors
        reasoning_depth = self._calculate_reasoning_depth(scalars)
        empathy_level = self._calculate_empathy_level(scalars)
        persistence_factor = self._calculate_persistence(scalars)
        caution_factor = self._calculate_caution(scalars)

        # Derive behavioral flags
        acknowledge_uncertainty = scalars["omnihumility"] > 1.1
        show_reasoning_chain = scalars["omnitransparency"] > self.HIGH_TRANSPARENCY_THRESHOLD
        include_historical_context = (
            scalars["omniwisdom"] > 1.2 if "omniwisdom" in scalars else True
        )
        provide_alternatives = scalars["omnipatience"] > 1.15
        express_confidence_level = scalars["omnitransparency"] > 0.1

        return PersonalityProfile(
            tone=tone,
            verbosity=verbosity,
            reasoning_depth=reasoning_depth,
            empathy_level=empathy_level,
            persistence_factor=persistence_factor,
            caution_factor=caution_factor,
            acknowledge_uncertainty=acknowledge_uncertainty,
            show_reasoning_chain=show_reasoning_chain,
            include_historical_context=include_historical_context,
            provide_alternatives=provide_alternatives,
            express_confidence_level=express_confidence_level,
            source_scalars=scalars,
        )

    def _derive_tone(self, scalars: dict[str, float]) -> CommunicationTone:
        """Derive communication tone from scalars."""
        compassion = scalars.get("omnicompassion", 1.0)
        determination = scalars.get("omnidetermination", 1.0)
        transparency = scalars.get("omnitransparency", 0.1)
        humility = scalars.get("omnihumility", 1.0)

        # High compassion prioritizes supportive tone
        if compassion > self.HIGH_COMPASSION_THRESHOLD:
            return CommunicationTone.SUPPORTIVE

        # High determination with moderate compassion is direct
        if determination > self.HIGH_DETERMINATION_THRESHOLD and compassion < 1.25:
            return CommunicationTone.DIRECT

        # High transparency indicates thorough communication
        if transparency > self.HIGH_TRANSPARENCY_THRESHOLD:
            return CommunicationTone.THOROUGH

        # High humility with uncertainty suggests cautious tone
        if humility > 1.2:
            return CommunicationTone.CAUTIOUS

        return CommunicationTone.NEUTRAL

    def _derive_verbosity(self, scalars: dict[str, float]) -> VerbosityLevel:
        """Derive verbosity level from scalars."""
        explainability = scalars.get("omniexplainability", 0.5)
        transparency = scalars.get("omnitransparency", 0.1)
        patience = scalars.get("omnipatience", 1.0)

        # High explainability and transparency = comprehensive
        if (
            explainability > self.HIGH_EXPLAINABILITY_THRESHOLD
            and transparency > self.HIGH_TRANSPARENCY_THRESHOLD
        ):
            return VerbosityLevel.COMPREHENSIVE

        # High explainability = detailed
        if explainability > self.HIGH_EXPLAINABILITY_THRESHOLD:
            return VerbosityLevel.DETAILED

        # Moderate patience allows standard verbosity
        if patience > 1.1:
            return VerbosityLevel.STANDARD

        return VerbosityLevel.MINIMAL

    def _calculate_reasoning_depth(self, scalars: dict[str, float]) -> int:
        """Calculate reasoning chain depth to expose (1-5)."""
        transparency = scalars.get("omnitransparency", 0.1)
        explainability = scalars.get("omniexplainability", 0.5)

        # Combine factors
        combined = transparency * 5 + explainability
        depth = int(min(5, max(1, combined)))

        return depth

    def _calculate_empathy_level(self, scalars: dict[str, float]) -> float:
        """Calculate empathy level (0.0-1.0)."""
        compassion = scalars.get("omnicompassion", 1.0)
        benevolence = scalars.get("omnibenevolence", 0.5)

        # Normalize to 0-1 range
        empathy = ((compassion - 1.0) / 0.5 + benevolence) / 2
        return min(1.0, max(0.0, empathy))

    def _calculate_persistence(self, scalars: dict[str, float]) -> float:
        """Calculate persistence factor (0.0-1.0)."""
        determination = scalars.get("omnidetermination", 1.0)
        vigilance = scalars.get("omnivigilance", 1.0)

        # Normalize
        persistence = ((determination - 1.0) + (vigilance - 1.0)) / 0.6
        return min(1.0, max(0.0, persistence))

    def _calculate_caution(self, scalars: dict[str, float]) -> float:
        """Calculate caution factor (0.0-1.0)."""
        humility = scalars.get("omnihumility", 1.0)

        # Higher humility = more caution about certainty claims
        return min(1.0, max(0.0, (humility - 1.0) / 0.3))

    def get_modifiers(
        self,
        severity: float,
        confidence: float,
        anomaly_detected: bool,
        profile: PersonalityProfile | None = None,
        domain: str | None = None,
    ) -> CommunicationModifiers:
        """
        Get context-specific communication modifiers.

        Args:
            severity: Detection severity (0-1)
            confidence: Detection confidence (0-1)
            anomaly_detected: Whether anomaly was detected
            profile: Pre-computed personality profile
            domain: Domain context

        Returns:
            CommunicationModifiers for text shaping
        """
        if profile is None:
            profile = self.get_profile(domain)

        modifiers = CommunicationModifiers()

        # Opening acknowledgment based on empathy and severity
        if profile.empathy_level > 0.5 and severity > 0.7 and anomaly_detected:
            modifiers.opening_acknowledgment = "I understand this finding may be concerning."
        elif profile.empathy_level > 0.7:
            modifiers.opening_acknowledgment = "I want to provide you with clear information."

        # Confidence framing
        if profile.express_confidence_level:
            if confidence > 0.9:
                modifiers.confidence_framing = "with high confidence"
            elif confidence > 0.7:
                modifiers.confidence_framing = "with reasonable confidence"
            elif confidence > 0.5:
                modifiers.confidence_framing = "with moderate confidence"
            else:
                modifiers.confidence_framing = "with limited confidence"

        # Uncertainty framing
        if profile.acknowledge_uncertainty and confidence < 0.7:
            if confidence < 0.5:
                modifiers.uncertainty_framing = (
                    "Note: Uncertainty is high. This finding should be verified."
                )
            else:
                modifiers.uncertainty_framing = "Some uncertainty remains in this assessment."

        # Support statement based on compassion
        if profile.tone == CommunicationTone.SUPPORTIVE:
            modifiers.support_statement = "I'm here to help clarify any questions."

        # Follow-up prompt based on patience
        if profile.provide_alternatives:
            modifiers.follow_up_prompt = "Would you like me to explore alternative interpretations?"

        # Intensity words based on tone
        if profile.tone == CommunicationTone.DIRECT:
            modifiers.intensity_words = ["immediately", "requires", "must", "critical"]
        elif profile.tone == CommunicationTone.CAUTIOUS:
            modifiers.intensity_words = ["potentially", "suggests", "may indicate"]

        # Softening words based on empathy
        if profile.empathy_level > 0.6:
            modifiers.softening_words = [
                "appears to",
                "suggests",
                "indicates",
                "points toward",
            ]

        return modifiers

    def shape_text(
        self,
        raw_text: str,
        profile: PersonalityProfile,
        modifiers: CommunicationModifiers,
    ) -> str:
        """
        Shape raw text according to personality and modifiers.

        Args:
            raw_text: Original text to shape
            profile: Personality profile
            modifiers: Communication modifiers

        Returns:
            Shaped text with personality applied
        """
        parts = []

        # Add opening acknowledgment
        if modifiers.opening_acknowledgment:
            parts.append(modifiers.opening_acknowledgment)

        # Add main text with softening if appropriate
        shaped_main = raw_text
        if modifiers.softening_words and profile.tone in (
            CommunicationTone.SUPPORTIVE,
            CommunicationTone.CAUTIOUS,
        ):
            # Apply light softening (don't overdo it)
            shaped_main = shaped_main.replace("detected", "appears to be detected")
            shaped_main = shaped_main.replace("Detected", "Appears to be detected")

        parts.append(shaped_main)

        # Add confidence framing
        if modifiers.confidence_framing and "confidence" not in raw_text.lower():
            parts.append(f"This assessment is made {modifiers.confidence_framing}.")

        # Add uncertainty framing
        if modifiers.uncertainty_framing:
            parts.append(modifiers.uncertainty_framing)

        # Add support statement
        if modifiers.support_statement:
            parts.append(modifiers.support_statement)

        # Add follow-up prompt
        if modifiers.follow_up_prompt:
            parts.append(modifiers.follow_up_prompt)

        return " ".join(parts)

    def get_greeting(self, domain: str | None = None) -> str:
        """
        Get appropriate greeting based on personality.

        This is NOT "How can I help you today?" engagement bait.
        It's context-appropriate acknowledgment.

        Args:
            domain: Domain context

        Returns:
            Appropriate greeting
        """
        profile = self.get_profile(domain)

        if profile.tone == CommunicationTone.DIRECT:
            return "Mercury Agent ready. Awaiting input."
        elif profile.tone == CommunicationTone.SUPPORTIVE:
            return "Mercury Agent online. Ready to analyze your data."
        elif profile.tone == CommunicationTone.THOROUGH:
            return (
                "Mercury Agent initialized. "
                "Full reasoning transparency enabled. Ready for analysis."
            )
        elif profile.tone == CommunicationTone.CAUTIOUS:
            return "Mercury Agent active. " "Note: All findings include uncertainty quantification."
        return "Mercury Agent operational."

    def get_uncertainty_statement(
        self,
        confidence: float,
        epistemic: float = 0.0,
        aleatoric: float = 0.0,
        domain: str | None = None,
    ) -> str:
        """
        Generate personality-appropriate uncertainty statement.

        Args:
            confidence: Overall confidence
            epistemic: Epistemic uncertainty
            aleatoric: Aleatoric uncertainty
            domain: Domain context

        Returns:
            Uncertainty statement shaped by personality
        """
        profile = self.get_profile(domain)

        if not profile.acknowledge_uncertainty:
            return ""

        parts = []

        # Base confidence statement
        if confidence < 0.3:
            parts.append(
                f"Confidence is low ({confidence:.0%}). " "Multiple interpretations are possible."
            )
        elif confidence < 0.5:
            parts.append(
                f"Moderate uncertainty ({confidence:.0%} confidence). " "Verification recommended."
            )
        elif confidence < 0.7:
            parts.append(
                f"Reasonable confidence ({confidence:.0%}), " "though some uncertainty remains."
            )

        # Decomposition if thorough
        if profile.verbosity in (VerbosityLevel.DETAILED, VerbosityLevel.COMPREHENSIVE):
            if epistemic > 0.1:
                parts.append(f"Model uncertainty (epistemic): {epistemic:.0%}.")
            if aleatoric > 0.1:
                parts.append(f"Data variability (aleatoric): {aleatoric:.0%}.")

        return " ".join(parts)

    def get_statistics(self) -> dict[str, Any]:
        """Get personality engine statistics."""
        gosnn = self._get_gosnn()
        return {
            "gosnn_connected": gosnn is not None,
            "domain_overrides_configured": list(self._domain_overrides.keys()),
            "default_scalars": self._default_scalars,
        }
