# Copyright (C) 2025 Steel Security Advisors LLC
"""Narrative Engine - Truth-Dense Communication Synthesis.

Transforms detection results into human-readable communication while maintaining
maximum truth density. This is NOT about engagement - it's about transparency.

Philosophy:
    "An Agent that does not engage or communicate to retain users like almost
    all LLMs, but provides transparency and truth in every response."

Key Features:
    - Converts detection results to natural language with full reasoning chains
    - Shapes communication style via omni-scalar personality parameters
    - Surfaces memory context for historical awareness
    - Quantifies uncertainty transparently
    - Provides actionable recommendations based on confidence levels

Integration Points:
    - CognitiveOrchestrator: Receives analysis results with reasoning chains
    - GlobalOmniScalarNetwork: Gets personality scalars for tone shaping
    - NeuralMemoryLayer: Retrieves historical context
    - LLMAdapter: Optional enhancement via language model

Not performative "aliveness" - genuine transparency through structure.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from omni_mercury_engine.core.centralized_constants import ETHICAL

logger = logging.getLogger(__name__)


class NarrativeStyle(Enum):
    """Communication style modes shaped by omni-scalars."""

    CLINICAL = "clinical"  # Precise, minimal, data-focused
    EXPLANATORY = "explanatory"  # Detailed reasoning, educational
    URGENT = "urgent"  # Critical alerts, action-oriented
    ANALYTICAL = "analytical"  # Deep dive, hypothesis exploration
    SUPPORTIVE = "supportive"  # Reassuring, context-providing


class ConfidenceLevel(Enum):
    """Confidence classification for transparency."""

    VERY_LOW = "very_low"  # <30% - High uncertainty, multiple hypotheses
    LOW = "low"  # 30-50% - Uncertain, requires verification
    MODERATE = "moderate"  # 50-70% - Reasonable confidence
    HIGH = "high"  # 70-90% - Strong evidence
    VERY_HIGH = "very_high"  # >90% - Near-certain


@dataclass
class ReasoningChainNarrative:
    """Verbalized reasoning chain for transparency."""

    steps: list[str]
    hypotheses_considered: list[str]
    hypotheses_rejected: list[str]
    final_conclusion: str
    confidence_rationale: str


@dataclass
class NarrativeResult:
    """Complete narrative output with full transparency."""

    # Primary communication
    summary: str  # One-line summary
    detailed_explanation: str  # Full explanation with reasoning
    reasoning_chain: ReasoningChainNarrative | None  # Verbalized reasoning

    # Transparency metrics
    confidence_level: ConfidenceLevel
    confidence_score: float
    uncertainty_disclosure: str  # Explicit uncertainty statement

    # Historical context
    historical_context: str | None  # Memory-informed context
    similar_past_events: list[str]  # References to past detections

    # Actionable output
    recommendations: list[str]
    urgency_level: str
    next_steps: list[str]

    # Metadata
    style_used: NarrativeStyle
    generation_time_ms: float
    scalars_applied: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "summary": self.summary,
            "detailed_explanation": self.detailed_explanation,
            "reasoning_chain": (
                {
                    "steps": self.reasoning_chain.steps,
                    "hypotheses_considered": self.reasoning_chain.hypotheses_considered,
                    "hypotheses_rejected": self.reasoning_chain.hypotheses_rejected,
                    "conclusion": self.reasoning_chain.final_conclusion,
                    "confidence_rationale": self.reasoning_chain.confidence_rationale,
                }
                if self.reasoning_chain
                else None
            ),
            "confidence": {
                "level": self.confidence_level.value,
                "score": self.confidence_score,
                "uncertainty_disclosure": self.uncertainty_disclosure,
            },
            "historical_context": self.historical_context,
            "similar_past_events": self.similar_past_events,
            "recommendations": self.recommendations,
            "urgency_level": self.urgency_level,
            "next_steps": self.next_steps,
            "metadata": {
                "style": self.style_used.value,
                "generation_time_ms": self.generation_time_ms,
                "scalars_applied": self.scalars_applied,
            },
        }


class NarrativeEngine:
    """Truth-Dense Communication Synthesis Engine.

    Transforms raw detection results into transparent, human-readable
    communication. Optimizes for truth density, not engagement.

    Key Principles:
        1. Every claim backed by evidence or explicitly uncertain
        2. Reasoning chains exposed, not hidden
        3. Historical context surfaced when relevant
        4. Recommendations proportional to confidence
        5. Style shaped by ethical scalars, not manipulation

    Usage:
        engine = NarrativeEngine()

        # From cognitive analysis result
        narrative = engine.synthesize(
            detection_result=cognitive_result.to_dict(),
            domain="medical",
            context={"patient_id": "P001", "vital_type": "heart_rate"}
        )

        print(narrative.summary)
        print(narrative.detailed_explanation)
        print(narrative.recommendations)
    """

    # Scalar thresholds for style selection
    URGENCY_THRESHOLD = 0.8
    ANALYTICAL_THRESHOLD = 0.6
    SUPPORTIVE_THRESHOLD = 1.25

    def __init__(
        self,
        default_style: NarrativeStyle = NarrativeStyle.EXPLANATORY,
        use_llm_enhancement: bool = False,
        max_reasoning_steps: int = 10,
    ) -> None:
        """Initialize Narrative Engine.

        Args:
            default_style: Default communication style
            use_llm_enhancement: Whether to use LLM for natural language polish
            max_reasoning_steps: Maximum reasoning steps to verbalize
        """
        self.default_style = default_style
        self.use_llm_enhancement = use_llm_enhancement
        self.max_reasoning_steps = max_reasoning_steps

        self._gosnn: Any | None = None  # Lazy-loaded
        self._llm_adapter: Any | None = None  # Lazy-loaded
        self._memory_surface: Any | None = None  # Set via set_memory_surface

        self._synthesis_count = 0
        self.logger = logging.getLogger(__name__)

    def set_memory_surface(self, memory_surface: Any) -> None:
        """Set memory surface for historical context retrieval."""
        self._memory_surface = memory_surface

    def _get_gosnn(self) -> Any:
        """Lazy-load Global Omni-Scalar Network."""
        if self._gosnn is None:
            try:
                from omni_mercury_engine.core.global_omni_scalar_network import (
                    get_global_scalar_network,
                )

                self._gosnn = get_global_scalar_network()
            except ImportError:
                self.logger.warning("GOSNN not available, using default scalars")
        return self._gosnn

    def synthesize(
        self,
        detection_result: dict[str, Any],
        domain: str | None = None,
        context: dict[str, Any] | None = None,
        style_override: NarrativeStyle | None = None,
    ) -> NarrativeResult:
        """Synthesize narrative from detection results.

        This is the primary method for transforming raw detection output
        into truth-dense, transparent communication.

        Args:
            detection_result: Output from CognitiveOrchestrator.analyze() or similar
            domain: Domain context (medical, security, etc.)
            context: Additional context for narrative
            style_override: Force specific communication style

        Returns:
            NarrativeResult with complete transparent communication
        """
        start_time = time.time()
        self._synthesis_count += 1
        context = context or {}

        # Extract core detection values
        anomaly_detected = detection_result.get("anomaly_detected", False)
        anomaly_score = detection_result.get("anomaly_score", 0.0)
        severity = detection_result.get("severity", 0.0)
        confidence = detection_result.get("confidence", 0.5)
        is_reliable = detection_result.get("is_reliable", True)

        # Get omni-scalars for style shaping
        scalars = self._get_personality_scalars(domain)

        # Determine appropriate style
        style = style_override or self._determine_style(
            anomaly_detected, severity, confidence, scalars
        )

        # Classify confidence level
        confidence_level = self._classify_confidence(confidence, is_reliable)

        # Build reasoning chain narrative
        reasoning_narrative = self._verbalize_reasoning_chain(
            detection_result.get("reasoning_chain", []),
            detection_result.get("causal_factors", []),
        )

        # Get historical context if memory surface available
        historical_context, similar_events = self._get_historical_context(detection_result, domain)

        # Generate uncertainty disclosure
        uncertainty_disclosure = self._generate_uncertainty_disclosure(
            confidence,
            is_reliable,
            detection_result.get("epistemic_uncertainty", 0.0),
            detection_result.get("aleatoric_uncertainty", 0.0),
        )

        # Synthesize summary
        summary = self._generate_summary(anomaly_detected, anomaly_score, severity, domain, style)

        # Synthesize detailed explanation
        detailed = self._generate_detailed_explanation(
            detection_result, domain, context, style, scalars
        )

        # Generate recommendations proportional to confidence
        recommendations = self._generate_recommendations(
            anomaly_detected,
            severity,
            confidence_level,
            domain,
            detection_result.get("recommendations", []),
        )

        # Determine urgency
        urgency = self._determine_urgency(anomaly_detected, severity, confidence)

        # Generate next steps
        next_steps = self._generate_next_steps(anomaly_detected, confidence_level, domain)

        generation_time = (time.time() - start_time) * 1000

        return NarrativeResult(
            summary=summary,
            detailed_explanation=detailed,
            reasoning_chain=reasoning_narrative,
            confidence_level=confidence_level,
            confidence_score=confidence,
            uncertainty_disclosure=uncertainty_disclosure,
            historical_context=historical_context,
            similar_past_events=similar_events,
            recommendations=recommendations,
            urgency_level=urgency,
            next_steps=next_steps,
            style_used=style,
            generation_time_ms=generation_time,
            scalars_applied=scalars,
        )

    def _get_personality_scalars(self, domain: str | None) -> dict[str, float]:
        """Get relevant omni-scalars for personality shaping."""
        gosnn = self._get_gosnn()
        if gosnn is None:
            return {
                "omnibenevolence": ETHICAL.BENEVOLENCE_IMMUTABLE,
                "omnitransparency": 0.18,
                "omniexplainability": 0.9,
                "omnicompassion": 1.30,
                "omnidetermination": 1.30,
            }

        return {
            "omnibenevolence": gosnn.get_scalar("omnibenevolence", ETHICAL.BENEVOLENCE_IMMUTABLE),
            "omnitransparency": gosnn.get_scalar("omnitransparency", 0.18),
            "omniexplainability": gosnn.get_scalar("omniexplainability", 0.9),
            "omnicompassion": gosnn.get_scalar("omnicompassion", 1.30),
            "omnidetermination": gosnn.get_scalar("omnidetermination", 1.30),
            "omnipatience": gosnn.get_scalar("omnipatience", 1.20),
            "omnivigilance": gosnn.get_scalar("omnivigilance", 1.20),
        }

    def _determine_style(
        self,
        anomaly_detected: bool,
        severity: float,
        confidence: float,
        scalars: dict[str, float],
    ) -> NarrativeStyle:
        """Determine communication style based on context and scalars."""
        # High severity demands urgency
        if anomaly_detected and severity > self.URGENCY_THRESHOLD:
            return NarrativeStyle.URGENT

        # High explainability scalar promotes detailed explanations
        if scalars.get("omniexplainability", 0) > self.ANALYTICAL_THRESHOLD:
            if confidence > 0.7:
                return NarrativeStyle.ANALYTICAL
            return NarrativeStyle.EXPLANATORY

        # High compassion with moderate confidence suggests supportive
        if scalars.get("omnicompassion", 0) > self.SUPPORTIVE_THRESHOLD:
            return NarrativeStyle.SUPPORTIVE

        # Low confidence defaults to clinical precision
        if confidence < 0.5:
            return NarrativeStyle.CLINICAL

        return self.default_style

    def _classify_confidence(self, confidence: float, is_reliable: bool) -> ConfidenceLevel:
        """Classify confidence level with reliability adjustment."""
        # Unreliable predictions get downgraded
        effective_confidence = confidence * (0.8 if not is_reliable else 1.0)

        if effective_confidence < 0.3:
            return ConfidenceLevel.VERY_LOW
        elif effective_confidence < 0.5:
            return ConfidenceLevel.LOW
        elif effective_confidence < 0.7:
            return ConfidenceLevel.MODERATE
        elif effective_confidence < 0.9:
            return ConfidenceLevel.HIGH
        return ConfidenceLevel.VERY_HIGH

    def _verbalize_reasoning_chain(
        self,
        reasoning_chain: list[dict[str, Any]],
        causal_factors: list[str],
    ) -> ReasoningChainNarrative | None:
        """Verbalize reasoning chain for transparency."""
        if not reasoning_chain:
            return None

        steps = []
        hypotheses_considered = []
        hypotheses_rejected = []

        for i, step in enumerate(reasoning_chain[: self.max_reasoning_steps]):
            rule = step.get("rule", "unknown_rule")
            conclusion = step.get("conclusion", "")
            conf = step.get("confidence", 0.5)

            step_text = f"Step {i + 1}: Applied {rule} -> {conclusion} (confidence: {conf:.0%})"
            steps.append(step_text)

            # Track hypotheses
            if conf > 0.6:
                hypotheses_considered.append(conclusion)
            else:
                hypotheses_rejected.append(f"{conclusion} (insufficient confidence)")

        # Add causal factors as hypotheses
        for factor in causal_factors[:5]:
            hypotheses_considered.append(f"Causal factor: {factor}")

        final_conclusion = (
            reasoning_chain[-1].get("conclusion", "No conclusion reached")
            if reasoning_chain
            else "Insufficient data for reasoning"
        )

        confidence_rationale = self._generate_confidence_rationale(
            reasoning_chain, hypotheses_rejected
        )

        return ReasoningChainNarrative(
            steps=steps,
            hypotheses_considered=hypotheses_considered,
            hypotheses_rejected=hypotheses_rejected,
            final_conclusion=final_conclusion,
            confidence_rationale=confidence_rationale,
        )

    def _generate_confidence_rationale(
        self, reasoning_chain: list[dict[str, Any]], rejected: list[str]
    ) -> str:
        """Generate explanation for confidence level."""
        if not reasoning_chain:
            return "Confidence limited by insufficient reasoning data."

        avg_step_conf = np.mean([s.get("confidence", 0.5) for s in reasoning_chain])
        n_steps = len(reasoning_chain)
        n_rejected = len(rejected)

        rationale_parts = []

        if avg_step_conf > 0.8:
            rationale_parts.append(
                f"Strong evidence chain ({n_steps} steps, avg confidence {avg_step_conf:.0%})"
            )
        elif avg_step_conf > 0.6:
            rationale_parts.append(
                f"Moderate evidence ({n_steps} steps, avg confidence {avg_step_conf:.0%})"
            )
        else:
            rationale_parts.append(
                f"Weak evidence chain ({n_steps} steps, avg confidence {avg_step_conf:.0%})"
            )

        if n_rejected > 0:
            rationale_parts.append(f"{n_rejected} alternative hypotheses ruled out")

        return ". ".join(rationale_parts) + "."

    def _get_historical_context(
        self, detection_result: dict[str, Any], domain: str | None
    ) -> tuple[str | None, list[str]]:
        """Retrieve historical context from memory surface."""
        if self._memory_surface is None:
            return None, []

        try:
            context = self._memory_surface.get_relevant_context(detection_result, domain)
            return context.summary, context.similar_event_ids
        except Exception as e:
            self.logger.debug(f"Memory surface lookup failed: {e}")
            return None, []

    def _generate_uncertainty_disclosure(
        self,
        confidence: float,
        is_reliable: bool,
        epistemic: float,
        aleatoric: float,
    ) -> str:
        """Generate explicit uncertainty disclosure."""
        parts = []

        # Overall confidence statement
        if confidence < 0.3:
            parts.append(
                f"High uncertainty (confidence: {confidence:.0%}). "
                "Multiple interpretations possible."
            )
        elif confidence < 0.5:
            parts.append(
                f"Moderate uncertainty (confidence: {confidence:.0%}). " "Verification recommended."
            )
        elif confidence < 0.7:
            parts.append(f"Reasonable confidence ({confidence:.0%}), but uncertainty remains.")
        elif confidence < 0.9:
            parts.append(f"High confidence ({confidence:.0%}).")
        else:
            parts.append(f"Very high confidence ({confidence:.0%}).")

        # Reliability warning
        if not is_reliable:
            parts.append("WARNING: This prediction is flagged as potentially unreliable.")

        # Uncertainty decomposition
        if epistemic > 0.1:
            parts.append(f"Epistemic uncertainty: {epistemic:.0%} (model knowledge gaps).")
        if aleatoric > 0.1:
            parts.append(f"Aleatoric uncertainty: {aleatoric:.0%} (inherent data variability).")

        return " ".join(parts)

    def _generate_summary(
        self,
        anomaly_detected: bool,
        anomaly_score: float,
        severity: float,
        domain: str | None,
        style: NarrativeStyle,
    ) -> str:
        """Generate one-line summary."""
        domain_prefix = f"[{domain.upper()}] " if domain else ""

        if not anomaly_detected:
            return (
                f"{domain_prefix}No anomaly detected. Systems operating within normal parameters."
            )

        severity_word = (
            "critical"
            if severity > 0.8
            else "significant" if severity > 0.6 else "moderate" if severity > 0.4 else "minor"
        )

        if style == NarrativeStyle.URGENT:
            return (
                f"{domain_prefix}ALERT: {severity_word.upper()} anomaly detected. "
                f"Score: {anomaly_score:.2f}. Severity: {severity:.0%}. Immediate review recommended."
            )
        elif style == NarrativeStyle.CLINICAL:
            return (
                f"{domain_prefix}Anomaly detected. "
                f"Score={anomaly_score:.3f}, Severity={severity:.3f}."
            )
        else:
            return (
                f"{domain_prefix}Detected {severity_word} anomaly "
                f"(score: {anomaly_score:.2f}, severity: {severity:.0%})."
            )

    def _generate_detailed_explanation(
        self,
        detection_result: dict[str, Any],
        domain: str | None,
        context: dict[str, Any],
        style: NarrativeStyle,
        scalars: dict[str, float],
    ) -> str:
        """Generate detailed explanation with full transparency."""
        parts = []

        anomaly_detected = detection_result.get("anomaly_detected", False)
        anomaly_score = detection_result.get("anomaly_score", 0.0)
        severity = detection_result.get("severity", 0.0)
        confidence = detection_result.get("confidence", 0.5)

        # Opening statement
        if anomaly_detected:
            parts.append(
                f"An anomaly has been detected with a score of {anomaly_score:.3f} "
                f"and severity rating of {severity:.0%}."
            )
        else:
            parts.append(
                "Analysis complete. No significant anomalies were detected in the provided data."
            )

        # Confidence context
        parts.append(f"Detection confidence: {confidence:.0%}.")

        # Reasoning chain summary
        reasoning_chain = detection_result.get("reasoning_chain", [])
        if reasoning_chain:
            parts.append(f"This conclusion is based on {len(reasoning_chain)} reasoning steps.")

            if style in (NarrativeStyle.EXPLANATORY, NarrativeStyle.ANALYTICAL):
                # Include key reasoning steps
                for step in reasoning_chain[:3]:
                    rule = step.get("rule", "")
                    conclusion = step.get("conclusion", "")
                    if rule and conclusion:
                        parts.append(f"  - {rule}: {conclusion}")

        # Causal factors
        causal_factors = detection_result.get("causal_factors", [])
        if causal_factors:
            parts.append("Identified causal relationships:")
            for factor in causal_factors[:3]:
                parts.append(f"  - {factor}")

        # Similar cases
        similar_cases = detection_result.get("similar_cases", [])
        if similar_cases:
            parts.append(f"This pattern has similarities to {len(similar_cases)} historical cases.")

        # Triggered indicators
        warnings = detection_result.get("warnings", [])
        if warnings:
            parts.append(f"Triggered {len(warnings)} warning indicators.")
            if style == NarrativeStyle.URGENT:
                for w in warnings[:2]:
                    if isinstance(w, dict):
                        parts.append(f"  - {w.get('message', str(w))}")

        # Domain-specific context
        if domain:
            parts.append(f"Analysis performed in {domain} domain context.")

        # Transparency scalar influence
        transparency = scalars.get("omnitransparency", 0)
        if transparency > 0.15:
            parts.append(
                "Full reasoning transparency enabled. "
                "All intermediate steps available in reasoning_chain."
            )

        return " ".join(parts)

    def _generate_recommendations(
        self,
        anomaly_detected: bool,
        severity: float,
        confidence_level: ConfidenceLevel,
        domain: str | None,
        existing_recommendations: list[str],
    ) -> list[str]:
        """Generate recommendations proportional to confidence."""
        recommendations = []

        # Include existing recommendations if confidence is sufficient
        if confidence_level in (ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH):
            recommendations.extend(existing_recommendations[:3])

        if not anomaly_detected:
            recommendations.append("Continue standard monitoring protocols.")
            return recommendations

        # Confidence-gated recommendations
        if confidence_level == ConfidenceLevel.VERY_HIGH:
            if severity > 0.8:
                recommendations.append("IMMEDIATE ACTION REQUIRED: Escalate to on-call team.")
            elif severity > 0.5:
                recommendations.append("Schedule review within 24 hours.")
            else:
                recommendations.append("Add to next routine review cycle.")

        elif confidence_level == ConfidenceLevel.HIGH:
            recommendations.append("Review detection with domain expert.")
            if severity > 0.6:
                recommendations.append("Consider additional data collection for validation.")

        elif confidence_level == ConfidenceLevel.MODERATE:
            recommendations.append(
                "Confidence is moderate. Cross-validate with additional data sources."
            )
            recommendations.append("Do not take irreversible action without verification.")

        elif confidence_level in (ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW):
            recommendations.append("LOW CONFIDENCE: Treat as preliminary finding only.")
            recommendations.append("Gather additional data before drawing conclusions.")
            recommendations.append("Consider alternative explanations.")

        # Domain-specific recommendations
        if domain == "medical":
            recommendations.append("Ensure clinical review before any treatment decisions.")
        elif domain == "security":
            recommendations.append("Document incident and preserve relevant logs.")
        elif domain == "infrastructure":
            recommendations.append("Check correlated systems for cascade effects.")

        return recommendations

    def _determine_urgency(self, anomaly_detected: bool, severity: float, confidence: float) -> str:
        """Determine urgency level."""
        if not anomaly_detected:
            return "routine"

        # Urgency requires both severity and confidence
        urgency_score = severity * confidence

        if urgency_score > 0.7:
            return "critical"
        elif urgency_score > 0.5:
            return "high"
        elif urgency_score > 0.3:
            return "moderate"
        elif urgency_score > 0.15:
            return "low"
        return "informational"

    def _generate_next_steps(
        self,
        anomaly_detected: bool,
        confidence_level: ConfidenceLevel,
        domain: str | None,
    ) -> list[str]:
        """Generate concrete next steps."""
        steps = []

        if not anomaly_detected:
            steps.append("No action required. Continue monitoring.")
            return steps

        # Universal first steps
        steps.append("Review detection details and reasoning chain.")

        if confidence_level in (ConfidenceLevel.VERY_LOW, ConfidenceLevel.LOW):
            steps.append("Gather additional supporting data.")
            steps.append("Check for data quality issues.")
            steps.append("Consider false positive possibility.")
        else:
            steps.append("Validate against known patterns.")
            steps.append("Determine appropriate response level.")

        # Domain-specific next steps
        if domain == "medical":
            steps.append("Consult clinical decision support guidelines.")
        elif domain == "security":
            steps.append("Initiate incident response protocol if warranted.")
        elif domain == "financial":
            steps.append("Review regulatory reporting requirements.")
        elif domain == "infrastructure":
            steps.append("Assess impact on dependent systems.")

        return steps

    def get_statistics(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            "synthesis_count": self._synthesis_count,
            "default_style": self.default_style.value,
            "use_llm_enhancement": self.use_llm_enhancement,
            "max_reasoning_steps": self.max_reasoning_steps,
            "gosnn_connected": self._gosnn is not None,
            "memory_surface_connected": self._memory_surface is not None,
        }
