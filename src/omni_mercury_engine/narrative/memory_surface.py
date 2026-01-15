"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

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
Memory Surface - Conversational Memory Integration

Surfaces memory context in communications, enabling responses that reference
past experiences, learned patterns, and historical predictions.

Key Behaviors:
    - "This pattern reminds me of the incident on January 3rd..."
    - "I've seen 47 similar anomalies this month - here's what I've learned..."
    - "My previous prediction for this pattern was correct 78% of the time."
    - "Historical context suggests this precedes escalation."

Integration Points:
    - NeuralMemoryLayer: Pattern detection and embeddings
    - AgentMemory: Episodic and semantic storage
    - CognitiveOrchestrator: Knowledge graph context

Not artificial nostalgia - genuine pattern-informed context.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


class MemoryRelevance(Enum):
    """Relevance classification for retrieved memories."""

    HIGHLY_RELEVANT = "highly_relevant"  # >0.8 similarity
    RELEVANT = "relevant"  # 0.6-0.8 similarity
    SOMEWHAT_RELEVANT = "somewhat_relevant"  # 0.4-0.6 similarity
    TANGENTIAL = "tangential"  # <0.4 similarity


@dataclass
class SimilarEvent:
    """A similar historical event."""

    event_id: str
    timestamp: float
    similarity_score: float
    domain: str | None
    outcome: str | None
    summary: str
    was_prediction_correct: bool | None = None


@dataclass
class PredictionHistory:
    """Historical prediction accuracy for a pattern type."""

    pattern_type: str
    total_predictions: int
    correct_predictions: int
    accuracy: float
    avg_lead_time_sec: float
    last_prediction: float


@dataclass
class MemoryContext:
    """Complete memory context for a detection."""

    # Summary narrative
    summary: str

    # Similar events
    similar_events: list[SimilarEvent]
    similar_event_ids: list[str]

    # Pattern statistics
    pattern_frequency: int  # How often seen in memory
    pattern_timespan_days: float  # How long we've tracked this
    first_seen: float | None
    last_seen: float | None

    # Prediction history
    prediction_accuracy: PredictionHistory | None

    # Learning insights
    learned_insights: list[str]

    # Relevance assessment
    overall_relevance: MemoryRelevance
    relevance_score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary": self.summary,
            "similar_events": [
                {
                    "event_id": e.event_id,
                    "timestamp": e.timestamp,
                    "similarity": e.similarity_score,
                    "domain": e.domain,
                    "outcome": e.outcome,
                    "summary": e.summary,
                }
                for e in self.similar_events
            ],
            "pattern_frequency": self.pattern_frequency,
            "pattern_timespan_days": self.pattern_timespan_days,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "prediction_accuracy": (
                {
                    "pattern_type": self.prediction_accuracy.pattern_type,
                    "accuracy": self.prediction_accuracy.accuracy,
                    "total_predictions": self.prediction_accuracy.total_predictions,
                }
                if self.prediction_accuracy
                else None
            ),
            "learned_insights": self.learned_insights,
            "relevance": {
                "level": self.overall_relevance.value,
                "score": self.relevance_score,
            },
        }


class MemorySurface:
    """
    Surfaces Memory Context in Communications.

    Bridges the neural memory layer and agent memory to provide contextual
    awareness in narrative generation. Enables Mercury to reference past
    experiences meaningfully.

    Key Capabilities:
        1. Similar event retrieval with semantic matching
        2. Pattern frequency and evolution tracking
        3. Prediction accuracy history
        4. Learning insight synthesis
        5. Contextual narrative generation

    Usage:
        surface = MemorySurface()

        # Connect to memory systems
        surface.set_neural_memory(neural_memory_layer)
        surface.set_agent_memory(agent_memory)

        # Get context for a detection
        context = surface.get_relevant_context(detection_result, domain="medical")

        print(context.summary)
        # "This pattern is similar to 3 previous events. Based on historical data,
        #  patterns like this escalated 67% of the time within 24 hours."
    """

    def __init__(
        self,
        max_similar_events: int = 5,
        similarity_threshold: float = 0.4,
        lookback_days: int = 90,
    ) -> None:
        """
        Initialize Memory Surface.

        Args:
            max_similar_events: Maximum similar events to retrieve
            similarity_threshold: Minimum similarity for relevance
            lookback_days: Days to look back for historical context
        """
        self.max_similar_events = max_similar_events
        self.similarity_threshold = similarity_threshold
        self.lookback_days = lookback_days

        self._neural_memory = None
        self._agent_memory = None
        self._knowledge_graph = None

        # Internal tracking
        self._pattern_registry: dict[str, dict[str, Any]] = {}
        self._prediction_outcomes: dict[str, list[dict[str, Any]]] = {}

        self._retrieval_count = 0
        self.logger = logging.getLogger(__name__)

    def set_neural_memory(self, neural_memory: Any) -> None:
        """Set neural memory layer reference."""
        self._neural_memory = neural_memory

    def set_agent_memory(self, agent_memory: Any) -> None:
        """Set agent memory reference."""
        self._agent_memory = agent_memory

    def set_knowledge_graph(self, knowledge_graph: Any) -> None:
        """Set knowledge graph reference."""
        self._knowledge_graph = knowledge_graph

    def get_relevant_context(
        self,
        detection_result: dict[str, Any],
        domain: str | None = None,
    ) -> MemoryContext:
        """
        Retrieve relevant memory context for a detection.

        Args:
            detection_result: Detection output to find context for
            domain: Domain context

        Returns:
            MemoryContext with historical awareness
        """
        self._retrieval_count += 1
        _current_time = time.time()  # Reserved for future time-based filtering

        # Get similar events
        similar_events = self._find_similar_events(detection_result, domain)

        # Get pattern statistics
        pattern_stats = self._get_pattern_statistics(detection_result, domain)

        # Get prediction history
        prediction_history = self._get_prediction_history(detection_result, domain)

        # Synthesize learned insights
        insights = self._synthesize_insights(
            similar_events, pattern_stats, prediction_history, detection_result
        )

        # Determine overall relevance
        relevance, relevance_score = self._assess_relevance(similar_events, pattern_stats)

        # Generate summary narrative
        summary = self._generate_context_summary(
            similar_events,
            pattern_stats,
            prediction_history,
            insights,
            domain,
        )

        return MemoryContext(
            summary=summary,
            similar_events=similar_events,
            similar_event_ids=[e.event_id for e in similar_events],
            pattern_frequency=pattern_stats.get("frequency", 0),
            pattern_timespan_days=pattern_stats.get("timespan_days", 0),
            first_seen=pattern_stats.get("first_seen"),
            last_seen=pattern_stats.get("last_seen"),
            prediction_accuracy=prediction_history,
            learned_insights=insights,
            overall_relevance=relevance,
            relevance_score=relevance_score,
        )

    def _find_similar_events(
        self,
        detection_result: dict[str, Any],
        domain: str | None,
    ) -> list[SimilarEvent]:
        """Find similar historical events using neural memory."""
        similar_events: list[SimilarEvent] = []

        # Try neural memory similarity search
        if self._neural_memory is not None:
            try:
                # Create embedding for current detection
                detection_features = self._extract_features(detection_result)
                query_embedding = self._neural_memory.vectorizer.transform(detection_features)

                # Find similar
                similar = self._neural_memory.get_similar_memories(
                    query_embedding, top_k=self.max_similar_events
                )

                for mem_embedding, similarity in similar:
                    if similarity >= self.similarity_threshold:
                        similar_events.append(
                            SimilarEvent(
                                event_id=mem_embedding.entry_id,
                                timestamp=mem_embedding.timestamp,
                                similarity_score=similarity,
                                domain=mem_embedding.metadata.get("domain"),
                                outcome=mem_embedding.metadata.get("outcome"),
                                summary=self._summarize_memory(mem_embedding),
                            )
                        )
            except Exception as e:
                self.logger.debug(f"Neural memory search failed: {e}")

        # Try agent episodic memory
        if self._agent_memory is not None and len(similar_events) < self.max_similar_events:
            try:
                episodic = self._agent_memory.retrieve_by_importance(
                    threshold=0.5, memory_type="episodic"
                )

                for mem in episodic[: self.max_similar_events - len(similar_events)]:
                    if isinstance(mem.content, dict):
                        # Compute simple feature similarity
                        similarity = self._compute_feature_similarity(detection_result, mem.content)

                        if similarity >= self.similarity_threshold:
                            similar_events.append(
                                SimilarEvent(
                                    event_id=mem.entry_id,
                                    timestamp=mem.timestamp,
                                    similarity_score=similarity,
                                    domain=mem.content.get("context", {}).get("domain"),
                                    outcome=mem.content.get("outcome"),
                                    summary=str(mem.content.get("event", ""))[:100],
                                )
                            )
            except Exception as e:
                self.logger.debug(f"Agent memory search failed: {e}")

        # Sort by similarity
        similar_events.sort(key=lambda x: x.similarity_score, reverse=True)
        return similar_events[: self.max_similar_events]

    def _extract_features(self, detection_result: dict[str, Any]) -> dict[str, Any]:
        """Extract features from detection result for embedding."""
        return {
            "anomaly_score": detection_result.get("anomaly_score", 0.0),
            "severity": detection_result.get("severity", 0.0),
            "confidence": detection_result.get("confidence", 0.5),
            "domain": detection_result.get("domain", "unknown"),
            "type": "detection",
        }

    def _summarize_memory(self, mem_embedding: Any) -> str:
        """Create summary of memory embedding."""
        metadata = mem_embedding.metadata
        parts = []

        if "event" in metadata:
            parts.append(str(metadata["event"])[:50])
        if "outcome" in metadata:
            parts.append(f"Outcome: {metadata['outcome']}")

        # Add timestamp context
        timestamp = mem_embedding.timestamp
        dt = datetime.fromtimestamp(timestamp)
        parts.append(f"({dt.strftime('%Y-%m-%d')})")

        return " ".join(parts) if parts else f"Memory {mem_embedding.entry_id}"

    def _compute_feature_similarity(
        self,
        detection: dict[str, Any],
        historical: dict[str, Any],
    ) -> float:
        """Compute simple feature-based similarity."""
        score = 0.0
        count = 0

        # Compare numerical features
        for key in ["anomaly_score", "severity", "confidence"]:
            if key in detection and key in historical.get("context", {}):
                diff = abs(detection[key] - historical["context"][key])
                score += 1.0 - min(diff, 1.0)
                count += 1

        # Domain match bonus
        det_domain = detection.get("domain", "")
        hist_domain = historical.get("context", {}).get("domain", "")
        if det_domain and det_domain == hist_domain:
            score += 0.3
            count += 1

        return score / count if count > 0 else 0.0

    def _get_pattern_statistics(
        self,
        detection_result: dict[str, Any],
        domain: str | None,
    ) -> dict[str, Any]:
        """Get statistics about this pattern type."""
        pattern_key = self._get_pattern_key(detection_result, domain)

        if pattern_key in self._pattern_registry:
            registry = self._pattern_registry[pattern_key]
            return registry

        # Build from memory if available
        stats = {
            "frequency": 0,
            "timespan_days": 0,
            "first_seen": None,
            "last_seen": None,
        }

        if self._neural_memory is not None:
            try:
                analysis = self._neural_memory.analyze()
                stats["frequency"] = analysis.get("patterns_detected", 0)
            except Exception:
                # Neural memory may not be fully initialized; use default stats
                pass

        return stats

    def _get_pattern_key(self, detection_result: dict[str, Any], domain: str | None) -> str:
        """Generate pattern key for tracking."""
        severity_bucket = int(detection_result.get("severity", 0) * 10)
        score_bucket = int(detection_result.get("anomaly_score", 0) * 10)
        return f"{domain or 'general'}:sev{severity_bucket}:score{score_bucket}"

    def _get_prediction_history(
        self,
        detection_result: dict[str, Any],
        domain: str | None,
    ) -> PredictionHistory | None:
        """Get prediction accuracy history for this pattern type."""
        pattern_key = self._get_pattern_key(detection_result, domain)

        if pattern_key in self._prediction_outcomes:
            outcomes = self._prediction_outcomes[pattern_key]
            if len(outcomes) >= 3:  # Need minimum data
                correct = sum(1 for o in outcomes if o.get("correct", False))
                total = len(outcomes)
                avg_lead = np.mean([o.get("lead_time", 0) for o in outcomes])

                return PredictionHistory(
                    pattern_type=pattern_key,
                    total_predictions=total,
                    correct_predictions=correct,
                    accuracy=correct / total if total > 0 else 0.0,
                    avg_lead_time_sec=float(avg_lead),
                    last_prediction=outcomes[-1].get("timestamp", 0),
                )

        return None

    def record_prediction_outcome(
        self,
        detection_result: dict[str, Any],
        domain: str | None,
        was_correct: bool,
        lead_time_sec: float = 0.0,
    ) -> None:
        """Record prediction outcome for accuracy tracking."""
        pattern_key = self._get_pattern_key(detection_result, domain)

        if pattern_key not in self._prediction_outcomes:
            self._prediction_outcomes[pattern_key] = []

        self._prediction_outcomes[pattern_key].append(
            {
                "timestamp": time.time(),
                "correct": was_correct,
                "lead_time": lead_time_sec,
            }
        )

        # Keep last 100 outcomes per pattern
        if len(self._prediction_outcomes[pattern_key]) > 100:
            self._prediction_outcomes[pattern_key] = self._prediction_outcomes[pattern_key][-100:]

    def _synthesize_insights(
        self,
        similar_events: list[SimilarEvent],
        pattern_stats: dict[str, Any],
        prediction_history: PredictionHistory | None,
        detection_result: dict[str, Any],
    ) -> list[str]:
        """Synthesize learned insights from memory."""
        insights = []

        # Frequency insight
        frequency = pattern_stats.get("frequency", 0)
        if frequency > 10:
            insights.append(f"This pattern type has been observed {frequency} times in memory.")
        elif frequency > 0:
            insights.append(f"Similar patterns have occurred {frequency} times previously.")

        # Outcome pattern insight
        outcomes_with_escalation = sum(
            1 for e in similar_events if e.outcome and "escalat" in e.outcome.lower()
        )
        if outcomes_with_escalation > 0 and len(similar_events) > 0:
            escalation_rate = outcomes_with_escalation / len(similar_events)
            if escalation_rate > 0.5:
                insights.append(
                    f"Historical data suggests {escalation_rate:.0%} of similar "
                    "patterns led to escalation."
                )

        # Prediction accuracy insight
        if prediction_history and prediction_history.total_predictions >= 5:
            acc = prediction_history.accuracy
            if acc > 0.8:
                insights.append(
                    f"Predictions for this pattern type have been highly accurate "
                    f"({acc:.0%} over {prediction_history.total_predictions} cases)."
                )
            elif acc > 0.6:
                insights.append(f"Prediction accuracy for this pattern: {acc:.0%}.")
            elif acc < 0.5:
                insights.append(
                    f"Note: Historical prediction accuracy for this pattern is "
                    f"limited ({acc:.0%}). Interpret with caution."
                )

        # Time pattern insight
        first_seen = pattern_stats.get("first_seen")
        if first_seen:
            days_tracking = (time.time() - first_seen) / 86400
            if days_tracking > 30:
                insights.append(f"This pattern type has been tracked for {days_tracking:.0f} days.")

        return insights

    def _assess_relevance(
        self,
        similar_events: list[SimilarEvent],
        pattern_stats: dict[str, Any],
    ) -> tuple[MemoryRelevance, float]:
        """Assess overall relevance of memory context."""
        if not similar_events:
            return MemoryRelevance.TANGENTIAL, 0.0

        # Compute relevance score
        max_similarity = max(e.similarity_score for e in similar_events)
        avg_similarity = np.mean([e.similarity_score for e in similar_events])
        frequency_factor = min(pattern_stats.get("frequency", 0) / 10, 1.0)

        relevance_score = 0.5 * max_similarity + 0.3 * avg_similarity + 0.2 * frequency_factor

        # Classify
        if relevance_score > 0.8:
            return MemoryRelevance.HIGHLY_RELEVANT, relevance_score
        elif relevance_score > 0.6:
            return MemoryRelevance.RELEVANT, relevance_score
        elif relevance_score > 0.4:
            return MemoryRelevance.SOMEWHAT_RELEVANT, relevance_score
        return MemoryRelevance.TANGENTIAL, relevance_score

    def _generate_context_summary(
        self,
        similar_events: list[SimilarEvent],
        pattern_stats: dict[str, Any],
        prediction_history: PredictionHistory | None,
        insights: list[str],
        domain: str | None,
    ) -> str:
        """Generate human-readable context summary."""
        parts = []

        # Similar events summary
        if similar_events:
            n_similar = len(similar_events)
            top_similarity = similar_events[0].similarity_score
            parts.append(
                f"This pattern is similar to {n_similar} previous event(s) "
                f"(top match: {top_similarity:.0%} similarity)."
            )

            # Reference specific event if highly relevant
            if top_similarity > 0.8:
                top_event = similar_events[0]
                dt = datetime.fromtimestamp(top_event.timestamp)
                parts.append(f"Most similar to event from {dt.strftime('%B %d, %Y')}.")
        else:
            parts.append("No highly similar historical events found.")

        # Pattern frequency
        frequency = pattern_stats.get("frequency", 0)
        if frequency > 0:
            parts.append(f"Pattern type observed {frequency} times in memory.")

        # Prediction context
        if prediction_history and prediction_history.accuracy > 0:
            parts.append(
                f"Historical prediction accuracy: {prediction_history.accuracy:.0%} "
                f"({prediction_history.total_predictions} predictions)."
            )

        # Key insight
        if insights:
            parts.append(insights[0])

        return " ".join(parts)

    def record_event(
        self,
        detection_result: dict[str, Any],
        domain: str | None,
        outcome: str | None = None,
    ) -> None:
        """Record an event for future memory retrieval."""
        # Update pattern registry
        pattern_key = self._get_pattern_key(detection_result, domain)
        current_time = time.time()

        if pattern_key not in self._pattern_registry:
            self._pattern_registry[pattern_key] = {
                "frequency": 0,
                "first_seen": current_time,
                "last_seen": current_time,
                "timespan_days": 0,
            }

        registry = self._pattern_registry[pattern_key]
        registry["frequency"] += 1
        registry["last_seen"] = current_time
        registry["timespan_days"] = (current_time - registry["first_seen"]) / 86400

        # Store in neural memory if available
        if self._neural_memory is not None:
            try:
                memory_content = {
                    **self._extract_features(detection_result),
                    "timestamp": current_time,
                    "domain": domain,
                    "outcome": outcome,
                }
                self._neural_memory.ingest_memories([memory_content])
            except Exception as e:
                self.logger.debug(f"Failed to ingest memory: {e}")

    def get_statistics(self) -> dict[str, Any]:
        """Get memory surface statistics."""
        return {
            "retrieval_count": self._retrieval_count,
            "pattern_types_tracked": len(self._pattern_registry),
            "prediction_patterns_tracked": len(self._prediction_outcomes),
            "neural_memory_connected": self._neural_memory is not None,
            "agent_memory_connected": self._agent_memory is not None,
            "knowledge_graph_connected": self._knowledge_graph is not None,
            "max_similar_events": self.max_similar_events,
            "similarity_threshold": self.similarity_threshold,
            "lookback_days": self.lookback_days,
        }
