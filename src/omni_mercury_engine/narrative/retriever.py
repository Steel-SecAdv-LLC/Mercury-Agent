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
Knowledge Retriever - Unified Information Search for Mercury Agent

Provides unified search across all Mercury's knowledge sources:
- Knowledge Graph (semantic search, traversal, similarity)
- Agent Memory (episodic events, semantic facts)
- Pattern History (historical anomalies, predictions)
- External Sources (when configured)

This enables Mercury to "know" things and reference them in conversation,
supporting truth-dense responses with evidence backing.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class RetrievalSource(Enum):
    """Sources of retrieved information."""

    KNOWLEDGE_GRAPH = "knowledge_graph"
    EPISODIC_MEMORY = "episodic_memory"
    SEMANTIC_MEMORY = "semantic_memory"
    PATTERN_HISTORY = "pattern_history"
    DETECTION_LOG = "detection_log"
    EXTERNAL = "external"


class QueryIntent(Enum):
    """Classified intent of a query."""

    SEARCH_FACTS = "search_facts"  # Looking for specific knowledge
    SEARCH_EVENTS = "search_events"  # Looking for past events
    SEARCH_PATTERNS = "search_patterns"  # Looking for patterns
    EXPLAIN_CONCEPT = "explain_concept"  # Wants explanation
    FIND_SIMILAR = "find_similar"  # Looking for similar items
    COMPARE = "compare"  # Comparing things
    PREDICT = "predict"  # Wants prediction
    STATUS = "status"  # System status
    HELP = "help"  # Help/guidance
    UNKNOWN = "unknown"


@dataclass
class RetrievalResult:
    """Result of a knowledge retrieval."""

    source: RetrievalSource
    content: Any
    relevance_score: float
    confidence: float
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source.value,
            "content": self.content if isinstance(self.content, (str, dict)) else str(self.content),
            "relevance": self.relevance_score,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class SearchContext:
    """Context for a search operation."""

    query: str
    intent: QueryIntent
    domain: str | None = None
    time_range_start: float | None = None
    time_range_end: float | None = None
    max_results: int = 10
    min_relevance: float = 0.3
    sources: list[RetrievalSource] | None = None


@dataclass
class SearchResponse:
    """Complete search response."""

    query: str
    intent: QueryIntent
    results: list[RetrievalResult]
    total_found: int
    search_time_ms: float
    summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "intent": self.intent.value,
            "results": [r.to_dict() for r in self.results],
            "total_found": self.total_found,
            "search_time_ms": self.search_time_ms,
            "summary": self.summary,
        }


class KnowledgeRetriever:
    """
    Unified Knowledge Search for Mercury Agent.

    Provides a single interface for searching across all knowledge sources,
    enabling Mercury to find and reference information in conversation.

    Key Capabilities:
        1. Multi-source search (knowledge graph, memory, patterns)
        2. Intent classification for query understanding
        3. Relevance ranking across sources
        4. Time-aware retrieval
        5. Domain-specific filtering

    Usage:
        retriever = KnowledgeRetriever()

        # Connect knowledge sources
        retriever.set_knowledge_graph(kg)
        retriever.set_agent_memory(memory)

        # Search
        response = retriever.search("What patterns preceded the last anomaly?")
        for result in response.results:
            print(f"{result.source}: {result.content} ({result.relevance_score:.0%})")
    """

    # Intent classification keywords
    INTENT_KEYWORDS = {
        QueryIntent.SEARCH_FACTS: ["what", "who", "where", "which", "define", "explain"],
        QueryIntent.SEARCH_EVENTS: ["when", "happened", "occurred", "event", "incident", "history"],
        QueryIntent.SEARCH_PATTERNS: ["pattern", "trend", "similar", "like", "recurring"],
        QueryIntent.EXPLAIN_CONCEPT: ["why", "how", "explain", "understand", "meaning"],
        QueryIntent.FIND_SIMILAR: ["similar", "like", "related", "comparable", "match"],
        QueryIntent.COMPARE: ["compare", "versus", "vs", "difference", "between"],
        QueryIntent.PREDICT: ["predict", "forecast", "will", "expect", "future"],
        QueryIntent.STATUS: ["status", "state", "running", "health", "operational"],
        QueryIntent.HELP: ["help", "how to", "guide", "assist", "support"],
    }

    def __init__(
        self,
        max_results: int = 10,
        default_min_relevance: float = 0.3,
    ) -> None:
        """
        Initialize Knowledge Retriever.

        Args:
            max_results: Default maximum results per search
            default_min_relevance: Default minimum relevance threshold
        """
        self.max_results = max_results
        self.default_min_relevance = default_min_relevance

        # Connected knowledge sources
        self._knowledge_graph = None
        self._agent_memory = None
        self._memory_surface = None
        self._pattern_registry: dict[str, list[dict[str, Any]]] = {}
        self._detection_log: list[dict[str, Any]] = []

        # Statistics
        self._search_count = 0
        self._total_results_returned = 0

        self.logger = logging.getLogger(__name__)

    def set_knowledge_graph(self, kg: Any) -> None:
        """Set knowledge graph for retrieval."""
        self._knowledge_graph = kg

    def set_agent_memory(self, memory: Any) -> None:
        """Set agent memory for retrieval."""
        self._agent_memory = memory

    def set_memory_surface(self, surface: Any) -> None:
        """Set memory surface for retrieval."""
        self._memory_surface = surface

    def log_detection(self, detection: dict[str, Any], domain: str | None = None) -> None:
        """Log a detection for future retrieval."""
        self._detection_log.append(
            {
                "detection": detection,
                "domain": domain,
                "timestamp": time.time(),
            }
        )

        # Keep last 1000 detections
        if len(self._detection_log) > 1000:
            self._detection_log = self._detection_log[-1000:]

    def search(
        self,
        query: str,
        domain: str | None = None,
        max_results: int | None = None,
        min_relevance: float | None = None,
        sources: list[RetrievalSource] | None = None,
    ) -> SearchResponse:
        """
        Search across all connected knowledge sources.

        Args:
            query: Natural language query
            domain: Optional domain filter
            max_results: Maximum results to return
            min_relevance: Minimum relevance score
            sources: Specific sources to search (None = all)

        Returns:
            SearchResponse with ranked results
        """
        start_time = time.time()
        self._search_count += 1

        max_results = max_results or self.max_results
        min_relevance = min_relevance or self.default_min_relevance

        # Classify intent
        intent = self._classify_intent(query)

        # Build search context
        context = SearchContext(
            query=query,
            intent=intent,
            domain=domain,
            max_results=max_results,
            min_relevance=min_relevance,
            sources=sources,
        )

        # Collect results from all sources
        all_results: list[RetrievalResult] = []

        # Search knowledge graph
        if self._should_search_source(RetrievalSource.KNOWLEDGE_GRAPH, sources):
            all_results.extend(self._search_knowledge_graph(context))

        # Search agent memory
        if self._should_search_source(RetrievalSource.EPISODIC_MEMORY, sources):
            all_results.extend(self._search_episodic_memory(context))

        if self._should_search_source(RetrievalSource.SEMANTIC_MEMORY, sources):
            all_results.extend(self._search_semantic_memory(context))

        # Search detection log
        if self._should_search_source(RetrievalSource.DETECTION_LOG, sources):
            all_results.extend(self._search_detection_log(context))

        # Search pattern history
        if self._should_search_source(RetrievalSource.PATTERN_HISTORY, sources):
            all_results.extend(self._search_pattern_history(context))

        # Filter by relevance
        filtered_results = [r for r in all_results if r.relevance_score >= min_relevance]

        # Rank by relevance
        filtered_results.sort(key=lambda x: x.relevance_score, reverse=True)

        # Limit results
        final_results = filtered_results[:max_results]

        self._total_results_returned += len(final_results)

        # Generate summary
        summary = self._generate_summary(query, final_results, intent)

        search_time = (time.time() - start_time) * 1000

        return SearchResponse(
            query=query,
            intent=intent,
            results=final_results,
            total_found=len(filtered_results),
            search_time_ms=search_time,
            summary=summary,
        )

    def _should_search_source(
        self,
        source: RetrievalSource,
        requested_sources: list[RetrievalSource] | None,
    ) -> bool:
        """Check if a source should be searched."""
        if requested_sources is None:
            return True
        return source in requested_sources

    def _classify_intent(self, query: str) -> QueryIntent:
        """Classify the intent of a query."""
        query_lower = query.lower()

        # Score each intent
        scores: dict[QueryIntent, int] = {intent: 0 for intent in QueryIntent}

        for intent, keywords in self.INTENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    scores[intent] += 1

        # Get highest scoring intent
        max_score = max(scores.values())
        if max_score > 0:
            for intent, score in scores.items():
                if score == max_score:
                    return intent

        return QueryIntent.UNKNOWN

    def _search_knowledge_graph(self, context: SearchContext) -> list[RetrievalResult]:
        """Search knowledge graph."""
        results = []

        if self._knowledge_graph is None:
            return results

        try:
            # Search by keyword in node labels
            query_words = context.query.lower().split()

            for node in self._knowledge_graph._nodes.values():
                label_lower = node.label.lower()
                matches = sum(1 for word in query_words if word in label_lower)

                if matches > 0:
                    relevance = matches / len(query_words)

                    # Boost by PageRank if available
                    if node.pagerank > 0:
                        relevance *= 1 + node.pagerank

                    results.append(
                        RetrievalResult(
                            source=RetrievalSource.KNOWLEDGE_GRAPH,
                            content={
                                "node_id": node.node_id,
                                "type": node.node_type.value,
                                "label": node.label,
                                "attributes": node.attributes,
                            },
                            relevance_score=min(1.0, relevance),
                            confidence=node.confidence,
                            timestamp=node.created_at,
                            metadata={"pagerank": node.pagerank},
                        )
                    )

            # Also try spreading activation for related concepts
            if results:
                top_node_id = results[0].content["node_id"]
                activations = self._knowledge_graph.spreading_activation(
                    [top_node_id], max_iterations=2
                )

                for node_id, activation in sorted(
                    activations.items(), key=lambda x: x[1], reverse=True
                )[:5]:
                    if node_id != top_node_id and node_id in self._knowledge_graph._nodes:
                        node = self._knowledge_graph._nodes[node_id]
                        results.append(
                            RetrievalResult(
                                source=RetrievalSource.KNOWLEDGE_GRAPH,
                                content={
                                    "node_id": node.node_id,
                                    "type": node.node_type.value,
                                    "label": node.label,
                                    "relation": "associated_via_activation",
                                },
                                relevance_score=activation * 0.8,  # Discount associated
                                confidence=node.confidence,
                                timestamp=node.created_at,
                                metadata={"activation": activation},
                            )
                        )

        except Exception as e:
            self.logger.debug(f"Knowledge graph search failed: {e}")

        return results

    def _search_episodic_memory(self, context: SearchContext) -> list[RetrievalResult]:
        """Search episodic memory for past events."""
        results = []

        if self._agent_memory is None:
            return results

        try:
            query_words = context.query.lower().split()

            for entry in self._agent_memory.episodic.values():
                content = entry.content
                if isinstance(content, dict):
                    event = str(content.get("event", "")).lower()
                    outcome = str(content.get("outcome", "")).lower()
                    text = event + " " + outcome

                    matches = sum(1 for word in query_words if word in text)
                    if matches > 0:
                        relevance = matches / len(query_words)

                        results.append(
                            RetrievalResult(
                                source=RetrievalSource.EPISODIC_MEMORY,
                                content=content,
                                relevance_score=min(1.0, relevance * entry.importance),
                                confidence=entry.importance,
                                timestamp=entry.timestamp,
                                metadata={"entry_id": entry.entry_id},
                            )
                        )

        except Exception as e:
            self.logger.debug(f"Episodic memory search failed: {e}")

        return results

    def _search_semantic_memory(self, context: SearchContext) -> list[RetrievalResult]:
        """Search semantic memory for facts."""
        results = []

        if self._agent_memory is None:
            return results

        try:
            # Use keyword search
            keyword_results = self._agent_memory.search_semantic(context.query)

            for entry in keyword_results:
                content = entry.content
                if isinstance(content, dict):
                    results.append(
                        RetrievalResult(
                            source=RetrievalSource.SEMANTIC_MEMORY,
                            content=content,
                            relevance_score=content.get("confidence", 0.5),
                            confidence=content.get("confidence", 0.5),
                            timestamp=entry.timestamp,
                            metadata={
                                "entry_id": entry.entry_id,
                                "category": content.get("category", "unknown"),
                            },
                        )
                    )

        except Exception as e:
            self.logger.debug(f"Semantic memory search failed: {e}")

        return results

    def _search_detection_log(self, context: SearchContext) -> list[RetrievalResult]:
        """Search detection history."""
        results = []

        query_words = context.query.lower().split()

        for log_entry in self._detection_log[-100:]:  # Last 100 entries
            detection = log_entry.get("detection", {})
            domain = log_entry.get("domain", "")

            # Filter by domain if specified
            if context.domain and domain != context.domain:
                continue

            # Check for anomaly-related queries
            if detection.get("anomaly_detected", False):
                # Score based on query relevance
                relevance = 0.3  # Base relevance for anomalies

                if "anomaly" in context.query.lower():
                    relevance += 0.3
                if "detection" in context.query.lower():
                    relevance += 0.2
                if domain and domain.lower() in context.query.lower():
                    relevance += 0.2

                if relevance > context.min_relevance:
                    results.append(
                        RetrievalResult(
                            source=RetrievalSource.DETECTION_LOG,
                            content={
                                "anomaly_score": detection.get("anomaly_score", 0),
                                "severity": detection.get("severity", 0),
                                "confidence": detection.get("confidence", 0),
                                "domain": domain,
                            },
                            relevance_score=min(1.0, relevance),
                            confidence=detection.get("confidence", 0.5),
                            timestamp=log_entry.get("timestamp", 0),
                            metadata={"domain": domain},
                        )
                    )

        return results

    def _search_pattern_history(self, context: SearchContext) -> list[RetrievalResult]:
        """Search pattern history from memory surface."""
        results = []

        if self._memory_surface is None:
            return results

        try:
            # Get pattern statistics
            for pattern_key, registry in self._memory_surface._pattern_registry.items():
                if isinstance(registry, dict):
                    frequency = registry.get("frequency", 0)

                    if frequency > 0:
                        relevance = 0.3  # Base

                        if "pattern" in context.query.lower():
                            relevance += 0.3
                        if "frequency" in context.query.lower():
                            relevance += 0.2

                        results.append(
                            RetrievalResult(
                                source=RetrievalSource.PATTERN_HISTORY,
                                content={
                                    "pattern_key": pattern_key,
                                    "frequency": frequency,
                                    "first_seen": registry.get("first_seen"),
                                    "last_seen": registry.get("last_seen"),
                                },
                                relevance_score=min(1.0, relevance),
                                confidence=min(
                                    1.0, frequency / 10
                                ),  # More frequent = more confident
                                timestamp=registry.get("last_seen", time.time()),
                                metadata={"pattern_key": pattern_key},
                            )
                        )

        except Exception as e:
            self.logger.debug(f"Pattern history search failed: {e}")

        return results

    def _generate_summary(
        self,
        query: str,
        results: list[RetrievalResult],
        intent: QueryIntent,
    ) -> str:
        """Generate summary of search results."""
        if not results:
            return f"No results found for '{query}'."

        n_results = len(results)
        top_source = results[0].source.value
        top_relevance = results[0].relevance_score

        summary_parts = [
            f"Found {n_results} result(s) for '{query}'.",
            f"Top result from {top_source} (relevance: {top_relevance:.0%}).",
        ]

        # Add intent-specific summary
        if intent == QueryIntent.SEARCH_EVENTS:
            event_count = sum(1 for r in results if r.source == RetrievalSource.EPISODIC_MEMORY)
            if event_count > 0:
                summary_parts.append(f"{event_count} historical event(s) found.")

        elif intent == QueryIntent.SEARCH_PATTERNS:
            pattern_count = sum(1 for r in results if r.source == RetrievalSource.PATTERN_HISTORY)
            if pattern_count > 0:
                summary_parts.append(f"{pattern_count} pattern(s) identified.")

        return " ".join(summary_parts)

    def get_statistics(self) -> dict[str, Any]:
        """Get retriever statistics."""
        return {
            "search_count": self._search_count,
            "total_results_returned": self._total_results_returned,
            "detection_log_size": len(self._detection_log),
            "knowledge_graph_connected": self._knowledge_graph is not None,
            "agent_memory_connected": self._agent_memory is not None,
            "memory_surface_connected": self._memory_surface is not None,
        }
