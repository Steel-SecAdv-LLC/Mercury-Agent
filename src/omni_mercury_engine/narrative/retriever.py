# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Knowledge Retriever - Unified Information Search for Mercury Agent.

Provides unified search across all Mercury's knowledge sources:
- Knowledge Graph (semantic search, traversal, similarity)
- Agent Memory (episodic events, semantic facts)
- Pattern History (historical anomalies, predictions)
- External Sources (when configured)

This enables Mercury to "know" things and reference them in conversation,
supporting truth-dense responses with evidence backing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from omni_mercury_engine.narrative.detection_memory import DetectionMemoryStore

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
    #: Who is asking. Detection-log entries are visible only to the principal
    #: that recorded them (or to everyone when recorded without one).
    principal: str | None = None
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


#: The whitelist's catch-all label (``cognitive.ethical_bounding.sanitize_domain``
#: collapses every unknown domain to it). It means "no domain was known", not
#: a domain of its own, so it matches everything on either side.
_GENERAL_DOMAIN = "general"

#: Query stems that name an anomaly, a detection, or a time frame. Substring
#: matched against the lower-cased query so inflections count.
_ANOMALY_STEMS: tuple[str, ...] = ("anomal", "outlier", "unusual", "flag", "alert", "incident")
_DETECTION_STEMS: tuple[str, ...] = ("detect", "event", "finding", "result")
_RECENCY_STEMS: tuple[str, ...] = ("recent", "lately", "today", "latest", "last", "happened")


def _principal_may_see(requested: str | None, recorded: Any) -> bool:
    """Whether the asking principal may read an entry recorded by ``recorded``.

    An entry recorded without a principal (narrated directly; no account was
    involved) is visible to everyone. Every other entry is visible only to the
    principal that recorded it, ``"anonymous"`` included: unauthenticated
    callers see unauthenticated detections and nothing else. A request without
    a principal is treated as anonymous rather than as unrestricted.
    """
    if recorded is None:
        return True
    return str(recorded) == (requested or "anonymous")


def _domains_compatible(requested: str | None, recorded: str) -> bool:
    """Whether a detection recorded under ``recorded`` answers a ``requested`` query.

    Strict equality was the previous rule, and it hid every detection the
    console ran from the console's own questions: ``/detect/univariate`` and
    ``/detect/multivariate`` carry no domain, so the mirror files them under
    ``general``, while the console asks under its CONTEXT domain. ``general``
    on either side is a wildcard; two specific domains must still agree.
    """
    if not requested or requested == _GENERAL_DOMAIN:
        return True
    if not recorded or recorded == _GENERAL_DOMAIN:
        return True
    return recorded == requested


class KnowledgeRetriever:
    """Unified Knowledge Search for Mercury Agent.

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
        # Stems, matched as substrings: "anomal" covers anomaly/anomalies/
        # anomalous, "flag" covers flagged, "alert" covers alerts. The table
        # used to carry only "event"/"happened"/..., so the console's own
        # wording ("Have there been any anomalies recently?") classified as
        # UNKNOWN and the reply apologised with the detection log full.
        QueryIntent.SEARCH_EVENTS: [
            "when",
            "happened",
            "occurred",
            "event",
            "incident",
            "history",
            "anomal",
            "flag",
            "alert",
            "recent",
            "lately",
            "today",
            "latest",
        ],
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
        """Initialize Knowledge Retriever.

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
        # Durable when MERCURY_AGENT_MEMORY_PATH is set: the newest entries are
        # loaded here and every logged detection is appended, so recall
        # survives a restart. Unset, the log is this list alone.
        self._detection_store = DetectionMemoryStore.from_env()
        self._detection_log: list[dict[str, Any]] = (
            self._detection_store.load() if self._detection_store is not None else []
        )

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

    def log_detection(
        self,
        detection: dict[str, Any],
        domain: str | None = None,
        principal: str | None = None,
    ) -> None:
        """Log a detection for future retrieval.

        ``principal`` is the account the detection belongs to (``"anonymous"``
        for an unauthenticated request). The retriever is one object per
        process, shared by every caller of the voice, so without this the
        detections of one account were recallable by any other; an entry is
        now visible only to its principal, and an entry recorded without one
        (narrated directly, no account involved) stays visible to all.
        """
        entry = {
            "detection": detection,
            "domain": domain,
            "principal": principal,
            "timestamp": time.time(),
        }
        self._detection_log.append(entry)
        if self._detection_store is not None:
            try:
                self._detection_store.append(entry)
            except Exception as exc:  # pragma: no cover - disk faults must not fail a detection
                logger.warning("detection memory not persisted: %s", exc)

        # Keep last 1000 detections
        if len(self._detection_log) > 1000:
            self._detection_log = self._detection_log[-1000:]

    def detections(
        self,
        domain: str | None = None,
        since: float | None = None,
        anomalous_only: bool = False,
        principal: str | None = None,
    ) -> list[dict[str, Any]]:
        """Logged detection entries, newest first.

        Each entry is ``{"detection": {...}, "domain": str | None, "timestamp":
        float}`` exactly as :meth:`log_detection` stored it. ``domain`` filters
        with the same wildcard rule the search uses (``general`` on either side
        matches); ``since`` is an epoch-seconds lower bound; ``anomalous_only``
        keeps the runs that flagged at least one sample; ``principal`` keeps the
        entries that principal may see (see :meth:`log_detection`).

        This is what the voice's counting, summary and "latest detection"
        answers read, so those answers are built from the record and never
        from the free-text scorer.
        """
        out: list[dict[str, Any]] = []
        for entry in reversed(self._detection_log):
            if since is not None and float(entry.get("timestamp", 0.0)) < since:
                continue
            if not _domains_compatible(domain, str(entry.get("domain") or "")):
                continue
            if not _principal_may_see(principal, entry.get("principal")):
                continue
            if anomalous_only and not entry.get("detection", {}).get("anomaly_detected", False):
                continue
            out.append(entry)
        return out

    def search(
        self,
        query: str,
        domain: str | None = None,
        max_results: int | None = None,
        min_relevance: float | None = None,
        sources: list[RetrievalSource] | None = None,
        principal: str | None = None,
    ) -> SearchResponse:
        """Search across all connected knowledge sources.

        Args:
            query: Natural language query
            domain: Optional domain filter
            max_results: Maximum results to return
            min_relevance: Minimum relevance score
            sources: Specific sources to search (None = all)
            principal: Who is asking; bounds the detection-log results to the
                entries that principal recorded (see :meth:`log_detection`)

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
            principal=principal,
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
        # Equal relevance is common (every anomalous detection-log entry scores
        # the same); break ties toward the most recent, or "what happened
        # recently?" is answered with the oldest record in the log.
        filtered_results.sort(key=lambda x: (x.relevance_score, x.timestamp or 0.0), reverse=True)

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
        scores: dict[QueryIntent, int] = dict.fromkeys(QueryIntent, 0)

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
        results: list[RetrievalResult] = []

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
        results: list[RetrievalResult] = []

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
        results: list[RetrievalResult] = []

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
        # Query words available via context.query.lower().split() for future keyword matching

        query_lower = context.query.lower()
        for log_entry in self._detection_log[-100:]:  # Last 100 entries
            detection = log_entry.get("detection", {})
            domain = log_entry.get("domain", "")

            if not _domains_compatible(context.domain, domain):
                continue
            if not _principal_may_see(context.principal, log_entry.get("principal")):
                continue

            # Check for anomaly-related queries
            if detection.get("anomaly_detected", False):
                # Score based on query relevance. Stems rather than whole
                # words: "anomalies", "anomalous", "detected", "flagged" and
                # "alerts" all name a detection; the exact-token match that
                # preceded this left the base score alone, which the
                # ``>`` floor below rejects, so the console's own question
                # ("any anomalies recently?") returned nothing.
                relevance = 0.3  # Base relevance for anomalies

                if any(stem in query_lower for stem in _ANOMALY_STEMS):
                    relevance += 0.3
                if any(stem in query_lower for stem in _DETECTION_STEMS):
                    relevance += 0.2
                if any(stem in query_lower for stem in _RECENCY_STEMS):
                    relevance += 0.1
                if domain and domain != _GENERAL_DOMAIN and domain.lower() in query_lower:
                    relevance += 0.2

                if relevance > context.min_relevance:
                    results.append(
                        RetrievalResult(
                            source=RetrievalSource.DETECTION_LOG,
                            content={
                                "anomaly_score": detection.get("anomaly_score", 0),
                                "severity": detection.get("severity"),
                                "severity_label": detection.get("severity_label"),
                                "confidence": detection.get("confidence", 0),
                                "domain": domain,
                                # The record itself, so the voice can say what
                                # ran, when, how many samples were flagged and
                                # which, instead of a bare score.
                                "method": detection.get("method"),
                                "anomaly_count": detection.get("anomaly_count"),
                                "n_samples": detection.get("n_samples"),
                                "threshold": detection.get("threshold"),
                                "indices": detection.get("indices"),
                                "timestamp": log_entry.get("timestamp"),
                                "detection_id": detection.get("detection_id"),
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
        results: list[RetrievalResult] = []

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
