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
Enhanced Anomaly Detection - Memory Graph and External Data Integration

Implements Phase 4 of the neuro-symbolic evolution:
- Internal memory-driven knowledge graph for pattern analysis
- Predictive models (Bayesian networks, HMMs) trained on success/failure histories
- External API integration framework for real-time data ingestion
- Value-extraction logic with ethical filtering

Research Sources:
- Bayesian Networks (Pearl, 1988)
- Hidden Markov Models (Rabiner, 1989)
- Knowledge Graphs for AI (Hogan et al., 2021)
- Real-time Anomaly Detection (Chandola et al., 2009)

Integration:
    This module enhances the NeurosymbolicFusionEngine with advanced
    anomaly detection capabilities including external data sources.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import httpx
import numpy as np


try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

logger = logging.getLogger(__name__)


class ExternalSourceCategory(Enum):
    """High-level categories of external data sources.

    Note: This enum represents abstract source categories for the cognitive
    anomaly detection module. For specific API data source types, see
    omni_mercury_engine.data_sources.DataSourceType instead.
    """

    GEOLOGICAL = "geological"
    ENVIRONMENTAL = "environmental"
    NEWS = "news"
    FINANCIAL = "financial"
    SOCIAL = "social"
    SECURITY = "security"
    HEALTH = "health"
    CUSTOM = "custom"


class PredictionType(Enum):
    """Types of predictions."""

    ANOMALY = "anomaly"
    TREND = "trend"
    ESCALATION = "escalation"
    OPPORTUNITY = "opportunity"
    RISK = "risk"


@dataclass
class ExternalDataPoint:
    """Data point from external source."""

    source_type: ExternalSourceCategory
    source_name: str
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.8
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictiveResult:
    """Result from predictive model."""

    prediction_id: str
    prediction_type: PredictionType
    probability: float
    time_horizon: float
    explanation: str
    contributing_factors: list[str]
    confidence_interval: tuple[float, float] = (0.0, 1.0)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValueExtraction:
    """Extracted value/opportunity from anomaly."""

    extraction_id: str
    anomaly_source: str
    value_type: str
    potential_benefit: float
    ethical_score: float
    is_benevolent: bool
    recommended_action: str
    explanation: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryKnowledgeGraph:
    """
    Knowledge graph built from accumulated memories.

    Uses memory entries as nodes and relationships as edges
    to enable pattern discovery and predictive analysis.
    """

    def __init__(self) -> None:
        """Initialize memory knowledge graph."""
        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()
        else:
            self.nodes: dict[str, dict[str, Any]] = {}
            self.edges: list[tuple[str, str, dict[str, Any]]] = []

        self._node_counter = 0
        self._edge_counter = 0

    def add_memory_node(
        self,
        memory_id: str,
        memory_type: str,
        content: dict[str, Any],
        importance: float = 0.5,
    ) -> str:
        """
        Add a memory as a node in the graph.

        Args:
            memory_id: Unique memory identifier
            memory_type: Type of memory (episodic, semantic, etc.)
            content: Memory content
            importance: Memory importance score

        Returns:
            Node ID
        """
        node_id = f"mem_{memory_id}"

        if NETWORKX_AVAILABLE:
            self.graph.add_node(
                node_id,
                memory_type=memory_type,
                content=content,
                importance=importance,
                timestamp=time.time(),
            )
        else:
            self.nodes[node_id] = {
                "memory_type": memory_type,
                "content": content,
                "importance": importance,
                "timestamp": time.time(),
            }

        self._node_counter += 1
        return node_id

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add a relationship between memory nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship_type: Type of relationship
            weight: Relationship weight
            metadata: Additional metadata

        Returns:
            Edge ID
        """
        self._edge_counter += 1
        edge_id = f"edge_{self._edge_counter}"

        edge_data = {
            "edge_id": edge_id,
            "relationship_type": relationship_type,
            "weight": weight,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }

        if NETWORKX_AVAILABLE:
            self.graph.add_edge(source_id, target_id, **edge_data)
        else:
            self.edges.append((source_id, target_id, edge_data))

        return edge_id

    def find_related_memories(
        self,
        node_id: str,
        max_depth: int = 2,
        min_weight: float = 0.5,
    ) -> list[tuple[str, float]]:
        """
        Find memories related to a given node.

        Args:
            node_id: Starting node ID
            max_depth: Maximum traversal depth
            min_weight: Minimum edge weight to follow

        Returns:
            List of (node_id, relevance_score) tuples
        """
        related: list[tuple[str, float]] = []

        if NETWORKX_AVAILABLE:
            if not self.graph.has_node(node_id):
                return related

            for depth in range(1, max_depth + 1):
                try:
                    paths = nx.single_source_shortest_path_length(self.graph, node_id, cutoff=depth)
                    for target, dist in paths.items():
                        if target != node_id:
                            relevance = 1.0 / (dist + 1)
                            related.append((target, relevance))
                except nx.NetworkXError:
                    # Graph traversal failed; continue with partial results
                    pass
        else:
            visited = {node_id}
            current_level = [node_id]

            for depth in range(max_depth):
                next_level = []
                for current in current_level:
                    for src, tgt, data in self.edges:
                        if src == current and tgt not in visited:
                            if data.get("weight", 1.0) >= min_weight:
                                visited.add(tgt)
                                next_level.append(tgt)
                                relevance = 1.0 / (depth + 2)
                                related.append((tgt, relevance))
                        elif tgt == current and src not in visited:
                            if data.get("weight", 1.0) >= min_weight:
                                visited.add(src)
                                next_level.append(src)
                                relevance = 1.0 / (depth + 2)
                                related.append((src, relevance))
                current_level = next_level

        return sorted(related, key=lambda x: x[1], reverse=True)

    def compute_centrality(self) -> dict[str, float]:
        """Compute node centrality scores."""
        if NETWORKX_AVAILABLE and self.graph.number_of_nodes() > 0:
            try:
                result: dict[str, float] = nx.pagerank(self.graph, alpha=0.85)
                return result
            except Exception:
                # PageRank computation failed; return uniform centrality
                return {n: 1.0 / self.graph.number_of_nodes() for n in self.graph.nodes()}
        else:
            n_nodes = len(self.nodes)
            if n_nodes == 0:
                return {}
            return dict.fromkeys(self.nodes, 1.0 / n_nodes)

    def get_statistics(self) -> dict[str, Any]:
        """Get graph statistics."""
        if NETWORKX_AVAILABLE:
            return {
                "num_nodes": self.graph.number_of_nodes(),
                "num_edges": self.graph.number_of_edges(),
                "density": nx.density(self.graph) if self.graph.number_of_nodes() > 1 else 0,
            }
        else:
            return {
                "num_nodes": len(self.nodes),
                "num_edges": len(self.edges),
                "density": len(self.edges) / (len(self.nodes) ** 2) if self.nodes else 0,
            }


class BayesianPredictor:
    """
    Bayesian predictor for anomaly forecasting.

    Uses Beta-Bernoulli conjugate prior for probability estimation
    based on success/failure histories.
    """

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        """
        Initialize Bayesian predictor.

        Args:
            prior_alpha: Prior alpha parameter (pseudo-successes)
            prior_beta: Prior beta parameter (pseudo-failures)
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.contexts: dict[str, dict[str, float]] = {}

    def update(self, context: str, success: bool) -> None:
        """
        Update beliefs based on observation.

        Args:
            context: Context identifier
            success: Whether the observation was a success
        """
        if context not in self.contexts:
            self.contexts[context] = {
                "alpha": self.prior_alpha,
                "beta": self.prior_beta,
            }

        if success:
            self.contexts[context]["alpha"] += 1
        else:
            self.contexts[context]["beta"] += 1

    def predict(self, context: str) -> tuple[float, tuple[float, float]]:
        """
        Predict probability for a context.

        Args:
            context: Context identifier

        Returns:
            Tuple of (probability, (lower_bound, upper_bound))
        """
        if context not in self.contexts:
            alpha = self.prior_alpha
            beta = self.prior_beta
        else:
            alpha = self.contexts[context]["alpha"]
            beta = self.contexts[context]["beta"]

        mean = alpha / (alpha + beta)

        variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        std = np.sqrt(variance)

        lower = max(0, mean - 2 * std)
        upper = min(1, mean + 2 * std)

        return mean, (lower, upper)

    def get_confidence(self, context: str) -> float:
        """Get confidence in prediction for context."""
        if context not in self.contexts:
            return 0.5

        alpha = self.contexts[context]["alpha"]
        beta = self.contexts[context]["beta"]
        n = alpha + beta - self.prior_alpha - self.prior_beta

        return min(1.0, n / 100)


class HiddenMarkovPredictor:
    """
    Hidden Markov Model for sequence-based anomaly prediction.

    Lightweight implementation for detecting state transitions
    that may indicate anomalies.
    """

    def __init__(self, n_states: int = 3) -> None:
        """
        Initialize HMM predictor.

        Args:
            n_states: Number of hidden states
        """
        self.n_states = n_states

        np.random.seed(42)
        self.transition_matrix = np.ones((n_states, n_states)) / n_states
        self.emission_probs: dict[str, np.ndarray[Any, Any]] = {}
        self.initial_probs = np.ones(n_states) / n_states

        self.state_history: list[int] = []
        self.observation_history: list[str] = []

    def observe(self, observation: str) -> int:
        """
        Process an observation and update state.

        Args:
            observation: Observed symbol

        Returns:
            Most likely current state
        """
        if observation not in self.emission_probs:
            self.emission_probs[observation] = np.random.dirichlet(np.ones(self.n_states))

        if not self.state_history:
            state_probs = self.initial_probs * self.emission_probs[observation]
        else:
            prev_state = self.state_history[-1]
            state_probs = self.transition_matrix[prev_state] * self.emission_probs[observation]

        state_probs /= state_probs.sum() + 1e-10
        current_state = int(np.argmax(state_probs))

        self.state_history.append(current_state)
        self.observation_history.append(observation)

        if len(self.state_history) > 1:
            prev = self.state_history[-2]
            self.transition_matrix[prev, current_state] += 0.1
            self.transition_matrix[prev] /= self.transition_matrix[prev].sum()

        return current_state

    def predict_next_state(self) -> tuple[int, float]:
        """
        Predict the next most likely state.

        Returns:
            Tuple of (predicted_state, probability)
        """
        if not self.state_history:
            return 0, 1.0 / self.n_states

        current_state = self.state_history[-1]
        next_probs = self.transition_matrix[current_state]
        predicted_state = int(np.argmax(next_probs))
        probability = float(next_probs[predicted_state])

        return predicted_state, probability

    def detect_anomaly(self, threshold: float = 0.1) -> bool:
        """
        Detect if current state transition is anomalous.

        Args:
            threshold: Probability threshold for anomaly

        Returns:
            True if anomalous transition detected
        """
        if len(self.state_history) < 2:
            return False

        prev_state = self.state_history[-2]
        current_state = self.state_history[-1]
        transition_prob = self.transition_matrix[prev_state, current_state]

        return bool(transition_prob < threshold)


class ExternalDataSource(ABC):
    """Abstract base class for external data sources."""

    @abstractmethod
    def fetch(self) -> list[ExternalDataPoint]:
        """Fetch data from the source."""
        pass

    @abstractmethod
    def get_source_type(self) -> ExternalSourceCategory:
        """Get the type of this data source."""
        pass


class SimulatedGeologicalSource(ExternalDataSource):
    """Simulated geological data source (USGS-style) for development/testing.

    Generates synthetic earthquake data for testing anomaly detection pipelines.
    In production, implement a real USGS API client that extends ExternalDataSource.
    """

    def __init__(self) -> None:
        self.source_name = "simulated_usgs"

    def fetch(self) -> list[ExternalDataPoint]:
        """Fetch simulated geological data for testing."""
        return [
            ExternalDataPoint(
                source_type=ExternalSourceCategory.GEOLOGICAL,
                source_name=self.source_name,
                data={
                    "event_type": "earthquake",
                    "magnitude": np.random.uniform(2.0, 6.0),
                    "depth_km": np.random.uniform(5, 50),
                    "latitude": np.random.uniform(-90, 90),
                    "longitude": np.random.uniform(-180, 180),
                },
                confidence=0.9,
            )
        ]

    def get_source_type(self) -> ExternalSourceCategory:
        return ExternalSourceCategory.GEOLOGICAL


class SimulatedEnvironmentalSource(ExternalDataSource):
    """Simulated environmental data source (NOAA-style) for development/testing.

    Generates synthetic weather and environmental data for testing anomaly detection.
    In production, implement a real NOAA API client that extends ExternalDataSource.
    """

    def __init__(self) -> None:
        self.source_name = "simulated_noaa"

    def fetch(self) -> list[ExternalDataPoint]:
        """Fetch simulated environmental data for testing."""
        return [
            ExternalDataPoint(
                source_type=ExternalSourceCategory.ENVIRONMENTAL,
                source_name=self.source_name,
                data={
                    "event_type": "weather_alert",
                    "severity": np.random.choice(["low", "moderate", "high", "extreme"]),
                    "temperature_c": np.random.uniform(-20, 45),
                    "wind_speed_kmh": np.random.uniform(0, 150),
                    "precipitation_mm": np.random.uniform(0, 100),
                },
                confidence=0.85,
            )
        ]

    def get_source_type(self) -> ExternalSourceCategory:
        return ExternalSourceCategory.ENVIRONMENTAL


class USGSEarthquakeSource(ExternalDataSource):
    """Real USGS Earthquake API client for production use.

    Fetches real-time earthquake data from the USGS Earthquake Hazards Program API.
    API Documentation: https://earthquake.usgs.gov/fdsnws/event/1/

    This is a free API that requires no authentication.
    Rate limits are generous for reasonable use cases.

    Example:
        source = USGSEarthquakeSource(min_magnitude=4.0, max_results=10)
        detector.register_external_source("usgs_earthquakes", source)
    """

    USGS_API_BASE = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    def __init__(
        self,
        min_magnitude: float = 2.5,
        max_results: int = 20,
        days_back: int = 7,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Initialize USGS earthquake data source.

        Args:
            min_magnitude: Minimum earthquake magnitude to fetch (default 2.5)
            max_results: Maximum number of results to return (default 20)
            days_back: Number of days to look back for earthquakes (default 7)
            timeout_seconds: HTTP request timeout in seconds (default 30)
        """
        self.source_name = "usgs_earthquake"
        self.min_magnitude = min_magnitude
        self.max_results = max_results
        self.days_back = days_back
        self.timeout_seconds = timeout_seconds
        self._client = httpx.Client(timeout=timeout_seconds)

    def fetch(self) -> list[ExternalDataPoint]:
        """Fetch real earthquake data from USGS API.

        Returns:
            List of ExternalDataPoint objects with earthquake data.
            Returns empty list if API call fails (with warning logged).
        """
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(days=self.days_back)

        params = {
            "format": "geojson",
            "starttime": start_time.strftime("%Y-%m-%d"),
            "endtime": end_time.strftime("%Y-%m-%d"),
            "minmagnitude": self.min_magnitude,
            "limit": self.max_results,
            "orderby": "time",
        }

        try:
            response = self._client.get(self.USGS_API_BASE, params=params)
            response.raise_for_status()
            data = response.json()

            results: list[ExternalDataPoint] = []
            for feature in data.get("features", []):
                props = feature.get("properties", {})
                coords = feature.get("geometry", {}).get("coordinates", [0, 0, 0])

                magnitude = props.get("mag", 0.0)
                confidence = min(0.99, 0.7 + (magnitude / 10.0) * 0.3)

                results.append(
                    ExternalDataPoint(
                        source_type=ExternalSourceCategory.GEOLOGICAL,
                        source_name=self.source_name,
                        data={
                            "event_type": "earthquake",
                            "event_id": feature.get("id", "unknown"),
                            "magnitude": magnitude,
                            "magnitude_type": props.get("magType", "unknown"),
                            "depth_km": coords[2] if len(coords) > 2 else 0.0,
                            "latitude": coords[1] if len(coords) > 1 else 0.0,
                            "longitude": coords[0] if len(coords) > 0 else 0.0,
                            "place": props.get("place", "Unknown location"),
                            "time_utc": props.get("time", 0),
                            "tsunami_alert": props.get("tsunami", 0) == 1,
                            "felt_reports": props.get("felt", 0),
                            "alert_level": props.get("alert", "none"),
                            "significance": props.get("sig", 0),
                        },
                        confidence=confidence,
                        timestamp=float(int(props.get("time", 0) or 0) / 1000),
                    )
                )

            logger.info(f"USGS API: Fetched {len(results)} earthquakes (M>={self.min_magnitude})")
            return results

        except httpx.HTTPStatusError as e:
            logger.warning(f"USGS API HTTP error: {e.response.status_code}")
            return []
        except httpx.RequestError as e:
            logger.warning(f"USGS API request error: {e}")
            return []
        except (KeyError, ValueError) as e:
            logger.warning(f"USGS API parse error: {e}")
            return []

    def get_source_type(self) -> ExternalSourceCategory:
        return ExternalSourceCategory.GEOLOGICAL

    def __del__(self) -> None:
        """Clean up HTTP client on deletion."""
        if hasattr(self, "_client"):
            self._client.close()


class NOAAWeatherSource(ExternalDataSource):
    """Real NOAA Weather API client for production use.

    Fetches real-time weather alerts from the National Weather Service API.
    API Documentation: https://www.weather.gov/documentation/services-web-api

    This is a free API that requires no authentication.
    A User-Agent header is required per NOAA API guidelines.

    Example:
        source = NOAAWeatherSource(state="CA")
        detector.register_external_source("noaa_weather", source)
    """

    NOAA_API_BASE = "https://api.weather.gov"

    def __init__(
        self,
        state: str | None = None,
        zone: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Initialize NOAA weather data source.

        Args:
            state: Two-letter state code to filter alerts (e.g., "CA", "TX")
            zone: Specific NWS zone ID (e.g., "CAZ006")
            timeout_seconds: HTTP request timeout in seconds (default 30)
        """
        self.source_name = "noaa_weather"
        self.state = state
        self.zone = zone
        self.timeout_seconds = timeout_seconds
        self._client = httpx.Client(
            timeout=timeout_seconds,
            headers={
                "User-Agent": "MercuryAgent/1.2.0 (steel.sa.llc@gmail.com)",
                "Accept": "application/geo+json",
            },
        )

    def fetch(self) -> list[ExternalDataPoint]:
        """Fetch real weather alerts from NOAA API.

        Returns:
            List of ExternalDataPoint objects with weather alert data.
            Returns empty list if API call fails (with warning logged).
        """
        url = f"{self.NOAA_API_BASE}/alerts/active"
        params: dict[str, str] = {"status": "actual"}

        if self.state:
            params["area"] = self.state
        if self.zone:
            params["zone"] = self.zone

        try:
            response = self._client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            results: list[ExternalDataPoint] = []
            for feature in data.get("features", []):
                props = feature.get("properties", {})

                severity = props.get("severity", "Unknown")
                severity_map = {"Extreme": 0.99, "Severe": 0.9, "Moderate": 0.75, "Minor": 0.6}
                confidence = severity_map.get(severity, 0.5)

                results.append(
                    ExternalDataPoint(
                        source_type=ExternalSourceCategory.ENVIRONMENTAL,
                        source_name=self.source_name,
                        data={
                            "event_type": "weather_alert",
                            "alert_id": props.get("id", "unknown"),
                            "event": props.get("event", "Unknown"),
                            "severity": severity,
                            "certainty": props.get("certainty", "Unknown"),
                            "urgency": props.get("urgency", "Unknown"),
                            "headline": props.get("headline", ""),
                            "description": props.get("description", "")[:500],
                            "instruction": props.get("instruction", "")[:500],
                            "area_desc": props.get("areaDesc", "Unknown area"),
                            "effective": props.get("effective", ""),
                            "expires": props.get("expires", ""),
                            "sender": props.get("senderName", "NWS"),
                        },
                        confidence=confidence,
                    )
                )

            logger.info(f"NOAA API: Fetched {len(results)} active weather alerts")
            return results

        except httpx.HTTPStatusError as e:
            logger.warning(f"NOAA API HTTP error: {e.response.status_code}")
            return []
        except httpx.RequestError as e:
            logger.warning(f"NOAA API request error: {e}")
            return []
        except (KeyError, ValueError) as e:
            logger.warning(f"NOAA API parse error: {e}")
            return []

    def get_source_type(self) -> ExternalSourceCategory:
        return ExternalSourceCategory.ENVIRONMENTAL

    def __del__(self) -> None:
        """Clean up HTTP client on deletion."""
        if hasattr(self, "_client"):
            self._client.close()


class ExternalDataIntegrator:
    """
    Integrates external data sources for real-time anomaly detection.

    Manages multiple data sources and aligns external patterns
    with internal memory patterns.
    """

    def __init__(self) -> None:
        """Initialize external data integrator."""
        self.sources: dict[str, ExternalDataSource] = {}
        self.data_buffer: list[ExternalDataPoint] = []
        self.max_buffer_size = 1000

    def register_source(self, name: str, source: ExternalDataSource) -> None:
        """
        Register an external data source.

        Args:
            name: Source name
            source: Data source instance
        """
        self.sources[name] = source
        logger.info(f"Registered external source: {name}")

    def fetch_all(self) -> list[ExternalDataPoint]:
        """
        Fetch data from all registered sources.

        Returns:
            List of data points from all sources
        """
        all_data = []

        for name, source in self.sources.items():
            try:
                data = source.fetch()
                all_data.extend(data)
                logger.debug(f"Fetched {len(data)} points from {name}")
            except Exception as e:
                logger.error(f"Error fetching from {name}: {e}")

        self.data_buffer.extend(all_data)
        if len(self.data_buffer) > self.max_buffer_size:
            self.data_buffer = self.data_buffer[-self.max_buffer_size :]

        return all_data

    def align_with_internal(
        self,
        internal_patterns: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Align external data with internal patterns.

        Args:
            internal_patterns: Patterns from internal analysis

        Returns:
            List of aligned pattern pairs
        """
        alignments = []

        for external in self.data_buffer[-100:]:
            for internal in internal_patterns:
                similarity = self._compute_similarity(external, internal)
                if similarity > 0.5:
                    alignments.append(
                        {
                            "external": external,
                            "internal": internal,
                            "similarity": similarity,
                            "timestamp": time.time(),
                        }
                    )

        return alignments

    def _compute_similarity(
        self,
        external: ExternalDataPoint,
        internal: dict[str, Any],
    ) -> float:
        """Compute similarity between external and internal data."""
        score = 0.0

        if external.source_type.value in str(internal.get("type", "")):
            score += 0.3

        ext_conf = external.confidence
        int_conf = internal.get("confidence", 0.5)
        score += 0.3 * (1 - abs(ext_conf - int_conf))

        time_diff = abs(external.timestamp - internal.get("timestamp", time.time()))
        if time_diff < 3600:
            score += 0.4 * (1 - time_diff / 3600)

        return float(score)

    def get_statistics(self) -> dict[str, Any]:
        """Get integrator statistics."""
        return {
            "num_sources": len(self.sources),
            "buffer_size": len(self.data_buffer),
            "source_types": [s.get_source_type().value for s in self.sources.values()],
        }


class ValueExtractor:
    """
    Extract value/opportunities from detected anomalies.

    Identifies benevolent intervention opportunities while
    filtering through ethical constraints.
    """

    def __init__(self, benevolence_threshold: float = 0.99) -> None:
        """
        Initialize value extractor.

        Args:
            benevolence_threshold: Minimum benevolence score for actions
        """
        self.benevolence_threshold = benevolence_threshold
        self._extraction_counter = 0

    def extract(
        self,
        anomaly: dict[str, Any],
        ethical_score: float,
    ) -> ValueExtraction | None:
        """
        Extract value from an anomaly if ethically appropriate.

        Args:
            anomaly: Anomaly data
            ethical_score: Ethical evaluation score

        Returns:
            ValueExtraction if opportunity found, None otherwise
        """
        is_benevolent = ethical_score >= self.benevolence_threshold

        if not is_benevolent:
            return None

        value_type = self._determine_value_type(anomaly)
        potential_benefit = self._estimate_benefit(anomaly, value_type)
        recommended_action = self._recommend_action(anomaly, value_type)

        if potential_benefit < 0.1:
            return None

        self._extraction_counter += 1
        extraction_id = f"extract_{self._extraction_counter:06d}"

        return ValueExtraction(
            extraction_id=extraction_id,
            anomaly_source=anomaly.get("id", "unknown"),
            value_type=value_type,
            potential_benefit=potential_benefit,
            ethical_score=ethical_score,
            is_benevolent=is_benevolent,
            recommended_action=recommended_action,
            explanation=self._generate_explanation(anomaly, value_type, potential_benefit),
        )

    def _determine_value_type(self, anomaly: dict[str, Any]) -> str:
        """Determine the type of value opportunity."""
        anomaly_type = anomaly.get("type", "unknown")

        if anomaly_type in ["escalation", "trend"]:
            return "early_warning"
        elif anomaly_type == "novelty":
            return "discovery"
        elif anomaly_type == "correlation":
            return "insight"
        else:
            return "monitoring"

    def _estimate_benefit(self, anomaly: dict[str, Any], value_type: str) -> float:
        """Estimate potential benefit of acting on anomaly."""
        base_benefit = anomaly.get("confidence", 0.5)

        type_multipliers = {
            "early_warning": 1.5,
            "discovery": 1.3,
            "insight": 1.2,
            "monitoring": 1.0,
        }

        multiplier = type_multipliers.get(value_type, 1.0)
        return float(min(1.0, base_benefit * multiplier))

    def _recommend_action(self, anomaly: dict[str, Any], value_type: str) -> str:
        """Recommend action based on anomaly and value type."""
        recommendations = {
            "early_warning": "Alert stakeholders and prepare response protocols",
            "discovery": "Document findings and investigate further",
            "insight": "Integrate into knowledge base for future reference",
            "monitoring": "Continue observation and log for pattern analysis",
        }
        return recommendations.get(value_type, "Review and assess")

    def _generate_explanation(
        self,
        anomaly: dict[str, Any],
        value_type: str,
        benefit: float,
    ) -> str:
        """Generate explanation for value extraction."""
        return (
            f"Identified {value_type} opportunity from anomaly "
            f"'{anomaly.get('id', 'unknown')}' with estimated benefit of {benefit:.0%}. "
            f"This represents a benevolent intervention opportunity."
        )


class EnhancedAnomalyDetector:
    """
    Enhanced Anomaly Detector with memory graph and external integration.

    Main interface for Phase 4 capabilities combining internal
    memory-driven patterns with external data sources.

    The detector supports both real and simulated data sources. By default,
    simulated sources are registered for development/testing. In production,
    register real data sources using register_external_source().
    """

    def __init__(
        self,
        benevolence_threshold: float = 0.99,
        hmm_states: int = 3,
        use_simulated_sources: bool = True,
    ):
        """
        Initialize enhanced anomaly detector.

        Args:
            benevolence_threshold: Minimum benevolence score
            hmm_states: Number of HMM states
            use_simulated_sources: If True, register simulated data sources for
                development/testing. Set to False in production and register
                real data sources using register_external_source().
        """
        self.benevolence_threshold = benevolence_threshold
        self.use_simulated_sources = use_simulated_sources

        self.memory_graph = MemoryKnowledgeGraph()
        self.bayesian_predictor = BayesianPredictor()
        self.hmm_predictor = HiddenMarkovPredictor(n_states=hmm_states)
        self.external_integrator = ExternalDataIntegrator()
        self.value_extractor = ValueExtractor(benevolence_threshold=benevolence_threshold)

        if use_simulated_sources:
            self._register_simulated_sources()
        self._prediction_counter = 0

        mode = "simulated" if use_simulated_sources else "production"
        logger.info(f"EnhancedAnomalyDetector initialized in {mode} mode")

    def _register_simulated_sources(self) -> None:
        """Register simulated data sources for development/testing.

        These sources generate synthetic data for testing purposes.
        In production, use register_external_source() to add real data feeds.
        """
        self.external_integrator.register_source(
            "geological_simulated", SimulatedGeologicalSource()
        )
        self.external_integrator.register_source(
            "environmental_simulated", SimulatedEnvironmentalSource()
        )

    def register_external_source(self, name: str, source: ExternalDataSource) -> None:
        """Register a real external data source for production use.

        Args:
            name: Unique name for the data source
            source: ExternalDataSource implementation (e.g., USGS, NOAA API client)
        """
        self.external_integrator.register_source(name, source)

    def add_memory(
        self,
        memory_id: str,
        memory_type: str,
        content: dict[str, Any],
        importance: float = 0.5,
        related_to: list[str] | None = None,
    ) -> str:
        """
        Add a memory to the knowledge graph.

        Args:
            memory_id: Memory identifier
            memory_type: Type of memory
            content: Memory content
            importance: Importance score
            related_to: List of related memory IDs

        Returns:
            Node ID in the graph
        """
        node_id = self.memory_graph.add_memory_node(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content,
            importance=importance,
        )

        if related_to:
            for related_id in related_to:
                related_node = f"mem_{related_id}"
                self.memory_graph.add_relationship(
                    source_id=node_id,
                    target_id=related_node,
                    relationship_type="related_to",
                    weight=0.8,
                )

        return node_id

    def update_predictor(self, context: str, success: bool) -> None:
        """
        Update Bayesian predictor with observation.

        Args:
            context: Context identifier
            success: Whether observation was successful
        """
        self.bayesian_predictor.update(context, success)

    def observe_sequence(self, observation: str) -> int:
        """
        Process sequential observation through HMM.

        Args:
            observation: Observation symbol

        Returns:
            Current state
        """
        return self.hmm_predictor.observe(observation)

    def predict(
        self,
        context: str,
        include_external: bool = True,
    ) -> PredictiveResult:
        """
        Generate prediction for a context.

        Args:
            context: Context identifier
            include_external: Whether to include external data

        Returns:
            PredictiveResult with prediction details
        """
        self._prediction_counter += 1
        prediction_id = f"pred_{self._prediction_counter:06d}"

        bayes_prob, bayes_interval = self.bayesian_predictor.predict(context)
        # Note: bayes_confidence available via self.bayesian_predictor.get_confidence(context)

        hmm_state, hmm_prob = self.hmm_predictor.predict_next_state()
        hmm_anomaly = self.hmm_predictor.detect_anomaly()

        combined_prob = 0.6 * bayes_prob + 0.4 * hmm_prob

        if hmm_anomaly:
            prediction_type = PredictionType.ANOMALY
        elif combined_prob > 0.7:
            prediction_type = PredictionType.RISK
        elif combined_prob > 0.5:
            prediction_type = PredictionType.TREND
        else:
            prediction_type = PredictionType.OPPORTUNITY

        contributing_factors = [
            f"Bayesian probability: {bayes_prob:.2%}",
            f"HMM state: {hmm_state}, probability: {hmm_prob:.2%}",
        ]

        if include_external:
            external_data = self.external_integrator.fetch_all()
            if external_data:
                contributing_factors.append(f"External data points: {len(external_data)}")

        explanation = self._generate_prediction_explanation(
            prediction_type, combined_prob, contributing_factors
        )

        return PredictiveResult(
            prediction_id=prediction_id,
            prediction_type=prediction_type,
            probability=combined_prob,
            time_horizon=3600.0,
            explanation=explanation,
            contributing_factors=contributing_factors,
            confidence_interval=bayes_interval,
        )

    def extract_value(
        self,
        anomaly: dict[str, Any],
        ethical_score: float,
    ) -> ValueExtraction | None:
        """
        Extract value from anomaly if ethically appropriate.

        Args:
            anomaly: Anomaly data
            ethical_score: Ethical evaluation score

        Returns:
            ValueExtraction if opportunity found
        """
        return self.value_extractor.extract(anomaly, ethical_score)

    def analyze_memory_patterns(self, memory_id: str) -> dict[str, Any]:
        """
        Analyze patterns related to a memory.

        Args:
            memory_id: Memory identifier

        Returns:
            Analysis results
        """
        node_id = f"mem_{memory_id}"
        related = self.memory_graph.find_related_memories(node_id)
        centrality = self.memory_graph.compute_centrality()

        return {
            "memory_id": memory_id,
            "related_memories": related[:10],
            "centrality_score": centrality.get(node_id, 0.0),
            "graph_stats": self.memory_graph.get_statistics(),
        }

    def _generate_prediction_explanation(
        self,
        prediction_type: PredictionType,
        probability: float,
        factors: list[str],
    ) -> str:
        """Generate explanation for prediction."""
        return (
            f"Prediction: {prediction_type.value.upper()} with {probability:.0%} probability. "
            f"Based on: {'; '.join(factors[:3])}"
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get detector statistics."""
        return {
            "predictions_made": self._prediction_counter,
            "memory_graph": self.memory_graph.get_statistics(),
            "external_sources": self.external_integrator.get_statistics(),
            "bayesian_contexts": len(self.bayesian_predictor.contexts),
            "hmm_observations": len(self.hmm_predictor.observation_history),
        }
