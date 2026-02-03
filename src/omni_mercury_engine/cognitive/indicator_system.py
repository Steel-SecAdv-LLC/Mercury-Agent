"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

from __future__ import annotations


"""
Indicator Development System

Transforms patterns into actionable indicators for threat anticipation:
- Indicator generation from anomaly patterns
- Indicator validation and refinement
- Warning and collection management
- Intelligence requirements tracking

Research Sources:
- Army FM 2-0: Indicator development and collection management
- CISA All-Source Intelligence
- Warning intelligence tradecraft
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


class IndicatorType(Enum):
    """Types of intelligence indicators."""

    PREPARATORY = "preparatory"  # Activities before an event
    EXECUTION = "execution"  # Activities during an event
    AFTERMATH = "aftermath"  # Activities after an event
    PATTERN = "pattern"  # Recurring behavioral patterns
    ANOMALY = "anomaly"  # Deviations from baseline
    THRESHOLD = "threshold"  # Value-based triggers


class IndicatorStatus(Enum):
    """Status of an indicator."""

    ACTIVE = "active"
    TRIGGERED = "triggered"
    EXPIRED = "expired"
    DEPRECATED = "deprecated"
    VALIDATING = "validating"


class WarningLevel(Enum):
    """Warning levels for indicators."""

    WATCH = "watch"  # Monitor closely
    WARNING = "warning"  # Increased concern
    ALERT = "alert"  # Immediate attention
    CRITICAL = "critical"  # Immediate action required


@dataclass
class Indicator:
    """An intelligence indicator."""

    indicator_id: str
    name: str
    description: str
    indicator_type: IndicatorType
    pattern: dict[str, Any]  # The pattern to detect
    threshold: float  # Trigger threshold
    confidence_required: float
    source_patterns: list[str]  # Patterns that generated this indicator
    domain: str
    status: IndicatorStatus = IndicatorStatus.ACTIVE
    trigger_count: int = 0
    false_positive_count: int = 0
    true_positive_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_triggered: float | None = None
    expiration: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.indicator_id,
            "name": self.name,
            "description": self.description,
            "type": self.indicator_type.value,
            "threshold": self.threshold,
            "status": self.status.value,
            "triggers": self.trigger_count,
            "precision": self.precision,
            "domain": self.domain,
        }

    @property
    def precision(self) -> float:
        """Calculate indicator precision (PPV)."""
        total = self.true_positive_count + self.false_positive_count
        return self.true_positive_count / total if total > 0 else 0.5


@dataclass
class Warning:
    """A warning generated from an indicator."""

    warning_id: str
    indicator: Indicator
    level: WarningLevel
    observation: dict[str, Any]
    confidence: float
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False
    resolved: bool = False
    resolution: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.warning_id,
            "indicator": self.indicator.name,
            "level": self.level.value,
            "confidence": self.confidence,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
        }


@dataclass
class IntelligenceRequirement:
    """A priority intelligence requirement (PIR)."""

    pir_id: str
    question: str
    priority: int  # 1 = highest
    domain: str
    associated_indicators: list[str]
    collection_tasks: list[str]
    status: str = "active"
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    answer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.pir_id,
            "question": self.question,
            "priority": self.priority,
            "domain": self.domain,
            "status": self.status,
            "indicators": self.associated_indicators,
        }


class IndicatorDevelopmentSystem:
    """
    Indicator Development and Warning System.

    Transforms anomaly detection patterns into actionable indicators:

    1. PATTERN ANALYSIS: Identify recurring patterns in anomalies
    2. INDICATOR GENERATION: Create indicators from patterns
    3. INDICATOR VALIDATION: Test and refine indicators
    4. WARNING GENERATION: Generate warnings when triggered
    5. COLLECTION MANAGEMENT: Guide intelligence collection

    This bridges anomaly detection and intelligence operations.
    """

    PHI = (1 + np.sqrt(5)) / 2  # Golden ratio

    def __init__(
        self,
        min_pattern_occurrences: int = 3,
        default_confidence_threshold: float = 0.7,
        warning_cooldown_seconds: float = 300,
        enable_auto_deprecation: bool = True,
    ):
        """
        Initialize Indicator Development System.

        Args:
            min_pattern_occurrences: Min occurrences to become indicator
            default_confidence_threshold: Default confidence for triggers
            warning_cooldown_seconds: Cooldown between same warnings
            enable_auto_deprecation: Auto-deprecate low-precision indicators
        """
        self.min_pattern_occurrences = min_pattern_occurrences
        self.default_confidence_threshold = default_confidence_threshold
        self.warning_cooldown_seconds = warning_cooldown_seconds
        self.enable_auto_deprecation = enable_auto_deprecation

        # Storage
        self._indicators: dict[str, Indicator] = {}
        self._warnings: list[Warning] = []
        self._pirs: dict[str, IntelligenceRequirement] = {}
        self._pattern_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._domain_index: dict[str, list[str]] = defaultdict(list)

        # Statistics
        self._stats = {
            "indicators_created": 0,
            "warnings_generated": 0,
            "true_positives": 0,
            "false_positives": 0,
        }

        logger.info("IndicatorDevelopmentSystem initialized")

    def develop_indicator(
        self,
        pattern: dict[str, Any],
        name: str,
        description: str,
        indicator_type: IndicatorType,
        domain: str,
        threshold: float | None = None,
        source_patterns: list[str] | None = None,
    ) -> Indicator:
        """
        Develop a new indicator from a pattern.

        Args:
            pattern: The pattern to detect
            name: Human-readable name
            description: Description of what this indicates
            indicator_type: Type of indicator
            domain: Domain (cyber, medical, etc.)
            threshold: Optional custom threshold
            source_patterns: Patterns that generated this

        Returns:
            Developed indicator
        """
        indicator_id = f"ind_{domain}_{int(time.time() * 1000)}"

        indicator = Indicator(
            indicator_id=indicator_id,
            name=name,
            description=description,
            indicator_type=indicator_type,
            pattern=pattern,
            threshold=threshold or self._calculate_optimal_threshold(pattern),
            confidence_required=self.default_confidence_threshold,
            source_patterns=source_patterns or [],
            domain=domain,
        )

        self._indicators[indicator_id] = indicator
        self._domain_index[domain].append(indicator_id)
        self._stats["indicators_created"] += 1

        logger.info(f"Developed indicator: {name} ({indicator_id})")
        return indicator

    def develop_from_anomalies(
        self,
        anomalies: list[dict[str, Any]],
        domain: str,
        min_support: float = 0.3,
    ) -> list[Indicator]:
        """
        Automatically develop indicators from anomaly patterns.

        Uses frequent pattern mining to identify recurring anomaly signatures.

        Args:
            anomalies: List of detected anomalies
            domain: Domain for indicators
            min_support: Minimum support threshold

        Returns:
            List of developed indicators
        """
        if len(anomalies) < self.min_pattern_occurrences:
            return []

        # Store in pattern history
        for anomaly in anomalies:
            pattern_key = self._extract_pattern_key(anomaly)
            self._pattern_history[pattern_key].append(anomaly)

        # Find frequent patterns
        frequent_patterns = self._find_frequent_patterns(anomalies, min_support)

        indicators = []
        for pattern, support, occurrences in frequent_patterns:
            # Generate name and description
            name = self._generate_indicator_name(pattern)
            description = self._generate_indicator_description(pattern, support)

            # Determine type
            indicator_type = self._infer_indicator_type(pattern)

            indicator = self.develop_indicator(
                pattern=pattern,
                name=name,
                description=description,
                indicator_type=indicator_type,
                domain=domain,
                source_patterns=[str(o) for o in occurrences[:5]],
            )
            indicators.append(indicator)

        logger.info(f"Developed {len(indicators)} indicators from {len(anomalies)} anomalies")
        return indicators

    def evaluate(
        self,
        observation: dict[str, Any],
        domain: str | None = None,
    ) -> list[Warning]:
        """
        Evaluate an observation against active indicators.

        Args:
            observation: Current observation to evaluate
            domain: Optional domain filter

        Returns:
            List of triggered warnings
        """
        warnings = []

        # Get relevant indicators
        if domain:
            indicator_ids = self._domain_index.get(domain, [])
        else:
            indicator_ids = list(self._indicators.keys())

        for ind_id in indicator_ids:
            indicator = self._indicators.get(ind_id)
            if not indicator or indicator.status != IndicatorStatus.ACTIVE:
                continue

            # Check if indicator matches
            match_score = self._match_pattern(observation, indicator.pattern)

            if match_score >= indicator.threshold:
                # Check cooldown
                if self._is_in_cooldown(indicator):
                    continue

                # Generate warning
                warning = self._generate_warning(indicator, observation, match_score)
                warnings.append(warning)
                self._warnings.append(warning)

                # Update indicator stats
                indicator.trigger_count += 1
                indicator.last_triggered = time.time()

        if warnings:
            self._stats["warnings_generated"] += len(warnings)
            logger.info(f"Generated {len(warnings)} warnings")

        return warnings

    def validate_indicator(
        self,
        indicator_id: str,
        is_true_positive: bool,
        feedback: dict[str, Any] | None = None,
    ) -> None:
        """
        Validate an indicator based on outcome.

        Args:
            indicator_id: Indicator to validate
            is_true_positive: Whether the trigger was correct
            feedback: Optional feedback
        """
        if indicator_id not in self._indicators:
            return

        indicator = self._indicators[indicator_id]

        if is_true_positive:
            indicator.true_positive_count += 1
            self._stats["true_positives"] += 1
        else:
            indicator.false_positive_count += 1
            self._stats["false_positives"] += 1

        # Auto-deprecation check
        if self.enable_auto_deprecation and indicator.precision < 0.3:
            if indicator.trigger_count >= 10:
                indicator.status = IndicatorStatus.DEPRECATED
                logger.warning(f"Deprecated low-precision indicator: {indicator_id}")

        logger.debug(f"Validated indicator {indicator_id}: TP={is_true_positive}")

    def create_pir(
        self,
        question: str,
        priority: int,
        domain: str,
        associated_indicators: list[str] | None = None,
    ) -> IntelligenceRequirement:
        """
        Create a Priority Intelligence Requirement.

        Args:
            question: The intelligence question
            priority: Priority level (1 = highest)
            domain: Domain
            associated_indicators: Related indicators

        Returns:
            Created PIR
        """
        pir_id = f"pir_{domain}_{int(time.time())}"

        # Generate collection tasks
        collection_tasks = self._generate_collection_tasks(question, domain)

        pir = IntelligenceRequirement(
            pir_id=pir_id,
            question=question,
            priority=priority,
            domain=domain,
            associated_indicators=associated_indicators or [],
            collection_tasks=collection_tasks,
        )

        self._pirs[pir_id] = pir
        logger.info(f"Created PIR: {pir_id}")

        return pir

    def get_collection_priorities(
        self,
        domain: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get prioritized collection guidance.

        Args:
            domain: Optional domain filter

        Returns:
            Prioritized collection tasks
        """
        priorities = []

        # From PIRs
        pirs = [p for p in self._pirs.values() if domain is None or p.domain == domain]
        pirs.sort(key=lambda p: p.priority)

        for pir in pirs:
            for task in pir.collection_tasks:
                priorities.append(
                    {
                        "task": task,
                        "priority": pir.priority,
                        "source": f"PIR: {pir.question[:50]}...",
                        "domain": pir.domain,
                    }
                )

        # From indicators needing validation
        for ind in self._indicators.values():
            if ind.status == IndicatorStatus.VALIDATING:
                priorities.append(
                    {
                        "task": f"Validate indicator: {ind.name}",
                        "priority": 3,
                        "source": "Indicator validation",
                        "domain": ind.domain,
                    }
                )

        return priorities

    def _calculate_optimal_threshold(self, pattern: dict[str, Any]) -> float:
        """Calculate optimal threshold for a pattern."""
        # Base threshold on pattern complexity
        complexity = len(pattern)
        base_threshold = 0.6

        # More complex patterns can have lower thresholds
        adjusted = base_threshold - 0.05 * min(complexity, 4)
        return max(0.4, min(0.9, adjusted))

    def _extract_pattern_key(self, anomaly: dict[str, Any]) -> str:
        """Extract a hashable key from an anomaly for pattern matching."""
        # Use type and key features
        parts = [
            anomaly.get("type", "unknown"),
            str(anomaly.get("severity", 0)),
        ]
        return "_".join(parts)

    def _find_frequent_patterns(
        self,
        anomalies: list[dict[str, Any]],
        min_support: float,
    ) -> list[tuple[dict[str, Any], float, list[dict[str, Any]]]]:
        """Find frequent patterns in anomalies."""
        # Simple frequency-based pattern finding
        pattern_counts: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for anomaly in anomalies:
            key = self._extract_pattern_key(anomaly)
            pattern_counts[key].append(anomaly)

        frequent = []
        min_count = int(len(anomalies) * min_support)

        for _key, occurrences in pattern_counts.items():
            if len(occurrences) >= max(min_count, self.min_pattern_occurrences):
                # Extract common features
                pattern = self._extract_common_features(occurrences)
                support = len(occurrences) / len(anomalies)
                frequent.append((pattern, support, occurrences))

        # Sort by support
        frequent.sort(key=lambda x: x[1], reverse=True)
        return frequent[:10]  # Top 10 patterns

    def _extract_common_features(
        self,
        occurrences: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Extract common features from multiple occurrences."""
        if not occurrences:
            return {}

        common = {}
        first = occurrences[0]

        for key, value in first.items():
            if all(o.get(key) == value for o in occurrences):
                common[key] = value
            elif all(isinstance(o.get(key), (int, float)) for o in occurrences):
                # For numeric values, use range
                values = [o[key] for o in occurrences]
                common[f"{key}_range"] = (min(values), max(values))

        return common

    def _generate_indicator_name(self, pattern: dict[str, Any]) -> str:
        """Generate a name for an indicator."""
        parts = []
        if "type" in pattern:
            parts.append(str(pattern["type"]).title())
        if "severity" in pattern:
            parts.append(f"Severity{pattern['severity']}")

        return "_".join(parts) if parts else "PatternIndicator"

    def _generate_indicator_description(
        self,
        pattern: dict[str, Any],
        support: float,
    ) -> str:
        """Generate description for an indicator."""
        return (
            f"Indicator based on pattern observed in {support:.0%} of anomalies. "
            f"Pattern features: {list(pattern.keys())}"
        )

    def _infer_indicator_type(self, pattern: dict[str, Any]) -> IndicatorType:
        """Infer indicator type from pattern."""
        if "preparation" in str(pattern).lower():
            return IndicatorType.PREPARATORY
        elif "threshold" in pattern or any(isinstance(v, tuple) for v in pattern.values()):
            return IndicatorType.THRESHOLD
        elif "anomaly" in str(pattern).lower():
            return IndicatorType.ANOMALY
        else:
            return IndicatorType.PATTERN

    def _match_pattern(
        self,
        observation: dict[str, Any],
        pattern: dict[str, Any],
    ) -> float:
        """Calculate match score between observation and pattern."""
        if not pattern:
            return 0.0

        matches: float = 0
        total = len(pattern)

        for key, expected in pattern.items():
            actual = observation.get(key)

            if key.endswith("_range") and isinstance(expected, tuple):
                # Range check
                base_key = key[:-6]
                actual = observation.get(base_key)
                if actual is not None and expected[0] <= actual <= expected[1]:
                    matches += 1
            elif actual == expected:
                matches += 1
            elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                # Fuzzy numeric match
                if abs(actual - expected) < expected * 0.2:
                    matches += 0.5

        return matches / total if total > 0 else 0.0

    def _is_in_cooldown(self, indicator: Indicator) -> bool:
        """Check if indicator is in cooldown period."""
        if indicator.last_triggered is None:
            return False
        elapsed = time.time() - indicator.last_triggered
        return elapsed < self.warning_cooldown_seconds

    def _generate_warning(
        self,
        indicator: Indicator,
        observation: dict[str, Any],
        match_score: float,
    ) -> Warning:
        """Generate a warning from a triggered indicator."""
        # Determine warning level based on match score and indicator precision
        if match_score > 0.9 and indicator.precision > 0.7:
            level = WarningLevel.CRITICAL
        elif match_score > 0.8:
            level = WarningLevel.ALERT
        elif match_score > 0.7:
            level = WarningLevel.WARNING
        else:
            level = WarningLevel.WATCH

        return Warning(
            warning_id=f"warn_{int(time.time() * 1000)}",
            indicator=indicator,
            level=level,
            observation=observation,
            confidence=match_score * indicator.precision,
        )

    def _generate_collection_tasks(
        self,
        question: str,
        domain: str,
    ) -> list[str]:
        """Generate collection tasks for a PIR."""
        tasks = [
            f"Monitor {domain} data sources for relevant information",
            f"Query historical records related to: {question[:50]}",
            f"Coordinate with {domain} subject matter experts",
        ]

        if "who" in question.lower():
            tasks.append("Identify key actors and relationships")
        if "when" in question.lower():
            tasks.append("Establish timeline of events")
        if "how" in question.lower():
            tasks.append("Analyze methods and techniques used")

        return tasks

    def get_statistics(self) -> dict[str, Any]:
        """Get system statistics."""
        active_indicators = sum(
            1 for i in self._indicators.values() if i.status == IndicatorStatus.ACTIVE
        )

        return {
            **self._stats,
            "total_indicators": len(self._indicators),
            "active_indicators": active_indicators,
            "total_warnings": len(self._warnings),
            "open_pirs": sum(1 for p in self._pirs.values() if p.status == "active"),
            "overall_precision": (
                self._stats["true_positives"]
                / (self._stats["true_positives"] + self._stats["false_positives"])
                if (self._stats["true_positives"] + self._stats["false_positives"]) > 0
                else 0.5
            ),
        }
