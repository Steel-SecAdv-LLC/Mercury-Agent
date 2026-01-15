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
Proactive Monitor - Background Vigilance with Initiative

Enables Mercury to proactively initiate communication when patterns warrant it.
This is NOT about engagement - it's about duty to inform when truth demands it.

Key Principles:
    - Speak up when silence would be a disservice
    - Initiative thresholds prevent noise
    - Vigilance shaped by omni-scalars
    - Pattern accumulation triggers escalation
    - Domain-specific vigilance levels

Features:
    - Background monitoring threads (non-blocking)
    - Initiative threshold configuration per domain
    - Pattern accumulation and escalation logic
    - Scheduled vigilance reports
    - Integration with NarrativeEngine for communication

This enables "aliveness" through responsible initiative, not performative behavior.
"""

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np


logger = logging.getLogger(__name__)


class VigilanceLevel(Enum):
    """Vigilance levels for monitoring intensity."""

    PASSIVE = "passive"  # Log only, no initiative
    ATTENTIVE = "attentive"  # Monitor, alert on high severity
    VIGILANT = "vigilant"  # Active monitoring, moderate thresholds
    HEIGHTENED = "heightened"  # Post-incident, lower thresholds
    CRITICAL = "critical"  # Emergency mode, immediate alerting


class InitiativeType(Enum):
    """Types of proactive initiative."""

    ANOMALY_ALERT = "anomaly_alert"  # Detected anomaly
    PATTERN_EMERGENCE = "pattern_emergence"  # New pattern detected
    ESCALATION = "escalation"  # Pattern accumulation escalation
    PREDICTION = "prediction"  # Predicted future anomaly
    CALIBRATION = "calibration"  # Model recalibration needed
    MEMORY_INSIGHT = "memory_insight"  # Historical pattern match
    SCHEDULED_REPORT = "scheduled_report"  # Regular vigilance report


@dataclass
class InitiativeThreshold:
    """Configuration for when to take initiative."""

    min_anomaly_score: float = 0.7  # Minimum score to alert
    min_confidence: float = 0.6  # Minimum confidence to alert
    min_severity: float = 0.5  # Minimum severity to alert
    pattern_accumulation_count: int = 3  # Patterns before escalation
    pattern_accumulation_window_sec: float = 3600.0  # Time window for accumulation
    cooldown_sec: float = 300.0  # Minimum time between same-type alerts


@dataclass
class InitiativeEvent:
    """Event generated when initiative threshold is crossed."""

    event_id: str
    initiative_type: InitiativeType
    timestamp: float
    domain: str | None

    # Detection context
    anomaly_score: float
    confidence: float
    severity: float

    # Communication
    summary: str
    details: dict[str, Any]
    recommendations: list[str]

    # Metadata
    triggered_by: str  # What triggered this initiative
    vigilance_level: VigilanceLevel
    priority: int = 3  # 1=highest, 5=lowest

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "type": self.initiative_type.value,
            "timestamp": self.timestamp,
            "domain": self.domain,
            "scores": {
                "anomaly": self.anomaly_score,
                "confidence": self.confidence,
                "severity": self.severity,
            },
            "summary": self.summary,
            "details": self.details,
            "recommendations": self.recommendations,
            "triggered_by": self.triggered_by,
            "vigilance_level": self.vigilance_level.value,
            "priority": self.priority,
        }


@dataclass
class PatternAccumulator:
    """Tracks pattern accumulation for escalation."""

    domain: str
    patterns: deque = field(default_factory=lambda: deque(maxlen=100))
    last_escalation: float = 0.0

    def add_pattern(
        self,
        pattern_type: str,
        score: float,
        timestamp: float,
    ) -> None:
        """Add pattern observation."""
        self.patterns.append({"type": pattern_type, "score": score, "timestamp": timestamp})

    def count_recent(self, window_sec: float, current_time: float) -> int:
        """Count patterns in recent time window."""
        cutoff = current_time - window_sec
        return sum(1 for p in self.patterns if p["timestamp"] >= cutoff)

    def get_avg_severity(self, window_sec: float, current_time: float) -> float:
        """Get average severity in recent window."""
        cutoff = current_time - window_sec
        recent_scores = [p["score"] for p in self.patterns if p["timestamp"] >= cutoff]
        return float(np.mean(recent_scores)) if recent_scores else 0.0


class ProactiveMonitor:
    """
    Background Vigilance with Initiative Thresholds.

    Monitors detection streams and initiates communication when patterns
    warrant it. This enables Mercury to "speak up" when silence would
    be a disservice to truth.

    Key Behaviors:
        - Background thread monitors detection queue
        - Initiative only when thresholds crossed
        - Pattern accumulation triggers escalation
        - Domain-specific vigilance levels
        - Cooldown prevents noise

    Usage:
        monitor = ProactiveMonitor()

        # Register callback for initiative events
        monitor.on_initiative(lambda event: notify_user(event))

        # Start monitoring
        monitor.start()

        # Submit detection results for monitoring
        monitor.submit(detection_result)

        # Adjust vigilance based on context
        monitor.set_vigilance(VigilanceLevel.HEIGHTENED, domain="security")
    """

    def __init__(
        self,
        default_vigilance: VigilanceLevel = VigilanceLevel.ATTENTIVE,
        enable_scheduled_reports: bool = True,
        report_interval_sec: float = 3600.0,
    ) -> None:
        """
        Initialize Proactive Monitor.

        Args:
            default_vigilance: Default vigilance level
            enable_scheduled_reports: Whether to generate periodic reports
            report_interval_sec: Interval between scheduled reports
        """
        self.default_vigilance = default_vigilance
        self.enable_scheduled_reports = enable_scheduled_reports
        self.report_interval_sec = report_interval_sec

        # Per-domain configuration
        self._vigilance_levels: dict[str, VigilanceLevel] = {}
        self._thresholds: dict[str, InitiativeThreshold] = {}
        self._accumulators: dict[str, PatternAccumulator] = {}

        # Event tracking
        self._event_counter = 0
        self._event_queue: queue.Queue = queue.Queue()
        self._initiative_callbacks: list[Callable[[InitiativeEvent], None]] = []
        self._last_initiative: dict[str, float] = {}  # For cooldown tracking

        # Background monitoring
        self._monitoring_thread: threading.Thread | None = None
        self._report_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()
        self._running = False

        # Statistics
        self._detections_processed = 0
        self._initiatives_generated = 0
        self._escalations_triggered = 0

        # Initialize default thresholds
        self._initialize_domain_thresholds()

        self.logger = logging.getLogger(__name__)

    def _initialize_domain_thresholds(self) -> None:
        """Initialize domain-specific default thresholds."""
        # Medical domain - lower thresholds, higher sensitivity
        self._thresholds["medical"] = InitiativeThreshold(
            min_anomaly_score=0.6,
            min_confidence=0.5,
            min_severity=0.4,
            pattern_accumulation_count=2,
            cooldown_sec=180.0,
        )

        # Security domain - balanced thresholds
        self._thresholds["security"] = InitiativeThreshold(
            min_anomaly_score=0.7,
            min_confidence=0.6,
            min_severity=0.5,
            pattern_accumulation_count=3,
            cooldown_sec=300.0,
        )

        # Infrastructure - moderate thresholds
        self._thresholds["infrastructure"] = InitiativeThreshold(
            min_anomaly_score=0.7,
            min_confidence=0.6,
            min_severity=0.5,
            pattern_accumulation_count=4,
            cooldown_sec=600.0,
        )

        # Default for unknown domains
        self._thresholds["_default"] = InitiativeThreshold()

    def on_initiative(self, callback: Callable[[InitiativeEvent], None]) -> None:
        """Register callback for initiative events."""
        self._initiative_callbacks.append(callback)

    def set_vigilance(
        self,
        level: VigilanceLevel,
        domain: str | None = None,
    ) -> None:
        """Set vigilance level for a domain or globally."""
        if domain:
            self._vigilance_levels[domain] = level
            self.logger.info(f"Vigilance for {domain} set to {level.value}")
        else:
            self.default_vigilance = level
            self.logger.info(f"Default vigilance set to {level.value}")

    def set_threshold(
        self,
        threshold: InitiativeThreshold,
        domain: str | None = None,
    ) -> None:
        """Set initiative threshold for a domain."""
        domain_key = domain or "_default"
        self._thresholds[domain_key] = threshold

    def start(self) -> None:
        """Start background monitoring."""
        if self._running:
            return

        self._running = True
        self._shutdown_event.clear()

        # Start monitoring thread
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="ProactiveMonitor",
            daemon=True,
        )
        self._monitoring_thread.start()

        # Start scheduled report thread if enabled
        if self.enable_scheduled_reports:
            self._report_thread = threading.Thread(
                target=self._report_loop,
                name="ProactiveReporter",
                daemon=True,
            )
            self._report_thread.start()

        self.logger.info("Proactive monitoring started")

    def stop(self) -> None:
        """Stop background monitoring."""
        if not self._running:
            return

        self._running = False
        self._shutdown_event.set()

        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=2.0)
        if self._report_thread:
            self._report_thread.join(timeout=2.0)

        self.logger.info("Proactive monitoring stopped")

    def submit(
        self,
        detection_result: dict[str, Any],
        domain: str | None = None,
    ) -> None:
        """Submit detection result for proactive monitoring."""
        self._event_queue.put((detection_result, domain, time.time()))

    def _monitoring_loop(self) -> None:
        """Main monitoring loop (runs in background thread)."""
        while not self._shutdown_event.is_set():
            try:
                # Non-blocking get with timeout
                result, domain, timestamp = self._event_queue.get(timeout=0.5)
                self._process_detection(result, domain, timestamp)
                self._detections_processed += 1
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")

    def _report_loop(self) -> None:
        """Scheduled report generation loop."""
        last_report = time.time()

        while not self._shutdown_event.is_set():
            time.sleep(10.0)  # Check every 10 seconds

            if time.time() - last_report >= self.report_interval_sec:
                self._generate_scheduled_report()
                last_report = time.time()

    def _process_detection(
        self,
        detection_result: dict[str, Any],
        domain: str | None,
        timestamp: float,
    ) -> None:
        """Process a detection result for initiative triggers."""
        vigilance = self._vigilance_levels.get(domain or "", self.default_vigilance)

        # Passive mode - no initiative
        if vigilance == VigilanceLevel.PASSIVE:
            return

        threshold = self._thresholds.get(domain or "", self._thresholds["_default"])

        # Extract values
        anomaly_detected = detection_result.get("anomaly_detected", False)
        anomaly_score = detection_result.get("anomaly_score", 0.0)
        confidence = detection_result.get("confidence", 0.5)
        severity = detection_result.get("severity", 0.0)

        # Apply vigilance modifiers
        vigilance_modifier = self._get_vigilance_modifier(vigilance)
        effective_threshold_score = threshold.min_anomaly_score * vigilance_modifier
        effective_threshold_conf = threshold.min_confidence * vigilance_modifier
        effective_threshold_sev = threshold.min_severity * vigilance_modifier

        # Check for direct initiative trigger
        if anomaly_detected:
            should_alert = (
                anomaly_score >= effective_threshold_score
                and confidence >= effective_threshold_conf
                and severity >= effective_threshold_sev
            )

            if should_alert and self._check_cooldown(
                InitiativeType.ANOMALY_ALERT, domain, threshold.cooldown_sec
            ):
                self._generate_initiative(
                    initiative_type=InitiativeType.ANOMALY_ALERT,
                    detection_result=detection_result,
                    domain=domain,
                    timestamp=timestamp,
                    vigilance=vigilance,
                    triggered_by="threshold_crossed",
                )

        # Track for pattern accumulation
        if anomaly_detected:
            self._track_pattern(domain, anomaly_score, timestamp, threshold)

    def _get_vigilance_modifier(self, vigilance: VigilanceLevel) -> float:
        """Get threshold modifier based on vigilance level."""
        return {
            VigilanceLevel.PASSIVE: 1.0,  # No effect (no initiative anyway)
            VigilanceLevel.ATTENTIVE: 1.0,  # Standard thresholds
            VigilanceLevel.VIGILANT: 0.9,  # 10% lower thresholds
            VigilanceLevel.HEIGHTENED: 0.75,  # 25% lower thresholds
            VigilanceLevel.CRITICAL: 0.5,  # 50% lower thresholds
        }[vigilance]

    def _check_cooldown(
        self,
        initiative_type: InitiativeType,
        domain: str | None,
        cooldown_sec: float,
    ) -> bool:
        """Check if cooldown has elapsed for this type of initiative."""
        key = f"{initiative_type.value}:{domain or 'global'}"
        last_time = self._last_initiative.get(key, 0.0)

        if time.time() - last_time < cooldown_sec:
            return False

        self._last_initiative[key] = time.time()
        return True

    def _track_pattern(
        self,
        domain: str | None,
        score: float,
        timestamp: float,
        threshold: InitiativeThreshold,
    ) -> None:
        """Track pattern for accumulation-based escalation."""
        domain_key = domain or "_global"

        if domain_key not in self._accumulators:
            self._accumulators[domain_key] = PatternAccumulator(domain=domain_key)

        accumulator = self._accumulators[domain_key]
        accumulator.add_pattern("anomaly", score, timestamp)

        # Check for escalation
        recent_count = accumulator.count_recent(
            threshold.pattern_accumulation_window_sec, timestamp
        )

        if recent_count >= threshold.pattern_accumulation_count:
            # Check escalation cooldown (longer than regular cooldown)
            escalation_cooldown = threshold.cooldown_sec * 3

            if timestamp - accumulator.last_escalation >= escalation_cooldown:
                accumulator.last_escalation = timestamp
                self._escalations_triggered += 1

                avg_severity = accumulator.get_avg_severity(
                    threshold.pattern_accumulation_window_sec, timestamp
                )

                self._generate_initiative(
                    initiative_type=InitiativeType.ESCALATION,
                    detection_result={
                        "anomaly_detected": True,
                        "anomaly_score": avg_severity,
                        "confidence": 0.8,  # Pattern accumulation increases confidence
                        "severity": avg_severity,
                    },
                    domain=domain,
                    timestamp=timestamp,
                    vigilance=self._vigilance_levels.get(domain or "", self.default_vigilance),
                    triggered_by=f"pattern_accumulation_{recent_count}_in_window",
                )

    def _generate_initiative(
        self,
        initiative_type: InitiativeType,
        detection_result: dict[str, Any],
        domain: str | None,
        timestamp: float,
        vigilance: VigilanceLevel,
        triggered_by: str,
    ) -> None:
        """Generate an initiative event."""
        self._event_counter += 1
        self._initiatives_generated += 1

        anomaly_score = detection_result.get("anomaly_score", 0.0)
        confidence = detection_result.get("confidence", 0.5)
        severity = detection_result.get("severity", 0.0)

        # Generate summary based on type
        summary = self._generate_initiative_summary(
            initiative_type, anomaly_score, severity, domain
        )

        # Generate recommendations
        recommendations = self._generate_initiative_recommendations(
            initiative_type, severity, confidence, domain
        )

        # Determine priority
        priority = self._calculate_priority(initiative_type, severity, confidence, vigilance)

        event = InitiativeEvent(
            event_id=f"init_{self._event_counter}_{int(timestamp)}",
            initiative_type=initiative_type,
            timestamp=timestamp,
            domain=domain,
            anomaly_score=anomaly_score,
            confidence=confidence,
            severity=severity,
            summary=summary,
            details=detection_result,
            recommendations=recommendations,
            triggered_by=triggered_by,
            vigilance_level=vigilance,
            priority=priority,
        )

        # Notify callbacks
        for callback in self._initiative_callbacks:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Initiative callback error: {e}")

    def _generate_initiative_summary(
        self,
        initiative_type: InitiativeType,
        anomaly_score: float,
        severity: float,
        domain: str | None,
    ) -> str:
        """Generate human-readable initiative summary."""
        domain_prefix = f"[{domain.upper()}] " if domain else ""
        severity_word = (
            "Critical"
            if severity > 0.8
            else "High" if severity > 0.6 else "Moderate" if severity > 0.4 else "Low"
        )

        if initiative_type == InitiativeType.ANOMALY_ALERT:
            return (
                f"{domain_prefix}{severity_word} severity anomaly detected. "
                f"Score: {anomaly_score:.2f}. Proactive alert triggered."
            )
        elif initiative_type == InitiativeType.ESCALATION:
            return (
                f"{domain_prefix}ESCALATION: Multiple anomalies accumulated. "
                f"Average severity: {severity:.0%}. Pattern warrants attention."
            )
        elif initiative_type == InitiativeType.PREDICTION:
            return (
                f"{domain_prefix}Predictive alert: Elevated risk detected. "
                f"Probability: {anomaly_score:.0%}."
            )
        elif initiative_type == InitiativeType.SCHEDULED_REPORT:
            return f"{domain_prefix}Scheduled vigilance report generated."
        else:
            return f"{domain_prefix}Proactive initiative: {initiative_type.value}"

    def _generate_initiative_recommendations(
        self,
        initiative_type: InitiativeType,
        severity: float,
        confidence: float,
        domain: str | None,
    ) -> list[str]:
        """Generate recommendations for initiative."""
        recommendations = []

        if initiative_type == InitiativeType.ANOMALY_ALERT:
            if severity > 0.8:
                recommendations.append("Immediate review recommended.")
            else:
                recommendations.append("Review at next convenient opportunity.")

            if confidence < 0.7:
                recommendations.append("Consider gathering additional data for validation.")

        elif initiative_type == InitiativeType.ESCALATION:
            recommendations.append("Pattern accumulation detected - investigate root cause.")
            recommendations.append("Consider increasing monitoring vigilance.")

        elif initiative_type == InitiativeType.PREDICTION:
            recommendations.append("Predicted anomaly - consider preemptive action.")
            recommendations.append("Monitor closely over prediction horizon.")

        # Domain-specific
        if domain == "medical":
            recommendations.append("Ensure clinical review before any action.")
        elif domain == "security":
            recommendations.append("Preserve relevant logs and evidence.")

        return recommendations

    def _calculate_priority(
        self,
        initiative_type: InitiativeType,
        severity: float,
        confidence: float,
        vigilance: VigilanceLevel,
    ) -> int:
        """Calculate priority (1=highest, 5=lowest)."""
        base_priority = 3

        # Type adjustments
        if initiative_type == InitiativeType.ESCALATION:
            base_priority -= 1
        elif initiative_type == InitiativeType.SCHEDULED_REPORT:
            base_priority += 1

        # Severity adjustments
        if severity > 0.8:
            base_priority -= 1
        elif severity < 0.3:
            base_priority += 1

        # Vigilance adjustments
        if vigilance in (VigilanceLevel.HEIGHTENED, VigilanceLevel.CRITICAL):
            base_priority -= 1

        return max(1, min(5, base_priority))

    def _generate_scheduled_report(self) -> None:
        """Generate scheduled vigilance report."""
        now = time.time()

        # Aggregate statistics
        total_patterns = sum(
            acc.count_recent(self.report_interval_sec, now) for acc in self._accumulators.values()
        )

        _summary_parts = [  # Reserved for future report formatting
            f"Vigilance Report (interval: {self.report_interval_sec / 3600:.1f}h)",
            f"Detections processed: {self._detections_processed}",
            f"Initiatives generated: {self._initiatives_generated}",
            f"Escalations triggered: {self._escalations_triggered}",
            f"Patterns tracked: {total_patterns}",
        ]

        self._generate_initiative(
            initiative_type=InitiativeType.SCHEDULED_REPORT,
            detection_result={
                "anomaly_detected": False,
                "anomaly_score": 0.0,
                "confidence": 1.0,
                "severity": 0.0,
            },
            domain=None,
            timestamp=now,
            vigilance=self.default_vigilance,
            triggered_by="scheduled_interval",
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get monitor statistics."""
        return {
            "running": self._running,
            "default_vigilance": self.default_vigilance.value,
            "domain_vigilance_levels": {k: v.value for k, v in self._vigilance_levels.items()},
            "detections_processed": self._detections_processed,
            "initiatives_generated": self._initiatives_generated,
            "escalations_triggered": self._escalations_triggered,
            "active_accumulators": len(self._accumulators),
            "pending_queue_size": self._event_queue.qsize(),
        }
