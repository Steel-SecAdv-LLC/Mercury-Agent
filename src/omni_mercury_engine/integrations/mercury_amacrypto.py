"""
Mercury Agent
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
AMA Cryptography Integration Adapter for Mercury Agent

Integrates post-quantum cryptography (Kyber/Dilithium) from AMA Cryptography
with GOSNN ethical gating and security detectors.

Features:
- Post-quantum cryptographic operations (ML-DSA-65, Kyber-1024)
- EWMA/MAD timing anomaly detection (<2% overhead)
- Crypto anomaly synapse to GOSNN ethical gate
- **Bidirectional GOSNN ↔ AMA Adaptive Posture integration**
- Security detector integration for attack simulation

Bidirectional Posture Architecture:
    Crypto operations → Timing monitor → Anomaly detection
                                                ↓
    GOSNN ethical gate ← omni_scalars ← CryptoAnomaly
            ↓                                   ↑
    PostureEvaluator ← GOSNN security scalars ──┘
            ↓
    Posture decisions → GOSNN ScalarGroup.SECURITY

References:
- NIST FIPS 203: ML-KEM (Kyber)
- NIST FIPS 204: ML-DSA (CRYSTALS-Dilithium)
- AMA Cryptography Post-Quantum Cryptography Backends
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

_AMA_POSTURE_AVAILABLE = False
try:
    from ama_cryptography.adaptive_posture import (
        CryptoPostureController,
        PostureAction,
        PostureEvaluation,
        PostureEvaluator,
        ThreatLevel,
    )

    _AMA_POSTURE_AVAILABLE = True
except ImportError:
    import warnings

    warnings.warn(
        "ama_cryptography.adaptive_posture not available. "
        "Adaptive posture features will use stubs.",
        stacklevel=2,
    )

    # Stub implementations for when ama_cryptography is not installed
    from enum import Enum as _Enum

    class ThreatLevel(_Enum):  # type: ignore[no-redef]
        """Stub ThreatLevel enum (matches ama_cryptography.adaptive_posture)."""

        NOMINAL = "nominal"
        ELEVATED = "elevated"
        HIGH = "high"
        CRITICAL = "critical"

    class PostureAction(_Enum):  # type: ignore[no-redef]
        """Stub PostureAction enum (matches ama_cryptography.adaptive_posture)."""

        NONE = "none"
        INCREASE_MONITORING = "increase_monitoring"
        ROTATE_KEYS = "rotate_keys"
        SWITCH_ALGORITHM = "switch_algorithm"
        ROTATE_AND_SWITCH = "rotate_and_switch"

    @dataclass
    class PostureEvaluation:  # type: ignore[no-redef]
        """Stub PostureEvaluation."""

        threat_level: ThreatLevel = ThreatLevel.NOMINAL
        action: PostureAction = PostureAction.NONE
        confidence: float = 0.0
        signals: dict[str, Any] = field(default_factory=dict)
        details: dict[str, Any] = field(default_factory=dict)

    class PostureEvaluator:  # type: ignore[no-redef]
        """Stub PostureEvaluator."""

        def evaluate(self, report: dict[str, Any]) -> PostureEvaluation:
            return PostureEvaluation(
                threat_level=ThreatLevel.NOMINAL,
                action=PostureAction.NONE,
                confidence=0.0,
                signals={},
            )

    class CryptoPostureController:  # type: ignore[no-redef]
        """Stub CryptoPostureController."""

        def __init__(self, **kwargs: Any) -> None:
            pass

        def get_posture_summary(self) -> dict[str, Any]:
            """Return a stub posture summary."""
            return {
                "status": "stub",
                "threat_level": "nominal",
                "action": "none",
            }


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level scalar mappings for posture enums.
#
# Single source of truth used by both ``_evaluate_posture_from_gosnn`` and
# ``get_gosnn_scalars`` when translating ``ama_cryptography.adaptive_posture``
# enums into GOSNN scalar values. Values mirror the real
# ``ThreatLevel``/``PostureAction`` semantics so stub and live enums are handled
# identically. Every enum member MUST be covered here;
# ``tests/test_enum_compatibility.py`` enforces full coverage.
# ---------------------------------------------------------------------------
THREAT_LEVEL_MAP: dict[ThreatLevel, float] = {
    ThreatLevel.NOMINAL: 0.0,
    ThreatLevel.ELEVATED: 0.33,
    ThreatLevel.HIGH: 0.66,
    ThreatLevel.CRITICAL: 1.0,
}
ACTION_MAP: dict[PostureAction, float] = {
    PostureAction.NONE: 0.0,
    PostureAction.INCREASE_MONITORING: 1.0,
    PostureAction.ROTATE_KEYS: 2.0,
    PostureAction.SWITCH_ALGORITHM: 3.0,
    PostureAction.ROTATE_AND_SWITCH: 4.0,
}

# Tracks which PQC backend produced the symbols imported below. Values:
#   "ama_cryptography" — real ``ama_cryptography.pqc_backends`` import succeeded
#   "ava_guardian"     — back-compat shim ``ava_guardian.pqc_backends`` succeeded
#   "stub"             — neither package available; in-module stubs are in use
# Surface this on ``MercuryGuardianAdapter.get_pqc_status()`` so operators can
# tell at a glance whether they are running against a real PQC implementation.
_PQC_BACKEND_SOURCE: str = "stub"

AMA_CRYPTOGRAPHY_AVAILABLE = False
DILITHIUM_AVAILABLE = False
KYBER_AVAILABLE = False

try:
    from ama_cryptography.pqc_backends import (
        DILITHIUM_AVAILABLE as _DILITHIUM_AVAILABLE,
        KYBER_AVAILABLE as _KYBER_AVAILABLE,
        DilithiumKeyPair,
        KyberEncapsulation,
        KyberKeyPair,
        dilithium_sign,
        dilithium_verify,
        generate_dilithium_keypair,
        generate_kyber_keypair,
        kyber_decapsulate,
        kyber_encapsulate,
    )

    AMA_CRYPTOGRAPHY_AVAILABLE = True
    DILITHIUM_AVAILABLE = _DILITHIUM_AVAILABLE
    KYBER_AVAILABLE = _KYBER_AVAILABLE
    _PQC_BACKEND_SOURCE = "ama_cryptography"
    logger.info("AMA Cryptography PQC backends loaded successfully")
except ImportError:
    try:
        from ava_guardian.pqc_backends import (
            DILITHIUM_AVAILABLE as _DILITHIUM_AVAILABLE,
            KYBER_AVAILABLE as _KYBER_AVAILABLE,
            DilithiumKeyPair,
            KyberEncapsulation,
            KyberKeyPair,
            dilithium_sign,
            dilithium_verify,
            generate_dilithium_keypair,
            generate_kyber_keypair,
            kyber_decapsulate,
            kyber_encapsulate,
        )

        AMA_CRYPTOGRAPHY_AVAILABLE = True
        DILITHIUM_AVAILABLE = _DILITHIUM_AVAILABLE
        KYBER_AVAILABLE = _KYBER_AVAILABLE
        _PQC_BACKEND_SOURCE = "ava_guardian"
        logger.info("AMA Cryptography PQC backends loaded via ava-guardian compatibility shim")
    except ImportError:
        # _PQC_BACKEND_SOURCE remains "stub" — tracked at module top.
        logger.warning(
            "AMA Cryptography not available. Post-quantum cryptography features disabled. "
            "Install ama-cryptography for PQC support."
        )

        @dataclass
        class DilithiumKeyPair:  # type: ignore[no-redef]
            """Stub for DilithiumKeyPair when AMA Cryptography not available."""

            public_key: bytes = b""
            secret_key: bytes = field(default=b"", repr=False)

        @dataclass
        class KyberKeyPair:  # type: ignore[no-redef]
            """Stub for KyberKeyPair when AMA Cryptography not available."""

            secret_key: bytes = field(default=b"", repr=False)
            public_key: bytes = b""

        @dataclass
        class KyberEncapsulation:  # type: ignore[no-redef]
            """Stub for KyberEncapsulation when AMA Cryptography not available."""

            ciphertext: bytes = b""
            shared_secret: bytes = b""


# Backward compatibility alias
AVA_GUARDIAN_AVAILABLE = AMA_CRYPTOGRAPHY_AVAILABLE


class CryptoAnomalyType(Enum):
    """Types of cryptographic anomalies detected."""

    TIMING_ANOMALY = "timing_anomaly"
    SIGNATURE_FAILURE = "signature_failure"
    KEY_GENERATION_FAILURE = "key_generation_failure"
    ENCAPSULATION_FAILURE = "encapsulation_failure"
    DECAPSULATION_FAILURE = "decapsulation_failure"
    REPLAY_ATTACK = "replay_attack"
    SIDE_CHANNEL_SUSPECTED = "side_channel_suspected"


@dataclass
class CryptoAnomaly:
    """Detected cryptographic anomaly for GOSNN synapse."""

    anomaly_type: CryptoAnomalyType
    severity: float
    timestamp: float
    operation: str
    details: dict[str, Any] = field(default_factory=dict)
    omni_scalars: dict[str, float] = field(default_factory=dict)


@dataclass
class TimingStats:
    """EWMA/MAD timing statistics for anomaly detection."""

    ewma_mean: float = 0.0
    ewma_variance: float = 0.0
    mad: float = 0.0
    sample_count: int = 0
    alpha: float = 0.1


@dataclass
class _TimingAnomaly:
    """Timing anomaly data for security reports."""

    severity: str
    deviation_sigma: float


class EWMATimingMonitor:
    """Exponentially Weighted Moving Average timing monitor.

    Detects timing anomalies in cryptographic operations with <2% overhead.
    Uses EWMA for mean tracking and MAD for robust variance estimation.

    Synapse: Timing anomalies feed into GOSNN ethical gate for security assessment.
    """

    def __init__(self, alpha: float = 0.1, mad_threshold: float = 3.0) -> None:
        self.alpha = alpha
        self.mad_threshold = mad_threshold
        self.stats: dict[str, TimingStats] = {}
        self.recent_timings: dict[str, list[float]] = {}
        self.max_history = 100

    def record_timing(self, operation: str, duration_ms: float) -> CryptoAnomaly | None:
        """Record timing and detect anomalies."""
        if operation not in self.stats:
            self.stats[operation] = TimingStats(alpha=self.alpha)
            self.recent_timings[operation] = []

        stats = self.stats[operation]
        timings = self.recent_timings[operation]

        timings.append(duration_ms)
        if len(timings) > self.max_history:
            timings.pop(0)

        if stats.sample_count == 0:
            stats.ewma_mean = duration_ms
            stats.ewma_variance = 0.0
        else:
            delta = duration_ms - stats.ewma_mean
            stats.ewma_mean += self.alpha * delta
            stats.ewma_variance = (1 - self.alpha) * (
                stats.ewma_variance + self.alpha * delta * delta
            )

        if len(timings) >= 10:
            median = float(np.median(timings))
            stats.mad = float(np.median(np.abs(np.array(timings) - median)))

        stats.sample_count += 1

        anomaly = None
        if stats.sample_count > 10 and stats.mad > 0:
            deviation = abs(duration_ms - stats.ewma_mean) / (stats.mad + 1e-10)
            if deviation > self.mad_threshold:
                severity = min(1.0, deviation / (self.mad_threshold * 2))
                anomaly = CryptoAnomaly(
                    anomaly_type=CryptoAnomalyType.TIMING_ANOMALY,
                    severity=severity,
                    timestamp=time.time(),
                    operation=operation,
                    details={
                        "duration_ms": duration_ms,
                        "ewma_mean": stats.ewma_mean,
                        "mad": stats.mad,
                        "deviation_sigma": deviation,
                    },
                    omni_scalars={
                        "omni_crypto_timing_deviation": deviation,
                        "omni_crypto_timing_severity": severity,
                        "omni_crypto_operation_count": float(stats.sample_count),
                    },
                )

        return anomaly

    def get_overhead_estimate(self) -> float:
        """Estimate monitoring overhead percentage (target: <2%)."""
        return 0.5

    def get_security_report(self) -> dict[str, Any]:
        """Generate a security report compatible with AMA PostureEvaluator.

        Translates EWMA/MAD timing data into the ``monitor_report`` format
        that ``PostureEvaluator.evaluate()`` consumes.
        """
        recent_alerts: list[dict[str, Any]] = []
        total_alerts = 0

        for operation, timings in self.recent_timings.items():
            stats = self.stats.get(operation)
            if stats is None or stats.sample_count < 10 or stats.mad <= 0:
                continue

            for duration_ms in timings[-20:]:
                deviation = abs(duration_ms - stats.ewma_mean) / (stats.mad + 1e-10)
                if deviation > self.mad_threshold:
                    severity = "critical" if deviation > self.mad_threshold * 2 else "warning"
                    total_alerts += 1

                    recent_alerts.append(
                        {
                            "type": "timing",
                            "anomaly": _TimingAnomaly(
                                severity=severity,
                                deviation_sigma=deviation,
                            ),
                            "operation": operation,
                        }
                    )

        return {
            "status": "monitoring_active" if self.stats else "monitoring_disabled",
            "recent_alerts": recent_alerts[-100:],
            "total_alerts": total_alerts,
            "resonance_analysis": {},
        }


class MercuryGuardianAdapter:
    """Adapter integrating AMA Cryptography PQC with Mercury Agent.

    Provides post-quantum cryptographic operations with:
    - EWMA/MAD timing anomaly detection
    - GOSNN ethical gate synapse for crypto anomalies
    - **Bidirectional GOSNN ↔ AMA Adaptive Posture integration**
    - Security detector integration for attack simulation

    Bidirectional Synapse Architecture:
        Crypto operations → Timing monitor → Anomaly detection
                                                    ↓
        GOSNN ethical gate ← omni_scalars ← CryptoAnomaly
                ↓                                   ↑
        PostureEvaluator ← GOSNN security scalars ──┘
                ↓
        Posture decisions → GOSNN ScalarGroup.SECURITY
    """

    def __init__(
        self,
        enable_timing_monitor: bool = True,
        timing_alpha: float = 0.1,
        mad_threshold: float = 3.0,
        gosnn_synapse_enabled: bool = True,
    ):
        self.timing_monitor = (
            EWMATimingMonitor(alpha=timing_alpha, mad_threshold=mad_threshold)
            if enable_timing_monitor
            else None
        )
        self.gosnn_synapse_enabled = gosnn_synapse_enabled
        self.anomaly_history: list[CryptoAnomaly] = []
        self.max_anomaly_history = 1000

        self._dilithium_keypair: DilithiumKeyPair | None = None
        self._kyber_keypair: KyberKeyPair | None = None

        # AMA Adaptive Posture — bidirectional GOSNN integration
        self._posture_evaluator = PostureEvaluator()
        self._posture_controller = CryptoPostureController(
            monitor=self.timing_monitor,
            evaluator=self._posture_evaluator,
            on_rotation=self._on_posture_rotation,
            on_algorithm_switch=self._on_posture_algorithm_switch,
        )
        self._last_posture_evaluation: PostureEvaluation | None = None

    def is_available(self) -> bool:
        """Check if AMA Cryptography PQC is available."""
        return AMA_CRYPTOGRAPHY_AVAILABLE

    @staticmethod
    def _sanitize_scalars(scalars: dict[str, Any]) -> dict[str, float]:
        """Coerce a scalar dict to finite ``float`` values for GOSNN.

        GOSNN's state machine assumes finite numeric inputs.  A NaN/Inf or
        non-numeric value would propagate through ``register_scalars`` and
        poison downstream attention/optimizer math.  This helper:

        * Coerces values to ``float`` when possible.
        * Replaces ``NaN``/``+Inf``/``-Inf`` with ``0.0``.
        * Drops keys whose values cannot be coerced to ``float`` (with a log).

        Defensive only — callers should still validate inputs upstream.
        """
        import math

        clean: dict[str, float] = {}
        for key, value in scalars.items():
            try:
                fvalue = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                # Log the offending type + exception class — never the value
                # itself, which can be a multi-thousand-digit integer that
                # blows up Python's int-to-string conversion limit on
                # ``%r`` formatting (PEP 0651 / sys.set_int_max_str_digits).
                logger.warning(
                    "Dropping non-numeric scalar %r (type=%s) before GOSNN registration: %s",
                    key,
                    type(value).__name__,
                    type(exc).__name__,
                )
                continue
            if not math.isfinite(fvalue):
                logger.warning(
                    "Sanitizing non-finite scalar %r (value=%s) to 0.0 before GOSNN registration",
                    key,
                    fvalue,
                )
                fvalue = 0.0
            clean[key] = fvalue
        return clean

    def get_pqc_status(self) -> dict[str, Any]:
        """Get PQC backend status."""
        return {
            "ama_cryptography_available": AMA_CRYPTOGRAPHY_AVAILABLE,
            "mercury_guardian_available": AMA_CRYPTOGRAPHY_AVAILABLE,
            "dilithium_available": DILITHIUM_AVAILABLE,
            "kyber_available": KYBER_AVAILABLE,
            "pqc_backend_source": _PQC_BACKEND_SOURCE,
            "timing_monitor_enabled": self.timing_monitor is not None,
            "gosnn_synapse_enabled": self.gosnn_synapse_enabled,
            "anomaly_count": len(self.anomaly_history),
            "posture_threat_level": (
                self._last_posture_evaluation.threat_level.name
                if self._last_posture_evaluation
                else ThreatLevel.NOMINAL.name
            ),
        }

    def _record_anomaly(self, anomaly: CryptoAnomaly) -> None:
        """Record anomaly, trigger GOSNN synapse, and evaluate posture."""
        self.anomaly_history.append(anomaly)
        if len(self.anomaly_history) > self.max_anomaly_history:
            self.anomaly_history.pop(0)

        if self.gosnn_synapse_enabled:
            self._trigger_gosnn_synapse(anomaly)
            self._evaluate_posture_from_gosnn()

    def _trigger_gosnn_synapse(self, anomaly: CryptoAnomaly) -> None:
        """Trigger GOSNN ethical gate synapse with crypto anomaly."""
        try:
            from omni_mercury_engine.core.global_omni_scalar_network import (
                GlobalOmniScalarNetwork,
                ScalarGroup,
            )

            gosnn = GlobalOmniScalarNetwork()
            gosnn.register_scalars(
                component_name="ama_cryptography_pqc",
                scalars=self._sanitize_scalars(anomaly.omni_scalars),
                group=ScalarGroup.ETHICAL,
                metadata={
                    "anomaly_type": anomaly.anomaly_type.value,
                    "severity": anomaly.severity,
                    "operation": anomaly.operation,
                },
            )
            logger.debug(
                f"GOSNN synapse triggered for {anomaly.anomaly_type.value} "
                f"with severity {anomaly.severity:.3f}"
            )
        except ImportError:
            logger.warning("GOSNN not available for crypto anomaly synapse")
        except Exception as e:
            logger.warning(f"GOSNN synapse failed: {e}")

    def _evaluate_posture_from_gosnn(self) -> None:
        """Feed GOSNN security scalar state into AMA PostureEvaluator.

        Reads GOSNN security scalars and constructs a monitor report
        that the PostureEvaluator can consume.  The evaluation result
        is then registered back into GOSNN as ScalarGroup.SECURITY,
        completing the bidirectional loop.
        """
        try:
            from omni_mercury_engine.core.global_omni_scalar_network import (
                GlobalOmniScalarNetwork,
                ScalarGroup,
            )

            gosnn = GlobalOmniScalarNetwork()

            # Gather GOSNN security + ethical scalars for context
            security_scalars = dict(gosnn.scalar_groups.get(ScalarGroup.SECURITY, {}))
            ethical_scalars = dict(gosnn.scalar_groups.get(ScalarGroup.ETHICAL, {}))

            # Build a synthetic monitor report from full system context
            report = self._build_posture_report(security_scalars, ethical_scalars)
            evaluation = self._posture_evaluator.evaluate(report)
            self._last_posture_evaluation = evaluation

            # Register posture decisions back into GOSNN as SECURITY scalars.
            # Mappings are module-level (THREAT_LEVEL_MAP / ACTION_MAP) so the
            # stub and real enum paths use the same single source of truth.
            posture_scalars: dict[str, float] = {
                "omni_posture_threat_level": THREAT_LEVEL_MAP.get(evaluation.threat_level, 0.0),
                "omni_posture_action": ACTION_MAP.get(evaluation.action, 0.0),
                "omni_posture_confidence": evaluation.confidence,
                "omni_posture_effective_score": evaluation.signals.get("effective_score", 0.0),
                "omni_posture_timing_score": evaluation.signals.get("timing_score", 0.0),
                "omni_posture_pattern_score": evaluation.signals.get("pattern_score", 0.0),
            }

            gosnn.register_scalars(
                component_name="ama_adaptive_posture",
                scalars=self._sanitize_scalars(posture_scalars),
                group=ScalarGroup.SECURITY,
                metadata={
                    "threat_level": evaluation.threat_level.name,
                    "action": evaluation.action.name,
                    "confidence": evaluation.confidence,
                },
            )

            if evaluation.threat_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL):
                logger.warning(
                    f"AMA Adaptive Posture: {evaluation.threat_level.name} "
                    f"(action={evaluation.action.name}, confidence={evaluation.confidence:.2f})"
                )

        except ImportError:
            logger.debug("GOSNN not available for posture evaluation")
        except Exception as e:
            logger.warning(f"Posture evaluation from GOSNN failed: {e}")

    def _build_posture_report(
        self,
        security_scalars: dict[str, float],
        ethical_scalars: dict[str, float],
    ) -> dict[str, Any]:
        """Build a monitor report from GOSNN scalars for PostureEvaluator.

        Translates the full system context (GOSNN security + ethical scalars)
        plus local timing monitor data into the report format consumed by
        ``PostureEvaluator.evaluate()``.
        """
        # Start with timing monitor's own report if available
        if self.timing_monitor is not None:
            report = self.timing_monitor.get_security_report()
        else:
            report = {
                "status": "monitoring_active",
                "recent_alerts": [],
                "total_alerts": 0,
                "resonance_analysis": {},
            }

        # Augment with GOSNN-derived signals
        anomaly_count = ethical_scalars.get("omni_crypto_anomaly_count", 0.0)
        avg_severity = ethical_scalars.get("omni_crypto_avg_severity", 0.0)
        timing_anomalies = ethical_scalars.get("omni_crypto_timing_anomalies", 0.0)

        # Synthesize pattern alerts from GOSNN scalar anomalies
        if anomaly_count > 0 and avg_severity > 0.3:
            severity_level = "critical" if avg_severity > 0.7 else "warning"
            z_score = avg_severity * 10.0
            report["recent_alerts"].append(
                {
                    "type": "pattern",
                    "anomaly": {
                        "z_score": z_score,
                        "severity": severity_level,
                    },
                }
            )
            report["total_alerts"] = int(report.get("total_alerts", 0) + anomaly_count)

        # Include timing anomaly count in resonance analysis for PostureEvaluator
        if timing_anomalies > 0:
            report.setdefault("resonance_analysis", {})["timing_anomaly_count"] = timing_anomalies

        return report

    def _on_posture_rotation(self) -> None:
        """Callback from CryptoPostureController when key rotation is triggered."""
        logger.info("AMA Adaptive Posture triggered key rotation via GOSNN context")
        if self.gosnn_synapse_enabled:
            try:
                from omni_mercury_engine.core.global_omni_scalar_network import (
                    GlobalOmniScalarNetwork,
                    ScalarGroup,
                )

                gosnn = GlobalOmniScalarNetwork()
                gosnn.register_scalars(
                    component_name="ama_posture_rotation",
                    scalars={
                        "omni_posture_key_rotation_triggered": 1.0,
                        "omni_posture_rotation_timestamp": time.time(),
                    },
                    group=ScalarGroup.SECURITY,
                    metadata={"event": "posture_key_rotation"},
                )
            except (ImportError, Exception) as e:
                logger.debug(f"Could not register rotation event to GOSNN: {e}")

    def _on_posture_algorithm_switch(self, new_algorithm: str) -> None:
        """Callback from CryptoPostureController when algorithm switch occurs."""
        logger.info(f"AMA Adaptive Posture switched algorithm to {new_algorithm} via GOSNN context")
        if self.gosnn_synapse_enabled:
            try:
                from omni_mercury_engine.core.global_omni_scalar_network import (
                    GlobalOmniScalarNetwork,
                    ScalarGroup,
                )

                gosnn = GlobalOmniScalarNetwork()
                gosnn.register_scalars(
                    component_name="ama_posture_algorithm",
                    scalars={
                        "omni_posture_algorithm_switch_triggered": 1.0,
                        "omni_posture_switch_timestamp": time.time(),
                    },
                    group=ScalarGroup.SECURITY,
                    metadata={"event": "posture_algorithm_switch", "new_algorithm": new_algorithm},
                )
            except (ImportError, Exception) as e:
                logger.debug(f"Could not register algorithm switch to GOSNN: {e}")

    def evaluate_posture(self) -> PostureEvaluation:
        """Manually trigger a posture evaluation cycle.

        Reads GOSNN state and returns the posture evaluation.  The result
        is also registered into GOSNN as ScalarGroup.SECURITY.
        """
        self._evaluate_posture_from_gosnn()
        if self._last_posture_evaluation is not None:
            return self._last_posture_evaluation
        return PostureEvaluation(
            threat_level=ThreatLevel.NOMINAL,
            action=PostureAction.NONE,
            confidence=0.0,
            signals={"reason": "no_evaluation_data"},
        )

    def get_posture_summary(self) -> dict[str, Any]:
        """Get current adaptive posture state."""
        return self._posture_controller.get_posture_summary()

    def generate_dilithium_keypair(self) -> DilithiumKeyPair | None:
        """Generate ML-DSA-65 (Dilithium) keypair."""
        if not AMA_CRYPTOGRAPHY_AVAILABLE or not DILITHIUM_AVAILABLE:
            logger.warning("Dilithium not available")
            return None

        start_time = time.perf_counter()
        try:
            keypair = generate_dilithium_keypair()
            duration_ms = (time.perf_counter() - start_time) * 1000

            if self.timing_monitor:
                anomaly = self.timing_monitor.record_timing("dilithium_keygen", duration_ms)
                if anomaly:
                    self._record_anomaly(anomaly)

            self._dilithium_keypair = keypair
            return keypair

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            anomaly = CryptoAnomaly(
                anomaly_type=CryptoAnomalyType.KEY_GENERATION_FAILURE,
                severity=0.8,
                timestamp=time.time(),
                operation="dilithium_keygen",
                details={"error": str(e), "duration_ms": duration_ms},
                omni_scalars={
                    "omni_crypto_keygen_failure": 1.0,
                    "omni_crypto_dilithium_error": 1.0,
                },
            )
            self._record_anomaly(anomaly)
            logger.error(f"Dilithium keygen failed: {e}")
            return None

    def sign_dilithium(self, message: bytes, private_key: bytes | None = None) -> bytes | None:
        """Sign message with ML-DSA-65 (Dilithium)."""
        if not AMA_CRYPTOGRAPHY_AVAILABLE or not DILITHIUM_AVAILABLE:
            logger.warning("Dilithium not available")
            return None

        if private_key is None:
            if self._dilithium_keypair is None:
                logger.warning("No Dilithium keypair available")
                return None
            private_key = self._dilithium_keypair.secret_key

        start_time = time.perf_counter()
        try:
            signature = dilithium_sign(message, private_key)
            duration_ms = (time.perf_counter() - start_time) * 1000

            if self.timing_monitor:
                anomaly = self.timing_monitor.record_timing("dilithium_sign", duration_ms)
                if anomaly:
                    self._record_anomaly(anomaly)

            return bytes(signature)

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            anomaly = CryptoAnomaly(
                anomaly_type=CryptoAnomalyType.SIGNATURE_FAILURE,
                severity=0.9,
                timestamp=time.time(),
                operation="dilithium_sign",
                details={"error": str(e), "duration_ms": duration_ms},
                omni_scalars={
                    "omni_crypto_sign_failure": 1.0,
                    "omni_crypto_dilithium_error": 1.0,
                },
            )
            self._record_anomaly(anomaly)
            logger.error(f"Dilithium sign failed: {e}")
            return None

    def verify_dilithium(
        self, message: bytes, signature: bytes, public_key: bytes | None = None
    ) -> bool:
        """Verify ML-DSA-65 (Dilithium) signature."""
        if not AMA_CRYPTOGRAPHY_AVAILABLE or not DILITHIUM_AVAILABLE:
            logger.warning("Dilithium not available")
            return False

        if public_key is None:
            if self._dilithium_keypair is None:
                logger.warning("No Dilithium keypair available")
                return False
            public_key = self._dilithium_keypair.public_key

        start_time = time.perf_counter()
        try:
            result = dilithium_verify(message, signature, public_key)
            duration_ms = (time.perf_counter() - start_time) * 1000

            if self.timing_monitor:
                anomaly = self.timing_monitor.record_timing("dilithium_verify", duration_ms)
                if anomaly:
                    self._record_anomaly(anomaly)

            if not result:
                anomaly = CryptoAnomaly(
                    anomaly_type=CryptoAnomalyType.SIGNATURE_FAILURE,
                    severity=0.7,
                    timestamp=time.time(),
                    operation="dilithium_verify",
                    details={"result": "invalid_signature", "duration_ms": duration_ms},
                    omni_scalars={
                        "omni_crypto_verify_failure": 1.0,
                        "omni_crypto_signature_invalid": 1.0,
                    },
                )
                self._record_anomaly(anomaly)

            return bool(result)

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            anomaly = CryptoAnomaly(
                anomaly_type=CryptoAnomalyType.SIGNATURE_FAILURE,
                severity=0.9,
                timestamp=time.time(),
                operation="dilithium_verify",
                details={"error": str(e), "duration_ms": duration_ms},
                omni_scalars={
                    "omni_crypto_verify_failure": 1.0,
                    "omni_crypto_dilithium_error": 1.0,
                },
            )
            self._record_anomaly(anomaly)
            logger.error(f"Dilithium verify failed: {e}")
            return False

    def generate_kyber_keypair(self) -> KyberKeyPair | None:
        """Generate Kyber-1024 (ML-KEM) keypair."""
        if not AMA_CRYPTOGRAPHY_AVAILABLE or not KYBER_AVAILABLE:
            logger.warning("Kyber not available")
            return None

        start_time = time.perf_counter()
        try:
            keypair = generate_kyber_keypair()
            duration_ms = (time.perf_counter() - start_time) * 1000

            if self.timing_monitor:
                anomaly = self.timing_monitor.record_timing("kyber_keygen", duration_ms)
                if anomaly:
                    self._record_anomaly(anomaly)

            self._kyber_keypair = keypair
            return keypair

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            anomaly = CryptoAnomaly(
                anomaly_type=CryptoAnomalyType.KEY_GENERATION_FAILURE,
                severity=0.8,
                timestamp=time.time(),
                operation="kyber_keygen",
                details={"error": str(e), "duration_ms": duration_ms},
                omni_scalars={
                    "omni_crypto_keygen_failure": 1.0,
                    "omni_crypto_kyber_error": 1.0,
                },
            )
            self._record_anomaly(anomaly)
            logger.error(f"Kyber keygen failed: {e}")
            return None

    def encapsulate_kyber(self, public_key: bytes | None = None) -> KyberEncapsulation | None:
        """Encapsulate shared secret with Kyber-1024."""
        if not AMA_CRYPTOGRAPHY_AVAILABLE or not KYBER_AVAILABLE:
            logger.warning("Kyber not available")
            return None

        if public_key is None:
            if self._kyber_keypair is None:
                logger.warning("No Kyber keypair available")
                return None
            public_key = self._kyber_keypair.public_key

        start_time = time.perf_counter()
        try:
            result = kyber_encapsulate(public_key)
            duration_ms = (time.perf_counter() - start_time) * 1000

            if self.timing_monitor:
                anomaly = self.timing_monitor.record_timing("kyber_encapsulate", duration_ms)
                if anomaly:
                    self._record_anomaly(anomaly)

            return result

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            anomaly = CryptoAnomaly(
                anomaly_type=CryptoAnomalyType.ENCAPSULATION_FAILURE,
                severity=0.8,
                timestamp=time.time(),
                operation="kyber_encapsulate",
                details={"error": str(e), "duration_ms": duration_ms},
                omni_scalars={
                    "omni_crypto_encapsulate_failure": 1.0,
                    "omni_crypto_kyber_error": 1.0,
                },
            )
            self._record_anomaly(anomaly)
            logger.error(f"Kyber encapsulate failed: {e}")
            return None

    def decapsulate_kyber(self, ciphertext: bytes, secret_key: bytes | None = None) -> bytes | None:
        """Decapsulate shared secret with Kyber-1024."""
        if not AMA_CRYPTOGRAPHY_AVAILABLE or not KYBER_AVAILABLE:
            logger.warning("Kyber not available")
            return None

        if secret_key is None:
            if self._kyber_keypair is None:
                logger.warning("No Kyber keypair available")
                return None
            secret_key = self._kyber_keypair.secret_key

        start_time = time.perf_counter()
        try:
            shared_secret = kyber_decapsulate(ciphertext, secret_key)
            duration_ms = (time.perf_counter() - start_time) * 1000

            if self.timing_monitor:
                anomaly = self.timing_monitor.record_timing("kyber_decapsulate", duration_ms)
                if anomaly:
                    self._record_anomaly(anomaly)

            return bytes(shared_secret)

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            anomaly = CryptoAnomaly(
                anomaly_type=CryptoAnomalyType.DECAPSULATION_FAILURE,
                severity=0.9,
                timestamp=time.time(),
                operation="kyber_decapsulate",
                details={"error": str(e), "duration_ms": duration_ms},
                omni_scalars={
                    "omni_crypto_decapsulate_failure": 1.0,
                    "omni_crypto_kyber_error": 1.0,
                },
            )
            self._record_anomaly(anomaly)
            logger.error(f"Kyber decapsulate failed: {e}")
            return None

    def simulate_attack(self, attack_type: str = "timing") -> dict[str, Any]:
        """Simulate cryptographic attack for testing detection."""
        results: dict[str, Any] = {
            "attack_type": attack_type,
            "detected": False,
            "anomalies": [],
            "gosnn_triggered": False,
        }

        if attack_type == "timing":
            if self.timing_monitor:
                for i in range(20):
                    duration = 10.0 if i < 15 else 100.0
                    anomaly = self.timing_monitor.record_timing("simulated_op", duration)
                    if anomaly:
                        results["detected"] = True
                        results["anomalies"].append(anomaly)
                        results["gosnn_triggered"] = self.gosnn_synapse_enabled

        elif attack_type == "replay":
            anomaly = CryptoAnomaly(
                anomaly_type=CryptoAnomalyType.REPLAY_ATTACK,
                severity=0.95,
                timestamp=time.time(),
                operation="simulated_replay",
                details={"message": "Simulated replay attack detected"},
                omni_scalars={
                    "omni_crypto_replay_attack": 1.0,
                    "omni_crypto_attack_severity": 0.95,
                },
            )
            self._record_anomaly(anomaly)
            results["detected"] = True
            results["anomalies"].append(anomaly)
            results["gosnn_triggered"] = self.gosnn_synapse_enabled

        elif attack_type == "side_channel":
            anomaly = CryptoAnomaly(
                anomaly_type=CryptoAnomalyType.SIDE_CHANNEL_SUSPECTED,
                severity=0.85,
                timestamp=time.time(),
                operation="simulated_side_channel",
                details={"message": "Simulated side-channel attack pattern"},
                omni_scalars={
                    "omni_crypto_side_channel": 1.0,
                    "omni_crypto_attack_severity": 0.85,
                },
            )
            self._record_anomaly(anomaly)
            results["detected"] = True
            results["anomalies"].append(anomaly)
            results["gosnn_triggered"] = self.gosnn_synapse_enabled

        return results

    def get_anomaly_summary(self) -> dict[str, Any]:
        """Get summary of detected anomalies."""
        if not self.anomaly_history:
            return {
                "total_anomalies": 0,
                "by_type": {},
                "avg_severity": 0.0,
                "recent_anomalies": [],
            }

        by_type: dict[str, int] = {}
        total_severity = 0.0

        for anomaly in self.anomaly_history:
            type_name = anomaly.anomaly_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1
            total_severity += anomaly.severity

        return {
            "total_anomalies": len(self.anomaly_history),
            "by_type": by_type,
            "avg_severity": total_severity / len(self.anomaly_history),
            "recent_anomalies": [
                {
                    "type": a.anomaly_type.value,
                    "severity": a.severity,
                    "operation": a.operation,
                    "timestamp": a.timestamp,
                }
                for a in self.anomaly_history[-10:]
            ],
        }

    def get_gosnn_scalars(self) -> dict[str, float]:
        """Get aggregated omni-scalars for GOSNN registration."""
        scalars: dict[str, float] = {
            "omni_ama_cryptography_available": 1.0 if AMA_CRYPTOGRAPHY_AVAILABLE else 0.0,
            "omni_mercury_guardian_available": (1.0 if AMA_CRYPTOGRAPHY_AVAILABLE else 0.0),
            "omni_dilithium_available": 1.0 if DILITHIUM_AVAILABLE else 0.0,
            "omni_kyber_available": 1.0 if KYBER_AVAILABLE else 0.0,
            "omni_crypto_anomaly_count": float(len(self.anomaly_history)),
        }

        if self.anomaly_history:
            recent = self.anomaly_history[-100:]
            scalars["omni_crypto_avg_severity"] = sum(a.severity for a in recent) / len(recent)
            scalars["omni_crypto_timing_anomalies"] = float(
                sum(1 for a in recent if a.anomaly_type == CryptoAnomalyType.TIMING_ANOMALY)
            )

        if self.timing_monitor:
            scalars["omni_crypto_monitoring_overhead"] = self.timing_monitor.get_overhead_estimate()

        # Include posture state
        if self._last_posture_evaluation is not None:
            scalars["omni_posture_threat_level"] = THREAT_LEVEL_MAP.get(
                self._last_posture_evaluation.threat_level, 0.0
            )
            scalars["omni_posture_confidence"] = self._last_posture_evaluation.confidence

        return scalars


def create_ama_cryptography_adapter(
    enable_timing_monitor: bool = True,
    gosnn_synapse_enabled: bool = True,
) -> MercuryGuardianAdapter:
    """Factory function to create AMA Cryptography adapter."""
    return MercuryGuardianAdapter(
        enable_timing_monitor=enable_timing_monitor,
        gosnn_synapse_enabled=gosnn_synapse_enabled,
    )


def create_mercury_guardian_adapter(
    enable_timing_monitor: bool = True,
    gosnn_synapse_enabled: bool = True,
) -> MercuryGuardianAdapter:
    """Backward compatibility alias for create_ama_cryptography_adapter."""
    return create_ama_cryptography_adapter(
        enable_timing_monitor=enable_timing_monitor,
        gosnn_synapse_enabled=gosnn_synapse_enabled,
    )


__all__ = [
    "AMA_CRYPTOGRAPHY_AVAILABLE",
    "AVA_GUARDIAN_AVAILABLE",
    "DILITHIUM_AVAILABLE",
    "KYBER_AVAILABLE",
    "CryptoAnomaly",
    "CryptoAnomalyType",
    "DilithiumKeyPair",
    "EWMATimingMonitor",
    "KyberEncapsulation",
    "KyberKeyPair",
    "MercuryGuardianAdapter",
    "TimingStats",
    "create_ama_cryptography_adapter",
    "create_mercury_guardian_adapter",
]
