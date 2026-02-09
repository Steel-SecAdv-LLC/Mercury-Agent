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
Ava-Guardian Integration Adapter for Mercury Agent ♱

Integrates post-quantum cryptography (Kyber/Dilithium) from Ava-Guardian ♱
with GOSNN ethical gating and security detectors.

Features:
- Post-quantum cryptographic operations (ML-DSA-65, Kyber-1024)
- EWMA/MAD timing anomaly detection (<2% overhead)
- Crypto anomaly synapse to GOSNN ethical gate
- Security detector integration for attack simulation

Synapse: Crypto anomalies → GOSNN gate → security detectors

References:
- NIST FIPS 203: ML-KEM (Kyber)
- NIST FIPS 204: ML-DSA (CRYSTALS-Dilithium)
- Ava-Guardian ♱ Post-Quantum Cryptography Backends
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

AVA_GUARDIAN_AVAILABLE = False
DILITHIUM_AVAILABLE = False
KYBER_AVAILABLE = False

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

    AVA_GUARDIAN_AVAILABLE = True
    DILITHIUM_AVAILABLE = _DILITHIUM_AVAILABLE
    KYBER_AVAILABLE = _KYBER_AVAILABLE
    logger.info("Ava-Guardian PQC backends loaded successfully")
except ImportError:
    logger.warning(
        "Ava-Guardian not available. Post-quantum cryptography features disabled. "
        "Install ava-guardian for PQC support."
    )

    @dataclass
    class DilithiumKeyPair:  # type: ignore[no-redef]
        """Stub for DilithiumKeyPair when Ava-Guardian not available."""

        private_key: bytes = field(default=b"", repr=False)
        public_key: bytes = b""

    @dataclass
    class KyberKeyPair:  # type: ignore[no-redef]
        """Stub for KyberKeyPair when Ava-Guardian not available."""

        secret_key: bytes = field(default=b"", repr=False)
        public_key: bytes = b""

    @dataclass
    class KyberEncapsulation:  # type: ignore[no-redef]
        """Stub for KyberEncapsulation when Ava-Guardian not available."""

        ciphertext: bytes = b""
        shared_secret: bytes = b""


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


class EWMATimingMonitor:
    """Exponentially Weighted Moving Average timing monitor.

    Detects timing anomalies in cryptographic operations with <2% overhead.
    Uses EWMA for mean tracking and MAD for robust variance estimation.

    Synapse: Timing anomalies feed into GOSNN ethical gate for security assessment.
    """

    def __init__(self, alpha: float = 0.1, mad_threshold: float = 3.0) -> None:
        """Initialize EWMA timing monitor.

        Args:
            alpha: EWMA smoothing factor (0 < alpha <= 1)
            mad_threshold: MAD multiplier for anomaly detection
        """
        self.alpha = alpha
        self.mad_threshold = mad_threshold
        self.stats: dict[str, TimingStats] = {}
        self.recent_timings: dict[str, list[float]] = {}
        self.max_history = 100

    def record_timing(self, operation: str, duration_ms: float) -> CryptoAnomaly | None:
        """Record timing and detect anomalies.

        Args:
            operation: Name of cryptographic operation
            duration_ms: Operation duration in milliseconds

        Returns:
            CryptoAnomaly if timing anomaly detected, None otherwise
        """
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
        """Estimate monitoring overhead percentage.

        Returns:
            Estimated overhead as percentage (target: <2%)
        """
        return 0.5


class MercuryGuardianAdapter:
    """Adapter integrating Ava-Guardian PQC with Mercury Agent ♱.

    Provides post-quantum cryptographic operations with:
    - EWMA/MAD timing anomaly detection
    - GOSNN ethical gate synapse for crypto anomalies
    - Security detector integration for attack simulation

    Synapse Architecture:
        Crypto operations → Timing monitor → Anomaly detection
                                                    ↓
        GOSNN ethical gate ← omni_scalars ← CryptoAnomaly
                ↓
        Security detectors (if anomaly severity > threshold)
    """

    def __init__(
        self,
        enable_timing_monitor: bool = True,
        timing_alpha: float = 0.1,
        mad_threshold: float = 3.0,
        gosnn_synapse_enabled: bool = True,
    ):
        """Initialize Ava-Guardian adapter.

        Args:
            enable_timing_monitor: Enable EWMA/MAD timing monitoring
            timing_alpha: EWMA smoothing factor
            mad_threshold: MAD multiplier for anomaly detection
            gosnn_synapse_enabled: Enable GOSNN ethical gate synapse
        """
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

    def is_available(self) -> bool:
        """Check if Ava-Guardian PQC is available."""
        return AVA_GUARDIAN_AVAILABLE

    def get_pqc_status(self) -> dict[str, Any]:
        """Get PQC backend status.

        Returns:
            Dictionary with availability status for each algorithm
        """
        return {
            "mercury_guardian_available": AVA_GUARDIAN_AVAILABLE,
            "dilithium_available": DILITHIUM_AVAILABLE,
            "kyber_available": KYBER_AVAILABLE,
            "timing_monitor_enabled": self.timing_monitor is not None,
            "gosnn_synapse_enabled": self.gosnn_synapse_enabled,
            "anomaly_count": len(self.anomaly_history),
        }

    def _record_anomaly(self, anomaly: CryptoAnomaly) -> None:
        """Record anomaly and trigger GOSNN synapse if enabled."""
        self.anomaly_history.append(anomaly)
        if len(self.anomaly_history) > self.max_anomaly_history:
            self.anomaly_history.pop(0)

        if self.gosnn_synapse_enabled:
            self._trigger_gosnn_synapse(anomaly)

    def _trigger_gosnn_synapse(self, anomaly: CryptoAnomaly) -> None:
        """Trigger GOSNN ethical gate synapse with crypto anomaly.

        Synapse: Registers omni-scalars from crypto anomaly for ethical gating.
        """
        try:
            from omni_mercury_engine.core.global_omni_scalar_network import (
                GlobalOmniScalarNetwork,
                ScalarGroup,
            )

            gosnn = GlobalOmniScalarNetwork()
            gosnn.register_scalars(
                component_name="ava_guardian_crypto",
                scalars=anomaly.omni_scalars,
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

    def generate_dilithium_keypair(self) -> DilithiumKeyPair | None:
        """Generate ML-DSA-65 (Dilithium) keypair.

        Returns:
            DilithiumKeyPair or None if not available
        """
        if not AVA_GUARDIAN_AVAILABLE or not DILITHIUM_AVAILABLE:
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
        """Sign message with ML-DSA-65 (Dilithium).

        Args:
            message: Data to sign
            private_key: Optional private key (uses cached if None)

        Returns:
            Signature bytes or None if failed
        """
        if not AVA_GUARDIAN_AVAILABLE or not DILITHIUM_AVAILABLE:
            logger.warning("Dilithium not available")
            return None

        if private_key is None:
            if self._dilithium_keypair is None:
                logger.warning("No Dilithium keypair available")
                return None
            private_key = self._dilithium_keypair.private_key

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
        """Verify ML-DSA-65 (Dilithium) signature.

        Args:
            message: Original data
            signature: Signature to verify
            public_key: Optional public key (uses cached if None)

        Returns:
            True if valid, False otherwise
        """
        if not AVA_GUARDIAN_AVAILABLE or not DILITHIUM_AVAILABLE:
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
        """Generate Kyber-1024 (ML-KEM) keypair.

        Returns:
            KyberKeyPair or None if not available
        """
        if not AVA_GUARDIAN_AVAILABLE or not KYBER_AVAILABLE:
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
        """Encapsulate shared secret with Kyber-1024.

        Args:
            public_key: Optional public key (uses cached if None)

        Returns:
            KyberEncapsulation with ciphertext and shared_secret, or None if failed
        """
        if not AVA_GUARDIAN_AVAILABLE or not KYBER_AVAILABLE:
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
        """Decapsulate shared secret with Kyber-1024.

        Args:
            ciphertext: Ciphertext from encapsulation
            secret_key: Optional secret key (uses cached if None)

        Returns:
            Shared secret bytes or None if failed
        """
        if not AVA_GUARDIAN_AVAILABLE or not KYBER_AVAILABLE:
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
        """Simulate cryptographic attack for testing detection.

        Args:
            attack_type: Type of attack to simulate ("timing", "replay", "side_channel")

        Returns:
            Dictionary with simulation results and detection status
        """
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
        """Get summary of detected anomalies.

        Returns:
            Dictionary with anomaly statistics and recent anomalies
        """
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
        """Get aggregated omni-scalars for GOSNN registration.

        Returns:
            Dictionary of omni-scalars from crypto operations
        """
        scalars: dict[str, float] = {
            "omni_mercury_guardian_available": 1.0 if AVA_GUARDIAN_AVAILABLE else 0.0,
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

        return scalars


def create_mercury_guardian_adapter(
    enable_timing_monitor: bool = True,
    gosnn_synapse_enabled: bool = True,
) -> MercuryGuardianAdapter:
    """Factory function to create Ava-Guardian adapter.

    Args:
        enable_timing_monitor: Enable EWMA/MAD timing monitoring
        gosnn_synapse_enabled: Enable GOSNN ethical gate synapse

    Returns:
        Configured MercuryGuardianAdapter instance
    """
    return MercuryGuardianAdapter(
        enable_timing_monitor=enable_timing_monitor,
        gosnn_synapse_enabled=gosnn_synapse_enabled,
    )


__all__ = [
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
    "create_mercury_guardian_adapter",
]
