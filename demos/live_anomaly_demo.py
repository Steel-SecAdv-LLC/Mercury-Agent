#!/usr/bin/env python3
"""
Mercury Agent ♱ - Live Anomaly Detection Demo
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

"""
Live Anomaly Detection Demo

This script demonstrates Mercury Agent ♱'s real-time anomaly detection
capabilities using simulated streaming data. It showcases:

1. Real-time data stream processing
2. Multi-domain anomaly detection (security, medical, environmental)
3. Ethical AI governance with benevolence scoring
4. Visual output with detection timestamps and confidence scores

Usage:
    python demos/live_anomaly_demo.py --domain security
    python demos/live_anomaly_demo.py --domain medical
    python demos/live_anomaly_demo.py --domain environmental
    python demos/live_anomaly_demo.py --all
"""

import argparse
import json
import time
from collections.abc import Generator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest

BANNER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ███╗   ███╗███████╗██████╗  ██████╗██╗   ██╗██████╗ ██╗   ██╗             ║
║   ████╗ ████║██╔════╝██╔══██╗██╔════╝██║   ██║██╔══██╗╚██╗ ██╔╝             ║
║   ██╔████╔██║█████╗  ██████╔╝██║     ██║   ██║██████╔╝ ╚████╔╝              ║
║   ██║╚██╔╝██║██╔══╝  ██╔══██╗██║     ██║   ██║██╔══██╗  ╚██╔╝               ║
║   ██║ ╚═╝ ██║███████╗██║  ██║╚██████╗╚██████╔╝██║  ██║   ██║                ║
║   ╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝                ║
║                                                                              ║
║                    ♱ AGENT - Live Anomaly Detection Demo                     ║
║                                                                              ║
║   Neuro-Symbolic AI | Ethical Governance | Civilization-First                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


@dataclass
class AnomalyDetection:
    """Represents a detected anomaly."""

    timestamp: str
    domain: str
    anomaly_score: float
    is_anomaly: bool
    confidence: float
    benevolence_score: float
    details: dict[str, Any]


@dataclass
class StreamStats:
    """Statistics for the data stream."""

    total_samples: int
    anomalies_detected: int
    detection_rate: float
    avg_confidence: float
    avg_benevolence: float
    runtime_seconds: float


class LiveAnomalyDetector:
    """Real-time anomaly detection engine."""

    def __init__(self, domain: str = "security", contamination: float = 0.1) -> None:
        self.domain = domain
        self.contamination = contamination
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        self._is_fitted = False
        self._detection_history: list[AnomalyDetection] = []

    def _generate_stream_data(
        self,
        n_samples: int = 100,
        anomaly_ratio: float = 0.15,
    ) -> Generator[tuple[np.ndarray, bool], None, None]:
        """Generate simulated streaming data for the specified domain."""
        rng = np.random.default_rng(int(time.time()) % 2**31)

        for i in range(n_samples):
            is_anomaly = rng.random() < anomaly_ratio

            if self.domain == "security":
                data = self._generate_security_sample(rng, is_anomaly)
            elif self.domain == "medical":
                data = self._generate_medical_sample(rng, is_anomaly)
            elif self.domain == "environmental":
                data = self._generate_environmental_sample(rng, is_anomaly)
            else:
                data = self._generate_generic_sample(rng, is_anomaly)

            yield data, is_anomaly

    def _generate_security_sample(self, rng: np.random.Generator, is_anomaly: bool) -> np.ndarray:
        """Generate network traffic-like data."""
        if is_anomaly:
            return np.array(
                [
                    rng.exponential(5000),
                    rng.exponential(10000),
                    rng.integers(100, 1000),
                    rng.uniform(0.5, 1.0),
                    rng.uniform(0.3, 0.8),
                    rng.exponential(50),
                    rng.uniform(0.7, 1.0),
                    rng.integers(50, 200),
                ]
            )
        else:
            return np.array(
                [
                    rng.exponential(500),
                    rng.exponential(1000),
                    rng.integers(1, 50),
                    rng.uniform(0.0, 0.2),
                    rng.uniform(0.0, 0.1),
                    rng.exponential(5),
                    rng.uniform(0.0, 0.3),
                    rng.integers(1, 20),
                ]
            )

    def _generate_medical_sample(self, rng: np.random.Generator, is_anomaly: bool) -> np.ndarray:
        """Generate vital signs-like data."""
        if is_anomaly:
            return np.array(
                [
                    rng.normal(120, 15),
                    rng.normal(160, 20),
                    rng.normal(50, 10),
                    rng.normal(28, 5),
                    rng.normal(90, 5),
                    rng.normal(39.0, 0.5),
                    rng.uniform(0.5, 1.0),
                    rng.uniform(0.6, 1.0),
                ]
            )
        else:
            return np.array(
                [
                    rng.normal(75, 10),
                    rng.normal(120, 10),
                    rng.normal(80, 8),
                    rng.normal(16, 2),
                    rng.normal(98, 1),
                    rng.normal(37.0, 0.3),
                    rng.uniform(0.0, 0.2),
                    rng.uniform(0.0, 0.2),
                ]
            )

    def _generate_environmental_sample(
        self, rng: np.random.Generator, is_anomaly: bool
    ) -> np.ndarray:
        """Generate environmental sensor-like data."""
        if is_anomaly:
            return np.array(
                [
                    rng.normal(1050, 20),
                    rng.normal(45, 10),
                    rng.normal(95, 5),
                    rng.normal(80, 20),
                    rng.uniform(0.5, 1.0),
                    rng.normal(150, 30),
                    rng.uniform(0.6, 1.0),
                    rng.normal(8.5, 0.5),
                ]
            )
        else:
            return np.array(
                [
                    rng.normal(1013, 5),
                    rng.normal(20, 5),
                    rng.normal(50, 10),
                    rng.normal(10, 5),
                    rng.uniform(0.0, 0.2),
                    rng.normal(50, 10),
                    rng.uniform(0.0, 0.2),
                    rng.normal(7.83, 0.1),
                ]
            )

    def _generate_generic_sample(self, rng: np.random.Generator, is_anomaly: bool) -> np.ndarray:
        """Generate generic anomaly data."""
        if is_anomaly:
            return rng.normal(5, 2, 8)
        else:
            return rng.normal(0, 1, 8)

    def _compute_benevolence_score(self, anomaly_score: float, confidence: float) -> float:
        """Compute ethical benevolence score for the detection."""
        base_score = 0.95
        detection_bonus = 0.03 if anomaly_score > 0.5 else 0.01
        confidence_bonus = confidence * 0.02
        return min(0.99, base_score + detection_bonus + confidence_bonus)

    def _categorize_heart_rate(self, hr: float) -> str:
        """Categorize heart rate into clinical buckets (no raw values logged)."""
        if hr < 60:
            return "bradycardia"
        elif hr > 100:
            return "tachycardia"
        else:
            return "normal"

    def _categorize_blood_pressure(self, systolic: float, diastolic: float) -> str:
        """Categorize blood pressure into clinical buckets (no raw values logged)."""
        if systolic < 90 or diastolic < 60:
            return "hypotension"
        elif systolic >= 180 or diastolic >= 120:
            return "hypertensive-crisis"
        elif systolic >= 140 or diastolic >= 90:
            return "stage2-hypertension"
        elif systolic >= 130 or diastolic >= 80:
            return "stage1-hypertension"
        elif systolic >= 120:
            return "elevated"
        else:
            return "normal"

    def _get_domain_details(self, data: np.ndarray, anomaly_score: float) -> dict[str, Any]:
        """Get domain-specific details for the detection."""
        if self.domain == "security":
            return {
                "src_bytes": float(data[0]),
                "dst_bytes": float(data[1]),
                "connection_count": int(data[2]),
                "error_rate": float(data[3]),
                "threat_level": (
                    "HIGH" if anomaly_score > 0.7 else "MEDIUM" if anomaly_score > 0.4 else "LOW"
                ),
            }
        elif self.domain == "medical":
            return {
                "heart_rate": float(data[0]),
                "systolic_bp": float(data[1]),
                "diastolic_bp": float(data[2]),
                "resp_rate": float(data[3]),
                "spo2": float(data[4]),
                "temperature": float(data[5]),
                "alert_level": (
                    "CRITICAL"
                    if anomaly_score > 0.7
                    else "WARNING" if anomaly_score > 0.4 else "NORMAL"
                ),
            }
        elif self.domain == "environmental":
            return {
                "pressure_hpa": float(data[0]),
                "temperature_c": float(data[1]),
                "humidity_pct": float(data[2]),
                "wind_speed_kmh": float(data[3]),
                "aqi": float(data[5]),
                "schumann_hz": float(data[7]),
                "severity": (
                    "SEVERE"
                    if anomaly_score > 0.7
                    else "MODERATE" if anomaly_score > 0.4 else "NORMAL"
                ),
            }
        else:
            return {"raw_features": data.tolist()}

    def run_live_detection(
        self,
        n_samples: int = 50,
        delay_ms: int = 200,
        verbose: bool = True,
    ) -> StreamStats:
        """
        Run live anomaly detection on streaming data.

        Args:
            n_samples: Number of samples to process
            delay_ms: Delay between samples in milliseconds
            verbose: Whether to print detection output

        Returns:
            StreamStats with detection statistics
        """
        start_time = time.time()

        training_data = []
        for data, _ in self._generate_stream_data(n_samples=200, anomaly_ratio=0.1):
            training_data.append(data)

        training_array = np.array(training_data)
        self.model.fit(training_array)
        self._is_fitted = True

        if verbose:
            print(f"\n{'='*70}")
            print(f"  LIVE DETECTION - Domain: {self.domain.upper()}")
            print(f"  Processing {n_samples} samples with {delay_ms}ms delay")
            print(f"{'='*70}\n")

        anomalies_detected = 0
        total_confidence = 0.0
        total_benevolence = 0.0

        for i, (data, ground_truth) in enumerate(
            self._generate_stream_data(n_samples=n_samples, anomaly_ratio=0.15)
        ):
            raw_score = -self.model.score_samples(data.reshape(1, -1))[0]
            anomaly_score = (raw_score - raw_score.min()) / (
                raw_score.max() - raw_score.min() + 1e-8
            )
            anomaly_score = float(np.clip(raw_score / 0.5, 0, 1))

            prediction = self.model.predict(data.reshape(1, -1))[0]
            is_anomaly = prediction == -1

            confidence = (
                min(0.99, 0.7 + anomaly_score * 0.3)
                if is_anomaly
                else 0.8 + (1 - anomaly_score) * 0.15
            )
            benevolence = self._compute_benevolence_score(anomaly_score, confidence)

            detection = AnomalyDetection(
                timestamp=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                domain=self.domain,
                anomaly_score=round(anomaly_score, 4),
                is_anomaly=is_anomaly,
                confidence=round(confidence, 4),
                benevolence_score=round(benevolence, 4),
                details=self._get_domain_details(data, anomaly_score),
            )

            self._detection_history.append(detection)

            if is_anomaly:
                anomalies_detected += 1

            total_confidence += confidence
            total_benevolence += benevolence

            if verbose:
                self._print_detection(i + 1, n_samples, detection)

            time.sleep(delay_ms / 1000)

        runtime = time.time() - start_time

        stats = StreamStats(
            total_samples=n_samples,
            anomalies_detected=anomalies_detected,
            detection_rate=anomalies_detected / n_samples,
            avg_confidence=total_confidence / n_samples,
            avg_benevolence=total_benevolence / n_samples,
            runtime_seconds=runtime,
        )

        if verbose:
            self._print_summary(stats)

        return stats

    def _print_detection(self, sample_num: int, total: int, detection: AnomalyDetection) -> None:
        """Print a single detection result."""
        status = "ANOMALY" if detection.is_anomaly else "NORMAL "
        color_code = "\033[91m" if detection.is_anomaly else "\033[92m"
        reset_code = "\033[0m"

        print(
            f"[{sample_num:3d}/{total}] {detection.timestamp} | "
            f"{color_code}{status}{reset_code} | "
            f"Score: {detection.anomaly_score:.3f} | "
            f"Conf: {detection.confidence:.3f} | "
            f"Benev: {detection.benevolence_score:.3f}"
        )

        if detection.is_anomaly:
            details = detection.details
            if self.domain == "security":
                print(
                    f"         └─ Threat: {details.get('threat_level', 'N/A')} | "
                    f"Bytes: {details.get('src_bytes', 0):.0f}/{details.get('dst_bytes', 0):.0f}"
                )
            elif self.domain == "medical":
                # Use categorized values instead of raw vitals for privacy
                hr_category = self._categorize_heart_rate(details.get("heart_rate", 75))
                bp_category = self._categorize_blood_pressure(
                    details.get("systolic_bp", 120), details.get("diastolic_bp", 80)
                )
                print(
                    f"         └─ Alert: {details.get('alert_level', 'N/A')} | "
                    f"HR: {hr_category} | BP: {bp_category}"
                )
            elif self.domain == "environmental":
                print(
                    f"         └─ Severity: {details.get('severity', 'N/A')} | "
                    f"Temp: {details.get('temperature_c', 0):.1f}°C | "
                    f"Pressure: {details.get('pressure_hpa', 0):.0f}hPa"
                )

    def _print_summary(self, stats: StreamStats) -> None:
        """Print detection summary."""
        print(f"\n{'='*70}")
        print("  DETECTION SUMMARY")
        print(f"{'='*70}")
        print(f"  Total Samples:      {stats.total_samples}")
        print(f"  Anomalies Detected: {stats.anomalies_detected}")
        print(f"  Detection Rate:     {stats.detection_rate:.2%}")
        print(f"  Avg Confidence:     {stats.avg_confidence:.4f}")
        print(f"  Avg Benevolence:    {stats.avg_benevolence:.4f}")
        print(f"  Runtime:            {stats.runtime_seconds:.2f}s")
        print(f"{'='*70}\n")

    def save_results(self, output_path: str | Path) -> None:
        """Save detection results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def convert_to_serializable(obj: Any) -> Any:
            """Convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, bool):
                return obj
            return obj

        detections = [convert_to_serializable(asdict(d)) for d in self._detection_history]

        results = {
            "domain": self.domain,
            "timestamp": datetime.now(UTC).isoformat(),
            "detections": detections,
            "summary": {
                "total": len(self._detection_history),
                "anomalies": sum(1 for d in self._detection_history if d.is_anomaly),
            },
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {output_path}")


def main() -> None:
    """Main entry point for the live demo."""
    parser = argparse.ArgumentParser(
        description="Mercury Agent ♱ Live Anomaly Detection Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demos/live_anomaly_demo.py --domain security
  python demos/live_anomaly_demo.py --domain medical --samples 100
  python demos/live_anomaly_demo.py --all --delay 100
        """,
    )
    parser.add_argument(
        "--domain",
        choices=["security", "medical", "environmental"],
        default="security",
        help="Domain for anomaly detection (default: security)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run demo for all domains",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=30,
        help="Number of samples to process (default: 30)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=150,
        help="Delay between samples in ms (default: 150)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (optional)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    print(BANNER)

    domains = ["security", "medical", "environmental"] if args.all else [args.domain]

    for domain in domains:
        detector = LiveAnomalyDetector(domain=domain)
        stats = detector.run_live_detection(
            n_samples=args.samples,
            delay_ms=args.delay,
            verbose=not args.quiet,
        )

        if args.output:
            output_path = args.output.replace(".json", f"_{domain}.json")
            detector.save_results(output_path)

    print("\n♱ Demo complete. Mercury Agent ♱ - Civilization-First AI")


if __name__ == "__main__":
    main()
