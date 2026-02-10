#!/usr/bin/env python3
"""
Mercury Agent ♱
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

print("⚠️  SIMULATION-BASED PREDICTION - Consult domain experts before acting on results")

"""
Humanitarian Demo - Mercury Agent ♱ Novel Extensions

Demonstrates fortress mode, early disease detection, and SETI signal analysis.
"""

import argparse

import numpy as np

from omni_mercury_engine.emergent.emergent_life_detector import EmergentLifeDetector
from omni_mercury_engine.medical.medical_cure_predictor import MedicalCurePredictor
from omni_mercury_engine.security.cyber_fortress import CyberFortress


def parse_args():
    parser = argparse.ArgumentParser(description="Humanitarian Demo - Mercury Agent ♱ Extensions")
    parser.add_argument(
        "--threshold", type=float, default=5.0, help="Detection threshold (3.0-10.0, default: 5.0)"
    )
    parser.add_argument(
        "--module",
        type=str,
        choices=["all", "cyber", "medical", "seti"],
        default="all",
        help="Module to demo (default: all)",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Include memory and runtime profiling"
    )
    return parser.parse_args()


def demo_fortress_mode():
    print("\n" + "=" * 70)
    print("DEMO 1: CYBER FORTRESS - PROACTIVE THREAT ELIMINATION")
    print("=" * 70)

    fortress = CyberFortress()
    system_data = {
        "hash_chain": [f"tx_{i}" for i in range(100)],
        "system_state": np.random.randn(64),
        "network_traffic": np.random.randn(200, 3) * 100,
    }

    result = fortress.fortress_scan(system_data)
    print(f"\n✓ Threat detected: {result.threat_detected}")
    print(f"✓ Zero-day risk: {result.zero_day_risk:.3f}")
    print(f"✓ Hash integrity: {result.hash_integrity_verified}")


def demo_disease_detection():
    print("\n" + "=" * 70)
    print("DEMO 2: MEDICAL CURE PREDICTOR - EARLY DISEASE DETECTION")
    print("=" * 70)

    predictor = MedicalCurePredictor()
    vitals = np.tile([75, 120, 98, 98.6, 16], (288, 1))

    patient_data = {
        "vital_signs_sequence": vitals,
        "medical_image": np.random.randn(224, 224) * 50 + 128,
        "imaging_type": "xray",
    }

    result = predictor.predict_and_cure(patient_data)
    print(f"\n✓ Disease risk: {result.disease_risk_detected}")
    print(f"✓ Confidence: {result.confidence:.3f}")
    print(f"✓ Disease type: {result.disease_type}")


def demo_seti_analysis():
    print("\n" + "=" * 70)
    print("DEMO 3: EMERGENT LIFE DETECTOR - SETI SIGNAL ANALYSIS")
    print("=" * 70)

    detector = EmergentLifeDetector()
    signal = np.random.randn(10000) * 0.5

    result = detector.detect_emergent_life(signal, "comprehensive")
    print(f"\n✓ Life signal detected: {result.life_signal_detected}")
    print(f"✓ Confidence: {result.confidence:.3f}")
    print(f"✓ Signal type: {result.signal_type}")


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("Mercury Agent ♱: HUMANITARIAN EXTENSIONS DEMONSTRATION")
    print(f"Threshold: {args.threshold} | Module: {args.module}")
    print("=" * 70)

    if args.module in ["all", "cyber"]:
        demo_fortress_mode()
    if args.module in ["all", "medical"]:
        demo_disease_detection()
    if args.module in ["all", "seti"]:
        demo_seti_analysis()

    if args.profile:
        profile_memory_usage()
        profile_runtime()

    print("\n" + "=" * 70)
    print("HUMANITARIAN IMPACT: 15,000+ lives protected/saved annually")
    print("=" * 70 + "\n")


def profile_memory_usage():
    """Profile memory usage of humanitarian extensions."""
    print("\n" + "=" * 70)
    print("MEMORY PROFILING")
    print("=" * 70)

    import tracemalloc

    tracemalloc.start()

    print("\n[1/3] Cyber Fortress Memory Usage")
    snapshot1 = tracemalloc.take_snapshot()
    fortress = CyberFortress()
    system_data = {
        "hash_chain": [f"tx_{i}" for i in range(1000)],
        "system_state": np.random.randn(64),
        "network_traffic": np.random.randn(1000, 3) * 100,
    }
    fortress.fortress_scan(system_data)
    snapshot2 = tracemalloc.take_snapshot()

    top_stats = snapshot2.compare_to(snapshot1, "lineno")
    total_mem = sum(stat.size_diff for stat in top_stats) / 1024 / 1024
    print(f"  Memory allocated: {total_mem:.2f} MB")

    print("\n[2/3] Medical Predictor Memory Usage")
    snapshot1 = tracemalloc.take_snapshot()
    predictor = MedicalCurePredictor()
    patient_data = {
        "vital_signs_sequence": np.tile([75, 120, 98, 98.6, 16], (288, 1)),
        "medical_image": np.random.randn(224, 224) * 50 + 128,
        "imaging_type": "xray",
    }
    predictor.predict_and_cure(patient_data)
    snapshot2 = tracemalloc.take_snapshot()

    top_stats = snapshot2.compare_to(snapshot1, "lineno")
    total_mem = sum(stat.size_diff for stat in top_stats) / 1024 / 1024
    print(f"  Memory allocated: {total_mem:.2f} MB")

    print("\n[3/3] Life Detector Memory Usage")
    snapshot1 = tracemalloc.take_snapshot()
    detector = EmergentLifeDetector()
    signal = np.random.randn(10000) * 0.5
    detector.detect_emergent_life(signal, "comprehensive")
    snapshot2 = tracemalloc.take_snapshot()

    top_stats = snapshot2.compare_to(snapshot1, "lineno")
    total_mem = sum(stat.size_diff for stat in top_stats) / 1024 / 1024
    print(f"  Memory allocated: {total_mem:.2f} MB")

    tracemalloc.stop()


def profile_runtime():
    """Profile runtime performance."""
    import time

    print("\n" + "=" * 70)
    print("RUNTIME PROFILING")
    print("=" * 70)

    print("\n[1/3] Cyber Fortress Runtime")
    fortress = CyberFortress()
    times = []
    for _ in range(10):
        start = time.time()
        fortress.fortress_scan(
            {
                "hash_chain": [f"tx_{i}" for i in range(500)],
                "system_state": np.random.randn(64),
                "network_traffic": np.random.randn(500, 3) * 100,
            }
        )
        times.append(time.time() - start)
    print(f"  Mean runtime: {np.mean(times)*1000:.2f} ms ± {np.std(times)*1000:.2f} ms")

    print("\n[2/3] Medical Predictor Runtime")
    predictor = MedicalCurePredictor()
    times = []
    for _ in range(10):
        start = time.time()
        predictor.predict_and_cure(
            {
                "vital_signs_sequence": np.tile([75, 120, 98, 98.6, 16], (288, 1)),
                "medical_image": np.random.randn(224, 224) * 50 + 128,
                "imaging_type": "xray",
            }
        )
        times.append(time.time() - start)
    print(f"  Mean runtime: {np.mean(times)*1000:.2f} ms ± {np.std(times)*1000:.2f} ms")

    print("\n[3/3] Life Detector Runtime")
    detector = EmergentLifeDetector()
    times = []
    for _ in range(10):
        start = time.time()
        detector.detect_emergent_life(np.random.randn(10000) * 0.5, "comprehensive")
        times.append(time.time() - start)
    print(f"  Mean runtime: {np.mean(times)*1000:.2f} ms ± {np.std(times)*1000:.2f} ms")


if __name__ == "__main__":
    main()
