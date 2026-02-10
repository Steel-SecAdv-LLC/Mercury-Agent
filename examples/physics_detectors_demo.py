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

"""
Physics-Inspired Anomaly Detection Demo - Mercury Agent ♱ v1.4.0

Demonstrates the advanced physics-inspired anomaly detection modules:
1. SpectralVibrationDetector - Frequency-domain analysis with GNN/CNN
2. AccelerationDynamicsDetector - Kinematic and phase space analysis
3. UIUXAnomalyDetector - User interaction behavioral analysis
4. AdvancedPhysicsIntegratedDetector - Unified multi-modal fusion
"""

import argparse
import time

import numpy as np

from omni_mercury_engine.detectors.acceleration_dynamics import (
    AccelerationDynamicsDetector,
)
from omni_mercury_engine.detectors.advanced_physics_integration import (
    AdvancedPhysicsIntegratedDetector,
    PhysicsDetectorType,
)
from omni_mercury_engine.detectors.spectral_vibration import (
    SpectralAnalysisMode,
    SpectralVibrationDetector,
)
from omni_mercury_engine.detectors.uiux_anomaly import (
    InteractionType,
    UIUXAnomalyDetector,
    UserInteraction,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Physics-Inspired Anomaly Detection Demo - Mercury Agent v1.4.0"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection threshold (0.0-1.0, default: 0.5)",
    )
    parser.add_argument(
        "--module",
        type=str,
        choices=["all", "spectral", "dynamics", "uiux", "integrated"],
        default="all",
        help="Module to demo (default: all)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Include memory and runtime profiling",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed analysis results",
    )
    return parser.parse_args()


def generate_normal_vibration_signal(
    n_samples: int = 4096, sample_rate: float = 1000.0
) -> np.ndarray:
    """Generate a normal vibration signal with harmonic components."""
    t = np.linspace(0, n_samples / sample_rate, n_samples)
    # Fundamental frequency with harmonics
    signal = (
        np.sin(2 * np.pi * 50 * t)  # 50 Hz fundamental
        + 0.3 * np.sin(2 * np.pi * 100 * t)  # 2nd harmonic
        + 0.1 * np.sin(2 * np.pi * 150 * t)  # 3rd harmonic
        + 0.05 * np.random.randn(n_samples)  # Small noise
    )
    return signal


def generate_anomalous_vibration_signal(
    n_samples: int = 4096, sample_rate: float = 1000.0
) -> np.ndarray:
    """Generate an anomalous vibration signal with bearing fault signature."""
    t = np.linspace(0, n_samples / sample_rate, n_samples)
    # Normal components
    signal = np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 100 * t)
    # Add bearing fault signature (characteristic frequencies)
    fault_freq = 137.5  # Simulated BPFO
    signal += 0.5 * np.sin(2 * np.pi * fault_freq * t)
    # Add impulse responses (simulating bearing impacts)
    for i in range(0, n_samples, int(sample_rate / 20)):
        if i + 50 < n_samples:
            signal[i : i + 50] += (
                0.8
                * np.exp(-np.linspace(0, 5, 50))
                * np.sin(2 * np.pi * 500 * np.linspace(0, 0.05, 50))
            )
    return signal


def generate_normal_motion(n_samples: int = 1000, dt: float = 0.01) -> np.ndarray:
    """Generate normal uniform motion with small oscillations."""
    t = np.linspace(0, n_samples * dt, n_samples)
    # Smooth sinusoidal motion
    position = 5.0 * np.sin(0.5 * t) + 0.02 * np.random.randn(n_samples)
    return position


def generate_chaotic_motion(n_samples: int = 1000, dt: float = 0.01) -> np.ndarray:
    """Generate chaotic-like motion using Lorenz attractor x-component."""
    # Simplified chaotic signal
    t = np.linspace(0, n_samples * dt, n_samples)
    # Mix of frequencies creating chaotic-like behavior
    signal = (
        np.sin(2.0 * t)
        + 0.5 * np.sin(3.14159 * t)
        + 0.3 * np.sin(5.7 * t)
        + 0.2 * np.sin(8.3 * t + np.cumsum(0.1 * np.random.randn(n_samples)))
    )
    # Add occasional sudden jumps
    for i in range(5):
        idx = np.random.randint(100, n_samples - 100)
        signal[idx : idx + 10] += 3.0 * np.random.randn(10)
    return signal


def generate_normal_user_session(n_interactions: int = 50) -> list[UserInteraction]:
    """Generate normal user interaction session."""
    interactions = []
    current_time = 0.0

    np.random.seed(42)

    for i in range(n_interactions):
        # Variable timing (human-like)
        current_time += 0.5 + np.random.exponential(1.0)

        # Mix of interaction types
        if i % 10 == 0:
            int_type = InteractionType.PAGE_VIEW
        elif i % 5 == 0:
            int_type = InteractionType.SCROLL
        else:
            int_type = InteractionType.CLICK

        interaction = UserInteraction(
            timestamp=current_time,
            interaction_type=int_type,
            x=100 + np.random.randint(-20, 200),
            y=200 + np.random.randint(-20, 300),
            element_id=f"btn_{i % 10}",
            element_type="button",
            page_url=f"/page_{i % 5}",
            viewport_width=1920,
            viewport_height=1080,
        )
        interactions.append(interaction)

    return interactions


def generate_bot_user_session(n_interactions: int = 50) -> list[UserInteraction]:
    """Generate bot-like user interaction session."""
    interactions = []
    current_time = 0.0

    for i in range(n_interactions):
        # Perfectly regular timing (bot-like)
        current_time += 0.5

        # Linear mouse movement (suspicious)
        interaction = UserInteraction(
            timestamp=current_time,
            interaction_type=InteractionType.CLICK,
            x=100 + i * 5,  # Linear movement
            y=200 + i * 5,
            element_id=f"btn_{i % 3}",
            element_type="button",
            page_url=f"/page_{i % 2}",
            viewport_width=1920,
            viewport_height=1080,
        )
        interactions.append(interaction)

    return interactions


def generate_rage_click_session(n_interactions: int = 50) -> list[UserInteraction]:
    """Generate user session with rage clicks."""
    interactions = []
    current_time = 0.0

    np.random.seed(123)

    for i in range(n_interactions):
        if 20 <= i < 28:
            # Rage click sequence (rapid clicks on same element)
            current_time += 0.08  # Very fast
            x, y = 500, 300  # Same position
            element_id = "slow_button"
        else:
            # Normal interaction
            current_time += 0.5 + np.random.exponential(0.8)
            x = 100 + np.random.randint(0, 400)
            y = 100 + np.random.randint(0, 300)
            element_id = f"btn_{i % 10}"

        interaction = UserInteraction(
            timestamp=current_time,
            interaction_type=InteractionType.CLICK,
            x=x,
            y=y,
            element_id=element_id,
            element_type="button",
            page_url=f"/page_{i % 5}",
            viewport_width=1920,
            viewport_height=1080,
        )
        interactions.append(interaction)

    return interactions


def demo_spectral_vibration(threshold: float = 0.5, verbose: bool = False):
    """Demonstrate SpectralVibrationDetector."""
    print("\n" + "=" * 70)
    print("DEMO 1: SPECTRAL VIBRATION DETECTOR - FREQUENCY DOMAIN ANALYSIS")
    print("=" * 70)

    detector = SpectralVibrationDetector(
        {
            "threshold": threshold,
            "sample_rate": 1000.0,
            "analysis_mode": SpectralAnalysisMode.COMPREHENSIVE,
        }
    )

    # Train on normal vibration
    print("\n[Training] Fitting on normal vibration signal...")
    normal_signal = generate_normal_vibration_signal()
    detector.fit(normal_signal)
    print("  Training complete.")

    # Test 1: Normal signal
    print("\n[Test 1] Analyzing normal vibration signal...")
    test_normal = generate_normal_vibration_signal(n_samples=2048)
    result_normal = detector.detect(test_normal)

    print(f"  Is anomaly: {result_normal.get('is_anomaly', False)}")
    print(f"  Anomaly score: {result_normal.get('anomaly_score', 0.0):.4f}")
    print(f"  Spectral entropy: {result_normal.get('spectral_entropy', 0.0):.4f}")

    if verbose:
        print(f"  Dominant frequencies: {result_normal.get('dominant_frequencies', [])[:5]}")
        print(f"  Harmonic distortion: {result_normal.get('harmonic_distortion', 0.0):.4f}")

    # Test 2: Anomalous signal (bearing fault)
    print("\n[Test 2] Analyzing anomalous vibration signal (simulated bearing fault)...")
    test_anomaly = generate_anomalous_vibration_signal(n_samples=2048)
    result_anomaly = detector.detect(test_anomaly)

    print(f"  Is anomaly: {result_anomaly.get('is_anomaly', False)}")
    print(f"  Anomaly score: {result_anomaly.get('anomaly_score', 0.0):.4f}")
    print(f"  Vibration signature: {result_anomaly.get('vibration_signature', 'unknown')}")

    if verbose:
        print(f"  Spectral entropy: {result_anomaly.get('spectral_entropy', 0.0):.4f}")
        print(f"  Harmonic distortion: {result_anomaly.get('harmonic_distortion', 0.0):.4f}")


def demo_acceleration_dynamics(threshold: float = 0.5, verbose: bool = False):
    """Demonstrate AccelerationDynamicsDetector."""
    print("\n" + "=" * 70)
    print("DEMO 2: ACCELERATION DYNAMICS DETECTOR - KINEMATIC ANALYSIS")
    print("=" * 70)

    detector = AccelerationDynamicsDetector(
        {
            "threshold": threshold,
            "time_step": 0.01,
            "jerk_sensitivity": 2.0,
            "chaos_threshold": 0.1,
        }
    )

    # Train on normal motion
    print("\n[Training] Fitting on normal oscillatory motion...")
    normal_motion = generate_normal_motion()
    detector.fit(normal_motion)
    print("  Training complete.")

    # Test 1: Normal motion
    print("\n[Test 1] Analyzing normal motion signal...")
    test_normal = generate_normal_motion(n_samples=500)
    result_normal = detector.detect(test_normal)

    print(f"  Is anomaly: {result_normal.get('is_anomaly', False)}")
    print(f"  Anomaly score: {result_normal.get('anomaly_score', 0.0):.4f}")
    print(f"  Motion state: {result_normal.get('motion_state', 'unknown')}")
    print(f"  Lyapunov exponent: {result_normal.get('lyapunov_exponent', 0.0):.4f}")
    print(f"  Is chaotic: {result_normal.get('is_chaotic', False)}")

    if verbose:
        print(f"  Mean velocity: {result_normal.get('mean_velocity', 0.0):.4f}")
        print(f"  Mean acceleration: {result_normal.get('mean_acceleration', 0.0):.4f}")

    # Test 2: Chaotic motion
    print("\n[Test 2] Analyzing chaotic motion signal...")
    test_chaotic = generate_chaotic_motion(n_samples=500)
    result_chaotic = detector.detect(test_chaotic)

    print(f"  Is anomaly: {result_chaotic.get('is_anomaly', False)}")
    print(f"  Anomaly score: {result_chaotic.get('anomaly_score', 0.0):.4f}")
    print(f"  Motion state: {result_chaotic.get('motion_state', 'unknown')}")
    print(f"  Lyapunov exponent: {result_chaotic.get('lyapunov_exponent', 0.0):.4f}")
    print(f"  Is chaotic: {result_chaotic.get('is_chaotic', False)}")
    print(f"  Jerk anomaly: {result_chaotic.get('jerk_anomaly', False)}")


def demo_uiux_anomaly(threshold: float = 0.5, verbose: bool = False):
    """Demonstrate UIUXAnomalyDetector."""
    print("\n" + "=" * 70)
    print("DEMO 3: UI/UX ANOMALY DETECTOR - BEHAVIORAL ANALYSIS")
    print("=" * 70)

    detector = UIUXAnomalyDetector(
        {
            "threshold": threshold,
            "rage_click_threshold": 0.2,
            "rage_click_count": 4,
            "bot_detection_threshold": 0.7,
        }
    )

    # Train on normal user behavior
    print("\n[Training] Fitting on normal user session...")
    normal_session = generate_normal_user_session()
    detector.fit(normal_session)
    print("  Training complete.")

    # Test 1: Normal user session
    print("\n[Test 1] Analyzing normal user session...")
    test_normal = generate_normal_user_session(n_interactions=40)
    result_normal = detector.detect(test_normal)

    print(f"  Is anomaly: {result_normal.get('is_anomaly', False)}")
    print(f"  Anomaly score: {result_normal.get('anomaly_score', 0.0):.4f}")
    print(f"  Behavior class: {result_normal.get('behavior_class', 'unknown')}")
    print(f"  Bot probability: {result_normal.get('bot_probability', 0.0):.4f}")

    if verbose and "click_analysis" in result_normal:
        ca = result_normal["click_analysis"]
        print(f"  Rage clicks: {ca.rage_clicks}")
        print(f"  Dead clicks: {ca.dead_clicks}")

    # Test 2: Bot-like session
    print("\n[Test 2] Analyzing bot-like user session...")
    test_bot = generate_bot_user_session(n_interactions=40)
    result_bot = detector.detect(test_bot)

    print(f"  Is anomaly: {result_bot.get('is_anomaly', False)}")
    print(f"  Anomaly score: {result_bot.get('anomaly_score', 0.0):.4f}")
    print(f"  Behavior class: {result_bot.get('behavior_class', 'unknown')}")
    print(f"  Bot probability: {result_bot.get('bot_probability', 0.0):.4f}")

    # Test 3: Rage click session
    print("\n[Test 3] Analyzing session with rage clicks...")
    test_rage = generate_rage_click_session(n_interactions=40)
    result_rage = detector.detect(test_rage)

    print(f"  Is anomaly: {result_rage.get('is_anomaly', False)}")
    print(f"  Anomaly score: {result_rage.get('anomaly_score', 0.0):.4f}")
    print(f"  Anomaly categories: {result_rage.get('anomaly_categories', [])}")

    if "click_analysis" in result_rage:
        ca = result_rage["click_analysis"]
        print(f"  Rage clicks detected: {ca.rage_clicks}")


def demo_integrated_physics(threshold: float = 0.5, verbose: bool = False):
    """Demonstrate AdvancedPhysicsIntegratedDetector."""
    print("\n" + "=" * 70)
    print("DEMO 4: INTEGRATED PHYSICS DETECTOR - MULTI-MODAL FUSION")
    print("=" * 70)

    detector = AdvancedPhysicsIntegratedDetector(
        {
            "threshold": threshold,
            "enabled_detectors": [
                PhysicsDetectorType.SPECTRAL,
                PhysicsDetectorType.DYNAMICS,
                PhysicsDetectorType.UIUX,
            ],
            "fusion_weights": {
                "spectral": 0.4,
                "dynamics": 0.3,
                "uiux": 0.3,
            },
        }
    )

    # Prepare multi-modal training data
    print("\n[Training] Fitting integrated detector on multi-modal data...")
    train_data = {
        "spectral": generate_normal_vibration_signal(),
        "dynamics": generate_normal_motion(),
        "uiux": generate_normal_user_session(),
    }
    detector.fit(train_data)
    print("  Training complete.")

    # Test 1: All normal
    print("\n[Test 1] Analyzing normal multi-modal data...")
    test_normal = {
        "spectral": generate_normal_vibration_signal(n_samples=2048),
        "dynamics": generate_normal_motion(n_samples=500),
        "uiux": generate_normal_user_session(n_interactions=40),
    }
    result_normal = detector.detect(test_normal)

    print(f"  Is anomaly: {result_normal.get('is_anomaly', False)}")
    print(f"  Fused anomaly score: {result_normal.get('fused_anomaly_score', 0.0):.4f}")

    if verbose:
        print(f"  3R alignment: {result_normal.get('three_r_alignment', {})}")

    # Test 2: Anomalies in all modalities
    print("\n[Test 2] Analyzing anomalous multi-modal data...")
    test_anomaly = {
        "spectral": generate_anomalous_vibration_signal(n_samples=2048),
        "dynamics": generate_chaotic_motion(n_samples=500),
        "uiux": generate_bot_user_session(n_interactions=40),
    }
    result_anomaly = detector.detect(test_anomaly)

    print(f"  Is anomaly: {result_anomaly.get('is_anomaly', False)}")
    print(f"  Fused anomaly score: {result_anomaly.get('fused_anomaly_score', 0.0):.4f}")

    # Show individual detector contributions
    if "spectral_result" in result_anomaly:
        print(
            f"  Spectral score: {result_anomaly['spectral_result'].get('anomaly_score', 0.0):.4f}"
        )
    if "dynamics_result" in result_anomaly:
        print(
            f"  Dynamics score: {result_anomaly['dynamics_result'].get('anomaly_score', 0.0):.4f}"
        )
    if "uiux_result" in result_anomaly:
        print(f"  UI/UX score: {result_anomaly['uiux_result'].get('anomaly_score', 0.0):.4f}")


def profile_memory_usage():
    """Profile memory usage of physics detectors."""
    print("\n" + "=" * 70)
    print("MEMORY PROFILING")
    print("=" * 70)

    import tracemalloc

    tracemalloc.start()

    print("\n[1/4] SpectralVibrationDetector Memory Usage")
    snapshot1 = tracemalloc.take_snapshot()
    detector = SpectralVibrationDetector()
    signal = generate_normal_vibration_signal(n_samples=8192)
    detector.fit(signal)
    detector.detect(signal)
    snapshot2 = tracemalloc.take_snapshot()

    top_stats = snapshot2.compare_to(snapshot1, "lineno")
    total_mem = sum(stat.size_diff for stat in top_stats) / 1024 / 1024
    print(f"  Memory allocated: {total_mem:.2f} MB")

    print("\n[2/4] AccelerationDynamicsDetector Memory Usage")
    snapshot1 = tracemalloc.take_snapshot()
    detector = AccelerationDynamicsDetector()
    motion = generate_normal_motion(n_samples=2000)
    detector.fit(motion)
    detector.detect(motion)
    snapshot2 = tracemalloc.take_snapshot()

    top_stats = snapshot2.compare_to(snapshot1, "lineno")
    total_mem = sum(stat.size_diff for stat in top_stats) / 1024 / 1024
    print(f"  Memory allocated: {total_mem:.2f} MB")

    print("\n[3/4] UIUXAnomalyDetector Memory Usage")
    snapshot1 = tracemalloc.take_snapshot()
    detector = UIUXAnomalyDetector()
    session = generate_normal_user_session(n_interactions=100)
    detector.fit(session)
    detector.detect(session)
    snapshot2 = tracemalloc.take_snapshot()

    top_stats = snapshot2.compare_to(snapshot1, "lineno")
    total_mem = sum(stat.size_diff for stat in top_stats) / 1024 / 1024
    print(f"  Memory allocated: {total_mem:.2f} MB")

    print("\n[4/4] AdvancedPhysicsIntegratedDetector Memory Usage")
    snapshot1 = tracemalloc.take_snapshot()
    detector = AdvancedPhysicsIntegratedDetector()
    data = {
        "spectral": generate_normal_vibration_signal(),
        "dynamics": generate_normal_motion(),
        "uiux": generate_normal_user_session(),
    }
    detector.fit(data)
    detector.detect(data)
    snapshot2 = tracemalloc.take_snapshot()

    top_stats = snapshot2.compare_to(snapshot1, "lineno")
    total_mem = sum(stat.size_diff for stat in top_stats) / 1024 / 1024
    print(f"  Memory allocated: {total_mem:.2f} MB")

    tracemalloc.stop()


def profile_runtime():
    """Profile runtime performance."""
    print("\n" + "=" * 70)
    print("RUNTIME PROFILING")
    print("=" * 70)

    n_runs = 5

    print("\n[1/4] SpectralVibrationDetector Runtime")
    detector = SpectralVibrationDetector()
    signal = generate_normal_vibration_signal()
    detector.fit(signal)

    times = []
    for _ in range(n_runs):
        test_signal = generate_normal_vibration_signal(n_samples=2048)
        start = time.time()
        detector.detect(test_signal)
        times.append(time.time() - start)
    print(f"  Mean runtime: {np.mean(times)*1000:.2f} ms +/- {np.std(times)*1000:.2f} ms")

    print("\n[2/4] AccelerationDynamicsDetector Runtime")
    detector = AccelerationDynamicsDetector()
    motion = generate_normal_motion()
    detector.fit(motion)

    times = []
    for _ in range(n_runs):
        test_motion = generate_normal_motion(n_samples=500)
        start = time.time()
        detector.detect(test_motion)
        times.append(time.time() - start)
    print(f"  Mean runtime: {np.mean(times)*1000:.2f} ms +/- {np.std(times)*1000:.2f} ms")

    print("\n[3/4] UIUXAnomalyDetector Runtime")
    detector = UIUXAnomalyDetector()
    session = generate_normal_user_session()
    detector.fit(session)

    times = []
    for _ in range(n_runs):
        test_session = generate_normal_user_session(n_interactions=40)
        start = time.time()
        detector.detect(test_session)
        times.append(time.time() - start)
    print(f"  Mean runtime: {np.mean(times)*1000:.2f} ms +/- {np.std(times)*1000:.2f} ms")

    print("\n[4/4] AdvancedPhysicsIntegratedDetector Runtime")
    detector = AdvancedPhysicsIntegratedDetector()
    data = {
        "spectral": generate_normal_vibration_signal(),
        "dynamics": generate_normal_motion(),
        "uiux": generate_normal_user_session(),
    }
    detector.fit(data)

    times = []
    for _ in range(n_runs):
        test_data = {
            "spectral": generate_normal_vibration_signal(n_samples=2048),
            "dynamics": generate_normal_motion(n_samples=500),
            "uiux": generate_normal_user_session(n_interactions=40),
        }
        start = time.time()
        detector.detect(test_data)
        times.append(time.time() - start)
    print(f"  Mean runtime: {np.mean(times)*1000:.2f} ms +/- {np.std(times)*1000:.2f} ms")


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("Mercury Agent: PHYSICS-INSPIRED ANOMALY DETECTION DEMO (v1.4.0)")
    print(f"Threshold: {args.threshold} | Module: {args.module}")
    print("=" * 70)

    if args.module in ["all", "spectral"]:
        demo_spectral_vibration(args.threshold, args.verbose)

    if args.module in ["all", "dynamics"]:
        demo_acceleration_dynamics(args.threshold, args.verbose)

    if args.module in ["all", "uiux"]:
        demo_uiux_anomaly(args.threshold, args.verbose)

    if args.module in ["all", "integrated"]:
        demo_integrated_physics(args.threshold, args.verbose)

    if args.profile:
        profile_memory_usage()
        profile_runtime()

    print("\n" + "=" * 70)
    print("PHYSICS-INSPIRED DETECTORS: Advanced multi-domain anomaly detection")
    print("    - GNN/CNN spectral analysis for predictive maintenance")
    print("    - Kinematic phase space for motion anomalies")
    print("    - UI/UX behavioral analysis for bot detection")
    print("    - GOSNN-aligned ethical fusion with 3R mechanism")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
