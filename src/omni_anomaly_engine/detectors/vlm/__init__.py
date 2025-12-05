"""
OMNI ♱ AVA (O♱A)
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
Vision-Language Model (VLM) Anomaly Detection Module

Zero-shot and customizable anomaly detection using Large Vision-Language Models.
Enables training-free detection with natural language anomaly descriptions.

Key Features:
    - Zero-shot detection without training
    - User-defined anomaly types via text prompts
    - Context-aware visual question answering
    - Training-free deployment for new domains

Algorithms:
    - AnyAnomaly: Customizable VAD with context-aware VQA (WACV 2026)
    - LAVAD: Training-free LLM-based VAD (CVPR 2024)
    - ALFA: Runtime prompt adaptation for zero-shot AD (ACM MM 2024)

Research Sources:
    - AnyAnomaly: "Zero-Shot Customizable Video Anomaly Detection with LVLM"
    - LAVAD: "Harnessing Large Language Models for Training-free Video Anomaly Detection"
    - ALFA: "Do LLMs Understand Visual Anomalies?"

Example:
    Basic usage::

        from omni_anomaly_engine.detectors.vlm import AnyAnomalyDetector

        detector = AnyAnomalyDetector(
            anomaly_description="person falling down or lying on ground"
        )

        # No training needed - zero-shot detection
        results = detector.detect(video_frames)
        print(f"Anomaly detected at frames: {results['anomaly_frames']}")
"""

from omni_anomaly_engine.detectors.vlm.anyanomaly import AnyAnomalyDetector
from omni_anomaly_engine.detectors.vlm.base_vlm import BaseVLMDetector, VLMConfig
from omni_anomaly_engine.detectors.vlm.context_providers import (
    PositionContextProvider,
    TemporalContextProvider,
)
from omni_anomaly_engine.detectors.vlm.lavad import LAVADDetector
from omni_anomaly_engine.detectors.vlm.lvlm_backends import LVLMBackend, get_lvlm_backend

__all__ = [
    # Detectors
    "AnyAnomalyDetector",
    # Base classes
    "BaseVLMDetector",
    "LAVADDetector",
    # Backends
    "LVLMBackend",
    # Context providers
    "PositionContextProvider",
    "TemporalContextProvider",
    "VLMConfig",
    "get_lvlm_backend",
]
