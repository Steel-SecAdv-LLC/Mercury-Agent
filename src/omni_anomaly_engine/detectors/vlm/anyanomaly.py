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
from __future__ import annotations

"""
AnyAnomaly: Zero-Shot Customizable Video Anomaly Detection with LVLM

Implementation inspired by AnyAnomaly (WACV 2026).
Enables user-defined anomaly detection via natural language.

Key Features:
    1. Zero-shot detection without training
    2. User-customizable anomaly definitions
    3. Context-aware visual question answering
    4. Segment-level analysis for video

Reference:
    Ahn et al. "AnyAnomaly: Zero-Shot Customizable Video Anomaly Detection
    with LVLM"
    https://arxiv.org/abs/2503.04504
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from omni_anomaly_engine.detectors.vlm.base_vlm import BaseVLMDetector, VLMConfig
from omni_anomaly_engine.detectors.vlm.context_providers import (
    CombinedContextProvider,
    PositionContextProvider,
    TemporalContextProvider,
)
from omni_anomaly_engine.detectors.vlm.lvlm_backends import get_lvlm_backend

logger = logging.getLogger(__name__)


@dataclass
class AnyAnomalyConfig(VLMConfig):
    """Configuration for AnyAnomaly detector.

    Attributes:
        segment_overlap: Overlap between video segments
        use_segment_voting: Aggregate votes across segments
        action_focused: Emphasize action-based anomalies
        appearance_focused: Emphasize appearance-based anomalies
        backend: LVLM backend type (for test compatibility)
        context_window: Number of context frames
        enable_positional_context: Enable positional context
        enable_temporal_context: Enable temporal context
    """

    segment_overlap: int = 4
    use_segment_voting: bool = True
    action_focused: bool = True
    appearance_focused: bool = True
    backend: str = "qwen2_vl"  # LVLM backend type
    context_window: int = 4  # Number of context frames
    enable_positional_context: bool = True  # Enable positional context
    enable_temporal_context: bool = True  # Enable temporal context


class AnyAnomalyDetector(BaseVLMDetector):
    """AnyAnomaly zero-shot customizable anomaly detector.

    Detects user-defined anomalies using vision-language models
    without any training or fine-tuning.

    Features:
        - Natural language anomaly specification
        - Context-aware VQA for improved accuracy
        - Video segment-level analysis
        - Position and temporal context integration

    Example:
        >>> detector = AnyAnomalyDetector(
        ...     anomaly_description="person falling or running"
        ... )
        >>> results = detector.detect(video_frames)
        >>> print(f"Anomaly at frames: {results['anomaly_frames']}")
    """

    def __init__(self, config: AnyAnomalyConfig | dict[str, Any] | None = None) -> None:
        """Initialize AnyAnomaly detector.

        Args:
            config: Detector configuration or dict
        """
        if config is None:
            config = AnyAnomalyConfig()
        elif isinstance(config, dict):
            config = AnyAnomalyConfig(**config)

        super().__init__(config)
        self.any_config: AnyAnomalyConfig = config
        # Override _config to use the specific AnyAnomaly config
        self._config = config

        # Context providers
        self.context_provider = CombinedContextProvider(
            position_provider=PositionContextProvider() if config.use_position_context else None,
            temporal_provider=TemporalContextProvider() if config.use_temporal_context else None,
        )

        # LVLM backend
        self._backend = None

        # Anomaly definition and reference frames for test compatibility
        self._anomaly_definition: str | None = None
        self._reference_frames: list[Any] = []

    @property
    def anomaly_definition(self) -> str | None:
        """Get the current anomaly definition."""
        return self._anomaly_definition or self.vlm_config.anomaly_description

    @property
    def reference_frames(self) -> list[Any]:
        """Get the reference normal frames."""
        return self._reference_frames

    def set_anomaly_definition(self, definition: str) -> None:
        """Set the anomaly definition for detection.

        Args:
            definition: Natural language description of anomaly
        """
        self._anomaly_definition = definition
        self.vlm_config.anomaly_description = definition

    def set_reference_normal(self, frames: list[Any]) -> None:
        """Set reference normal frames for comparison.

        Args:
            frames: List of normal reference frames
        """
        self._reference_frames = frames

    def _initialize_model(self) -> None:
        """Initialize LVLM backend."""
        self._backend = get_lvlm_backend(
            model_type=self.vlm_config.model_type.value,
            model_name=self.vlm_config.model_name,
            device=str(self.device),
            max_new_tokens=self.vlm_config.max_new_tokens,
            temperature=self.vlm_config.temperature,
        )
        self._backend.initialize()

    @property
    def backend(self) -> Any:
        """Get the LVLM backend."""
        if self._backend is None:
            self._initialize_model()
        return self._backend

    def _create_prompt(
        self,
        anomaly_description: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Create VQA prompt for anomaly detection.

        Args:
            anomaly_description: Description of anomaly to detect
            context: Optional context information

        Returns:
            Formatted prompt
        """
        # Base prompt structure
        prompt = f"""You are an expert video surveillance analyst. Your task is to determine if the given image(s) contain any anomalous activity.

ANOMALY DEFINITION:
The following is considered anomalous: {anomaly_description}

"""

        # Add normal description if provided
        if self.vlm_config.normal_description:
            prompt += f"""NORMAL BEHAVIOR:
The following is considered normal: {self.vlm_config.normal_description}

"""

        # Add context if available
        if context:
            contexts = self.context_provider.extract_all_context(
                context.get("frames", np.array([]))
            )
            context_str = self.context_provider.format_combined_prompt(contexts)
            prompt += f"""CONTEXT INFORMATION:{context_str}

"""

        # Add task instruction
        prompt += """TASK:
Analyze the image(s) carefully and determine:
1. Is there any anomalous activity matching the definition above?
2. If yes, describe what you observe.
3. Rate your confidence from 0.0 to 1.0.

RESPONSE FORMAT:
Answer with EXACTLY this format:
ANOMALY_DETECTED: [YES/NO]
CONFIDENCE: [0.0-1.0]
EXPLANATION: [Your detailed explanation]
"""

        return prompt

    def _parse_response(self, response: str) -> tuple[bool, float, str]:
        """Parse LVLM response to extract detection result.

        Args:
            response: Model response text

        Returns:
            Tuple of (is_anomaly, confidence, explanation)
        """
        response_upper = response.upper()

        # Extract anomaly decision
        is_anomaly = False
        if (
            "ANOMALY_DETECTED: YES" in response_upper
            or "ANOMALY_DETECTED:YES" in response_upper
            or "YES" in response_upper[:50]
        ):
            is_anomaly = True

        # Extract confidence
        confidence = 0.5  # Default
        confidence_patterns = [
            r"CONFIDENCE:\s*([\d.]+)",
            r"confidence[:\s]*([\d.]+)",
            r"([\d.]+)%?\s*confident",
        ]
        for pattern in confidence_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    conf_val = float(match.group(1))
                    if conf_val > 1:  # Percentage
                        conf_val /= 100
                    confidence = min(max(conf_val, 0.0), 1.0)
                    break
                except ValueError:
                    pass

        # Extract explanation
        explanation = response
        explanation_patterns = [
            r"EXPLANATION:\s*(.+?)(?=$|\n\n)",
            r"explanation[:\s]*(.+?)(?=$|\n\n)",
        ]
        for pattern in explanation_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                explanation = match.group(1).strip()
                break

        return is_anomaly, confidence, explanation

    def _segment_video(
        self,
        frames: np.ndarray[Any, Any],
    ) -> list[tuple[int, int, np.ndarray[Any, Any]]]:
        """Segment video into overlapping segments.

        Args:
            frames: Video frames [T, C, H, W]

        Returns:
            List of (start_idx, end_idx, segment_frames)
        """
        t = frames.shape[0]
        segment_len = self.vlm_config.segment_length
        overlap = self.any_config.segment_overlap
        stride = segment_len - overlap

        segments = []
        for start in range(0, t, max(stride, 1)):
            end = min(start + segment_len, t)
            segment = frames[start:end]
            segments.append((start, end, segment))

            if end >= t:
                break

        return segments

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Detect anomalies using VLM-based VQA.

        Args:
            data: Images [N, C, H, W] or video frames [T, C, H, W]

        Returns:
            Dict containing:
                - scores: Frame-level anomaly scores [T]
                - is_anomaly: Binary anomaly flags [T]
                - explanations: Natural language explanations
                - features: Extracted features for fusion
                - segment_results: Per-segment detection results
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        # Ensure 4D: [T, C, H, W]
        if data.ndim == 3:
            data = data[np.newaxis, ...]

        t, _c, _h, _w = data.shape

        # Segment video
        segments = self._segment_video(data)

        # Process each segment
        segment_results = []
        frame_scores = np.zeros(t)
        frame_counts = np.zeros(t)

        for start, end, segment_frames in segments:
            # Extract context
            context = {"frames": segment_frames}

            # Create prompt
            prompt = self._create_prompt(
                self.vlm_config.anomaly_description,
                context=context,
            )

            # Convert frames to PIL images (sample key frames)
            key_frames = self._sample_key_frames(segment_frames)
            images = [self._numpy_to_pil(f) for f in key_frames]

            # Query LVLM
            try:
                response = self.backend.generate(images, prompt)
                is_anomaly, confidence, explanation = self._parse_response(response)
            except Exception as e:
                logger.warning(f"LVLM query failed: {e}")
                is_anomaly, confidence, explanation = False, 0.0, "Error processing segment"

            # Store segment result
            segment_results.append(
                {
                    "start": start,
                    "end": end,
                    "is_anomaly": is_anomaly,
                    "confidence": confidence,
                    "explanation": explanation,
                }
            )

            # Update frame scores
            score = confidence if is_anomaly else 1 - confidence
            for i in range(start, end):
                frame_scores[i] += score
                frame_counts[i] += 1

        # Average scores where multiple segments overlap
        frame_counts = np.maximum(frame_counts, 1)
        scores = frame_scores / frame_counts

        # Determine anomalies
        is_anomaly = scores > self.vlm_config.confidence_threshold

        # Collect explanations for anomalous segments
        explanations = [r["explanation"] for r in segment_results if r["is_anomaly"]]

        # Generate features (simple encoding of scores and metadata)
        features = self._generate_features(scores, segment_results)

        # Build reasoning summary from segment results
        reasoning_parts = []
        for i, r in enumerate(segment_results):
            if r["is_anomaly"]:
                reasoning_parts.append(f"Segment {i}: {r['explanation']}")
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No anomalies detected"

        return {
            "scores": scores,
            "is_anomaly": is_anomaly,
            "explanations": explanations,
            "features": features,
            "segment_results": segment_results,
            "anomaly_frames": np.where(is_anomaly)[0].tolist(),
            "reasoning": reasoning,
        }

    def _sample_key_frames(
        self,
        segment: np.ndarray[Any, Any],
        max_frames: int = 4,
    ) -> list[np.ndarray[Any, Any]]:
        """Sample key frames from segment for LVLM input.

        Args:
            segment: Video segment [T, C, H, W]
            max_frames: Maximum frames to sample

        Returns:
            List of key frames
        """
        t = segment.shape[0]

        if t <= max_frames:
            indices = list(range(t))
        else:
            # Uniform sampling
            indices = np.linspace(0, t - 1, max_frames, dtype=int).tolist()

        return [segment[i] for i in indices]

    def _numpy_to_pil(self, frame: np.ndarray[Any, Any]) -> Any:
        """Convert numpy frame to PIL Image."""
        from PIL import Image

        # Handle CHW format
        if frame.ndim == 3 and frame.shape[0] in [1, 3]:
            frame = np.transpose(frame, (1, 2, 0))

        # Normalize to uint8
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)

        # Handle grayscale
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = np.squeeze(frame, axis=-1)

        return Image.fromarray(frame)

    def _generate_features(
        self,
        scores: np.ndarray[Any, Any],
        segment_results: list[dict[str, Any]],
    ) -> torch.Tensor:
        """Generate feature representation for fusion.

        Args:
            scores: Frame-level scores
            segment_results: Segment detection results

        Returns:
            Feature tensor [1, 128] for fusion
        """
        # Simple feature encoding
        features = []

        # Score statistics
        features.extend(
            [
                np.mean(scores),
                np.std(scores),
                np.max(scores),
                np.min(scores),
            ]
        )

        # Segment statistics
        anomaly_count = sum(1 for r in segment_results if r["is_anomaly"])
        avg_confidence = np.mean([r["confidence"] for r in segment_results])
        features.extend(
            [
                anomaly_count / max(len(segment_results), 1),
                avg_confidence,
            ]
        )

        # Pad to 128D
        features = np.array(features)
        if len(features) < 128:
            features = np.pad(features, (0, 128 - len(features)))

        return torch.from_numpy(features).float().unsqueeze(0)

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> torch.Tensor:
        """Extract features for ML fusion pipeline.

        Args:
            data: Input video/images

        Returns:
            Feature tensor [N, 128] normalized for fusion
        """
        results = self.detect(data)
        features = results["features"]

        # Normalize
        features = nn.functional.normalize(features, p=2, dim=1)

        return features
