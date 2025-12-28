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
Base classes for Vision-Language Model anomaly detection.

Provides unified interface for zero-shot VLM-based anomaly detection
using Large Vision-Language Models (LVLMs).
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

from omni_mercury_engine.core.base import BaseDetector

if TYPE_CHECKING:
    import numpy as np


class LVLMType(Enum):
    """Supported LVLM backend types."""

    QWEN2_VL = "qwen2_vl"
    MINICPM_V = "minicpm_v"
    LLAVA = "llava"
    CHAT_UNIVI = "chat_univi"
    INTERNVL = "internvl"
    LOCAL_CUSTOM = "local_custom"


class ContextType(Enum):
    """Types of context for VQA."""

    POSITION = "position"  # Spatial context
    TEMPORAL = "temporal"  # Temporal sequence context
    SEMANTIC = "semantic"  # Scene understanding context


@dataclass
class VLMConfig:
    """Configuration for VLM-based anomaly detectors.

    Attributes:
        model_name: LVLM model identifier
        model_type: Type of LVLM backend
        anomaly_description: Natural language description of anomaly
        normal_description: Description of normal behavior (optional)
        device: Computation device
        max_new_tokens: Maximum tokens for generation
        temperature: Sampling temperature
        use_position_context: Enable spatial context
        use_temporal_context: Enable temporal context
        segment_length: Number of frames per segment for video
        confidence_threshold: Minimum confidence for anomaly detection
    """

    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    model_type: LVLMType = LVLMType.QWEN2_VL
    anomaly_description: str = "unusual or abnormal activity"
    normal_description: str | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 256
    temperature: float = 0.1
    use_position_context: bool = True
    use_temporal_context: bool = True
    segment_length: int = 16
    confidence_threshold: float = 0.5
    batch_size: int = 1


class BaseVLMDetector(BaseDetector):
    """Abstract base class for VLM-based anomaly detectors.

    Provides common functionality for zero-shot anomaly detection
    using Vision-Language Models.

    Features:
        - Zero-shot detection without training
        - Natural language anomaly specification
        - Context-aware visual question answering
        - Video and image support
    """

    def __init__(self, config: VLMConfig | dict[str, Any] | None = None) -> None:
        """Initialize VLM detector.

        Args:
            config: Detector configuration
        """
        if config is None:
            self.vlm_config = VLMConfig()
        elif isinstance(config, dict):
            self.vlm_config = VLMConfig(**config)
        else:
            self.vlm_config = config

        # Expose config property for test compatibility
        self._config = self.vlm_config

        # Initialize BaseDetector attributes manually (avoid calling __init__ which sets self.config)
        self.threshold = 0.5
        self._is_fitted = True  # VLM doesn't need fitting - mark as ready

        self.device = torch.device(self.vlm_config.device)
        self._model: Any = None
        self._processor: Any = None

    @property
    def config(self) -> VLMConfig:
        """Get the detector configuration."""
        return self._config

    @property
    def model(self) -> Any:
        """Get the LVLM model."""
        if self._model is None:
            self._initialize_model()
        return self._model

    @property
    def processor(self) -> Any:
        """Get the LVLM processor/tokenizer."""
        if self._processor is None:
            self._initialize_model()
        return self._processor

    def _initialize_model(self) -> None:
        """Initialize the LVLM model and processor.

        Note:
            Subclasses must override this method.
        """
        raise NotImplementedError(
            "Subclasses must implement _initialize_model() for VLM detectors."
        )

    def _create_prompt(
        self,
        anomaly_description: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Create the VQA prompt for anomaly detection.

        Args:
            anomaly_description: Description of anomaly to detect
            context: Optional context information

        Returns:
            Formatted prompt string

        Note:
            Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement _create_prompt() for VLM detectors.")

    def _parse_response(self, response: str) -> tuple[bool, float, str]:
        """Parse LVLM response to extract anomaly decision.

        Args:
            response: Model response text

        Returns:
            Tuple of (is_anomaly, confidence, explanation)

        Note:
            Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement _parse_response() for VLM detectors.")

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> BaseVLMDetector:
        """VLM detectors are zero-shot - no fitting required.

        Args:
            data: Ignored (included for interface compatibility)

        Returns:
            Self for method chaining
        """
        # Zero-shot - no training needed
        self._is_fitted = True
        return self

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Detect anomalies using VLM.

        Args:
            data: Images [N, C, H, W] or video frames [T, C, H, W]

        Returns:
            Dict containing:
                - scores: Anomaly confidence scores
                - is_anomaly: Binary anomaly flags
                - explanations: Natural language explanations
                - features: Extracted features for fusion

        Note:
            Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement detect() for VLM detectors.")

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> torch.Tensor:
        """Extract features for ML fusion pipeline.

        Args:
            data: Input images/video

        Returns:
            Feature tensor for fusion

        Note:
            Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement extract_features() for VLM detectors.")

    def _sample_frames(
        self,
        video: torch.Tensor,
        n_frames: int = 8,
    ) -> list[torch.Tensor]:
        """Sample frames uniformly from video for VLM processing.

        Args:
            video: Video tensor [T, H, W, C] or [T, C, H, W]
            n_frames: Number of frames to sample

        Returns:
            List of sampled frame tensors
        """
        total_frames = video.shape[0]

        if total_frames <= n_frames:
            # Return all frames if video is shorter than requested
            return [video[i] for i in range(total_frames)]

        # Uniform sampling
        indices = torch.linspace(0, total_frames - 1, n_frames).long()
        return [video[idx] for idx in indices]

    def set_anomaly_description(self, description: str) -> None:
        """Update the anomaly description for detection.

        Args:
            description: New anomaly description
        """
        self.vlm_config.anomaly_description = description

    def set_normal_description(self, description: str) -> None:
        """Update the normal behavior description.

        Args:
            description: Normal behavior description
        """
        self.vlm_config.normal_description = description


class VQAResult:
    """Container for Visual Question Answering results."""

    def __init__(
        self,
        is_anomaly: bool,
        confidence: float,
        explanation: str,
        raw_response: str,
        frame_indices: list[int] | None = None,
    ):
        """Initialize VQA result.

        Args:
            is_anomaly: Whether anomaly was detected
            confidence: Confidence score [0, 1]
            explanation: Natural language explanation
            raw_response: Raw model response
            frame_indices: Relevant frame indices for video
        """
        self.is_anomaly = is_anomaly
        self.confidence = confidence
        self.explanation = explanation
        self.raw_response = raw_response
        self.frame_indices = frame_indices or []

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_anomaly": self.is_anomaly,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "raw_response": self.raw_response,
            "frame_indices": self.frame_indices,
        }
