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
Context providers for VLM-based anomaly detection.

Implements position and temporal context extraction for
context-aware visual question answering.

Key innovations from AnyAnomaly (WACV 2026):
    - Position Context: Enhances object localization analysis
    - Temporal Context: Improves action understanding across frames
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


logger = logging.getLogger(__name__)


@dataclass
class ContextInfo:
    """Container for extracted context information."""

    context_type: str
    description: str
    features: np.ndarray[Any, Any] | None = None
    metadata: dict[str, Any] | None = None


class BaseContextProvider(ABC):
    """Abstract base class for context providers."""

    @abstractmethod
    def extract_context(
        self,
        frames: np.ndarray[Any, Any] | torch.Tensor,
        **kwargs: Any,
    ) -> ContextInfo:
        """Extract context from input frames.

        Args:
            frames: Input frames [T, C, H, W] or [C, H, W]
            **kwargs: Additional arguments

        Returns:
            Extracted context information
        """
        pass

    @abstractmethod
    def format_context_prompt(self, context: ContextInfo) -> str:
        """Format context as text prompt addition.

        Args:
            context: Extracted context

        Returns:
            Text description to add to prompt
        """
        pass


class PositionContextProvider(BaseContextProvider):
    """Position context provider for spatial awareness.

    Extracts spatial information about objects and regions
    to enhance object-level analysis.
    """

    def __init__(
        self,
        grid_size: tuple[int, int] = (3, 3),
        use_saliency: bool = True,
    ):
        """Initialize position context provider.

        Args:
            grid_size: Spatial grid for region descriptions
            use_saliency: Whether to use saliency detection
        """
        self.grid_size = grid_size
        self.use_saliency = use_saliency

        # Region labels for grid
        self.region_labels = self._create_region_labels()

    def _create_region_labels(self) -> dict[tuple[int, int], str]:
        """Create human-readable labels for grid regions."""
        h, w = self.grid_size

        # Position descriptors
        v_labels = {0: "top", h // 2: "center", h - 1: "bottom"}
        h_labels = {0: "left", w // 2: "center", w - 1: "right"}

        labels = {}
        for i in range(h):
            for j in range(w):
                v = v_labels.get(i, "upper" if i < h // 2 else "lower")
                h_pos = h_labels.get(j, "left" if j < w // 2 else "right")

                if v == "center" and h_pos == "center":
                    labels[(i, j)] = "center"
                elif v == "center":
                    labels[(i, j)] = h_pos
                elif h_pos == "center":
                    labels[(i, j)] = v
                else:
                    labels[(i, j)] = f"{v}-{h_pos}"

        return labels

    def extract_context(
        self,
        frames: np.ndarray[Any, Any] | torch.Tensor,
        **kwargs: Any,
    ) -> ContextInfo:
        """Extract spatial position context.

        Args:
            frames: Input frames

        Returns:
            Position context information
        """
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()

        # Handle single image or video
        if frames.ndim == 3:  # [C, H, W]
            frames = frames[np.newaxis, ...]  # Add time dim

        _t, _c, _h, _w = frames.shape

        # Compute activity/saliency per region
        region_activity = self._compute_region_activity(frames)

        # Find most active regions
        active_regions = self._find_active_regions(region_activity)

        # Build description
        description = self._build_position_description(active_regions)

        return ContextInfo(
            context_type="position",
            description=description,
            features=region_activity,
            metadata={
                "active_regions": active_regions,
                "grid_size": self.grid_size,
            },
        )

    def _compute_region_activity(self, frames: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute activity level for each grid region.

        Uses frame difference and gradient magnitude as activity proxy.
        """
        t, _c, h, w = frames.shape
        gh, gw = self.grid_size

        region_h = h // gh
        region_w = w // gw

        activity = np.zeros((gh, gw))

        for i in range(gh):
            for j in range(gw):
                y1, y2 = i * region_h, (i + 1) * region_h
                x1, x2 = j * region_w, (j + 1) * region_w

                region = frames[:, :, y1:y2, x1:x2]

                # Temporal difference (if multiple frames)
                temp_diff = np.abs(np.diff(region, axis=0)).mean() if t > 1 else 0

                # Spatial gradient (edge/texture)
                gray = region.mean(axis=1)  # [T, H, W]
                grad_y = np.abs(np.diff(gray, axis=1)).mean()
                grad_x = np.abs(np.diff(gray, axis=2)).mean()

                activity[i, j] = temp_diff + 0.5 * (grad_y + grad_x)

        # Normalize
        if activity.max() > 0:
            activity = activity / activity.max()

        return activity

    def _find_active_regions(
        self,
        activity: np.ndarray[Any, Any],
        threshold: float = 0.3,
    ) -> list[tuple[tuple[int, int], float]]:
        """Find regions with high activity."""
        active = []
        for i in range(activity.shape[0]):
            for j in range(activity.shape[1]):
                if activity[i, j] > threshold:
                    active.append(((i, j), float(activity[i, j])))

        # Sort by activity level
        active.sort(key=lambda x: x[1], reverse=True)
        return active[:5]  # Top 5 regions

    def _build_position_description(
        self,
        active_regions: list[tuple[tuple[int, int], float]],
    ) -> str:
        """Build natural language position description."""
        if not active_regions:
            return "Activity is distributed evenly across the image."

        descriptions = []
        for (i, j), score in active_regions:
            region_name = self.region_labels.get((i, j), f"region ({i},{j})")
            descriptions.append(f"the {region_name} area (activity: {score:.1%})")

        if len(descriptions) == 1:
            return f"Main activity is in {descriptions[0]}."

        return f"Activity detected in: {', '.join(descriptions[:-1])}, and {descriptions[-1]}."

    def format_context_prompt(self, context: ContextInfo) -> str:
        """Format position context as prompt addition."""
        return f"\n[Spatial Context: {context.description}]"


class TemporalContextProvider(BaseContextProvider):
    """Temporal context provider for action understanding.

    Extracts temporal dynamics across video frames to improve
    action and motion-based anomaly detection.
    """

    def __init__(
        self,
        window_size: int = 8,
        motion_threshold: float = 0.1,
    ):
        """Initialize temporal context provider.

        Args:
            window_size: Number of frames to analyze together
            motion_threshold: Threshold for significant motion
        """
        self.window_size = window_size
        self.motion_threshold = motion_threshold

    def extract_context(
        self,
        frames: np.ndarray[Any, Any] | torch.Tensor,
        **kwargs: Any,
    ) -> ContextInfo:
        """Extract temporal context from video frames.

        Args:
            frames: Video frames [T, C, H, W]

        Returns:
            Temporal context information
        """
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()

        if frames.ndim == 3:  # Single image
            return ContextInfo(
                context_type="temporal",
                description="Single frame - no temporal context available.",
                features=None,
                metadata={"num_frames": 1},
            )

        t, _c, _h, _w = frames.shape

        # Compute motion features
        motion_magnitude = self._compute_motion(frames)
        motion_pattern = self._analyze_motion_pattern(motion_magnitude)

        # Detect significant events
        events = self._detect_events(motion_magnitude)

        # Build description
        description = self._build_temporal_description(motion_pattern, events, t)

        return ContextInfo(
            context_type="temporal",
            description=description,
            features=motion_magnitude,
            metadata={
                "num_frames": t,
                "motion_pattern": motion_pattern,
                "events": events,
            },
        )

    def _compute_motion(self, frames: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute frame-to-frame motion magnitude."""
        # Frame differences
        diffs = np.abs(np.diff(frames.astype(float), axis=0))

        # Motion magnitude per frame
        motion = diffs.mean(axis=(1, 2, 3))  # [T-1]

        return motion

    def _analyze_motion_pattern(self, motion: np.ndarray[Any, Any]) -> str:
        """Analyze overall motion pattern."""
        if len(motion) == 0:
            return "static"

        mean_motion = motion.mean()
        std_motion = motion.std()

        if mean_motion < self.motion_threshold * 0.5:
            return "static"
        elif mean_motion < self.motion_threshold:
            return "low_motion"
        elif std_motion / max(mean_motion, 1e-6) > 0.5:
            return "variable_motion"
        else:
            return "consistent_motion"

    def _detect_events(
        self,
        motion: np.ndarray[Any, Any],
        threshold_factor: float = 2.0,
    ) -> list[dict[str, Any]]:
        """Detect significant motion events."""
        events = []

        if len(motion) < 2:
            return events

        mean_motion = motion.mean()
        threshold = max(mean_motion * threshold_factor, self.motion_threshold)

        # Find peaks
        for i in range(1, len(motion) - 1):
            if motion[i] > threshold:
                if motion[i] > motion[i - 1] and motion[i] > motion[i + 1]:
                    events.append(
                        {
                            "frame": i,
                            "magnitude": float(motion[i]),
                            "type": "sudden_motion",
                        }
                    )

        # Detect start/stop of motion
        for i in range(len(motion) - 1):
            diff = motion[i + 1] - motion[i]
            if abs(diff) > threshold:
                event_type = "motion_start" if diff > 0 else "motion_stop"
                events.append(
                    {
                        "frame": i,
                        "magnitude": float(abs(diff)),
                        "type": event_type,
                    }
                )

        return events[:5]  # Limit to top 5 events

    def _build_temporal_description(
        self,
        pattern: str,
        events: list[dict[str, Any]],
        num_frames: int,
    ) -> str:
        """Build natural language temporal description."""
        pattern_descriptions = {
            "static": "The scene is mostly static with minimal movement.",
            "low_motion": "There is subtle movement in the scene.",
            "variable_motion": "Motion varies significantly across the sequence.",
            "consistent_motion": "There is steady, continuous motion throughout.",
        }

        base = pattern_descriptions.get(pattern, "Motion pattern is unclear.")

        if not events:
            return base

        event_strs = []
        for event in events[:3]:
            frame = event["frame"]
            etype = event["type"].replace("_", " ")
            event_strs.append(f"{etype} at frame {frame}")

        return f"{base} Notable events: {'; '.join(event_strs)}."

    def format_context_prompt(self, context: ContextInfo) -> str:
        """Format temporal context as prompt addition."""
        return f"\n[Temporal Context: {context.description}]"


class CombinedContextProvider:
    """Combines multiple context providers for rich context."""

    def __init__(
        self,
        position_provider: PositionContextProvider | None = None,
        temporal_provider: TemporalContextProvider | None = None,
    ):
        """Initialize combined provider.

        Args:
            position_provider: Optional position context provider
            temporal_provider: Optional temporal context provider
        """
        self.position_provider = position_provider or PositionContextProvider()
        self.temporal_provider = temporal_provider or TemporalContextProvider()

    def extract_all_context(
        self,
        frames: np.ndarray[Any, Any] | torch.Tensor,
    ) -> dict[str, ContextInfo]:
        """Extract all context types.

        Args:
            frames: Input frames

        Returns:
            Dict mapping context type to context info
        """
        contexts = {}

        contexts["position"] = self.position_provider.extract_context(frames)
        contexts["temporal"] = self.temporal_provider.extract_context(frames)

        return contexts

    def format_combined_prompt(
        self,
        contexts: dict[str, ContextInfo],
    ) -> str:
        """Format all contexts as prompt addition.

        Args:
            contexts: Dict of context info

        Returns:
            Combined context string for prompt
        """
        parts = []

        if "position" in contexts:
            parts.append(self.position_provider.format_context_prompt(contexts["position"]))

        if "temporal" in contexts:
            parts.append(self.temporal_provider.format_context_prompt(contexts["temporal"]))

        return "\n".join(parts)


# Aliases for test compatibility
class PositionalContextExtractor(PositionContextProvider):
    """Alias for PositionContextProvider for test compatibility."""

    def extract(self, image: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Extract positional context from image.

        Args:
            image: Input image [C, H, W]

        Returns:
            Context dictionary
        """
        context = self.extract_context(image)
        return {
            "type": context.context_type,
            "description": context.description,
            "features": context.features,
            "metadata": context.metadata,
        }


class TemporalContextExtractor(TemporalContextProvider):
    """Alias for TemporalContextProvider for test compatibility."""

    def extract(self, frames: list[Any] | np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Extract temporal context from frames.

        Args:
            frames: List of frames or video tensor

        Returns:
            Context dictionary
        """
        if isinstance(frames, list):
            # Convert list of tensors to numpy array
            if len(frames) > 0:
                if isinstance(frames[0], torch.Tensor):
                    frames = torch.stack(frames).cpu().numpy()
                else:
                    frames = np.stack(frames)
            else:
                frames = np.array([])

        context = self.extract_context(frames)
        return {
            "type": context.context_type,
            "description": context.description,
            "features": context.features,
            "metadata": context.metadata,
        }
