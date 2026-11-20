# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AnyAnomaly: Zero-Shot Customizable Video Anomaly Detection with LVLM.

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

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from omni_mercury_engine.detectors.provenance import DecisionSource
from omni_mercury_engine.detectors.vlm.base_vlm import BaseVLMDetector, VLMConfig
from omni_mercury_engine.detectors.vlm.context_providers import (
    CombinedContextProvider,
    PositionContextProvider,
    TemporalContextProvider,
)
from omni_mercury_engine.detectors.vlm.lvlm_cache import get_frozen_cached_backend

logger = logging.getLogger(__name__)

#: Documented fallback probabilities when the LVLM states a verdict but no
#: parseable number (B3 audit cure): a bare YES must land above the default
#: 0.5 alarm threshold (the old 0.5 default meant a YES without a confidence
#: could never alarm), a bare NO below it, and a response with neither a
#: verdict nor a number is the neutral 0.5 that never alarms.
YES_DEFAULT_PROBABILITY = 0.6
NO_DEFAULT_PROBABILITY = 0.4
NO_EVIDENCE_PROBABILITY = 0.5

#: Anchored verdict patterns.  The prompt's ``ANOMALY_DETECTED:`` protocol
#: and the BLIP-VQA backend's actual ``ANOMALY: yes`` format
#: (``lvlm_backends.BLIPVQABackend.generate``) are accepted explicitly;
#: the first-50-characters fallback requires a word boundary, so ``EYES``
#: or ``YESTERDAY`` never reads as a YES, and the prompt echo
#: ``ANOMALY_DETECTED: [YES/NO]`` never parses as a verdict.
_VERDICT_PATTERNS = (
    re.compile(r"ANOMALY[_\s]?DETECTED\s*:\s*\[?\s*(YES|NO)\b", re.IGNORECASE),
    re.compile(r"\bANOMALY\s*:\s*\[?\s*(YES|NO)\b", re.IGNORECASE),
)
_VERDICT_WORD = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)

#: Literal template tokens from the prompt's RESPONSE FORMAT block.  A model
#: that echoes them ("ANOMALY_DETECTED: [YES/NO]", "CONFIDENCE: [0.0-1.0]")
#: has answered nothing, so they are removed before any parsing.
_TEMPLATE_ECHO = re.compile(r"\[\s*(?:YES\s*/\s*NO|0\.0\s*-\s*1\.0)\s*\]", re.IGNORECASE)

#: Probability patterns, most specific first.  ``ANOMALY_PROBABILITY`` is
#: the field the reworded prompt asks for; ``CONFIDENCE`` keeps backend and
#: legacy responses parseable (both now mean the probability THAT THE SCENE
#: IS ANOMALOUS, never "confidence in my verdict", which inverted NO-verdict
#: scores).
_PROBABILITY_PATTERNS = (
    re.compile(r"ANOMALY[_\s]?PROBABILITY\s*:\s*\[?\s*([\d.]+)", re.IGNORECASE),
    re.compile(r"CONFIDENCE\s*:\s*\[?\s*([\d.]+)", re.IGNORECASE),
    re.compile(r"([\d.]+)\s*%?\s*confident", re.IGNORECASE),
)


@dataclass
class AnyAnomalyConfig(VLMConfig):
    """Configuration for AnyAnomaly detector.

    Attributes:
        segment_overlap: Overlap between video segments
        use_segment_voting: Aggregate votes across segments
        action_focused: Emphasize action-based anomalies
        appearance_focused: Emphasize appearance-based anomalies
        context_window: Number of context frames
        enable_positional_context: Alias for the inherited
            ``use_position_context`` field on :class:`VLMConfig`.
            ``AnyAnomalyDetector.__init__`` propagates this onto
            ``use_position_context`` so either field name drives the
            same context-provider construction.
        enable_temporal_context: Alias for the inherited
            ``use_temporal_context``; same propagation contract.

    Note:
        The LVLM backend is selected via the inherited ``model_type``
        field on :class:`VLMConfig`, which ``AnyAnomalyDetector._initialize_model``
        forwards to :func:`omni_mercury_engine.detectors.vlm.lvlm_backends.get_lvlm_backend`.
        There is no separate ``backend`` field; a configuration like
        ``AnyAnomalyConfig(model_type=LVLMType.QWEN2_VL)`` is the
        supported pattern.  ``LVLMType`` is exported from
        :mod:`omni_mercury_engine.detectors.vlm.base_vlm`.
    """

    segment_overlap: int = 4
    use_segment_voting: bool = True
    action_focused: bool = True
    appearance_focused: bool = True
    context_window: int = 4  # Number of context frames
    # Alias fields: propagated onto the inherited ``use_position_context`` /
    # ``use_temporal_context`` in ``AnyAnomalyDetector.__init__`` so the new
    # field names are not inert.  Default to ``None`` so an explicit user
    # value overrides the inherited default; ``None`` means "do not override".
    enable_positional_context: bool | None = None
    enable_temporal_context: bool | None = None


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

    decision_source = (
        DecisionSource(
            kind="published",
            description=(
                "Segment-level zero-shot scoring follows the published "
                "AnyAnomaly method: an LVLM answers visual questions about "
                "the user's natural-language anomaly definition, with "
                "position and temporal context providers."
            ),
            citation=(
                'Ahn et al., "AnyAnomaly: Zero-Shot Customizable Video '
                'Anomaly Detection with LVLM", WACV 2026, arXiv:2503.04504'
            ),
        ),
        DecisionSource(
            kind="experimental",
            description=(
                "The binary flag cuts the LVLM confidence score at "
                "vlm_config.confidence_threshold (default 0.5): the paper "
                "reports ranking metrics rather than an alarm threshold, "
                "and no labelled corpus here calibrates one, so binary "
                "verdicts are experimental."
            ),
        ),
    )

    def __init__(self, config: AnyAnomalyConfig | dict[str, Any] | None = None) -> None:
        """Initialize AnyAnomaly detector.

        Args:
            config: Detector configuration or dict
        """
        if config is None:
            config = AnyAnomalyConfig()
        elif isinstance(config, dict):
            config = AnyAnomalyConfig(**config)

        # Propagate the AnyAnomaly-specific alias fields onto the
        # inherited VLMConfig fields they shadow.  Without this, a
        # caller that sets ``enable_temporal_context=False`` on
        # AnyAnomalyConfig would still get a temporal context
        # provider because the constructor below reads
        # ``config.use_temporal_context`` (the inherited field), which
        # defaults to True regardless of the alias value.  ``None``
        # means "no override", preserving the inherited default.
        if config.enable_positional_context is not None:
            config.use_position_context = config.enable_positional_context
        if config.enable_temporal_context is not None:
            config.use_temporal_context = config.enable_temporal_context

        super().__init__(config)
        self.any_config: AnyAnomalyConfig = config

        # Context providers
        self.context_provider = CombinedContextProvider(
            position_provider=PositionContextProvider() if config.use_position_context else None,
            temporal_provider=TemporalContextProvider() if config.use_temporal_context else None,
        )

        # LVLM backend
        self._backend: Any = None

        # Anomaly definition and reference frames for test compatibility
        self._anomaly_definition: str | None = None
        self._reference_frames: list[Any] = []

    #: ``context_provider`` is built deterministically in ``__init__`` from
    #: configuration; ``_backend`` is the lazily-loaded pretrained LVLM.
    #: ``_reference_frames`` (user-supplied detection context) is decomposed
    #: below so numpy frames survive the weights_only load.
    _UNSAVED_ATTRS = BaseVLMDetector._UNSAVED_ATTRS | {
        "context_provider",
        "_backend",
        "_reference_frames",
    }

    def _learned_state(self) -> dict[str, Any]:
        """Default state plus reference frames stored as tensors."""
        state = super()._learned_state()
        state["_reference_frames_payload"] = [
            frame if isinstance(frame, torch.Tensor) else torch.from_numpy(np.asarray(frame))
            for frame in self._reference_frames
        ]
        state["_reference_frames_were_tensors"] = [
            isinstance(frame, torch.Tensor) for frame in self._reference_frames
        ]
        return state

    def _restore_learned_state(self, state: dict[str, Any]) -> None:
        """Restore default state, then decode the reference frames."""
        state = dict(state)
        payload = state.pop("_reference_frames_payload", None)
        were_tensors = state.pop("_reference_frames_were_tensors", None)
        super()._restore_learned_state(state)
        if payload is not None:
            if were_tensors is None:
                were_tensors = [True] * len(payload)
            self._reference_frames = [
                frame if was_tensor else frame.cpu().numpy()
                for frame, was_tensor in zip(payload, were_tensors, strict=True)
            ]

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
        """Initialize the LVLM backend through the process-wide cache.

        Repeated detector instances reuse one loaded backend
        (:func:`get_frozen_cached_backend`); a cached instance whose frozen
        configuration (device, generation settings, revision) differs from
        this detector's is never reused — a dedicated backend is built
        instead, so no state leaks across instances.
        """
        self._backend = get_frozen_cached_backend(
            model_type=self.vlm_config.model_type.value,
            model_name=self.vlm_config.model_name,
            device=str(self.device),
            max_new_tokens=self.vlm_config.max_new_tokens,
            temperature=self.vlm_config.temperature,
            revision=self.vlm_config.revision,
        )

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
        # Add task instruction.  The number asked for is the probability
        # THAT THE SCENE IS ANOMALOUS — not "confidence in your answer",
        # whose NO-verdict readings inverted the score (B3 audit cure).
        prompt += """TASK:
Analyze the image(s) carefully and determine:
1. Is there any anomalous activity matching the definition above?
2. If yes, describe what you observe.
3. State the probability that the scene is anomalous, from 0.0 (certainly
   normal) to 1.0 (certainly anomalous).

RESPONSE FORMAT:
Answer with EXACTLY this format:
ANOMALY_DETECTED: [YES/NO]
ANOMALY_PROBABILITY: [0.0-1.0]
EXPLANATION: [Your detailed explanation]
"""
        return prompt

    @staticmethod
    def _parse_anomaly_probability(response: str) -> float | None:
        """Extract the stated probability that the scene is anomalous.

        Args:
            response: Model response text (template echoes already removed).

        Returns:
            The clamped probability, or ``None`` when no number parses.
        """
        for pattern in _PROBABILITY_PATTERNS:
            match = pattern.search(response)
            if not match:
                continue
            try:
                value = float(match.group(1))
            except ValueError:
                continue  # Invalid number, try the next pattern
            if value > 1:  # Percentage
                value /= 100
            return min(max(value, 0.0), 1.0)
        return None

    @staticmethod
    def _parse_verdict(response: str) -> bool | None:
        """Extract the YES/NO verdict with anchored or word-boundary matching.

        Args:
            response: Model response text (template echoes already removed).

        Returns:
            ``True`` for YES, ``False`` for NO, ``None`` when no verdict
            parses.
        """
        for pattern in _VERDICT_PATTERNS:
            match = pattern.search(response)
            if match:
                return match.group(1).upper() == "YES"
        # Fallback: a standalone YES/NO near the start of the answer.  The
        # word boundary keeps "EYES"/"YESTERDAY" from reading as YES (the
        # old substring check alarmed on both).
        match = _VERDICT_WORD.search(response[:50])
        if match:
            return match.group(1).upper() == "YES"
        return None

    def _parse_response(self, response: str) -> tuple[bool, float, str]:
        """Parse LVLM response to extract detection result.

        The returned ``confidence`` is the probability that the scene is
        anomalous (what the prompt asks for), so it is the frame score
        directly for either verdict.  Resolution order (B3 audit cure):

        1. A parsed number is that probability and decides the flag
           (``> confidence_threshold``), whatever the verdict says — this
           removes the "confidence in a NO" inversion class entirely.
        2. With no number, a parsed YES defaults to the documented
           :data:`YES_DEFAULT_PROBABILITY` (0.6, above the default
           threshold, so a bare YES alarms), a parsed NO to
           :data:`NO_DEFAULT_PROBABILITY` (0.4).
        3. With neither, the neutral :data:`NO_EVIDENCE_PROBABILITY` (0.5)
           — never an alarm.

        Args:
            response: Model response text

        Returns:
            Tuple of (is_anomaly, confidence, explanation)
        """
        cleaned = _TEMPLATE_ECHO.sub(" ", response)

        probability = self._parse_anomaly_probability(cleaned)
        verdict = self._parse_verdict(cleaned)

        if probability is not None:
            confidence = probability
            is_anomaly = probability > self.vlm_config.confidence_threshold
        elif verdict is not None:
            confidence = YES_DEFAULT_PROBABILITY if verdict else NO_DEFAULT_PROBABILITY
            is_anomaly = verdict
        else:
            confidence = NO_EVIDENCE_PROBABILITY
            is_anomaly = False

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
        data = self._validate_frames(data)
        t = data.shape[0]

        # Segment video
        segments = self._segment_video(data)

        # Process each segment
        segment_results = []
        failed_segments: list[tuple[int, int, str]] = []
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

            # Query LVLM. A backend failure contributes NO score: the
            # historical fallback (is_anomaly=False, confidence=0.0) fed
            # score = 1 - confidence = 1.0 into every covered frame, so a
            # backend outage silently marked the whole clip anomalous.
            try:
                response = self.backend.generate(images, prompt)
                is_anomaly, confidence, explanation = self._parse_response(response)
            except Exception as e:
                logger.warning(f"LVLM query failed: {e}")
                failed_segments.append((start, end, str(e)))
                segment_results.append(
                    {
                        "start": start,
                        "end": end,
                        "is_anomaly": False,
                        "confidence": 0.0,
                        "explanation": f"LVLM backend failed: {e}",
                        "backend_failed": True,
                    }
                )
                continue

            # Store segment result
            segment_results.append(
                {
                    "start": start,
                    "end": end,
                    "is_anomaly": is_anomaly,
                    "confidence": confidence,
                    "explanation": explanation,
                    "backend_failed": False,
                }
            )

            # Update frame scores.  ``confidence`` is already the model's
            # probability that the scene is anomalous (see _parse_response),
            # so it is the score for either verdict — the old
            # ``1 - confidence`` reading for NO inverted responses such as
            # "NO, CONFIDENCE: 0.1" into a 0.9 anomaly score.
            score = confidence
            for i in range(start, end):
                frame_scores[i] += score
                frame_counts[i] += 1

        # Fail loud when backend failures leave frames with no real signal
        # (overlapping successful segments may still cover every frame).
        if failed_segments:
            uncovered = int(np.count_nonzero(frame_counts == 0))
            if uncovered:
                raise RuntimeError(
                    f"AnyAnomaly LVLM backend failed on {len(failed_segments)} of "
                    f"{len(segment_results)} segments, leaving {uncovered} frame(s) "
                    f"with no real signal (first error: {failed_segments[0][2]}). "
                    "Refusing to fabricate scores for uncovered frames."
                )

        # Average scores where multiple segments overlap
        frame_counts = np.maximum(frame_counts, 1)
        scores = frame_scores / frame_counts

        # Determine anomalies (per-frame mask; distinct from the per-segment
        # ``is_anomaly`` bool reused in the segment loop above).
        anomaly_mask = np.asarray(scores > self.vlm_config.confidence_threshold, dtype=bool)

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
            "is_anomaly": anomaly_mask,
            "explanations": explanations,
            "features": features,
            "segment_results": segment_results,
            "anomaly_frames": np.where(anomaly_mask)[0].tolist(),
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
        features = np.array(features)  # type: ignore[assignment, unused-ignore]
        if len(features) < 128:
            features = np.pad(features, (0, 128 - len(features)))  # type: ignore[assignment, unused-ignore]

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
