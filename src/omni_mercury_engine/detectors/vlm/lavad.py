# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""LAVAD: Harnessing Large Language Models for Training-free Video Anomaly Detection.

Implementation inspired by LAVAD (CVPR 2024).
Uses LLMs for temporal aggregation and anomaly scoring without training.

Key Features:
    1. Training-free detection using pre-trained LLMs
    2. VLM captioning + LLM reasoning pipeline
    3. Temporal aggregation across frames
    4. Works on real-world surveillance scenarios

Reference:
    Zanella et al. "Harnessing Large Language Models for Training-free
    Video Anomaly Detection"
    https://lucazanella.github.io/lavad/
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
from omni_mercury_engine.detectors.vlm.lvlm_cache import get_frozen_cached_backend
from omni_mercury_engine.detectors.vlm.text_match import content_words, find_word_matches

logger = logging.getLogger(__name__)

#: Released default values of the canonical model fields and their alias
#: fields.  ``__post_init__`` compares against these to tell "left at the
#: default" from "explicitly set" when reconciling each alias pair.
_CAPTION_MODEL_DEFAULT = "Salesforce/blip-image-captioning-base"
_REASONING_MODEL_DEFAULT = "Qwen/Qwen2.5-0.5B-Instruct"
_LLM_MODEL_DEFAULT = "meta-llama/Meta-Llama-3-8B-Instruct"
_VLM_MODEL_DEFAULT = "Salesforce/blip-image-captioning-base"


@dataclass
class LAVADConfig(VLMConfig):
    """Configuration for LAVAD detector.

    Attributes:
        caption_model: Model for frame captioning
        reasoning_backend: Backend serving the text reasoning stage
            (``"hf_text"``: a local instruction-tuned language model)
        reasoning_model: Model for temporal reasoning (LLM).  The shipped
            default is the pinned, CPU-runnable Qwen2.5-0.5B-Instruct; the
            previous default (Llama 3 8B) was never loaded -- the reasoning
            prompt went to the captioner, which returns nothing for text.
        reasoning_revision: Immutable revision for a non-default
            ``reasoning_model`` (the default is pinned by its backend)
        reasoning_mode: ``"frame_likelihood"`` (default): the language model
            judges each frame's caption against the anomaly description and
            the frame's score is its own probability of answering Yes.
            ``"window_generate"``: LAVAD-style temporal reasoning over a
            sliding window of captions with a generated, parsed verdict --
            for capable reasoners (measured: the shipped 0.5B model answers
            YES on all-normal clips this way, and conflates frames when
            neighbouring captions are in the prompt).
        window_size: Number of captions for temporal context
        use_llm_reasoning: Use LLM for anomaly reasoning
        llm_model: Released alias for ``reasoning_model``, wired in
            ``__post_init__``: setting either field takes effect, the
            canonical ``reasoning_model`` wins when both are set, and after
            construction the alias always equals the canonical field.
        vlm_model: Released alias for ``caption_model``; same wiring
            contract as ``llm_model``.
        sampling_fps: Caption sampling rate in frames per second.  Used when
            the caller passes the clip's real frame rate to
            ``detect(data, source_fps=...)``: frames are then captioned at
            this rate (stride ``round(source_fps / sampling_fps)``) and
            uncaptioned frames inherit the nearest captioned frame's caption
            and score.  Without ``source_fps`` every frame is captioned.
        temporal_window: Alias for the ``window_size`` field above.
            ``LAVADDetector.__init__`` propagates an explicit value onto
            ``window_size`` so the field is not inert; ``None`` (the
            default) means "do not override".
    """

    # Shipped default: the SHA-pinned, CPU-runnable BLIP captioner
    # (BLIPCaptionBackend) so the caption stage is genuinely operational
    # out of the box; any registered backend/model can be configured.
    caption_backend: str = "blip_caption"
    caption_model: str = _CAPTION_MODEL_DEFAULT
    reasoning_backend: str = "hf_text"
    reasoning_model: str = _REASONING_MODEL_DEFAULT
    reasoning_revision: str | None = None
    reasoning_mode: str = "frame_likelihood"
    window_size: int = 5
    use_llm_reasoning: bool = True
    # Released alias fields (wired in __post_init__; they were declared but
    # never read before the B4 audit cure). The declared defaults are the
    # released ones; after construction each alias mirrors its canonical
    # field, so the pair can never disagree.
    llm_model: str = _LLM_MODEL_DEFAULT  # Alias for reasoning_model
    vlm_model: str = _VLM_MODEL_DEFAULT  # Alias for caption_model
    sampling_fps: float = 2.0  # Caption sampling rate (see docstring)
    temporal_window: int | None = None  # Alias for window_size (see docstring)

    def __post_init__(self) -> None:
        """Wire the released alias fields onto their canonical fields.

        Setting either name has effect: an alias changed from its released
        default propagates onto a canonical field left at its default; when
        both are set, the canonical field wins; and after construction each
        alias equals its canonical field, so reads through either name see
        the model that will actually load.
        """
        if self.reasoning_model == _REASONING_MODEL_DEFAULT and (
            self.llm_model != _LLM_MODEL_DEFAULT
        ):
            self.reasoning_model = self.llm_model
        self.llm_model = self.reasoning_model

        if self.caption_model == _CAPTION_MODEL_DEFAULT and self.vlm_model != _VLM_MODEL_DEFAULT:
            self.caption_model = self.vlm_model
        self.vlm_model = self.caption_model


class LAVADDetector(BaseVLMDetector):
    """LAVAD training-free video anomaly detector.

    Uses a two-stage pipeline:
    1. VLM generates captions for each frame
    2. LLM reasons about anomalies from caption sequence

    Achieves SOTA on UCF-Crime and XD-Violence without any training.

    Example:
        >>> detector = LAVADDetector()
        >>> detector.set_anomaly_description("violent activity")
        >>> results = detector.detect(video_frames)
    """

    decision_source = (
        DecisionSource(
            kind="published",
            description=(
                "Training-free window scoring follows the published LAVAD "
                "pipeline: a VLM captions each frame and an LLM reasons "
                "over the caption sequence to score temporal windows."
            ),
            citation=(
                'Zanella et al., "Harnessing Large Language Models for '
                'Training-free Video Anomaly Detection", CVPR 2024, '
                "https://lucazanella.github.io/lavad/"
            ),
        ),
        DecisionSource(
            kind="experimental",
            description=(
                "The binary flag cuts the LLM-derived window score at "
                "vlm_config.confidence_threshold (default 0.5): the paper "
                "reports ranking metrics rather than an alarm threshold, "
                "and no labelled corpus here calibrates one, so binary "
                "verdicts are experimental."
            ),
        ),
    )

    def __init__(self, config: LAVADConfig | dict[str, Any] | None = None) -> None:
        """Initialize LAVAD detector.

        Args:
            config: Detector configuration
        """
        if config is None:
            config = LAVADConfig()
        elif isinstance(config, dict):
            config = LAVADConfig(**config)

        # ``temporal_window`` is an alias for ``window_size`` (it was a dead
        # field): an explicit value overrides, ``None`` keeps the default.
        if config.temporal_window is not None:
            config.window_size = config.temporal_window

        super().__init__(config)
        self.lavad_config: LAVADConfig = config

        self._caption_model: Any = None
        self._reasoning_model: Any = None

        # Scene context for test compatibility
        self._scene_context: dict[str, Any] | None = None

    #: The captioning and reasoning models are pretrained checkpoints loaded
    #: lazily by name from the config, never weights learned here.
    _UNSAVED_ATTRS = BaseVLMDetector._UNSAVED_ATTRS | {
        "_caption_model",
        "_reasoning_model",
    }

    @property
    def scene_context(self) -> dict[str, Any] | None:
        """Get the current scene context."""
        return self._scene_context

    def set_scene_context(
        self,
        scene_description: str,
        expected_activities: list[str] | None = None,
        anomaly_types: list[str] | None = None,
    ) -> None:
        """Set scene context for detection.

        Args:
            scene_description: Description of the scene
            expected_activities: List of expected normal activities
            anomaly_types: List of anomaly types to detect
        """
        self._scene_context = {
            "scene_description": scene_description,
            "expected_activities": expected_activities or [],
            "anomaly_types": anomaly_types or [],
        }

    def detect_video(
        self,
        video: np.ndarray[Any, Any] | torch.Tensor,
        source_fps: float | None = None,
    ) -> dict[str, Any]:
        """Detect anomalies in video.

        Args:
            video: Video tensor [T, C, H, W]
            source_fps: The clip's real frame rate; enables caption
                subsampling at ``config.sampling_fps`` (see :meth:`detect`).

        Returns:
            Detection results with frame_scores
        """
        result = self.detect(video, source_fps=source_fps)
        # Add frame_scores alias for test compatibility
        result["frame_scores"] = result["scores"]
        return result

    def _initialize_model(self) -> None:
        """Initialize captioning and reasoning models.

        The captioning backend comes through the process-wide cache
        (:func:`get_frozen_cached_backend`), so repeated detector instances
        reuse one loaded model; a cached instance with a different frozen
        configuration is never reused.  The default is the shipped
        SHA-pinned BLIP captioner; a missing transformers install or an
        unpinned/unlisted model still fails loud through the backend and
        SafeHFLoader — never a silent mock.
        """
        self._caption_model = get_frozen_cached_backend(
            model_type=self.lavad_config.caption_backend,
            model_name=self.lavad_config.caption_model,
            device=str(self.device),
        )

        logger.info("LAVAD models initialized")

    @property
    def reasoner(self) -> Any:
        """The text language model that reasons over the frame captions."""
        if self._reasoning_model is None:
            self._reasoning_model = get_frozen_cached_backend(
                model_type=self.lavad_config.reasoning_backend,
                model_name=self.lavad_config.reasoning_model,
                device=str(self.device),
                revision=self.lavad_config.reasoning_revision,
            )
        return self._reasoning_model

    @property
    def caption_model(self) -> Any:
        """Get captioning model."""
        if self._caption_model is None:
            self._initialize_model()
        return self._caption_model

    def _create_prompt(
        self,
        anomaly_description: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Create caption prompt for frame."""
        return (
            "Describe this surveillance camera image in detail. "
            "Focus on: people present, their actions, objects, "
            "and any unusual or concerning activities."
        )

    def _create_reasoning_prompt(
        self,
        captions: list[str],
        anomaly_description: str,
    ) -> str:
        """Create LLM reasoning prompt from captions.

        Args:
            captions: Sequence of frame captions
            anomaly_description: Target anomaly description

        Returns:
            Reasoning prompt for LLM
        """
        caption_text = "\n".join(f"Frame {i + 1}: {cap}" for i, cap in enumerate(captions))

        prompt = f"""You are analyzing a sequence of surveillance video frame descriptions to detect anomalies.

FRAME DESCRIPTIONS:
{caption_text}

ANOMALY TO DETECT:
{anomaly_description}

TASK:
Based on the sequence of frame descriptions above:
1. Analyze the progression of events across frames
2. Determine if the described anomaly is occurring
3. Identify which frames (if any) show anomalous activity
4. Rate your confidence from 0.0 to 1.0

RESPONSE FORMAT:
ANOMALY_DETECTED: [YES/NO]
ANOMALY_FRAMES: [comma-separated frame numbers, or "none"]
CONFIDENCE: [0.0-1.0]
EXPLANATION: [Your reasoning based on the frame descriptions]
"""
        return prompt

    @staticmethod
    def _frame_question(caption: str, anomaly_description: str) -> str:
        """The yes/no question the language model answers for one frame."""
        return (
            f"A video frame is described as: {caption.strip().rstrip('.')}.\n"
            f"Does this frame show {anomaly_description.strip().rstrip('?.')}? "
            "Answer Yes or No."
        )

    def _parse_response(self, response: str) -> tuple[bool, float, str]:
        """Parse LLM response."""
        response_upper = response.upper()

        # Anomaly detection
        is_anomaly = "ANOMALY_DETECTED: YES" in response_upper

        # Confidence
        confidence = 0.5
        match = re.search(r"CONFIDENCE:\s*([\d.]+)", response, re.IGNORECASE)
        if match:
            try:
                confidence = float(match.group(1))
                confidence = min(max(confidence, 0.0), 1.0)
            except ValueError:
                pass  # Invalid confidence value, use default 0.5

        # Explanation
        explanation = response
        match = re.search(r"EXPLANATION:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
        if match:
            explanation = match.group(1).strip()

        return is_anomaly, confidence, explanation

    def _parse_anomaly_frames(self, response: str) -> list[int]:
        """Parse which frames are anomalous from response."""
        match = re.search(r"ANOMALY_FRAMES:\s*(.+?)(?=\n|$)", response, re.IGNORECASE)
        if not match:
            return []

        frames_str = match.group(1).strip()
        if frames_str.lower() == "none":
            return []

        frames = []
        for part in frames_str.replace(" ", "").split(","):
            try:
                frames.append(int(part) - 1)  # Convert to 0-indexed
            except ValueError:
                pass  # Skip non-integer frame references

        return frames

    def _generate_caption(self, frame: np.ndarray[Any, Any]) -> str:
        """Generate caption for a single frame.

        Args:
            frame: Frame array [C, H, W]

        Returns:
            Caption string
        """
        from PIL import Image

        # Convert to PIL
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

        pil_image = Image.fromarray(frame)

        # Generate caption
        prompt = self._create_prompt(self.vlm_config.anomaly_description)
        caption = self.caption_model.generate([pil_image], prompt)

        return str(caption)

    def _caption_stride(self, source_fps: float | None) -> int:
        """Caption-sampling stride derived from ``sampling_fps``.

        Args:
            source_fps: The clip's real frame rate, or ``None`` when unknown.

        Returns:
            How many source frames one captioned frame stands for (1 when
            the frame rate is unknown or already at/below ``sampling_fps``).

        Raises:
            ValueError: If ``source_fps`` or ``config.sampling_fps`` is not
                positive.
        """
        if source_fps is None:
            return 1
        sampling_fps = self.lavad_config.sampling_fps
        if source_fps <= 0 or sampling_fps <= 0:
            raise ValueError(
                f"source_fps ({source_fps}) and sampling_fps ({sampling_fps}) must be positive"
            )
        return max(1, round(source_fps / sampling_fps))

    def detect(
        self,
        data: np.ndarray[Any, Any] | torch.Tensor,
        source_fps: float | None = None,
    ) -> dict[str, Any]:
        """Detect anomalies using caption + reasoning pipeline.

        Args:
            data: Video frames [T, C, H, W]
            source_fps: The clip's real frame rate.  When given, frames are
                captioned at ``config.sampling_fps`` instead of one caption
                per frame (the audited efficiency fix: the field was dead
                and every frame was captioned); uncaptioned frames inherit
                the nearest captioned frame's caption and score.  ``None``
                (default) captions every frame.

        Returns:
            Detection results dict; ``caption_frame_indices`` lists the
            frames that were actually captioned.
        """
        data = self._validate_frames(data)
        t = data.shape[0]

        # Stage 1: Generate captions at the configured sampling rate.
        stride = self._caption_stride(source_fps)
        sampled_indices = list(range(0, t, stride))
        logger.info(f"Generating captions for {len(sampled_indices)} of {t} frames...")
        caption_by_sample = {i: self._generate_caption(data[i]) for i in sampled_indices}
        nearest_sample = [min(sampled_indices, key=lambda s: abs(s - i)) for i in range(t)]
        captions = [caption_by_sample[nearest_sample[i]] for i in range(t)]

        # Stage 2: language-model judgement of the captions
        frame_scores = np.zeros(t)
        frame_counts = np.zeros(t)
        all_explanations: list[dict[str, Any]] = []

        window_size = min(self.lavad_config.window_size, t)
        mode = self.lavad_config.reasoning_mode
        if self.lavad_config.use_llm_reasoning and mode not in (
            "frame_likelihood",
            "window_generate",
        ):
            raise ValueError(
                f"unknown reasoning_mode {mode!r}; expected 'frame_likelihood' or 'window_generate'"
            )
        frame_likelihood = self.lavad_config.use_llm_reasoning and mode == "frame_likelihood"
        if frame_likelihood:
            description = self.vlm_config.anomaly_description
            # One judgement per captioned frame; frames sharing a caption
            # share its score.
            p_by_sample = {
                s: float(
                    self.reasoner.yes_probability(
                        self._frame_question(caption_by_sample[s], description)
                    )
                )
                for s in sampled_indices
            }
            for i in range(t):
                p_yes = p_by_sample[nearest_sample[i]]
                frame_scores[i], frame_counts[i] = p_yes, 1.0
                if p_yes > self.vlm_config.confidence_threshold:
                    all_explanations.append(
                        {
                            "frame": i,
                            "caption": captions[i],
                            "confidence": p_yes,
                            "window": (i, i + 1),
                        }
                    )

        for start in range(0, t - window_size + 1) if not frame_likelihood else ():
            end = start + window_size
            window_captions = captions[start:end]

            # Create reasoning prompt
            if self.lavad_config.use_llm_reasoning:
                reasoning_prompt = self._create_reasoning_prompt(
                    window_captions,
                    self.vlm_config.anomaly_description,
                )

                # Query the language model.  A failure or an answer without a
                # verdict raises: scoring the window as normal instead (as
                # this did) would report "no anomaly" for frames nobody
                # reasoned about.
                response = self.reasoner.generate([], reasoning_prompt)
                if "ANOMALY_DETECTED:" not in response.upper():
                    raise RuntimeError(
                        "LAVAD reasoning model returned no ANOMALY_DETECTED verdict for "
                        f"frames {start}..{end - 1}: {response[:200]!r}"
                    )
                is_anomaly, confidence, explanation = self._parse_response(response)
                anomaly_frames = self._parse_anomaly_frames(response)
            else:
                # Simple heuristic based on caption keywords
                is_anomaly, confidence, explanation = self._simple_analysis(window_captions)
                anomaly_frames = list(range(window_size)) if is_anomaly else []

            # Update frame scores.  A YES verdict without a parseable frame
            # list localises nothing, so the whole window carries the
            # verdict's score: the old ``base_score * 0.5`` cap meant such a
            # YES at confidence <= 1.0 could never exceed the strict 0.5
            # threshold and the alarm was silently lost.
            base_score = confidence if is_anomaly else 0.0
            no_localization = is_anomaly and not anomaly_frames
            for frame_idx in range(start, end):
                local_idx = frame_idx - start
                if no_localization or local_idx in anomaly_frames:
                    frame_scores[frame_idx] += confidence
                else:
                    frame_scores[frame_idx] += base_score * 0.5
                frame_counts[frame_idx] += 1

            if is_anomaly:
                window_explanation = {
                    "window": (start, end),
                    "explanation": explanation,
                    "confidence": confidence,
                }
                if no_localization:
                    window_explanation["no_localization"] = True
                all_explanations.append(window_explanation)

        # Average overlapping window scores.  No normalisation by the clip's
        # maximum: dividing by it (as before) put the top frame of every clip
        # at 1.0, so even an all-normal clip always flagged a frame.
        frame_counts = np.maximum(frame_counts, 1)
        scores = frame_scores / frame_counts

        # Threshold for anomaly detection (per-frame mask; distinct from the
        # per-window ``is_anomaly`` bool reused in the windowing loop above).
        anomaly_mask = np.asarray(scores > self.vlm_config.confidence_threshold, dtype=bool)

        # Generate features
        features = self._generate_features(scores, captions)

        return {
            "scores": scores,
            "is_anomaly": anomaly_mask,
            "captions": captions,
            "caption_frame_indices": sampled_indices,
            "explanations": all_explanations,
            "features": features,
            "anomaly_frames": np.where(anomaly_mask)[0].tolist(),
        }

    def _simple_analysis(
        self,
        captions: list[str],
    ) -> tuple[bool, float, str]:
        """Simple keyword-based analysis without LLM.

        Args:
            captions: Frame captions

        Matching is whole-word and case-insensitive (``"hit"`` no longer
        matches inside ``"white"`` nor ``"run"`` inside ``"brunette"``), and
        the anomaly description contributes only stop-word-free content
        words, so its ``"or"`` never matches ``"door"``/``"floor"``.

        Returns:
            Tuple of (is_anomaly, confidence, explanation)
        """
        # Anomaly keywords
        anomaly_keywords = [
            "fight",
            "violence",
            "attack",
            "hit",
            "punch",
            "kick",
            "fall",
            "falling",
            "fell",
            "collapse",
            "run",
            "running",
            "chase",
            "flee",
            "weapon",
            "gun",
            "knife",
            "blood",
            "fire",
            "smoke",
            "explosion",
            "accident",
            "crash",
            "collision",
            "suspicious",
            "unusual",
            "strange",
            "abnormal",
        ]

        # Check target anomaly description (content words only)
        target_words = content_words(self.vlm_config.anomaly_description, exclude=anomaly_keywords)

        # Search captions with whole-word matching
        matched_keywords = []
        for caption in captions:
            matched_keywords.extend(find_word_matches(caption, anomaly_keywords + target_words))

        if matched_keywords:
            confidence = min(len(set(matched_keywords)) * 0.2, 0.9)
            explanation = f"Detected keywords: {', '.join(set(matched_keywords))}"
            return True, confidence, explanation

        return False, 0.2, "No anomaly indicators found in captions."

    def _generate_features(
        self,
        scores: np.ndarray[Any, Any],
        captions: list[str],
    ) -> torch.Tensor:
        """Generate feature representation.

        Args:
            scores: Frame scores
            captions: Frame captions

        Returns:
            Feature tensor [1, 128]
        """
        features = []

        # Score statistics
        features.extend(
            [
                np.mean(scores),
                np.std(scores),
                np.max(scores),
                np.min(scores),
                np.percentile(scores, 75),
                np.percentile(scores, 25),
            ]
        )

        # Caption statistics (simple)
        avg_caption_len = np.mean([len(c.split()) for c in captions])
        features.append(avg_caption_len / 100.0)  # Normalize

        # Anomaly duration
        is_anomaly = scores > 0.5
        anomaly_ratio = is_anomaly.mean()
        features.append(anomaly_ratio)

        # Pad to 128D
        features = np.array(features)  # type: ignore[assignment, unused-ignore]
        if len(features) < 128:
            features = np.pad(features, (0, 128 - len(features)))  # type: ignore[assignment, unused-ignore]

        return torch.from_numpy(features).float().unsqueeze(0)

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> torch.Tensor:
        """Extract features for ML fusion pipeline."""
        results = self.detect(data)
        features = results["features"]
        return nn.functional.normalize(features, p=2, dim=1)
