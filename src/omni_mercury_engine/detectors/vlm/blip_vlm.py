# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""BLIP Vision-Language Model Detector for Zero-Shot Anomaly Detection.

Implements BLIPVLMDetector using Salesforce BLIP model for image captioning
and anomaly detection with 128D feature normalization for fusion pipeline.

Key Features:
    1. Zero-shot anomaly detection via image captioning
    2. 128D normalized feature extraction for DetectorRegistry integration
    3. Hard fail-fast when HuggingFace transformers unavailable (Phase 2
       audit cure: silent mock degradation is forbidden — the constructor
       raises ``NotImplementedError`` rather than substituting a synthetic
       mock model that would silently corrupt downstream fusion scores)
    4. Interpretable anomaly explanations via natural language

Reference:
    Li et al. "BLIP: Bootstrapping Language-Image Pre-training for Unified
    Vision-Language Understanding and Generation"
    https://arxiv.org/abs/2201.12086
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from omni_mercury_engine.detectors.provenance import DecisionSource
from omni_mercury_engine.detectors.vlm.base_vlm import (
    BaseVLMDetector,
    LVLMType,
    VLMConfig,
)
from omni_mercury_engine.detectors.vlm.text_match import content_words, count_word_matches
from omni_mercury_engine.security.model_policy import SafeHFLoader, UnsafeModelError

logger = logging.getLogger(__name__)

# Optional imports.  The two optional dependencies have **different**
# failure contracts -- they are not interchangeable:
#
#   * ``transformers`` (extras_require[vlm]) is **required** for the
#     detector to operate at all; the constructor and every dispatch
#     site raise ``NotImplementedError`` when it is missing (Phase 2
#     audit cure -- silent mock degradation is forbidden, see class
#     docstring for the full rationale).
#   * ``PIL`` (Pillow) is genuinely **optional**: when present it
#     enables ``ndarray`` / tensor → ``Image.fromarray`` conversion
#     for raw image inputs.  When absent the detector still runs on
#     callers that pass tensor inputs (the model accepts them
#     directly); only the raw-numpy-image input path is disabled,
#     logged once at import time.
HAS_TRANSFORMERS = False
HAS_PIL = False

try:
    from transformers import BlipForConditionalGeneration, BlipProcessor

    HAS_TRANSFORMERS = True
except ImportError:
    # ``transformers`` is an optional dependency (extras_require[vlm]).  The
    # ``# type: ignore`` annotations mirror the PIL fallback below: mypy sees
    # these names as reassignments of imported types when ``transformers`` is
    # present in the type-checking env, but at runtime they are only reached
    # when the import failed and the names are not yet bound.
    BlipProcessor = None  # type: ignore[assignment, misc, unused-ignore]
    BlipForConditionalGeneration = None  # type: ignore[assignment, misc, unused-ignore]
    logger.debug(
        "transformers not available - BLIP VLM detector disabled "
        "(constructor will raise NotImplementedError; Phase 2 audit forbids silent mock degradation)"
    )

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    Image = None  # type: ignore[assignment, unused-ignore]
    logger.debug("PIL not available - image processing will be limited")

# Feature dimension for fusion pipeline normalization
FEATURE_DIM = 128

#: Documented neutral confidence for a caption carrying no keyword evidence
#: (and for a failed caption).  A caption with evidence never receives this
#: value silently: its confidence is the evidence ratio, and the evidence
#: counts travel in ``detect``'s ``details`` so a genuine 1-vs-1 tie (also
#: confidence 0.5) is distinguishable from no evidence at all.
NO_EVIDENCE_CONFIDENCE = 0.5


def caption_keyword_vote(
    caption: str,
    anomaly_keywords: list[str],
    normal_keywords: list[str],
    anomaly_description: str = "",
) -> tuple[bool, float, str, dict[str, int]]:
    """Vote on a caption with word-boundary keyword evidence.

    Audit cure (B1).  Matching is case-insensitive and whole-word, so
    ``"usual"`` no longer matches inside ``"unusual"`` nor ``"normal"``
    inside ``"abnormal"``; the description contributes only stop-word-free
    content words (``"or"`` never matches ``"floor"``), deduplicated against
    the anomaly keywords.  The vote itself:

    * ``confidence = anomaly_evidence / (anomaly_evidence + normal_evidence)``
      when any evidence exists;
    * with no evidence at all, the documented neutral
      :data:`NO_EVIDENCE_CONFIDENCE` (0.5) and no alarm;
    * ``is_anomaly`` iff there is strictly more anomaly than normal evidence,
      so one genuine anomaly keyword with no normal keyword alarms (the old
      ``total / (total + 1)`` formula pinned that case to exactly 0.5 and it
      never alarmed).

    Args:
        caption: Generated caption text.
        anomaly_keywords: Keywords counting as anomaly evidence.
        normal_keywords: Keywords counting as normal evidence.
        anomaly_description: Free-text anomaly description whose content
            words also count as anomaly evidence.

    Returns:
        Tuple of ``(is_anomaly, confidence, explanation, evidence)`` where
        ``evidence`` carries the raw ``anomaly_evidence`` / ``normal_evidence``
        counts.
    """
    anomaly_terms = [kw.lower() for kw in anomaly_keywords]
    description_terms = content_words(anomaly_description, exclude=anomaly_terms)

    anomaly_matches = count_word_matches(caption, anomaly_terms)
    description_matches = count_word_matches(caption, description_terms)
    normal_matches = count_word_matches(caption, normal_keywords)

    total_anomaly = anomaly_matches + description_matches
    evidence = {"anomaly_evidence": total_anomaly, "normal_evidence": normal_matches}

    if total_anomaly + normal_matches == 0:
        confidence = NO_EVIDENCE_CONFIDENCE
        is_anomaly = False
    else:
        confidence = total_anomaly / (total_anomaly + normal_matches)
        is_anomaly = total_anomaly > normal_matches

    explanation = (
        f"Caption: {caption}. Anomaly evidence: {total_anomaly} "
        f"(keywords: {anomaly_matches}, description terms: {description_matches}), "
        f"Normal evidence: {normal_matches}"
    )
    return is_anomaly, confidence, explanation, evidence


#: Seed of the fixed feature projection.  No trained weights exist for it, so it
#: is a fixed random map of BLIP's pooled embedding; unseeded, every detector
#: instance drew a different map (the same image gave unit vectors up to 0.39
#: apart per component), so features fused or compared across instances or
#: processes were not comparable.
FEATURE_PROJECTION_SEED = 20260723


@dataclass
class BLIPConfig(VLMConfig):
    """Configuration for BLIP VLM detector.

    Attributes:
        model_name: HuggingFace model identifier for BLIP
        model_type: LVLM type (set to LOCAL_CUSTOM for BLIP)
        anomaly_keywords: Keywords indicating anomaly in captions
        normal_keywords: Keywords indicating normal behavior
        feature_dim: Output feature dimension for fusion (default 128)
        use_vqa: When True, the prompt built by ``_create_prompt`` (the
            anomaly description plus any context) is passed to the BLIP
            processor as conditional text, steering caption generation
            toward the anomaly question; the echoed prompt prefix is
            stripped from the decoded caption before the keyword vote.
            When False (default), captions are unconditional.
        caption_max_length: Maximum caption length
        revision: HuggingFace revision (commit SHA preferred) for the
            built-in ``Salesforce/blip-*`` Hub IDs. ``SafeHFLoader``
            requires a pin for every remote load -- supply the SHA
            you have validated. Local-disk paths in ``model_name``
            bypass this requirement.
    """

    model_name: str = "Salesforce/blip-image-captioning-base"
    model_type: LVLMType = LVLMType.LOCAL_CUSTOM
    anomaly_keywords: list[str] | None = None
    normal_keywords: list[str] | None = None
    feature_dim: int = FEATURE_DIM
    use_vqa: bool = False
    caption_max_length: int = 50
    # SafeHFLoader requires a revision pin for remote loads. Operators
    # supply the commit SHA they want; local paths bypass the pin.
    revision: str | None = None

    def __post_init__(self) -> None:
        """Initialize default keywords if not provided."""
        if self.anomaly_keywords is None:
            self.anomaly_keywords = [
                "unusual",
                "abnormal",
                "strange",
                "odd",
                "suspicious",
                "dangerous",
                "threat",
                "anomaly",
                "error",
                "fault",
                "damage",
                "broken",
                "fire",
                "smoke",
                "flood",
                "crash",
                "accident",
                "emergency",
            ]
        if self.normal_keywords is None:
            self.normal_keywords = [
                "normal",
                "typical",
                "usual",
                "regular",
                "standard",
                "ordinary",
                "common",
                "expected",
            ]


class FeatureProjection(nn.Module):
    """Projects BLIP features to 128D for fusion pipeline.

    This module normalizes variable-dimension BLIP embeddings to a fixed 128D output for consistent
    integration with DetectorRegistry.
    """

    def __init__(self, input_dim: int = 768, output_dim: int = FEATURE_DIM) -> None:
        """Initialize feature projection.

        Args:
            input_dim: Input feature dimension from BLIP (typically 768)
            output_dim: Output dimension for fusion (default 128)
        """
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features to output dimension.

        Args:
            x: Input features [batch, input_dim]

        Returns:
            Projected features [batch, output_dim]
        """
        return self.projection(x)


class BLIPVLMDetector(BaseVLMDetector):
    """BLIP-based zero-shot anomaly detector.

    Uses Salesforce BLIP model for image captioning and anomaly detection.
    Provides 128D normalized features for DetectorRegistry integration.

    Features:
        - Zero-shot detection via image captioning
        - Natural language anomaly explanations
        - 128D feature normalization for fusion pipeline
        - Hard fail-fast when transformers unavailable (constructor raises
          ``NotImplementedError`` to comply with the Phase 2 audit ban on
          silent mock degradation; install ``mercury-agent-global[vlm]`` to enable)

    Example:
        >>> # SafeHFLoader requires a revision pin for Hub IDs; supply
        >>> # the commit SHA you have validated for the chosen model.
        >>> detector = BLIPVLMDetector(
        ...     config=BLIPConfig(
        ...         anomaly_description="fire or smoke",
        ...         revision="<validated-commit-sha>",
        ...     )
        ... )
        >>> results = detector.detect(image_tensor)
        >>> print(f"Anomaly: {results['is_anomaly']}, Score: {results['scores'][0]:.3f}")
        >>> features = detector.extract_features(image_tensor)
        >>> print(f"Feature shape: {features.shape}")  # [1, 128]
    """

    decision_source = DecisionSource(
        kind="experimental",
        description=(
            "Caption keyword vote with whole-word matching: confidence = anomaly evidence / "
            "(anomaly + normal evidence), alarming on strictly more anomaly than normal "
            "evidence; a caption with no evidence scores the documented neutral 0.5 and does "
            "not alarm (evidence counts are reported in details, so a tie is distinguishable "
            "from no evidence). The caption comes from the pinned BLIP captioner; the keyword "
            "lists are hand-set, with no recorded calibration."
        ),
    )

    # Operator-known BLIP variants. SafeHFLoader rejects anything outside.
    ALLOWED_MODELS: frozenset[str] = frozenset(
        {
            "Salesforce/blip-image-captioning-base",
            "Salesforce/blip-image-captioning-large",
            "Salesforce/blip-vqa-base",
            "Salesforce/blip-vqa-capfilt-large",
        }
    )

    def __init__(self, config: BLIPConfig | dict[str, Any] | None = None) -> None:
        """Initialize BLIP VLM detector.

        Args:
            config: Detector configuration or dict
        """
        if config is None:
            config = BLIPConfig()
        elif isinstance(config, dict):
            config = BLIPConfig(**config)

        super().__init__(config)
        self.blip_config: BLIPConfig = config

        # Feature projection for 128D normalization
        self._feature_projection: FeatureProjection | None = None
        self._hidden_dim: int = 768  # BLIP base hidden dimension

        # Check availability
        self._has_transformers = HAS_TRANSFORMERS
        self._has_pil = HAS_PIL

        if not self._has_transformers:
            raise NotImplementedError(
                "transformers not available — BLIP VLM cannot operate. "
                "Install with: pip install transformers. "
                "Silent mock degradation is not permitted (Phase 2 audit cure)."
            )

    def _initialize_model(self) -> None:
        """Initialize BLIP model and processor.

        Loads the BLIP model from HuggingFace if ``transformers`` is
        available; otherwise raises ``NotImplementedError`` (the Phase 2
        audit forbids silent mock degradation, so there is no synthetic
        fallback model).
        """
        if not self._has_transformers:
            raise NotImplementedError(
                "transformers not available — BLIP VLM cannot initialize. "
                "Silent mock degradation is not permitted (Phase 2 audit cure)."
            )

        try:
            logger.info(f"Loading BLIP model: {self.blip_config.model_name}")

            self._processor = SafeHFLoader.load_processor(
                BlipProcessor,
                self.blip_config.model_name,
                revision=self.blip_config.revision,
                allowlist=self.ALLOWED_MODELS,
            )
            self._model = SafeHFLoader.load_model(
                BlipForConditionalGeneration,
                self.blip_config.model_name,
                revision=self.blip_config.revision,
                allowlist=self.ALLOWED_MODELS,
            ).to(self.device)
            self._model.eval()

            # Initialize feature projection (a fixed, seeded random map; see
            # FEATURE_PROJECTION_SEED).  The global RNG is left untouched.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(FEATURE_PROJECTION_SEED)
                projection = FeatureProjection(
                    input_dim=self._hidden_dim,
                    output_dim=self.blip_config.feature_dim,
                )
            self._feature_projection = projection.to(self.device).eval()

            logger.info(f"BLIP model loaded successfully on {self.device}")

        except UnsafeModelError:
            # SafeHFLoader policy refusal (missing revision, non-SHA
            # revision, allowlist miss, ...). Surface so the operator
            # sees the actionable supply-chain error instead of the
            # generic "BLIP model not loaded" message that the caller
            # would otherwise hit.
            self._model = None
            self._processor = None
            raise
        except Exception as e:
            logger.error(f"Failed to load BLIP model: {e}")
            self._model = None
            self._processor = None

    def _create_prompt(
        self,
        anomaly_description: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Create prompt for BLIP anomaly detection.

        Args:
            anomaly_description: Description of anomaly to detect
            context: Optional context information

        Returns:
            Formatted prompt string for BLIP
        """
        # BLIP uses simple prompts for conditional generation
        base_prompt = f"Analyze this image for anomalies: {anomaly_description}"

        if context:
            if "location" in context:
                base_prompt += f" Location: {context['location']}."
            if "time" in context:
                base_prompt += f" Time: {context['time']}."

        return base_prompt

    def _parse_response(self, response: str) -> tuple[bool, float, str]:
        """Parse BLIP caption response to extract anomaly decision.

        Delegates to :func:`caption_keyword_vote` (whole-word matching,
        majority-evidence vote; see its docstring for the full rules).

        Args:
            response: Generated caption from BLIP

        Returns:
            Tuple of (is_anomaly, confidence, explanation)
        """
        is_anomaly, confidence, explanation, _ = caption_keyword_vote(
            response,
            self.blip_config.anomaly_keywords or [],
            self.blip_config.normal_keywords or [],
            self.blip_config.anomaly_description,
        )
        return is_anomaly, confidence, explanation

    def _preprocess_image(self, data: np.ndarray[Any, Any] | torch.Tensor) -> list[Any]:
        """Preprocess image data for BLIP.

        Args:
            data: Image tensor [C, H, W] or [N, C, H, W] or numpy array

        Returns:
            List of PIL Images or processed tensors
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        # Ensure 4D tensor [N, C, H, W]
        if data.dim() == 3:
            data = data.unsqueeze(0)

        images = []
        for i in range(data.shape[0]):
            img = data[i]

            # Convert to PIL if available
            if self._has_pil and Image is not None:
                # Assume [C, H, W] format, convert to [H, W, C]
                if img.shape[0] in [1, 3, 4]:  # Channel-first
                    img = img.permute(1, 2, 0)

                # Normalize to 0-255 if needed
                if img.max() <= 1.0:
                    img = (img * 255).byte()

                img_np = img.cpu().numpy().astype(np.uint8)

                # Handle grayscale
                if img_np.shape[-1] == 1:
                    img_np = np.repeat(img_np, 3, axis=-1)

                images.append(Image.fromarray(img_np))
            else:
                images.append(img)  # type: ignore[arg-type, unused-ignore]

        return images

    @staticmethod
    def _strip_prompt_echo(caption: str, prompt: str) -> str:
        """Remove the conditional-text prefix BLIP echoes into its output.

        BLIP conditional generation completes the given text, so the decoded
        string starts with the prompt itself.  The prompt names the anomaly
        description, so leaving it in the caption would make the keyword
        vote match the detector's own question.

        Args:
            caption: Decoded BLIP output.
            prompt: The conditional text that was passed to the processor.

        Returns:
            The caption with a leading (case-insensitive) prompt echo removed.
        """
        stripped = caption.strip()
        prompt_clean = prompt.strip()
        if prompt_clean and stripped.lower().startswith(prompt_clean.lower()):
            return stripped[len(prompt_clean) :].strip()
        return stripped

    def _encode_features(self, inputs: Any) -> torch.Tensor:
        """Project one processed image's BLIP vision features to ``feature_dim``.

        Args:
            inputs: Processor output holding ``pixel_values`` for one image.

        Returns:
            Projected (un-normalized) feature tensor ``[1, feature_dim]``.

        Raises:
            RuntimeError: If the vision forward fails — a zero vector is
                indistinguishable from a real feature downstream, so the
                failure surfaces instead of fabricating one.
        """
        if self._model is None or self._feature_projection is None:
            raise RuntimeError(
                "BLIP model not loaded — cannot extract features. "
                "Silent mock degradation is not permitted (Phase 2 audit cure)."
            )
        try:
            with torch.no_grad():
                outputs = self._model.vision_model(pixel_values=inputs["pixel_values"])
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    feat = outputs.pooler_output
                else:
                    feat = outputs.last_hidden_state.mean(dim=1)
                projected: torch.Tensor = self._feature_projection(feat)
                return projected
        except Exception as e:
            raise RuntimeError(f"BLIP feature extraction failed: {e}") from e

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Detect anomalies using BLIP image captioning.

        Each image is preprocessed and encoded exactly once: the same
        processor inputs drive caption generation and feature extraction
        (previously ``detect`` called ``extract_features(data)``, re-encoding
        every image a second time).

        With ``use_vqa=True`` the prompt from :meth:`_create_prompt` is
        passed as BLIP's conditional text, steering generation toward the
        anomaly description; the echoed prompt prefix is stripped from the
        decoded caption so the keyword vote never matches the question
        itself.

        A caption-generation failure is explicit, never a silent neutral:
        the frame scores the documented neutral 0.5 and does not alarm (the
        base contract's ``is_anomaly`` is a binary flag array with no
        abstain value), and ``details[i]["caption_failed"]`` marks it.

        Args:
            data: Images [N, C, H, W] or single image [C, H, W]

        Returns:
            Dict containing:
                - scores: Anomaly confidence scores [N]
                - is_anomaly: Binary anomaly flags [N]
                - explanations: Natural language explanations
                - captions: Raw BLIP captions ("" where generation failed)
                - details: Per-image evidence counts and ``caption_failed``
                - features: 128D features for fusion
        """
        # Ensure model is initialized
        if self._model is None and self._has_transformers:
            self._initialize_model()
        if self._model is None or self._processor is None or self._feature_projection is None:
            raise RuntimeError(
                "BLIP model not loaded — cannot generate captions. "
                "Silent mock degradation is not permitted (Phase 2 audit cure)."
            )

        images = self._preprocess_image(data)

        vqa_prompt: str | None = None
        if self.blip_config.use_vqa:
            vqa_prompt = self._create_prompt(self.blip_config.anomaly_description)

        scores: list[float] = []
        is_anomaly_list: list[bool] = []
        explanations: list[str] = []
        captions: list[str] = []
        details: list[dict[str, Any]] = []
        feature_list: list[torch.Tensor] = []

        for img in images:
            # Encode once per image; the encoding feeds both the caption
            # generation below and the feature extraction.
            try:
                if vqa_prompt is not None:
                    inputs = self._processor(img, text=vqa_prompt, return_tensors="pt").to(
                        self.device
                    )
                else:
                    inputs = self._processor(img, return_tensors="pt").to(self.device)
            except Exception as e:
                raise RuntimeError(f"BLIP preprocessing failed: {e}") from e

            caption = ""
            failure_reason: str | None = None
            try:
                with torch.no_grad():
                    output_ids = self._model.generate(
                        **inputs,
                        max_length=self.blip_config.caption_max_length,
                    )
                caption = self._processor.decode(output_ids[0], skip_special_tokens=True)
                if vqa_prompt is not None:
                    caption = self._strip_prompt_echo(caption, vqa_prompt)
            except Exception as e:
                logger.warning(f"BLIP inference failed: {e}")
                failure_reason = str(e)

            captions.append(caption)
            feature_list.append(self._encode_features(inputs))

            if failure_reason is not None:
                scores.append(NO_EVIDENCE_CONFIDENCE)
                is_anomaly_list.append(False)
                explanations.append(f"BLIP caption generation failed: {failure_reason}")
                details.append(
                    {"caption_failed": True, "anomaly_evidence": 0, "normal_evidence": 0}
                )
                continue

            # Parse caption for anomaly detection
            is_anom, conf, expl, evidence = caption_keyword_vote(
                caption,
                self.blip_config.anomaly_keywords or [],
                self.blip_config.normal_keywords or [],
                self.blip_config.anomaly_description,
            )
            scores.append(conf)
            is_anomaly_list.append(is_anom)
            explanations.append(expl)
            details.append({"caption_failed": False, **evidence})

        features = torch.nn.functional.normalize(torch.cat(feature_list, dim=0), p=2, dim=-1)

        return {
            "scores": np.array(scores),
            "is_anomaly": np.array(is_anomaly_list),
            "explanations": explanations,
            "captions": captions,
            "details": details,
            "features": features,
        }

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> torch.Tensor:
        """Extract 128D normalized features for fusion pipeline.

        Args:
            data: Images [N, C, H, W] or single image [C, H, W]

        Returns:
            Feature tensor [N, 128] normalized for DetectorRegistry
        """
        # Ensure model is initialized
        if self._model is None and self._has_transformers:
            self._initialize_model()

        images = self._preprocess_image(data)

        if (
            self._model is not None
            and self._processor is not None
            and self._feature_projection is not None
        ):
            # Extract features from BLIP encoder
            features_list = []

            for img in images:
                try:
                    inputs = self._processor(img, return_tensors="pt").to(self.device)
                except Exception as e:
                    # A zero vector is indistinguishable from a real feature
                    # downstream; surface the failure instead of fabricating one.
                    raise RuntimeError(f"BLIP feature extraction failed: {e}") from e
                features_list.append(self._encode_features(inputs))

            features = torch.cat(features_list, dim=0)

        else:
            raise RuntimeError(
                "BLIP model not loaded — cannot extract features. "
                "Silent mock degradation is not permitted (Phase 2 audit cure)."
            )

        # L2 normalize features
        features = torch.nn.functional.normalize(features, p=2, dim=-1)

        return features

    def get_interpretability_score(self, caption: str) -> float:
        """Compute interpretability score for a caption.

        Higher scores indicate more informative captions.

        Args:
            caption: Generated caption

        Returns:
            Interpretability score [0, 1]
        """
        # Score based on caption length and keyword presence
        words = caption.split()
        length_score = min(len(words) / 20.0, 1.0)

        # Check for descriptive words
        descriptive_words = [
            "showing",
            "with",
            "containing",
            "displaying",
            "featuring",
            "including",
        ]
        desc_score = sum(1 for w in descriptive_words if w in caption.lower()) / len(
            descriptive_words
        )

        return 0.6 * length_score + 0.4 * desc_score


def create_blip_detector(
    anomaly_description: str = "unusual or abnormal activity",
    device: str | None = None,
    **kwargs: Any,
) -> BLIPVLMDetector:
    """Factory function to create BLIP VLM detector.

    Args:
        anomaly_description: Description of anomaly to detect
        device: Computation device (auto-detected if None)
        **kwargs: Additional config parameters

    Returns:
        Configured BLIPVLMDetector instance
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = BLIPConfig(
        anomaly_description=anomaly_description,
        device=device,
        **kwargs,
    )

    return BLIPVLMDetector(config)
