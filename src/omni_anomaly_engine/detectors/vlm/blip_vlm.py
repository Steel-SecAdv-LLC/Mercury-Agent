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
BLIP Vision-Language Model Detector for Zero-Shot Anomaly Detection.

Implements BLIPVLMDetector using Salesforce BLIP model for image captioning
and anomaly detection with 128D feature normalization for fusion pipeline.

Key Features:
    1. Zero-shot anomaly detection via image captioning
    2. 128D normalized feature extraction for DetectorRegistry integration
    3. Graceful fallback when HuggingFace transformers unavailable
    4. Interpretable anomaly explanations via natural language

Reference:
    Li et al. "BLIP: Bootstrapping Language-Image Pre-training for Unified
    Vision-Language Understanding and Generation"
    https://arxiv.org/abs/2201.12086
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from omni_anomaly_engine.detectors.vlm.base_vlm import (
    BaseVLMDetector,
    LVLMType,
    VLMConfig,
)

logger = logging.getLogger(__name__)

# Optional imports with graceful fallback
HAS_TRANSFORMERS = False
HAS_PIL = False

try:
    from transformers import BlipForConditionalGeneration, BlipProcessor

    HAS_TRANSFORMERS = True
except ImportError:
    BlipProcessor = None
    BlipForConditionalGeneration = None
    logger.debug("transformers not available - BLIP VLM will use mock implementation")

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    Image = None
    logger.debug("PIL not available - image processing will be limited")


# Feature dimension for fusion pipeline normalization
FEATURE_DIM = 128


@dataclass
class BLIPConfig(VLMConfig):
    """Configuration for BLIP VLM detector.

    Attributes:
        model_name: HuggingFace model identifier for BLIP
        model_type: LVLM type (set to LOCAL_CUSTOM for BLIP)
        anomaly_keywords: Keywords indicating anomaly in captions
        normal_keywords: Keywords indicating normal behavior
        feature_dim: Output feature dimension for fusion (default 128)
        use_vqa: Use Visual Question Answering mode
        caption_max_length: Maximum caption length
    """

    model_name: str = "Salesforce/blip-image-captioning-base"
    model_type: LVLMType = LVLMType.LOCAL_CUSTOM
    anomaly_keywords: list[str] | None = None
    normal_keywords: list[str] | None = None
    feature_dim: int = FEATURE_DIM
    use_vqa: bool = False
    caption_max_length: int = 50

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

    This module normalizes variable-dimension BLIP embeddings to a fixed
    128D output for consistent integration with DetectorRegistry.
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
        - Graceful fallback when transformers unavailable

    Example:
        >>> detector = BLIPVLMDetector(
        ...     config=BLIPConfig(anomaly_description="fire or smoke")
        ... )
        >>> results = detector.detect(image_tensor)
        >>> print(f"Anomaly: {results['is_anomaly']}, Score: {results['scores'][0]:.3f}")
        >>> features = detector.extract_features(image_tensor)
        >>> print(f"Feature shape: {features.shape}")  # [1, 128]
    """

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
        self._config = config

        # Feature projection for 128D normalization
        self._feature_projection: FeatureProjection | None = None
        self._hidden_dim: int = 768  # BLIP base hidden dimension

        # Check availability
        self._has_transformers = HAS_TRANSFORMERS
        self._has_pil = HAS_PIL

        if not self._has_transformers:
            logger.warning(
                "transformers not available - BLIP VLM will use mock implementation. "
                "Install with: pip install transformers"
            )

    def _initialize_model(self) -> None:
        """Initialize BLIP model and processor.

        Loads the BLIP model from HuggingFace or creates mock implementation
        if transformers is not available.
        """
        if not self._has_transformers:
            logger.info("Using mock BLIP implementation (transformers not available)")
            self._model = None
            self._processor = None
            return

        try:
            logger.info(f"Loading BLIP model: {self.blip_config.model_name}")

            self._processor = BlipProcessor.from_pretrained(
                self.blip_config.model_name
            )  # nosec B615 - model_name is user-configured, see module docstring for security guidance
            self._model = BlipForConditionalGeneration.from_pretrained(  # nosec B615 - model_name is user-configured
                self.blip_config.model_name
            ).to(
                self.device
            )
            self._model.eval()

            # Initialize feature projection
            self._feature_projection = FeatureProjection(
                input_dim=self._hidden_dim,
                output_dim=self.blip_config.feature_dim,
            ).to(self.device)

            logger.info(f"BLIP model loaded successfully on {self.device}")

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

        Args:
            response: Generated caption from BLIP

        Returns:
            Tuple of (is_anomaly, confidence, explanation)
        """
        response_lower = response.lower()

        # Count anomaly and normal keyword matches
        anomaly_matches = sum(
            1 for kw in self.blip_config.anomaly_keywords if kw.lower() in response_lower
        )
        normal_matches = sum(
            1 for kw in self.blip_config.normal_keywords if kw.lower() in response_lower
        )

        # Also check for user-specified anomaly description
        anomaly_desc_lower = self.blip_config.anomaly_description.lower()
        desc_words = anomaly_desc_lower.split()
        desc_matches = sum(1 for word in desc_words if word in response_lower)

        # Compute confidence based on keyword matches
        total_anomaly = anomaly_matches + desc_matches
        total_normal = normal_matches

        if total_anomaly + total_normal == 0:
            # No keywords found - use neutral score
            confidence = 0.5
            is_anomaly = False
        else:
            # Compute anomaly probability
            confidence = total_anomaly / (total_anomaly + total_normal + 1)
            is_anomaly = confidence > self.blip_config.confidence_threshold

        explanation = f"Caption: {response}. Anomaly keywords: {anomaly_matches}, Normal keywords: {normal_matches}"

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
                images.append(img)

        return images

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Detect anomalies using BLIP image captioning.

        Args:
            data: Images [N, C, H, W] or single image [C, H, W]

        Returns:
            Dict containing:
                - scores: Anomaly confidence scores [N]
                - is_anomaly: Binary anomaly flags [N]
                - explanations: Natural language explanations
                - captions: Raw BLIP captions
                - features: 128D features for fusion
        """
        # Ensure model is initialized
        if self._model is None and self._has_transformers:
            self._initialize_model()

        images = self._preprocess_image(data)

        scores = []
        is_anomaly_list = []
        explanations = []
        captions = []

        for img in images:
            if self._model is not None and self._processor is not None:
                # Use actual BLIP model
                try:
                    inputs = self._processor(img, return_tensors="pt").to(self.device)

                    with torch.no_grad():
                        output_ids = self._model.generate(
                            **inputs,
                            max_length=self.blip_config.caption_max_length,
                        )

                    caption = self._processor.decode(output_ids[0], skip_special_tokens=True)

                except Exception as e:
                    logger.warning(f"BLIP inference failed: {e}")
                    caption = "Unable to generate caption"
            else:
                # Mock implementation
                caption = self._generate_mock_caption(img)

            captions.append(caption)

            # Parse caption for anomaly detection
            is_anom, conf, expl = self._parse_response(caption)
            scores.append(conf)
            is_anomaly_list.append(is_anom)
            explanations.append(expl)

        # Extract features
        features = self.extract_features(data)

        return {
            "scores": np.array(scores),
            "is_anomaly": np.array(is_anomaly_list),
            "explanations": explanations,
            "captions": captions,
            "features": features,
        }

    def _generate_mock_caption(self, image: Any) -> str:
        """Generate mock caption when BLIP model unavailable.

        Args:
            image: Input image (PIL or tensor)

        Returns:
            Mock caption string
        """
        # Generate deterministic mock caption based on image statistics
        if isinstance(image, torch.Tensor):
            mean_val = image.float().mean().item()
            std_val = image.float().std().item()
        elif self._has_pil and hasattr(image, "getdata"):
            # PIL Image
            data = np.array(image)
            mean_val = data.mean() / 255.0
            std_val = data.std() / 255.0
        else:
            mean_val = 0.5
            std_val = 0.2

        # Generate caption based on statistics
        if std_val > 0.3:
            return "An image showing varied scene with multiple elements"
        elif mean_val < 0.3:
            return "A dark scene with low visibility"
        elif mean_val > 0.7:
            return "A bright scene with high exposure"
        else:
            return "A typical scene showing normal activity"

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

                    with torch.no_grad():
                        # Get encoder outputs
                        outputs = self._model.vision_model(**inputs)
                        # Use pooled output or mean of last hidden state
                        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                            feat = outputs.pooler_output
                        else:
                            feat = outputs.last_hidden_state.mean(dim=1)

                        # Project to 128D
                        feat = self._feature_projection(feat)
                        features_list.append(feat)

                except Exception as e:
                    logger.warning(f"Feature extraction failed: {e}")
                    # Return zero features on error
                    features_list.append(
                        torch.zeros(1, self.blip_config.feature_dim, device=self.device)
                    )

            features = torch.cat(features_list, dim=0)

        else:
            # Mock feature extraction
            features = self._generate_mock_features(images)

        # L2 normalize features
        features = torch.nn.functional.normalize(features, p=2, dim=-1)

        return features

    def _generate_mock_features(self, images: list[Any]) -> torch.Tensor:
        """Generate mock features when BLIP model unavailable.

        Args:
            images: List of images

        Returns:
            Mock feature tensor [N, 128]
        """
        n_images = len(images)
        features = torch.zeros(n_images, self.blip_config.feature_dim, device=self.device)

        for i, img in enumerate(images):
            # Generate deterministic features based on image statistics
            if isinstance(img, torch.Tensor):
                # Use image statistics as feature seed
                mean_val = img.float().mean().item()
                std_val = img.float().std().item()
            elif self._has_pil and hasattr(img, "getdata"):
                data = np.array(img)
                mean_val = data.mean() / 255.0
                std_val = data.std() / 255.0
            else:
                mean_val = 0.5
                std_val = 0.2

            # Generate pseudo-random features based on statistics
            torch.manual_seed(int(mean_val * 1000 + std_val * 100))
            features[i] = torch.randn(self.blip_config.feature_dim, device=self.device)

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
