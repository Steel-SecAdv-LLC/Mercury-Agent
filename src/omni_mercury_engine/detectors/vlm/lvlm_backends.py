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
LVLM Backend implementations for anomaly detection.

Provides unified interface for various Large Vision-Language Models:
- Qwen2-VL
- MiniCPM-V
- LLaVA
- InternVL

Security Note:
    When loading models from HuggingFace Hub, consider using specific revision
    hashes (e.g., "model-name@abc123") instead of branch names to ensure
    reproducibility and prevent supply chain attacks. Model names without
    revisions will load the latest version which may change unexpectedly.
    See: https://huggingface.co/docs/hub/security
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class LVLMBackend(ABC):
    """Abstract base class for LVLM backends."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ):
        """Initialize LVLM backend.

        Args:
            model_name: HuggingFace model identifier
            device: Computation device
            max_new_tokens: Maximum generation tokens
            temperature: Sampling temperature
        """
        self.model_name = model_name
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.model: Any = None
        self.processor: Any = None
        self._initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Load model and processor."""
        pass

    @abstractmethod
    def generate(
        self,
        images: list[Image.Image] | list[np.ndarray[Any, Any]],
        prompt: str,
    ) -> str:
        """Generate response for visual question.

        Args:
            images: Input images (single image or sequence)
            prompt: Text prompt/question

        Returns:
            Model response text
        """
        pass

    def _ensure_initialized(self) -> None:
        """Ensure model is loaded."""
        if not self._initialized:
            self.initialize()
            self._initialized = True

    def _to_pil(self, image: np.ndarray[Any, Any] | Image.Image) -> Image.Image:
        """Convert image to PIL format."""
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, np.ndarray):
            if image.ndim == 4:
                image = image[0]  # Take first if batched
            if image.shape[0] in [1, 3]:  # CHW format
                image = np.transpose(image, (1, 2, 0))
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            return Image.fromarray(image)
        raise ValueError(f"Unsupported image type: {type(image)}")


class Qwen2VLBackend(LVLMBackend):
    """Qwen2-VL backend for vision-language tasks."""

    def initialize(self) -> None:
        """Load Qwen2-VL model."""
        try:
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

            logger.info(f"Loading Qwen2-VL: {self.model_name}")

            self.processor = AutoProcessor.from_pretrained(self.model_name)  # nosec B615
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(  # nosec B615
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.model.eval()

            logger.info("Qwen2-VL loaded successfully")

        except ImportError as e:
            raise ImportError(
                "transformers and qwen-vl-utils required for Qwen2-VL. "
                "Install with: pip install transformers qwen-vl-utils"
            ) from e

    def generate(
        self,
        images: list[Image.Image] | list[np.ndarray[Any, Any]],
        prompt: str,
    ) -> str:
        """Generate response using Qwen2-VL."""
        self._ensure_initialized()

        # Convert images
        pil_images = [self._to_pil(img) for img in images]

        # Build conversation
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": img} for img in pil_images]
                + [{"type": "text", "text": prompt}],
            }
        ]

        # Process inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=pil_images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )

        # Decode
        response = self.processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )[0]

        return str(response)


class MiniCPMVBackend(LVLMBackend):
    """MiniCPM-V backend - efficient vision-language model."""

    def initialize(self) -> None:
        """Load MiniCPM-V model."""
        try:
            from transformers import AutoModel, AutoTokenizer

            logger.info(f"Loading MiniCPM-V: {self.model_name}")

            # nosec B615 - model_name is user-configured; see module docstring for security guidance
            self.model = AutoModel.from_pretrained(  # nosec B615
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            ).to(self.device)
            self.processor = AutoTokenizer.from_pretrained(  # nosec B615
                self.model_name,
                trust_remote_code=True,
            )
            self.model.eval()

            logger.info("MiniCPM-V loaded successfully")

        except ImportError as e:
            raise ImportError(
                "transformers required for MiniCPM-V. " "Install with: pip install transformers"
            ) from e

    def generate(
        self,
        images: list[Image.Image] | list[np.ndarray[Any, Any]],
        prompt: str,
    ) -> str:
        """Generate response using MiniCPM-V."""
        self._ensure_initialized()

        pil_images = [self._to_pil(img) for img in images]

        # MiniCPM-V specific chat format
        msgs = [{"role": "user", "content": [*pil_images, prompt]}]

        with torch.no_grad():
            response = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.processor,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )

        return str(response)


class LLaVABackend(LVLMBackend):
    """LLaVA backend for vision-language tasks."""

    def initialize(self) -> None:
        """Load LLaVA model."""
        try:
            from transformers import AutoProcessor, LlavaForConditionalGeneration

            logger.info(f"Loading LLaVA: {self.model_name}")

            self.processor = AutoProcessor.from_pretrained(self.model_name)  # nosec B615
            self.model = LlavaForConditionalGeneration.from_pretrained(  # nosec B615
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.model.eval()

            logger.info("LLaVA loaded successfully")

        except ImportError as e:
            raise ImportError(
                "transformers required for LLaVA. " "Install with: pip install transformers"
            ) from e

    def generate(
        self,
        images: list[Image.Image] | list[np.ndarray[Any, Any]],
        prompt: str,
    ) -> str:
        """Generate response using LLaVA."""
        self._ensure_initialized()

        pil_images = [self._to_pil(img) for img in images]

        # LLaVA conversation format
        conversation = [
            {
                "role": "user",
                "content": [{"type": "image"} for _ in pil_images]
                + [{"type": "text", "text": prompt}],
            }
        ]

        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(
            text=text,
            images=pil_images,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )

        response = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        return str(response)


class MockLVLMBackend(LVLMBackend):
    """Mock backend for testing without actual models."""

    def initialize(self) -> None:
        """No-op initialization."""
        logger.info("Using Mock LVLM backend (for testing)")

    def generate(
        self,
        images: list[Image.Image] | list[np.ndarray[Any, Any]],
        prompt: str,
    ) -> str:
        """Generate mock response."""
        # Simple heuristic for testing
        if "anomaly" in prompt.lower() or "unusual" in prompt.lower():
            return (
                "Based on my analysis of the image(s), I do not detect any "
                "clear anomalies. The scene appears normal with typical "
                "activity patterns. Confidence: 0.2"
            )
        return "The image shows a typical scene with no unusual elements."

    def vqa(
        self,
        image: Image.Image | np.ndarray[Any, Any] | torch.Tensor,
        question: str,
    ) -> str:
        """Visual Question Answering for test compatibility.

        Args:
            image: Input image
            question: Question about the image

        Returns:
            Answer string
        """
        return self.generate([image] if not isinstance(image, list) else image, question)


def get_lvlm_backend(
    model_type: str,
    model_name: str | None = None,
    device: str = "cuda",
    **kwargs: Any,
) -> LVLMBackend:
    """Factory function to get appropriate LVLM backend.

    Args:
        model_type: Type of LVLM ('qwen2_vl', 'minicpm_v', 'llava', 'mock')
        model_name: HuggingFace model identifier (optional, defaults to model_type)
        device: Computation device
        **kwargs: Additional backend arguments

    Returns:
        Configured LVLM backend
    """
    backends: dict[str, type[LVLMBackend]] = {
        "qwen2_vl": Qwen2VLBackend,
        "minicpm_v": MiniCPMVBackend,
        "llava": LLaVABackend,
        "mock": MockLVLMBackend,
    }

    # If model_name not provided, use model_type as model_name
    if model_name is None:
        model_name = model_type

    if model_type not in backends:
        logger.warning(f"Unknown model type {model_type}, using mock backend")
        model_type = "mock"

    backend_class = backends[model_type]
    return backend_class(model_name, device, **kwargs)
