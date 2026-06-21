# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""LVLM Backend implementations for anomaly detection.

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

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from PIL import Image

from omni_mercury_engine.security.model_policy import SafeHFLoader

logger = logging.getLogger(__name__)


class LVLMBackend(ABC):
    """Abstract base class for LVLM backends."""

    # Subclasses override with the set of HuggingFace ids they accept.
    # SafeHFLoader gates every from_pretrained call against this set so
    # an operator who passes an unexpected model id at config-time gets
    # an UnsafeModelError instead of a silent default-branch load.
    ALLOWED_MODELS: frozenset[str] = frozenset()

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        revision: str | None = None,
    ):
        """Initialize LVLM backend.

        Args:
            model_name: HuggingFace model identifier
            device: Computation device
            max_new_tokens: Maximum generation tokens
            temperature: Sampling temperature
            revision: Pinned revision (commit SHA preferred). Required
                for remote loads; ``None`` is only accepted for local
                paths.
        """
        self.model_name = model_name
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.revision = revision

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
            arr: np.ndarray[Any, Any] = image
            if arr.ndim == 4:
                arr = arr[0]  # Take first if batched
            if arr.shape[0] in [1, 3]:  # CHW format
                arr = np.transpose(arr, (1, 2, 0))
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            return Image.fromarray(arr)
        raise ValueError(f"Unsupported image type: {type(image)}")


class Qwen2VLBackend(LVLMBackend):
    """Qwen2-VL backend for vision-language tasks."""

    ALLOWED_MODELS: frozenset[str] = frozenset(
        {
            "Qwen/Qwen2-VL-2B-Instruct",
            "Qwen/Qwen2-VL-7B-Instruct",
            "Qwen/Qwen2-VL-72B-Instruct",
            # Qwen2.5-VL family.  ``VLMConfig.model_name`` defaults to
            # ``Qwen/Qwen2.5-VL-7B-Instruct``; without these entries the
            # out-of-the-box detector would hit SafeHFLoader's allowlist
            # gate and refuse even with a valid revision pin.
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/Qwen2.5-VL-72B-Instruct",
        }
    )

    def initialize(self) -> None:
        """Load Qwen2-VL model."""
        try:
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

            logger.info(f"Loading Qwen2-VL: {self.model_name}")

            self.processor = SafeHFLoader.load_processor(
                AutoProcessor,
                self.model_name,
                revision=self.revision,
                allowlist=self.ALLOWED_MODELS,
            )
            self.model = SafeHFLoader.load_model(
                Qwen2VLForConditionalGeneration,
                self.model_name,
                revision=self.revision,
                allowlist=self.ALLOWED_MODELS,
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

    ALLOWED_MODELS: frozenset[str] = frozenset(
        {
            "openbmb/MiniCPM-V-2_6",
            "openbmb/MiniCPM-Llama3-V-2_5",
            "openbmb/MiniCPM-V-2",
        }
    )

    def initialize(self) -> None:
        """Load MiniCPM-V model."""
        try:
            from transformers import AutoModel, AutoTokenizer

            logger.info(f"Loading MiniCPM-V: {self.model_name}")

            self.model = SafeHFLoader.load_model(
                AutoModel,
                self.model_name,
                revision=self.revision,
                allowlist=self.ALLOWED_MODELS,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            ).to(self.device)
            self.processor = SafeHFLoader.load_tokenizer(
                AutoTokenizer,
                self.model_name,
                revision=self.revision,
                allowlist=self.ALLOWED_MODELS,
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

    ALLOWED_MODELS: frozenset[str] = frozenset(
        {
            "llava-hf/llava-1.5-7b-hf",
            "llava-hf/llava-1.5-13b-hf",
            "llava-hf/llava-v1.6-mistral-7b-hf",
            "llava-hf/llava-v1.6-vicuna-7b-hf",
            "llava-hf/llava-v1.6-vicuna-13b-hf",
        }
    )

    def initialize(self) -> None:
        """Load LLaVA model."""
        try:
            from transformers import AutoProcessor, LlavaForConditionalGeneration

            logger.info(f"Loading LLaVA: {self.model_name}")

            self.processor = SafeHFLoader.load_processor(
                AutoProcessor,
                self.model_name,
                revision=self.revision,
                allowlist=self.ALLOWED_MODELS,
            )
            self.model = SafeHFLoader.load_model(
                LlavaForConditionalGeneration,
                self.model_name,
                revision=self.revision,
                allowlist=self.ALLOWED_MODELS,
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
    """Mock LVLM backend — hard-fails at construction.

    Phase 2 audit cure: silent mock degradation is not permitted in
    production.  Instantiating this class raises ``NotImplementedError``
    so operators are forced to configure a real vision-language model.
    """

    def initialize(self) -> None:
        """Initialize."""
        raise NotImplementedError(
            "MockLVLMBackend cannot be used in production. "
            "Configure a real LVLM backend (e.g. Qwen2VL, MiniCPMV, LLaVA)."
        )

    def generate(
        self,
        images: list[Image.Image] | list[np.ndarray[Any, Any]],
        prompt: str,
    ) -> str:
        """Generate."""
        raise NotImplementedError(
            "MockLVLMBackend cannot be used in production. "
            "Configure a real LVLM backend (e.g. Qwen2VL, MiniCPMV, LLaVA)."
        )

    def vqa(
        self,
        image: Image.Image | np.ndarray[Any, Any] | torch.Tensor,
        question: str,
    ) -> str:
        """Vqa."""
        raise NotImplementedError(
            "MockLVLMBackend cannot be used in production. "
            "Configure a real LVLM backend (e.g. Qwen2VL, MiniCPMV, LLaVA)."
        )


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
        # Phase 2 audit cure: silent fall-through to ``MockLVLMBackend``
        # for an unknown ``model_type`` is forbidden — production code
        # would otherwise route through a backend whose ``initialize``,
        # ``generate``, and ``vqa`` are all hard-fail stubs, surfacing
        # the loss of a real model only at first use rather than at
        # configuration time.  Raise here so the misconfiguration is
        # caught at the factory boundary.
        raise ValueError(
            f"Unknown LVLM model_type {model_type!r}. " f"Supported: {sorted(backends)}."
        )

    backend_class = backends[model_type]
    return backend_class(model_name, device, **kwargs)
