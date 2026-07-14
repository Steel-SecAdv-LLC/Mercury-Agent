# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Foundation Model Adapters for Time-Series Anomaly Detection.

Integrates state-of-the-art foundation models for time-series:
- TimeGPT: Nixtla's 100B+ parameter pre-trained model
- Chronos: Amazon's local inference model
- MOMENT: CMU's multi-task foundation model

Key Features:
    - Zero-shot anomaly detection
    - Fine-tuning on domain data
    - Ensemble predictions across models
    - Seamless integration with Mercury-Agent fusion pipeline

Uses lazy (PEP 562) imports, mirroring ``omni_mercury_engine.models``: the
tensor-surface adapters (chronos / timegpt / matrix_profile / ensemble /
base_foundation) import torch at module top, while the LLM adapters
(``llm_adapter`` / ``ollama_adapter`` — the Ollama, Anthropic, OpenAI and
other cloud backends) are pure ``requests``.  Eager package imports would
force the ~2 GB ML stack onto the cloud-LLM path; lazy exports keep
``FallbackLLMChain``/cloud adapters importable in a torch-free
environment.

Example:
    Basic usage with TimeGPT::

        from omni_mercury_engine.models.foundation import TimeGPTAdapter

        adapter = TimeGPTAdapter(api_key="your_key")
        anomalies = adapter.detect_anomalies(time_series_data)

    Ensemble usage::

        from omni_mercury_engine.models.foundation import FoundationEnsemble

        ensemble = FoundationEnsemble(models=['timegpt', 'chronos'])
        results = ensemble.detect(time_series_data)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Type-only imports for static analysis (CodeQL, mypy, etc.)
# These are not imported at runtime to support lazy loading
if TYPE_CHECKING:
    from omni_mercury_engine.models.foundation.base_foundation import (
        BaseFoundationAdapter as BaseFoundationAdapter,
        BaseFoundationModel as BaseFoundationModel,
        ForecastResult as ForecastResult,
        FoundationModelConfig as FoundationModelConfig,
    )
    from omni_mercury_engine.models.foundation.chronos_adapter import (
        ChronosAdapter as ChronosAdapter,
    )
    from omni_mercury_engine.models.foundation.ensemble import (
        FoundationEnsemble as FoundationEnsemble,
    )
    from omni_mercury_engine.models.foundation.matrix_profile import (
        MatrixProfileDetector as MatrixProfileAdapter,
        MatrixProfileDetector as MatrixProfileDetector,
    )
    from omni_mercury_engine.models.foundation.ollama_adapter import (
        FallbackLLMChain as FallbackLLMChain,
        ModelConfiguration as ModelConfiguration,
        ModelProfile as ModelProfile,
        OllamaConfig as OllamaConfig,
        OllamaLLMAdapter as OllamaLLMAdapter,
        OllamaModel as OllamaModel,
        TemplateLLMAdapter as TemplateLLMAdapter,
        create_fallback_chain as create_fallback_chain,
        create_ollama_adapter as create_ollama_adapter,
    )
    from omni_mercury_engine.models.foundation.timegpt_adapter import (
        TimeGPTAdapter as TimeGPTAdapter,
    )

__all__ = [
    "BaseFoundationAdapter",
    "BaseFoundationModel",
    "ChronosAdapter",
    "FallbackLLMChain",
    "ForecastResult",
    "FoundationEnsemble",
    "FoundationModelConfig",
    "MatrixProfileAdapter",
    "MatrixProfileDetector",
    "ModelConfiguration",
    "ModelProfile",
    "OllamaConfig",
    "OllamaLLMAdapter",
    "OllamaModel",
    "TemplateLLMAdapter",
    "TimeGPTAdapter",
    "create_fallback_chain",
    "create_ollama_adapter",
]

_LAZY_IMPORTS = {
    # Tensor-surface adapters (import torch at module top)
    "BaseFoundationAdapter": "omni_mercury_engine.models.foundation.base_foundation",
    "BaseFoundationModel": "omni_mercury_engine.models.foundation.base_foundation",
    "ForecastResult": "omni_mercury_engine.models.foundation.base_foundation",
    "FoundationModelConfig": "omni_mercury_engine.models.foundation.base_foundation",
    "ChronosAdapter": "omni_mercury_engine.models.foundation.chronos_adapter",
    "FoundationEnsemble": "omni_mercury_engine.models.foundation.ensemble",
    "MatrixProfileDetector": "omni_mercury_engine.models.foundation.matrix_profile",
    "TimeGPTAdapter": "omni_mercury_engine.models.foundation.timegpt_adapter",
    # LLM adapters (pure requests; must stay importable without torch)
    "FallbackLLMChain": "omni_mercury_engine.models.foundation.ollama_adapter",
    "ModelConfiguration": "omni_mercury_engine.models.foundation.ollama_adapter",
    "ModelProfile": "omni_mercury_engine.models.foundation.ollama_adapter",
    "OllamaConfig": "omni_mercury_engine.models.foundation.ollama_adapter",
    "OllamaLLMAdapter": "omni_mercury_engine.models.foundation.ollama_adapter",
    "OllamaModel": "omni_mercury_engine.models.foundation.ollama_adapter",
    "TemplateLLMAdapter": "omni_mercury_engine.models.foundation.ollama_adapter",
    "create_fallback_chain": "omni_mercury_engine.models.foundation.ollama_adapter",
    "create_ollama_adapter": "omni_mercury_engine.models.foundation.ollama_adapter",
}


def __getattr__(name: str) -> object:
    """Lazy import foundation adapters on first access."""
    if name == "MatrixProfileAdapter":
        # Compatibility alias for tests
        import importlib

        module = importlib.import_module("omni_mercury_engine.models.foundation.matrix_profile")
        return module.MatrixProfileDetector
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
