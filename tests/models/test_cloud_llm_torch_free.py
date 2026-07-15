# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Laziness contract: the cloud LLM path must not import torch.

The Anthropic/OpenAI/Cohere/Gemini adapters are pure ``requests``
transports.  Historically ``llm_adapter.py`` imported torch at module top
and ``models/foundation/__init__.py`` eagerly imported every
tensor-surface adapter, so selecting a cloud provider dragged the ~2 GB
ML stack into the process.  These tests pin the decoupling by importing
the whole cloud chain in a subprocess and asserting torch was never
pulled into ``sys.modules`` — which holds regardless of whether torch is
installed, and therefore fails loudly if an eager import is ever
reintroduced.  (Operational proof in a genuinely torch-free environment:
a core-only venv drives ``AnthropicCloudAdapter`` end to end; see the
[foundation]/[llm] extras notes in pyproject.toml.)
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

_PROOF = """
import sys
from omni_mercury_engine.models.foundation.ollama_adapter import (
    AnthropicCloudAdapter,
    FallbackLLMChain,
    OpenAICloudAdapter,
)
from omni_mercury_engine.models.foundation.llm_adapter import LLMConfig, LLMProvider
from omni_mercury_engine.models.foundation import FallbackLLMChain as lazy_chain

assert lazy_chain is FallbackLLMChain, "package __getattr__ must resolve the same class"

import numpy as np

adapter = AnthropicCloudAdapter(LLMConfig(provider=LLMProvider.ANTHROPIC, api_key="test-not-real"))
prompt = adapter._build_anomaly_prompt(np.arange(24.0).reshape(6, 4), context={"k": "v"})
assert "Numerical data" in prompt.user_prompt

assert "torch" not in sys.modules, "cloud LLM path imported torch"
print("torch-free-ok")
"""


def test_cloud_llm_chain_never_imports_torch() -> None:
    """Importing + operating the cloud adapters must leave torch unloaded."""
    result = subprocess.run(
        [sys.executable, "-c", _PROOF],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "torch-free-ok" in result.stdout


def test_tensor_inputs_still_recognised_when_torch_present() -> None:
    """With torch importable, a Tensor input keeps its dedicated prompt branch."""
    torch = __import__("torch")

    from omni_mercury_engine.models.foundation.llm_adapter import LLMConfig, LLMProvider
    from omni_mercury_engine.models.foundation.ollama_adapter import AnthropicCloudAdapter

    adapter = AnthropicCloudAdapter(
        LLMConfig(provider=LLMProvider.ANTHROPIC, api_key="test-not-real")
    )
    prompt = adapter._build_anomaly_prompt(torch.arange(12.0).reshape(3, 4), context=None)
    assert "Tensor data" in prompt.user_prompt


def test_no_vendor_default_model_ships_for_any_cloud_provider() -> None:
    """Every cloud adapter requires an explicit model_name; none ships a default.

    Vendor-neutral policy: Mercury privileges no provider. An adapter
    constructed without a model_name reports itself unavailable (so the
    chain falls back to the deterministic template) instead of silently
    talking to a vendor-chosen model; an explicit model_name plus key makes
    it available and is used verbatim. This also closes the retired-default
    failure mode (a hard-coded vendor id rotted upstream and 404'd on the
    first call).
    """
    from omni_mercury_engine.models.foundation.llm_adapter import LLMConfig, LLMProvider
    from omni_mercury_engine.models.foundation.ollama_adapter import (
        AnthropicCloudAdapter,
        CohereCloudAdapter,
        DeepSeekAdapter,
        GeminiCloudAdapter,
        HuggingFaceCloudAdapter,
        OpenAICloudAdapter,
        XAIGrokAdapter,
    )

    # type[Any]: `.model` is defined per concrete adapter, not on the base.
    cases: list[tuple[type[Any], LLMProvider]] = [
        (OpenAICloudAdapter, LLMProvider.OPENAI),
        (AnthropicCloudAdapter, LLMProvider.ANTHROPIC),
        (HuggingFaceCloudAdapter, LLMProvider.HUGGINGFACE),
        (XAIGrokAdapter, LLMProvider.XAI),
        (DeepSeekAdapter, LLMProvider.DEEPSEEK),
        (CohereCloudAdapter, LLMProvider.COHERE),
        (GeminiCloudAdapter, LLMProvider.GEMINI),
    ]
    for adapter_cls, provider in cases:
        unset = adapter_cls(LLMConfig(provider=provider, api_key="test-not-real"))
        assert unset.model == "", f"{adapter_cls.__name__} ships a default model"
        assert unset.is_available() is False, f"{adapter_cls.__name__} available without a model"

        explicit = adapter_cls(
            LLMConfig(
                provider=provider, model_name="operator-chosen-model", api_key="test-not-real"
            )
        )
        assert explicit.model == "operator-chosen-model"
        assert explicit.is_available() is True, f"{adapter_cls.__name__} rejected explicit model"


def test_mercury_provider_model_env_fallback(monkeypatch: Any) -> None:
    """Every provider gets the same MERCURY_<PROVIDER>_MODEL env surface.

    Precedence: explicit LLMConfig.model_name wins over the env variable;
    the env variable makes an otherwise model-less adapter operable; with
    neither, the adapter stays unavailable (no vendor default).
    """
    from omni_mercury_engine.models.foundation.llm_adapter import LLMConfig, LLMProvider
    from omni_mercury_engine.models.foundation.ollama_adapter import (
        AnthropicCloudAdapter,
        CohereCloudAdapter,
        DeepSeekAdapter,
        GeminiCloudAdapter,
        HuggingFaceCloudAdapter,
        OpenAICloudAdapter,
        XAIGrokAdapter,
    )

    cases: list[tuple[type[Any], LLMProvider, str]] = [
        (OpenAICloudAdapter, LLMProvider.OPENAI, "MERCURY_OPENAI_MODEL"),
        (AnthropicCloudAdapter, LLMProvider.ANTHROPIC, "MERCURY_ANTHROPIC_MODEL"),
        (HuggingFaceCloudAdapter, LLMProvider.HUGGINGFACE, "MERCURY_HUGGINGFACE_MODEL"),
        (XAIGrokAdapter, LLMProvider.XAI, "MERCURY_XAI_MODEL"),
        (DeepSeekAdapter, LLMProvider.DEEPSEEK, "MERCURY_DEEPSEEK_MODEL"),
        (CohereCloudAdapter, LLMProvider.COHERE, "MERCURY_COHERE_MODEL"),
        (GeminiCloudAdapter, LLMProvider.GEMINI, "MERCURY_GEMINI_MODEL"),
    ]
    for adapter_cls, provider, env_var in cases:
        monkeypatch.setenv(env_var, "operator-env-model")
        via_env = adapter_cls(LLMConfig(provider=provider, api_key="test-not-real"))
        assert via_env.model == "operator-env-model", adapter_cls.__name__
        assert via_env.is_available() is True, adapter_cls.__name__

        explicit = adapter_cls(
            LLMConfig(provider=provider, model_name="explicit-model", api_key="test-not-real")
        )
        assert explicit.model == "explicit-model", f"{adapter_cls.__name__}: env beat explicit"
        monkeypatch.delenv(env_var)


def test_identity_clause_reaches_every_backend_prompt() -> None:
    """Mercury Agent's product-identity contract rides every system prompt.

    Mercury Agent always identifies as Mercury Agent -- never under the
    backend model's own persona -- while staying transparent that an
    operator-configured LLM backend serves the language generation.
    """
    from omni_mercury_engine.models.foundation.llm_adapter import (
        MERCURY_IDENTITY_CLAUSE,
        LLMConfig,
        LLMProvider,
    )
    from omni_mercury_engine.models.foundation.ollama_adapter import AnthropicCloudAdapter
    from omni_mercury_engine.reasoning.backend import SYSTEM_PROMPT

    assert "Mercury Agent" in MERCURY_IDENTITY_CLAUSE
    assert "never claim to be a human" in MERCURY_IDENTITY_CLAUSE
    assert MERCURY_IDENTITY_CLAUSE in SYSTEM_PROMPT

    adapter = AnthropicCloudAdapter(
        LLMConfig(provider=LLMProvider.ANTHROPIC, api_key="test-not-real")
    )
    import numpy as np

    prompt = adapter._build_anomaly_prompt(np.arange(8.0).reshape(2, 4), context=None)
    assert MERCURY_IDENTITY_CLAUSE in prompt.system_prompt
