# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Air-gap contract for the LLM chain: MERCURY_OFFLINE constructs no cloud adapter.

Two layers are pinned here:

* **Baseline** (``MERCURY_OFFLINE`` unset): with no cloud configured the chain
  serves the local/template path and constructs no cloud adapter — local-first
  is the default, independent of the air-gap flag.
* **Hard air-gap** (``MERCURY_OFFLINE=1``): even with cloud explicitly enabled
  and a provider configured, no cloud adapter is ever constructed or called.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omni_mercury_engine.models.foundation.llm_adapter import LLMConfig, LLMProvider
from omni_mercury_engine.models.foundation.ollama_adapter import FallbackLLMChain

if TYPE_CHECKING:
    import pytest

_OPENAI_ADAPTER = "omni_mercury_engine.models.foundation.ollama_adapter.OpenAICloudAdapter"


def _openai_cloud_config() -> LLMConfig:
    """A cloud config (never expected to be constructed in these tests)."""
    return LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="some-model",
        api_key="test-key-not-used",
    )


class TestLLMAirGap:
    """MERCURY_OFFLINE provably keeps the chain on local + template only."""

    def test_baseline_local_only_without_cloud_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MERCURY_OFFLINE", raising=False)
        chain = FallbackLLMChain()  # enable_cloud defaults to False
        assert chain._cloud is None
        assert "cloud" not in chain._active_name  # ollama (if present) or template

    def test_offline_never_constructs_cloud(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MERCURY_OFFLINE", "1")

        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("cloud adapter constructed under MERCURY_OFFLINE")

        # Any attempt to build a cloud adapter raises — a green run proves zero
        # cloud construction even with cloud explicitly enabled + configured.
        monkeypatch.setattr(_OPENAI_ADAPTER, _boom)

        chain = FallbackLLMChain(enable_cloud=True, cloud_config=_openai_cloud_config())
        assert chain._cloud is None
        assert "cloud" not in chain._active_name
        # Defense in depth: a direct call also refuses under the air-gap.
        assert chain._create_cloud_adapter() is None

    def test_offline_truthy_spelling_also_blocks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Reuses the dataset-layer truthy contract (1/true/yes/on).
        monkeypatch.setenv("MERCURY_OFFLINE", "true")
        chain = FallbackLLMChain(enable_cloud=True, cloud_config=_openai_cloud_config())
        assert chain._cloud is None
