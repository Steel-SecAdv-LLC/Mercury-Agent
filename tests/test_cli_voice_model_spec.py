# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``mercury-agent voice --model`` names a provider and a model, and Ollama tags still work.

The flag used to build an Ollama chain the voice never consulted, so it
changed nothing a user read (the console said so). It now attaches the
named provider to the voice. ``provider:model`` names one of the shipped
providers; a bare name, or ``ollama:<name>``, is an Ollama model, and a
``mistral:7b``-style Ollama tag (a size, ``latest``, a quantisation) is not
mistaken for the Mistral provider.
"""

from __future__ import annotations

import pytest

from omni_mercury_engine.cli import _parse_voice_model


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("", ("", "")),
        ("llama3.2:3b", ("ollama", "llama3.2:3b")),
        ("mistral:7b", ("ollama", "mistral:7b")),
        ("mistral:latest", ("ollama", "mistral:latest")),
        ("mistral:7b-instruct-q4_K_M", ("ollama", "mistral:7b-instruct-q4_K_M")),
        ("ollama:mistral:7b", ("ollama", "mistral:7b")),
        ("mistral:mistral-large-latest", ("mistral", "mistral-large-latest")),
        ("openai:gpt-4o", ("openai", "gpt-4o")),
        ("OpenAI:gpt-4o", ("openai", "gpt-4o")),
        ("groq:llama-3.3-70b-versatile", ("groq", "llama-3.3-70b-versatile")),
        ("lmstudio:qwen2.5-7b-instruct", ("lmstudio", "qwen2.5-7b-instruct")),
        ("azure_openai:my-deployment", ("azure_openai", "my-deployment")),
        ("not-a-provider:thing", ("ollama", "not-a-provider:thing")),
    ],
)
def test_model_specs_resolve_to_a_provider_and_a_model(
    spec: str, expected: tuple[str, str]
) -> None:
    assert _parse_voice_model(spec) == expected
