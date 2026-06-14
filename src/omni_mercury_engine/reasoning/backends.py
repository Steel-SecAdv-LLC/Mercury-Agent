# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Concrete reasoning backends Mercury can call.

Three implementations of :class:`ReasoningBackend`:

* :class:`MockReasoningBackend` — deterministic, network-free; lets CI exercise
  the reasoning surface with no live model.
* :class:`LocalReasoningBackend` — offline-first, air-gap-safe; serves the
  local Ollama runtime when present and the builtin deterministic template
  otherwise. The intended local model is a Mercury-domain LoRA adapter
  (roadmap; see ``docs/ROADMAP.md``) loaded through this same offline chain.
* :class:`RemoteReasoningBackend` — network-capable; calls an operator-declared
  frontier model. No model name is hard-coded here — the model id and
  credentials come from an :class:`LLMConfig` the operator supplies.

All three thread the optional :class:`UsageLedger` from the LLM substrate so
provider-reported token spend is accounted regardless of which backend served
a call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omni_mercury_engine.models.foundation.ollama_adapter import (
    FallbackLLMChain,
    OllamaConfig,
)
from omni_mercury_engine.reasoning.backend import ReasoningBackend

if TYPE_CHECKING:
    from omni_mercury_engine.models.foundation.llm_adapter import LLMConfig
    from omni_mercury_engine.models.foundation.llm_usage import UsageLedger
    from omni_mercury_engine.security.sigma_immutable_gate import SigmaImmutableGate

__all__ = [
    "LocalReasoningBackend",
    "MockReasoningBackend",
    "ReasoningBackendUnavailableError",
    "RemoteReasoningBackend",
]


class ReasoningBackendUnavailableError(RuntimeError):
    """Raised when a network-capable backend is invoked under the air-gap.

    Fail-closed by design: a direct cloud call made while ``MERCURY_OFFLINE`` is
    set raises rather than silently substituting a weaker local/template answer
    for the cloud call the caller explicitly made — the unconscious-substitution
    Mercury's design rejects. (The router's serving path degrades gracefully to
    a local backend instead; this guard is only for direct, explicit use.)
    """


class MockReasoningBackend(ReasoningBackend):
    """Deterministic, offline, network-free backend for tests and CI.

    Produces a stable, prompt-derived string without loading any model or
    touching the network, so the reasoning surface and its ethics gate can be
    exercised with no live dependency.
    """

    @property
    def name(self) -> str:
        """Provenance label."""
        return "mock"

    @property
    def model(self) -> str:
        """Model identifier (none — this backend loads no model)."""
        return "mock"

    @property
    def is_offline(self) -> bool:
        """Always offline: this backend never touches the network."""
        return True

    def _generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Return a deterministic synthesis of the prompt's first line."""
        stripped = prompt.strip()
        head = stripped.splitlines()[0] if stripped else "empty prompt"
        return f"[mock-reasoning] {head}"


class LocalReasoningBackend(ReasoningBackend):
    """Offline-first reasoning over Mercury's local LLM chain.

    Backed by :class:`FallbackLLMChain` with cloud disabled: it serves the
    local Ollama runtime when reachable and the deterministic builtin-template
    adapter otherwise. Both paths are loopback/in-process, so this backend is
    air-gap-safe and :attr:`is_offline` is True. The intended local model is a
    Mercury-domain LoRA adapter served through this same chain; until one is
    trained the backend already runs fully offline on the shipped local path.
    """

    def __init__(
        self,
        *,
        ollama_config: OllamaConfig | None = None,
        usage_ledger: UsageLedger | None = None,
        ethics_enabled: bool = True,
        benevolence_scorer: Any | None = None,
        sigma_gate: SigmaImmutableGate | None = None,
    ) -> None:
        """Initialize the offline-first local backend.

        Args:
            ollama_config: Local Ollama configuration; defaults to
                :class:`OllamaConfig`.
            usage_ledger: Optional shared ledger threaded through the chain.
            ethics_enabled: Forwarded to :class:`ReasoningBackend`.
            benevolence_scorer: Forwarded to :class:`ReasoningBackend`.
            sigma_gate: Forwarded to :class:`ReasoningBackend`.
        """
        super().__init__(
            ethics_enabled=ethics_enabled,
            benevolence_scorer=benevolence_scorer,
            sigma_gate=sigma_gate,
        )
        self._ollama_config = ollama_config or OllamaConfig()
        self._chain = FallbackLLMChain(
            ollama_config=self._ollama_config,
            enable_cloud=False,
            usage_ledger=usage_ledger,
        )

    @property
    def name(self) -> str:
        """Provenance label."""
        return "local"

    @property
    def model(self) -> str:
        """Local model this backend is configured to serve."""
        return self._ollama_config.model

    @property
    def is_offline(self) -> bool:
        """Always offline: cloud is disabled; only loopback/in-process paths run."""
        return True

    def _generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate via the offline-first local chain."""
        return self._chain.generate(prompt, system_prompt)


class RemoteReasoningBackend(ReasoningBackend):
    """Network-capable reasoning over an operator-declared frontier model.

    Requires an explicit :class:`LLMConfig` (provider, model id, credentials
    via environment) — no model name is hard-coded here. It remains
    offline-first *within* the chain (a reachable local Ollama runtime is still
    preferred), but the backend is permitted to reach the network, so
    :attr:`is_offline` is False and Mercury's router will not select it under
    hard-offline mode.
    """

    def __init__(
        self,
        *,
        cloud_config: LLMConfig,
        ollama_config: OllamaConfig | None = None,
        usage_ledger: UsageLedger | None = None,
        ethics_enabled: bool = True,
        benevolence_scorer: Any | None = None,
        sigma_gate: SigmaImmutableGate | None = None,
    ) -> None:
        """Initialize the network-capable remote backend.

        Args:
            cloud_config: Operator-declared provider/model/credentials config.
            ollama_config: Local Ollama configuration for the offline-first
                leg of the chain; defaults to :class:`OllamaConfig`.
            usage_ledger: Optional shared ledger threaded through the chain.
            ethics_enabled: Forwarded to :class:`ReasoningBackend`.
            benevolence_scorer: Forwarded to :class:`ReasoningBackend`.
            sigma_gate: Forwarded to :class:`ReasoningBackend`.
        """
        super().__init__(
            ethics_enabled=ethics_enabled,
            benevolence_scorer=benevolence_scorer,
            sigma_gate=sigma_gate,
        )
        self._model = cloud_config.model_name
        self._chain = FallbackLLMChain(
            ollama_config=ollama_config or OllamaConfig(),
            enable_cloud=True,
            cloud_config=cloud_config,
            usage_ledger=usage_ledger,
        )

    @property
    def name(self) -> str:
        """Provenance label."""
        return "remote"

    @property
    def model(self) -> str:
        """Operator-declared model id this backend targets."""
        return self._model

    @property
    def is_offline(self) -> bool:
        """False: this backend is permitted to reach the network."""
        return False

    def _generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate via the network-capable chain.

        Fail-closed under the air-gap: a direct call (bypassing the router)
        while ``MERCURY_OFFLINE`` is set raises
        :class:`ReasoningBackendUnavailableError` rather than silently serving a
        weaker local/template answer. The router degrades gracefully to a local
        backend on the serving path; this guard is only for direct, explicit use.
        """
        from omni_mercury_engine.datasets.exceptions import offline_mode_active

        if offline_mode_active():
            raise ReasoningBackendUnavailableError(
                "RemoteReasoningBackend unavailable: MERCURY_OFFLINE air-gap active"
            )
        return self._chain.generate(prompt, system_prompt)
