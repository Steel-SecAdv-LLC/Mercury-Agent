# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""LLM model registry: capability/cost-aware model selection.

Mercury ships real adapters for ten LLM providers but, until this module,
no unified place to answer "which configured model satisfies this call's
requirements (capabilities, context, budget)?" — provider selection was
manual (`LLMConfig(provider=...)`) and availability-ordered only
(`FallbackLLMChain`). This registry is that substrate.

Two layers, with deliberately different provenance:

* :data:`PROVIDER_CATALOG` — **code-grounded** facts about the adapters
  Mercury ships (wire format, key env var, locality, whether the provider
  reports token usage). Every field is verifiable against
  ``models/foundation/{llm_adapter,ollama_adapter}.py`` and a drift gate in
  ``tests/models/test_llm_registry.py`` pins the catalog to
  ``IMPLEMENTED_LLM_PROVIDERS``.
* :class:`LLMModelSpec` / :class:`LLMModelRegistry` — **operator-declared**
  model facts (context windows, pricing). Market facts rot; the registry
  therefore ships empty of them and *requires provenance*: a spec carrying
  prices must carry ``pricing_as_of``. No price, context window, or
  capability is hard-coded here that the repository cannot verify.

This module is intentionally importable without torch (it is pure
configuration); the adapter layer is only touched by the test-time drift
gate.

Example:
    registry = LLMModelRegistry()
    registry.register(
        LLMModelSpec(
            provider="ollama",
            model_id="llama3.2:3b",
            context_window=8_192,
            capabilities=frozenset({"chat", "tool_use"}),
            notes="local open-weights model; no per-token cost",
        )
    )
    spec = registry.select_one(
        required_capabilities=("chat",),
        min_context=8_000,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "KNOWN_CAPABILITIES",
    "PROVIDER_CATALOG",
    "LLMModelRegistry",
    "LLMModelSpec",
    "ProviderFacts",
]

# Vetted capability vocabulary. Registration validates against this set so a
# typo ("tooluse") cannot silently make a model unselectable.
KNOWN_CAPABILITIES: frozenset[str] = frozenset(
    {
        "chat",
        "tool_use",
        "vision",
        "json_mode",
        "streaming",
        "embeddings",
    }
)


@dataclass(frozen=True)
class ProviderFacts:
    """Code-grounded facts about one shipped provider adapter.

    Every field is verifiable in-tree against the adapter implementations;
    the drift gate in ``tests/models/test_llm_registry.py`` pins this
    catalog to ``IMPLEMENTED_LLM_PROVIDERS``.

    Attributes:
        wire_format: Request/response wire format the adapter speaks.
        api_key_env_var: Environment variable the adapter reads the key
            from, or ``None`` when no key is involved.
        locality: ``"cloud"``, ``"local"`` (loopback-gated), or
            ``"builtin"`` (no model at all).
        requires_explicit_base_url: True when the adapter refuses to guess
            an endpoint (``CursorAdapter``).
        reports_token_usage: True when the adapter parses provider-reported
            token counts into the usage ledger (see ``llm_usage.py``);
            False marks calls that are recorded as unmetered or, for
            ``template``, not LLM calls at all.
    """

    wire_format: str
    api_key_env_var: str | None
    locality: str
    requires_explicit_base_url: bool
    reports_token_usage: bool


# Facts below mirror models/foundation/ollama_adapter.py adapter by adapter:
# the env var each __init__ reads, the endpoint family generate() posts to,
# the loopback gate (Ollama), the explicit-base_url refusal (Cursor), and
# which response payloads carry usage the adapters now record.
PROVIDER_CATALOG: dict[str, ProviderFacts] = {
    "openai": ProviderFacts(
        wire_format="openai-chat-completions",
        api_key_env_var="OPENAI_API_KEY",
        locality="cloud",
        requires_explicit_base_url=False,
        reports_token_usage=True,
    ),
    "anthropic": ProviderFacts(
        wire_format="anthropic-messages",
        api_key_env_var="ANTHROPIC_API_KEY",
        locality="cloud",
        requires_explicit_base_url=False,
        reports_token_usage=True,
    ),
    "huggingface": ProviderFacts(
        wire_format="hf-inference-text-generation",
        api_key_env_var="HUGGINGFACE_API_KEY",
        locality="cloud",
        requires_explicit_base_url=False,
        # The Inference API text-generation route returns no usage block;
        # calls are recorded as unmetered (reported=False).
        reports_token_usage=False,
    ),
    "ollama": ProviderFacts(
        wire_format="ollama-generate-chat",
        api_key_env_var=None,
        locality="local",
        requires_explicit_base_url=False,
        reports_token_usage=True,
    ),
    "template": ProviderFacts(
        wire_format="builtin-templates",
        api_key_env_var=None,
        locality="builtin",
        requires_explicit_base_url=False,
        reports_token_usage=False,
    ),
    "xai": ProviderFacts(
        wire_format="openai-chat-completions",
        api_key_env_var="XAI_API_KEY",
        locality="cloud",
        requires_explicit_base_url=False,
        reports_token_usage=True,
    ),
    "gemini": ProviderFacts(
        wire_format="gemini-generate-content",
        api_key_env_var="GEMINI_API_KEY",
        locality="cloud",
        requires_explicit_base_url=False,
        reports_token_usage=True,
    ),
    "cohere": ProviderFacts(
        wire_format="cohere-chat-v2",
        api_key_env_var="COHERE_API_KEY",
        locality="cloud",
        requires_explicit_base_url=False,
        reports_token_usage=True,
    ),
    "deepseek": ProviderFacts(
        wire_format="openai-chat-completions",
        api_key_env_var="DEEPSEEK_API_KEY",
        locality="cloud",
        requires_explicit_base_url=False,
        reports_token_usage=True,
    ),
    "cursor": ProviderFacts(
        wire_format="openai-chat-completions",
        api_key_env_var="CURSOR_API_KEY",
        locality="cloud",
        requires_explicit_base_url=True,
        reports_token_usage=True,
    ),
}


@dataclass(frozen=True)
class LLMModelSpec:
    """Operator-declared facts about one selectable model.

    Attributes:
        provider: Provider label; must be a key of :data:`PROVIDER_CATALOG`.
        model_id: Provider-side model identifier.
        context_window: Maximum context length in tokens (must be positive).
        capabilities: Subset of :data:`KNOWN_CAPABILITIES`.
        max_output_tokens: Provider output cap, when known.
        input_cost_per_mtok: USD per million input tokens, when declared.
        output_cost_per_mtok: USD per million output tokens, when declared.
        pricing_as_of: ISO date the prices were checked. **Required when
            any price is declared** — prices without provenance are how
            documentation rots.
        notes: Free-form operator notes.
    """

    provider: str
    model_id: str
    context_window: int
    capabilities: frozenset[str] = frozenset()
    max_output_tokens: int | None = None
    input_cost_per_mtok: float | None = None
    output_cost_per_mtok: float | None = None
    pricing_as_of: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        """Validate provider, capabilities, context, and price provenance."""
        if self.provider not in PROVIDER_CATALOG:
            raise ValueError(
                f"unknown provider {self.provider!r}; shipped adapters: "
                f"{sorted(PROVIDER_CATALOG)}"
            )
        if not self.model_id:
            raise ValueError("model_id must be a non-empty string")
        if self.context_window <= 0:
            raise ValueError(f"context_window must be positive, got {self.context_window}")
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError(f"max_output_tokens must be positive, got {self.max_output_tokens}")
        unknown = self.capabilities - KNOWN_CAPABILITIES
        if unknown:
            raise ValueError(
                f"unknown capabilities {sorted(unknown)}; vetted vocabulary: "
                f"{sorted(KNOWN_CAPABILITIES)}"
            )
        has_price = self.input_cost_per_mtok is not None or self.output_cost_per_mtok is not None
        for name in ("input_cost_per_mtok", "output_cost_per_mtok"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if has_price:
            if self.pricing_as_of is None:
                raise ValueError(
                    "declared prices require pricing_as_of (ISO date) — prices "
                    "without provenance are not registrable"
                )
            date.fromisoformat(self.pricing_as_of)  # raises ValueError if malformed

    @property
    def key(self) -> str:
        """Stable registry key, ``provider:model_id``."""
        return f"{self.provider}:{self.model_id}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "context_window": self.context_window,
            "capabilities": sorted(self.capabilities),
            "max_output_tokens": self.max_output_tokens,
            "input_cost_per_mtok": self.input_cost_per_mtok,
            "output_cost_per_mtok": self.output_cost_per_mtok,
            "pricing_as_of": self.pricing_as_of,
            "notes": self.notes,
        }


def _is_local(spec: LLMModelSpec) -> bool:
    """Whether the spec's provider runs locally (loopback) or in-process."""
    return PROVIDER_CATALOG[spec.provider].locality in ("local", "builtin")


def _effective_input_cost(spec: LLMModelSpec) -> float | None:
    """Input $/MTok used for budgeting and ordering.

    A declared price always wins. Otherwise a ``local``/``builtin`` model is
    genuinely free (``0.0``) — local inference has no per-token charge — while
    an *undeclared cloud* price stays unknown (``None``) so it is excluded from
    budget queries rather than assumed free. This is what makes free/local the
    baseline preference: a free local model beats any paid cloud on cost and
    wins ties.
    """
    if spec.input_cost_per_mtok is not None:
        return spec.input_cost_per_mtok
    if _is_local(spec):
        return 0.0
    return None


@dataclass
class LLMModelRegistry:
    """Instance-owned registry of selectable LLM model specs.

    Unlike ``SOTARegistry`` (a class-level catalog of in-tree model code),
    model specs are *deployment configuration* — context windows and prices
    belong to the operator's moment in time, not to the repository — so the
    registry is instance-based and starts empty.
    """

    _specs: dict[str, LLMModelSpec] = field(default_factory=dict)

    def register(self, spec: LLMModelSpec, *, replace: bool = False) -> None:
        """Register a model spec.

        Args:
            spec: Validated model spec.
            replace: Allow overwriting an existing ``provider:model_id``
                entry. Without it, re-registration raises so conflicting
                declarations cannot shadow each other silently.

        Raises:
            ValueError: If the key is already registered and ``replace`` is
                False.
        """
        if spec.key in self._specs and not replace:
            raise ValueError(f"{spec.key} already registered (pass replace=True to overwrite)")
        self._specs[spec.key] = spec

    def get(self, provider: str, model_id: str) -> LLMModelSpec:
        """Return the spec for ``provider:model_id``.

        Raises:
            KeyError: If not registered.
        """
        key = f"{provider}:{model_id}"
        if key not in self._specs:
            raise KeyError(f"{key} is not registered; known: {sorted(self._specs)}")
        return self._specs[key]

    def list_models(self) -> list[str]:
        """Return all registered keys, sorted."""
        return sorted(self._specs)

    def __len__(self) -> int:
        """Number of registered specs."""
        return len(self._specs)

    def select(
        self,
        *,
        required_capabilities: tuple[str, ...] = (),
        min_context: int = 0,
        max_input_cost_per_mtok: float | None = None,
        providers: tuple[str, ...] | None = None,
    ) -> list[LLMModelSpec]:
        """Return all specs satisfying the requirements, best-first.

        Filtering is conjunctive. **Free and local first** is the ordering rule:
        a ``local``/``builtin`` model is treated as genuinely free (cost ``0``),
        so it sorts ahead of any paid cloud model and wins ties. Ordering is
        deterministic: known cost ascending (free local leads at ``0``), then
        local/builtin ahead of cloud on equal cost, then key.

        When ``max_input_cost_per_mtok`` is given, local/builtin models always
        qualify (they are free); a cloud spec with **no declared price** cannot
        honestly satisfy a budget, so it is excluded (unknown cost, sorted
        last) rather than assumed free.

        Args:
            required_capabilities: Capabilities every result must declare.
            min_context: Minimum context window in tokens.
            max_input_cost_per_mtok: Input-price budget (USD per MTok).
            providers: Restrict to these provider labels.

        Raises:
            ValueError: If a required capability is outside the vetted
                vocabulary (a typo would otherwise just return ``[]``).
        """
        unknown = set(required_capabilities) - KNOWN_CAPABILITIES
        if unknown:
            raise ValueError(
                f"unknown required capabilities {sorted(unknown)}; vetted "
                f"vocabulary: {sorted(KNOWN_CAPABILITIES)}"
            )

        results = []
        for spec in self._specs.values():
            if providers is not None and spec.provider not in providers:
                continue
            if spec.context_window < min_context:
                continue
            if not set(required_capabilities) <= spec.capabilities:
                continue
            if max_input_cost_per_mtok is not None:
                effective_cost = _effective_input_cost(spec)
                if effective_cost is None:
                    # Undeclared cloud price: unknown, cannot honestly satisfy a
                    # budget. (Local/builtin are free and never reach here.)
                    continue
                if effective_cost > max_input_cost_per_mtok:
                    continue
            results.append(spec)

        def _ordering(spec: LLMModelSpec) -> tuple[bool, float, int, str]:
            effective_cost = _effective_input_cost(spec)
            return (
                effective_cost is None,  # unknown-cost (undeclared cloud) last
                effective_cost if effective_cost is not None else 0.0,  # cheapest first
                0 if _is_local(spec) else 1,  # local/free ahead of cloud on ties
                spec.key,  # deterministic final tiebreak
            )

        results.sort(key=_ordering)
        return results

    def select_one(self, **criteria: Any) -> LLMModelSpec:
        """Return the single best spec for the criteria.

        Args:
            **criteria: Forwarded to :meth:`select`.

        Raises:
            LookupError: If nothing satisfies the criteria; the message
                restates them so the caller can see what to relax.
        """
        matches = self.select(**criteria)
        if not matches:
            raise LookupError(
                f"no registered model satisfies {criteria!r} among " f"{self.list_models()}"
            )
        return matches[0]
