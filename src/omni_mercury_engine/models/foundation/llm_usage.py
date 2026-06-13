# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Provider-reported LLM token usage accounting.

Multi-provider operation is uncosted without token accounting: Mercury's
LLM adapters previously parsed only the generated text out of each provider
response and discarded the usage block the provider sent alongside it. This
module supplies the two pieces every adapter now feeds:

* :class:`LLMUsage` — one immutable record per successful generation,
  carrying the token counts **as reported by the provider's own response
  payload**. Counts are never estimated client-side: an adapter whose
  provider reports no usage (e.g. the HuggingFace Inference API
  text-generation route) records ``reported=False`` with ``None`` counts,
  so unmetered spend is visible instead of silently absent.
* :class:`UsageLedger` — a thread-safe aggregator. Aggregate totals are
  exact over the ledger's lifetime (running counters, O(1) to read);
  the per-call history is a bounded ring so long-running processes do not
  grow without bound.

Design notes:
    No process-global ledger is created here. Callers own ledger instances
    and attach them to adapters (``BaseLLMAdapter.attach_usage_ledger``) or
    thread one through a :class:`~omni_mercury_engine.models.foundation.
    ollama_adapter.FallbackLLMChain` — mirroring how the decision layer's
    ``DecisionLedger`` is wired explicitly rather than ambiently.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["LLMUsage", "UsageLedger"]


@dataclass(frozen=True)
class LLMUsage:
    """One provider-reported usage record for a single generation call.

    Attributes:
        provider: Provider label (an ``LLMProvider`` value, e.g. ``"openai"``).
        model: Model identifier the call was made with.
        prompt_tokens: Input-side tokens as reported by the provider, or
            ``None`` when the provider did not report them.
        completion_tokens: Output-side tokens as reported, or ``None``.
        total_tokens: Provider-reported total. When the provider reports
            prompt and completion but no total, the sum is used (still
            provider-derived, not estimated).
        reported: True when the provider's response carried any usage
            information. False marks an *unmetered* call — the call
            happened, but its cost is unknown — which the ledger surfaces
            separately rather than counting as zero.
    """

    provider: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reported: bool = True

    def __post_init__(self) -> None:
        """Validate counts and reconcile the total."""
        if not self.provider:
            raise ValueError("provider must be a non-empty string")
        if not self.model:
            raise ValueError("model must be a non-empty string")
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative int or None, got {value!r}")
        if (
            self.total_tokens is None
            and self.prompt_tokens is not None
            and self.completion_tokens is not None
        ):
            # Provider reported both sides but no total: derive it from the
            # provider's own numbers (frozen dataclass -> object.__setattr__).
            object.__setattr__(self, "total_tokens", self.prompt_tokens + self.completion_tokens)
        if not self.reported and any(
            getattr(self, name) is not None
            for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        ):
            raise ValueError("reported=False requires all token counts to be None")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return asdict(self)


@dataclass
class _Aggregate:
    """Running counters for one ``(provider, model)`` key."""

    calls: int = 0
    unreported_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, usage: LLMUsage) -> None:
        self.calls += 1
        if not usage.reported:
            self.unreported_calls += 1
        self.prompt_tokens += usage.prompt_tokens or 0
        self.completion_tokens += usage.completion_tokens or 0
        self.total_tokens += usage.total_tokens or 0

    def to_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "unreported_calls": self.unreported_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class UsageLedger:
    """Thread-safe accumulator of :class:`LLMUsage` records.

    Aggregate totals are exact over everything ever recorded (running
    counters; recording and reading are O(1) / O(#keys)). The per-call
    history kept for inspection is a bounded ring of the most recent
    ``max_recent`` records — totals are **not** affected by the bound.

    Attributes:
        max_recent: Capacity of the recent-records ring.
    """

    max_recent: int = 4096
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _recent: deque[LLMUsage] = field(init=False, repr=False)
    _by_model: dict[tuple[str, str], _Aggregate] = field(default_factory=dict, repr=False)
    _recorded: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        """Initialize the bounded recent-records ring."""
        if self.max_recent < 1:
            raise ValueError(f"max_recent must be >= 1, got {self.max_recent}")
        self._recent = deque(maxlen=self.max_recent)

    def record(self, usage: LLMUsage) -> None:
        """Record one usage entry (thread-safe)."""
        with self._lock:
            self._recent.append(usage)
            self._by_model.setdefault((usage.provider, usage.model), _Aggregate()).add(usage)
            self._recorded += 1

    def totals(self) -> dict[str, int]:
        """Return exact lifetime totals across all providers and models."""
        with self._lock:
            out = _Aggregate()
            for agg in self._by_model.values():
                out.calls += agg.calls
                out.unreported_calls += agg.unreported_calls
                out.prompt_tokens += agg.prompt_tokens
                out.completion_tokens += agg.completion_tokens
                out.total_tokens += agg.total_tokens
            return out.to_dict()

    def totals_by_model(self) -> dict[tuple[str, str], dict[str, int]]:
        """Return exact lifetime totals keyed by ``(provider, model)``."""
        with self._lock:
            return {key: agg.to_dict() for key, agg in self._by_model.items()}

    def recent(self, n: int | None = None) -> list[LLMUsage]:
        """Return the most recent records (newest last), up to ``n``."""
        with self._lock:
            items = list(self._recent)
        return items if n is None else items[-n:]

    def __len__(self) -> int:
        """Total number of records ever recorded (not the ring size)."""
        with self._lock:
            return self._recorded
