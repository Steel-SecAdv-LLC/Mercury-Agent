# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Typed inputs and outputs for Mercury's reasoning layer.

These shapes carry no provider detail. A :class:`ReasoningContext` states
*what Mercury wants reasoned about*; :class:`Explanation`, :class:`Hypothesis`,
and :class:`Report` state *what came back*, each stamped with provenance
(which backend, which model, whether Mercury's dual ethical gate cleared it).
The backend that produced them is an implementation detail Mercury swaps
without any of these shapes changing — the contract belongs to Mercury, not to
whatever language model happens to serve a call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["Explanation", "Hypothesis", "ReasoningContext", "Report"]


@dataclass(frozen=True)
class ReasoningContext:
    """A request for reasoning, framed in Mercury's own terms.

    Attributes:
        summary: One-line statement of what needs reasoning about.
        domain: Mercury domain hint; sanitized at the ethical boundary.
        evidence: Structured signals/findings Mercury already computed.
        severity: Per-call severity in ``[0, 1]`` (feeds the ethical gate).
        anomaly_prob: Per-call anomaly probability in ``[0, 1]``.
    """

    summary: str
    domain: str = "general"
    evidence: dict[str, Any] = field(default_factory=dict)
    severity: float = 0.0
    anomaly_prob: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return {
            "summary": self.summary,
            "domain": self.domain,
            "evidence": dict(self.evidence),
            "severity": self.severity,
            "anomaly_prob": self.anomaly_prob,
        }


@dataclass(frozen=True)
class Explanation:
    """A natural-language explanation of a Mercury finding.

    Attributes:
        text: The explanation prose.
        backend: Provenance label of the backend that produced it.
        model: Model identifier the backend served.
        gated: True when Mercury's dual ethical gate governed the call.
    """

    text: str
    backend: str
    model: str
    gated: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return {
            "text": self.text,
            "backend": self.backend,
            "model": self.model,
            "gated": self.gated,
        }


@dataclass(frozen=True)
class Hypothesis:
    """A single proposed explanation for the cognitive engine to weigh.

    Attributes:
        statement: The hypothesis itself.
        rationale: Why it is being proposed.
        confidence: Backend-reported confidence in ``[0, 1]`` (0 when unknown).
    """

    statement: str
    rationale: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return {
            "statement": self.statement,
            "rationale": self.rationale,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class Report:
    """A synthesized natural-language report over Mercury findings.

    Attributes:
        title: Report title.
        body: Report body prose.
        backend: Provenance label of the backend that produced it.
        model: Model identifier the backend served.
        gated: True when Mercury's dual ethical gate governed the call.
    """

    title: str
    body: str
    backend: str
    model: str
    gated: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return {
            "title": self.title,
            "body": self.body,
            "backend": self.backend,
            "model": self.model,
            "gated": self.gated,
        }
