# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mercury-owned reasoning backend interface.

Mercury Agent is the agent and the brain of record: its OODA loop,
neuro-symbolic detection, and dual ethical gates own the control flow. A
:class:`ReasoningBackend` is a *called dependency* — a swappable language /
reasoning engine Mercury invokes from inside that loop for explanation,
hypothesis proposal, and report synthesis. It is never the front of the
system, and Mercury is never a wrapper around it.

Two invariants hold for every backend, enforced here in the base class so a
concrete implementation cannot skip them:

* **Identity-preserving by construction** — the public methods speak Mercury's
  vocabulary (:mod:`omni_mercury_engine.reasoning.schemas`); no provider name
  appears in any signature.
* **Governed by Mercury's ethics, not the model's** — every reasoning
  operation passes Mercury's benevolence + σ_Immutable dual hard gate
  (:func:`enforce_dual_ethical_gate`) at the reasoning boundary before any
  generated content is surfaced. The gate fails closed: a violation raises
  :class:`~omni_mercury_engine.cognitive.ethical_bounding.EthicalConstraintViolationError`
  and no output is returned. The backend does not get to bypass Mercury's
  governance.
"""

from __future__ import annotations

import abc
from typing import Any

from omni_mercury_engine.cognitive.ethical_bounding import (
    MINIMUM_BENEVOLENCE_FLOOR,
    BenevolenceScorer,
    sanitize_domain,
)
from omni_mercury_engine.models.foundation.llm_adapter import MERCURY_IDENTITY_CLAUSE
from omni_mercury_engine.reasoning.schemas import (
    Explanation,
    Hypothesis,
    ReasoningContext,
    Report,
)
from omni_mercury_engine.security.sigma_immutable_gate import (
    SigmaImmutableGate,
    enforce_dual_ethical_gate,
    get_sigma_immutable_gate,
)

__all__ = ["ReasoningBackend"]

#: System prompt prepended to every call — the product-identity contract
#: (Mercury Agent speaks as Mercury Agent, backend disclosed on inquiry;
#: see MERCURY_IDENTITY_CLAUSE) plus the subordination contract: the model
#: is a subordinate engine, not the system. Mercury owns the decision.
SYSTEM_PROMPT: str = (
    MERCURY_IDENTITY_CLAUSE + " "
    "You are a subordinate reasoning engine invoked by Mercury Agent. "
    "Mercury owns the detection, the decision, and the final verdict; you "
    "provide concise, evidence-grounded language only. Do not claim to be "
    "the system or to make the decision yourself."
)

#: Positive-purpose keywords evidencing the defensive, truth-dense intent of
#: the reasoning boundary (mirrors the narrative-voice gate contract).
_PURPOSE_KEYWORDS: str = (
    "audit verify inform protect explain evidence fair oversight "
    "transparency care help support honesty"
)


class ReasoningBackend(abc.ABC):
    """Abstract reasoning engine Mercury calls; never the front of the system.

    Subclasses implement only the provider-specific :meth:`_generate` plus the
    :attr:`name`, :attr:`model`, and :attr:`is_offline` provenance properties.
    The governed public surface (:meth:`explain`, :meth:`propose_hypotheses`,
    :meth:`synthesize_report`) is final here so the ethics gate cannot be
    skipped by an implementation.
    """

    def __init__(
        self,
        *,
        ethics_enabled: bool = True,
        benevolence_scorer: Any | None = None,
        sigma_gate: Any | None = None,
    ) -> None:
        """Initialize the backend and bind Mercury's ethical gates.

        Args:
            ethics_enabled: When True (default), every reasoning operation is
                run through Mercury's dual hard ethical gate before output is
                surfaced. Disable only in trusted, offline tests.
            benevolence_scorer: Override the benevolence gate (dependency
                injection for tests); defaults to a floor-clamped
                :class:`BenevolenceScorer`.
            sigma_gate: Override the σ_Immutable gate (dependency injection
                for tests); defaults to the process-wide gate.
        """
        self.ethics_enabled = ethics_enabled
        self._benevolence_scorer = (
            benevolence_scorer
            if benevolence_scorer is not None
            else BenevolenceScorer(benevolence_threshold=MINIMUM_BENEVOLENCE_FLOOR)
        )
        self._sigma_gate: SigmaImmutableGate = (
            sigma_gate if sigma_gate is not None else get_sigma_immutable_gate()
        )

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short backend label for provenance (e.g. ``"local"``)."""

    @property
    @abc.abstractmethod
    def model(self) -> str:
        """Model identifier currently served by this backend."""

    @property
    @abc.abstractmethod
    def is_offline(self) -> bool:
        """True iff this backend is guaranteed to make no external network call."""

    @abc.abstractmethod
    def _generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate text for a built prompt (provider-specific).

        Never invoked before the ethical gate has cleared the operation.

        Args:
            prompt: The fully-built user prompt.
            system_prompt: Optional system prompt; defaults to
                :data:`SYSTEM_PROMPT` at the call sites.

        Returns:
            The generated text.
        """

    def _enforce(self, boundary: str, context: ReasoningContext) -> None:
        """Run Mercury's dual hard ethical gate for one reasoning operation.

        Mirrors the narrative-voice contract: benevolence first, σ_Immutable
        second, both fail-closed. On violation the call raises and no
        generated content is surfaced.

        Args:
            boundary: Fully-qualified boundary name for the audit trail.
            context: The reasoning request whose domain/severity govern the gate.
        """
        if not self.ethics_enabled:
            return
        safe_domain = sanitize_domain(context.domain)
        action = f"reasoning_backend:{safe_domain}:{_PURPOSE_KEYWORDS}"
        gate_context = {
            "purpose": "truth-dense reasoning over Mercury detections",
            "safety": "inform protect verify transparency evidence",
            "domain": safe_domain,
        }
        enforce_dual_ethical_gate(
            benevolence_scorer=self._benevolence_scorer,
            sigma_gate=self._sigma_gate,
            action=action,
            context=gate_context,
            boundary=boundary,
            domain=safe_domain,
            severity=context.severity,
            anomaly_prob=context.anomaly_prob,
            extra_details={"backend": self.name, "model": self.model},
        )

    def explain(self, context: ReasoningContext) -> Explanation:
        """Explain a Mercury finding in natural language.

        Args:
            context: The finding to explain.

        Returns:
            A provenance-stamped :class:`Explanation`.

        Raises:
            EthicalConstraintViolationError: If the dual ethical gate blocks
                the operation; no content is returned in that case.
        """
        self._enforce("reasoning_backend.explain", context)
        safe_domain = sanitize_domain(context.domain)
        prompt = (
            "Explain the following Mercury finding for an analyst, grounded in "
            "the evidence and no longer than a short paragraph.\n"
            f"Domain: {safe_domain}\n"
            f"Summary: {context.summary}\n"
            f"Evidence: {context.evidence}"
        )
        text = self._generate(prompt, SYSTEM_PROMPT)
        return Explanation(
            text=text, backend=self.name, model=self.model, gated=self.ethics_enabled
        )

    def propose_hypotheses(self, evidence: ReasoningContext) -> list[Hypothesis]:
        """Propose candidate hypotheses for Mercury's cognitive engine to weigh.

        Args:
            evidence: The evidence to reason over.

        Returns:
            A list of :class:`Hypothesis`, one per non-empty generated line.

        Raises:
            EthicalConstraintViolationError: If the dual ethical gate blocks
                the operation; no content is returned in that case.
        """
        self._enforce("reasoning_backend.propose_hypotheses", evidence)
        safe_domain = sanitize_domain(evidence.domain)
        prompt = (
            "Propose candidate hypotheses (one per line) that could account "
            "for the following Mercury evidence. Mercury, not you, will weigh "
            "and decide among them.\n"
            f"Domain: {safe_domain}\n"
            f"Summary: {evidence.summary}\n"
            f"Evidence: {evidence.evidence}"
        )
        text = self._generate(prompt, SYSTEM_PROMPT)
        return [
            Hypothesis(statement=line.strip(), rationale="proposed by reasoning backend")
            for line in text.splitlines()
            if line.strip()
        ]

    def synthesize_report(self, findings: ReasoningContext) -> Report:
        """Synthesize a natural-language report over Mercury findings.

        Args:
            findings: The findings to report on.

        Returns:
            A provenance-stamped :class:`Report`.

        Raises:
            EthicalConstraintViolationError: If the dual ethical gate blocks
                the operation; no content is returned in that case.
        """
        self._enforce("reasoning_backend.synthesize_report", findings)
        safe_domain = sanitize_domain(findings.domain)
        prompt = (
            "Synthesize a brief analyst report over the following Mercury "
            "findings, grounded strictly in the evidence.\n"
            f"Domain: {safe_domain}\n"
            f"Summary: {findings.summary}\n"
            f"Evidence: {findings.evidence}"
        )
        text = self._generate(prompt, SYSTEM_PROMPT)
        title = f"Mercury reasoning report: {safe_domain}"
        return Report(
            title=title, body=text, backend=self.name, model=self.model, gated=self.ethics_enabled
        )
