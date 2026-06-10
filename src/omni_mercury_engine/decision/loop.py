# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The closed ``identify -> interpret -> decide -> deter -> verify`` loop.

:class:`DecisionLoop` runs a detection result through the decision/abstention
gate (:class:`~omni_mercury_engine.decision.decider.DecisionAbstentionResponder`)
and then *verifies* the pass by appending the resulting
:class:`~omni_mercury_engine.decision.record.DecisionRecord` to a
:class:`~omni_mercury_engine.decision.ledger.DecisionLedger` and offering it to
an optional feedback sink -- the seam through which outcomes can flow back to
calibration, learning, or a human queue (the omnidirectional, closed-loop
property).

The responder's :meth:`decide` stays a pure function of the certificate; the
loop is the small stateful shell that adds the "verify" step around it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.decision.decider import DecisionAbstentionResponder
from omni_mercury_engine.decision.ledger import DecisionLedger

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from omni_mercury_engine.decision.record import DecisionRecord

#: A sink invoked with every completed decision -- the feedback seam back to
#: calibration / learning / a human queue.  It must not raise; the loop's job is
#: done once the ledger holds the record.
FeedbackSink = Callable[["DecisionRecord"], None]


class DecisionLoop:
    """Decide -> deter -> verify over detection results, with an audit ledger.

    Args:
        responder: The decision/abstention gate (defaults to a fresh
            :class:`DecisionAbstentionResponder`).
        ledger: The audit ledger the loop appends to (defaults to a fresh,
            unbounded :class:`DecisionLedger`).
        feedback: Optional sink invoked with each completed decision.
    """

    def __init__(
        self,
        responder: DecisionAbstentionResponder | None = None,
        *,
        ledger: DecisionLedger | None = None,
        feedback: FeedbackSink | None = None,
    ) -> None:
        """Wire a responder, an audit ledger, and an optional feedback sink."""
        self.responder = responder or DecisionAbstentionResponder()
        self.ledger = ledger if ledger is not None else DecisionLedger()
        self.feedback = feedback

    def step(
        self,
        detection_result: Mapping[str, Any],
        *,
        domain: str | None = None,
    ) -> DecisionRecord:
        """Decide on one detection result, record it, and fan out feedback.

        Args:
            detection_result: A ``detect_with_fusion``-style result mapping.
            domain: Optional domain hint.

        Returns:
            The :class:`DecisionRecord` for this pass (already recorded).
        """
        record = self.responder.decide(detection_result, domain=domain)
        self.ledger.record(record)
        if self.feedback is not None:
            self.feedback(record)
        return record

    def run(
        self,
        detection_results: Iterable[Mapping[str, Any]],
        *,
        domain: str | None = None,
    ) -> list[DecisionRecord]:
        """Run :meth:`step` over an iterable of results, returning the records."""
        return [self.step(result, domain=domain) for result in detection_results]

    def summary(self) -> dict[str, Any]:
        """Return the ledger's aggregate summary over everything seen so far."""
        return self.ledger.summary()


__all__ = ["DecisionLoop", "FeedbackSink"]
