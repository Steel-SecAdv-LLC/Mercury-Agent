"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.

Decision / Abstention / Response layer.

This package closes the ``identify -> interpret -> decide -> deter -> verify``
loop on top of the engine's calibrated detection. It converts a calibrated
``P(anomaly)`` (and, when present, a conformal label set carrying a
distribution-free coverage guarantee) into a *typed decision with a first-class
"don't-know"* (:class:`AbstentionPolicy`), then into a *proportionate,
reversible-by-default, ethically-gated response* (:class:`ResponsePlanner` /
:class:`ResponseActuator`), and records every pass to an append-only audit
ledger (:class:`DecisionResponseLoop` / :class:`AuditLedger`).

Design commitments (all enforced in code and tests):

* **Honest abstention.** ``ABSTAIN`` is a real verdict that maps to
  :attr:`~omni_mercury_engine.verifiers.three_state.ThreeState.UNAVAILABLE`,
  reusing the cross-repo honesty contract -- never coerced to a binary call.
* **Verifiable-only.** Decisions are pure functions of their inputs; every pass
  carries full provenance and is JSON-serialisable.
* **Reversible-by-default, fail-closed.** Abstentions never deter; irreversible /
  escalatory actions require explicit human authorization; every effectful action
  passes the ethical gate before actuation.
"""

from __future__ import annotations

from omni_mercury_engine.decision.abstention import (
    POLICY_VERSION,
    AbstentionPolicy,
    AbstentionThresholds,
)
from omni_mercury_engine.decision.confidence import (
    ConfidenceSignal,
    ConfidenceSource,
    confidence_batch_from_conformal_scores,
    confidence_from_conformal,
    confidence_from_engine_result,
)
from omni_mercury_engine.decision.loop import (
    AuditLedger,
    DecisionResponseLoop,
    FeedbackSink,
)
from omni_mercury_engine.decision.response import (
    Authorization,
    EthicalGate,
    ResponseActuator,
    ResponsePlanner,
    ResponseVetoError,
    deny_all_gate,
    permit_all_gate,
    threat_level_from_score,
)
from omni_mercury_engine.decision.types import (
    Decision,
    LoopResult,
    ResponseAction,
    ResponseOutcome,
    ResponseStatus,
    ResponseTier,
    Verdict,
    verdict_to_three_state,
)

__all__ = [
    "POLICY_VERSION",
    "AbstentionPolicy",
    "AbstentionThresholds",
    "AuditLedger",
    "Authorization",
    "ConfidenceSignal",
    "ConfidenceSource",
    "Decision",
    "DecisionResponseLoop",
    "EthicalGate",
    "FeedbackSink",
    "LoopResult",
    "ResponseAction",
    "ResponseActuator",
    "ResponseOutcome",
    "ResponsePlanner",
    "ResponseStatus",
    "ResponseTier",
    "ResponseVetoError",
    "Verdict",
    "confidence_batch_from_conformal_scores",
    "confidence_from_conformal",
    "confidence_from_engine_result",
    "deny_all_gate",
    "permit_all_gate",
    "threat_level_from_score",
    "verdict_to_three_state",
]
