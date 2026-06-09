# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Decision / abstention / response layer.

This package closes Mercury's loop from *interpret* to *deter*.  The engine's
calibrated detection certificate (calibrated probability + conformal coverage
set + ethical-gate verdict + neuro-symbolic agreement) goes in; a single
auditable :class:`DecisionRecord` comes out, carrying either a grounded label
or an explicit, principled **abstention** -- a "don't-know" gate split, using
the engine's own :class:`ThreeState` invariant, into a *resolvable* deferral
(``UNAVAILABLE``) and a *fail-closed* hold (``UNDECIDABLE``).  Each record is
paired with a bounded, non-destructive :class:`ResponsePlan` (notify /
recommend reversible countermeasures / escalate to a human / hold) so the loop
``identify -> interpret -> decide -> deter`` is genuinely closed without ever
authorising a destructive autonomous action.

The whole layer is pure Python and deterministic -- the same certificate always
yields the same record -- which is what makes the closed loop verifiable.

Typical use::

    from omni_mercury_engine.decision import DecisionAbstentionResponder

    responder = DecisionAbstentionResponder()
    record = responder.decide(engine.detect_with_fusion(x), domain="security")
    if record.abstained:
        ...  # route record.response to a human
    print(record.explain())

Or wired into the engine so every detection carries a ``"decision"`` key::

    engine.enable_decision_layer()
    result = engine.detect_with_fusion(x, domain="security")
    result["decision"]  # the DecisionRecord, as a dict
"""

from __future__ import annotations

from omni_mercury_engine.decision.bridge import to_agent_action, to_cap_alert
from omni_mercury_engine.decision.decider import DecisionAbstentionResponder
from omni_mercury_engine.decision.evidence import Evidence
from omni_mercury_engine.decision.policy import DecisionPolicy
from omni_mercury_engine.decision.record import DecisionRecord
from omni_mercury_engine.decision.response import ResponsePlan, ResponsePolicy
from omni_mercury_engine.decision.states import Disposition, ResponseAction
from omni_mercury_engine.verifiers.three_state import ThreeState

__all__ = [
    "DecisionAbstentionResponder",
    "DecisionPolicy",
    "DecisionRecord",
    "Disposition",
    "Evidence",
    "ResponseAction",
    "ResponsePlan",
    "ResponsePolicy",
    "ThreeState",
    "to_agent_action",
    "to_cap_alert",
]
