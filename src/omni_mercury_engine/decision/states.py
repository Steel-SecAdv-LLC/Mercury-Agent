# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Operational vocabulary for the decision / abstention / response layer.

The layer closes the loop ``identify -> interpret -> decide -> deter`` on top of
the calibrated detection certificate.  Two small, stable enums name the two
axes a closed loop needs:

* :class:`Disposition` -- the *operational stance* the loop takes on an event
  (act, clear, defer, hold).
* :class:`ResponseAction` -- the *bounded, non-destructive* response the loop
  recommends.

The transparency axis -- *did we actually decide?* -- is **not** re-invented here.
It reuses the engine's own cross-component invariant
:class:`omni_mercury_engine.verifiers.three_state.ThreeState`
(``GROUNDED`` / ``UNAVAILABLE`` / ``UNDECIDABLE``).  A :class:`Disposition`
is the action-oriented projection of a :class:`~...three_state.ThreeState`:

==================  =====================================================
``ThreeState``      :class:`Disposition`
==================  =====================================================
``GROUNDED``        :attr:`Disposition.ACT` (anomaly) /
                    :attr:`Disposition.CLEAR` (normal)
``UNAVAILABLE``     :attr:`Disposition.DEFER` (a resolvable "don't-know")
``UNDECIDABLE``     :attr:`Disposition.HOLD` (a fail-closed "don't-know")
==================  =====================================================

Every string value is the serialised wire format; pin them in tests so a
rename is deliberate.
"""

from __future__ import annotations

from enum import Enum


class Disposition(Enum):
    """The operational stance the closed loop takes on a single event.

    Mutually exclusive and exhaustive.  ``ACT`` and ``CLEAR`` are the two
    *grounded* stances (a label was decided); ``DEFER`` and ``HOLD`` are the
    two *abstaining* stances -- the explicit "don't-know" gate, split by
    whether the indecision is resolvable.
    """

    #: A calibrated, grounded anomaly call.  The loop recommends an active
    #: (but non-destructive) response and, above a severity bar, a human.
    ACT = "act"

    #: A calibrated, grounded "normal" call.  The loop stays passive (monitor).
    CLEAR = "clear"

    #: A *resolvable* abstention (maps to ``ThreeState.UNAVAILABLE``): the
    #: evidence this run is insufficient -- a calibrated ambiguity, a near-
    #: threshold band with no coverage certificate, neuro-symbolic
    #: disagreement, or distribution drift.  More data, recalibration, or a
    #: human could decide it, so the loop defers and asks for that.
    DEFER = "defer"

    #: A *fail-closed* abstention (maps to ``ThreeState.UNDECIDABLE``): the
    #: input is outside the model's certified scope (an atypical point no
    #: class explains) or a hard gate refused the boundary.  More of the same
    #: evidence cannot resolve it, so the loop holds and routes to a human.
    HOLD = "hold"


class ResponseAction(Enum):
    """A bounded, **non-destructive** response the loop may recommend.

    Every member is advisory or notifying -- the layer recommends and
    escalates, it never autonomously executes a destructive or irreversible
    action.  The catalogue is deliberately small and is aligned with the
    action vocabulary the existing autonomy loop already speaks
    (``flag_anomaly`` / ``escalate`` / ``investigate`` / ``log``), so a
    :class:`~omni_mercury_engine.decision.record.DecisionRecord` can drive the
    reinforcement-learning agent without a second vocabulary.
    """

    #: Passive observation only -- the grounded-normal stance.
    MONITOR = "monitor"

    #: Emit a standards-based notification (CAP 1.2) to operators.
    ALERT = "alert"

    #: Surface advisory, reversible countermeasures for human approval.
    #: Recommendations only -- never auto-applied.
    RECOMMEND_MITIGATION = "recommend_mitigation"

    #: Put a human in the loop before any consequential step.
    ESCALATE_TO_HUMAN = "escalate_to_human"

    #: Ask for the missing signal (more samples, recalibration, a label) that
    #: would turn a resolvable abstention into a decision.
    REQUEST_INPUT = "request_input"

    #: Fail-closed refusal: take no autonomous action and route to a human.
    HOLD = "hold"


__all__ = ["Disposition", "ResponseAction"]
