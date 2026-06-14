# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mercury's pluggable reasoning layer.

Mercury Agent is the agent and the brain of record. This subpackage gives it a
swappable, **subordinate** reasoning engine to call — for explanation,
hypothesis proposal, and report synthesis — without ever becoming a wrapper
around that engine.

* :class:`ReasoningBackend` — the Mercury-owned interface; its public methods
  speak Mercury's vocabulary and every call is governed by Mercury's dual hard
  ethical gate before any output is surfaced.
* :class:`MockReasoningBackend` / :class:`LocalReasoningBackend` /
  :class:`RemoteReasoningBackend` — a network-free test double, an
  offline-first local backend (free to run, air-gap-safe), and an
  operator-declared network-capable backend.
* :class:`ReasoningRouter` — offline-first routing: the local backend is the
  default and the floor; the remote backend is reached only on explicit opt-in
  and never under hard-offline mode.

Typed inputs/outputs live in :mod:`omni_mercury_engine.reasoning.schemas`.
"""

from __future__ import annotations

from omni_mercury_engine.reasoning.backend import ReasoningBackend
from omni_mercury_engine.reasoning.backends import (
    LocalReasoningBackend,
    MockReasoningBackend,
    RemoteReasoningBackend,
)
from omni_mercury_engine.reasoning.router import ReasoningRouter
from omni_mercury_engine.reasoning.schemas import (
    Explanation,
    Hypothesis,
    ReasoningContext,
    Report,
)

__all__ = [
    "Explanation",
    "Hypothesis",
    "LocalReasoningBackend",
    "MockReasoningBackend",
    "ReasoningBackend",
    "ReasoningContext",
    "ReasoningRouter",
    "RemoteReasoningBackend",
    "Report",
]
