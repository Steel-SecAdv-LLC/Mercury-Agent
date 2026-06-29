# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generalist subagent: the full main-agent pipeline; the fleet's routing floor.

The generalist is the internal routing floor (not a public pantheon member): any
task that attracts no specialist runs here. It is genuinely capable — it inherits
and runs the complete
:meth:`~omni_mercury_engine.agentic.mercury_a_agent.MercuryAgent.analyze`
pipeline (plan → reason → execute → learn) under the same ethical gate. It is not
a stub or a no-op fallback.
"""

from __future__ import annotations

from omni_mercury_engine.agentic.subagents.base import SubAgent


class GeneralistSubAgent(SubAgent):
    """A subagent with the main agent's full, unspecialized capability."""
