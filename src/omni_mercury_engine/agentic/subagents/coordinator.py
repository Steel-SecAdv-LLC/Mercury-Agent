# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Coordinator subagent: a real binding to one or more engine subsystems.

A :class:`CoordinatorSubAgent` is the pantheon member used when a role maps to a
real ``omni_mercury_engine`` subsystem that does not (yet) have a bespoke deep
specialization. Its work is genuine and fail-closed: it *binds* to its declared
subsystem(s) by importing them, introspects their real public API, and reports
live availability and capability. It never fabricates a domain result; when none
of its subsystems can be imported it fails honestly with
:class:`SubAgentExecutionError`.

This gives the root agent a real, auditable substrate — the live capability
surface of each bound subsystem — on which deeper, operation-specific behavior
can be layered per subsystem over time. The binding reflects the actual state of
the repository, so the signal is true by construction.
"""

from __future__ import annotations

import importlib
import inspect
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.agentic.subagents.base import SubAgent, SubAgentExecutionError
from omni_mercury_engine.agentic.subagents.roster import subsystem_module

if TYPE_CHECKING:
    from omni_mercury_engine.agentic.subagents.base import SubAgentTask

# Cap on enumerated public symbols per subsystem (keeps the report bounded).
_MAX_PUBLIC_SYMBOLS = 60


def _public_api(module: Any, module_path: str) -> list[str]:
    """Enumerate a module's own public classes/functions (deterministic, bounded)."""
    declared = getattr(module, "__all__", None)
    if isinstance(declared, (list, tuple)) and declared:
        names = [str(n) for n in declared if not str(n).startswith("_")]
    else:
        names = [
            name
            for name, obj in inspect.getmembers(module)
            if not name.startswith("_")
            and (inspect.isclass(obj) or inspect.isfunction(obj))
            and getattr(obj, "__module__", "").startswith(module_path)
        ]
    return sorted(set(names))[:_MAX_PUBLIC_SYMBOLS]


class CoordinatorSubAgent(SubAgent):
    """A subagent that binds to and assesses its real engine subsystem(s)."""

    def _perform(self, task: SubAgentTask) -> tuple[dict[str, Any], float, str]:
        """Bind to each declared subsystem and report genuine capability.

        Imports each subsystem module, records whether it is available and its
        live public API, and returns a real capability report. Confidence is the
        fraction of declared subsystems that bound successfully. Fails closed if
        none bind — never returns a fabricated result.
        """
        report: dict[str, dict[str, Any]] = {}
        bound: list[str] = []
        for sub in self.subsystems:
            module_path = subsystem_module(sub)
            try:
                module = importlib.import_module(module_path)
            except Exception as exc:  # genuine unavailability — surfaced, not hidden
                report[sub] = {
                    "available": False,
                    "module": module_path,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                continue
            api = _public_api(module, module_path)
            report[sub] = {
                "available": True,
                "module": module_path,
                "n_public": len(api),
                "public_api": api,
            }
            bound.append(sub)

        if not bound:
            raise SubAgentExecutionError(
                f"{self.id}: none of its bound subsystems {self.subsystems!r} are importable"
            )

        confidence = len(bound) / len(self.subsystems)
        output: dict[str, Any] = {
            "subagent": self.id,
            "role": self.role,
            "anchor": self.anchor_name,
            "autonomy_ceiling": self.autonomy_ceiling,
            "bound_subsystems": bound,
            "subsystems": report,
            "task": task.description,
        }
        reasoning = (
            f"{self.id} ({self.role}) bound {len(bound)}/{len(self.subsystems)} subsystem(s): "
            f"{', '.join(bound)}"
        )
        return output, confidence, reasoning
