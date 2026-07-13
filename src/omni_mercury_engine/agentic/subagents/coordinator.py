# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Coordinator subagent: a real **operator** over one or more engine subsystems.

A :class:`CoordinatorSubAgent` is the pantheon member used when a role maps to a
real ``omni_mercury_engine`` subsystem that does not have a bespoke deep
specialization. Its work is genuine and fail-closed. It is a true subsystem
*operator*: for each member, an adapter in
:mod:`~omni_mercury_engine.agentic.subagents.operations` invokes that member's
**real** engine entrypoint with inputs derived from ``task.payload`` (e.g.
``Helios_XVII`` computes real telemetry metrics, ``Kronos_XXII`` fits and runs a
real detector, ``Artemis_VI`` genuinely probes data-source reachability). The
result is the transparent output of that call (``mode="operation"``); it is never
fabricated.

When a member's entrypoint is *input-gated* and the payload lacks the required
inputs — or the operator is asked for a readiness probe via
``payload["mode"] == "introspect"`` — the coordinator falls back to the transparent
**binding report** (``mode="binding"``): it imports each declared subsystem,
introspects its live public API, and reports genuine availability/capability.
That report fails closed when *no* subsystem binds, so the signal is true by
construction either way. The binding report is the transparent no-input floor, never
the whole behavior.
"""

from __future__ import annotations

import importlib
import inspect
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.agentic.subagents.base import SubAgent, SubAgentExecutionError
from omni_mercury_engine.agentic.subagents.operations import OPERATIONS
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
    """A subagent that *operates* its real engine subsystem(s), transparently.

    Dispatches to the member's real-entrypoint adapter in
    :mod:`~omni_mercury_engine.agentic.subagents.operations`; falls back to a
    live binding/capability report when input-gated and uninvoked.
    """

    def _perform(self, task: SubAgentTask) -> tuple[dict[str, Any], float, str]:
        """Run the member's real entrypoint, else report its genuine binding.

        Dispatches to the operations adapter for this pantheon id, invoking the
        real ``omni_mercury_engine`` entrypoint with payload-derived inputs and
        returning its transparent output (``mode="operation"``). When the adapter is
        input-gated and the payload lacks its inputs, or the caller asks for a
        readiness probe (``payload["mode"] == "introspect"``), it falls back to
        the live binding report (``mode="binding"``). Fails closed throughout —
        never a fabricated result.
        """
        if str(task.payload.get("mode", "")) != "introspect":
            adapter = OPERATIONS.get(self.id)
            if adapter is not None:
                result = adapter(self, task)  # may raise SubAgentExecutionError (fail-closed)
                if result is not None:
                    output, confidence, reasoning = result
                    output.setdefault("subagent", self.id)
                    output.setdefault("role", self.role)
                    output["anchor"] = self.anchor_name
                    output["autonomy_ceiling"] = self.autonomy_ceiling
                    output["mode"] = "operation"
                    return output, float(confidence), reasoning
        return self._binding_report(task)

    def _binding_report(self, task: SubAgentTask) -> tuple[dict[str, Any], float, str]:
        """Transparent fallback: bind to each declared subsystem and report capability.

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
            "mode": "binding",
            "bound_subsystems": bound,
            "subsystems": report,
            "task": task.description,
        }
        reasoning = (
            f"{self.id} ({self.role}) bound {len(bound)}/{len(self.subsystems)} subsystem(s): "
            f"{', '.join(bound)}"
        )
        return output, confidence, reasoning
