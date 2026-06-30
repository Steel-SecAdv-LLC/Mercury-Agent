# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Subagent registry: the pantheon catalogue + deterministic task routing.

The registry is built directly from the declarative
:data:`~omni_mercury_engine.agentic.subagents.roster.ROSTER` (plus the internal
routing floor). It instantiates the right class per member — the deep
specialization named by ``impl_path``, or the generic
:class:`~omni_mercury_engine.agentic.subagents.coordinator.CoordinatorSubAgent`
— always threading the internal access sentinel so the boundary holds end to
end. It is access-guarded and never exposed publicly.

Routing is deterministic and explainable: a task is scored against each public
member's keywords (dominant) and domain (tie-break); the lexicographic id breaks
remaining ties. A task that attracts no specialist keyword routes to the
internal generalist floor — never to silence.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from omni_mercury_engine.agentic.subagents.base import (
    _INTERNAL,
    SubAgent,
    SubAgentAccessError,
    _InternalAccess,
)
from omni_mercury_engine.agentic.subagents.coordinator import CoordinatorSubAgent
from omni_mercury_engine.agentic.subagents.roster import (
    ALL_ENTRIES,
    GENERALIST_FLOOR,
    ROSTER,
)

if TYPE_CHECKING:
    from omni_mercury_engine.agentic.subagents.base import SubAgentTask
    from omni_mercury_engine.agentic.subagents.roster import RosterEntry


def _import_class(dotted_path: str) -> type[SubAgent]:
    """Import a ``module.Class`` dotted path and return the class object."""
    module_path, _, class_name = dotted_path.rpartition(".")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not (isinstance(cls, type) and issubclass(cls, SubAgent)):
        raise TypeError(f"{dotted_path!r} is not a SubAgent subclass")
    return cls


class SubAgentRegistry:
    """The pantheon catalogue with deterministic capability routing."""

    def __init__(self, access: _InternalAccess) -> None:
        """Construct the registry from the roster. Engine-mediated only.

        Raises:
            SubAgentAccessError: If ``access`` is not the internal sentinel.
        """
        if access is not _INTERNAL:
            raise SubAgentAccessError(
                "SubAgentRegistry is internal-only; obtain it via default_registry() "
                "behind the engine-mediated fleet."
            )
        self._entries: dict[str, RosterEntry] = {e.id: e for e in ALL_ENTRIES}

    def get(self, agent_id: str) -> RosterEntry:
        """Look up a roster entry by id.

        Raises:
            KeyError: If the id is unknown.
        """
        if agent_id not in self._entries:
            raise KeyError(f"unknown subagent {agent_id!r}; registered: {self.list_specialties()}")
        return self._entries[agent_id]

    def list_specialties(self) -> list[str]:
        """The 33 public pantheon ids, sorted (the internal floor is excluded)."""
        return sorted(e.id for e in self._entries.values() if not e.internal)

    def create(
        self, agent_id: str, access: _InternalAccess, *, seed: int | None = None
    ) -> SubAgent:
        """Instantiate the subagent for ``agent_id`` (engine-mediated only).

        Resolves the deep specialization class (``impl_path``) or falls back to
        :class:`CoordinatorSubAgent`, threading the internal sentinel through.
        """
        entry = self.get(agent_id)
        cls = _import_class(entry.impl_path) if entry.impl_path else CoordinatorSubAgent
        return cls(access=access, entry=entry, seed=seed)

    def match(self, task: SubAgentTask) -> str:
        """Route a task to the best public member, else the generalist floor.

        Scoring (deterministic): keyword overlap dominates; domain competence
        breaks ties; the lexicographic id breaks remaining ties. The internal
        floor is never an attractor — it is only the fallback.
        """
        description = task.description.lower()
        best_id: str | None = None
        best_key: tuple[int, int, str] = (-1, -1, "")
        for entry in ROSTER:  # public members only
            keyword_hits = sum(1 for kw in entry.keywords if kw in description)
            if keyword_hits == 0:
                continue
            domain_match = 1 if task.domain == entry.domain else 0
            key = (keyword_hits, domain_match, entry.id)
            if key > best_key:
                best_key = key
                best_id = entry.id
        return best_id if best_id is not None else GENERALIST_FLOOR.id


def default_registry(access: _InternalAccess) -> SubAgentRegistry:
    """Build the registry (pre-loaded from the roster)."""
    return SubAgentRegistry(access)
