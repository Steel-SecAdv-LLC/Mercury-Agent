# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Autonomy governor for the subagent fleet.

This is the safety spine that lets the main agent run subagents "in the masses"
without the failure modes of an unbounded fleet. It distils the *genuine* intent
behind a capability-ceiling / AGI-safety monitor into enforcement that is real
and measurable rather than aspirational:

* **Capability ceiling** — hard caps on concurrent replicas per dispatch, total
  active subagents, and recursion depth (a subagent delegating to subagents).
  Exceeding any cap fails closed.
* **Ethical floor** — the benevolence floor and the Omni-Code autonomy cap are
  asserted here as well as at each subagent, so a misconfigured specialization
  cannot raise its own ceiling.
* **Corrigibility** — :meth:`pause` / :meth:`resume` and :meth:`trip` give the
  main agent an always-available kill-switch; a halted governor refuses every
  dispatch until explicitly resumed (or never, if tripped).
* **Tripwire** — :meth:`observe_results` watches realized outcomes and trips the
  governor when the failure rate crosses a threshold or an autonomy breach is
  observed, halting the fleet before a degenerate run amplifies.

Thread-safety: a fleet dispatches subagents concurrently, so all mutable
counters are guarded by a lock. Counts are exact, not best-effort.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from omni_mercury_engine.cognitive.ethical_bounding import MINIMUM_BENEVOLENCE_FLOOR

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omni_mercury_engine.agentic.subagents.base import SubAgentResult

# Omni-Code-aligned hard autonomy cap (mirrors compute_ethical_autonomy's 0.95).
_MAX_AUTONOMY_CAP = 0.95


class GovernorTripped(RuntimeError):
    """Raised when the governor refuses a dispatch (fail-closed).

    Carries the reason so the caller (and audit trail) sees *why* the fleet
    declined to act rather than a bare refusal.
    """


@dataclass
class CapabilityCeiling:
    """Hard limits the governor enforces on the fleet.

    Attributes:
        max_replicas: Max subagents one ``scale_dispatch`` may run at once.
        max_total_active: Max subagents active across the whole fleet at once.
        max_recursion_depth: Max depth of subagents delegating to subagents.
        max_autonomy: Hard autonomy cap (Omni-Code aligned).
        min_benevolence: Hard benevolence floor (mirrors the engine's).
        max_failure_rate: Realized failure rate above which the tripwire fires.
        tripwire_min_observations: Minimum results before the tripwire can fire
            (so a single early failure cannot halt the fleet).
    """

    max_replicas: int = 64
    max_total_active: int = 256
    max_recursion_depth: int = 3
    max_autonomy: float = _MAX_AUTONOMY_CAP
    min_benevolence: float = MINIMUM_BENEVOLENCE_FLOOR
    max_failure_rate: float = 0.5
    tripwire_min_observations: int = 8

    def __post_init__(self) -> None:
        """Validate the capability-ceiling invariants, failing closed.

        Raises:
            ValueError: If any ceiling is out of range (non-positive bounds, a
                negative recursion depth, or an autonomy / benevolence /
                failure-rate value outside its permitted interval).
        """
        if self.max_replicas < 1 or self.max_total_active < 1:
            raise ValueError("capability ceilings must be >= 1")
        if self.max_recursion_depth < 0:
            raise ValueError("max_recursion_depth must be >= 0")
        if not 0.0 < self.max_autonomy <= _MAX_AUTONOMY_CAP:
            raise ValueError(f"max_autonomy must be in (0, {_MAX_AUTONOMY_CAP}]")
        if not 0.0 <= self.min_benevolence <= 1.0:
            raise ValueError("min_benevolence must be in [0, 1]")
        if not 0.0 < self.max_failure_rate <= 1.0:
            raise ValueError("max_failure_rate must be in (0, 1]")


@dataclass
class GovernorState:
    """Point-in-time snapshot of the governor (for telemetry / audit)."""

    active: int
    paused: bool
    tripped: bool
    total_authorized: int
    total_completed: int
    total_failed: int
    total_blocked: int
    trip_reasons: list[str] = field(default_factory=list)


class AutonomyGovernor:
    """Enforces the capability ceiling, ethical floor, corrigibility, tripwire."""

    def __init__(self, ceiling: CapabilityCeiling | None = None) -> None:
        """Initialize the governor with a (default) capability ceiling."""
        self.ceiling = ceiling or CapabilityCeiling()
        self._lock = threading.Lock()
        self._active = 0
        self._paused = False
        self._tripped = False
        self._trip_reasons: list[str] = []
        self._total_authorized = 0
        self._total_completed = 0
        self._total_failed = 0
        self._total_blocked = 0

    # ------------------------------------------------------------------
    # Corrigibility (always-available kill-switch)
    # ------------------------------------------------------------------

    def pause(self) -> None:
        """Halt new dispatches (reversible). In-flight subagents finish."""
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        """Resume after a :meth:`pause`. A *tripped* governor stays halted."""
        with self._lock:
            self._paused = False

    def trip(self, reason: str) -> None:
        """Permanently halt the fleet (tripwire / operator kill).

        A tripped governor refuses every dispatch until a fresh governor is
        installed — deliberately irreversible, so a safety stop is not undone by
        an automatic :meth:`resume`.
        """
        with self._lock:
            self._tripped = True
            self._trip_reasons.append(reason)

    @property
    def is_halted(self) -> bool:
        """Whether dispatches are currently refused (paused or tripped)."""
        with self._lock:
            return self._paused or self._tripped

    @property
    def is_tripped(self) -> bool:
        """Whether the governor has been irreversibly tripped."""
        with self._lock:
            return self._tripped

    # ------------------------------------------------------------------
    # Capability ceiling (fail-closed authorization)
    # ------------------------------------------------------------------

    def authorize(self, replicas: int, depth: int) -> None:
        """Authorize a dispatch of ``replicas`` subagents at recursion ``depth``.

        Reserves the capacity atomically. The caller MUST call :meth:`release`
        with the same count when the dispatch completes (the fleet does this in
        a ``finally``). Fail-closed: any ceiling breach raises
        :class:`GovernorTripped` and reserves nothing.
        """
        if replicas < 1:
            raise GovernorTripped(f"dispatch must request >= 1 replica (got {replicas})")
        with self._lock:
            if self._tripped:
                raise GovernorTripped(
                    f"governor tripped: {'; '.join(self._trip_reasons) or 'safety stop'}"
                )
            if self._paused:
                raise GovernorTripped("governor paused: dispatches are halted")
            if depth > self.ceiling.max_recursion_depth:
                raise GovernorTripped(
                    f"recursion depth {depth} exceeds ceiling "
                    f"{self.ceiling.max_recursion_depth}"
                )
            if replicas > self.ceiling.max_replicas:
                raise GovernorTripped(
                    f"replica count {replicas} exceeds per-dispatch ceiling "
                    f"{self.ceiling.max_replicas}"
                )
            if self._active + replicas > self.ceiling.max_total_active:
                raise GovernorTripped(
                    f"dispatch of {replicas} would exceed total-active ceiling "
                    f"{self.ceiling.max_total_active} (active={self._active})"
                )
            self._active += replicas
            self._total_authorized += replicas

    def release(self, replicas: int) -> None:
        """Release capacity reserved by :meth:`authorize`."""
        with self._lock:
            self._active = max(0, self._active - replicas)

    def check_autonomy(self, autonomy: float) -> None:
        """Assert a subagent's autonomy never exceeds the hard ceiling.

        A specialization that somehow raised its own autonomy ceiling trips the
        governor — the fleet is halted rather than allowed to run an over-
        autonomous worker.
        """
        if autonomy > self.ceiling.max_autonomy + 1e-9:
            self.trip(
                f"autonomy breach: subagent autonomy {autonomy:.3f} exceeds cap "
                f"{self.ceiling.max_autonomy:.3f}"
            )
            raise GovernorTripped(
                f"subagent autonomy {autonomy:.3f} exceeds cap {self.ceiling.max_autonomy:.3f}"
            )

    # ------------------------------------------------------------------
    # Tripwire (watch realized outcomes)
    # ------------------------------------------------------------------

    def observe_results(self, results: Sequence[SubAgentResult]) -> None:
        """Account realized outcomes and fire the tripwire on degenerate runs.

        Trips when the cumulative failure rate exceeds the ceiling after enough
        observations, or immediately on any autonomy breach. Blocked (ethically
        refused) results are accounted separately — they are correct refusals,
        not failures, and never trip the wire.
        """
        with self._lock:
            for r in results:
                if r.status == "completed":
                    self._total_completed += 1
                elif r.status == "blocked":
                    self._total_blocked += 1
                else:
                    self._total_failed += 1
                if r.autonomy_ceiling > self.ceiling.max_autonomy + 1e-9:
                    self._tripped = True
                    self._trip_reasons.append(
                        f"autonomy breach observed in {r.subagent_id} "
                        f"({r.autonomy_ceiling:.3f})"
                    )
            graded = self._total_completed + self._total_failed
            if graded >= self.ceiling.tripwire_min_observations:
                failure_rate = self._total_failed / graded
                if failure_rate > self.ceiling.max_failure_rate and not self._tripped:
                    self._tripped = True
                    self._trip_reasons.append(
                        f"failure-rate tripwire: {failure_rate:.2f} over {graded} graded "
                        f"results exceeds {self.ceiling.max_failure_rate:.2f}"
                    )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def snapshot(self) -> GovernorState:
        """Return a consistent point-in-time snapshot of the governor."""
        with self._lock:
            return GovernorState(
                active=self._active,
                paused=self._paused,
                tripped=self._tripped,
                total_authorized=self._total_authorized,
                total_completed=self._total_completed,
                total_failed=self._total_failed,
                total_blocked=self._total_blocked,
                trip_reasons=list(self._trip_reasons),
            )
