# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""A staged model registry with a one-command rollback.

A gated retrain writes its accepted candidate into a **staging** registry, never
straight to production. The registry tracks exactly two live pointers -- the
``active`` staged model and the ``previous`` one it superseded -- plus an
append-only history. If a staged model turns out bad, :meth:`ModelRegistry.rollback`
(surfaced as ``scripts/mercury_retrain_rollback.py``) swaps ``active`` back to
``previous`` in a single, audited, atomic operation. That is the "one-command
rollback" the closed loop's safety contract requires: no manual file surgery, no
ambiguity about what the previous good model was.

The registry file (``<staging_dir>/registry.json``) is written atomically
(temp-file + ``os.replace``) so a crash mid-write can never leave a half-written
pointer that would strand the rollback path.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omni_mercury_engine.cognitive.gate_audit import record_gate_decision

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelEntry:
    """A registered staged model pointer."""

    version: str
    model_path: str
    kind: str  # "candidate" | "baseline"
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping."""
        return {
            "version": self.version,
            "model_path": self.model_path,
            "kind": self.kind,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelEntry:
        """Rebuild a :class:`ModelEntry` from its stored mapping."""
        return cls(
            version=str(data["version"]),
            model_path=str(data["model_path"]),
            kind=str(data.get("kind", "candidate")),
            metrics=dict(data.get("metrics", {})),
        )


@dataclass(frozen=True)
class RollbackResult:
    """The outcome of a rollback attempt."""

    rolled_back: bool
    reason: str
    from_version: str | None = None
    to_version: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping."""
        return {
            "rolled_back": self.rolled_back,
            "reason": self.reason,
            "from_version": self.from_version,
            "to_version": self.to_version,
        }


class ModelRegistry:
    """Two-pointer (active/previous) staged model registry with atomic writes."""

    def __init__(self, staging_dir: str | os.PathLike[str]) -> None:
        """Open (or lazily create) the registry under ``staging_dir``."""
        self.staging_dir = Path(staging_dir)
        self.registry_path = self.staging_dir / "registry.json"

    def _read(self) -> dict[str, Any]:
        if not self.registry_path.is_file():
            return {"active": None, "previous": None, "history": []}
        state: dict[str, Any] = json.loads(self.registry_path.read_text(encoding="utf-8"))
        return state

    def _write(self, state: dict[str, Any]) -> None:
        """Atomically write the registry (temp-file + os.replace)."""
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.registry_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(self.registry_path)

    def active(self) -> ModelEntry | None:
        """Return the currently-active staged model, if any."""
        raw = self._read().get("active")
        return ModelEntry.from_dict(raw) if raw else None

    def previous(self) -> ModelEntry | None:
        """Return the model the active one superseded, if any."""
        raw = self._read().get("previous")
        return ModelEntry.from_dict(raw) if raw else None

    def register(self, entry: ModelEntry) -> None:
        """Promote ``entry`` to ``active`` (the prior active becomes ``previous``).

        Appended to ``history`` for the full audit trail. Idempotent on version:
        re-registering the current active version is a no-op (does not shuffle the
        previous pointer, which would corrupt the rollback target).
        """
        state = self._read()
        current = state.get("active")
        if current and current.get("version") == entry.version:
            logger.info("model version %s already active; register is a no-op", entry.version)
            return
        state["previous"] = current
        state["active"] = entry.as_dict()
        history = state.get("history", [])
        history.append(entry.as_dict())
        state["history"] = history
        self._write(state)
        record_gate_decision(
            decision="model_registered",
            source="feedback_loop:registry",
            disposition="approved",
            signals=("closed_loop", "staged_model"),
            reason=f"staged model {entry.version} registered as active ({entry.kind})",
            extra={"version": entry.version, "model_path": entry.model_path},
        )

    def rollback(self) -> RollbackResult:
        """Restore ``previous`` as ``active`` (the one-command rollback).

        Fail-safe: with no active or no previous model there is nothing to roll
        back to, so it reports ``rolled_back=False`` rather than corrupting state.

        **Monotonic, not a toggle.** A rollback restores the previous good model
        and then *clears* the ``previous`` pointer, so a repeated ``rollback()``
        (an operator double-click, a retry wrapper) is a no-op rather than
        swapping the just-removed candidate back into ``active``. Re-activating a
        model that was rolled back must go through a fresh, gate-checked
        :meth:`register` -- never through a blind second rollback. Without this,
        two rollbacks would silently re-arm the exact (possibly poisoned) model
        the first rollback removed, and audit it as an approved decision.
        """
        state = self._read()
        active = state.get("active")
        previous = state.get("previous")
        if not active:
            return RollbackResult(False, "no active model to roll back")
        if not previous:
            return RollbackResult(
                False,
                "no previous model to roll back to (already rolled back, or only "
                "one model registered); re-activate via a fresh gated register",
                from_version=active.get("version"),
            )
        # Restore the previous good model as active, then clear the previous
        # pointer: the rolled-back-from candidate is intentionally NOT restorable
        # via another rollback (that would re-arm a bad model). A subsequent
        # register() repopulates previous with the then-current active.
        state["active"] = previous
        state["previous"] = None
        history = state.get("history", [])
        history.append({**previous, "kind": "rollback_restore"})
        state["history"] = history
        self._write(state)
        result = RollbackResult(
            rolled_back=True,
            reason="rolled active back to previous staged model",
            from_version=active.get("version"),
            to_version=previous.get("version"),
        )
        record_gate_decision(
            decision="model_rollback",
            source="feedback_loop:registry",
            disposition="approved",
            signals=("closed_loop", "rollback"),
            reason=result.reason,
            extra={"from_version": result.from_version, "to_version": result.to_version},
        )
        logger.info("rolled back model %s -> %s", result.from_version, result.to_version)
        return result


def rollback_staging(staging_dir: str | os.PathLike[str]) -> RollbackResult:
    """One-command rollback of the staged model registry at ``staging_dir``."""
    return ModelRegistry(staging_dir).rollback()


__all__ = [
    "ModelEntry",
    "ModelRegistry",
    "RollbackResult",
    "rollback_staging",
]
