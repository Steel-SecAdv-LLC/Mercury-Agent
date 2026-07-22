# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Periodic maintenance sweeps for the platform stores.

Append-only and expiring state needs an owner or it grows without bound and
quietly degrades the whole deployment: the ``usage_ledger`` gains a row per
metered request forever, expired sessions and consumed email tokens pile up as
dead rows an attacker-facing query still scans, and stale rate-limit buckets
accumulate one row per client IP ever seen.

:func:`run_maintenance_sweep` prunes all of it in one pass — and doubles as
the **TOTP sealing migration**: any plaintext TOTP secret found while a stable
at-rest key is configured is sealed in place, so enabling
``MERCURY_DATA_ENC_KEY`` on an existing deployment upgrades every stored
secret at the next sweep with no manual step.

The server lifespan runs a sweep at startup and then every
``MERCURY_MAINTENANCE_INTERVAL_SECONDS`` (default 3600; ``0`` disables the
periodic loop). Sweeps are failure-isolated per store: one backend erroring is
logged and the rest still run — maintenance must never take the API down.
Everything is driven off the environment-selected backends, so a solo
in-memory deployment sweeps empty stores at zero cost.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

logger = logging.getLogger(__name__)

__all__ = [
    "maintenance_interval_seconds",
    "run_maintenance_sweep",
    "start_maintenance_task",
    "usage_retention",
]

#: Default usage-ledger retention. Must comfortably exceed the largest quota
#: window in use (pruning inside a live window would refund spent quota).
_DEFAULT_USAGE_RETENTION_DAYS = 30.0


def usage_retention() -> timedelta:
    """The configured usage-ledger retention window."""
    raw = os.getenv("MERCURY_USAGE_RETENTION_DAYS", "").strip()
    try:
        days = float(raw) if raw else _DEFAULT_USAGE_RETENTION_DAYS
    except ValueError:
        days = _DEFAULT_USAGE_RETENTION_DAYS
    if days <= 0:
        days = _DEFAULT_USAGE_RETENTION_DAYS
    return timedelta(days=days)


def maintenance_interval_seconds() -> float:
    """Seconds between periodic sweeps (0 disables the loop)."""
    raw = os.getenv("MERCURY_MAINTENANCE_INTERVAL_SECONDS", "").strip()
    try:
        return float(raw) if raw else 3600.0
    except ValueError:
        return 3600.0


def _isolated(label: str, results: dict[str, int], task: Callable[[], int]) -> None:
    """Run one sweep step, recording its count and isolating its failures."""
    try:
        results[label] = task()
    except Exception:
        logger.exception("maintenance step %s failed", label)
        results[label] = -1


def run_maintenance_sweep(now: datetime | None = None) -> dict[str, int]:
    """Prune expired/consumed state across every configured backend.

    Args:
        now: Injectable current time for deterministic tests.

    Returns:
        Per-step counts (``-1`` marks a step that errored; see logs).
    """
    from omni_mercury_engine.api.identity_store import (
        build_identity_store,
        identity_store_is_durable,
    )
    from omni_mercury_engine.api.rate_limit_store import (
        build_action_rate_limiter,
        build_shared_bucket_backend,
    )
    from omni_mercury_engine.api.secret_sealer import (
        build_secret_sealer,
        migrate_plaintext_totp_secrets,
    )
    from omni_mercury_engine.api.usage_ledger import build_usage_ledger

    moment = now or datetime.now(UTC)
    results: dict[str, int] = {}

    store = build_identity_store()
    _isolated("expired_sessions", results, lambda: store.prune_expired_sessions(moment))
    _isolated("email_tokens", results, lambda: store.prune_email_tokens(moment))
    _isolated(
        "sealed_totp_secrets",
        results,
        lambda: migrate_plaintext_totp_secrets(
            store, build_secret_sealer(store_is_durable=identity_store_is_durable())
        ),
    )

    ledger = build_usage_ledger()
    cutoff = moment - usage_retention()
    _isolated("usage_events", results, lambda: ledger.prune_before(cutoff))

    bucket_backend = build_shared_bucket_backend()
    if bucket_backend is not None:
        # A bucket idle for an hour has long since refilled; its row is dead.
        _isolated("rate_buckets", results, lambda: bucket_backend.prune_stale(time.time() - 3600))

    limiter = build_action_rate_limiter()
    counter_store = getattr(limiter, "_store", None)
    if counter_store is not None and hasattr(counter_store, "prune_stale"):
        # Fixed-window counters two days old are outside every action window.
        _isolated(
            "rate_counters",
            results,
            lambda: counter_store.prune_stale(int(time.time()) - 2 * 86400),
        )

    logger.info("maintenance sweep completed: %s", results)
    return results


async def _maintenance_loop(interval: float) -> None:
    """Run sweeps forever at ``interval`` seconds (cancelled at shutdown)."""
    while True:
        await asyncio.sleep(interval)
        try:
            await asyncio.to_thread(run_maintenance_sweep)
        except Exception:  # pragma: no cover - the loop must survive any sweep
            logger.exception("periodic maintenance sweep failed")


def start_maintenance_task() -> asyncio.Task[None] | None:
    """Run the startup sweep and start the periodic loop (if enabled).

    Called from the server lifespan. The startup sweep runs synchronously so
    the process begins life pruned (and TOTP migration applies before the
    first login); the periodic loop then keeps it that way.

    Returns:
        The loop task (cancel it at shutdown), or ``None`` when the periodic
        loop is disabled.
    """
    try:
        run_maintenance_sweep()
    except Exception:  # pragma: no cover - startup must not be blocked by a sweep
        logger.exception("startup maintenance sweep failed")

    interval = maintenance_interval_seconds()
    if interval <= 0:
        return None
    return asyncio.get_running_loop().create_task(_maintenance_loop(interval))


@contextlib.contextmanager
def maintenance_disabled_for_tests() -> Iterator[None]:
    """Context manager tests use to switch the periodic loop off."""
    previous = os.environ.get("MERCURY_MAINTENANCE_INTERVAL_SECONDS")
    os.environ["MERCURY_MAINTENANCE_INTERVAL_SECONDS"] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MERCURY_MAINTENANCE_INTERVAL_SECONDS", None)
        else:
            os.environ["MERCURY_MAINTENANCE_INTERVAL_SECONDS"] = previous
