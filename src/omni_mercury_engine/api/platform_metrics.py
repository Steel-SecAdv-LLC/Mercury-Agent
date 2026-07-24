# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Prometheus counters for the account/quota/mailer/maintenance platform.

The platform paths added for the free public service (registration, login,
per-action throttles, quotas, transactional mail, maintenance sweeps) were
observable only through logs. This module gives each a counter on the
``prometheus_client`` default registry, so the existing ``/metrics`` endpoint
(which appends the default-registry exposition — see
:func:`omni_mercury_engine.core.metrics.render_exposition`) serves them with
no new scrape target.

Design constraints, in order:

* **The core lane must not require prometheus.** The import is guarded; with
  ``prometheus_client`` absent every ``record_*`` function is a silent no-op.
* **Metrics must never break the request path.** Every recorder swallows its
  own failures; an instrumentation bug degrades observability, not serving.
* **Label cardinality is bounded.** Labels only ever carry closed sets the
  code defines (action names, outcome words, sweep step names) — never
  account ids, emails, IPs, or any other caller-controlled value.

Registration is lazy: a counter is created on the first event that touches
it, so importing this module (which the route layer always does) costs
nothing on deployments that never see the instrumented events.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "PROMETHEUS_AVAILABLE",
    "record_email",
    "record_login",
    "record_maintenance_sweep",
    "record_quota_denial",
    "record_registration",
    "record_throttle_config_mismatch",
    "record_throttle_denial",
]

try:
    from prometheus_client import Counter

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via the reload test
    PROMETHEUS_AVAILABLE = False

_counters: dict[str, Any] = {}
_counters_lock = threading.Lock()


def _counter(name: str, description: str, labelnames: tuple[str, ...] = ()) -> Any:
    """Return (creating on first use) the named default-registry counter.

    Args:
        name: Prometheus metric name.
        description: Help text.
        labelnames: Label names (fixed per metric; values must be bounded).

    Returns:
        The ``prometheus_client.Counter``, or ``None`` when prometheus is
        not installed.
    """
    if not PROMETHEUS_AVAILABLE:
        return None
    counter = _counters.get(name)
    if counter is None:
        with _counters_lock:
            counter = _counters.get(name)
            if counter is None:
                counter = Counter(name, description, list(labelnames))
                _counters[name] = counter
    return counter


def _inc(name: str, description: str, labels: dict[str, str] | None = None) -> None:
    """Increment a lazily created counter, never letting a failure escape."""
    try:
        counter = _counter(name, description, tuple(labels) if labels else ())
        if counter is None:
            return
        if labels:
            counter.labels(**labels).inc()
        else:
            counter.inc()
    except Exception:
        # Metrics must never break a request; exercised by
        # test_platform_metrics.test_metric_failure_never_breaks_the_caller.
        logger.exception("platform metric %s failed to record", name)


def record_registration() -> None:
    """Count one successful account registration."""
    _inc(
        "mercury_platform_registrations_total",
        "Successful account registrations",
    )


def record_login(outcome: str) -> None:
    """Count one login attempt by outcome.

    Args:
        outcome: ``"ok"``, ``"fail"``, or ``"2fa_challenged"`` — a closed set
            chosen by the route layer, never caller input.
    """
    _inc(
        "mercury_platform_logins_total",
        "Login attempts by outcome (ok / fail / 2fa_challenged)",
        {"outcome": outcome},
    )


def record_throttle_denial(action: str) -> None:
    """Count one per-action throttle denial (429).

    Args:
        action: The rule name that tripped (e.g. ``"login_ip"``) — drawn from
            the fixed action-rule table, so cardinality is bounded.
    """
    _inc(
        "mercury_platform_throttle_denials_total",
        "Per-action auth throttle denials by rule name",
        {"action": action},
    )


def record_throttle_config_mismatch(action: str) -> None:
    """Count a dispatched throttle action that has no configured rule.

    A mismatch means a request-path ``check()`` would silently allow-on-unknown
    (see ``rate_limit_store.ActionRateLimiter.check``); the boot-time
    cross-check records it here so an operator alert can catch the drift.

    Args:
        action: The dispatched action name missing from the rule table.
    """
    _inc(
        "mercury_platform_throttle_rule_mismatch_total",
        "Dispatched throttle actions missing from the configured rule table",
        {"action": action},
    )


def record_quota_denial(reason: str) -> None:
    """Count one quota denial (429) on a metered route.

    Args:
        reason: ``"requests"`` or ``"compute"`` — which ceiling denied.
    """
    _inc(
        "mercury_platform_quota_denials_total",
        "Per-account quota denials by exhausted ceiling",
        {"reason": reason},
    )


def record_email(outcome: str) -> None:
    """Count one transactional email delivery attempt.

    Args:
        outcome: ``"sent"`` or ``"failed"``.
    """
    _inc(
        "mercury_platform_emails_total",
        "Transactional email delivery attempts by outcome",
        {"outcome": outcome},
    )


def record_maintenance_sweep(results: dict[str, int]) -> None:
    """Record one maintenance sweep's per-step results.

    Each non-negative step count feeds ``…_maintenance_pruned_total{step=…}``
    (for the sealing migration the "pruned" count is the number of secrets
    sealed); a ``-1`` marks a step that errored and feeds
    ``…_maintenance_errors_total{step=…}`` instead. Step names come from
    :func:`omni_mercury_engine.api.maintenance.run_maintenance_sweep` — a
    fixed set, so cardinality is bounded.

    Args:
        results: The sweep's per-step counts.
    """
    for step, count in results.items():
        if count < 0:
            _inc(
                "mercury_platform_maintenance_errors_total",
                "Maintenance sweep steps that errored",
                {"step": step},
            )
            continue
        try:
            counter = _counter(
                "mercury_platform_maintenance_pruned_total",
                "Rows pruned (or TOTP secrets sealed) by maintenance sweeps, per step",
                ("step",),
            )
            if counter is not None:
                counter.labels(step=step).inc(count)
        except Exception:
            # Metrics must never break a sweep; exercised by
            # test_platform_metrics.test_metric_failure_never_breaks_the_caller.
            logger.exception("platform metric for sweep step %s failed to record", step)
