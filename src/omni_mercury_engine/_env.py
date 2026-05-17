"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

Runtime environment-mode primitive.

Mercury historically had no single source of truth for "am I running in
production?".  Individual modules invented ad-hoc env vars
(``AMA_REQUIRE_REAL_PQC``, ``MERCURY_PQC_REAL_AMA``,
``_GOSNN_TESTING_BYPASS``) for their own fail-closed gates, which works
for narrowly-scoped contracts (PQC, GOSNN) but does not scale to "every
optional collaborator that has a development-friendly stub should
hard-fail in production".

This module introduces a single canonical environment-mode flag,
``MERCURY_ENV``, and a tiny helper API consumed by any module that
wants to refuse to silently degrade in production.  The PQC import
gate (``_pqc_gate._enforce_pqc_production_gate``) stays orthogonal
because PQC has its own hard-required-build contract independent of
the development/production distinction; ``MERCURY_ENV`` is the broader
"don't let optional integrations silently drop to stub mode" switch.

Contract
========

``MERCURY_ENV`` accepts one of:

- ``"development"`` (default when the variable is unset or empty) —
  optional integrations may log warnings and continue with reduced
  functionality.
- ``"production"`` — any optional integration that would otherwise
  degrade silently MUST raise
  :class:`MercuryProductionConfigError` instead.

Any other value raises :class:`MercuryProductionConfigError` at the
first call to :func:`get_mercury_env`, so a typo in deployment
configuration is loud, not a silent fall-through to development mode.

Public API
==========

- :data:`MERCURY_ENV_VAR` — the env var name, exported as a constant
  so call sites and tests don't repeat the string literal.
- :func:`get_mercury_env` — returns the validated mode string.
- :func:`is_production` — convenience boolean.
- :func:`require_real_component` — call site helper that raises
  :class:`MercuryProductionConfigError` in production with a uniform
  message format and a remediation hint.
- :class:`MercuryProductionConfigError` — raised when a production
  process attempts to use a stub/mock collaborator.

The module is **import-safe**: it has no third-party dependencies, no
side effects at import time, and never reads the environment at
import time — every read goes through :func:`get_mercury_env`, which
re-reads the env on every call so tests can use
``monkeypatch.setenv`` without process restarts.
"""

from __future__ import annotations

import os

__all__ = [
    "MERCURY_ENV_DEVELOPMENT",
    "MERCURY_ENV_PRODUCTION",
    "MERCURY_ENV_VAR",
    "MercuryProductionConfigError",
    "get_mercury_env",
    "is_production",
    "require_real_component",
]


MERCURY_ENV_VAR = "MERCURY_ENV"
"""Canonical environment-mode env var name."""

MERCURY_ENV_DEVELOPMENT = "development"
"""Default development mode — optional integrations may degrade with a warning."""

MERCURY_ENV_PRODUCTION = "production"
"""Production mode — optional integrations must be configured or fail closed."""

_VALID_MODES: frozenset[str] = frozenset({MERCURY_ENV_DEVELOPMENT, MERCURY_ENV_PRODUCTION})


class MercuryProductionConfigError(RuntimeError):
    """Raised when a production process is missing a real collaborator.

    Distinct from :class:`ValueError` / :class:`RuntimeError` so callers
    that want to differentiate "operator misconfigured production" from
    "runtime failure of a configured component" can catch this class
    specifically and surface a different exit code or alert.

    Inherits from :class:`RuntimeError` (not :class:`Exception` directly)
    because every existing fail-closed gate in Mercury already raises
    :class:`RuntimeError` subclasses (see ``_pqc_gate``,
    ``security/safe_load``, ``security/sigma_immutable_gate``).  Keeping
    the same base class avoids surprising callers that already broaden
    their except clauses to :class:`RuntimeError`.
    """


def get_mercury_env() -> str:
    """Return the validated Mercury environment mode.

    Returns:
        Either :data:`MERCURY_ENV_DEVELOPMENT` or
        :data:`MERCURY_ENV_PRODUCTION`.  An unset, empty, or whitespace-
        only ``MERCURY_ENV`` value defaults to development.

    Raises:
        MercuryProductionConfigError: If ``MERCURY_ENV`` is set to a
            value other than ``"development"`` or ``"production"``
            (case-insensitive, whitespace-tolerant).  Refusing to
            normalise unknown values is intentional — a deployment
            that sets ``MERCURY_ENV=prod`` (instead of ``production``)
            must fail loudly rather than silently fall through to
            development mode.
    """
    raw = os.environ.get(MERCURY_ENV_VAR, "")
    normalised = raw.strip().lower()
    if not normalised:
        return MERCURY_ENV_DEVELOPMENT
    if normalised not in _VALID_MODES:
        raise MercuryProductionConfigError(
            f"{MERCURY_ENV_VAR}={raw!r} is not a recognised Mercury "
            f"environment mode.  Expected one of: "
            f"{sorted(_VALID_MODES)}.  Unset {MERCURY_ENV_VAR} to use "
            f"the default ({MERCURY_ENV_DEVELOPMENT!r}) or set it "
            f"explicitly to {MERCURY_ENV_PRODUCTION!r}."
        )
    return normalised


def is_production() -> bool:
    """Return ``True`` if Mercury is running in production mode."""
    return get_mercury_env() == MERCURY_ENV_PRODUCTION


def require_real_component(
    component: str,
    *,
    remediation: str,
) -> None:
    """Refuse to continue if ``MERCURY_ENV=production`` and a stub is in use.

    Call this at the top of any code path that would otherwise silently
    instantiate a mock, heuristic-only, or placeholder collaborator.
    In development mode this function is a no-op; in production mode
    it raises :class:`MercuryProductionConfigError` with a uniform
    message containing the component name and the caller-supplied
    remediation hint.

    Args:
        component: Human-readable name of the missing real collaborator
            (e.g. ``"narrative LLM provider"``).
        remediation: One-line instruction telling the operator how to
            configure a real provider.  Will be appended to the error
            message verbatim.

    Raises:
        MercuryProductionConfigError: When :func:`is_production`
            returns ``True``.

    Example:
        >>> require_real_component(
        ...     "narrative LLM provider",
        ...     remediation=(
        ...         "Pass llm_provider=<provider> to MercuryVoice() "
        ...         "or call OmniMercuryEngine.enable_llm_enhancement()."
        ...     ),
        ... )  # raises in production, no-op in development
    """
    if not is_production():
        return
    raise MercuryProductionConfigError(
        f"{MERCURY_ENV_VAR}={MERCURY_ENV_PRODUCTION} but no real "
        f"{component} is configured.  Mercury refuses to silently "
        f"degrade to a stub in production.\n"
        f"Remediation: {remediation}"
    )
