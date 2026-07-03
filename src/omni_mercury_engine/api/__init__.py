# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""REST API for Mercury Agent anomaly detection.

The package is split along the optional ``[api]`` extra boundary so that
``import omni_mercury_engine.api`` never requires FastAPI:

* The **framework-independent** auth surface — native JWT mint/verify, the
  ``User`` / ``Permission`` models, and the ``require_*`` authorization
  decorators — is imported eagerly and works with no optional extras.
* The **HTTP-server** surface — the FastAPI ``app``, the health/voice routers,
  and ``HealthChecker`` — lives behind the optional ``[api]`` extra
  (FastAPI/uvicorn) and is exposed lazily via :pep:`562` ``__getattr__``.
  Accessing one of those names without the extra installed raises an
  actionable :class:`ModuleNotFoundError` pointing at
  ``pip install 'mercury-agent[api]'`` rather than failing the whole
  package import.

Keeping the package importable without FastAPI is what lets the agentic
``api`` subsystem binding (and the in-process ``Eos_XVIII`` onboarding
coordinator) operate in lean installs that do not ship the web server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Framework-independent surface — always importable (FastAPI not required).
from .auth import APIKeyAuth, JWTAuth, Permission, User, require_permission, require_role

if TYPE_CHECKING:
    from .health import HealthChecker, get_health_checker, health_router
    from .server import app
    from .voice import (
        add_voice_routes,
        router as voice_router,
    )

# Lazily-served FastAPI-dependent names: public attr -> (submodule, symbol).
_LAZY: dict[str, tuple[str, str]] = {
    "app": (".server", "app"),
    "HealthChecker": (".health", "HealthChecker"),
    "get_health_checker": (".health", "get_health_checker"),
    "health_router": (".health", "health_router"),
    "add_voice_routes": (".voice", "add_voice_routes"),
    "voice_router": (".voice", "router"),
}

_API_EXTRA_HINT = (
    "Mercury's HTTP-server surface requires the optional [api] extra "
    "(FastAPI/uvicorn). Install it with: pip install 'mercury-agent[api]'"
)

# Import-root names whose absence genuinely means the [api] extra is missing.
# An ImportError for anything else is a real bug and must not be masked.
_API_EXTRA_PACKAGES = frozenset({"fastapi", "starlette", "uvicorn"})


def __getattr__(name: str) -> Any:
    """Lazily import FastAPI-dependent API names (:pep:`562`).

    Keeps ``import omni_mercury_engine.api`` working with no optional extras so
    the framework-independent auth surface — and the agentic ``api`` subsystem
    binding — stay available; surfaces a clear install hint when a server-side
    name is used without the ``[api]`` extra present.
    """
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    submodule, symbol = target
    import importlib

    try:
        module = importlib.import_module(submodule, __name__)
    except ImportError as exc:
        # Only rewrite the error when it is genuinely the missing [api] extra
        # (FastAPI/Starlette/uvicorn). A real ImportError *inside* an api
        # submodule (a bug, or an unrelated missing dependency) must propagate
        # unchanged so it is not masked as "install the [api] extra".
        missing_root = (getattr(exc, "name", "") or "").split(".", 1)[0]
        if missing_root in _API_EXTRA_PACKAGES:
            raise ModuleNotFoundError(
                f"{_API_EXTRA_HINT} (could not resolve {name!r}: {exc})"
            ) from exc
        raise
    return getattr(module, symbol)


def __dir__() -> list[str]:
    """Expose both eager and lazy names to ``dir()`` / tab-completion."""
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    # Auth names are framework-independent (eager); the health/server/voice
    # names require the optional ``[api]`` extra and are served lazily.
    "APIKeyAuth",
    "HealthChecker",
    "JWTAuth",
    "Permission",
    "User",
    "add_voice_routes",
    "app",
    "get_health_checker",
    "health_router",
    "require_permission",
    "require_role",
    "voice_router",
]
