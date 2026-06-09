# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Aggregate dispatcher for ``python -m tools <subcommand>``.

Lists curated operator tools and dispatches to their ``main()``
(or ``_cli()``) entry-point.  Each individual tool also remains
independently invocable via ``python -m <module>`` -- the dispatcher
does not replace that, it adds a single discoverable entry-point that
lists what is available and routes by name.

Two tool locations are supported:

* top-level ``tools/<name>.py`` (e.g. :mod:`tools.lyapunov_validator`),
  used for tools whose entry-point is ``_cli`` and whose import chain
  must remain dependency-light;
* ``omni_mercury_engine.tools.<name>`` (the package's operator-tool
  surface, e.g. :mod:`omni_mercury_engine.tools.sigma_immutable_verifier`),
  whose entry-point is ``main`` and which already ship with full
  certificate-envelope + Ed25519 sidecar infrastructure via
  :mod:`omni_mercury_engine.tools._base`.

Adding a new tool: implement ``main(argv: Sequence[str] | None = None)
-> int`` (or ``_cli(argv) -> int``) and add a ``(name → (module, attr))``
entry to :data:`_REGISTRY` below.  The dispatcher imports the module
lazily so a missing optional dep in one tool does not break
``python -m tools list`` for the others.

Every new tool must ship with:

* tests covering at minimum a CLI smoke path (exit codes + JSON schema),
* runtime API references verified against the actual production
  surface (no assumed function names — a tool that imports a symbol
  the production module does not export is a regression, not a feature),
* documentation under the README's "Reproducible Verification" section.

Untested or scaffolding-only tools must not land here; the registry is
the operator's contract.
"""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Sequence


# Every registered tool's entry-point must satisfy this protocol: it
# accepts an ``argv`` list (the subcommand-strip slice of ``sys.argv``)
# and returns an integer exit code.  Encoding the contract as a
# ``Protocol`` rather than ``Callable[[list[str]], int]`` lets entry-
# points use ``Sequence[str] | None = None`` defaults (the standard
# ``argparse`` idiom) without breaking the type check.
class _ToolEntrypoint(Protocol):
    def __call__(self, argv: Sequence[str] | None = ..., /) -> int: ...


# Registry of operator tools.  Each entry maps a CLI name to
# ``(module_dotted_path, entry_point_attr)``.  The entry-point attr is
# resolved at dispatch time, not import time, so a tool whose import
# chain is broken can still be listed (with an explicit error) instead
# of breaking the whole dispatcher.
_REGISTRY: dict[str, tuple[str, str]] = {
    "lyapunov_validator": ("tools.lyapunov_validator", "_cli"),
    # Universal equation optimizer: inventories the MATH_SPEC equation
    # surfaces, freezes the original equations as an immutable baseline,
    # searches a constrained candidate family under hard ethical / Lyapunov /
    # contraction gates, and emits versioned + rollback artifacts. Dependency-
    # light (stdlib + numpy); entry-point ``_cli``. Tests:
    # ``tests/tools/test_equation_optimizer.py`` (pipeline + CLI smoke).
    "equation_optimizer": ("tools.equation_optimizer", "_cli"),
    # Cryptographic evidence tools graduated from
    # ``omni_mercury_engine.tools.*`` after their behavioural tests
    # landed in ``tests/tools/test_new_tools.py``.  These ship with the
    # standard ``_base.run_tool`` envelope (Ed25519 sidecar, atomic
    # write, ``--require`` semantics) so the dispatcher does not need
    # any per-tool plumbing — it just routes by name.
    "sigma_immutable_verifier": (
        "omni_mercury_engine.tools.sigma_immutable_verifier",
        "main",
    ),
    "pqc_capability_probe": (
        "omni_mercury_engine.tools.pqc_capability_probe",
        "main",
    ),
    "kat_runner_standalone": (
        "omni_mercury_engine.tools.kat_runner_standalone",
        "main",
    ),
}


def _print_help(*, missing: str | None = None) -> None:
    if missing is not None:
        print(f"unknown tool: {missing!r}", file=sys.stderr)
    print(
        "usage: python -m tools <subcommand> [args...]\n\nAvailable subcommands:",
        file=sys.stderr,
    )
    # Print each tool with its actual module path so operators see the
    # correct ``python -m <module>`` invocation for the *registered*
    # location — the registry mixes top-level ``tools.*`` and packaged
    # ``omni_mercury_engine.tools.*`` modules, so a single blanket hint
    # like ``python -m tools.<name>`` would be misleading for the latter.
    name_width = max((len(n) for n in _REGISTRY), default=0)
    for name in sorted(_REGISTRY):
        module_path, _ = _REGISTRY[name]
        print(
            f"  {name:<{name_width}}  (also: python -m {module_path})",
            file=sys.stderr,
        )


def _resolve_entrypoint(name: str) -> _ToolEntrypoint:
    """Resolve ``name`` to its callable entry-point.

    Raises ``ImportError`` if the registered module cannot be imported,
    ``AttributeError`` if the entry-point attribute is missing, and
    ``TypeError`` if the registered attribute is not callable (which
    would otherwise surface as a ``TypeError: 'X' object is not
    callable`` at dispatch time with a less-helpful traceback).  The
    ``cast`` to ``_ToolEntrypoint`` is the type narrowing required by
    the Protocol contract — ``getattr`` returns ``Any`` and the explicit
    callability check above is what justifies the narrowing.
    """
    module_path, attr = _REGISTRY[name]
    mod = importlib.import_module(module_path)
    fn = getattr(mod, attr, None)
    if fn is None:
        raise AttributeError(f"tool {name!r} module {module_path!r} has no {attr!r} entry-point")
    if not callable(fn):
        raise TypeError(
            f"tool {name!r} entry-point {module_path}.{attr} is not callable "
            f"(got {type(fn).__name__})"
        )
    return cast("_ToolEntrypoint", fn)


def _main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        _print_help()
        return 0 if args else 2
    if args[0] == "list":
        for name in sorted(_REGISTRY):
            print(name)
        return 0

    name, *rest = args
    if name not in _REGISTRY:
        _print_help(missing=name)
        return 2
    try:
        fn = _resolve_entrypoint(name)
    except (ImportError, AttributeError, TypeError) as exc:
        print(
            f"ERROR: cannot dispatch to tool {name!r}: {exc}",
            file=sys.stderr,
        )
        return 2
    rc = fn(rest)
    return int(rc) if rc is not None else 0


if __name__ == "__main__":  # pragma: no cover - package entry-point
    raise SystemExit(_main())
