"""Aggregate dispatcher for ``python -m tools <subcommand>``.

Lists every operator tool under :mod:`tools` and dispatches to its
``main()`` (or ``_cli()``) entry-point.  Each individual tool also
remains independently invocable via ``python -m tools.<name>`` -- the
dispatcher does not replace that, it adds a single discoverable
entry-point that lists what is available and routes by name.

Adding a new tool: implement ``main(argv: Sequence[str] | None = None)
-> int`` (or ``_cli(argv) -> int``) in ``tools/<your_tool>.py`` and add
the module name to :data:`_REGISTRY` below.  The dispatcher imports the
module lazily so a missing optional dep in one tool does not break
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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# Registry of operator tools.  Each entry maps a CLI name to
# ``(module_dotted_path, entry_point_attr)``.  The entry-point attr is
# resolved at dispatch time, not import time, so a tool whose import
# chain is broken can still be listed (with an explicit error) instead
# of breaking the whole dispatcher.
_REGISTRY: dict[str, tuple[str, str]] = {
    "lyapunov_validator": ("tools.lyapunov_validator", "_cli"),
}


def _print_help(*, missing: str | None = None) -> None:
    if missing is not None:
        print(f"unknown tool: {missing!r}", file=sys.stderr)
    print(
        "usage: python -m tools <subcommand> [args...]\n" "\n" "Available subcommands:",
        file=sys.stderr,
    )
    for name in sorted(_REGISTRY):
        print(f"  {name}", file=sys.stderr)
    print(
        "\n" "Each subcommand also accepts ``python -m tools.<name>`` directly.",
        file=sys.stderr,
    )


def _resolve_entrypoint(name: str) -> object:
    module_path, attr = _REGISTRY[name]
    mod = importlib.import_module(module_path)
    fn = getattr(mod, attr, None)
    if fn is None:
        raise AttributeError(f"tool {name!r} module {module_path!r} has no {attr!r} entry-point")
    return fn


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
    except (ImportError, AttributeError) as exc:
        print(
            f"ERROR: cannot dispatch to tool {name!r}: {exc}",
            file=sys.stderr,
        )
        return 2
    rc = fn(rest)
    return int(rc) if rc is not None else 0


if __name__ == "__main__":  # pragma: no cover - package entry-point
    raise SystemExit(_main())
