"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.

------------------------------------------------------------------------

Operator tool: diff the public ``omni_mercury_engine.*`` re-export
surface between two refs (or between HEAD and a saved snapshot).

Catches accidental ABI-breaking removals before release.  Operates
purely on Python introspection so it requires no special parser.

Two modes:

* ``--snapshot path.json``  — emit a snapshot of the current public
  surface (top-level public attributes of ``omni_mercury_engine``).
* ``--against path.json``  — diff the current public surface against
  the saved snapshot.  Any removal is a hard finding.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.api_contract_diff/v1"
_MODULE = "omni_mercury_engine"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.api_contract_diff",
        description=(
            "Snapshot or diff the public re-export surface of "
            "omni_mercury_engine to catch ABI-breaking removals."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--snapshot",
        metavar="PATH",
        help="Emit a JSON snapshot of the current public surface to PATH.",
    )
    group.add_argument(
        "--against",
        metavar="PATH",
        help="Diff the current public surface against the snapshot at PATH.",
    )
    return parser


def _classify(obj: Any) -> str:
    if inspect.isclass(obj):
        return "class"
    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return "function"
    if inspect.ismodule(obj):
        return "module"
    return type(obj).__name__


def _signature(obj: Any) -> str | None:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


def _snapshot_surface() -> dict[str, Any]:
    mod = importlib.import_module(_MODULE)
    public = getattr(mod, "__all__", None)
    if public is None:
        public = sorted(
            name for name in dir(mod) if not name.startswith("_")
        )
    entries: dict[str, dict[str, Any]] = {}
    for name in sorted(public):
        obj = getattr(mod, name, None)
        if obj is None:
            continue
        entries[name] = {
            "kind": _classify(obj),
            "signature": _signature(obj),
            "module": getattr(obj, "__module__", None),
            "qualname": getattr(obj, "__qualname__", name),
        }
    return {
        "module": _MODULE,
        "version": getattr(mod, "__version__", None),
        "public_count": len(entries),
        "entries": entries,
    }


def _diff(saved: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    saved_e = saved.get("entries", {})
    current_e = current.get("entries", {})
    removed = sorted(set(saved_e) - set(current_e))
    added = sorted(set(current_e) - set(saved_e))
    changed: list[dict[str, Any]] = []
    for name in sorted(set(saved_e) & set(current_e)):
        s, c = saved_e[name], current_e[name]
        if s.get("signature") != c.get("signature") or s.get("kind") != c.get("kind"):
            changed.append({"name": name, "before": s, "after": c})
    return {
        "removed": removed,
        "added": added,
        "changed": changed,
    }


def _collect(args: argparse.Namespace) -> Certificate:
    if args.snapshot:
        snap = _snapshot_surface()
        Path(args.snapshot).write_text(json.dumps(snap, indent=2, sort_keys=True))
        return Certificate(
            tool="api_contract_diff",
            schema=_SCHEMA,
            status="ok",
            body={"mode": "snapshot", "output": args.snapshot, **snap},
        )

    saved = json.loads(Path(args.against).read_text())
    current = _snapshot_surface()
    diff = _diff(saved, current)
    warnings: list[str] = []
    for name in diff["removed"]:
        warnings.append(f"removed public symbol: {name}")
    for entry in diff["changed"]:
        warnings.append(
            f"signature changed: {entry['name']} "
            f"{entry['before'].get('signature')} → {entry['after'].get('signature')}"
        )
    status = "fail" if (diff["removed"] or diff["changed"]) else "ok"
    return Certificate(
        tool="api_contract_diff",
        schema=_SCHEMA,
        status=status,
        body={"mode": "diff", "against": args.against, "current": current, "diff": diff},
        warnings=warnings,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
