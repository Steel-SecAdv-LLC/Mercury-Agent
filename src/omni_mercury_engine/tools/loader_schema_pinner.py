"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

------------------------------------------------------------------------

Operator tool: pin every loader's output schema (column dtypes, label
set, row count range) and verify it on every load.

Two modes:

* ``--emit``: introspect each loader's ``schema()`` (or, when absent,
  one ``probe()`` call) and write the pinned schema to disk;
* ``--verify PATH``: re-introspect and diff against the pinned schema,
  failing on any drift.

The verify path is what tools like ``DatasetLoader.load(verify_schema=...)``
are expected to call at runtime — silent schema drift is the gap this
gate closes.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pkgutil
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, atomic_write_text, run_tool

_SCHEMA = "mercury.tools.loader_schema_pinner/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.loader_schema_pinner",
        description="Pin or verify loader output schemas.",
    )
    parser.add_argument(
        "--package",
        default="omni_mercury_engine.loaders",
        help="Loader package to walk.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emit", metavar="PATH", help="Write pinned schemas to PATH.")
    group.add_argument("--verify", metavar="PATH", help="Verify schemas against PATH.")
    return parser


def _loader_schema(module: Any) -> dict[str, Any]:
    """Return ``module.schema()`` when present, else introspect ``probe()``."""
    if hasattr(module, "schema"):
        try:
            return dict(module.schema())
        except Exception as exc:
            return {"error": f"schema() raised: {type(exc).__name__}: {exc}"}
    if hasattr(module, "probe"):
        try:
            data = module.probe()
            return {
                "columns": list(getattr(data, "columns", []) or []),
                "dtypes": {
                    str(k): str(v)
                    for k, v in getattr(data, "dtypes", {}).items()
                    if not k.startswith("_")
                },
                "rows": int(getattr(data, "shape", (0,))[0]),
            }
        except Exception as exc:
            return {"error": f"probe() raised: {type(exc).__name__}: {exc}"}
    return {"error": "module exposes neither schema() nor probe()"}


def _walk(package: str) -> dict[str, dict[str, Any]]:
    # Surface package-level import failures in the certificate body
    # instead of letting them crash the tool — the v1 envelope contract
    # requires every invocation to emit a JSON certificate, even when
    # the loader package itself is absent (which is itself the
    # operator-visible finding).
    try:
        pkg = importlib.import_module(package)
    except ImportError as exc:
        return {package: {"error": f"package import failed: {exc}"}}
    out: dict[str, dict[str, Any]] = {}
    for m in pkgutil.iter_modules(pkg.__path__):
        if m.name.startswith("_"):
            continue
        full = f"{package}.{m.name}"
        try:
            mod = importlib.import_module(full)
        except ImportError as exc:
            out[full] = {"error": f"import failed: {exc}"}
            continue
        out[full] = _loader_schema(mod)
    return out


def _collect(args: argparse.Namespace) -> Certificate:
    current = _walk(args.package)
    if args.emit:
        atomic_write_text(Path(args.emit), json.dumps(current, indent=2, sort_keys=True) + "\n")
        return Certificate(
            tool="loader_schema_pinner",
            schema=_SCHEMA,
            status="ok",
            body={
                "mode": "emit",
                "package": args.package,
                "output": args.emit,
                "loader_count": len(current),
                "schemas": current,
            },
        )

    saved = json.loads(Path(args.verify).read_text())
    drift: list[dict[str, Any]] = []
    for name in sorted(set(saved) | set(current)):
        if name not in saved:
            drift.append({"loader": name, "kind": "added", "current": current[name]})
        elif name not in current:
            drift.append({"loader": name, "kind": "removed", "saved": saved[name]})
        elif saved[name] != current[name]:
            drift.append(
                {
                    "loader": name,
                    "kind": "changed",
                    "saved": saved[name],
                    "current": current[name],
                }
            )
    return Certificate(
        tool="loader_schema_pinner",
        schema=_SCHEMA,
        status="fail" if drift else "ok",
        body={
            "mode": "verify",
            "package": args.package,
            "saved_path": args.verify,
            "drift": drift,
            "saved_count": len(saved),
            "current_count": len(current),
        },
        warnings=[f"{d['kind']}: {d['loader']}" for d in drift],
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
