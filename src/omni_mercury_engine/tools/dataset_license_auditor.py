# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

from __future__ import annotations

import argparse
import importlib
import pkgutil
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.dataset_license_auditor/v1"
_REQUIRED_FIELDS: frozenset[str] = frozenset({"spdx", "upstream_url", "redistribution"})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.dataset_license_auditor",
        description="Verify every loader declares an upstream licence + SPDX expression.",
    )
    parser.add_argument(
        "--package",
        default="omni_mercury_engine.loaders",
        help="Loader package to walk.",
    )
    return parser


def _collect(args: argparse.Namespace) -> Certificate:
    try:
        pkg = importlib.import_module(args.package)
    except ImportError as exc:
        return Certificate(
            tool="dataset_license_auditor",
            schema=_SCHEMA,
            status="warn",
            body={"package": args.package, "error": str(exc)},
            warnings=[f"loader package not importable: {exc}"],
        )

    loaders: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for m in pkgutil.iter_modules(pkg.__path__):
        if m.name.startswith("_"):
            continue
        full = f"{args.package}.{m.name}"
        try:
            mod = importlib.import_module(full)
        except ImportError as exc:
            missing.append(f"{full}: import failed: {exc}")
            continue
        lic = getattr(mod, "DATASET_LICENSE", None)
        if not isinstance(lic, dict):
            missing.append(f"{full}: missing DATASET_LICENSE")
            continue
        if not _REQUIRED_FIELDS.issubset(lic.keys()):
            absent = sorted(_REQUIRED_FIELDS - set(lic.keys()))
            missing.append(f"{full}: DATASET_LICENSE missing fields {absent}")
            continue
        loaders[full] = {k: str(v) for k, v in lic.items()}

    body: dict[str, Any] = {
        "package": args.package,
        "loaders": loaders,
        "loader_count": len(loaders),
        "missing": missing,
    }
    if missing:
        return Certificate(
            tool="dataset_license_auditor",
            schema=_SCHEMA,
            status="fail",
            body=body,
            warnings=missing,
        )
    return Certificate(
        tool="dataset_license_auditor",
        schema=_SCHEMA,
        status="ok",
        body=body,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
