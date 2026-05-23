"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

------------------------------------------------------------------------

Operator tool: reproducible-build probe.

Drives a wheel build twice with ``SOURCE_DATE_EPOCH`` pinned and
compares the digests.  Bit-reproducible builds are the supply-chain
primitive every other gate (SBOM, SLSA, signed_release_bundle) tacitly
assumes — this tool makes the assumption auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.reproducible_build_probe/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.reproducible_build_probe",
        description="Build twice with SOURCE_DATE_EPOCH pinned and diff digests.",
    )
    parser.add_argument("--project-root", default=".", help="Repo root to build.")
    parser.add_argument("--source-date-epoch", type=int, default=1_700_000_000)
    parser.add_argument(
        "--build-cmd",
        default="python -m build --wheel",
        help="Build command to execute in each iteration.",
    )
    return parser


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_once(project_root: Path, build_cmd: str, source_date_epoch: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["SOURCE_DATE_EPOCH"] = str(source_date_epoch)
    env["PYTHONHASHSEED"] = "0"
    env["TZ"] = "UTC"
    tmp = Path(tempfile.mkdtemp(prefix="mercury-build-"))
    try:
        # Run the build with --outdir to keep the wheel scoped to ``tmp``.
        cmd = build_cmd.split() + ["--outdir", str(tmp)]
        proc = subprocess.run(
            cmd, cwd=str(project_root), env=env, capture_output=True, text=True, check=False
        )
        if proc.returncode != 0:
            return {
                "ok": False,
                "stdout": proc.stdout[-2000:],
                "stderr": proc.stderr[-2000:],
                "returncode": proc.returncode,
            }
        wheels = sorted(tmp.glob("*.whl"))
        if not wheels:
            return {"ok": False, "stderr": "no wheel produced", "returncode": 0}
        return {
            "ok": True,
            "wheel": wheels[0].name,
            "sha256": _sha256_file(wheels[0]),
            "size": wheels[0].stat().st_size,
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _collect(args: argparse.Namespace) -> Certificate:
    project = Path(args.project_root).resolve()
    first = _build_once(project, args.build_cmd, args.source_date_epoch)
    if not first.get("ok"):
        return Certificate(
            tool="reproducible_build_probe",
            schema=_SCHEMA,
            status="fail",
            body={"project_root": str(project), "first": first},
            warnings=["first build failed"],
        )
    second = _build_once(project, args.build_cmd, args.source_date_epoch)
    body: dict[str, Any] = {
        "project_root": str(project),
        "source_date_epoch": int(args.source_date_epoch),
        "build_cmd": args.build_cmd,
        "first": first,
        "second": second,
    }
    if not second.get("ok"):
        return Certificate(
            tool="reproducible_build_probe",
            schema=_SCHEMA,
            status="fail",
            body=body,
            warnings=["second build failed"],
        )
    if first["sha256"] != second["sha256"]:
        return Certificate(
            tool="reproducible_build_probe",
            schema=_SCHEMA,
            status="fail",
            body=body,
            warnings=[f"wheel digest drift: {first['sha256']} != {second['sha256']}"],
        )
    return Certificate(
        tool="reproducible_build_probe",
        schema=_SCHEMA,
        status="ok",
        body=body,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
