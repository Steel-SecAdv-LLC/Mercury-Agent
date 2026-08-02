# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Operator tool: workflow-version drift gate.

Verifies that the ``ama-cryptography`` git ref pinned in
``pyproject.toml`` matches the ``AMA_REF`` env-var pinned in every
workflow that builds the AMA native library:

* ``.github/workflows/ci.yml``
* ``.github/workflows/pqc-production-check.yml``
* any other workflow that defines ``AMA_REF:``

We just hit this manually (AMA v4.0.0 vs v2.0).  A pre-commit / CI
gate turns the manual check into a structural one.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.workflow_version_drift_gate/v1"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.workflow_version_drift_gate",
        description=(
            "Assert pyproject.toml and every workflow's AMA_REF reference "
            "the same ama-cryptography git tag."
        ),
    )
    parser.add_argument(
        "--root",
        default=str(_REPO_ROOT),
        help="Repository root (default: detected from this file).",
    )
    return parser


# Regex for the pyproject.toml dependency line:
#   "ama-cryptography @ git+https://github.com/.../AMA-Cryptography.git@v4.0.0",
_PYPROJECT_PATTERN = re.compile(
    r"ama-cryptography\s*@\s*git\+https?://[^@]+@(?P<ref>[^\s\",]+)",
    re.IGNORECASE,
)
# Regex for the workflow env-var:
#   AMA_REF: v4.0.0
_WORKFLOW_PATTERN = re.compile(r"^\s*AMA_REF:\s*['\"]?(?P<ref>\S+?)['\"]?\s*$", re.MULTILINE)


def _scan_pyproject(root: Path) -> dict[str, Any]:
    path = root / "pyproject.toml"
    if not path.exists():
        return {"path": str(path), "error": "pyproject.toml not found"}
    text = path.read_text()
    refs = []
    for ln, line in enumerate(text.splitlines(), start=1):
        m = _PYPROJECT_PATTERN.search(line)
        if m:
            refs.append({"line": ln, "ref": m.group("ref")})
    return {"path": "pyproject.toml", "refs": refs}


def _scan_workflows(root: Path) -> list[dict[str, Any]]:
    wf_dir = root / ".github" / "workflows"
    if not wf_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(wf_dir.glob("*.yml")):
        text = path.read_text()
        refs = []
        for m in _WORKFLOW_PATTERN.finditer(text):
            line_no = text.count("\n", 0, m.start()) + 1
            refs.append({"line": line_no, "ref": m.group("ref")})
        if refs:
            out.append({"path": str(path.relative_to(root)), "refs": refs})
    return out


def _collect(args: argparse.Namespace) -> Certificate:
    root = Path(args.root).resolve()
    pyproject = _scan_pyproject(root)
    workflows = _scan_workflows(root)

    all_refs: set[str] = set()
    for r in pyproject.get("refs", []):
        all_refs.add(r["ref"])
    for wf in workflows:
        for r in wf["refs"]:
            all_refs.add(r["ref"])

    body: dict[str, Any] = {
        "root": str(root),
        "pyproject": pyproject,
        "workflows": workflows,
        "distinct_refs": sorted(all_refs),
    }

    warnings: list[str] = []
    if "error" in pyproject:
        warnings.append(pyproject["error"])
        status = "fail"
    elif not all_refs:
        warnings.append("no ama-cryptography refs found anywhere — pyproject pin missing?")
        status = "fail"
    elif len(all_refs) > 1:
        warnings.append(
            f"AMA git ref drift: {sorted(all_refs)} — pyproject.toml and every "
            "workflow's AMA_REF must reference the same tag"
        )
        status = "fail"
    else:
        status = "ok"

    return Certificate(
        tool="workflow_version_drift_gate",
        schema=_SCHEMA,
        status=status,
        body=body,
        warnings=warnings,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
