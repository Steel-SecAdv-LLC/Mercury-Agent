# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Operator tool: workflow-version drift gate.

Verifies that the ``ama-cryptography`` git ref pinned in
``pyproject.toml`` matches the AMA ref pinned in every workflow that
builds the AMA native library, and in the shared composite action's
default:

* ``.github/workflows/*.yml`` — the ``ama-ref:`` input passed to the
  ``build-ama-cryptography`` composite action (older workflows used an
  ``AMA_REF:`` env var; both forms are recognised).
* ``.github/actions/*/action.yml`` — the ``ama-ref`` input *default*.

We just hit this manually (AMA v3.3.0 vs v2.0).  A pre-commit / CI
gate turns the manual check into a structural one.

The scan matches only version-like values (``v4.0.0`` / ``4.0.0``), so a
templated ``AMA_REF: ${{ inputs.ama-ref }}`` inside the composite action is
skipped rather than parsed as a bogus ref. And because the workflows migrated
from the ``AMA_REF:`` env var to the ``ama-ref:`` action input, the gate now
FAILS when a build-AMA workflow is present but no ref parses — otherwise a
future key rename would silently make it vacuous (verify the pin against an
empty set and always pass), which is exactly the state this comment prevents.
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
# Regex for the AMA ref pinned in a workflow: either the modern composite-action
# input ``ama-ref: v4.0.0`` or the legacy env var ``AMA_REF: v4.0.0``. The value
# must be version-like (optional ``v`` then a digit) so a templated
# ``${{ inputs.ama-ref }}`` is not parsed as a ref.
_WORKFLOW_PATTERN = re.compile(
    r"^\s*(?:AMA_REF|ama-ref):\s*['\"]?(?P<ref>v?[0-9][^\s'\"]*)['\"]?\s*$",
    re.MULTILINE,
)
# Marker that a workflow actually builds AMA (invokes the composite action), used
# to distinguish "no AMA workflows" from "AMA workflows whose ref no longer
# parses" -- the latter must fail rather than pass vacuously.
_BUILD_AMA_MARKER = "build-ama-cryptography"


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


def _scan_ref_file(path: Path, root: Path) -> tuple[dict[str, Any] | None, bool]:
    """Scan one YAML file for AMA refs and whether it builds AMA.

    Returns ``(entry_or_None, builds_ama)``: the entry is ``None`` when the file
    carries no parsed ref, and ``builds_ama`` is True when the file invokes the
    build-AMA composite action (so a missing ref is a failure, not a no-op).
    """
    text = path.read_text()
    refs = [
        {"line": text.count("\n", 0, m.start()) + 1, "ref": m.group("ref")}
        for m in _WORKFLOW_PATTERN.finditer(text)
    ]
    builds_ama = _BUILD_AMA_MARKER in text
    entry = {"path": str(path.relative_to(root)), "refs": refs} if refs else None
    return entry, builds_ama


def _scan_workflows(root: Path) -> tuple[list[dict[str, Any]], bool]:
    """Scan workflows and the composite action for AMA refs.

    Returns ``(entries, builds_ama_seen)`` where ``builds_ama_seen`` is True when
    at least one scanned file invokes the build-AMA action.
    """
    out: list[dict[str, Any]] = []
    builds_ama_seen = False
    wf_dir = root / ".github" / "workflows"
    if wf_dir.is_dir():
        for path in sorted(wf_dir.glob("*.yml")):
            entry, builds_ama = _scan_ref_file(path, root)
            builds_ama_seen = builds_ama_seen or builds_ama
            if entry:
                out.append(entry)
    # The composite action's own default (``ama-ref``) is the pin used when a
    # workflow omits the input; verify it agrees too.
    actions_dir = root / ".github" / "actions"
    if actions_dir.is_dir():
        for path in sorted(actions_dir.glob("*/action.yml")):
            entry, _ = _scan_ref_file(path, root)
            if entry:
                out.append(entry)
    return out, builds_ama_seen


def _collect(args: argparse.Namespace) -> Certificate:
    root = Path(args.root).resolve()
    pyproject = _scan_pyproject(root)
    workflows, builds_ama_seen = _scan_workflows(root)

    all_refs: set[str] = set()
    for r in pyproject.get("refs", []):
        all_refs.add(r["ref"])
    workflow_ref_count = 0
    for wf in workflows:
        for r in wf["refs"]:
            all_refs.add(r["ref"])
            workflow_ref_count += 1

    body: dict[str, Any] = {
        "root": str(root),
        "pyproject": pyproject,
        "workflows": workflows,
        "builds_ama_seen": builds_ama_seen,
        "distinct_refs": sorted(all_refs),
    }

    warnings: list[str] = []
    if "error" in pyproject:
        warnings.append(pyproject["error"])
        status = "fail"
    elif not all_refs:
        warnings.append("no ama-cryptography refs found anywhere — pyproject pin missing?")
        status = "fail"
    elif builds_ama_seen and workflow_ref_count == 0:
        # A workflow (or the composite action) builds AMA, but not one AMA ref
        # parsed from any of them. That means the pattern no longer matches the
        # key the workflows use, so the gate would "verify" the pyproject pin
        # against an empty set and pass vacuously. Fail instead of rubber-stamping.
        warnings.append(
            "workflows invoke the build-AMA action but no AMA ref parsed from any "
            "workflow or composite action — the drift gate cannot verify the "
            "workflow pins (has the 'ama-ref:'/'AMA_REF:' key been renamed?)"
        )
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
