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

Two further surfaces restate the pin and are held to the ``pyproject.toml``
ref by version (a leading ``v`` is not significant there):

* ``src/omni_mercury_engine/_pqc_gate.py`` -- ``_AMA_REQUIRED_VERSION``, the
  version the import-time PQC gate refuses to run without.
* Markdown prose -- any paragraph that mentions AMA and says "pinned to
  ``vX.Y.Z``". ``CHANGELOG.md`` is a historical record and is not scanned.
  Three documents (CONTRIBUTING.md, docs/index.md, rust_crypto/README.md)
  still said v3.3.0 two releases after the pin moved to v4.0.0, which is what
  added this surface.

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
# The runtime pin enforced by the import-time PQC gate.
_RUNTIME_GATE = Path("src") / "omni_mercury_engine" / "_pqc_gate.py"
_RUNTIME_PATTERN = re.compile(
    r"^_AMA_REQUIRED_VERSION\s*=\s*['\"](?P<ref>[^'\"]+)['\"]", re.MULTILINE
)
# A prose pin statement ("pinned to `v4.0.0`", "pinned to **v4.0.0**"). Only
# paragraphs that mention AMA are considered, so pins of unrelated tools in the
# same file are not mistaken for the AMA pin.
_DOC_PIN_PATTERN = re.compile(r"(?i)\bpinned\s+to\s+[`*]*(?P<ref>v?\d+\.\d+\.\d+)")
# Markdown that is a historical record (past pins are preserved, not rewritten)
# or a local dataset cache rather than project documentation.
_DOC_SKIP_FILES = frozenset({"CHANGELOG.md"})
_DOC_SKIP_TOP_DIRS = frozenset({"data", "node_modules"})


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


def _scan_runtime_gate(root: Path) -> dict[str, Any] | None:
    """Return the import-time gate's required AMA version, if the gate exists."""
    path = root / _RUNTIME_GATE
    if not path.is_file():
        return None
    text = path.read_text()
    m = _RUNTIME_PATTERN.search(text)
    if m is None:
        return {"path": str(_RUNTIME_GATE), "error": "_AMA_REQUIRED_VERSION not found"}
    return {
        "path": str(_RUNTIME_GATE),
        "line": text.count("\n", 0, m.start()) + 1,
        "ref": m.group("ref"),
    }


def _scan_docs(root: Path) -> list[dict[str, Any]]:
    """Return every "pinned to vX.Y.Z" statement in an AMA-mentioning paragraph."""
    found: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root)
        if rel.name in _DOC_SKIP_FILES or rel.parts[0] in _DOC_SKIP_TOP_DIRS:
            continue
        if any(part.startswith(".") for part in rel.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for para in re.finditer(r"(?:.+\n?)+", text):
            if "AMA" not in para.group(0):
                continue
            for m in _DOC_PIN_PATTERN.finditer(para.group(0)):
                offset = para.start() + m.start()
                found.append(
                    {
                        "path": str(rel),
                        "line": text.count("\n", 0, offset) + 1,
                        "ref": m.group("ref"),
                    }
                )
    return found


def _version(ref: str) -> str:
    """Compare pins by version: ``v4.0.0`` and ``4.0.0`` name the same release."""
    return ref[1:] if ref[:1] in {"v", "V"} else ref


def _collect(args: argparse.Namespace) -> Certificate:
    root = Path(args.root).resolve()
    pyproject = _scan_pyproject(root)
    workflows, builds_ama_seen = _scan_workflows(root)
    runtime_gate = _scan_runtime_gate(root)
    docs = _scan_docs(root)

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
        "runtime_gate": runtime_gate,
        "docs": docs,
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

    if status == "ok":
        pinned = _version(next(iter(all_refs)))
        restated = ([runtime_gate] if runtime_gate is not None else []) + docs
        stale = [
            entry for entry in restated if "error" in entry or _version(str(entry["ref"])) != pinned
        ]
        for entry in stale:
            if "error" in entry:
                warnings.append(f"{entry['path']}: {entry['error']}")
            else:
                warnings.append(
                    f"{entry['path']}:{entry['line']}: states AMA {entry['ref']} but "
                    f"pyproject.toml pins {pinned}"
                )
        if stale:
            status = "fail"

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
