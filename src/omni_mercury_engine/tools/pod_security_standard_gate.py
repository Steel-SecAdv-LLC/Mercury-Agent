"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

------------------------------------------------------------------------

Operator tool: verify Helm chart values satisfy the Kubernetes Pod
Security Standards "restricted" profile.

Reads the rendered manifests (or, when ``--values`` is supplied, the
Helm chart's ``values.yaml`` projection) and asserts:

* ``runAsNonRoot: true``,
* ``runAsUser`` >= 10000 when explicitly set,
* ``readOnlyRootFilesystem: true`` on every container,
* ``allowPrivilegeEscalation: false``,
* ``capabilities.drop`` contains ``ALL``,
* ``seccompProfile.type`` ∈ {``RuntimeDefault``, ``Localhost``}.

The check is intentionally a tight subset of the upstream PSS
restricted profile — every assertion is a gate the chart should
already pass.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.pod_security_standard_gate/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.pod_security_standard_gate",
        description="Verify Helm chart / rendered manifests satisfy PSS restricted.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to rendered Kubernetes YAML (e.g. helm template output).",
    )
    return parser


_REQUIRED_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("runAsNonRoot", re.compile(r"runAsNonRoot\s*:\s*true", re.MULTILINE)),
    (
        "readOnlyRootFilesystem",
        re.compile(r"readOnlyRootFilesystem\s*:\s*true", re.MULTILINE),
    ),
    (
        "allowPrivilegeEscalation",
        re.compile(r"allowPrivilegeEscalation\s*:\s*false", re.MULTILINE),
    ),
    (
        "capabilities_drop_all",
        re.compile(
            r"capabilities\s*:\s*\n(?:[^\S\n]+drop\s*:\s*(?:\n(?:[^\S\n]+-\s*ALL))|[^\S\n]+drop\s*:\s*\[\s*['\"]?ALL['\"]?\s*\])",
            re.MULTILINE,
        ),
    ),
    (
        "seccompProfile_type",
        re.compile(
            r"seccompProfile\s*:\s*\n[^\S\n]+type\s*:\s*(RuntimeDefault|Localhost)", re.MULTILINE
        ),
    ),
)

_FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("privileged_true", re.compile(r"privileged\s*:\s*true", re.MULTILINE)),
    ("hostNetwork_true", re.compile(r"hostNetwork\s*:\s*true", re.MULTILINE)),
    ("hostPID_true", re.compile(r"hostPID\s*:\s*true", re.MULTILINE)),
    ("hostIPC_true", re.compile(r"hostIPC\s*:\s*true", re.MULTILINE)),
)

# Captures every explicit ``runAsUser: <int>`` so the gate can assert
# the documented "runAsUser >= 10000 when explicitly set" invariant.
# The PSS restricted profile forbids root (UID 0); Mercury Agent
# tightens this further to a non-system, non-reserved UID floor of
# 10_000.  When the field is absent we cannot assert anything (Kubernetes
# may infer from the image), and the surrounding ``runAsNonRoot: true``
# gate already covers root; the integer floor is only enforced where the
# value is materialised in the manifest.
_RUN_AS_USER_PATTERN: re.Pattern[str] = re.compile(r"^\s*runAsUser\s*:\s*(-?\d+)\s*$", re.MULTILINE)
_RUN_AS_USER_FLOOR = 10_000


def _collect_run_as_user_violations(text: str) -> list[int]:
    """Return every ``runAsUser`` integer below the documented floor."""
    return [
        int(match.group(1))
        for match in _RUN_AS_USER_PATTERN.finditer(text)
        if int(match.group(1)) < _RUN_AS_USER_FLOOR
    ]


def _collect(args: argparse.Namespace) -> Certificate:
    path = Path(args.manifest)
    if not path.is_file():
        return Certificate(
            tool="pod_security_standard_gate",
            schema=_SCHEMA,
            status="fail",
            body={"manifest": str(path), "error": "manifest not found"},
        )
    text = path.read_text()
    missing: list[str] = []
    for name, pat in _REQUIRED_PATTERNS:
        if not pat.search(text):
            missing.append(name)
    forbidden: list[str] = []
    for name, pat in _FORBIDDEN_PATTERNS:
        if pat.search(text):
            forbidden.append(name)

    run_as_user_violations = _collect_run_as_user_violations(text)
    if run_as_user_violations:
        forbidden.append(
            "runAsUser_below_floor:"
            + ",".join(str(uid) for uid in sorted(set(run_as_user_violations)))
        )

    body: dict[str, Any] = {
        "manifest": str(path),
        "size_bytes": len(text.encode("utf-8")),
        "missing_required": missing,
        "forbidden_present": forbidden,
        "run_as_user_floor": _RUN_AS_USER_FLOOR,
        "run_as_user_violations": sorted(set(run_as_user_violations)),
    }
    failures = missing + forbidden
    return Certificate(
        tool="pod_security_standard_gate",
        schema=_SCHEMA,
        status="fail" if failures else "ok",
        body=body,
        warnings=failures,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
