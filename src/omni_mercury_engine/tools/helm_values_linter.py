# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Operator tool: Helm values security-posture linter.

Validates ``helm/mercury-agent/values.yaml`` against the documented
container-deployment contract:

* resource requests/limits set;
* security context: ``runAsNonRoot: true``, ``readOnlyRootFilesystem: true``,
  ``allowPrivilegeEscalation: false``, drop ALL capabilities;
* network policies present;
* image pinned by digest (not floating tag).

Trivy scans CVEs; this gate audits chart composition.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.helm_values_linter/v1"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_VALUES = _REPO_ROOT / "helm" / "mercury-agent" / "values.yaml"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.helm_values_linter",
        description="Lint helm/mercury-agent/values.yaml for security posture.",
    )
    parser.add_argument(
        "--values",
        default=str(_DEFAULT_VALUES),
        help="Path to values.yaml (default: helm/mercury-agent/values.yaml).",
    )
    return parser


def _dig(d: Any, *keys: str) -> Any:
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _collect(args: argparse.Namespace) -> Certificate:
    path = Path(args.values)
    if not path.exists():
        return Certificate(
            tool="helm_values_linter",
            schema=_SCHEMA,
            status="warn",
            body={"path": str(path), "exists": False},
            warnings=[f"values file not found: {path} (no Helm chart in repo yet)"],
        )
    if importlib.util.find_spec("yaml") is None:
        return Certificate(
            tool="helm_values_linter",
            schema=_SCHEMA,
            status="fail",
            body={"path": str(path)},
            warnings=["PyYAML not installed; cannot parse values.yaml"],
        )
    # PyYAML is declared in ``pyproject.toml`` mypy overrides; no
    # ``type: ignore`` is required.
    import yaml

    doc = yaml.safe_load(path.read_text()) or {}

    findings: list[str] = []

    # Image pinning by digest
    img = _dig(doc, "image") or {}
    tag = str(img.get("tag", ""))
    digest = str(img.get("digest", ""))
    if not digest and not tag.startswith("sha256:"):
        findings.append("image.digest not set (image is not pinned by content digest)")

    # Resource requests + limits
    resources = _dig(doc, "resources") or {}
    for kind in ("requests", "limits"):
        section = resources.get(kind) or {}
        for k in ("cpu", "memory"):
            if k not in section:
                findings.append(f"resources.{kind}.{k} missing")

    # Pod security context
    sec = _dig(doc, "securityContext") or _dig(doc, "podSecurityContext") or {}
    if not sec.get("runAsNonRoot"):
        findings.append("securityContext.runAsNonRoot must be true")
    if sec.get("runAsUser", 0) == 0:
        findings.append("securityContext.runAsUser is root (0)")

    # Container security context
    csec = _dig(doc, "containerSecurityContext") or {}
    if not csec.get("readOnlyRootFilesystem", False):
        findings.append("containerSecurityContext.readOnlyRootFilesystem must be true")
    if csec.get("allowPrivilegeEscalation", True):
        findings.append("containerSecurityContext.allowPrivilegeEscalation must be false")
    cap_drop = (csec.get("capabilities") or {}).get("drop") or []
    if "ALL" not in cap_drop:
        findings.append("containerSecurityContext.capabilities.drop must include ALL")

    # Network policies
    if not _dig(doc, "networkPolicy", "enabled"):
        findings.append("networkPolicy.enabled is not true")

    body: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "image": img,
        "findings": findings,
    }
    status = "fail" if findings else "ok"
    return Certificate(
        tool="helm_values_linter",
        schema=_SCHEMA,
        status=status,
        body=body,
        warnings=findings,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
