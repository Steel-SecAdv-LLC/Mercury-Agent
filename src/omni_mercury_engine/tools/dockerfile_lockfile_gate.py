# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.dockerfile_lockfile_gate/v1"

# Unpinned install heuristics.  We deliberately allow ``apt-get update``
# (which is required to refresh the index) but require any ``install``
# directive to use ``=`` for the package version.
_APT_INSTALL = re.compile(
    r"\bapt(?:-get)?\s+install\s+([^\n]+)",
    re.IGNORECASE,
)
# Only ``apk add`` is version-pinnable.  ``apk update`` (and ``apk
# upgrade``) refresh the package index but take no version-pinnable
# arguments — matching them here produced false positives that
# flagged switch tokens from neighbouring ``apk update`` lines as
# unpinned packages.  Operator flags such as ``--no-cache`` or
# ``--virtual <name>`` on ``apk add`` itself are stripped by the
# argument tokeniser (``_check_apk``), so we only need to anchor on
# the ``apk add`` keyword pair here.
_APK_INSTALL = re.compile(r"\bapk\s+add\b\s+([^\n]+)", re.IGNORECASE)
_PIP_INSTALL = re.compile(r"\bpip\s+install\s+([^\n]+)", re.IGNORECASE)
_FROM_DIGEST = re.compile(r"^FROM\s+\S+@sha256:[0-9a-f]{64}", re.IGNORECASE | re.MULTILINE)
_FROM_LINE = re.compile(
    r"^FROM\s+([^\s@]+)(?:@(sha256:[0-9a-f]{64}))?", re.IGNORECASE | re.MULTILINE
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.dockerfile_lockfile_gate",
        description="Enforce digest/version pins on every Dockerfile install.",
    )
    parser.add_argument("--dockerfile", default="Dockerfile")
    return parser


def _check_apt(arg: str) -> list[str]:
    issues: list[str] = []
    for tok in arg.split():
        if tok in {"-y", "--yes", "--no-install-recommends", "--", "&&"}:
            continue
        if tok.startswith("-"):
            continue
        if "=" not in tok:
            issues.append(f"apt package not version-pinned: {tok}")
    return issues


def _check_apk(arg: str) -> list[str]:
    issues: list[str] = []
    for tok in arg.split():
        if tok.startswith("-") or tok in {"--no-cache", "--no-progress"}:
            continue
        if "=" not in tok and "~=" not in tok:
            issues.append(f"apk package not version-pinned: {tok}")
    return issues


def _check_pip(arg: str) -> list[str]:
    issues: list[str] = []
    for tok in arg.split():
        if tok.startswith("-") or tok in {"--no-cache-dir", "--require-hashes"}:
            continue
        if tok.startswith(("/", ".", "git+")):
            continue
        if "==" not in tok and not tok.endswith(".txt"):
            issues.append(f"pip package not version-pinned: {tok}")
    return issues


def _collect(args: argparse.Namespace) -> Certificate:
    path = Path(args.dockerfile)
    if not path.is_file():
        return Certificate(
            tool="dockerfile_lockfile_gate",
            schema=_SCHEMA,
            status="fail",
            body={"dockerfile": str(path), "error": "Dockerfile not found"},
        )
    text = path.read_text()
    issues: list[str] = []

    base_images: list[dict[str, Any]] = []
    digest_pinned: list[bool] = []
    for m in _FROM_LINE.finditer(text):
        ref, digest = m.group(1), m.group(2)
        base_images.append({"image": ref, "digest": digest})
        digest_pinned.append(bool(digest))
    if base_images and not all(digest_pinned):
        for img in base_images:
            if not img["digest"]:
                issues.append(f"FROM line not digest-pinned: {img['image']}")

    for m in _APT_INSTALL.finditer(text):
        issues.extend(_check_apt(m.group(1)))
    for m in _APK_INSTALL.finditer(text):
        issues.extend(_check_apk(m.group(1)))
    for m in _PIP_INSTALL.finditer(text):
        issues.extend(_check_pip(m.group(1)))

    body: dict[str, Any] = {
        "dockerfile": str(path),
        "base_images": base_images,
        "issues": sorted(set(issues)),
    }
    return Certificate(
        tool="dockerfile_lockfile_gate",
        schema=_SCHEMA,
        status="fail" if issues else "ok",
        body=body,
        warnings=sorted(set(issues)),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
