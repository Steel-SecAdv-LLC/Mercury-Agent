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

Operator tool: runtime-image composition auditor.

Audits what's *actually* inside the Mercury runtime image (or any
filesystem rooted at ``--root``) — non-root user, no dev tools, no
apt cache, correct entrypoint, correct LD_LIBRARY_PATH for AMA.
Trivy scans CVEs; this tool scans composition.

Two modes:

* ``--mode dockerfile`` (default when only ``Dockerfile`` is present)
  — static-scan the repo's Dockerfile for the same posture invariants.
* ``--mode rootfs --root /var/lib/docker/...``  — walk an extracted
  rootfs and check live artefacts.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.image_surface_auditor/v1"
_REPO_ROOT = Path(__file__).resolve().parents[3]

_DEV_TOOLS = ("gcc", "g++", "make", "cmake", "git", "curl", "wget", "vim", "apt", "dpkg")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.image_surface_auditor",
        description=(
            "Audit the runtime image composition: non-root user, no dev "
            "tools, no apt cache, correct entrypoint, AMA LD_LIBRARY_PATH."
        ),
    )
    parser.add_argument(
        "--mode",
        default="dockerfile",
        choices=["dockerfile", "rootfs"],
        help="Audit source (default: dockerfile).",
    )
    parser.add_argument(
        "--dockerfile",
        default=str(_REPO_ROOT / "Dockerfile"),
        help="Path to the Dockerfile (dockerfile mode).",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Extracted rootfs directory (rootfs mode).",
    )
    return parser


def _audit_dockerfile(path: Path) -> tuple[list[str], dict[str, Any]]:
    findings: list[str] = []
    if not path.exists():
        return [f"Dockerfile not found: {path}"], {"exists": False}
    text = path.read_text()
    lines = text.splitlines()
    facts: dict[str, Any] = {"path": str(path), "exists": True, "lines": len(lines)}

    user_lines = [ln for ln in lines if re.match(r"\s*USER\s+", ln, re.IGNORECASE)]
    facts["USER_directives"] = user_lines
    if not user_lines:
        findings.append("no USER directive — container will run as root")
    else:
        last = user_lines[-1].strip().split(None, 1)[1].strip().strip("\"'")
        if last in {"root", "0"}:
            findings.append(f"last USER directive sets root user: {last!r}")

    entrypoint_lines = [ln for ln in lines if re.match(r"\s*ENTRYPOINT\s+", ln, re.IGNORECASE)]
    facts["ENTRYPOINT_directives"] = entrypoint_lines
    if not entrypoint_lines:
        findings.append("no ENTRYPOINT directive")

    if not re.search(r"LD_LIBRARY_PATH.*ama", text, re.IGNORECASE):
        findings.append("LD_LIBRARY_PATH does not reference AMA library path")

    if not re.search(r"(rm\s+-rf\s+/var/lib/apt/lists|apt-get\s+clean)", text, re.IGNORECASE):
        findings.append("no apt cache cleanup (rm -rf /var/lib/apt/lists OR apt-get clean)")

    for tool in _DEV_TOOLS:
        # Crude but effective: warn if any dev tool is installed and not removed in the same RUN.
        if re.search(rf"\bapt(-get)?\s+install[^\n]*\b{tool}\b", text, re.IGNORECASE) and not re.search(
            rf"\bapt-get\s+(purge|remove)[^\n]*\b{tool}\b", text, re.IGNORECASE
        ):
            findings.append(f"dev tool {tool!r} installed but not purged in the final image")
    return findings, facts


def _audit_rootfs(root: Path) -> tuple[list[str], dict[str, Any]]:
    findings: list[str] = []
    facts: dict[str, Any] = {"root": str(root)}
    if not root.is_dir():
        return [f"--root not a directory: {root}"], facts
    bin_dirs = [root / "usr" / "bin", root / "bin", root / "usr" / "sbin", root / "sbin"]
    found_tools: list[str] = []
    for tool in _DEV_TOOLS:
        for d in bin_dirs:
            if (d / tool).exists():
                found_tools.append(f"{d.relative_to(root)}/{tool}")
                break
    facts["dev_tools_present"] = found_tools
    if found_tools:
        findings.append(f"dev tools present in image: {found_tools}")

    apt_lists = root / "var" / "lib" / "apt" / "lists"
    if apt_lists.exists() and any(apt_lists.iterdir()):
        findings.append("/var/lib/apt/lists is non-empty (apt cache not cleaned)")

    passwd = (root / "etc" / "passwd").read_text(errors="replace") if (root / "etc" / "passwd").exists() else ""
    non_root_users = [ln for ln in passwd.splitlines() if not ln.startswith("root:")]
    facts["non_root_user_count"] = len(non_root_users)
    if not non_root_users:
        findings.append("/etc/passwd has no non-root users — the entrypoint cannot drop privileges")
    return findings, facts


def _collect(args: argparse.Namespace) -> Certificate:
    if args.mode == "dockerfile":
        findings, facts = _audit_dockerfile(Path(args.dockerfile))
    else:
        if not args.root:
            raise ValueError("--root is required in rootfs mode")
        findings, facts = _audit_rootfs(Path(args.root))

    body: dict[str, Any] = {"mode": args.mode, **facts, "findings": findings}
    status = "fail" if findings else "ok"
    return Certificate(
        tool="image_surface_auditor",
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
