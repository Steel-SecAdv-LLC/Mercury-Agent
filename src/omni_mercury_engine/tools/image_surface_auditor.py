# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Operator tool: runtime-image composition auditor.

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
        if re.search(
            rf"\bapt(-get)?\s+install[^\n]*\b{tool}\b", text, re.IGNORECASE
        ) and not re.search(rf"\bapt-get\s+(purge|remove)[^\n]*\b{tool}\b", text, re.IGNORECASE):
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

    passwd = (
        (root / "etc" / "passwd").read_text(errors="replace")
        if (root / "etc" / "passwd").exists()
        else ""
    )
    non_root_users = [ln for ln in passwd.splitlines() if not ln.startswith("root:")]
    facts["non_root_user_count"] = len(non_root_users)
    if not non_root_users:
        findings.append("/etc/passwd has no non-root users — the entrypoint cannot drop privileges")

    # --- ENTRYPOINT --------------------------------------------------------
    # Docker / OCI store the entrypoint in the image config (a JSON file
    # produced by ``docker save`` or ``skopeo copy``).  Tools that
    # extract an image typically drop the config alongside the rootfs
    # (e.g. ``manifest.json`` + ``<sha>.json`` for ``docker save``,
    # ``index.json`` + ``blobs/...`` for OCI layout).  We search both
    # the rootfs root and its parent for any config-like JSON and look
    # for a non-trivial ``Entrypoint``.
    entrypoint_source, entrypoint = _find_image_entrypoint(root)
    facts["entrypoint_source"] = entrypoint_source
    facts["entrypoint"] = entrypoint
    if entrypoint_source is None:
        findings.append(
            "no image config (manifest.json / config.json) found next to rootfs — "
            "cannot verify ENTRYPOINT posture"
        )
    elif not entrypoint:
        findings.append(
            f"image config {entrypoint_source} declares no ENTRYPOINT — "
            "container will inherit shell from base image"
        )

    # --- LD_LIBRARY_PATH ---------------------------------------------------
    # ``LD_LIBRARY_PATH`` may be set in three places that survive into an
    # extracted rootfs: ``/etc/environment``, ``/etc/profile`` (+ its
    # ``.d/`` snippets), and ``/etc/ld.so.conf.d/`` (the loader's own
    # config).  Mercury Agent's invariant is that *one* of them must
    # point at the AMA library path so ``libama_cryptography.so`` is
    # picked up over any system-shipped fallback.
    ld_sources, ld_value, ld_references_ama = _find_ld_library_path(root)
    facts["ld_library_path_sources"] = ld_sources
    facts["ld_library_path"] = ld_value
    facts["ld_library_path_references_ama"] = ld_references_ama
    if not ld_sources:
        findings.append(
            "no LD_LIBRARY_PATH configuration found in /etc/environment, "
            "/etc/profile*, or /etc/ld.so.conf.d/ — AMA library path not pinned"
        )
    elif not ld_references_ama:
        findings.append(
            "LD_LIBRARY_PATH configuration does not reference an AMA library path "
            f"(observed: {ld_value!r})"
        )

    return findings, facts


def _find_image_entrypoint(root: Path) -> tuple[str | None, list[str]]:
    """Locate the OCI / Docker image config and extract ``Entrypoint``.

    Returns ``(source_path_or_None, entrypoint_list)``.  ``source_path``
    is ``None`` when no config-like JSON is found in or next to the
    rootfs (which is itself a finding — the auditor cannot validate
    something that isn't there).
    """
    import json as _json

    # ``docker save`` layout: rootfs is one of several layer tarballs,
    # but the *image* tarball also contains a top-level ``manifest.json``
    # pointing at a per-image config JSON.  When the operator extracts
    # the whole tarball into ``<dir>/`` and points us at ``<dir>/rootfs``
    # we should still see ``<dir>/manifest.json``.
    candidates = [
        root / "manifest.json",
        root.parent / "manifest.json",
        root / "config.json",
        root.parent / "config.json",
        root / "image_config.json",
        root.parent / "image_config.json",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            payload = _json.loads(candidate.read_text())
        except (OSError, _json.JSONDecodeError):
            continue
        entrypoint = _extract_entrypoint_field(payload, candidate.parent)
        if entrypoint is not None:
            return str(candidate), entrypoint
    return None, []


def _extract_entrypoint_field(payload: Any, base: Path) -> list[str] | None:
    """Pull ``Entrypoint`` out of a Docker or OCI image config payload."""
    import json as _json

    if isinstance(payload, list):
        # ``docker save``'s ``manifest.json`` is a top-level list whose
        # entries reference a ``Config`` JSON inside the same tarball.
        for entry in payload:
            cfg_ref = entry.get("Config") if isinstance(entry, dict) else None
            if isinstance(cfg_ref, str):
                cfg_path = base / cfg_ref
                if cfg_path.is_file():
                    try:
                        cfg_payload = _json.loads(cfg_path.read_text())
                    except (OSError, _json.JSONDecodeError):
                        continue
                    return _extract_entrypoint_field(cfg_payload, base)
        return None
    if not isinstance(payload, dict):
        return None
    # OCI image config: ``config.Entrypoint``
    config_block = payload.get("config")
    if isinstance(config_block, dict) and "Entrypoint" in config_block:
        ep = config_block.get("Entrypoint")
        return [str(x) for x in ep] if isinstance(ep, list) else ([str(ep)] if ep else [])
    # Some tooling flattens the field to the top level.
    if "Entrypoint" in payload:
        ep = payload.get("Entrypoint")
        return [str(x) for x in ep] if isinstance(ep, list) else ([str(ep)] if ep else [])
    return None


def _find_ld_library_path(root: Path) -> tuple[list[str], str, bool]:
    """Resolve effective LD_LIBRARY_PATH from common rootfs config files.

    Returns ``(sources, joined_value, references_ama)`` where
    ``sources`` is the list of files that contributed a value,
    ``joined_value`` is the concatenated colon-separated path, and
    ``references_ama`` is True when any contributing path token names
    AMA (case-insensitive).
    """
    sources: list[str] = []
    values: list[str] = []
    env_pattern = re.compile(r"^\s*(?:export\s+)?LD_LIBRARY_PATH\s*=\s*\"?([^\"\n]+)\"?\s*$")
    config_files: list[Path] = [
        root / "etc" / "environment",
        root / "etc" / "profile",
    ]
    profile_dir = root / "etc" / "profile.d"
    if profile_dir.is_dir():
        config_files.extend(sorted(profile_dir.glob("*.sh")))
    for path in config_files:
        if not path.is_file():
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            match = env_pattern.match(line)
            if match:
                sources.append(str(path.relative_to(root)))
                values.append(match.group(1))

    ld_conf_dir = root / "etc" / "ld.so.conf.d"
    if ld_conf_dir.is_dir():
        for conf in sorted(ld_conf_dir.glob("*.conf")):
            try:
                text = conf.read_text(errors="replace")
            except OSError:
                continue
            paths = [
                ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")
            ]
            if paths:
                sources.append(str(conf.relative_to(root)))
                values.extend(paths)

    joined = ":".join(values)
    references_ama = bool(re.search(r"\bama\b|libama_cryptography", joined, re.IGNORECASE))
    return sources, joined, references_ama


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
