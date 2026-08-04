# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Operator tool: handwritten secret scanner with allow-list baseline.

Walks the tree from --root and flags every file whose bytes match any
of:

* a curated regex set for common secret formats (AWS access keys,
  Google API keys, GitHub PATs/Apps, JWT, PEM private keys, Slack
  tokens, generic ``password = "..."`` assignments);
* a Shannon-entropy threshold over base64/base64url-shaped runs longer
  than the configured floor.

A finding can be allow-listed by adding its
``{path, line, secret_hash}`` triple to ``.secrets.baseline`` (default
``.secrets.baseline`` at the repo root) — re-runs of the scanner read
the baseline, drop matching findings, and only fail on *new* secrets.

The scanner is pre-commit-friendly (no network, no external tools) and
deliberately handwritten so Mercury does not pull in
``gitleaks``/``detect-secrets`` as runtime dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, atomic_write_text, run_tool

_SCHEMA = "mercury.tools.secret_scan_baseline/v1"

# Regex catalogue.  Order is preserved so the longest, most-specific
# patterns win on overlapping matches.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    (
        "aws-secret-key",
        re.compile(
            r"(?i)aws(?:.{0,20})?(?:secret|access)?[_-]?key(?:.{0,20})?[:=]\s*['\"]([A-Za-z0-9/+=]{40})['\"]"
        ),
    ),
    ("github-pat", re.compile(r"\bghp_[A-Za-z0-9]{36}\b")),
    ("github-app", re.compile(r"\bghs_[A-Za-z0-9]{36}\b")),
    ("github-fine-grained", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{82}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("slack-token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,48}\b")),
    ("stripe-live", re.compile(r"\bsk_live_[A-Za-z0-9]{24,}\b")),
    ("stripe-test", re.compile(r"\bsk_test_[A-Za-z0-9]{24,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}\b")),
    ("pem-private-key", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----")),
    (
        "generic-password",
        re.compile(
            r"(?i)\b(?:password|passwd|pwd|secret|token)\b\s*[:=]\s*['\"]([^'\"\s]{8,})['\"]"
        ),
    ),
)

# Default extensions to skip.  Binary blobs trigger entropy hits with
# no real signal value; keep the list conservative.
_DEFAULT_SKIP_EXTS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".bin",
    ".whl",
    ".tar",
    ".gz",
    ".zip",
    ".onnx",
    ".pt",
    ".pth",
    ".npy",
    ".npz",
    ".pyc",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.secret_scan_baseline",
        description=(
            "Handwritten secret scanner with .secrets.baseline allow-list. "
            "Pre-commit-friendly and dependency-free."
        ),
    )
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--baseline",
        default=".secrets.baseline",
        help="Allow-list file (JSON list of {path,line,secret_hash}).",
    )
    parser.add_argument(
        "--entropy-min",
        type=float,
        default=4.5,
        help="Minimum Shannon entropy for entropy-only findings.",
    )
    parser.add_argument(
        "--entropy-run-min",
        type=int,
        default=24,
        help="Minimum length of a base64/base64url run to entropy-score.",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write current findings to --baseline (use to bootstrap).",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[".venv", "node_modules", ".git", "dist", "build", "__pycache__"],
        help="Path fragment to skip (repeatable).",
    )
    return parser


def _shannon(s: str) -> float:
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    total = len(s)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


_BASE64_RUN = re.compile(r"[A-Za-z0-9+/=_-]{16,}")


def _secret_hash(s: str) -> str:
    """Salted SHA-256 of the candidate so baselines do not leak secrets."""
    return hashlib.sha256(f"mercury-secret-scan:{s}".encode()).hexdigest()


def _scan_file(
    path: Path,
    *,
    entropy_min: float,
    entropy_run_min: int,
) -> list[dict[str, Any]]:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []
    findings: list[dict[str, Any]] = []
    for ln, line in enumerate(text.splitlines(), start=1):
        for kind, pat in _PATTERNS:
            for m in pat.finditer(line):
                secret = m.group(0)
                findings.append(
                    {
                        "kind": kind,
                        "line": ln,
                        "secret_hash": _secret_hash(secret),
                        # Two characters is enough to locate the value on the named
                        # line without the report itself becoming a partial
                        # disclosure (8 leading chars of a key is real key material).
                        "preview": secret[:2] + f"...({len(secret)} chars)",
                    }
                )
        for m in _BASE64_RUN.finditer(line):
            run = m.group(0)
            if len(run) < entropy_run_min:
                continue
            ent = _shannon(run)
            if ent >= entropy_min:
                findings.append(
                    {
                        "kind": "entropy",
                        "line": ln,
                        "entropy": round(ent, 3),
                        "secret_hash": _secret_hash(run),
                        "preview": run[:2] + f"...({len(run)} chars)",
                    }
                )
    return findings


def _walk(root: Path, excludes: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if any(frag in rel for frag in excludes):
            continue
        if p.suffix.lower() in _DEFAULT_SKIP_EXTS:
            continue
        if p.is_file():
            out.append(p)
    return out


def _collect(args: argparse.Namespace) -> Certificate:
    root = Path(args.root).resolve()
    baseline_path = Path(args.baseline)
    if not baseline_path.is_absolute():
        baseline_path = (root / args.baseline).resolve()
    baseline: list[dict[str, Any]] = []
    if baseline_path.exists():
        try:
            baseline = json.loads(baseline_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return Certificate(
                tool="secret_scan_baseline",
                schema=_SCHEMA,
                status="fail",
                body={
                    "root": str(root),
                    "baseline": str(baseline_path),
                    "error": f"baseline unparseable: {exc}",
                },
            )

    baseline_keys: set[tuple[str, int, str]] = {
        (b["path"], int(b["line"]), b["secret_hash"]) for b in baseline if "secret_hash" in b
    }

    all_findings: list[dict[str, Any]] = []
    new_findings: list[dict[str, Any]] = []
    for f in _walk(root, args.exclude):
        rel = f.relative_to(root).as_posix()
        for finding in _scan_file(
            f,
            entropy_min=args.entropy_min,
            entropy_run_min=args.entropy_run_min,
        ):
            entry = {"path": rel, **finding}
            all_findings.append(entry)
            key = (rel, int(finding["line"]), finding["secret_hash"])
            if key not in baseline_keys:
                new_findings.append(entry)

    if args.update_baseline:
        atomic_write_text(
            baseline_path,
            json.dumps(all_findings, indent=2, sort_keys=True) + "\n",
        )

    body: dict[str, Any] = {
        "root": str(root),
        "baseline": str(baseline_path),
        "baseline_entries": len(baseline_keys),
        "total_findings": len(all_findings),
        "new_findings": new_findings,
        "files_scanned": len(_walk(root, args.exclude)),
    }
    status = "fail" if new_findings else "ok"
    warnings = (
        [f"{len(new_findings)} new candidate secret(s) outside baseline"] if new_findings else []
    )
    return Certificate(
        tool="secret_scan_baseline",
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
