"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

------------------------------------------------------------------------

Operator tool: emit a SLSA v1.0 in-toto provenance attestation.

The CycloneDX SBOM (see :mod:`sbom_emitter`) answers "what's in this
artefact"; the SLSA provenance answers "who built it, how, and from
what source".  The attestation is in-toto v1.0 statement format with
the SLSA v1.0 provenance predicate.

The tool runs offline: it walks the supplied artefact paths to compute
their SHA-256 subjects, reads ``$GITHUB_*`` env-vars (or git fallback)
for the builder identity, and emits a canonicalised JSON ready for
detached signing via ``--sign-key-hex`` or out-of-band cosign.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.slsa_provenance_emitter/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.slsa_provenance_emitter",
        description=("Emit a SLSA v1.0 in-toto provenance attestation for the supplied artefacts."),
    )
    parser.add_argument(
        "artifacts",
        nargs="+",
        help="One or more built artefacts (wheel, container image tarball, ...).",
    )
    parser.add_argument(
        "--builder-id",
        default=os.environ.get("MERCURY_BUILDER_ID") or os.environ.get("GITHUB_WORKFLOW_REF"),
        help=(
            "Builder identity URI (e.g. 'https://github.com/actions/runner@v2'). "
            "Defaults to $MERCURY_BUILDER_ID or $GITHUB_WORKFLOW_REF."
        ),
    )
    parser.add_argument(
        "--build-type",
        default="https://slsa.dev/provenance/v1",
        help="SLSA buildType URI.",
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="Source repo (owner/name).  Defaults to $GITHUB_REPOSITORY.",
    )
    parser.add_argument(
        "--ref",
        default=os.environ.get("GITHUB_REF") or os.environ.get("GITHUB_SHA"),
        help="Source ref (refs/heads/main, tag/v1.7.0, ...).  Defaults to $GITHUB_REF / $GITHUB_SHA.",
    )
    return parser


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def _collect(args: argparse.Namespace) -> Certificate:
    subjects: list[dict[str, Any]] = []
    missing: list[str] = []
    for a in args.artifacts:
        p = Path(a)
        if not p.is_file():
            missing.append(a)
            continue
        subjects.append({"name": p.name, "digest": {"sha256": _sha256_file(p)}})

    if missing:
        return Certificate(
            tool="slsa_provenance_emitter",
            schema=_SCHEMA,
            status="fail",
            body={"missing_artifacts": missing},
            warnings=[f"missing artefact: {m}" for m in missing],
        )

    commit = _git_commit()
    predicate: dict[str, Any] = {
        "buildDefinition": {
            "buildType": args.build_type,
            "externalParameters": {
                "ref": args.ref,
                "repository": args.repo,
            },
            "internalParameters": {
                "mercury_env": os.environ.get("MERCURY_ENV", "development"),
            },
            "resolvedDependencies": (
                [{"uri": f"git+https://github.com/{args.repo}", "digest": {"sha1": commit}}]
                if (args.repo and commit)
                else []
            ),
        },
        "runDetails": {
            "builder": {"id": args.builder_id or "https://github.com/actions/runner"},
            "metadata": {
                "invocationId": os.environ.get("GITHUB_RUN_ID")
                or os.environ.get("MERCURY_BUILD_ID"),
                "startedOn": _dt.datetime.now(_dt.UTC).isoformat(),
            },
        },
    }

    attestation = {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": subjects,
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": predicate,
    }

    body: dict[str, Any] = {
        "attestation": attestation,
        "subject_count": len(subjects),
        "builder_id": predicate["runDetails"]["builder"]["id"],
        "repo": args.repo,
        "ref": args.ref,
    }
    warnings: list[str] = []
    if not args.repo:
        warnings.append("repo not pinned ($GITHUB_REPOSITORY unset)")
    if commit is None:
        warnings.append("git HEAD unknown — provenance resolvedDependencies is empty")
    status = "warn" if warnings else "ok"
    return Certificate(
        tool="slsa_provenance_emitter",
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
