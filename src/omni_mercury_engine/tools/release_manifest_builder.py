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

Operator tool: release-time manifest builder.

Emits a single JSON manifest pinning every contractual constant of a
Mercury release tag — versions, dependency refs, container digest, the
fusion-weight tuple, λ, σ thresholds, and the benevolence threshold.
Today this information is spread across CHANGELOG, release notes,
workflow files and runtime constants; one manifest collapses the
release attestation into a reviewable artefact.
"""

from __future__ import annotations

import argparse
import importlib.metadata as _md
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.release_manifest_builder/v1"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.release_manifest_builder",
        description="Emit a JSON release manifest pinning every contractual constant.",
    )
    parser.add_argument(
        "--container-digest",
        default=os.environ.get("MERCURY_CONTAINER_DIGEST"),
        help="Optional container image digest (sha256:...).  Defaults to $MERCURY_CONTAINER_DIGEST.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional release tag (e.g. v1.7.0).  If omitted, derived from `git describe`.",
    )
    return parser


def _git(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=str(_REPO_ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _dep_version(name: str) -> str | None:
    try:
        return _md.version(name)
    except _md.PackageNotFoundError:
        return None


def _read_ama_ref() -> str | None:
    text = (_REPO_ROOT / "pyproject.toml").read_text(errors="replace")
    import re

    m = re.search(
        r"ama-cryptography\s*@\s*git\+https?://[^@]+@(?P<ref>[^\s\",]+)",
        text,
        re.IGNORECASE,
    )
    return m.group("ref") if m else None


def _fusion_weights() -> dict[str, float] | str:
    try:
        from omni_mercury_engine.ml.three_r_attention import PHI

        phi_sum = PHI + 2.0
        return {
            "w_R": PHI / phi_sum,
            "w_H": 1.0 / phi_sum,
            "w_O": 1.0 / phi_sum,
        }
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc}"


def _ethical_constants() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        from omni_mercury_engine.core.centralized_constants import ETHICAL, LYAPUNOV

        out["benevolence_immutable"] = float(ETHICAL.BENEVOLENCE_IMMUTABLE)
        out["lambda_convergence"] = float(LYAPUNOV.LAMBDA_CONVERGENCE)
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    try:
        from omni_mercury_engine.security.sigma_immutable_gate import (
            SIGMA_ETHICAL_BAND_END,
            SIGMA_IMMUTABLE_DEFAULT_THRESHOLD,
        )

        out["sigma_immutable_default_threshold"] = float(SIGMA_IMMUTABLE_DEFAULT_THRESHOLD)
        out["sigma_ethical_band_end"] = float(SIGMA_ETHICAL_BAND_END)
    except Exception as exc:  # noqa: BLE001
        out["sigma_error"] = str(exc)
    return out


def _collect(args: argparse.Namespace) -> Certificate:
    tag = args.tag or _git(["describe", "--tags", "--always", "--dirty"])
    head = _git(["rev-parse", "HEAD"])
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])

    body: dict[str, Any] = {
        "tag": tag,
        "git_head": head,
        "git_branch": branch,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            name: _dep_version(name)
            for name in (
                "omni-mercury-engine",
                "numpy",
                "scipy",
                "scikit-learn",
                "torch",
                "cryptography",
                "ama-cryptography",
                "pynacl",
                "fastapi",
                "uvicorn",
                "pyyaml",
            )
        },
        "ama_cryptography_ref": _read_ama_ref(),
        "container_digest": args.container_digest,
        "fusion_weights": _fusion_weights(),
        "constants": _ethical_constants(),
    }

    warnings: list[str] = []
    if body["ama_cryptography_ref"] is None:
        warnings.append("ama-cryptography git ref not pinned in pyproject.toml")
    if body["container_digest"] is None:
        warnings.append("container_digest not supplied (set MERCURY_CONTAINER_DIGEST)")

    status = "ok" if not warnings else "warn"
    return Certificate(
        tool="release_manifest_builder",
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
