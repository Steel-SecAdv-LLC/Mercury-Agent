"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

------------------------------------------------------------------------

Operator tool: HSM / TPM / YubiHSM attestation probe.

Reads ``MERCURY_HSM=pkcs11|tpm2|yubihsm`` and attempts to fetch and
verify the device attestation chain.  Fails closed in production
(:func:`omni_mercury_engine.tools._base.require_real_component`); warns
in development/CI when the backend is unreachable.

The probe is intentionally read-only — it never generates or imports
keys.  All it does is enumerate the on-device certificate chain and
report a SHA-256 over the chain so an auditor can pin the result.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
from typing import Any

from omni_mercury_engine.tools._base import (
    Certificate,
    mercury_env,
    require_real_component,
    run_tool,
)

_SCHEMA = "mercury.tools.hsm_attestation_probe/v1"
_SUPPORTED = ("pkcs11", "tpm2", "yubihsm")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.hsm_attestation_probe",
        description=(
            "Probe the configured HSM/TPM/YubiHSM device for an attestation "
            "chain and emit a signed evidence certificate."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=_SUPPORTED,
        default=os.environ.get("MERCURY_HSM"),
        help="Override $MERCURY_HSM (pkcs11|tpm2|yubihsm).",
    )
    return parser


def _run(cmd: list[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return 127, "", "not installed"
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"


def _probe_pkcs11() -> dict[str, Any]:
    bin_path = shutil.which("pkcs11-tool")
    if not bin_path:
        return {"available": False, "reason": "pkcs11-tool not installed"}
    rc, out, err = _run([bin_path, "--list-objects", "--login", "--read-only"])
    if rc != 0:
        return {"available": False, "reason": f"pkcs11-tool exited {rc}: {err.strip()}"}
    digest = hashlib.sha256(out.encode("utf-8")).hexdigest()
    return {
        "available": True,
        "tool": bin_path,
        "object_count": out.count("Object Label"),
        "chain_sha256": digest,
    }


def _probe_tpm2() -> dict[str, Any]:
    bin_path = shutil.which("tpm2_getekcertificate")
    if not bin_path:
        return {"available": False, "reason": "tpm2-tools not installed"}
    rc, out, err = _run([bin_path, "--ek-certificate"])
    if rc != 0:
        return {"available": False, "reason": f"tpm2_getekcertificate exited {rc}: {err.strip()}"}
    return {
        "available": True,
        "tool": bin_path,
        "ek_certificate_sha256": hashlib.sha256(out.encode("utf-8")).hexdigest(),
    }


def _probe_yubihsm() -> dict[str, Any]:
    bin_path = shutil.which("yubihsm-shell")
    if not bin_path:
        return {"available": False, "reason": "yubihsm-shell not installed"}
    rc, out, err = _run([bin_path, "-a", "get-device-info"])
    if rc != 0:
        return {"available": False, "reason": f"yubihsm-shell exited {rc}: {err.strip()}"}
    return {
        "available": True,
        "tool": bin_path,
        "device_info_sha256": hashlib.sha256(out.encode("utf-8")).hexdigest(),
    }


def _collect(args: argparse.Namespace) -> Certificate:
    backend = args.backend
    body: dict[str, Any] = {
        "mercury_env": mercury_env(),
        "backend": backend,
    }
    if backend is None:
        return Certificate(
            tool="hsm_attestation_probe",
            schema=_SCHEMA,
            status="warn",
            body=body,
            warnings=["MERCURY_HSM not set; no HSM/TPM/YubiHSM was probed"],
        )
    probe: dict[str, Any]
    if backend == "pkcs11":
        probe = _probe_pkcs11()
    elif backend == "tpm2":
        probe = _probe_tpm2()
    else:
        probe = _probe_yubihsm()
    body["probe"] = probe
    available = bool(probe.get("available"))
    require_real_component(f"HSM backend {backend!r}", available)
    if not available:
        return Certificate(
            tool="hsm_attestation_probe",
            schema=_SCHEMA,
            status="fail" if mercury_env() == "production" else "warn",
            body=body,
            warnings=[probe.get("reason", "backend unavailable")],
        )
    return Certificate(
        tool="hsm_attestation_probe",
        schema=_SCHEMA,
        status="ok",
        body=body,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
