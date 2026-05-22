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

Shared infrastructure for ``omni_mercury_engine.tools.*`` operator
utilities.

This module is intentionally kept dependency-free (stdlib + numpy via
caller-supplied paths only).  It defines the contracts every operator
tool in the package follows:

* a single, machine-readable JSON document on stdout (always), with a
  small set of common envelope fields (``schema``, ``tool``,
  ``mercury_version``, ``generated_at``, ``status``);
* stable, documented exit codes (:data:`EXIT_OK`,
  :data:`EXIT_FAIL`, :data:`EXIT_USAGE`, :data:`EXIT_DEPENDENCY`);
* optional Ed25519 detached signature over the certificate bytes so
  that an external auditor can re-verify the artefact independently of
  the issuing host;
* a single ``run_tool(...)`` driver that converts an in-process
  ``Certificate`` into the canonical CLI output (file write + signature
  side-car) and returns the right exit code.

Tools must never print free-form text to stdout — stdout is a JSON
channel for downstream automation.  Human-readable progress goes to
stderr.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------
#
# These are part of the operator contract — pre-commit hooks, Helm
# tests, and CI gates pin on these numbers.  Do not renumber.
EXIT_OK: int = 0
"""Tool ran, evidence collected, no policy violation detected."""

EXIT_FAIL: int = 1
"""Tool ran but the audited condition failed (signature invalid,
config out of schema, gate did not fire, drift detected, ...).
Pre-commit / CI must treat this as a hard error."""

EXIT_USAGE: int = 2
"""CLI usage error (missing argument, unparsable file, ...).
Identical semantics to ``argparse``'s default for unknown flags."""

EXIT_DEPENDENCY: int = 3
"""A required runtime dependency is missing (AMA Cryptography not
installed, torch not installed, GPU absent, ...).  Distinct from
:data:`EXIT_FAIL` so an operator can tell ``"system not built right"``
apart from ``"system built right but the artefact is bad"``."""


# ---------------------------------------------------------------------------
# Mercury version pin
# ---------------------------------------------------------------------------


def mercury_version() -> str:
    """Return the installed Mercury Agent version string.

    Falls back to ``"unknown"`` only when the package is being executed
    from a non-installed checkout *and* ``__version__`` is unavailable —
    in normal use this is the ``omni_mercury_engine.__version__``
    string emitted from ``__init__.py``.
    """
    try:
        from omni_mercury_engine import __version__ as v

        return str(v)
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Certificate envelope
# ---------------------------------------------------------------------------


@dataclass
class Certificate:
    """A signed-or-unsigned JSON evidence record produced by a tool.

    Attributes:
        tool: Short, machine-readable tool name (e.g.
            ``"sigma_immutable_verifier"``).  Used as the document type.
        schema: Versioned schema identifier
            (e.g. ``"mercury.tools.sigma_immutable_verifier/v1"``).
        status: One of ``"ok"`` / ``"fail"`` / ``"warn"``.
        body: Tool-specific evidence payload.  Must be JSON-serialisable
            with ``default=str`` semantics — Path/Decimal/numpy scalars
            are auto-stringified by :func:`to_json_bytes`.
        warnings: Non-fatal observations.  Empty list when none.
    """

    tool: str
    schema: str
    status: str
    body: Mapping[str, Any]
    warnings: list[str] = field(default_factory=list)

    def envelope(self) -> dict[str, Any]:
        """Return the full envelope (header + body) as a plain dict."""
        return {
            "schema": self.schema,
            "tool": self.tool,
            "mercury_version": mercury_version(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": self.status,
            "warnings": list(self.warnings),
            "body": dict(self.body),
        }


def to_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Canonical JSON byte serialisation used for hashing/signing.

    * ``sort_keys=True`` so the bytes are deterministic.
    * ``separators=(",", ":")`` so the digest is stable across
      pretty-printing changes.
    * ``default=str`` so :class:`~pathlib.Path`, :class:`~enum.Enum`,
      and numpy scalar types serialise without per-call adapters.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


# ---------------------------------------------------------------------------
# Optional Ed25519 signing
# ---------------------------------------------------------------------------


def sign_certificate_ed25519(payload: bytes, secret_key_hex: str) -> dict[str, str]:
    """Sign ``payload`` with Ed25519 and return the signature record.

    The signature record matches the same shape used by
    :func:`omni_mercury_engine.security.sigma_immutable_corpus.sign_and_persist_corpus`
    so external auditors only need to know one verification routine.

    Args:
        payload: Bytes to sign.  Caller is responsible for canonicalising
            (use :func:`to_json_bytes`).
        secret_key_hex: 32-byte Ed25519 secret seed, hex-encoded.  Use
            ``secrets.token_hex(32)`` to generate.

    Raises:
        ValueError: secret_key_hex is not exactly 64 hex characters.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    if len(secret_key_hex) != 64:
        raise ValueError(
            f"Ed25519 secret_key_hex must be 32 bytes (64 hex chars); got {len(secret_key_hex)}"
        )
    sk_bytes = bytes.fromhex(secret_key_hex)
    sk = Ed25519PrivateKey.from_private_bytes(sk_bytes)
    pk_bytes = sk.public_key().public_bytes_raw()
    signature = sk.sign(payload)
    return {
        "algorithm": "ed25519",
        "public_key_hex": pk_bytes.hex(),
        "signature_hex": signature.hex(),
        "payload_sha3_256": hashlib.sha3_256(payload).hexdigest(),
    }


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the shared ``--output`` / ``--sign-key-hex`` flags.

    Every tool accepts:

    * ``--output PATH``: write the JSON certificate to ``PATH``
      instead of stdout (stdout still receives a one-line summary
      when an output path is given);
    * ``--sign-key-hex HEX``: Ed25519 secret seed (64 hex chars).
      When supplied the tool writes a ``<output>.sig.json`` side-car
      next to the certificate.  Without ``--output`` the signature
      record is embedded in the envelope under ``"signature"``.
    * ``--require``: when set, the tool exits non-zero unless
      ``status == "ok"``.  Operators use this to convert evidence
      collection into a hard gate.
    """
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Write the JSON certificate to PATH instead of stdout.",
    )
    parser.add_argument(
        "--sign-key-hex",
        default=None,
        help="32-byte Ed25519 secret seed (64 hex chars) used to sign the certificate.",
    )
    parser.add_argument(
        "--require",
        action="store_true",
        help="Exit non-zero unless status == 'ok'.",
    )


def emit(certificate: Certificate, args: argparse.Namespace) -> int:
    """Write the certificate per CLI args and return the exit code.

    * Honours ``--output`` (path) and ``--sign-key-hex`` (Ed25519 seed).
    * Maps ``status`` → exit code: ``"ok"``→0, ``"warn"``→0 (warnings
      are non-fatal by default), anything else→1.  ``--require``
      escalates ``"warn"`` to 1 as well.
    """
    envelope = certificate.envelope()
    canonical = to_json_bytes(envelope)

    signature: dict[str, str] | None = None
    if args.sign_key_hex:
        signature = sign_certificate_ed25519(canonical, args.sign_key_hex)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(canonical)
        if signature is not None:
            sig_path = out.with_suffix(out.suffix + ".sig.json")
            sig_path.write_text(json.dumps(signature, sort_keys=True, indent=2))
            print(f"wrote certificate: {out}\nwrote signature  : {sig_path}", file=sys.stderr)
        else:
            print(f"wrote certificate: {out}", file=sys.stderr)
    else:
        if signature is not None:
            envelope_with_sig = dict(envelope)
            envelope_with_sig["signature"] = signature
            sys.stdout.write(
                json.dumps(envelope_with_sig, sort_keys=True, indent=2, default=str) + "\n"
            )
        else:
            sys.stdout.write(json.dumps(envelope, sort_keys=True, indent=2, default=str) + "\n")

    status = certificate.status
    if status == "ok":
        return EXIT_OK
    if status == "warn" and not args.require:
        return EXIT_OK
    return EXIT_FAIL


def run_tool(
    build_parser: Callable[[], argparse.ArgumentParser],
    collect: Callable[[argparse.Namespace], Certificate],
    argv: list[str] | None = None,
) -> int:
    """Standard ``__main__`` driver for every tool in this package.

    Args:
        build_parser: Zero-arg factory returning the tool's argparse
            parser (must NOT install the common args; the driver does).
        collect: Callable that takes the parsed namespace and returns
            a :class:`Certificate` (or raises).
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Exit-code semantics:
        ``EXIT_OK``         — ``collect`` returned an ``"ok"`` certificate.
        ``EXIT_FAIL``       — certificate ``status`` was not ``"ok"`` (or
                              ``"warn"`` under ``--require``).
        ``EXIT_USAGE``      — argparse rejected the arguments.
        ``EXIT_DEPENDENCY`` — ``collect`` raised a
                              :class:`DependencyMissing`.
    """
    parser = build_parser()
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    try:
        certificate = collect(args)
    except DependencyMissing as exc:
        print(f"dependency missing: {exc}", file=sys.stderr)
        return EXIT_DEPENDENCY
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — top-level CLI boundary, must not crash
        # Operator-visible failure with a structured fallback envelope so
        # downstream automation still receives parseable JSON.
        tb = traceback.format_exc() if os.environ.get("MERCURY_TOOLS_DEBUG") else None
        fallback = Certificate(
            tool=parser.prog or "unknown",
            schema="mercury.tools.error/v1",
            status="fail",
            body={
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                **({"traceback": tb} if tb else {}),
            },
        )
        return emit(fallback, args)
    return emit(certificate, args)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DependencyMissing(RuntimeError):
    """Raised by a tool's collector when a required runtime is absent.

    The driver maps this to :data:`EXIT_DEPENDENCY` so callers can
    distinguish ``"AMA not installed"`` from ``"corpus signature
    invalid"`` — they require different remediation.
    """


__all__ = [
    "Certificate",
    "DependencyMissing",
    "EXIT_DEPENDENCY",
    "EXIT_FAIL",
    "EXIT_OK",
    "EXIT_USAGE",
    "add_common_arguments",
    "emit",
    "mercury_version",
    "run_tool",
    "sign_certificate_ed25519",
    "to_json_bytes",
]
