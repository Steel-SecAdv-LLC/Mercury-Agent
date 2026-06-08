# Copyright (C) 2025 Steel Security Advisors LLC
"""Tool ran, evidence collected, no policy violation detected."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------
#
# These are part of the operator contract — pre-commit hooks, Helm
# tests, and CI gates pin on these numbers.  Do not renumber.
EXIT_OK: int = 0
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
        """Serialise the certificate (header + body) as a plain dict."""
        return {
            "schema": self.schema,
            "tool": self.tool,
            "mercury_version": mercury_version(),
            "generated_at": datetime.now(UTC).isoformat(),
            "status": self.status,
            "warnings": list(self.warnings),
            "body": _coerce_deterministic(dict(self.body)),
        }


# ---------------------------------------------------------------------------
# Deterministic coercion
# ---------------------------------------------------------------------------
#
# Tools must produce byte-identical output (modulo ``generated_at``) for
# byte-identical input.  Two failure modes the audit caught in practice:
#
#   1. Plain ``set`` instances embedded in the body — Python set
#      iteration order is insertion order, but reconstructed sets
#      (e.g. ``set(a) | set(b)``) are not guaranteed across runs.
#   2. Plain ``float`` values printed via ``repr`` — ``json.dumps``
#      itself is deterministic for finite floats, but ``f"{v!r}"``
#      inside error strings is sensitive to NaN/Inf representation.
#
# ``_coerce_deterministic`` walks the body recursively and converts
# every ``set``/``frozenset`` into a sorted list (string-keyed by
# default) and normalises non-finite floats to a stable token.

_NON_FINITE = {float("inf"): "inf", float("-inf"): "-inf"}


def _coerce_deterministic(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _coerce_deterministic(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return sorted(_coerce_deterministic(item) for item in obj)
    if isinstance(obj, (list, tuple)):
        return [_coerce_deterministic(item) for item in obj]
    if isinstance(obj, float):
        # NaN compares unequal to itself; pin to a stable token so two
        # runs with NaN-bearing inputs still diff to nothing.
        if obj != obj:  # noqa: PLR0124 — NaN check is intentional
            return "NaN"
        if obj in _NON_FINITE:
            return _NON_FINITE[obj]
    return obj


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
# Envelope validator (handwritten — no jsonschema dependency)
# ---------------------------------------------------------------------------
#
# Every tool round-trips its emitted envelope through this validator
# before writing to disk.  Drift fails the run closed.  The schema
# string format is ``mercury.tools.<name>/v<version>`` so a future v2
# can drop fields without silently downgrading v1 consumers.

_SCHEMA_RE = re.compile(r"^mercury\.tools\.[a-z][a-z0-9_]*/v\d+$")
_TOOL_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_STATUS_VALUES: frozenset[str] = frozenset({"ok", "warn", "fail"})


class EnvelopeValidationError(ValueError):
    """Raised when a certificate envelope drifts from the v1 contract."""


def validate_envelope(envelope: Mapping[str, Any]) -> None:
    """Validate a certificate envelope against the v1 contract.

    Raises :class:`EnvelopeValidationError` on the first drift.  The
    contract is intentionally minimal so v1 stays stable across
    releases:

    * ``schema`` matches ``mercury.tools.<name>/v<n>``;
    * ``tool`` matches the schema's tool segment;
    * ``mercury_version`` is a non-empty string;
    * ``generated_at`` is an RFC3339-ish ISO string (parseable by
      :func:`datetime.fromisoformat`);
    * ``status`` is one of ``ok``/``warn``/``fail``;
    * ``warnings`` is a list of strings;
    * ``body`` is a dict.

    The validator is handwritten (no ``jsonschema`` dependency) per the
    operator-tools brief: native first.
    """
    required = {"schema", "tool", "mercury_version", "generated_at", "status", "warnings", "body"}
    missing = required - set(envelope.keys())
    if missing:
        raise EnvelopeValidationError(f"missing envelope fields: {sorted(missing)}")
    extra = set(envelope.keys()) - required - {"signature"}
    if extra:
        raise EnvelopeValidationError(f"unknown envelope fields: {sorted(extra)}")

    schema = envelope["schema"]
    if not isinstance(schema, str) or not _SCHEMA_RE.match(schema):
        raise EnvelopeValidationError(
            f"schema must match 'mercury.tools.<name>/v<n>', got {schema!r}"
        )

    tool = envelope["tool"]
    if not isinstance(tool, str) or not _TOOL_RE.match(tool):
        raise EnvelopeValidationError(f"tool must match '[a-z][a-z0-9_]*', got {tool!r}")

    # Schema tool-segment must match the declared tool field.
    schema_tool = schema.removeprefix("mercury.tools.").rsplit("/v", 1)[0]
    if schema_tool != tool:
        raise EnvelopeValidationError(f"schema tool segment {schema_tool!r} != tool field {tool!r}")

    mv = envelope["mercury_version"]
    if not isinstance(mv, str) or not mv:
        raise EnvelopeValidationError(f"mercury_version must be a non-empty string, got {mv!r}")

    ts = envelope["generated_at"]
    if not isinstance(ts, str) or not ts:
        raise EnvelopeValidationError(f"generated_at must be an ISO string, got {ts!r}")
    try:
        datetime.fromisoformat(ts)
    except ValueError as exc:
        raise EnvelopeValidationError(f"generated_at not parseable: {exc}") from exc

    status = envelope["status"]
    if status not in _STATUS_VALUES:
        raise EnvelopeValidationError(
            f"status must be one of {sorted(_STATUS_VALUES)}, got {status!r}"
        )

    warnings = envelope["warnings"]
    if not isinstance(warnings, list) or not all(isinstance(w, str) for w in warnings):
        raise EnvelopeValidationError("warnings must be list[str]")

    body = envelope["body"]
    if not isinstance(body, dict):
        raise EnvelopeValidationError(f"body must be a dict, got {type(body).__name__}")

    sig = envelope.get("signature")
    if sig is not None:
        if not isinstance(sig, dict):
            raise EnvelopeValidationError("signature must be a dict when present")
        for field_name in ("algorithm", "public_key_hex", "signature_hex", "payload_sha3_256"):
            if field_name not in sig:
                raise EnvelopeValidationError(f"signature is missing required field {field_name!r}")


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


def verify_certificate_ed25519(payload: bytes, signature_record: Mapping[str, Any]) -> bool:
    """Verify a detached Ed25519 signature record against ``payload``.

    Returns True on success, raises :class:`ValueError` on a malformed
    record, and returns False on a cryptographic verification failure.
    Used by :class:`signed_release_bundle` and the round-trip tests.
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    if signature_record.get("algorithm") != "ed25519":
        raise ValueError(f"expected algorithm 'ed25519', got {signature_record.get('algorithm')!r}")
    expected_digest = signature_record.get("payload_sha3_256")
    actual_digest = hashlib.sha3_256(payload).hexdigest()
    if expected_digest != actual_digest:
        return False
    pk = Ed25519PublicKey.from_public_bytes(bytes.fromhex(signature_record["public_key_hex"]))
    try:
        pk.verify(bytes.fromhex(signature_record["signature_hex"]), payload)
    except InvalidSignature:
        return False
    return True


# ---------------------------------------------------------------------------
# Atomic file writes
# ---------------------------------------------------------------------------


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically.

    Uses ``tempfile.NamedTemporaryFile`` in the same directory followed by
    ``os.replace`` — POSIX guarantees same-directory rename is atomic on
    every supported filesystem (ext4, xfs, btrfs, apfs).  A crash mid-write
    leaves either the old file untouched or the new file in place; never
    a half-written manifest.

    Every tool that writes to disk uses this helper (centralised in
    :func:`emit`) so the atomic-replace invariant is enforced by the
    framework, not re-implemented per-tool.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.replace(tmp.name, path)
    except Exception:
        try:
            tmp.close()
        except Exception:
            pass
        try:
            os.unlink(tmp.name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    """Wrapper around :func:`atomic_write_bytes` for text content."""
    atomic_write_bytes(path, text.encode("utf-8"))


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the shared CLI flags used by every tool.

    Every tool accepts:

    * ``--output PATH``: write the JSON certificate to ``PATH``
      instead of stdout (stdout still receives a one-line summary
      when an output path is given);
    * ``--sign-key-hex HEX``: Ed25519 secret seed (64 hex chars).
      When supplied the tool writes a ``<output>.sig.json`` side-car
      next to the certificate.  Without ``--output`` the signature
      record is embedded in the envelope under ``"signature"``.  The
      signature is computed over the **exact bytes written to disk**,
      not a re-serialised copy.
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

    * Validates the envelope against the v1 contract; drift becomes a
      ``schema_validation_failed`` certificate (the original tool body
      is preserved under ``body.original`` for forensic value).
    * Atomically writes the certificate bytes to ``--output`` (when set)
      and signs **those exact bytes** when ``--sign-key-hex`` is set.
    * Maps ``status`` → exit code: ``"ok"``→0, ``"warn"``→0 (warnings
      are non-fatal by default), anything else→1.  ``--require``
      escalates ``"warn"`` to 1 as well.
    """
    envelope = certificate.envelope()
    try:
        validate_envelope(envelope)
    except EnvelopeValidationError as exc:
        # Replace the drifted envelope with a fail-closed one — preserve
        # the original body for the operator to debug.
        fallback = Certificate(
            tool=certificate.tool if _TOOL_RE.match(certificate.tool or "") else "unknown_tool",
            schema=(
                certificate.schema
                if _SCHEMA_RE.match(certificate.schema or "")
                else "mercury.tools.error/v1"
            ),
            status="fail",
            body={
                "error": "schema_validation_failed",
                "detail": str(exc),
                "original": dict(certificate.body),
            },
            warnings=[f"envelope rejected: {exc}"] + list(certificate.warnings),
        )
        envelope = fallback.envelope()
        # The fallback envelope itself is guaranteed valid by construction.
        validate_envelope(envelope)
        certificate = fallback

    canonical = to_json_bytes(envelope)

    signature: dict[str, str] | None = None
    if args.sign_key_hex and not args.output:
        # Inline-signature path: sign the canonical *embedded* envelope
        # bytes and round-trip the augmented envelope through stdout.
        signature = sign_certificate_ed25519(canonical, args.sign_key_hex)

    if args.output:
        out = Path(args.output)
        atomic_write_bytes(out, canonical)
        if args.sign_key_hex:
            # Sign the exact bytes we just wrote (read-back to guarantee
            # we're signing what an external auditor will read).
            written = out.read_bytes()
            signature = sign_certificate_ed25519(written, args.sign_key_hex)
            sig_path = out.with_suffix(out.suffix + ".sig.json")
            atomic_write_bytes(
                sig_path, (json.dumps(signature, sort_keys=True, indent=2) + "\n").encode("utf-8")
            )
            print(f"wrote certificate: {out}\nwrote signature  : {sig_path}", file=sys.stderr)
        else:
            print(f"wrote certificate: {out}", file=sys.stderr)
    elif signature is not None:
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
    except Exception as exc:
        # Operator-visible failure with a structured fallback envelope so
        # downstream automation still receives parseable JSON.
        tb = traceback.format_exc() if os.environ.get("MERCURY_TOOLS_DEBUG") else None
        prog = parser.prog or "unknown"
        # ``parser.prog`` defaults to the python -m form; pull the last
        # path segment so the tool name validates against ``_TOOL_RE``.
        tool_name = prog.rsplit(".", 1)[-1] if "." in prog else prog
        if not _TOOL_RE.match(tool_name):
            tool_name = "unknown_tool"
        fallback = Certificate(
            tool=tool_name,
            schema=f"mercury.tools.{tool_name}/v1",
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


# ---------------------------------------------------------------------------
# Production-vs-development environment selection
# ---------------------------------------------------------------------------


def mercury_env() -> str:
    """Return the active deployment environment.

    Reads ``MERCURY_ENV`` and accepts ``production`` / ``staging`` /
    ``development`` / ``ci``.  Unknown values fall through to
    ``development`` so a typo never silently lifts a fail-closed gate
    into a production warning.
    """
    raw = os.environ.get("MERCURY_ENV", "development").strip().lower()
    if raw in {"production", "staging", "development", "ci"}:
        return raw
    return "development"


def require_real_component(name: str, present: bool, env: str | None = None) -> None:
    """Fail closed in production when a real component is absent.

    Tools that have a stub fallback (HSM, hardware RNG, secure enclave)
    must call this before emitting an ``ok`` certificate.  In production
    the stub fallback is unacceptable; in development it is reported as
    a warning by the caller, not enforced here.
    """
    if not present and (env or mercury_env()) == "production":
        raise RuntimeError(
            f"{name} is unavailable but MERCURY_ENV=production requires the real component"
        )


__all__ = [
    "EXIT_DEPENDENCY",
    "EXIT_FAIL",
    "EXIT_OK",
    "EXIT_USAGE",
    "Certificate",
    "DependencyMissing",
    "EnvelopeValidationError",
    "add_common_arguments",
    "atomic_write_bytes",
    "atomic_write_text",
    "emit",
    "mercury_env",
    "mercury_version",
    "require_real_component",
    "run_tool",
    "sign_certificate_ed25519",
    "to_json_bytes",
    "validate_envelope",
    "verify_certificate_ed25519",
]
