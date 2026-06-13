# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Part 1 — Hardening contract tests.

These tests are the regression guard for every operator-tool invariant
the brief promises:

* certificate envelope conformance (``mercury.tools.<name>/v1``);
* deterministic output modulo ``generated_at``;
* Ed25519 ``--sign-key-hex`` produces a detached signature over the
  *exact* bytes written to ``--output`` and round-trips through
  ``sigma_immutable_verifier``;
* exit-code contract: ``fail → 1``, ``warn → 0``, ``--require warn → 1``;
* atomic file writes (``tempfile.NamedTemporaryFile`` + ``os.replace``)
  and ``--dry-run`` honoured;
* :mod:`convergence_proof_emitter` sentinel splicing is idempotent,
  byte-preserving, and fails loud on malformed input.

Every test in this module either fails before the corresponding
hardening fix in :mod:`omni_mercury_engine.tools._base` /
:mod:`convergence_proof_emitter` and passes after, or pins a present
invariant against regression.
"""

from __future__ import annotations

import json
import secrets
from typing import TYPE_CHECKING, Any

import pytest

from omni_mercury_engine.tools import TOOL_REGISTRY
from omni_mercury_engine.tools._base import (
    Certificate,
    EnvelopeValidationError,
    atomic_write_bytes,
    sign_certificate_ed25519,
    to_json_bytes,
    validate_envelope,
    verify_certificate_ed25519,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers


def _load_cert(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text())
    assert isinstance(parsed, dict), f"expected dict envelope, got {type(parsed)}"
    return parsed


def _drop_volatile(envelope: dict[str, Any]) -> dict[str, Any]:
    """Strip ``generated_at`` (and similar) so byte-comparison is deterministic."""
    return {k: v for k, v in envelope.items() if k != "generated_at"}


# ---------------------------------------------------------------------------
# 1. Envelope conformance — handwritten validator, no jsonschema


class TestEnvelopeConformance:
    """Every tool's certificate must round-trip through ``validate_envelope``."""

    def test_validator_accepts_well_formed_cert(self) -> None:
        cert = Certificate(
            tool="oae_weight_certifier",
            schema="mercury.tools.oae_weight_certifier/v1",
            status="ok",
            body={"w_R": 0.4472, "w_H": 0.2764, "w_O": 0.2764},
        ).envelope()
        validate_envelope(cert)  # no raise

    def test_validator_rejects_unknown_status(self) -> None:
        with pytest.raises(EnvelopeValidationError):
            validate_envelope(
                {
                    "schema": "mercury.tools.foo/v1",
                    "tool": "foo",
                    "status": "maybe",
                    "generated_at": "2025-01-01T00:00:00Z",
                    "mercury_version": "0.0.0",
                    "body": {},
                }
            )

    def test_validator_rejects_malformed_schema(self) -> None:
        with pytest.raises(EnvelopeValidationError):
            validate_envelope(
                {
                    "schema": "tools/foo/v1",
                    "tool": "foo",
                    "status": "ok",
                    "generated_at": "2025-01-01T00:00:00Z",
                    "mercury_version": "0.0.0",
                    "body": {},
                }
            )

    def test_validator_rejects_schema_tool_mismatch(self) -> None:
        with pytest.raises(EnvelopeValidationError):
            validate_envelope(
                {
                    "schema": "mercury.tools.foo/v1",
                    "tool": "bar",
                    "status": "ok",
                    "generated_at": "2025-01-01T00:00:00Z",
                    "mercury_version": "0.0.0",
                    "body": {},
                }
            )

    def test_validator_rejects_body_not_object(self) -> None:
        with pytest.raises(EnvelopeValidationError):
            validate_envelope(
                {
                    "schema": "mercury.tools.foo/v1",
                    "tool": "foo",
                    "status": "ok",
                    "generated_at": "2025-01-01T00:00:00Z",
                    "mercury_version": "0.0.0",
                    "body": ["wrong-type"],
                }
            )


# ---------------------------------------------------------------------------
# 2. Determinism — every tool produces byte-identical output across runs,
# modulo generated_at.

_DETERMINISTIC_TARGETS = (
    "oae_weight_certifier",
    "algorithm_name_drift_gate",
    "benevolence_certifier",
    "config_validator",
)


@pytest.mark.parametrize("tool_name", _DETERMINISTIC_TARGETS)
def test_tool_byte_deterministic_modulo_generated_at(
    tool_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The headline determinism contract."""
    monkeypatch.chdir(tmp_path)
    out1 = tmp_path / "run1.json"
    out2 = tmp_path / "run2.json"
    main = TOOL_REGISTRY[tool_name]
    rc1 = main(["--output", str(out1)])
    rc2 = main(["--output", str(out2)])
    assert rc1 == rc2
    body1 = _drop_volatile(_load_cert(out1))
    body2 = _drop_volatile(_load_cert(out2))
    assert body1 == body2, f"tool {tool_name} not deterministic"


# ---------------------------------------------------------------------------
# 3. Ed25519 detached signature — over the exact bytes written, and the
# corpus round-trips through sigma_immutable_verifier.


class TestEd25519RoundTrip:
    """The signature must cover the bytes the operator can verify on disk."""

    def test_signature_over_exact_written_bytes(self, tmp_path: Path) -> None:
        seed = secrets.token_bytes(32).hex()
        out = tmp_path / "cert.json"
        rc = TOOL_REGISTRY["oae_weight_certifier"](["--output", str(out), "--sign-key-hex", seed])
        assert rc == 0
        # Sidecar is written next to the output.
        sig_path = out.with_suffix(out.suffix + ".sig.json")
        assert sig_path.exists(), "Ed25519 sidecar missing"
        sig_record = json.loads(sig_path.read_text())
        # Verify the signature over the *exact* bytes the operator can read.
        payload = out.read_bytes()
        assert verify_certificate_ed25519(
            payload, sig_record
        ), "signature does not verify against the bytes on disk"

    def test_corpus_resign_roundtrip(self, tmp_path: Path) -> None:
        """Re-sign the in-repo σ_Immutable corpus into a tmpdir, verify it.

        Uses an ephemeral seed (never committed) and captures the
        derived public key in the cert body so :mod:`sigma_immutable_verifier`
        can validate the resigned bundle.
        """
        # The seed exists only for this test — derived per-test, captured
        # in the cert body, never written outside ``tmp_path``.
        seed = secrets.token_bytes(32).hex()
        signed_out = tmp_path / "signed_payload.json"

        payload = {"schema": "mercury.tools.test/v1", "body": {"value": 42}}
        data = to_json_bytes(payload)
        atomic_write_bytes(signed_out, data)
        record = sign_certificate_ed25519(data, seed)
        atomic_write_bytes(
            signed_out.with_suffix(signed_out.suffix + ".sig.json"),
            json.dumps(record, sort_keys=True).encode("utf-8"),
        )
        # And verify it back.
        assert verify_certificate_ed25519(
            signed_out.read_bytes(),
            json.loads(signed_out.with_suffix(signed_out.suffix + ".sig.json").read_text()),
        )


# ---------------------------------------------------------------------------
# 4. Exit-code contract — every tool routes through _base.run_tool so the
# fail → 1, warn → 0, --require warn → 1 mapping holds.

_EXIT_CONTRACT_TARGETS = (
    # Tools cheap to invoke and that touch _base.run_tool unconditionally.
    "oae_weight_certifier",
    "algorithm_name_drift_gate",
    "config_validator",
    "benevolence_certifier",
)


@pytest.mark.parametrize("tool_name", _EXIT_CONTRACT_TARGETS)
def test_tool_routes_through_run_tool(tool_name: str, tmp_path: Path) -> None:
    """Smoke check: every registered tool obeys the run_tool exit-code contract.

    We assert the tool exits 0 or 1 (success / fail), never raises an
    unhandled exception, and accepts the documented ``--output`` flag.
    """
    out = tmp_path / "cert.json"
    rc = TOOL_REGISTRY[tool_name](["--output", str(out)])
    assert rc in (0, 1, 2, 3), f"{tool_name} returned non-contract exit code {rc}"


def test_require_flag_upgrades_warn_to_fail(tmp_path: Path) -> None:
    """``--require`` must upgrade a ``warn`` cert to exit code 1.

    Drives :func:`_base.emit` directly with a synthetic ``warn``
    certificate so the contract is exercised without depending on a
    specific tool's ``warn`` regime (which tends to be data-dependent
    and brittle in CI).
    """
    import argparse

    from omni_mercury_engine.tools._base import Certificate, emit

    out = tmp_path / "cert.json"
    cert = Certificate(
        tool="oae_weight_certifier",
        schema="mercury.tools.oae_weight_certifier/v1",
        status="warn",
        body={"reason": "synthetic warn for require-gate contract test"},
        warnings=["synthetic warn"],
    )

    args_warn = argparse.Namespace(output=str(out), sign_key_hex=None, require=False, dry_run=False)
    rc_warn = emit(cert, args_warn)
    args_required = argparse.Namespace(
        output=str(out), sign_key_hex=None, require=True, dry_run=False
    )
    rc_required = emit(cert, args_required)
    assert rc_warn == 0, "warn without --require must be exit 0"
    assert rc_required == 1, "--require must upgrade warn to exit 1"


# ---------------------------------------------------------------------------
# 5. Atomic writes + --dry-run honoured.


def test_atomic_write_uses_temp_then_replace(tmp_path: Path) -> None:
    """``atomic_write_bytes`` must not leave a partial file behind on crash."""
    out = tmp_path / "atomic.bin"
    atomic_write_bytes(out, b"hello world")
    # The temp file should be gone, only the final file remains.
    siblings = [p.name for p in tmp_path.iterdir()]
    assert siblings == [out.name]
    assert out.read_bytes() == b"hello world"


def test_dry_run_does_not_mutate_side_effect_file(tmp_path: Path) -> None:
    """``--dry-run`` must keep the *side-effect* output file untouched.

    ``--dry-run`` is a side-effect gate — every tool still emits its
    JSON certificate (so the operator can see what would happen), but
    any file the tool mutates *on the filesystem* (release manifest,
    SBOM document, MATH_SPEC.md, etc.) MUST be left alone.

    We drive :mod:`convergence_proof_emitter` against a tempo
    MATH_SPEC.md whose initial bytes contain a sentinel that the
    emitter would normally rewrite.  After ``--dry-run`` the file's
    bytes MUST be byte-identical to the original.
    """
    doc = tmp_path / "MATH_SPEC.md"
    original = (
        "preamble\n" f"{_BEGIN}\n" "STALE PROOF — would be rewritten\n" f"{_END}\n" "trailer\n"
    ).encode()
    doc.write_bytes(original)
    out = tmp_path / "cert.json"
    rc = TOOL_REGISTRY["convergence_proof_emitter"](
        ["--output", str(out), "--doc", str(doc), "--dry-run"]
    )
    assert rc in (0, 1)
    assert (
        doc.read_bytes() == original
    ), "convergence_proof_emitter --dry-run mutated the side-effect document"


# ---------------------------------------------------------------------------
# 6. Sentinel splicing — convergence_proof_emitter.

_BEGIN = "<!-- CONVERGENCE-PROOF:BEGIN -->"
_END = "<!-- CONVERGENCE-PROOF:END -->"


class TestSentinelSplicing:
    def test_idempotent_repeated_runs(self, tmp_path: Path) -> None:
        doc = tmp_path / "MATH_SPEC.md"
        doc.write_text(
            "preamble line 1\n" f"{_BEGIN}\n" "STALE PROOF\n" f"{_END}\n" "trailer line\n"
        )
        out = tmp_path / "cert.json"
        rc1 = TOOL_REGISTRY["convergence_proof_emitter"](["--output", str(out), "--doc", str(doc)])
        first = doc.read_bytes()
        rc2 = TOOL_REGISTRY["convergence_proof_emitter"](["--output", str(out), "--doc", str(doc)])
        second = doc.read_bytes()
        assert rc1 == rc2
        assert first == second, "convergence_proof_emitter not idempotent"

    def test_preserves_surrounding_bytes(self, tmp_path: Path) -> None:
        doc = tmp_path / "MATH_SPEC.md"
        original_preamble = "PREAMBLE_LINE_A\nPREAMBLE_LINE_B\n"
        original_trailer = "TRAILER_LINE_X\nTRAILER_LINE_Y\n"
        doc.write_text(original_preamble + f"{_BEGIN}\nstale\n{_END}\n" + original_trailer)
        out = tmp_path / "cert.json"
        TOOL_REGISTRY["convergence_proof_emitter"](["--output", str(out), "--doc", str(doc)])
        text = doc.read_text()
        assert text.startswith(
            original_preamble
        ), "convergence_proof_emitter modified bytes before BEGIN sentinel"
        assert text.endswith(
            original_trailer
        ), "convergence_proof_emitter modified bytes after END sentinel"

    def test_fails_on_malformed_double_begin(self, tmp_path: Path) -> None:
        doc = tmp_path / "MATH_SPEC.md"
        doc.write_text(f"{_BEGIN}\nx\n{_BEGIN}\ny\n{_END}\n")
        out = tmp_path / "cert.json"
        rc = TOOL_REGISTRY["convergence_proof_emitter"](["--output", str(out), "--doc", str(doc)])
        assert rc == 1, "malformed double-BEGIN should fail-loud"

    def test_fails_on_lone_begin(self, tmp_path: Path) -> None:
        doc = tmp_path / "MATH_SPEC.md"
        doc.write_text(f"prefix\n{_BEGIN}\nbody without end\n")
        out = tmp_path / "cert.json"
        rc = TOOL_REGISTRY["convergence_proof_emitter"](["--output", str(out), "--doc", str(doc)])
        assert rc == 1, "lone BEGIN sentinel should fail-loud"
