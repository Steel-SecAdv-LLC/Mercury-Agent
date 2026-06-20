# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Guardrail: the pyca ``cryptography`` library stays inside its boundary.

Dependency sovereignty for the crypto layer. AMA Cryptography owns the
post-quantum path (ML-DSA-65 / Kyber-1024 / SPHINCS+). The third-party
``cryptography`` (pyca) package is *intentional but bounded*: it supplies the
classical Ed25519 half of the hybrid signature, AEAD/KDF primitives, and the
Known-Answer-Test (KAT) validation harnesses. That split is correct — but only
as long as pyca cannot quietly leak into modules it has no business in, which is
exactly how a PQC path silently degrades to classical-only crypto.

This guard fails loudly if any ``src/omni_mercury_engine`` module imports
``cryptography`` from **outside** the allowlisted crypto/security/tools surface.
It is the crypto-layer sibling of ``tests/test_no_sklearn_in_src.py``: lock the
verified boundary so a future change cannot move pyca onto an AMA-owned path
without this test going red.
"""

from __future__ import annotations

import ast
from pathlib import Path

# Repo root = two levels up from this file: tests/<this file> -> tests -> repo.
_REPO = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO / "src" / "omni_mercury_engine"

# Package-relative prefixes permitted to import pyca ``cryptography``. These are
# the modules that legitimately own classical-hybrid crypto + KAT validation:
#   crypto/    -- classical Ed25519 hybrid half, AEAD/KDF primitives
#   security/  -- crypto_api surface that composes AMA + classical
#   tools/     -- standalone KAT runners / PQC capability probes
_ALLOWED_PREFIXES: tuple[str, ...] = ("crypto/", "security/", "tools/")


def _src_python_files() -> list[Path]:
    """All shipped engine modules."""
    return [p for p in _SRC_ROOT.rglob("*.py")]


def _imports_pyca_cryptography(path: Path) -> bool:
    """True iff the module has a real ``import cryptography`` / ``from cryptography`` (AST)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                a.name == "cryptography" or a.name.startswith("cryptography.") for a in node.names
            ):
                return True
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "cryptography" or mod.startswith("cryptography."):
                return True
    return False


def _rel(path: Path) -> str:
    """Path relative to the engine package root, forward-slashed."""
    return path.relative_to(_SRC_ROOT).as_posix()


def _is_allowed(path: Path) -> bool:
    return _rel(path).startswith(_ALLOWED_PREFIXES)


def test_pyca_cryptography_confined_to_crypto_security_tools() -> None:
    """No engine module outside crypto/security/tools may import pyca cryptography."""
    offenders = sorted(
        _rel(p) for p in _src_python_files() if _imports_pyca_cryptography(p) and not _is_allowed(p)
    )
    assert not offenders, (
        "pyca `cryptography` must stay inside the crypto/security/tools boundary "
        "(AMA owns the PQC path). Move the primitive behind the AMA/crypto API, or "
        "justify and extend _ALLOWED_PREFIXES. Offending modules:\n  " + "\n  ".join(offenders)
    )


def test_guard_actually_observes_known_importers() -> None:
    """Non-vacuity: the scan really sees the (allowlisted) pyca importers.

    If the scan silently found nothing, the boundary test above would pass
    vacuously. Pin that the known classical-crypto importers are detected.
    """
    importers = {_rel(p) for p in _src_python_files() if _imports_pyca_cryptography(p)}
    assert importers, "expected to find pyca importers under crypto/security/tools; found none"
    # every detected importer must already be inside the allowed boundary
    assert all(
        prefix_ok for prefix_ok in (s.startswith(_ALLOWED_PREFIXES) for s in importers)
    ), f"a pyca importer sits outside the allowed boundary: {sorted(importers)}"
