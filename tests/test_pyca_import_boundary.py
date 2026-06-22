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
            # Absolute imports only (level == 0): a relative ``from .cryptography
            # import ...`` is a local module, not the pyca package.
            mod = node.module or ""
            if node.level == 0 and (mod == "cryptography" or mod.startswith("cryptography.")):
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


def test_scan_is_non_vacuous_and_importers_in_boundary() -> None:
    """The engine scan is real, and any pyca importers it finds are in-boundary.

    Non-vacuity is pinned on the *scan* (engine modules are actually walked), not
    on pyca being present: eliminating the pyca dependency entirely is a
    legitimate future hardening that must not false-fail this guard, so an empty
    importer set is allowed — only an *out-of-boundary* importer fails.
    """
    files = _src_python_files()
    assert files, "engine module scan is empty — the boundary test could pass vacuously."
    importers = {_rel(p) for p in files if _imports_pyca_cryptography(p)}
    assert all(
        s.startswith(_ALLOWED_PREFIXES) for s in importers
    ), f"a pyca importer sits outside the allowed boundary: {sorted(importers)}"


def test_scanner_detects_absolute_and_ignores_relative(tmp_path: Path) -> None:
    """``_imports_pyca_cryptography`` flags absolute pyca imports and ignores a
    relative import of a local module named ``cryptography`` (node.level > 0).

    Fixture-based so it stays meaningful even if the repo ever drops pyca, and so
    it pins the absolute-vs-relative distinction directly.
    """
    (tmp_path / "abs1.py").write_text("import cryptography\n")
    (tmp_path / "abs2.py").write_text("from cryptography.hazmat.primitives import hashes\n")
    (tmp_path / "rel.py").write_text("from .cryptography import helper\n")
    (tmp_path / "benign.py").write_text("import numpy  # cryptography in a comment\n")
    assert _imports_pyca_cryptography(tmp_path / "abs1.py") is True
    assert _imports_pyca_cryptography(tmp_path / "abs2.py") is True
    assert _imports_pyca_cryptography(tmp_path / "rel.py") is False  # relative -> local
    assert _imports_pyca_cryptography(tmp_path / "benign.py") is False
