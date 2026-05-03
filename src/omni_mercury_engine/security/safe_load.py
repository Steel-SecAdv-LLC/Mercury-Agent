"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

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

Pickle-free training-data loader.

Mercury Agent does not deserialize Python pickles. Pickle is a
stack-based VM whose opcodes can resolve any importable callable; even
"safe" whitelists are a brittle defense over a structurally hostile
format. We accept training data only as numpy ``.npz`` archives, which
are pure binary tensor containers with no execution semantics.

This module provides:

* :func:`safe_load_training_data` -- the only sanctioned entry point
  for loading on-disk training tensors. Enforces magic bytes, size
  ceiling, and ``allow_pickle=False``.
* :func:`sign_npz` / :func:`verify_npz_signature` -- optional HMAC-SHA-256
  provenance via a sidecar ``.npz.sig`` file, using the ``cryptography``
  library already in the project's runtime dependencies. No third-party
  signing format introduced.

The pickle-based code path that previously lived inline in
``omni_mercury_engine.engine.OmniMercuryEngine.train_fusion_model`` has
been deleted. Legacy ``.pkl`` payloads must be converted once via
``python -m omni_mercury_engine.tools.migrate_pkl`` (an isolated
subprocess that never touches the engine).
"""

from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_MAX_BYTES",
    "NPZ_MAGIC",
    "SIG_SUFFIX",
    "UnsafePayloadError",
    "safe_load_training_data",
    "sign_npz",
    "verify_npz_signature",
]


# .npz files are zip archives; the zip "local file header" magic is PK\x03\x04.
NPZ_MAGIC: bytes = b"PK\x03\x04"

# 64 MiB default ceiling. Override per-call when a larger payload is expected.
DEFAULT_MAX_BYTES: int = 64 * 1024 * 1024

# Sidecar signature suffix.
SIG_SUFFIX: str = ".sig"


class UnsafePayloadError(ValueError):
    """Raised when a payload is rejected by the safe loader.

    The exception message describes the precise reason (size, magic,
    pickle content, signature mismatch). It never echoes payload bytes.
    """


def _validate_path(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    if not p.exists():
        raise UnsafePayloadError(f"Training data path does not exist: {p}")
    if not p.is_file():
        raise UnsafePayloadError(f"Training data path is not a regular file: {p}")
    return p


def _validate_size(p: Path, max_bytes: int) -> int:
    size = p.stat().st_size
    if size <= 0:
        raise UnsafePayloadError(f"Training data file is empty: {p}")
    if size > max_bytes:
        raise UnsafePayloadError(
            f"Training data exceeds size ceiling: {size} bytes > {max_bytes} bytes "
            f"(raise max_bytes explicitly if this payload is trusted)"
        )
    return size


def _validate_magic(p: Path) -> None:
    with p.open("rb") as f:
        head = f.read(len(NPZ_MAGIC))
    if head != NPZ_MAGIC:
        raise UnsafePayloadError(
            f"File does not have .npz magic bytes (got {head!r}); "
            f"only numpy .npz archives are accepted. Convert legacy .pkl "
            f"files with `python -m omni_mercury_engine.tools.migrate_pkl`."
        )


def safe_load_training_data(
    path: str | os.PathLike[str],
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    verify_key: bytes | None = None,
) -> dict[str, np.ndarray]:
    """Load a training payload from a numpy ``.npz`` archive.

    This is the only sanctioned loader. Pickle is **not** supported and
    will not be supported. ``allow_pickle=False`` is enforced
    unconditionally.

    Parameters
    ----------
    path:
        Filesystem path to a ``.npz`` archive.
    max_bytes:
        Reject the file if it is larger than this many bytes. Default
        :data:`DEFAULT_MAX_BYTES` (64 MiB). Raise this explicitly when a
        larger payload is expected from a trusted source.
    verify_key:
        If provided, a sidecar ``<path>.sig`` file must exist and
        contain a valid HMAC-SHA-256 of the archive bytes under this
        key. The archive will not be loaded if verification fails.

    Returns
    -------
    dict
        Mapping of array name to ``numpy.ndarray``. The caller is
        responsible for converting to tensors.

    Raises
    ------
    UnsafePayloadError
        If any safety check fails (missing file, wrong magic, size
        ceiling, signature mismatch, or attempted pickle content).
    """
    p = _validate_path(path)
    _validate_size(p, max_bytes)
    _validate_magic(p)

    if verify_key is not None:
        verify_npz_signature(p, verify_key)

    try:
        with np.load(str(p), allow_pickle=False) as archive:
            # Materialize eagerly so the caller gets plain numpy arrays
            # rather than a lazy NpzFile that could be tampered with on
            # disk between iteration steps.
            return {name: np.array(archive[name]) for name in archive.files}
    except ValueError as exc:
        # numpy raises ValueError when a .npz contains pickled objects
        # and allow_pickle=False. Translate to our explicit error type.
        if "allow_pickle" in str(exc).lower() or "pickle" in str(exc).lower():
            raise UnsafePayloadError(
                f"Refusing to load {p}: archive contains pickled Python "
                f"objects. Re-export training data using only "
                f"numpy-native dtypes (int, float, bool, complex, "
                f"unicode, bytes)."
            ) from exc
        raise UnsafePayloadError(f"Failed to parse {p} as .npz: {exc}") from exc


# --------------------------------------------------------------------------- #
# Optional HMAC-SHA-256 provenance.
# --------------------------------------------------------------------------- #


def _sig_path(path: str | os.PathLike[str]) -> Path:
    return Path(str(path) + SIG_SUFFIX)


def sign_npz(path: str | os.PathLike[str], key: bytes) -> Path:
    """Compute HMAC-SHA-256 over the file contents and write a sidecar.

    The sidecar path is ``<path>.sig`` and contains a single hex digest
    (64 characters, no trailing whitespace). This format is intentionally
    simple so it can be inspected, signed by external tooling, or
    re-implemented without depending on any pickle-adjacent format.

    Parameters
    ----------
    path:
        Filesystem path to a ``.npz`` archive that already exists.
    key:
        HMAC key. Must be at least 32 bytes; longer is fine.

    Returns
    -------
    pathlib.Path
        The sidecar signature path.
    """
    if not isinstance(key, (bytes, bytearray)):
        raise TypeError(f"key must be bytes, got {type(key).__name__}")
    if len(key) < 32:
        raise ValueError("HMAC key must be at least 32 bytes")
    p = _validate_path(path)

    digest = _file_hmac(p, bytes(key))
    sig_path = _sig_path(p)
    sig_path.write_text(digest.hex(), encoding="ascii")
    # Best-effort: tighten permissions so the signature is owner-only.
    try:
        os.chmod(sig_path, 0o600)
    except OSError:
        pass
    return sig_path


def verify_npz_signature(path: str | os.PathLike[str], key: bytes) -> None:
    """Verify a sidecar HMAC-SHA-256 signature, raising on mismatch.

    Parameters
    ----------
    path:
        Filesystem path to a ``.npz`` archive.
    key:
        HMAC key, same as was passed to :func:`sign_npz`. Must be at
        least 32 bytes.

    Raises
    ------
    UnsafePayloadError
        If the sidecar is missing, malformed, or does not match.
    """
    if not isinstance(key, (bytes, bytearray)):
        raise TypeError(f"key must be bytes, got {type(key).__name__}")
    if len(key) < 32:
        raise ValueError("HMAC key must be at least 32 bytes")

    p = _validate_path(path)
    sig_path = _sig_path(p)
    if not sig_path.exists():
        raise UnsafePayloadError(f"Missing signature sidecar: {sig_path}")

    try:
        expected_hex = sig_path.read_text(encoding="ascii").strip()
        expected = bytes.fromhex(expected_hex)
    except (OSError, ValueError) as exc:
        raise UnsafePayloadError(f"Malformed signature sidecar {sig_path}: {exc}") from exc

    if len(expected) != hashlib.sha256().digest_size:
        raise UnsafePayloadError(
            f"Signature length {len(expected)} != SHA-256 digest size "
            f"{hashlib.sha256().digest_size}"
        )

    actual = _file_hmac(p, bytes(key))
    if not hmac.compare_digest(actual, expected):
        raise UnsafePayloadError(
            f"HMAC signature mismatch for {p}: payload has been "
            f"modified or signed with a different key"
        )


def _file_hmac(p: Path, key: bytes, *, chunk: int = 1 << 20) -> bytes:
    """Stream HMAC-SHA-256 over a file in 1 MiB blocks."""
    mac = hmac.new(key, digestmod=hashlib.sha256)
    with p.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            mac.update(buf)
    return mac.digest()


def assert_pickle_path_removed(loaded_module: Any) -> None:
    """Sentinel used by tests: confirm engine.py has no pickle import path.

    Imports the engine source and asserts neither ``pickle`` nor
    ``_RestrictedUnpickler`` are referenced inside ``train_fusion_model``.
    Tests call this so a future refactor cannot silently reintroduce the
    dangerous code path without breaking CI.
    """
    import inspect

    src = inspect.getsource(loaded_module.OmniMercuryEngine.train_fusion_model)
    forbidden = ("pickle.", "import pickle", "_RestrictedUnpickler", ".pkl", ".pickle")
    found = [tok for tok in forbidden if tok in src]
    if found:
        raise AssertionError(
            f"train_fusion_model contains forbidden pickle tokens: {found}. "
            f"The pickle code path was removed in v1.7.0 and must not "
            f"be reintroduced."
        )
