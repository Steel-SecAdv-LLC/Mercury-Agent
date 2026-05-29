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

Pickle-free training-data loader.

Mercury Agent does not deserialize Python pickles. Pickle is a
stack-based VM whose opcodes can resolve any importable callable; even
"safe" whitelists are a brittle defense over a structurally hostile
format. We accept training data only as numpy ``.npz`` archives, which
are pure binary tensor containers with no execution semantics.

This module provides:

* :func:`safe_load_training_data` -- the only sanctioned entry point
  for loading on-disk training tensors. Enforces magic bytes, on-disk
  size ceiling, zip central-directory inspection (per-entry and total
  uncompressed-size limits, entry-count cap, suspicious-name guard,
  compression-ratio guard against zip bombs), and
  ``allow_pickle=False``.
* :func:`sign_npz` / :func:`verify_npz_signature` -- optional HMAC-SHA-256
  provenance via a sidecar ``.npz.sig`` file. Implemented with the
  Python standard library (``hmac`` and ``hashlib``) so the loader has
  no third-party crypto dependency at import time and can be used in
  minimal-install environments.

The pickle-based code path that previously lived inline in
``omni_mercury_engine.engine.OmniMercuryEngine.train_fusion_model`` has
been deleted. Legacy ``.pkl`` payloads must be converted once via
``python -m omni_mercury_engine.tools.migrate_pkl`` (an isolated
subprocess that never touches the engine).
"""

from __future__ import annotations

import hashlib
import hmac
import zipfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import os

__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_ENTRIES",
    "DEFAULT_MAX_UNCOMPRESSED_BYTES",
    "NPZ_MAGIC",
    "SIG_SUFFIX",
    "UnsafePayloadError",
    "safe_load_training_data",
    "sign_npz",
    "verify_npz_signature",
]


# .npz files are zip archives; the zip "local file header" magic is PK\x03\x04.
NPZ_MAGIC: bytes = b"PK\x03\x04"

# Default 64 MiB on-disk ceiling. Override per-call when a larger payload is
# expected from a trusted source.
DEFAULT_MAX_BYTES: int = 64 * 1024 * 1024

# Default 1 GiB total uncompressed ceiling. .npz is a zip container, so a
# small file on disk can expand into very large arrays; this guard bounds the
# decompression-bomb attack surface independently of on-disk size.
DEFAULT_MAX_UNCOMPRESSED_BYTES: int = 1024 * 1024 * 1024

# Default upper bound on the number of entries in the archive. Legitimate
# Mercury training payloads contain a handful of named arrays; an archive
# with thousands of entries is almost certainly hostile.
DEFAULT_MAX_ENTRIES: int = 256

# Sidecar signature suffix.
SIG_SUFFIX: str = ".sig"


class UnsafePayloadError(ValueError):
    """
    Raised when a payload is rejected by the safe loader.

    The exception message describes the precise reason (size, magic, pickle content, signature
    mismatch). It never echoes payload bytes.
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


def _validate_zip_central_directory(
    p: Path,
    *,
    max_uncompressed_bytes: int,
    max_entries: int,
) -> None:
    r"""
    Inspect the zip central directory before letting numpy decompress.

    .npz is a zip container. Without this guard, a small file on disk
    can expand to tens of GiB of numpy arrays in memory (decompression
    bomb). We read only the central directory metadata and reject the
    archive before any decompression happens.

    Rejects, in order:

    * Files that aren't valid zip archives (corrupt or truncated).
    * Archives with more than ``max_entries`` members.
    * Per-entry uncompressed sizes greater than ``max_uncompressed_bytes``.
    * Cumulative uncompressed size greater than ``max_uncompressed_bytes``.
    * Compression ratios per entry greater than 1000:1, which is
      characteristic of zip-bomb constructions and never produced by
      legitimate numpy savez output.
    * Entry names that contain path-traversal components (``..``,
      leading ``/``, drive letters) or ``\\`` (backslash) -- numpy
      doesn't write any of these, so any presence indicates tampering.
      We reject backslashes outright because POSIX path parsing keeps
      ``\\`` as a literal character (so ``..\\escape.npy`` would slip
      past a naive parts-check on a POSIX runtime), and Windows zip
      consumers would interpret it as a directory separator on
      extraction. Names are then parsed with ``PurePosixPath`` for
      ``..`` detection so behaviour is identical regardless of the
      platform Mercury Agent runs on.
    """
    try:
        with zipfile.ZipFile(p, "r") as zf:
            infos = zf.infolist()
    except zipfile.BadZipFile as exc:
        raise UnsafePayloadError(f"{p} is not a valid zip archive: {exc}") from exc

    if len(infos) == 0:
        raise UnsafePayloadError(f"{p} is an empty zip archive")
    if len(infos) > max_entries:
        raise UnsafePayloadError(
            f"{p} has {len(infos)} entries (max {max_entries}); refusing to load"
        )

    cumulative = 0
    for info in infos:
        name = info.filename
        if (
            "\\" in name  # see docstring: backslash is hostile in zip names
            or name.startswith("/")
            or ".." in PurePosixPath(name).parts
            or (len(name) >= 2 and name[1] == ":")
        ):
            raise UnsafePayloadError(
                f"{p} contains suspicious entry name {name!r}; refusing to load"
            )
        if info.file_size < 0:
            raise UnsafePayloadError(f"{p} entry {name!r} reports negative uncompressed size")
        if info.file_size > max_uncompressed_bytes:
            raise UnsafePayloadError(
                f"{p} entry {name!r} uncompressed size {info.file_size} "
                f"exceeds limit {max_uncompressed_bytes}"
            )
        # Compression-ratio guard. A modest ratio is normal for numpy
        # arrays; anything beyond ~1000:1 is a bomb signature.
        if info.compress_size > 0:
            ratio = info.file_size / info.compress_size
            if ratio > 1000:
                raise UnsafePayloadError(
                    f"{p} entry {name!r} has compression ratio {ratio:.0f}:1 "
                    f"(>1000:1); refusing to load (zip-bomb signature)"
                )
        cumulative += info.file_size
        if cumulative > max_uncompressed_bytes:
            raise UnsafePayloadError(
                f"{p} cumulative uncompressed size exceeds {max_uncompressed_bytes} "
                f"bytes; refusing to load"
            )


def safe_load_training_data(
    path: str | os.PathLike[str],
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_uncompressed_bytes: int = DEFAULT_MAX_UNCOMPRESSED_BYTES,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    verify_key: bytes | None = None,
) -> dict[str, np.ndarray[Any, Any]]:
    """
    Load a training payload from a numpy ``.npz`` archive.

    This is the only sanctioned loader. Pickle is **not** supported and
    will not be supported. ``allow_pickle=False`` is enforced
    unconditionally. The archive is also screened for zip-bomb
    decompression attacks before any data is read.

    Parameters
    ----------
    path:
        Filesystem path to a ``.npz`` archive.
    max_bytes:
        Reject the file if its on-disk size exceeds this many bytes.
        Default :data:`DEFAULT_MAX_BYTES` (64 MiB). Raise this
        explicitly when a larger payload is expected from a trusted
        source.
    max_uncompressed_bytes:
        Reject the archive if any single entry, or the cumulative
        uncompressed payload, exceeds this many bytes. Default
        :data:`DEFAULT_MAX_UNCOMPRESSED_BYTES` (1 GiB). Tighten this
        for memory-constrained workers.
    max_entries:
        Reject the archive if it contains more than this many entries.
        Default :data:`DEFAULT_MAX_ENTRIES` (256). Legitimate Mercury
        training payloads contain a handful of named arrays.
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
        If any safety check fails (missing file, wrong magic, on-disk
        size ceiling, zip-bomb signature, suspicious entry name,
        signature mismatch, or attempted pickle content).
    """
    p = _validate_path(path)
    _validate_size(p, max_bytes)
    _validate_magic(p)
    _validate_zip_central_directory(
        p,
        max_uncompressed_bytes=max_uncompressed_bytes,
        max_entries=max_entries,
    )

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
    except zipfile.BadZipFile as exc:
        # Central-directory check passed earlier, but the archive became
        # invalid by the time numpy tried to read it (TOCTOU: the file
        # was modified or truncated between our pre-check and np.load).
        raise UnsafePayloadError(
            f"{p} became an invalid zip between validation and load (TOCTOU): {exc}"
        ) from exc
    except OSError as exc:
        # Filesystem-level failure: file deleted, permission revoked,
        # disk error, etc. Surface as an UnsafePayloadError so callers
        # see one exception type from this loader.
        raise UnsafePayloadError(f"Failed to read {p}: {exc}") from exc
    except (KeyError, EOFError) as exc:
        # numpy.NpzFile can raise KeyError on a malformed entry name and
        # EOFError on a truncated DEFLATE stream. Translate both.
        raise UnsafePayloadError(f"Malformed .npz {p}: {exc}") from exc


# --------------------------------------------------------------------------- #
# Optional HMAC-SHA-256 provenance.
# --------------------------------------------------------------------------- #


def _sig_path(path: str | os.PathLike[str]) -> Path:
    return Path(str(path) + SIG_SUFFIX)


def sign_npz(path: str | os.PathLike[str], key: bytes) -> Path:
    """
    Compute HMAC-SHA-256 over the file contents and write a sidecar.

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
        sig_path.chmod(0o600)
    except OSError:
        pass
    return sig_path


def verify_npz_signature(path: str | os.PathLike[str], key: bytes) -> None:
    """
    Verify a sidecar HMAC-SHA-256 signature, raising on mismatch.

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


# The "pickle path is gone" guardian lives in
# ``tests/security/test_safe_load.py::test_engine_train_fusion_model_has_no_pickle_path``.
# That test reads ``engine.py`` directly as text so it runs even when
# optional ML deps are absent, which is the most reliable contract for
# CI. No additional sentinel is exported from this module on purpose --
# duplication only invites the two checks to drift.
