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

Tests for ``omni_mercury_engine.security.safe_load``.

These tests pin two contracts:

1. **Pickle is gone.** ``train_fusion_model`` must not contain any
   pickle import or ``.pkl`` branch. A guardian test asserts the
   source text directly so a future refactor cannot quietly bring it
   back.
2. **The .npz loader is strict.** Any deviation from a clean numpy
   archive -- wrong magic, oversized file, pickled objects inside
   the archive, missing or mismatched HMAC -- raises
   :class:`UnsafePayloadError`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from omni_mercury_engine.security.safe_load import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_ENTRIES,
    DEFAULT_MAX_UNCOMPRESSED_BYTES,
    NPZ_MAGIC,
    SIG_SUFFIX,
    UnsafePayloadError,
    safe_load_training_data,
    sign_npz,
    verify_npz_signature,
)

# --------------------------------------------------------------------------- #
# Fixtures.
# --------------------------------------------------------------------------- #


@pytest.fixture
def good_npz(tmp_path: Path) -> Path:
    """A well-formed training archive."""
    path = tmp_path / "good.npz"
    np.savez(
        str(path),
        modality_a=np.random.rand(8, 16).astype(np.float32),
        modality_b=np.random.rand(8, 4).astype(np.float32),
        labels=np.array([0, 1] * 4, dtype=np.int64),
    )
    return path


@pytest.fixture
def hmac_key() -> bytes:
    return b"\x11" * 32


# --------------------------------------------------------------------------- #
# Guardian: pickle path must remain deleted.
# --------------------------------------------------------------------------- #


def test_engine_train_fusion_model_has_no_pickle_path() -> None:
    """train_fusion_model source must not reference pickle.

    This is the single most important contract: the dangerous code path
    we removed must not be reintroduced under any guise.

    Implemented as a text scan of ``engine.py`` so the test runs even
    when optional ML deps (torch, etc.) are not installed.
    """
    import omni_mercury_engine

    engine_path = Path(omni_mercury_engine.__file__).resolve().parent / "engine.py"
    text = engine_path.read_text(encoding="utf-8")

    # Locate the train_fusion_model body. We scan only that method to
    # avoid false positives in unrelated parts of the file.
    marker = "def train_fusion_model("
    start = text.find(marker)
    assert start != -1, "train_fusion_model not found in engine.py"
    # Method ends at the next top-level method (4-space indent + def) or EOF.
    rest = text[start:]
    end_rel = -1
    for i, line in enumerate(rest.splitlines(keepends=True)):
        if i == 0:
            offset = len(line)
            continue
        # Sibling method: starts with exactly 4 spaces then "def ".
        if line.startswith("    def ") and not line.startswith("        "):
            end_rel = offset
            break
        offset += len(line)
    body = rest if end_rel == -1 else rest[:end_rel]

    # Tokens that imply pickle CODE EXECUTION. Mere string mentions of
    # ".pkl" inside error messages (e.g. "convert legacy .pkl files via
    # ...") are intentional and harmless.
    forbidden = (
        "import pickle",
        "from pickle ",
        "pickle.load",
        "pickle.Unpickler",
        "_RestrictedUnpickler",
    )
    found = [tok for tok in forbidden if tok in body]
    assert not found, (
        f"train_fusion_model contains forbidden pickle tokens: {found}. "
        f"The pickle code path was removed and must not be reintroduced."
    )


# --------------------------------------------------------------------------- #
# Happy path: legitimate .npz archives load correctly.
# --------------------------------------------------------------------------- #


def test_load_legitimate_npz(good_npz: Path) -> None:
    out = safe_load_training_data(good_npz)
    assert set(out.keys()) == {"modality_a", "modality_b", "labels"}
    assert out["labels"].dtype == np.int64
    assert out["modality_a"].shape == (8, 16)


def test_load_legitimate_npz_with_uint8_image_data(tmp_path: Path) -> None:
    """The new loader accepts dtypes the old whitelist would reject."""
    path = tmp_path / "img.npz"
    arr = np.random.randint(0, 255, size=(4, 8, 8), dtype=np.uint8)
    np.savez(str(path), images=arr, labels=np.zeros(4, dtype=np.int64))
    out = safe_load_training_data(path)
    np.testing.assert_array_equal(out["images"], arr)


def test_load_legitimate_npz_with_bool_mask(tmp_path: Path) -> None:
    path = tmp_path / "mask.npz"
    arr = np.array([True, False, True], dtype=np.bool_)
    np.savez(str(path), mask=arr, labels=np.zeros(3, dtype=np.int64))
    out = safe_load_training_data(path)
    np.testing.assert_array_equal(out["mask"], arr)


def test_load_legitimate_npz_with_float16(tmp_path: Path) -> None:
    path = tmp_path / "fp16.npz"
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    np.savez(str(path), x=arr, labels=np.zeros(3, dtype=np.int64))
    out = safe_load_training_data(path)
    np.testing.assert_array_equal(out["x"], arr)


# --------------------------------------------------------------------------- #
# Path / filesystem rejections.
# --------------------------------------------------------------------------- #


def test_missing_path_rejected(tmp_path: Path) -> None:
    with pytest.raises(UnsafePayloadError, match="does not exist"):
        safe_load_training_data(tmp_path / "missing.npz")


def test_directory_rejected(tmp_path: Path) -> None:
    with pytest.raises(UnsafePayloadError, match="not a regular file"):
        safe_load_training_data(tmp_path)


def test_empty_file_rejected(tmp_path: Path) -> None:
    p = tmp_path / "empty.npz"
    p.write_bytes(b"")
    with pytest.raises(UnsafePayloadError, match="empty"):
        safe_load_training_data(p)


# --------------------------------------------------------------------------- #
# Magic bytes / format rejections.
# --------------------------------------------------------------------------- #


def test_wrong_magic_rejected(tmp_path: Path) -> None:
    p = tmp_path / "fake.npz"
    p.write_bytes(b"NOTAZIP" + b"\x00" * 100)
    with pytest.raises(UnsafePayloadError, match="magic"):
        safe_load_training_data(p)


def test_pickle_file_renamed_to_npz_rejected(tmp_path: Path) -> None:
    """Renaming a .pkl to .npz must not bypass the loader."""
    import pickle

    p = tmp_path / "evil.npz"
    with p.open("wb") as f:
        pickle.dump({"x": [1, 2, 3]}, f)
    with pytest.raises(UnsafePayloadError, match="magic"):
        safe_load_training_data(p)


def test_npz_containing_object_dtype_rejected(tmp_path: Path) -> None:
    """numpy refuses object arrays without allow_pickle; we surface that."""
    p = tmp_path / "objs.npz"
    # Build an .npz containing an object-dtype array. numpy pickles such
    # arrays into the archive on save, so they exercise the
    # allow_pickle=False guard on read.
    arr = np.array([{"k": 1}, {"k": 2}], dtype=object)
    np.savez(str(p), x=arr)
    with pytest.raises(UnsafePayloadError):
        safe_load_training_data(p)


# --------------------------------------------------------------------------- #
# Size ceiling.
# --------------------------------------------------------------------------- #


def test_size_ceiling_enforced(tmp_path: Path) -> None:
    p = tmp_path / "small.npz"
    np.savez(str(p), x=np.zeros(10, dtype=np.float32), labels=np.zeros(10, dtype=np.int64))
    with pytest.raises(UnsafePayloadError, match="size ceiling"):
        safe_load_training_data(p, max_bytes=10)


def test_size_ceiling_default_is_64mib() -> None:
    assert DEFAULT_MAX_BYTES == 64 * 1024 * 1024


def test_size_ceiling_can_be_raised(tmp_path: Path, good_npz: Path) -> None:
    # Default is 64 MiB; our test archive is well under that.
    out = safe_load_training_data(good_npz, max_bytes=128 * 1024 * 1024)
    assert "labels" in out


# --------------------------------------------------------------------------- #
# HMAC signing roundtrip.
# --------------------------------------------------------------------------- #


def test_sign_and_verify_roundtrip(good_npz: Path, hmac_key: bytes) -> None:
    sig_path = sign_npz(good_npz, hmac_key)
    assert sig_path == Path(str(good_npz) + SIG_SUFFIX)
    assert sig_path.exists()
    # Sidecar contains exactly one 64-char hex SHA-256.
    digest_hex = sig_path.read_text(encoding="ascii").strip()
    assert len(digest_hex) == 64
    int(digest_hex, 16)  # parse-checks hex
    # And it verifies.
    verify_npz_signature(good_npz, hmac_key)


def test_sign_then_load_with_verify(good_npz: Path, hmac_key: bytes) -> None:
    sign_npz(good_npz, hmac_key)
    out = safe_load_training_data(good_npz, verify_key=hmac_key)
    assert "labels" in out


def test_verify_rejects_tampered_payload(good_npz: Path, hmac_key: bytes) -> None:
    sign_npz(good_npz, hmac_key)
    # Mutate one byte well past the magic.
    raw = good_npz.read_bytes()
    tampered = raw[:200] + bytes([(raw[200] ^ 0xFF) & 0xFF]) + raw[201:]
    good_npz.write_bytes(tampered)
    with pytest.raises(UnsafePayloadError, match="signature mismatch"):
        verify_npz_signature(good_npz, hmac_key)


def test_verify_rejects_wrong_key(good_npz: Path, hmac_key: bytes) -> None:
    sign_npz(good_npz, hmac_key)
    other_key = b"\x22" * 32
    with pytest.raises(UnsafePayloadError, match="signature mismatch"):
        verify_npz_signature(good_npz, other_key)


def test_verify_rejects_missing_sidecar(good_npz: Path, hmac_key: bytes) -> None:
    with pytest.raises(UnsafePayloadError, match="Missing signature sidecar"):
        verify_npz_signature(good_npz, hmac_key)


def test_verify_rejects_malformed_sidecar(good_npz: Path, hmac_key: bytes) -> None:
    Path(str(good_npz) + SIG_SUFFIX).write_text("not-hex-not-64-chars", encoding="ascii")
    with pytest.raises(UnsafePayloadError):
        verify_npz_signature(good_npz, hmac_key)


def test_load_with_verify_fails_when_sidecar_missing(good_npz: Path, hmac_key: bytes) -> None:
    with pytest.raises(UnsafePayloadError, match="Missing signature sidecar"):
        safe_load_training_data(good_npz, verify_key=hmac_key)


def test_short_key_rejected(good_npz: Path) -> None:
    with pytest.raises(ValueError, match="at least 32 bytes"):
        sign_npz(good_npz, b"\x11" * 16)
    with pytest.raises(ValueError, match="at least 32 bytes"):
        verify_npz_signature(good_npz, b"\x11" * 16)


def test_non_bytes_key_rejected(good_npz: Path) -> None:
    with pytest.raises(TypeError, match="key must be bytes"):
        sign_npz(good_npz, "not-bytes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="key must be bytes"):
        verify_npz_signature(good_npz, "not-bytes")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Zip-bomb / decompression-DoS defense.
# --------------------------------------------------------------------------- #


def _make_bomb_npz(path: Path, *, ratio_target: float = 5000.0) -> None:
    """Write a real zip with a highly compressible (all-zero) payload.

    Zeros compress extraordinarily well with DEFLATE, so the resulting
    archive has a >1000:1 compress/uncompress ratio in its central
    directory -- the same shape a real zip-bomb would have. We exercise
    the loader's rejection path with a real archive rather than a
    fabricated header so the test is robust to any future change in
    how the loader inspects the zip.
    """
    import zipfile

    big_payload = b"\x00" * int(ratio_target * 4096)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("payload.npy", big_payload)

    # Sanity: confirm we hit a hostile ratio.
    with zipfile.ZipFile(path, "r") as zf:
        info = zf.infolist()[0]
        assert info.compress_size > 0
        assert (info.file_size / info.compress_size) > 1000, (
            "test fixture failed to produce a zip-bomb-shaped ratio"
        )


def test_zip_bomb_compression_ratio_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bomb.npz"
    _make_bomb_npz(p)
    with pytest.raises(UnsafePayloadError, match="zip-bomb signature"):
        safe_load_training_data(p)


def test_too_many_entries_rejected(tmp_path: Path) -> None:
    """An archive with thousands of entries is rejected before decompression."""
    import zipfile

    p = tmp_path / "many.npz"
    with zipfile.ZipFile(p, "w") as zf:
        for i in range(DEFAULT_MAX_ENTRIES + 5):
            zf.writestr(f"entry_{i}.npy", b"\x00")
    with pytest.raises(UnsafePayloadError, match="entries"):
        safe_load_training_data(p)


def test_corrupt_zip_rejected(tmp_path: Path) -> None:
    """File starts with .npz magic but has truncated central directory."""
    p = tmp_path / "corrupt.npz"
    p.write_bytes(NPZ_MAGIC + b"\x00" * 200)
    with pytest.raises(UnsafePayloadError, match="not a valid zip archive"):
        safe_load_training_data(p)


def test_path_traversal_entry_rejected(tmp_path: Path) -> None:
    """Entry with ``..`` in its path is rejected even if size is fine."""
    import zipfile

    p = tmp_path / "evil.npz"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("../escape.npy", b"\x00" * 16)
    with pytest.raises(UnsafePayloadError, match="suspicious entry name"):
        safe_load_training_data(p)


def test_backslash_path_traversal_entry_rejected(tmp_path: Path) -> None:
    """Backslash-laced traversal must be rejected on POSIX too.

    On POSIX, ``Path('..\\\\escape.npy').parts`` is a single-component
    tuple because backslash is a literal filename character. A naive
    check on ``parts`` would let this through. We reject any backslash
    in the entry name outright -- numpy never writes backslashes, and
    a Windows extractor would treat ``\\`` as a directory separator.
    """
    import zipfile

    p = tmp_path / "evil_bs.npz"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("..\\escape.npy", b"\x00" * 16)
    with pytest.raises(UnsafePayloadError, match="suspicious entry name"):
        safe_load_training_data(p)


def test_embedded_backslash_entry_rejected(tmp_path: Path) -> None:
    """Even names without a leading ``..`` but with backslashes are rejected."""
    import zipfile

    p = tmp_path / "evil_embed.npz"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("foo\\..\\bar.npy", b"\x00" * 16)
    with pytest.raises(UnsafePayloadError, match="suspicious entry name"):
        safe_load_training_data(p)


def test_drive_letter_entry_rejected(tmp_path: Path) -> None:
    """Windows-style drive-letter prefix is rejected."""
    import zipfile

    p = tmp_path / "evil_drive.npz"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("C:evil.npy", b"\x00" * 16)
    with pytest.raises(UnsafePayloadError, match="suspicious entry name"):
        safe_load_training_data(p)


def test_uncompressed_size_ceiling_enforced(tmp_path: Path, good_npz: Path) -> None:
    """Caller-supplied tighter uncompressed-size cap is honored."""
    # good_npz contains a few small float arrays; cap to 1 byte to force rejection.
    with pytest.raises(UnsafePayloadError, match="uncompressed"):
        safe_load_training_data(good_npz, max_uncompressed_bytes=1)


def test_default_uncompressed_ceiling_is_one_gib() -> None:
    assert DEFAULT_MAX_UNCOMPRESSED_BYTES == 1024 * 1024 * 1024


def test_default_max_entries_is_256() -> None:
    assert DEFAULT_MAX_ENTRIES == 256


# --------------------------------------------------------------------------- #
# Exception translation: every failure path raises UnsafePayloadError.
# --------------------------------------------------------------------------- #


def test_toctou_corruption_after_validation_raises_unsafe_payload(
    good_npz: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file that passes the central-directory check but is corrupt by
    the time numpy reads it must surface as UnsafePayloadError, not as
    a raw zipfile.BadZipFile leaked to the caller.

    We simulate the TOCTOU window by monkey-patching np.load to raise
    BadZipFile -- exactly what happens if the file is truncated or
    rewritten between our pre-check and numpy's own zip parse.
    """
    import zipfile as _zipfile_mod

    import numpy as _np_mod

    def _explode(*_args: object, **_kwargs: object) -> object:
        raise _zipfile_mod.BadZipFile("simulated TOCTOU corruption")

    monkeypatch.setattr(_np_mod, "load", _explode)
    with pytest.raises(UnsafePayloadError, match="TOCTOU"):
        safe_load_training_data(good_npz)


def test_oserror_during_load_raises_unsafe_payload(
    good_npz: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Filesystem failures during np.load (e.g., file deleted, EIO) are
    translated to UnsafePayloadError so callers see one exception type."""
    import numpy as _np_mod

    def _explode(*_args: object, **_kwargs: object) -> object:
        raise OSError("simulated disk error")

    monkeypatch.setattr(_np_mod, "load", _explode)
    with pytest.raises(UnsafePayloadError, match="Failed to read"):
        safe_load_training_data(good_npz)


def test_keyerror_during_load_raises_unsafe_payload(
    good_npz: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KeyError from a malformed NpzFile entry is translated."""
    import numpy as _np_mod

    def _explode(*_args: object, **_kwargs: object) -> object:
        raise KeyError("malformed entry")

    monkeypatch.setattr(_np_mod, "load", _explode)
    with pytest.raises(UnsafePayloadError, match=r"Malformed \.npz"):
        safe_load_training_data(good_npz)


# --------------------------------------------------------------------------- #
# Constants are exported and stable.
# --------------------------------------------------------------------------- #


def test_npz_magic_is_zip_local_header() -> None:
    assert NPZ_MAGIC == b"PK\x03\x04"


def test_sig_suffix_default() -> None:
    assert SIG_SUFFIX == ".sig"
