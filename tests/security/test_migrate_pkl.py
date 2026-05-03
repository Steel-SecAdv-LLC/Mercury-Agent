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

Tests for the one-shot legacy ``.pkl`` migration CLI.

We invoke the tool via ``subprocess`` rather than importing it, because
the production behaviour is to re-launch in a hardened subprocess
(scrubbed env: ``PYTHONNOUSERSITE=1``, ``PYTHONDONTWRITEBYTECODE=1``,
no ``PYTHONSTARTUP``, no ``PYTHONPATH``). Importing in-process would
skip exactly the hardening we want to verify.
"""

from __future__ import annotations

import pickle  # only used to *create* test fixtures, never to load engine data
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from omni_mercury_engine.security.safe_load import (
    safe_load_training_data,
    verify_npz_signature,
)


def _run_tool(*args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the migration tool exactly the way an operator would."""
    # S603: command list is built from sys.executable, the module path of
    # the tool under test, and test-controlled args. No shell=True.
    return subprocess.run(  # noqa: S603  # nosec B603
        [sys.executable, "-m", "omni_mercury_engine.tools.migrate_pkl", *args],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def legacy_pkl(tmp_path: Path) -> Path:
    """A legitimate legacy training pickle in the expected schema."""
    payload = {
        "features": {
            "modality_a": np.random.rand(8, 16).astype(np.float32),
            "modality_b": np.random.rand(8, 4).astype(np.float32),
        },
        "labels": np.array([0, 1] * 4, dtype=np.int64),
    }
    path = tmp_path / "legacy.pkl"
    with path.open("wb") as f:
        pickle.dump(payload, f)
    return path


# --------------------------------------------------------------------------- #
# Hardened-subprocess relaunch.
# --------------------------------------------------------------------------- #


def test_relaunches_under_hardened_subprocess(tmp_path: Path, legacy_pkl: Path) -> None:
    """
    The top-level invocation prints the relaunch banner; the child runs
    in a hardened subprocess (scrubbed env). We verify by reading the
    stderr banner and confirming the child exits successfully.
    """
    out = _run_tool("--input", str(legacy_pkl), "--output", str(tmp_path / "out.npz"))
    assert "hardened subprocess" in out.stderr
    assert out.returncode == 0


# --------------------------------------------------------------------------- #
# Happy path: convert legitimate .pkl to .npz, then load via safe_load.
# --------------------------------------------------------------------------- #


def test_round_trip_pkl_to_npz(tmp_path: Path, legacy_pkl: Path) -> None:
    out_path = tmp_path / "out.npz"
    proc = _run_tool("--input", str(legacy_pkl), "--output", str(out_path))
    assert proc.returncode == 0, proc.stderr
    assert out_path.exists()

    loaded = safe_load_training_data(out_path)
    assert "labels" in loaded
    assert "modality_a" in loaded
    assert "modality_b" in loaded
    assert loaded["labels"].dtype == np.int64


def test_signing_during_migration(tmp_path: Path, legacy_pkl: Path) -> None:
    out_path = tmp_path / "signed.npz"
    key_hex = "11" * 32
    proc = _run_tool(
        "--input",
        str(legacy_pkl),
        "--output",
        str(out_path),
        "--sign-key-hex",
        key_hex,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_path.exists()
    assert (Path(str(out_path) + ".sig")).exists()
    verify_npz_signature(out_path, bytes.fromhex(key_hex))


# --------------------------------------------------------------------------- #
# Schema rejections. Object dtypes, malformed roots must be refused.
# --------------------------------------------------------------------------- #


def test_rejects_pickle_with_non_dict_root(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pkl"
    with bad.open("wb") as f:
        pickle.dump([1, 2, 3], f)
    proc = _run_tool("--input", str(bad), "--output", str(tmp_path / "out.npz"))
    assert proc.returncode == 3
    assert "must be a dict" in proc.stderr


def test_rejects_pickle_with_missing_keys(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pkl"
    with bad.open("wb") as f:
        pickle.dump({"only_features": {}}, f)
    proc = _run_tool("--input", str(bad), "--output", str(tmp_path / "out.npz"))
    assert proc.returncode == 3


def test_rejects_object_dtype_features(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pkl"
    payload = {
        "features": {"weird": np.array([{"k": 1}, {"k": 2}], dtype=object)},
        "labels": np.array([0, 1], dtype=np.int64),
    }
    with bad.open("wb") as f:
        pickle.dump(payload, f)
    proc = _run_tool("--input", str(bad), "--output", str(tmp_path / "out.npz"))
    assert proc.returncode == 3
    assert "object dtype" in proc.stderr


# --------------------------------------------------------------------------- #
# Filesystem safety: refuse to overwrite existing outputs.
# --------------------------------------------------------------------------- #


def test_refuses_to_overwrite_existing_output(tmp_path: Path, legacy_pkl: Path) -> None:
    existing = tmp_path / "out.npz"
    existing.write_bytes(b"placeholder")
    proc = _run_tool("--input", str(legacy_pkl), "--output", str(existing))
    assert proc.returncode == 2
    assert "refusing to overwrite" in proc.stderr
    # File untouched.
    assert existing.read_bytes() == b"placeholder"


def test_max_bytes_enforced(tmp_path: Path, legacy_pkl: Path) -> None:
    proc = _run_tool(
        "--input",
        str(legacy_pkl),
        "--output",
        str(tmp_path / "out.npz"),
        "--max-bytes",
        "10",
    )
    assert proc.returncode == 2
    assert "max-bytes" in proc.stderr
