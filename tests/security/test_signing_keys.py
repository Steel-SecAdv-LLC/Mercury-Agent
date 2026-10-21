# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Owner-only ML-DSA-65 key files (security/signing_keys.py)."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import pytest

from omni_mercury_engine.security import signing_keys
from omni_mercury_engine.security.signing_keys import SigningKeyError

if TYPE_CHECKING:
    from pathlib import Path


def _crypto() -> None:
    try:
        from omni_mercury_engine.security.crypto_api import MLDSAProvider

        MLDSAProvider()
    except (ImportError, RuntimeError):  # pragma: no cover - AMA is required in CI
        pytest.skip("ML-DSA-65 needs the AMA native backend")


def test_generate_then_load_round_trips(tmp_path: Path) -> None:
    _crypto()
    path = tmp_path / "keys" / "adjudication.key"
    public_hex = signing_keys.generate_keyfile(path)
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600
    public_key, secret_key = signing_keys.load_keypair(path)
    assert public_key.hex() == public_hex
    assert secret_key


def test_an_existing_key_is_never_overwritten(tmp_path: Path) -> None:
    _crypto()
    path = tmp_path / "adjudication.key"
    signing_keys.generate_keyfile(path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        signing_keys.generate_keyfile(path)
    assert path.read_bytes() == before


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_a_group_readable_key_is_refused(tmp_path: Path) -> None:
    _crypto()
    path = tmp_path / "adjudication.key"
    signing_keys.generate_keyfile(path)
    path.chmod(0o640)
    with pytest.raises(SigningKeyError, match="chmod 600"):
        signing_keys.load_keypair(path)


def test_a_mismatched_pair_is_refused(tmp_path: Path) -> None:
    _crypto()
    first, second = tmp_path / "a.key", tmp_path / "b.key"
    signing_keys.generate_keyfile(first)
    signing_keys.generate_keyfile(second)
    document = json.loads(first.read_text())
    document["public_key_hex"] = json.loads(second.read_text())["public_key_hex"]
    first.write_text(json.dumps(document))
    first.chmod(0o600)
    with pytest.raises(SigningKeyError, match="does not match"):
        signing_keys.load_keypair(first)


@pytest.mark.parametrize(
    "content",
    ["not json", json.dumps({"schema": "other", "algorithm": "ML-DSA-65"}), json.dumps([])],
)
def test_malformed_files_are_refused(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.key"
    path.write_text(content)
    path.chmod(0o600)
    with pytest.raises(SigningKeyError):
        signing_keys.load_keypair(path)


def test_missing_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SigningKeyError, match="not readable"):
        signing_keys.load_keypair(tmp_path / "absent.key")


def test_env_helpers(tmp_path: Path) -> None:
    assert signing_keys.keypair_from_env(environ={}) is None
    assert signing_keys.pinned_public_key_from_env(environ={}) is None
    env = {signing_keys.ADJUDICATION_PUBLIC_KEY_ENV: " ABCD "}
    assert signing_keys.pinned_public_key_from_env(environ=env) == "abcd"
    with pytest.raises(SigningKeyError, match="not a hex"):
        signing_keys.pinned_public_key_from_env(
            environ={signing_keys.ADJUDICATION_PUBLIC_KEY_ENV: "xyz"}
        )
