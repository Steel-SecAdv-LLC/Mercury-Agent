# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The AMA compiler resolver picks the right path on every image shape.

Each case runs the real script against fake compilers on a PATH that holds
nothing else, so the outcome depends only on the fakes that are present.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "resolve_ama_compiler.sh"
_BASH = shutil.which("bash")
_CUT = shutil.which("cut")

pytestmark = pytest.mark.skipif(
    _BASH is None or _CUT is None or os.name != "posix",
    reason="needs POSIX bash and cut",
)


def _fake_compiler(bin_dir: Path, name: str, version: str) -> None:
    path = bin_dir / name
    path.write_text(
        "#!/bin/sh\n" f'[ "$1" = -dumpfullversion ] && echo {version} || exit 2\n',
        encoding="utf-8",
    )
    path.chmod(0o755)


def _resolve(tmp_path: Path, compilers: dict[str, str]) -> dict[str, str]:
    assert _BASH is not None and _CUT is not None  # guarded by the skipif marker
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "cut").symlink_to(_CUT)
    for name, version in compilers.items():
        _fake_compiler(bin_dir, name, version)
    env = {"PATH": str(bin_dir)}
    proc = subprocess.run(
        [_BASH, str(_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    fields: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        key, _, value = line.partition("=")
        fields[key] = value
    return fields


def test_image_default_gcc_meeting_the_floor_is_used(tmp_path: Path) -> None:
    got = _resolve(tmp_path, {"gcc": "13.3.0", "g++": "13.3.0"})
    assert got == {
        "cc": "gcc",
        "cxx": "g++",
        "version": "13.3.0",
        "source": "image-default",
        "need-apt": "false",
    }


def test_default_below_the_floor_falls_back_to_image_gcc_12(tmp_path: Path) -> None:
    got = _resolve(
        tmp_path,
        {"gcc": "11.4.0", "g++": "11.4.0", "gcc-12": "12.5.0", "g++-12": "12.5.0"},
    )
    assert got["cc"] == "gcc-12"
    assert got["cxx"] == "g++-12"
    assert got["source"] == "image-gcc-12"
    assert got["need-apt"] == "false"


def test_nothing_usable_requests_the_apt_install(tmp_path: Path) -> None:
    got = _resolve(tmp_path, {})
    assert got["cc"] == "/usr/bin/gcc-12"
    assert got["cxx"] == "/usr/bin/g++-12"
    assert got["version"] == "12"
    assert got["source"] == "apt-gcc-12"
    assert got["need-apt"] == "true"


def test_the_c_and_c_plus_plus_compilers_must_both_meet_the_floor(tmp_path: Path) -> None:
    """A new C compiler with an old C++ one is not a usable toolchain."""
    got = _resolve(tmp_path, {"gcc": "13.3.0", "g++": "10.5.0"})
    assert got["source"] == "apt-gcc-12"


def test_a_missing_cxx_compiler_is_not_a_toolchain(tmp_path: Path) -> None:
    got = _resolve(tmp_path, {"gcc": "13.3.0"})
    assert got["source"] == "apt-gcc-12"


def test_the_floor_is_inclusive(tmp_path: Path) -> None:
    got = _resolve(tmp_path, {"gcc": "12.0.0", "g++": "12.0.0"})
    assert got["source"] == "image-default"
