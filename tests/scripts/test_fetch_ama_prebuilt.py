# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The prebuilt AMA native backend: download, verify, unpack, or step aside.

``scripts/fetch_ama_prebuilt.sh`` is the first thing every AMA install path
runs (``scripts/build_ama_native.sh`` for the Dockerfile and local installs,
the ``build-ama-cryptography`` composite for CI). Its contract is the exit
status: 0 with a verified archive unpacked, 2 when no archive applies so the
caller builds from source, 1 when a downloaded archive fails its checksum so
the caller stops. These tests drive it against a ``file://`` release mirror.
"""

from __future__ import annotations

import hashlib
import io
import os
import platform
import re
import subprocess
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FETCH = REPO / "scripts" / "fetch_ama_prebuilt.sh"
BUILD = REPO / "scripts" / "build_ama_native.sh"
ACTION = REPO / ".github" / "actions" / "build-ama-cryptography" / "action.yml"
WORKFLOW = REPO / ".github" / "workflows" / "ama-prebuilt.yml"
REF = "v9.9.9-test"


def _platform_suffix() -> str:
    os_name = {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}[platform.system()]
    arch = {"x86_64": "x86_64", "AMD64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}[
        platform.machine()
    ]
    return f"{os_name}-{arch}"


def _archive_bytes(*, with_wheel: bool = True, extra: bytes = b"") -> bytes:
    """A release archive: one wheel per Python, each bundling the library."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        members = []
        if with_wheel:
            for tag in ("cp311", "cp312", "cp313", "cp314"):
                members.append(
                    (
                        f"ama_cryptography-9.9.9-{tag}-{tag}-linux_x86_64.whl",
                        b"PK\x05\x06" + b"\0" * 18,
                    )
                )
        if extra:
            members.append(("README", extra))
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _release(tmp_path: Path, archive: bytes | None, *, sums_for: bytes | None = None) -> Path:
    """Lay out a release directory: the archive (if any) and SHA256SUMS."""
    release = tmp_path / "release"
    release.mkdir()
    name = f"ama-cryptography-{REF}-{_platform_suffix()}.tar.gz"
    if archive is not None:
        (release / name).write_bytes(archive)
    digest_source = sums_for if sums_for is not None else archive
    if digest_source is not None:
        digest = hashlib.sha256(digest_source).hexdigest()
        (release / "SHA256SUMS").write_text(f"{digest}  {name}\n", encoding="utf-8")
    return release


def _fetch(tmp_path: Path, release: Path | None, **env: str) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "AMA_REF": REF,
        "AMA_PREBUILT_BASE_URL": release.as_uri() if release else "file:///nonexistent",
        **env,
    }
    return subprocess.run(
        ["bash", str(FETCH), str(tmp_path / "dest")],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.skipif(platform.system() == "Windows", reason="the script runs under bash")
class TestFetch:
    def test_verified_archive_is_unpacked_and_its_path_printed(self, tmp_path: Path) -> None:
        release = _release(tmp_path, _archive_bytes())
        result = _fetch(tmp_path, release)
        assert result.returncode == 0, result.stderr
        dest = Path(result.stdout.strip())
        assert dest == tmp_path / "dest"
        assert len(list(dest.glob("ama_cryptography-9.9.9-cp31*.whl"))) == 4
        assert "verified (sha256" in result.stderr
        assert "4 wheel(s)" in result.stderr

    def test_checksum_mismatch_stops_the_caller(self, tmp_path: Path) -> None:
        good = _archive_bytes()
        tampered = _archive_bytes(extra=b"one more member")
        release = _release(tmp_path, tampered, sums_for=good)
        result = _fetch(tmp_path, release)
        assert result.returncode == 1
        assert "does not match SHA256SUMS" in result.stderr
        assert not (tmp_path / "dest").exists()

    def test_release_without_an_archive_for_this_platform_defers_to_source(
        self, tmp_path: Path
    ) -> None:
        release = _release(tmp_path, None, sums_for=b"other")
        (release / "SHA256SUMS").write_text(
            "0" * 64 + f"  ama-cryptography-{REF}-plan9-mips.tar.gz\n", encoding="utf-8"
        )
        result = _fetch(tmp_path, release)
        assert result.returncode == 2
        assert "release has no ama-cryptography" in result.stderr

    def test_missing_sums_defers_to_source(self, tmp_path: Path) -> None:
        release = tmp_path / "release"
        release.mkdir()
        result = _fetch(tmp_path, release)
        assert result.returncode == 2
        assert "no SHA256SUMS" in result.stderr

    def test_disabled_by_environment_defers_without_touching_the_network(
        self, tmp_path: Path
    ) -> None:
        result = _fetch(tmp_path, None, AMA_PREBUILT="0")
        assert result.returncode == 2
        assert "AMA_PREBUILT=0" in result.stderr

    def test_archive_without_a_wheel_is_rejected(self, tmp_path: Path) -> None:
        no_wheel = _release(tmp_path, _archive_bytes(with_wheel=False, extra=b"nothing useful"))
        result = _fetch(tmp_path, no_wheel)
        assert result.returncode == 1
        assert "carries no ama_cryptography wheel" in result.stderr


class TestCallersAgreeOnTheContract:
    """The archive name, exit codes and ordering are shared by every path."""

    def test_build_script_tries_the_prebuilt_before_cloning(self) -> None:
        text = BUILD.read_text(encoding="utf-8")
        fetch_at = text.index("fetch_ama_prebuilt.sh")
        clone_at = text.index("git clone")
        assert fetch_at < clone_at
        # Only "no archive" (2) falls through; a failed checksum (1) exits.
        assert re.search(r'\[ "\$\{rc\}" -eq 2 \] \|\| exit "\$\{rc\}"', text)

    def test_composite_action_skips_every_build_step_on_a_prebuilt(self) -> None:
        text = ACTION.read_text(encoding="utf-8")
        assert text.index("fetch_ama_prebuilt.sh") < text.index("Resolve AMA C/C++ compiler")
        guarded = re.findall(r"if: steps\.prebuilt\.outputs\.found != 'true'", text)
        # compiler, cache key, cache restore, toolchain, floors, build, install
        assert len(guarded) == 7, len(guarded)

    def test_workflow_archive_name_matches_the_fetch_script(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        fetch = FETCH.read_text(encoding="utf-8")
        assert 'archive="ama-cryptography-${AMA_REF}-${os}-${arch}.tar.gz"' in fetch
        assert "ama-cryptography-${AMA_REF}-${PLATFORM}.tar.gz" in workflow
        platforms = set(re.findall(r"platform: ([a-z]+-[a-z0-9_]+)", workflow))
        assert platforms == {
            "linux-x86_64",
            "linux-aarch64",
            "macos-aarch64",
            "macos-x86_64",
            "windows-x86_64",
        }
        # One wheel per supported Python, each tagged for its interpreter.
        assert set(re.findall(r'python-version: "(3\.1\d)"', workflow)) == {
            "3.11",
            "3.12",
            "3.13",
            "3.14",
        }
        # The default download location is the release this workflow publishes.
        assert "tag_name: ama-native-${{ inputs.ama-ref }}" in workflow
        assert "releases/download/ama-native-${AMA_REF}" in fetch
        # Every installer picks the wheel by tag rather than by file name.
        for text in (
            BUILD.read_text(encoding="utf-8"),
            ACTION.read_text(encoding="utf-8"),
            workflow,
        ):
            assert "--no-index --find-links" in text

    def test_dockerfile_ships_the_fetch_script_and_curl(self) -> None:
        dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")
        assert "scripts/fetch_ama_prebuilt.sh" in dockerfile
        assert re.search(r"^\s+curl \\\s*$", dockerfile, re.M)

    def test_scripts_are_bash_clean(self) -> None:
        for script in (FETCH, BUILD):
            subprocess.run(["bash", "-n", str(script)], check=True, timeout=30)
