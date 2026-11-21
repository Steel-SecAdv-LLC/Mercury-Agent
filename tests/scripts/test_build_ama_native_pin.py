# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``scripts/build_ama_native.sh``: the ref is a tag, the commit is pinned beside it.

The Dockerfile once passed a commit id as ``AMA_REF``, to pin AMA immutably.
The script's whole chain is keyed on the *tag* -- the prebuilt archive name,
the wheel version (``${AMA_REF#v}``) and ``git clone --branch``, which takes a
branch or tag only -- so every image build failed inside git with
``Remote branch <sha> not found``.  Nothing in the repository's own gates
built the image, so it was found by CI after the push.

The pin is now two values: ``AMA_REF`` stays the tag, and ``AMA_COMMIT`` is
the commit that tag must resolve to.  These tests drive the real script
against a local fixture repository, with a stand-in ``python`` on ``PATH``
that refuses every ``pip install``, so nothing is fetched from the network
and nothing can be installed into the interpreter running the tests.

``test_the_dockerfile_passes_a_tag_and_a_commit`` fails on the prior
Dockerfile; the rest fail on the prior script.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BUILD = REPO / "scripts" / "build_ama_native.sh"
DOCKERFILE = REPO / "Dockerfile"
LOCK = REPO / "uv.lock"
TAG = "v9.9.9"
#: The exit status the stand-in ``python`` gives a refused ``pip install``.
INSTALL_REFUSED = 97

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="the script drives a POSIX toolchain"),
    pytest.mark.skipif(
        shutil.which("bash") is None or shutil.which("git") is None,
        reason="the script needs bash and git",
    ),
]


def _git(cwd: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
    }
    done = subprocess.run(
        ["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True, timeout=60
    )
    return done.stdout.strip()


@pytest.fixture
def upstream(tmp_path: Path) -> tuple[Path, str]:
    """A repository with one tagged commit, standing in for AMA's."""
    repo = tmp_path / "upstream"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "README").write_text("fixture\n")
    _git(repo, "add", "README")
    _git(repo, "commit", "-q", "-m", "fixture")
    _git(repo, "tag", TAG)
    return repo, _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def stub_python(tmp_path: Path) -> tuple[Path, Path]:
    """A ``python`` that reports a pip, refuses every install, and logs both."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "python-calls.log"
    stub = bin_dir / "python"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$*" >> "{log}"\n'
        'if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then\n'
        '  if [ "$3" = "--version" ]; then echo "pip 0 (stand-in)"; exit 0; fi\n'
        f"  exit {INSTALL_REFUSED}\n"
        "fi\n"
        f'exec "{sys.executable}" "$@"\n'
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    return bin_dir, log


def _run(
    stub: tuple[Path, Path], tmp_path: Path, **overrides: str
) -> subprocess.CompletedProcess[str]:
    bin_dir, _ = stub
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "AMA_PREBUILT": "0",  # no download: straight to the source path
        "AMA_BUILD_DIR": str(tmp_path / "checkout"),
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        **overrides,
    }
    return subprocess.run(
        ["bash", str(BUILD)], env=env, capture_output=True, text=True, timeout=120, check=False
    )


def _installs(stub: tuple[Path, Path]) -> list[str]:
    _, log = stub
    lines = log.read_text().splitlines() if log.exists() else []
    return [line for line in lines if " install" in f" {line}"]


class TestTheRefIsATag:
    def test_a_commit_id_as_the_ref_is_refused_with_the_remedy(
        self, stub_python: tuple[Path, Path], tmp_path: Path
    ) -> None:
        commit = "702e253a137f98ffe452d2f6e614fcc175112598"
        done = _run(stub_python, tmp_path, AMA_REF=commit)
        assert done.returncode == 1
        assert "AMA_REF takes a tag" in done.stderr
        assert f"AMA_COMMIT={commit}" in done.stderr  # what to do instead
        assert not (tmp_path / "checkout").exists()  # refused before any clone
        assert _installs(stub_python) == []

    @pytest.mark.parametrize(
        "bad", ["702e253", "Z" * 40, "702E253A137F98FFE452D2F6E614FCC175112598"]
    )
    def test_a_malformed_commit_pin_is_refused(
        self, stub_python: tuple[Path, Path], tmp_path: Path, bad: str
    ) -> None:
        done = _run(stub_python, tmp_path, AMA_REF=TAG, AMA_COMMIT=bad)
        assert done.returncode == 1
        assert "AMA_COMMIT must be a full 40-character" in done.stderr
        assert _installs(stub_python) == []


@pytest.mark.skipif(
    not any(shutil.which(cc) for cc in ("cc", "gcc", "clang")),
    reason="the script's source path preflight needs a C compiler",
)
class TestTheCommitIsPinned:
    def test_a_tag_that_resolves_elsewhere_is_refused_before_anything_is_installed(
        self, upstream: tuple[Path, str], stub_python: tuple[Path, Path], tmp_path: Path
    ) -> None:
        repo, head = upstream
        expected = "0" * 40
        done = _run(stub_python, tmp_path, AMA_REF=TAG, AMA_REPO=str(repo), AMA_COMMIT=expected)
        assert done.returncode == 1
        assert f"resolved to {head}, expected {expected}" in done.stderr
        assert not (tmp_path / "checkout").exists()  # the refused checkout is removed
        assert _installs(stub_python) == []  # the interpreter was never touched

    def test_the_pinned_commit_is_accepted_and_the_build_goes_on(
        self, upstream: tuple[Path, str], stub_python: tuple[Path, Path], tmp_path: Path
    ) -> None:
        repo, head = upstream
        done = _run(stub_python, tmp_path, AMA_REF=TAG, AMA_REPO=str(repo), AMA_COMMIT=head)
        assert f"is the pinned commit {head}" in done.stdout
        # It went on to install the build tools, which the stand-in refuses:
        # that refusal, not the pin, is what stopped the script.
        assert done.returncode == INSTALL_REFUSED
        assert len(_installs(stub_python)) == 1

    def test_without_a_pin_the_tag_is_taken_as_it_resolves(
        self, upstream: tuple[Path, str], stub_python: tuple[Path, Path], tmp_path: Path
    ) -> None:
        repo, _ = upstream
        done = _run(stub_python, tmp_path, AMA_REF=TAG, AMA_REPO=str(repo))
        assert "pinned commit" not in done.stdout
        assert done.returncode == INSTALL_REFUSED  # reached the build, unpinned as before


class TestTheDockerfile:
    def test_the_dockerfile_passes_a_tag_and_a_commit(self) -> None:
        """Fails on the prior Dockerfile, whose AMA_REF was the commit id."""
        text = DOCKERFILE.read_text()
        ref = re.search(r"^ARG AMA_REF=(\S+)", text, re.MULTILINE)
        commit = re.search(r"^ARG AMA_COMMIT=(\S+)", text, re.MULTILINE)
        assert ref is not None and commit is not None
        assert re.fullmatch(r"v\d+\.\d+\.\d+", ref.group(1)), ref.group(1)
        assert re.fullmatch(r"[0-9a-f]{40}", commit.group(1)), commit.group(1)
        # Both reach the script, on the line that runs it.
        run_line = next(
            line for line in text.splitlines() if "build_ama_native.sh" in line and "RUN" in line
        )
        assert 'AMA_REF="${AMA_REF}"' in run_line
        assert 'AMA_COMMIT="${AMA_COMMIT}"' in run_line

    def test_the_dockerfile_tag_is_the_scripts_default(self) -> None:
        """One tag across the image, the script and the session setup."""
        ref = re.search(r"^ARG AMA_REF=(\S+)", DOCKERFILE.read_text(), re.MULTILINE)
        default = re.search(r'^AMA_REF="\$\{AMA_REF:-(\S+)\}"', BUILD.read_text(), re.MULTILINE)
        assert ref is not None and default is not None
        assert ref.group(1) == default.group(1)

    def test_the_dockerfile_commit_is_the_one_the_lockfile_resolves(self) -> None:
        """The lockfile records which commit the tag resolved to; the image pins the same."""
        text = DOCKERFILE.read_text()
        ref = re.search(r"^ARG AMA_REF=(\S+)", text, re.MULTILINE)
        commit = re.search(r"^ARG AMA_COMMIT=(\S+)", text, re.MULTILINE)
        assert ref is not None and commit is not None
        locked = re.search(
            r"AMA-Cryptography\.git\?rev=" + re.escape(ref.group(1)) + r"#([0-9a-f]{40})",
            LOCK.read_text(),
        )
        assert locked is not None, f"uv.lock does not resolve AMA at {ref.group(1)}"
        assert commit.group(1) == locked.group(1)
