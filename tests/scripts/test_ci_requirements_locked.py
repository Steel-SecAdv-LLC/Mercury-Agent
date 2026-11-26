# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The hash-locked CI requirement files agree with what they stand beside.

``requirements/ci/*.txt`` (compiled by ``scripts/lock_ci_requirements.sh``)
are what the workflows, the Dockerfile and the AMA build install with
``--require-hashes``. Where one of them re-installs into an environment
``uv.lock`` already resolved, its pins must be the lock's, or the
re-install would move a locked package; the pip floor must hold the
CVE-2026-6357 line; and the AMA wheel requirement must hash exactly the
wheels it is given.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_REPO = Path(__file__).resolve().parents[2]
_REQ = _REPO / "requirements" / "ci"


def _pins(path: Path) -> dict[str, str]:
    return {
        name.lower(): version
        for name, version in re.findall(
            r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)", path.read_text(), re.MULTILINE
        )
    }


def _locked() -> dict[str, set[str]]:
    lock = tomllib.loads((_REPO / "uv.lock").read_text(encoding="utf-8"))
    versions: dict[str, set[str]] = {}
    for package in lock["package"]:
        versions.setdefault(package["name"].lower(), set()).add(package["version"])
    return versions


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _REPO / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_image_floors_are_exactly_the_locked_versions() -> None:
    """The Dockerfile re-applies these after ``uv sync --locked``: equal or it moves them."""
    locked = _locked()
    for name, version in _pins(_REQ / "image-floors.txt").items():
        assert name in locked and locked[name] == {version}, (name, version, locked.get(name))


def test_the_pip_floor_is_hashed_and_at_or_above_26_1() -> None:
    pins = _pins(_REQ / "pip.txt")
    major, minor = (int(x) for x in pins["pip"].split(".")[:2])
    assert (major, minor) >= (26, 1)
    assert "--hash=sha256:" in (_REQ / "pip.txt").read_text()


def test_the_cpu_torch_file_cannot_resolve_a_cuda_build() -> None:
    text = (_REQ / "torch-cpu.txt").read_text()
    pins = _pins(_REQ / "torch-cpu.txt")
    assert pins["torch"].endswith("+cpu") and pins["torchvision"].endswith("+cpu")
    assert not re.search(r"^(nvidia-|triton)", text, re.MULTILINE)


def test_every_compiled_file_names_its_regeneration_command() -> None:
    for path in sorted(_REQ.glob("*.txt")):
        assert "bash scripts/lock_ci_requirements.sh" in path.read_text(), path.name


def test_the_dockerfile_copies_what_its_installs_read() -> None:
    dockerfile = (_REPO / "Dockerfile").read_text(encoding="utf-8")
    copy_at = dockerfile.index("COPY requirements/ci/ /app/requirements/ci/")
    first_install = dockerfile.index("RUN pip install --require-hashes")
    assert copy_at < first_install
    for helper in ("install_ama_wheel.sh", "ama_wheel_requirement.py"):
        assert f"scripts/{helper}" in dockerfile, helper
    dockerignore = (_REPO / ".dockerignore").read_text(encoding="utf-8")
    assert not re.search(r"^/?requirements/?$", dockerignore, re.MULTILINE)


class TestAmaWheelRequirement:
    @pytest.fixture(scope="class")
    def module(self) -> ModuleType:
        return _load("ama_wheel_requirement")

    def test_hashes_every_wheel_of_the_version(self, module: ModuleType, tmp_path: Path) -> None:
        a = tmp_path / "ama_cryptography-5.0.0-cp312-cp312-linux_x86_64.whl"
        b = tmp_path / "ama_cryptography-5.0.0-cp313-cp313-linux_x86_64.whl"
        other = tmp_path / "ama_cryptography-4.9.0-cp312-cp312-linux_x86_64.whl"
        for path, body in ((a, b"a"), (b, b"b"), (other, b"c")):
            path.write_bytes(body)
        line = module.requirement("5.0.0", tmp_path)
        expected = " ".join(
            f"--hash=sha256:{hashlib.sha256(body).hexdigest()}" for body in (b"a", b"b")
        )
        assert line == f"ama-cryptography==5.0.0 {expected}\n"

    def test_no_wheel_is_an_error_not_an_empty_file(
        self, module: ModuleType, tmp_path: Path
    ) -> None:
        out = tmp_path / "req.txt"
        assert module.main(["5.0.0", str(tmp_path), str(out)]) == 1
        assert not out.exists()
        assert module.main(["5.0.0"]) == 2

    def test_writes_the_file(self, module: ModuleType, tmp_path: Path) -> None:
        (tmp_path / "ama_cryptography-5.0.0-py3-none-any.whl").write_bytes(b"w")
        out = tmp_path / "req.txt"
        assert module.main(["5.0.0", str(tmp_path), str(out)]) == 0
        assert out.read_text().startswith("ama-cryptography==5.0.0 --hash=sha256:")


def test_the_hardening_gate_reads_the_pip_file_floor(tmp_path: Path) -> None:
    """The CVE guard counts the file form only while the file pins pip >= 26.1."""
    hardening = _load("check_workflow_hardening")
    line = 'python -m pip install --require-hashes -r "${GITHUB_WORKSPACE}/requirements/ci/pip.txt"'
    assert hardening.is_pip_floor(line)
    low = tmp_path / "pip.txt"
    low.write_text("pip==26.0 \\\n    --hash=sha256:" + "c" * 64 + "\n")
    assert not hardening.pip_floor_file_pins_floor(low)
    low.write_text("pip==26.3 \\\n    --hash=sha256:" + "c" * 64 + "\n")
    assert hardening.pip_floor_file_pins_floor(low)
    assert not hardening.is_pip_floor("python -m pip install -r requirements/ci/pip.txt")


def test_a_single_platform_file_is_compiled_for_its_lanes_interpreter() -> None:
    """For one platform uv hashes only the wheels one interpreter installs.

    torch-cpu.txt was first compiled for Python 3.11 while the verifiers lane
    runs 3.12: its ``markupsafe`` carried the cp311 hash only, and the real
    install failed with a hash mismatch on the cp312 wheel. The spec now names
    the lane's version; this holds the two equal.
    """
    spec = (_REQ / "torch-cpu.in").read_text(encoding="utf-8")
    args = re.search(r"^# uv-compile-args: (.*)$", spec, re.MULTILINE)
    assert args is not None and "--python-platform" in args.group(1)
    compiled_for = re.search(r"--python-version (\S+)", args.group(1))
    assert compiled_for is not None, "a --python-platform spec must name --python-version"
    lane = (_REPO / ".github" / "workflows" / "verifiers.yml").read_text(encoding="utf-8")
    lane_version = re.search(r'python-version: "([\d.]+)"', lane)
    assert lane_version is not None
    assert compiled_for.group(1) == lane_version.group(1)


#: Published advisories against CI tool pins and the first fixed release.
#: Each entry was a Dependabot alert on ``main`` after #415 (2026-11-23); the
#: pinned version must stay at or above the fix. ``set`` is the lock file stem.
_ADVISORY_FLOORS: dict[str, tuple[str, str, str]] = {
    "uv": ("uv", "0.13.4", "GHSA-hq85-5cpj-mr62"),
    "mcp": ("semgrep", "1.30.0", "GHSA-84m7-p3x7-pcfv and four more"),
    "opentelemetry-api": ("semgrep", "1.45.2", "GHSA-xcq7-p3pf-p88h, GHSA-mf5v-w3w3-f6cg"),
    "wcmatch": ("semgrep", "11.2", "GHSA-53pm-wgxf-c6qf"),
    "filelock": ("audit", "4.2.1", "GHSA-v4x8-3wr5-3wq8 and twelve more"),
}


@pytest.mark.parametrize("package", sorted(_ADVISORY_FLOORS))
def test_ci_tool_pins_stay_at_or_above_their_advisory_fix(package: str) -> None:
    from packaging.version import Version

    stem, floor, advisory = _ADVISORY_FLOORS[package]
    pinned = _pins(_REQ / f"{stem}.txt").get(package)
    assert pinned is not None, f"{package} missing from {stem}.txt"
    assert Version(pinned) >= Version(
        floor
    ), f"{stem}.txt pins {package}=={pinned}, below {floor} ({advisory})"


def _override_specs(path: Path) -> list[str]:
    lines = (line.split("#", 1)[0].strip() for line in path.read_text().splitlines())
    return [line for line in lines if line]


def test_every_override_file_is_compiled_in_and_its_specifiers_hold() -> None:
    """An override only protects a lock compiled with it, at a version it admits."""
    from packaging.requirements import Requirement

    overrides = sorted(_REQ.glob("*.overrides"))
    assert {p.stem for p in overrides} >= {"semgrep", "audit"}
    for path in overrides:
        spec = (_REQ / f"{path.stem}.in").read_text()
        args = re.search(r"^# uv-compile-args: (.*)$", spec, re.MULTILINE)
        assert args and f"--override requirements/ci/{path.name}" in args.group(1), path.name
        pins = _pins(_REQ / f"{path.stem}.txt")
        for line in _override_specs(path):
            req = Requirement(line)
            name = req.name.lower()
            assert name in pins, (path.name, name)
            assert req.specifier.contains(pins[name], prereleases=True), (
                path.name,
                line,
                pins[name],
            )


def test_overridden_sets_install_without_pips_resolver() -> None:
    """pip's resolver rejects an overridden pin; the closure is already complete and hashed."""
    surfaces = [*(_REPO / ".github" / "workflows").glob("*.yml"), _REPO / "Dockerfile"]
    surfaces += [*(_REPO / "scripts").glob("*.sh")]
    stems = {p.stem for p in _REQ.glob("*.overrides")}
    seen = 0
    for surface in surfaces:
        for line in surface.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith("#"):
                continue
            for stem in stems:
                if f"requirements/ci/{stem}.txt" in line and "install" in line:
                    seen += 1
                    assert "--no-deps" in line, (surface.name, line.strip())
    assert seen >= 3  # ci.yml audit + semgrep, security.yml audit


def test_the_security_lane_audits_every_ci_lock() -> None:
    """A vulnerable tool pin fails the pull request, not a Dependabot alert after merge."""
    workflow = (_REPO / ".github" / "workflows" / "security.yml").read_text(encoding="utf-8")
    step = re.search(r"for lock in requirements/ci/\*\.txt; do\n(.*?)\n\s*done", workflow, re.S)
    assert step is not None
    assert 'pip-audit -r "${lock}"' in step.group(1)
    assert "--ignore-vuln" not in step.group(1)
