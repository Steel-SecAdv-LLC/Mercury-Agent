# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for scripts/check_pypi_publishable.py (release PyPI preflight).

Each test builds a minimal wheel or sdist in ``tmp_path`` with exactly the
metadata under test, so no real build and no network is needed; the PyPI JSON
API is injected.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_MOD = REPO_ROOT / "scripts" / "check_pypi_publishable.py"
_spec = importlib.util.spec_from_file_location("check_pypi_publishable", _MOD)
assert _spec is not None and _spec.loader is not None
preflight = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = preflight
_spec.loader.exec_module(preflight)

_REPO = "Steel-SecAdv-LLC/Mercury-Agent"


def _metadata(name: str, version: str, requires: list[str]) -> str:
    lines = ["Metadata-Version: 2.4", f"Name: {name}", f"Version: {version}"]
    lines += [f"Requires-Dist: {req}" for req in requires]
    return "\n".join(lines) + "\n\n"


def _wheel(
    directory: Path,
    name: str = "demo-pkg",
    version: str = "1.0.0",
    requires: list[str] | None = None,
) -> Path:
    stem = f"{name.replace('-', '_')}-{version}"
    path = directory / f"{stem}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{stem}.dist-info/METADATA", _metadata(name, version, requires or []))
    return path


def _sdist(
    directory: Path,
    name: str = "demo-pkg",
    version: str = "1.0.0",
    requires: list[str] | None = None,
) -> Path:
    stem = f"{name.replace('-', '_')}-{version}"
    path = directory / f"{stem}.tar.gz"
    data = _metadata(name, version, requires or []).encode()
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo(f"{stem}/PKG-INFO")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    return path


def _fetch(status: int, payload: Any = None) -> Any:
    def fetch(url: str, timeout: float) -> tuple[int, Any]:
        return status, payload

    return fetch


def test_clean_wheel_and_sdist_are_publishable_offline(tmp_path: Path) -> None:
    _wheel(tmp_path, requires=["numpy>=2.4", 'torch>=2.2; extra == "ml"'])
    _sdist(tmp_path, requires=["numpy>=2.4"])
    report = preflight.Report()
    dists = preflight.check_distributions(tmp_path, report)
    assert len(dists) == 2
    assert report.publishable, report.errors
    assert (report.name, report.version) == ("demo-pkg", "1.0.0")


@pytest.mark.parametrize(
    "requirement",
    [
        'ama-cryptography @ git+https://example.invalid/ama.git@v5.0.0 ; extra == "pqc"',
        "pkg[extra] @ https://example.invalid/pkg-1.0.tar.gz",
    ],
)
def test_direct_url_dependency_is_rejected(tmp_path: Path, requirement: str) -> None:
    _wheel(tmp_path, requires=["numpy>=2.4", requirement])
    report = preflight.Report()
    preflight.check_distributions(tmp_path, report)
    assert not report.publishable
    assert any("direct URL" in e for e in report.errors)


def test_this_projects_pqc_extra_is_a_direct_reference() -> None:
    """The real pyproject still pins AMA by git URL, which PyPI refuses."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "ama-cryptography @ git+" in text
    assert preflight._DIRECT_REFERENCE.match("ama-cryptography @ git+https://x/y.git@v1")


def test_non_distribution_files_warn_and_staging_excludes_them(tmp_path: Path) -> None:
    src = tmp_path / "dist"
    src.mkdir()
    wheel = _wheel(src)
    (src / "SBOM.spdx.json").write_text("{}", encoding="utf-8")
    (src / "provenance.json").write_text("{}", encoding="utf-8")
    report = preflight.Report()
    dists = preflight.check_distributions(src, report)
    assert report.publishable
    assert any("SBOM.spdx.json" in w for w in report.warnings)
    staged = tmp_path / "pypi-dist"
    preflight.stage(dists, staged)
    assert sorted(p.name for p in staged.iterdir()) == [wheel.name]


def test_empty_directory_is_not_publishable(tmp_path: Path) -> None:
    report = preflight.Report()
    assert preflight.check_distributions(tmp_path, report) == []
    assert not report.publishable


def test_mismatched_names_are_rejected(tmp_path: Path) -> None:
    _wheel(tmp_path, name="demo-pkg")
    _sdist(tmp_path, name="other-pkg")
    report = preflight.Report()
    preflight.check_distributions(tmp_path, report)
    assert any("disagree" in e for e in report.errors)


def test_free_name_passes_index_check() -> None:
    report = preflight.Report(name="demo-pkg", version="1.0.0")
    preflight.check_index(report, repository=_REPO, fetch=_fetch(404))
    assert report.index_checked and report.publishable


def test_name_owned_by_another_project_is_rejected() -> None:
    payload = {
        "info": {
            "name": "demo-pkg",
            "version": "0.1.10",
            "author": "someone else",
            "home_page": "http://unrelated.example",
            "project_urls": {"Homepage": "http://unrelated.example"},
        },
        "releases": {"0.1.10": []},
    }
    report = preflight.Report(name="demo-pkg", version="2.1.0")
    preflight.check_index(report, repository=_REPO, fetch=_fetch(200, payload))
    assert not report.publishable
    assert "belongs to another project" in report.errors[0]


def test_own_project_passes_unless_version_exists() -> None:
    payload = {
        "info": {"name": "demo-pkg", "project_urls": {"Repository": f"https://github.com/{_REPO}"}},
        "releases": {"1.0.0": []},
    }
    fresh = preflight.Report(name="demo-pkg", version="1.1.0")
    preflight.check_index(fresh, repository=_REPO, fetch=_fetch(200, payload))
    assert fresh.publishable
    dup = preflight.Report(name="demo-pkg", version="1.0.0")
    preflight.check_index(dup, repository=_REPO, fetch=_fetch(200, payload))
    assert any("already on PyPI" in e for e in dup.errors)


def test_unreachable_index_fails_closed() -> None:
    def fetch(url: str, timeout: float) -> tuple[int, Any]:
        raise OSError("network down")

    report = preflight.Report(name="demo-pkg", version="1.0.0")
    preflight.check_index(report, repository=_REPO, fetch=fetch)
    assert not report.publishable and not report.index_checked


def test_main_offline_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = tmp_path / "good"
    good.mkdir()
    _wheel(good)
    assert preflight.main([str(good), "--offline"]) == 0
    bad = tmp_path / "bad"
    bad.mkdir()
    _wheel(bad, requires=["x @ https://example.invalid/x.whl"])
    assert preflight.main([str(bad), "--offline", "--github"]) == 1
    assert "::error title=PyPI preflight::" in capsys.readouterr().out
