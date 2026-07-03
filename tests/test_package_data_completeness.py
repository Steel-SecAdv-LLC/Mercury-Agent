# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every non-Python runtime data file under ``src/omni_mercury_engine`` must be declared in ``pyproject.toml``'s ``[tool.setuptools.package-data]``.

CI installs the package with ``pip install -e .`` (editable), which resolves
``Path(__file__).parent``-relative loads against the source checkout
regardless of what is declared here -- so a missing declaration is invisible
to every CI job. A real (non-editable) install -- ``pip install ".[all]"``
with no ``-e``, which is exactly what the Dockerfile and any real
``pip install mercury-agent`` do -- silently drops undeclared files from the
installed package. For files the σ_Immutable ethical gate reads (the corpus,
its signature bundle, and its trained weights), that means every detection
call then fails closed with ``EthicalConstraintViolationError`` the moment
the package is installed for real, while the editable-install CI gate stays
green throughout. This test pins the invariant a static code review cannot
catch: it has to actually compare the declared globs against the files
really present on disk.
"""

from __future__ import annotations

import tomllib
from fnmatch import fnmatch
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE_ROOT = _REPO_ROOT / "src" / "omni_mercury_engine"

# Files that exist on disk but are never read by installed runtime code
# (operator/training scripts, not the installed package's own import graph)
# and so are intentionally excluded from package-data.
_NOT_RUNTIME_DATA = {
    "security/sigma_immutable_registry.json",  # written by scripts/train_sigma_immutable.py only
}


def _declared_package_data_globs() -> list[str]:
    with open(_REPO_ROOT / "pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)
    package_data = pyproject["tool"]["setuptools"]["package-data"]
    return list(package_data["omni_mercury_engine"])


def _non_python_data_files() -> list[str]:
    """Relative (POSIX) paths of every non-.py/.pyi file under the package."""
    files = []
    for path in _PACKAGE_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        if path.suffix in (".py", ".pyi", ".pyc"):
            continue
        rel = path.relative_to(_PACKAGE_ROOT).as_posix()
        if rel in _NOT_RUNTIME_DATA:
            continue
        files.append(rel)
    return files


class TestPackageDataCompleteness:
    def test_every_runtime_data_file_is_declared(self) -> None:
        globs = _declared_package_data_globs()
        data_files = _non_python_data_files()
        assert data_files, "sanity check: expected at least one non-.py data file on disk"

        uncovered = [f for f in data_files if not any(fnmatch(f, g) for g in globs)]
        assert not uncovered, (
            "These runtime data files exist under src/omni_mercury_engine/ but are "
            "not covered by [tool.setuptools.package-data] in pyproject.toml, so a "
            "non-editable install silently ships without them: "
            f"{sorted(uncovered)}. Add a matching glob."
        )

    def test_sigma_immutable_artifacts_are_declared(self) -> None:
        """Pin the three files whose absence makes every detection call
        raise EthicalConstraintViolationError under a real install."""
        globs = _declared_package_data_globs()
        required = (
            "security/sigma_immutable_corpus.json",
            "security/sigma_immutable_corpus.sig.json",
            "security/sigma_immutable_weights.pt",
        )
        for rel in required:
            assert any(
                fnmatch(rel, g) for g in globs
            ), f"{rel} is not covered by any package-data glob {globs}"
            assert (_PACKAGE_ROOT / rel).is_file(), f"{rel} is missing from the source tree"
