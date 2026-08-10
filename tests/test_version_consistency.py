# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""``[tool.semantic_release].version_variable`` must name rewritable literals.

semantic-release bumps a version by finding ``NAME = "<old>"`` and rewriting the
quoted value. A target it cannot pattern-match is skipped in silence -- no
error, no warning, just a version that stops moving.

That is what the previous configuration had. It listed
``src/omni_mercury_engine/__init__.py:__version__``, but that line is
``__version__ = _get_version()``: a call, not a literal, so no release could
ever have rewritten it. The hand-maintained literal that a release genuinely
must move is ``_version.py:_FALLBACK_VERSION``.

This module checks the property rather than the specific spelling, so the
guard survives the config being re-pointed at some other file later.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"


def _pyproject() -> dict:
    with _PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def _version_targets() -> list[str]:
    return list(_pyproject()["tool"]["semantic_release"]["version_variable"])


def _literal_assignment(variable: str) -> re.Pattern[str]:
    """The assignment shape semantic-release can rewrite."""
    return re.compile(
        rf"^\s*{re.escape(variable)}\s*(?::[^=]+)?=\s*(?P<quote>[\"']).*?(?P=quote)",
        re.MULTILINE,
    )


class TestVersionTargets:
    def test_at_least_one_target_is_configured(self) -> None:
        assert _version_targets()

    @pytest.mark.parametrize("target", _version_targets())
    def test_target_file_exists(self, target: str) -> None:
        path, _, _variable = target.rpartition(":")
        assert (_REPO_ROOT / path).is_file(), target

    @pytest.mark.parametrize("target", _version_targets())
    def test_target_is_a_rewritable_string_literal(self, target: str) -> None:
        path, _, variable = target.rpartition(":")
        source = (_REPO_ROOT / path).read_text(encoding="utf-8")
        assert _literal_assignment(variable).search(source), (
            f"{target!r} does not name a quoted string-literal assignment, so "
            "semantic-release cannot bump it. Point the entry at the literal "
            "that actually holds the version."
        )

    def test_a_call_expression_would_be_rejected(self) -> None:
        """Guards the check above from being vacuous."""
        assert not _literal_assignment("__version__").search("__version__ = _get_version()")
        assert _literal_assignment("__version__").search('__version__ = "2.1.0"')


class TestVersionsAgree:
    """Every place the version is written down must say the same thing."""

    def test_fallback_literal_matches_project_version(self) -> None:
        from omni_mercury_engine._version import _FALLBACK_VERSION

        assert _pyproject()["project"]["version"] == _FALLBACK_VERSION

    def test_resolved_version_matches_project_version(self) -> None:
        import omni_mercury_engine

        assert omni_mercury_engine.__version__ == _pyproject()["project"]["version"]

    def test_helm_chart_version_matches(self) -> None:
        import yaml

        chart = yaml.safe_load((_REPO_ROOT / "helm" / "mercury-agent" / "Chart.yaml").read_text())
        project_version = _pyproject()["project"]["version"]
        assert chart["version"] == project_version
        assert chart["appVersion"] == project_version

    def test_kubernetes_workload_labels_match(self) -> None:
        """`app.kubernetes.io/version` drifted to 2.0.0 across seven sites."""
        project_version = _pyproject()["project"]["version"]
        pattern = re.compile(r'app\.kubernetes\.io/version:\s*"?([^"\s]+)"?')
        checked = 0
        for manifest in (_REPO_ROOT / "k8s").rglob("*.yaml"):
            for found in pattern.findall(manifest.read_text(encoding="utf-8")):
                checked += 1
                assert found == project_version, f"{manifest}: {found}"
        assert checked >= 6, "expected the version label on both workloads"

    def test_kustomize_app_info_version_matches(self) -> None:
        import yaml

        kustomization = yaml.safe_load(
            (_REPO_ROOT / "k8s" / "base" / "kustomization.yaml").read_text(encoding="utf-8")
        )
        literals = {
            generator["name"]: generator.get("literals", [])
            for generator in kustomization["configMapGenerator"]
        }
        assert f"version={_pyproject()['project']['version']}" in literals["mercury-agent-app-info"]
