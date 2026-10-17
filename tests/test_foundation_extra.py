# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the ``[foundation]`` extra's dependency contract.

Every adapter in ``models/foundation/`` imports torch at module top (the
fusion tensor surface), and the chronos adapter needs
``chronos.ChronosPipeline`` from the ``chronos-forecasting`` package.  The
extra must therefore compose the project's own ``[ml]`` extra and declare
``chronos-forecasting``; historically it declared only stumpy/nixtla, so
``pip install .[foundation]`` produced an environment where the chronos
detector could never become operational and the package it enabled could
not even be imported without a separate ``[ml]`` install.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _foundation_extra() -> list[str]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    extras = data["project"]["optional-dependencies"]
    assert "foundation" in extras, "missing [foundation] extra"
    return [dep.lower() for dep in extras["foundation"]]


def _project_name() -> str:
    """The distribution's own name, so the assertion survives renames."""
    import tomllib

    with (Path(__file__).resolve().parents[1] / "pyproject.toml").open("rb") as handle:
        return str(tomllib.load(handle)["project"]["name"])


def test_foundation_extra_declares_chronos_forecasting() -> None:
    joined = " ".join(_foundation_extra())
    assert "chronos-forecasting" in joined, (
        "[foundation] must declare chronos-forecasting: ChronosAdapter._load_model "
        "imports chronos.ChronosPipeline and is inert without it"
    )


def test_foundation_extra_composes_ml() -> None:
    assert any(f"{_project_name()}[ml]" in dep for dep in _foundation_extra()), (
        "[foundation] must compose mercury-agent-csi[ml]: every models/foundation adapter "
        "imports torch at module top"
    )


def test_foundation_extra_reachable_from_all() -> None:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    extras = data["project"]["optional-dependencies"]
    assert any("foundation" in dep for dep in extras["all"])
