# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Composition contract for the umbrella ``[all]`` extra.

``[all]`` advertises "Full installation with all features": every extra that
gates a *runtime feature* must be reachable from it, or a full install ships
a feature that fails at first use.  Observed instance: ``[compliance]``
(openpyxl) was missing, so the NIST CSF live-reference mode of a
``pip install .[all]`` environment raised ``ModuleNotFoundError`` (caught by
the network-marked ``tests/test_nist_csf_integrator.py`` live lane).

Deliberately NOT required in ``[all]``: ``dev``/``loadtest``/``benchmark-comparison``
(tooling, not runtime features), ``pqc`` (a git-pinned native build installed
by ``scripts/build_ama_native.sh``, not resolvable from PyPI), and
``performance`` (an opt-in numba acceleration lane whose absence changes
speed, not capability).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_RUNTIME_FEATURE_EXTRAS = (
    "ml",
    "visual",
    "vlm",
    "foundation",
    "medical",
    "face",
    "api",
    "sota",
    "llm",
    "drift",
    "fairness",
    "streaming",
    "optimization",
    "domains",
    "gui",
    "explainability",
    "compliance",
)


def test_all_extra_reaches_every_runtime_feature_extra() -> None:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    extras = data["project"]["optional-dependencies"]
    all_spec = " ".join(extras["all"])
    inner = all_spec[all_spec.index("[") + 1 : all_spec.index("]")]
    reachable = {name.strip() for name in inner.split(",")}
    missing = [name for name in _RUNTIME_FEATURE_EXTRAS if name not in reachable]
    assert not missing, (
        f"[all] does not reach runtime feature extra(s) {missing}; a full "
        "install would ship those features broken at first use"
    )
