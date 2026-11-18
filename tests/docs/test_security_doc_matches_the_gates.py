# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``SECURITY.md`` describes the gates the code runs, not the ones it used to.

Until 2026-11-18 its hard-gate section said every boundary ran a mandatory
``Benevolence (>= 0.99)`` gate with "no advisory mode" and reserved
``check="benevolence"`` for it. ``CAPABILITY_MATRIX.md`` section 6 had recorded
that bar as removed, the boundary code raises ``check="harm_uplift"``, and the
doc lint passed all the same: it checks that a path exists, not that a
sentence is true.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from omni_mercury_engine.cognitive import decision_gate
from omni_mercury_engine.security import sigma_immutable_gate

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SECURITY = (_REPO_ROOT / "SECURITY.md").read_text(encoding="utf-8")


def _section(title: str) -> str:
    start = _SECURITY.index(f"### {title}")
    end = _SECURITY.find("\n### ", start + 1)
    return _SECURITY[start : end if end != -1 else len(_SECURITY)]


def test_the_boundary_section_names_the_gates_that_run() -> None:
    section = _section("Hard-gate boundary contract")

    assert "harm-uplift gate" in section
    assert "no advisory mode" not in " ".join(section.split())
    assert "advisory" in section  # what the benevolence score now is


def test_every_check_code_the_section_lists_is_one_the_gates_raise() -> None:
    section = " ".join(_section("Hard-gate boundary contract").split())
    listed = re.search(r"codes these two raise are (.+?)\.", section)
    assert listed is not None
    codes = set(re.findall(r'`"([a-z_]+)"`', listed.group(1)))
    raised = set(
        re.findall(
            r'check="([a-z_]+)"',
            inspect.getsource(decision_gate) + inspect.getsource(sigma_immutable_gate),
        )
    )

    assert codes == {"harm_uplift", "sigma_immutable", "gosnn_unavailable"}
    assert codes <= raised
    assert "benevolence" not in codes


def test_no_validation_is_claimed_that_was_not_entered() -> None:
    """The matrix says CAVP/CMVP was never entered; no sentence here may say otherwise."""
    assert "ACVP-validated" not in _SECURITY
    assert "FIPS-validated" not in _SECURITY
    assert "FIPS validated" not in _SECURITY
