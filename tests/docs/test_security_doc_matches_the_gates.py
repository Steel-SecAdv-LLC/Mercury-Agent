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


def test_the_section_names_everything_that_turns_a_gate_off() -> None:
    """Was: "neither has a flag that turns it off", beside a flag that turns one off."""
    from omni_mercury_engine import engine
    from omni_mercury_engine.reasoning import backend

    section = " ".join(_section("Hard-gate boundary contract").split())

    assert "neither has a flag" not in section
    # sigma_Immutable's one switch, and its default.
    assert "_GOSNN_TESTING_BYPASS" in section
    assert engine._GOSNN_TESTING_BYPASS is False
    assert "if _GOSNN_TESTING_BYPASS:" in inspect.getsource(engine)
    # The reasoning backend's, and its default.
    assert "ethics_enabled=False" in section
    parameters = inspect.signature(backend.ReasoningBackend.__init__).parameters
    assert parameters["ethics_enabled"].default is True
    # The harm gate reads nothing that could skip it.
    gate_source = inspect.getsource(decision_gate)
    assert "os.environ" not in gate_source and "getenv" not in gate_source
    assert set(inspect.signature(decision_gate.enforce_decision_boundary).parameters) == {
        "subject",
        "advisory_scorer",
        "harm_classifier",
    }


def test_the_reading_the_section_describes_is_the_one_the_gate_does() -> None:
    section = " ".join(_section("Hard-gate boundary contract").split())

    assert f"{decision_gate.MAX_SUBJECT_CHARS:,} characters" in section
    assert f"{decision_gate.MAX_ASSESSED_CHARS:,} characters" in section
    assert f'details["reason"] == "{decision_gate.UNASSESSABLE}"' in section


def test_the_policy_names_the_ruleset_version_the_code_has() -> None:
    """A cache of verdicts is thrown away on this number; the policy must not name an old one."""
    from omni_mercury_engine.core.centralized_constants import ETHICAL

    policy = (_REPO_ROOT / "docs" / "HARM_POLICY.md").read_text(encoding="utf-8")
    stated = re.search(r"`RULESET_VERSION` is \*\*(\d+)\*\*", policy)

    assert stated is not None
    assert int(stated.group(1)) == ETHICAL.RULESET_VERSION


def test_no_validation_is_claimed_that_was_not_entered() -> None:
    """The matrix says CAVP/CMVP was never entered; no sentence here may say otherwise."""
    assert "ACVP-validated" not in _SECURITY
    assert "FIPS-validated" not in _SECURITY
    assert "FIPS validated" not in _SECURITY
