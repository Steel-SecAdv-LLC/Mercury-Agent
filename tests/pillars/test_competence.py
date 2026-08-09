# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pillar: competence — the quality bar is the one that is actually enforced.

Competence is not "we aim for 85 % coverage". It is "a pull request below the
stated floor cannot merge", and the floor being stated is the one CI blocks on.
Three properties:

* **The declared floor is the enforced floor.** ``pyproject.toml``'s
  ``fail_under`` matches ``COVERAGE_THRESHOLD_CORE`` in ``ci.yml``, and both
  test lanes pass ``--cov-fail-under`` from those env vars. The project used to
  declare ``fail_under = 85`` while every lane ran at 33/62 — a number no lane
  enforced, which reads as a guarantee and was not one. 85 survives as a
  labelled aspiration in prose, not as a config key pretending to be a gate.
* **No green-washed lane.** A test file whose skip condition is never satisfied
  in CI contributes a passing lane and zero evidence. The numba fast-path
  parity tests were exactly that: gated on the ``performance`` extra, which no
  CI lane installed. ``[all]`` now includes it.
* **Nothing swallows a red test run.** No pytest step in CI is
  ``continue-on-error``.
"""

from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _ci_text() -> str:
    return CI_WORKFLOW.read_text(encoding="utf-8")


def _ci_floors() -> tuple[int, int]:
    """(CORE, FULL) from the ci.yml env block — the authoritative source."""
    text = _ci_text()
    core = re.search(r"^\s*COVERAGE_THRESHOLD_CORE:\s*(\d+)\s*$", text, re.M)
    full = re.search(r"^\s*COVERAGE_THRESHOLD_FULL:\s*(\d+)\s*$", text, re.M)
    assert core and full, "ci.yml no longer declares the coverage threshold env vars"
    return int(core.group(1)), int(full.group(1))


def _pyproject() -> dict[str, Any]:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


class TestTheDeclaredFloorIsTheEnforcedFloor:
    def test_pyproject_fail_under_matches_the_ci_core_floor(self) -> None:
        core, _ = _ci_floors()
        declared = _pyproject()["tool"]["coverage"]["report"]["fail_under"]
        assert declared == core, (
            f"pyproject declares fail_under={declared} but CI enforces {core}; "
            "a floor nothing runs at is not a floor"
        )

    def test_both_lanes_enforce_a_floor_from_the_env_block(self) -> None:
        text = _ci_text()
        assert "--cov-fail-under=${{ env.COVERAGE_THRESHOLD_CORE }}" in text
        assert "--cov-fail-under=${{ env.COVERAGE_THRESHOLD_FULL }}" in text

    def test_the_floors_are_the_measured_ones(self) -> None:
        """Pinned so a graduation is deliberate and its docs move with it."""
        assert _ci_floors() == (33, 62)

    def test_the_full_floor_is_at_least_the_core_floor(self) -> None:
        core, full = _ci_floors()
        assert full >= core

    def test_every_coverage_lane_either_gates_or_says_it_does_not(self) -> None:
        """A ``--cov`` run with no floor measures without gating — say so, in place.

        Ungated coverage is legitimate (a subpackage slice for Codecov flags is
        not comparable to the whole-tree floors), but it must be labelled at the
        call site, or a reader counts it as a gate it is not.
        """
        text = _ci_text()
        blocks = text.split("--cov-report=term-missing")
        # Each block that produced coverage must be followed either by a
        # ``--cov-fail-under`` or by the explicit report-only declaration.
        ungated_unlabelled = []
        for index, block in enumerate(blocks[:-1]):
            if "--cov=src/omni_mercury_engine" not in block:
                continue
            tail = blocks[index + 1][:1200]
            if "--cov-fail-under=" in tail or "COVERAGE REPORT-ONLY" in tail:
                continue
            ungated_unlabelled.append(tail[:200])
        assert not ungated_unlabelled, ungated_unlabelled

    def test_eighty_five_is_labelled_an_aspiration_where_it_appears(self) -> None:
        """If the old number is still written down, it must not read as a gate."""
        text = _ci_text()
        for match in re.finditer(r"fail_under = 85|85-point|85 %", text):
            window = text[max(0, match.start() - 400) : match.end() + 400].lower()
            assert "aspiration" in window or "target" in window, match.group(0)


class TestNoGreenWashedLane:
    def test_the_all_extra_installs_the_numba_fast_path_dependency(self) -> None:
        """Without this the JIT parity tests skip in every lane and prove nothing."""
        extras = _pyproject()["project"]["optional-dependencies"]
        assert "performance" in extras
        all_spec = " ".join(extras["all"])
        assert "performance" in all_spec, (
            "tests/test_native_acceleration.py's numba lane is gated on the "
            "``performance`` extra; if no CI lane installs it the lane is green "
            "by skipping, not by passing"
        )

    def test_the_full_suite_lane_installs_the_all_extra(self) -> None:
        assert 'pip install -e ".[all,dev]"' in _ci_text()

    def test_the_numba_parity_tests_exist_and_are_collected_by_the_full_lane(self) -> None:
        target = REPO_ROOT / "tests" / "test_native_acceleration.py"
        assert target.is_file()
        # The ml-tests lane runs ``pytest tests/``, i.e. the whole tree.
        assert re.search(r"pytest tests/ \\", _ci_text())

    @pytest.mark.skipif(
        importlib.util.find_spec("numba") is None,
        reason="numba is an optional extra; CI's [all,dev] lane installs it",
    )
    def test_the_numba_fast_paths_actually_execute_when_numba_is_present(self) -> None:
        """The lane is only meaningful if the JIT paths run when the extra is in."""
        import numpy as np

        from omni_mercury_engine.detectors import spatial

        assert spatial.NUMBA_AVAILABLE is True, (
            "numba is importable but the spatial detector did not pick up its "
            "JIT lane -- the fast path is dead code in this configuration"
        )
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, (64, 4))
        detector = spatial.SpatialAnomalyDetector()
        detector.fit(data)
        # Exercises ``_compute_distance_scores``, which branches into the
        # JIT kernels when NUMBA_AVAILABLE -- the fast path itself, not a proxy.
        scores = np.asarray(detector._compute_distance_scores(data))
        assert scores.shape[0] == 64
        assert np.all(np.isfinite(scores))

        result = detector.detect(data)
        assert "scores" in result or "is_anomaly" in result


class TestNothingSwallowsARedRun:
    def test_no_pytest_step_is_continue_on_error(self) -> None:
        """Every ``continue-on-error`` in ci.yml must be a checkout retry."""
        lines = _ci_text().splitlines()
        offenders: list[str] = []
        for index, line in enumerate(lines):
            if "continue-on-error: true" not in line:
                continue
            # Look back for the step's ``- name:`` and forward for its body.
            context = "\n".join(lines[max(0, index - 6) : index + 14]).lower()
            # Transient-infrastructure retries are legitimate: each has a final
            # attempt WITHOUT continue-on-error, so a genuine outage still reds.
            if "attempt" in context and ("checkout" in context or "buildx" in context):
                continue
            offenders.append(context)
        assert not offenders, offenders[:2]

    def test_the_test_lanes_declare_their_dependency_contract(self) -> None:
        """``MERCURY_REQUIRES_ML`` turns a broken install into a red, not skips."""
        text = _ci_text()
        assert text.count('MERCURY_REQUIRES_ML: "1"') >= 2

    def test_the_conftest_gate_aborts_when_the_contract_is_unmet(self) -> None:
        conftest = (REPO_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
        assert "MERCURY_REQUIRES_ML" in conftest
        assert "pytest_sessionstart" in conftest


class TestTheFloorsAreDocumentedConsistently:
    """A floor quoted differently in the docs is a floor nobody can rely on."""

    def test_no_document_quotes_a_superseded_floor_as_current(self) -> None:
        core, full = _ci_floors()
        stale = {"15", "25", "30", "50", "55", "95"} - {str(core), str(full)}
        pattern = re.compile(r"COVERAGE_THRESHOLD_(?:CORE|FULL)\s*[=:]\s*(\d+)", re.IGNORECASE)
        offenders: list[str] = []
        for doc in ("README.md", "ARCHITECTURE.md", "CONTRIBUTING.md"):
            body = (REPO_ROOT / doc).read_text(encoding="utf-8")
            for match in pattern.finditer(body):
                value = match.group(1)
                if value in stale:
                    offenders.append(f"{doc}: {match.group(0)}")
        assert not offenders, offenders
