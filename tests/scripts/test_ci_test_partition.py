# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Splitting the suite for the coverage lane drops no test and loses no coverage gate.

The ML Tests lane runs the whole suite under coverage.  As one pytest session
that needed more memory than the hosted runner has (measured: 5.9 GB after
collection, 13.6 GB peak at four workers on a 16 GB machine; on the runner,
with the pinned models loaded too, the job was killed at 36%), so the lane
runs the suite as several sessions that append to one coverage file.

A split is only safe if nothing falls between the parts.  These tests hold
that ``scripts/ci_test_partition.py`` puts every test file pytest collects in
exactly one part, for any number of parts, and that the workflow step built
on it still runs every part, appends coverage, and enforces the floor once,
at the end, over all of it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from scripts import ci_test_partition as split

_WORKFLOW = _REPO / ".github" / "workflows" / "ci.yml"


@pytest.fixture(autouse=True)
def _one_reading_of_the_tree(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Every scan in a test sees the same files.

    Another test (``tests/scripts/test_xdist_group.py``) writes a scratch test
    file under ``tests/`` and removes it, possibly on another worker while
    these run; read twice, the tree could differ between the two readings.
    """
    snapshot = split.collected_files()
    real = split.collected_files

    def frozen(tests: Path = split.TESTS, patterns: tuple[str, ...] | None = None) -> list[Path]:
        return list(snapshot) if tests == split.TESTS else real(tests, patterns)

    monkeypatch.setattr(split, "collected_files", frozen)
    return snapshot


def _under(entry: str) -> set[Path]:
    """Every test file pytest collects when handed ``tests/<entry>``."""
    target = split.TESTS / entry
    return {path for path in split.collected_files() if path == target or target in path.parents}


class TestNoTestFallsBetweenTheParts:
    @pytest.mark.parametrize("parts", [1, 2, 3, 4, 7])
    def test_the_parts_are_disjoint_and_together_are_the_whole_suite(self, parts: int) -> None:
        assigned = split.partition(parts)
        assert len(assigned) == parts
        names = [name for part in assigned for name in part]
        assert len(names) == len(set(names)), "an entry is in two parts"
        everything = set(split.collected_files())
        covered: set[Path] = set()
        for part in assigned:
            for name in part:
                files = _under(name)
                assert files, f"tests/{name} holds no test file"
                assert not covered & files
                covered |= files
        assert covered == everything
        assert len(everything) > 1000  # the scan is not vacuous

    def test_it_follows_pytests_own_file_patterns(self, tmp_path: Path) -> None:
        """A file named for either configured pattern is a test; a helper beside it is not."""
        assert set(split.collected_patterns()) == {"test_*.py", "*_test.py"}
        tree = tmp_path / "tests"
        (tree / "a" / "deep").mkdir(parents=True)
        (tree / "b").mkdir()
        (tree / "helpers").mkdir()
        (tree / "a" / "test_one.py").write_text("")
        (tree / "a" / "deep" / "two_test.py").write_text("")
        (tree / "b" / "test_three.py").write_text("")
        (tree / "helpers" / "builders.py").write_text("")
        (tree / "test_top.py").write_text("")
        (tree / "conftest.py").write_text("")
        patterns = split.collected_patterns()
        assert split.entries(tree, patterns) == {"a": 2, "b": 1, "test_top.py": 1}
        assert split.partition(2, tree, patterns) == [["a"], ["b", "test_top.py"]]

    def test_the_parts_are_balanced(self) -> None:
        sized = split.entries()
        weights = [sum(sized[name] for name in part) for part in split.partition(3)]
        assert max(weights) - min(weights) <= max(sized.values())
        assert max(weights) < 0.4 * sum(weights)

    def test_the_assignment_depends_on_nothing_but_the_tree(self) -> None:
        assert split.partition(3) == split.partition(3)

    def test_the_command_prints_paths_pytest_accepts(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        printed: list[str] = []
        for index in range(3):
            assert split.main(["3", str(index)]) == 0
            lines = capsys.readouterr().out.split()
            assert lines and all(line.startswith("tests/") for line in lines)
            printed += lines
        assert sorted(printed) == sorted(f"tests/{name}" for name in split.entries())
        with pytest.raises(SystemExit):
            split.main(["3", "3"])
        with pytest.raises(ValueError, match="at least 1"):
            split.partition(0)


@pytest.fixture(scope="module")
def step() -> dict[str, Any]:
    """The ML Tests step that runs the suite."""
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["ml-tests"]["steps"]
    found: dict[str, Any] = next(s for s in steps if "ci_test_partition.py" in str(s.get("run")))
    return found


class TestTheLaneStillRunsAndGatesEverything:
    def test_every_part_is_run_from_the_partition(self, step: dict[str, Any]) -> None:
        script = step["run"]
        parts = int(step["env"]["SUITE_PARTS"])
        assert parts >= 2
        assert 'for part in $(seq 0 "$last")' in script and "last=$((SUITE_PARTS - 1))" in script
        assert 'python scripts/ci_test_partition.py "$SUITE_PARTS" "$part"' in script
        assert 'pytest "${paths[@]}"' in script
        # No session of this lane names the whole tree: that is the run that did not fit.
        assert not re.search(r"pytest\s+tests/\s", script)

    def test_coverage_is_one_file_and_the_floor_is_enforced_once_over_all_of_it(
        self, step: dict[str, Any]
    ) -> None:
        script = step["run"]
        assert "--cov=src/omni_mercury_engine" in script
        later = script.index('if [ "$part" -gt 0 ]; then')
        assert 0 < script.index("append=(--cov-append)", later) - later < 80
        final = script.index('if [ "$part" -eq "$last" ]; then')
        floor = script.index("--cov-fail-under=${{ env.COVERAGE_THRESHOLD_FULL }}")
        assert 0 < floor - final < 200, "the floor belongs to the last part only"
        assert script.count("--cov-fail-under") == 1
        # Every other part reports nothing: it only adds to the data file.
        assert script.index("report=(--cov-report=)") < final
        assert '"${append[@]}" "${report[@]}"' in script
        workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
        assert int(workflow["env"]["COVERAGE_THRESHOLD_FULL"]) >= 62

    def test_a_failing_part_fails_the_step_without_hiding_the_others(
        self, step: dict[str, Any]
    ) -> None:
        script = step["run"]
        assert '"${append[@]}" "${report[@]}" || status=1' in script
        assert script.rstrip().endswith('exit "$status"')
        assert "holds no tests" in script  # an empty part is an error, not a pass

    def test_each_parts_peak_memory_is_reported(self, step: dict[str, Any]) -> None:
        """The margin on the runner is measured in the job log, not assumed."""
        script = step["run"]
        assert "free -m" in script and "suite-memory.log" in script
        assert "::notice::suite part" in script and "peak memory in use" in script
        assert 'kill "$sampler"' in script

    def test_the_lane_keeps_its_workers_timeout_and_must_run_gates(
        self, step: dict[str, Any]
    ) -> None:
        script, env = step["run"], step["env"]
        for flag in ("-n 4", "--timeout=300", "--dist=loadgroup"):
            assert flag in script
        for gate in (
            "MERCURY_REQUIRES_ML",
            "MERCURY_REQUIRES_BROWSER",
            "MERCURY_REQUIRES_HF_MODELS",
            "MERCURY_FULL_LEDGER_CHECK",
            "AMA_REQUIRE_REAL_PQC",
        ):
            assert str(env[gate]) in ("1", "true")
        assert "--deselect" not in script and "-k " not in script and "--ignore" not in script
