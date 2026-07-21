# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for :mod:`scripts.run_sigma_mutation_gate`.

The harness is exercised end-to-end against a small fixture module and a
real (but fast) test command, covering:

* deterministic mutation-site enumeration across every operator class,
* ``# pragma: no mutate`` exemption,
* single-site application producing a compilable mutant that differs
  from the original,
* byte-exact restoration of the target file after every mutant —
  including when the test command fails,
* kill/survive classification driven by a genuine assertion,
* deterministic stride sampling,
* the red-baseline abort (exit 2) that prevents a fabricated 100% score,
* the ``--fail-under`` gate outcome in both directions,
* the JSON report artifact.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_sigma_mutation_gate import (
    enumerate_sites,
    main,
    mutate_source,
    run_gate,
    stride_sample,
)

#: Fixture module: one function per operator class the harness mutates.
FIXTURE_SOURCE = '''\
"""Fixture for mutation-gate tests."""

THRESHOLD = 0.5
LIMIT = 3
ENABLED = True
EXEMPT = 10  # pragma: no mutate


def clamp_low(value):
    if value < THRESHOLD and ENABLED:
        return value + 1.0
    return value


def is_valid(flag):
    return not flag
'''

#: Test command that pins the fixture's behaviour: any single mutation of
#: the comparison, the boolean operator, the arithmetic, the constants, or
#: the ``not`` must flip at least one of these assertions.
KILLING_TEST = (
    "import fixture_mod as m; "
    "assert m.clamp_low(0.4) == 1.4; "
    "assert m.clamp_low(0.6) == 0.6; "
    "assert m.clamp_low(0.5) == 0.5; "
    "assert m.is_valid(False) is True; "
    "assert m.is_valid(True) is False; "
    "assert m.LIMIT == 3; "
    "assert m.EXEMPT == 10"
)

#: Test command that cannot detect any mutation (always passes).
BLIND_TEST = "import fixture_mod"


@pytest.fixture()
def fixture_repo(tmp_path: Path) -> tuple[Path, str]:
    """A throwaway repo root containing the fixture module."""
    target = tmp_path / "fixture_mod.py"
    target.write_text(FIXTURE_SOURCE, encoding="utf-8")
    return tmp_path, "fixture_mod.py"


def _test_cmd(body: str) -> str:
    return f'{sys.executable} -c "{body}"'


# =============================================================================
# Enumeration
# =============================================================================


def test_enumeration_is_deterministic_and_complete(fixture_repo: tuple[Path, str]) -> None:
    root, rel = fixture_repo
    sites_a = enumerate_sites(root / rel, rel)
    sites_b = enumerate_sites(root / rel, rel)
    assert [s.description for s in sites_a] == [s.description for s in sites_b]

    operators = {s.operator for s in sites_a}
    assert "compare_swap[0]" in operators
    assert "bool_swap" in operators
    assert "arith_swap" in operators
    assert "not_removal" in operators
    assert "bool_flip" in operators
    assert "int_tweak" in operators
    assert "float_tweak" in operators


def test_pragma_no_mutate_exempts_line(fixture_repo: tuple[Path, str]) -> None:
    root, rel = fixture_repo
    sites = enumerate_sites(root / rel, rel)
    exempt_lines = {s.lineno for s in sites}
    # EXEMPT = 10 lives on the pragma line and must not be enumerated.
    pragma_line = FIXTURE_SOURCE.splitlines().index("EXEMPT = 10  # pragma: no mutate") + 1
    assert pragma_line not in exempt_lines


# =============================================================================
# Mutation application
# =============================================================================


def test_each_site_produces_distinct_compilable_mutant(
    fixture_repo: tuple[Path, str],
) -> None:
    root, rel = fixture_repo
    source = (root / rel).read_text(encoding="utf-8")
    sites = enumerate_sites(root / rel, rel)
    assert sites, "fixture must enumerate at least one site"
    for site in sites:
        mutated = mutate_source(source, site.index)
        assert mutated is not None, f"site {site.index} ({site.description}) was invalid"
        compile(mutated, "<mutant>", "exec")
        # The mutated module must differ semantically from the original —
        # unparse both to normalise formatting before comparing.
        assert ast.unparse(ast.parse(mutated)) != ast.unparse(ast.parse(source)), (
            f"site {site.index} ({site.operator}: {site.description}) produced "
            "an identical module"
        )


def test_out_of_range_site_returns_none(fixture_repo: tuple[Path, str]) -> None:
    root, rel = fixture_repo
    source = (root / rel).read_text(encoding="utf-8")
    assert mutate_source(source, 10_000) is None


# =============================================================================
# Stride sampling
# =============================================================================


def test_stride_sample_bounds_and_determinism(fixture_repo: tuple[Path, str]) -> None:
    root, rel = fixture_repo
    sites = enumerate_sites(root / rel, rel)
    sampled = stride_sample(sites, 3)
    assert len(sampled) == 3
    assert sampled == stride_sample(sites, 3)
    indices = [s.index for s in sampled]
    assert indices == sorted(indices)
    # Zero or oversize bounds return everything.
    assert stride_sample(sites, 0) == sites
    assert stride_sample(sites, len(sites) + 5) == sites


# =============================================================================
# Gate end-to-end
# =============================================================================


def test_killing_tests_pass_gate_and_restore_file(fixture_repo: tuple[Path, str]) -> None:
    root, rel = fixture_repo
    original_bytes = (root / rel).read_bytes()
    exit_code = run_gate(
        targets=[rel],
        test_cmd=_test_cmd(KILLING_TEST),
        fail_under=95.0,
        max_mutants=0,
        test_timeout=60.0,
        repo_root=root,
    )
    assert exit_code == 0
    assert (root / rel).read_bytes() == original_bytes


def test_blind_tests_fail_gate_and_restore_file(fixture_repo: tuple[Path, str]) -> None:
    root, rel = fixture_repo
    original_bytes = (root / rel).read_bytes()
    exit_code = run_gate(
        targets=[rel],
        test_cmd=_test_cmd(BLIND_TEST),
        fail_under=50.0,
        max_mutants=0,
        test_timeout=60.0,
        repo_root=root,
    )
    assert exit_code == 1
    assert (root / rel).read_bytes() == original_bytes


def test_red_baseline_aborts_with_exit_2(fixture_repo: tuple[Path, str]) -> None:
    """A failing baseline must abort — not fabricate a perfect score."""
    root, rel = fixture_repo
    exit_code = run_gate(
        targets=[rel],
        test_cmd=_test_cmd("import fixture_mod; assert False"),
        fail_under=50.0,
        max_mutants=0,
        test_timeout=60.0,
        repo_root=root,
    )
    assert exit_code == 2


def test_missing_target_exits_2(tmp_path: Path) -> None:
    exit_code = run_gate(
        targets=["does_not_exist.py"],
        test_cmd=_test_cmd(BLIND_TEST),
        fail_under=50.0,
        max_mutants=0,
        test_timeout=60.0,
        repo_root=tmp_path,
    )
    assert exit_code == 2


def test_report_artifact_written(fixture_repo: tuple[Path, str], tmp_path: Path) -> None:
    root, rel = fixture_repo
    report_path = tmp_path / "mutation_report.json"
    exit_code = run_gate(
        targets=[rel],
        test_cmd=_test_cmd(KILLING_TEST),
        fail_under=95.0,
        max_mutants=4,
        test_timeout=60.0,
        repo_root=root,
        report_path=str(report_path),
    )
    assert exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["n_selected"] == 4
    assert report["killed"] + report["survived"] + report["invalid"] == 4
    assert report["kill_rate_percent"] >= 95.0
    assert len(report["outcomes"]) == 4
    statuses = {outcome["status"] for outcome in report["outcomes"]}
    assert statuses.issubset({"killed", "survived", "timeout_killed", "invalid"})


def test_main_list_mode_runs_no_tests(fixture_repo: tuple[Path, str]) -> None:
    root, rel = fixture_repo
    exit_code = main(
        [
            "--targets",
            rel,
            "--repo-root",
            str(root),
            "--test-cmd",
            _test_cmd("raise SystemExit(1)"),  # would fail if executed
            "--list",
        ]
    )
    assert exit_code == 0
