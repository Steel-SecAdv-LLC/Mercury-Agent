# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Atheris fuzz lane is wired end to end.

``.github/workflows/fuzz.yml`` runs ``tests/fuzz/atheris_cap_parser.py`` with
a seed corpus and a token dictionary, from the ``ci-fuzz`` group in uv.lock.
Each piece can be renamed or dropped on its own, and the lane would then fuzz
nothing (or fail to start) without any test noticing; these checks hold the
pieces together. The harness itself imports Atheris, which installs only on
Linux x86_64, so its contract is exercised here through its source and the
parsers directly rather than by importing it.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from omni_mercury_engine.alerting.cap_generator import CAPAlertGenerator
from omni_mercury_engine.watch.world import parse_cap

_REPO = Path(__file__).resolve().parents[2]
_HARNESS = _REPO / "tests" / "fuzz" / "atheris_cap_parser.py"
_DICT = _REPO / "tests" / "fuzz" / "cap.dict"
_WORKFLOW = _REPO / ".github" / "workflows" / "fuzz.yml"
_SEEDS = sorted((_REPO / "tests" / "watch" / "data").glob("*.cap.xml"))


def test_the_workflow_runs_the_harness_with_its_dictionary_and_seeds() -> None:
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    assert "python tests/fuzz/atheris_cap_parser.py" in workflow
    assert "-dict=tests/fuzz/cap.dict" in workflow
    assert "cp tests/watch/data/*.cap.xml" in workflow
    assert "--group ci-fuzz" in workflow
    assert re.search(r'python-version: "3\.1[2-9]"', workflow), "Atheris needs CPython >= 3.12"
    assert "-max_total_time=" in workflow


def test_the_harness_is_an_atheris_entry_point() -> None:
    source = _HARNESS.read_text(encoding="utf-8")
    assert re.search(r"^import atheris$", source, re.MULTILINE)
    assert "atheris.Setup(sys.argv, fuzz_one_input)" in source
    assert "atheris.instrument_imports(" in source


def test_atheris_is_locked_in_the_ci_fuzz_group() -> None:
    pyproject = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    group = pyproject["dependency-groups"]["ci-fuzz"]
    assert any(spec.startswith("atheris==") for spec in group), group
    lock = tomllib.loads((_REPO / "uv.lock").read_text(encoding="utf-8"))
    assert any(package["name"] == "atheris" for package in lock["package"])


def test_the_dictionary_is_valid_libfuzzer_syntax() -> None:
    entries = [
        line
        for line in _DICT.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert len(entries) >= 20
    for entry in entries:
        assert re.fullmatch(r'"(?:[^"\\]|\\.)+"', entry), entry


def test_every_seed_satisfies_the_contract_the_harness_checks() -> None:
    """A seed that broke the contract would crash the lane before it fuzzed."""
    assert len(_SEEDS) >= 2
    for seed in _SEEDS:
        document = seed.read_bytes().decode("utf-8", errors="replace")
        assert isinstance(parse_cap(document), dict), seed.name
        assert isinstance(CAPAlertGenerator.validate_cap_xml(document), bool)
