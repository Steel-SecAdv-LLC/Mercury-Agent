#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mutation-testing gate for the σ_Immutable hot path (ROADMAP row 8).

Systematically mutates ``security/sigma_immutable_gate.py`` and
``security/sigma_immutable_corpus.py`` one operator at a time, runs the
σ_Immutable-focused test subset against each mutant, and fails the gate
when the measured mutation-kill rate drops below the configured floor.
A surviving mutant means the test suite cannot distinguish the real
σ_Immutable enforcement logic from a broken variant — exactly the class
of silent-weakening regression the ethical gates must be immune to.

Why not ``mutmut`` / ``cosmic-ray``: both drive mutation through their
own import/trampoline machinery, which fights this package's import-time
native-PQC gate (``_pqc_gate`` refuses package import unless the real AMA
backend loads) and neither offers a deterministic, stride-sampled runtime
bound suitable for a blocking CI lane.  This harness applies mutants
**in place** (with byte-exact restoration guaranteed by a ``finally``
block), so the package's import machinery — editable install, native
backend, conftest gates — is exactly the one production tests use.

Mutation operators (deterministic, enumerated in source order):

* comparison swaps: ``<`` ↔ ``<=``, ``>`` ↔ ``>=``, ``==`` ↔ ``!=``
* arithmetic swaps: ``+`` ↔ ``-``, ``*`` → ``+``
* boolean-operator swaps: ``and`` ↔ ``or``
* ``not X`` → ``X``
* numeric-constant tweaks: int ``n`` → ``n + 1``, float ``f`` → ``f + 0.1``
* boolean-constant flips: ``True`` ↔ ``False``

Lines carrying a ``# pragma: no mutate`` comment are exempt.  A mutant
whose source fails to compile is counted as *invalid* and excluded from
the denominator.  A mutant whose test run exceeds ``--test-timeout`` is
counted as **killed** (an infinite-loop mutant is a detected mutant —
the standard mutation-testing convention).

Before any mutation runs, the harness executes the test command once
against the unmutated tree; a failing baseline aborts with exit 2, since
a red baseline would count every mutant as killed and fabricate a
perfect score.

Exit codes
----------
* ``0`` — kill rate ≥ ``--fail-under`` floor.
* ``1`` — kill rate below the floor (survivors are listed).
* ``2`` — configuration / baseline error (missing target, failing
  baseline test run, no mutants generated).

Usage
-----
::

    python scripts/run_sigma_mutation_gate.py                  # full run, default floor
    python scripts/run_sigma_mutation_gate.py --max-mutants 80 # bounded CI run
    python scripts/run_sigma_mutation_gate.py --list           # enumerate only, no test runs
    python scripts/run_sigma_mutation_gate.py --report out.json
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The σ_Immutable hot path (ROADMAP row 8).
DEFAULT_TARGETS = (
    "src/omni_mercury_engine/security/sigma_immutable_gate.py",
    "src/omni_mercury_engine/security/sigma_immutable_corpus.py",
)

#: σ_Immutable-focused test subset: fast, deterministic files that pin the
#: gate's evaluate/enforce semantics, the corpus loader, and the trained
#: classifier's discrimination floor.  ``-x`` short-circuits so killed
#: mutants exit on the first failing test.
#:
#: The ``*_semantics`` file exists specifically for this gate: the
#: first full measurement (2026-07-21) showed the interface-level subset
#: alone killed only 9.7% of mutants — constant tweaks and operator
#: swaps in the projection band math, the vector builders, and the
#: corpus-generation arithmetic all survived.  The semantic suite pins
#: those closed forms directly; do not remove it from this command
#: without re-measuring the kill rate.
DEFAULT_TEST_CMD = (
    "pytest -x -q -p no:cacheprovider "
    "tests/security/test_sigma_immutable_gate_semantics.py "
    "tests/security/test_sigma_immutable_corpus_semantics.py "
    "tests/security/test_sigma_immutable_kat.py "
    "tests/security/test_sigma_immutable_pqc_classification.py "
    "tests/test_sigma_immutable_discrimination.py"
)

#: Comparison-operator swap table.
_COMPARE_SWAPS: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
}

#: Arithmetic-operator swap table.
_ARITH_SWAPS: dict[type[ast.operator], type[ast.operator]] = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Add,
}

_PRAGMA = "no mutate"


@dataclass
class MutationSite:
    """One applicable mutation, addressed by file + enumeration index."""

    target: str  # repo-relative path
    index: int  # position in this file's deterministic enumeration
    operator: str  # human-readable operator name
    lineno: int
    col_offset: int
    description: str


@dataclass
class MutantOutcome:
    """Result of testing one mutant."""

    site: MutationSite
    status: str  # "killed" | "survived" | "timeout_killed" | "invalid"
    duration_seconds: float


def _pragma_lines(source: str) -> set[int]:
    """Line numbers (1-based) carrying a ``# pragma: no mutate`` comment."""
    exempt: set[int] = set()
    for i, line in enumerate(source.splitlines(), start=1):
        if _PRAGMA in line and "#" in line:
            exempt.add(i)
    return exempt


class _SiteEnumerator(ast.NodeVisitor):
    """Enumerate every applicable mutation site in deterministic order."""

    def __init__(self, target: str, exempt_lines: set[int]) -> None:
        self.target = target
        self.exempt_lines = exempt_lines
        self.sites: list[MutationSite] = []

    def _add(self, node: ast.AST, operator: str, description: str) -> None:
        lineno = getattr(node, "lineno", 0)
        if lineno in self.exempt_lines:
            return
        self.sites.append(
            MutationSite(
                target=self.target,
                index=len(self.sites),
                operator=operator,
                lineno=lineno,
                col_offset=getattr(node, "col_offset", 0),
                description=description,
            )
        )

    def visit_Compare(self, node: ast.Compare) -> None:
        for position, op in enumerate(node.ops):
            replacement = _COMPARE_SWAPS.get(type(op))
            if replacement is not None:
                self._add(
                    node,
                    f"compare_swap[{position}]",
                    f"{type(op).__name__} -> {replacement.__name__}",
                )
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        replacement = _ARITH_SWAPS.get(type(node.op))
        if replacement is not None:
            self._add(node, "arith_swap", f"{type(node.op).__name__} -> {replacement.__name__}")
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        flipped = "Or" if isinstance(node.op, ast.And) else "And"
        self._add(node, "bool_swap", f"{type(node.op).__name__} -> {flipped}")
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if isinstance(node.op, ast.Not):
            self._add(node, "not_removal", "not X -> X")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        value = node.value
        if type(value) is bool:
            self._add(node, "bool_flip", f"{value} -> {not value}")
        elif type(value) is int:
            self._add(node, "int_tweak", f"{value} -> {value + 1}")
        elif type(value) is float:
            self._add(node, "float_tweak", f"{value} -> {value + 0.1}")
        self.generic_visit(node)


class _MutationApplier(ast.NodeTransformer):
    """Apply the mutation at one enumerated site index.

    Re-runs the same deterministic enumeration order as
    :class:`_SiteEnumerator`; when the running counter matches the target
    index the node is rewritten.
    """

    def __init__(self, target_index: int, exempt_lines: set[int]) -> None:
        self.target_index = target_index
        self.exempt_lines = exempt_lines
        self.counter = -1
        self.applied: str | None = None

    def _next_matches(self, node: ast.AST) -> bool:
        if getattr(node, "lineno", 0) in self.exempt_lines:
            return False
        self.counter += 1
        return self.counter == self.target_index

    def _count_only(self, node: ast.AST) -> None:
        if getattr(node, "lineno", 0) not in self.exempt_lines:
            self.counter += 1

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        if getattr(node, "lineno", 0) not in self.exempt_lines:
            for position, op in enumerate(node.ops):
                replacement = _COMPARE_SWAPS.get(type(op))
                if replacement is None:
                    continue
                self.counter += 1
                if self.counter == self.target_index:
                    new_node = copy.deepcopy(node)
                    new_node.ops[position] = replacement()
                    self.applied = f"compare_swap[{position}]"
                    return ast.copy_location(self.generic_visit(new_node), node)
        return self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        if type(node.op) in _ARITH_SWAPS:
            if self._next_matches(node):
                new_node = copy.deepcopy(node)
                new_node.op = _ARITH_SWAPS[type(node.op)]()
                self.applied = "arith_swap"
                return ast.copy_location(self.generic_visit(new_node), node)
        return self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        if self._next_matches(node):
            new_node = copy.deepcopy(node)
            new_node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
            self.applied = "bool_swap"
            return ast.copy_location(self.generic_visit(new_node), node)
        return self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        if isinstance(node.op, ast.Not):
            if self._next_matches(node):
                self.applied = "not_removal"
                return ast.copy_location(self.generic_visit(node.operand), node)
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        value = node.value
        if type(value) is bool:
            if self._next_matches(node):
                self.applied = "bool_flip"
                return ast.copy_location(ast.Constant(value=not value), node)
        elif type(value) is int:
            if self._next_matches(node):
                self.applied = "int_tweak"
                return ast.copy_location(ast.Constant(value=value + 1), node)
        elif type(value) is float:
            if self._next_matches(node):
                self.applied = "float_tweak"
                return ast.copy_location(ast.Constant(value=value + 0.1), node)
        return node


def enumerate_sites(target: Path, repo_relative: str) -> list[MutationSite]:
    """Enumerate every mutation site in ``target`` deterministically."""
    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source)
    enumerator = _SiteEnumerator(repo_relative, _pragma_lines(source))
    enumerator.visit(tree)
    return enumerator.sites


def mutate_source(source: str, site_index: int) -> str | None:
    """Return the mutated source for ``site_index``, or None if invalid.

    Invalid means the mutation could not be applied (index out of range)
    or the mutated source no longer compiles.
    """
    tree = ast.parse(source)
    applier = _MutationApplier(site_index, _pragma_lines(source))
    mutated_tree = applier.visit(tree)
    if applier.applied is None:
        return None
    ast.fix_missing_locations(mutated_tree)
    mutated_source = ast.unparse(mutated_tree)
    try:
        compile(mutated_source, "<mutant>", "exec")
    except (SyntaxError, ValueError):
        return None
    return mutated_source


def stride_sample(sites: list[MutationSite], max_mutants: int) -> list[MutationSite]:
    """Deterministic stride sample spanning the full site list evenly."""
    if max_mutants <= 0 or max_mutants >= len(sites):
        return list(sites)
    step = len(sites) / max_mutants
    picked = sorted({int(i * step) for i in range(max_mutants)})
    return [sites[i] for i in picked]


def run_test_command(cmd: list[str], timeout: float, cwd: Path) -> tuple[bool, bool]:
    """Run the test command; returns (passed, timed_out)."""
    try:
        completed = subprocess.run(  # noqa: S603 - operator-supplied test command, same trust boundary as the CI job invoking this gate
            cmd,
            cwd=cwd,
            timeout=timeout,
            capture_output=True,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, True
    return completed.returncode == 0, False


def run_gate(
    targets: Sequence[str],
    test_cmd: str,
    fail_under: float,
    max_mutants: int,
    test_timeout: float,
    repo_root: Path,
    report_path: str | None = None,
    list_only: bool = False,
) -> int:
    """Execute the mutation gate; returns the process exit code."""
    all_sites: list[MutationSite] = []
    sources: dict[str, str] = {}
    for rel in targets:
        path = repo_root / rel
        if not path.is_file():
            print(f"ERROR: mutation target not found: {rel}", file=sys.stderr)
            return 2
        sources[rel] = path.read_text(encoding="utf-8")
        sites = enumerate_sites(path, rel)
        all_sites.extend(sites)
        print(f"  {rel}: {len(sites)} mutation sites")

    if not all_sites:
        print("ERROR: no mutation sites enumerated", file=sys.stderr)
        return 2

    selected = stride_sample(all_sites, max_mutants)
    print(f"==> {len(all_sites)} sites total, {len(selected)} selected for execution")

    if list_only:
        for site in selected:
            print(
                f"  [{site.target}:{site.lineno}:{site.col_offset}] "
                f"{site.operator}: {site.description}"
            )
        return 0

    cmd = shlex.split(test_cmd)

    print("==> Baseline test run (unmutated tree)")
    baseline_start = time.monotonic()
    baseline_passed, baseline_timeout = run_test_command(cmd, test_timeout, repo_root)
    baseline_duration = time.monotonic() - baseline_start
    if baseline_timeout or not baseline_passed:
        print(
            "ERROR: baseline test run failed or timed out — a red baseline would "
            "count every mutant as killed and fabricate a perfect score",
            file=sys.stderr,
        )
        return 2
    print(f"    baseline green in {baseline_duration:.1f}s")

    outcomes: list[MutantOutcome] = []
    killed = survived = invalid = 0
    for ordinal, site in enumerate(selected, start=1):
        original = sources[site.target]
        mutated = mutate_source(original, site.index)
        if mutated is None:
            invalid += 1
            outcomes.append(MutantOutcome(site=site, status="invalid", duration_seconds=0.0))
            continue

        target_path = repo_root / site.target
        start = time.monotonic()
        try:
            target_path.write_text(mutated, encoding="utf-8")
            passed, timed_out = run_test_command(cmd, test_timeout, repo_root)
        finally:
            target_path.write_text(original, encoding="utf-8")
        duration = time.monotonic() - start

        if timed_out:
            status = "timeout_killed"
            killed += 1
        elif passed:
            status = "survived"
            survived += 1
        else:
            status = "killed"
            killed += 1
        outcomes.append(MutantOutcome(site=site, status=status, duration_seconds=duration))
        print(
            f"  [{ordinal}/{len(selected)}] {site.target}:{site.lineno} "
            f"{site.operator} ({site.description}) -> {status} ({duration:.1f}s)"
        )

    evaluated = killed + survived
    kill_rate = (100.0 * killed / evaluated) if evaluated else 0.0
    print(
        f"==> mutation results: {killed} killed, {survived} survived, "
        f"{invalid} invalid — kill rate {kill_rate:.1f}% (floor {fail_under:.1f}%)"
    )
    if survived:
        print("==> surviving mutants (tests cannot detect these changes):")
        for outcome in outcomes:
            if outcome.status == "survived":
                site = outcome.site
                print(
                    f"    {site.target}:{site.lineno}:{site.col_offset} "
                    f"{site.operator}: {site.description}"
                )

    if report_path:
        report = {
            "targets": list(targets),
            "test_cmd": test_cmd,
            "n_sites_total": len(all_sites),
            "n_selected": len(selected),
            "killed": killed,
            "survived": survived,
            "invalid": invalid,
            "kill_rate_percent": kill_rate,
            "fail_under_percent": fail_under,
            "baseline_seconds": baseline_duration,
            "outcomes": [asdict(outcome) for outcome in outcomes],
        }
        Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"==> report written to {report_path}")

    if evaluated == 0:
        print("ERROR: every selected mutant was invalid", file=sys.stderr)
        return 2
    return 0 if kill_rate >= fail_under else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(DEFAULT_TARGETS),
        help="repo-relative files to mutate (default: the σ_Immutable hot path)",
    )
    parser.add_argument(
        "--test-cmd",
        default=DEFAULT_TEST_CMD,
        help="test command executed per mutant (default: σ_Immutable test subset)",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=70.0,
        help="minimum mutation kill rate percent (default: 70)",
    )
    parser.add_argument(
        "--max-mutants",
        type=int,
        default=0,
        help="bound execution to N stride-sampled mutants (0 = all)",
    )
    parser.add_argument(
        "--test-timeout",
        type=float,
        default=600.0,
        help="per-mutant test-run timeout in seconds; a timeout counts as killed",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="write a JSON report to this path",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="enumerate (sampled) mutation sites without running tests",
    )
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
        help="repository root (tests run with this as the working directory)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return run_gate(
        targets=args.targets,
        test_cmd=args.test_cmd,
        fail_under=args.fail_under,
        max_mutants=args.max_mutants,
        test_timeout=args.test_timeout,
        repo_root=Path(args.repo_root),
        report_path=args.report,
        list_only=args.list,
    )


if __name__ == "__main__":
    sys.exit(main())
