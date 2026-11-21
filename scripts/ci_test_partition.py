# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Split the test tree into N parts for the CI coverage lane.

The full suite under coverage does not fit a 16 GB hosted runner as one
pytest session: the xdist workers hold 5.9 GB as soon as collection ends and
then grow as tests run (measured 2026-11-21: 13.6 GB at four workers on a
16 GB machine, before the pinned models CI also loads).  The ML Tests lane
and the release gate therefore run the suite as N sessions that append to one
coverage file (``--cov-append``) and enforce the coverage floor on the last
one, over all of them.  A new session starts new workers, so what one part accumulated is
returned before the next begins.  This prints the test paths of one part.

Every top-level entry of ``tests/`` that holds tests (a directory with a test
file anywhere under it, or a test file) belongs to exactly one part.  Parts
are filled largest entry first, each entry going to the part that is lightest
so far by number of test files, so the parts are balanced and the assignment
depends on nothing but the tree.  ``tests/scripts/test_ci_test_partition.py``
holds that the parts are disjoint and that together they are every test file
pytest would collect: no test is left out of CI by being split.

Usage::

    python scripts/ci_test_partition.py 3 0        # paths of part 0 of 3, one per line
    python scripts/ci_test_partition.py 3          # the size of every part
"""

from __future__ import annotations

import argparse
import fnmatch
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TESTS = REPO / "tests"


def collected_patterns(repo: Path = REPO) -> tuple[str, ...]:
    """The file-name patterns pytest collects (``python_files`` in pyproject.toml)."""
    config = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
    options = config.get("tool", {}).get("pytest", {}).get("ini_options", {})
    return tuple(options.get("python_files", ("test_*.py",)))


def is_test_file(path: Path, patterns: tuple[str, ...]) -> bool:
    """Whether pytest would collect ``path`` as a test module."""
    return path.suffix == ".py" and any(fnmatch.fnmatch(path.name, p) for p in patterns)


def collected_files(tests: Path = TESTS, patterns: tuple[str, ...] | None = None) -> list[Path]:
    """Every test module under ``tests``, sorted."""
    found = patterns if patterns is not None else collected_patterns()
    return sorted(
        path
        for path in tests.rglob("*.py")
        if "__pycache__" not in path.parts and is_test_file(path, found)
    )


def entries(tests: Path = TESTS, patterns: tuple[str, ...] | None = None) -> dict[str, int]:
    """Top-level entries of ``tests`` that hold tests, with their test-file counts."""
    found: dict[str, int] = {}
    for path in collected_files(tests, patterns):
        top = path.relative_to(tests).parts[0]
        found[top] = found.get(top, 0) + 1
    return found


def partition(
    parts: int, tests: Path = TESTS, patterns: tuple[str, ...] | None = None
) -> list[list[str]]:
    """Assign every entry to one of ``parts`` parts, balanced by test-file count.

    Args:
        parts: How many parts (at least 1).
        tests: The test tree.
        patterns: Test-file patterns (default: pytest's, from pyproject.toml).

    Returns:
        ``parts`` lists of entry names, each sorted.

    Raises:
        ValueError: If ``parts`` is less than 1.
    """
    if parts < 1:
        raise ValueError("parts must be at least 1")
    assigned: list[list[str]] = [[] for _ in range(parts)]
    weight = [0] * parts
    sized = entries(tests, patterns)
    for name in sorted(sized, key=lambda entry: (-sized[entry], entry)):
        lightest = min(range(parts), key=lambda index: (weight[index], index))
        assigned[lightest].append(name)
        weight[lightest] += sized[name]
    return [sorted(names) for names in assigned]


def main(argv: list[str] | None = None) -> int:
    """Print one part's test paths, or the size of every part."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("parts", type=int, help="how many parts the suite is split into")
    parser.add_argument("index", type=int, nargs="?", help="which part to print (0-based)")
    args = parser.parse_args(argv)
    if args.parts < 1:
        parser.error("parts must be at least 1")
    split = partition(args.parts)
    if args.index is None:
        sized = entries()
        for index, names in enumerate(split):
            files = sum(sized[name] for name in names)
            print(f"part {index}: {len(names)} entries, {files} test files")
        return 0
    if not 0 <= args.index < args.parts:
        parser.error(f"index must be in [0, {args.parts})")
    for name in split[args.index]:
        print(f"tests/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
