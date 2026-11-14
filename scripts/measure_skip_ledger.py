#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measure which tests can never run, and gate the answer.

A skipped test reports as a pass. The suite holds several hundred skip guards
and almost all of them are sound: a test that needs torch skips where torch is
absent and runs in the lanes that install it. Two kinds are not sound, and
neither shows up in a green run:

* **A test no lane can run.** ``pytest.importorskip("x")`` skips wherever
  ``x`` is missing. When no extra installs ``x``, that is everywhere: the test
  is green in every lane and has never executed. ``gymnasium`` was the case
  that prompted this script -- ``test_gymnasium_vector_environments`` skipped
  in all lanes because no extra carried it.
* **A test that is switched off.** ``@pytest.mark.skip``, an unconditional
  ``pytest.skip(...)`` and ``@pytest.mark.xfail`` each park a test without a
  condition under which it comes back.

This script reads ``tests/`` with :mod:`ast` (nothing is imported, so it needs
no ML or native install) and ``uv.lock`` for what the fullest test lane
installs, and reports:

1. every dependency that guards a test through ``importorskip``, the
   distribution that provides it, and whether the full lane installs it;
2. every unconditionally skipped test and every expected failure.

The result is rendered between ``<!-- SKIP-LEDGER:START -->`` and
``<!-- SKIP-LEDGER:END -->`` in ``docs/SKIP_LEDGER.md``. Volatile totals (how
many files guard on torch) are printed by the plain report and deliberately
kept out of the gated block, so adding an ordinary guarded test does not
require regenerating a document.

Usage::

    python scripts/measure_skip_ledger.py                       # full report
    python scripts/measure_skip_ledger.py --json                # machine form
    python scripts/measure_skip_ledger.py --update docs/SKIP_LEDGER.md
    python scripts/measure_skip_ledger.py --check docs/SKIP_LEDGER.md

Exit codes:
    0  Report printed, block updated, or (``--check``) current with no violation.
    1  ``--check`` found a stale block or a violation: a guard on a dependency
       the full lane does not install, an unconditional skip, or an expected
       failure, that is not declared in this file with a reason.
    2  A source is missing or the markers are absent.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TESTS = REPO / "tests"
LOCK = REPO / "uv.lock"

LEDGER_START = "<!-- SKIP-LEDGER:START -->"
LEDGER_END = "<!-- SKIP-LEDGER:END -->"

#: The distribution ``uv.lock`` is written for.
PROJECT = "mercury-agent-global"

#: The first-party import package; a guard on one of its modules is a guard on
#: that module's own optional imports, not on an installable distribution.
FIRST_PARTY = "omni_mercury_engine"

#: Extras the fullest test lane installs (``ml-tests`` in ``ci.yml``:
#: ``uv sync --locked --extra all --extra dev``). Pinned to that command by
#: ``tests/test_measure_skip_ledger.py``.
FULL_LANE_EXTRAS: tuple[str, ...] = ("all", "dev")

#: Import name -> the distributions that provide it, where the two differ by
#: more than case and ``_``/``-``. Any one of them being installed is enough.
IMPORT_TO_DISTRIBUTIONS: dict[str, tuple[str, ...]] = {
    "PIL": ("pillow",),
    "chronos": ("chronos-forecasting",),
    "cv2": ("opencv-python", "opencv-python-headless", "opencv-contrib-python"),
    "sklearn": ("scikit-learn",),
    "xdist": ("pytest-xdist",),
    "yaml": ("pyyaml",),
}

#: Guards on a dependency the full lane does not install, each with the reason
#: it is acceptable and where the test does run.
ALLOWED_UNINSTALLED: dict[str, str] = {
    "ama_cryptography": (
        "built from source in every test lane by "
        "`.github/actions/build-ama-cryptography`; the `[pqc]` extra is kept out of "
        "`[all]` on purpose (`tests/test_all_extra_contract.py`)"
    ),
}

#: Unconditional skips and expected failures, keyed ``path::name``, each with
#: the reason the test is parked and what would bring it back. Empty: none.
ALLOWED_PARKED: dict[str, str] = {}

_BRANCHING = (
    ast.If,
    ast.Try,
    ast.ExceptHandler,
    ast.While,
    ast.For,
    ast.AsyncFor,
    ast.With,
    ast.AsyncWith,
    ast.Match,
    ast.IfExp,
    ast.BoolOp,
    ast.Lambda,
)


@dataclass(frozen=True)
class Parked:
    """One test that is switched off without a condition."""

    path: str
    line: int
    name: str
    kind: str  # "skip" | "xfail"
    reason: str

    @property
    def key(self) -> str:
        """The ``path::name`` identifier used in :data:`ALLOWED_PARKED`."""
        return f"{self.path}::{self.name}"


@dataclass
class Scan:
    """Everything the static pass finds in ``tests/``."""

    guards: dict[str, set[str]] = field(default_factory=dict)  # import target -> files
    dynamic_guards: list[str] = field(default_factory=list)  # "path:line"
    parked: list[Parked] = field(default_factory=list)
    conditional_skips: int = 0
    skipif_marks: int = 0
    module_skips: int = 0
    files: int = 0


def _dotted(node: ast.AST) -> str | None:
    """``pytest.mark.skip`` for the matching attribute chain, else ``None``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _reason(call: ast.AST) -> str:
    """The ``reason=`` (or first positional string) of a skip/xfail, if literal."""
    if not isinstance(call, ast.Call):
        return ""
    for keyword in call.keywords:
        if keyword.arg == "reason" and isinstance(keyword.value, ast.Constant):
            return str(keyword.value.value)
    for arg in call.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
    return ""


def _enclosing(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef | None, bool]:
    """Return the function a node sits in and whether a branch lies between them."""
    branched = False
    current = node
    while current in parents:
        parent = parents[current]
        if isinstance(parent, _BRANCHING):
            branched = True
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return parent, branched
        current = parent
    return None, branched


def _returns_before(function: ast.FunctionDef | ast.AsyncFunctionDef, line: int) -> bool:
    """Whether the function can leave before ``line`` (a ``return`` or ``yield``)."""
    return any(
        isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom)) and node.lineno < line
        for node in ast.walk(function)
    )


def _under_module_branch(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """Whether a definition sits under a module-level ``if``/``try``."""
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.If, ast.Try)):
            return True
    return False


def _module_string_tables(tree: ast.Module) -> dict[str, list[str]]:
    """Module-level ``NAME = {...: "module", ...}`` dicts whose values are all strings.

    ``importorskip(TABLE[key])`` guards on every value of ``TABLE``; resolving
    the table keeps such a guard inside the ledger instead of calling it
    unaccountable.
    """
    tables: dict[str, list[str]] = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target, value = statement.targets[0], statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            target, value = statement.target, statement.value
        else:
            continue
        if not (isinstance(target, ast.Name) and isinstance(value, ast.Dict)):
            continue
        values = [
            item.value
            for item in value.values
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        ]
        if values and len(values) == len(value.values):
            tables[target.id] = values
    return tables


def _guard_targets(argument: ast.AST | None, tables: dict[str, list[str]]) -> list[str] | None:
    """Module names an ``importorskip`` argument can take; ``None`` if not static."""
    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
        return [argument.value]
    if (
        isinstance(argument, ast.Subscript)
        and isinstance(argument.value, ast.Name)
        and argument.value.id in tables
    ):
        return tables[argument.value.id]
    return None


def _scan_file(path: Path, rel: str, scan: Scan) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
    tables = _module_string_tables(tree)
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name in {"pytest.importorskip", "importorskip"}:
                targets = _guard_targets(node.args[0] if node.args else None, tables)
                if targets is None:
                    scan.dynamic_guards.append(f"{rel}:{node.lineno}")
                else:
                    for target in targets:
                        scan.guards.setdefault(target, set()).add(rel)
            elif name == "pytest.skip":
                function, branched = _enclosing(node, parents)
                if function is None:
                    scan.module_skips += 1
                elif branched or _returns_before(function, node.lineno):
                    scan.conditional_skips += 1
                else:
                    scan.parked.append(
                        Parked(rel, node.lineno, function.name, "skip", _reason(node))
                    )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for decorator in node.decorator_list:
                marker = decorator.func if isinstance(decorator, ast.Call) else decorator
                name = _dotted(marker)
                if name == "pytest.mark.skipif":
                    scan.skipif_marks += 1
                elif name == "pytest.mark.skip":
                    if _under_module_branch(node, parents):
                        scan.conditional_skips += 1
                    else:
                        scan.parked.append(
                            Parked(rel, decorator.lineno, node.name, "skip", _reason(decorator))
                        )
                elif name == "pytest.mark.xfail":
                    scan.parked.append(
                        Parked(rel, decorator.lineno, node.name, "xfail", _reason(decorator))
                    )


def scan_tests(root: Path = TESTS) -> Scan:
    """Statically inventory every skip construct under ``root``."""
    scan = Scan()
    for path in sorted(root.rglob("*.py")):
        scan.files += 1
        _scan_file(path, path.relative_to(root.parent).as_posix(), scan)
    scan.parked.sort(key=lambda parked: (parked.path, parked.line))
    return scan


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def full_lane_distributions(
    lock_path: Path = LOCK, extras: tuple[str, ...] = FULL_LANE_EXTRAS
) -> set[str]:
    """Distributions the full test lane installs, read from ``uv.lock``.

    The walk starts at the project's own dependencies plus the named extras
    and follows every package's dependencies, including the extras a
    dependency is requested with. Environment markers are ignored, so the set
    is what the lane installs on at least one supported platform and Python.
    """
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    packages = {_normalise(package["name"]): package for package in lock["package"]}
    root = packages[_normalise(PROJECT)]
    pending: list[tuple[str, tuple[str, ...]]] = [(_normalise(PROJECT), extras)]
    seen: set[tuple[str, str]] = set()
    installed: set[str] = set()
    while pending:
        name, wanted = pending.pop()
        package = packages.get(name)
        if package is None:
            continue
        installed.add(name)
        entries = [] if (name, "") in seen else list(package.get("dependencies", []))
        seen.add((name, ""))
        for extra in wanted:
            if (name, extra) in seen:
                continue
            seen.add((name, extra))
            entries += package.get("optional-dependencies", {}).get(extra, [])
        for entry in entries:
            pending.append((_normalise(entry["name"]), tuple(entry.get("extra", ()))))
    installed.discard(_normalise(root["name"]))
    return installed


def provider(target: str, installed: set[str]) -> tuple[str, bool]:
    """Return ``(distribution, installed_by_full_lane)`` for one import target."""
    top = target.split(".", 1)[0]
    if top == FIRST_PARTY:
        return "first-party", True
    if top in sys.stdlib_module_names:
        return "standard library", True
    candidates = IMPORT_TO_DISTRIBUTIONS.get(top, (_normalise(top),))
    for candidate in candidates:
        if _normalise(candidate) in installed:
            return _normalise(candidate), True
    return _normalise(candidates[0]), False


@dataclass(frozen=True)
class GuardRow:
    """One optional import that guards tests."""

    module: str
    distribution: str
    installed: bool
    files: int


def guard_rows(scan: Scan, installed: set[str]) -> list[GuardRow]:
    """One row per top-level import target, sorted by name."""
    grouped: dict[str, set[str]] = {}
    for target, files in scan.guards.items():
        grouped.setdefault(target.split(".", 1)[0], set()).update(files)
    rows = []
    for top, files in grouped.items():
        distribution, present = provider(top, installed)
        rows.append(GuardRow(top, distribution, present, len(files)))
    return sorted(rows, key=lambda row: row.module.lower())


def violations(scan: Scan, rows: list[GuardRow]) -> list[str]:
    """What must be fixed or declared before the ledger can pass."""
    problems = []
    for row in rows:
        if not row.installed and row.module not in ALLOWED_UNINSTALLED:
            problems.append(
                f"tests guard on `{row.module}` ({row.files} file(s)), but no extra the full "
                f"lane installs ({', '.join(FULL_LANE_EXTRAS)}) provides `{row.distribution}`: "
                "those tests skip in every lane. Add the distribution to an extra, or declare "
                "it in ALLOWED_UNINSTALLED with the reason."
            )
    for location in scan.dynamic_guards:
        problems.append(
            f"{location}: importorskip() is called with a computed name; write the module "
            "name as a literal so the ledger can account for it."
        )
    for parked in scan.parked:
        if parked.key not in ALLOWED_PARKED:
            what = "an expected failure" if parked.kind == "xfail" else "unconditionally skipped"
            problems.append(
                f"{parked.path}:{parked.line} `{parked.name}` is {what}. Give the skip a "
                "condition, fix the test, or declare it in ALLOWED_PARKED with the reason."
            )
    declared = {row.module for row in rows if not row.installed}
    for name in sorted(set(ALLOWED_UNINSTALLED) - declared):
        problems.append(f"ALLOWED_UNINSTALLED lists `{name}`, which no longer needs it; remove it.")
    live = {parked.key for parked in scan.parked}
    for key in sorted(set(ALLOWED_PARKED) - live):
        problems.append(f"ALLOWED_PARKED lists `{key}`, which is no longer parked; remove it.")
    return problems


def render_block(scan: Scan, rows: list[GuardRow]) -> str:
    """Render the gated block: stable facts only, no volatile totals."""
    lines = [
        LEDGER_START,
        "",
        "_Generated by `python scripts/measure_skip_ledger.py --update docs/SKIP_LEDGER.md` "
        "and gated in CI (`--check docs/SKIP_LEDGER.md`); do not hand-edit between the "
        "markers._",
        "",
        "### Tests that are switched off",
        "",
    ]
    if scan.parked:
        lines += ["| Test | Kind | Reason given | Declared because |", "|---|---|---|---|"]
        for parked in scan.parked:
            declared = ALLOWED_PARKED.get(parked.key, "**not declared**")
            lines.append(
                f"| `{parked.path}::{parked.name}` | {parked.kind} | "
                f"{parked.reason or '(none)'} | {declared} |"
            )
    else:
        lines.append(
            "None. No test carries `@pytest.mark.skip`, an unconditional `pytest.skip(...)` "
            "or `@pytest.mark.xfail`."
        )
    lines += [
        "",
        "### Dependencies that guard tests",
        "",
        'A test behind `pytest.importorskip("x")` runs only where `x` is installed. '
        f"\"Full lane\" is `uv sync --locked {' '.join('--extra ' + e for e in FULL_LANE_EXTRAS)}`.",
        "",
        "| Import | Provided by | Installed by the full lane |",
        "|---|---|---|",
    ]
    for row in rows:
        if row.installed:
            verdict = "yes"
        else:
            verdict = "**no** — " + ALLOWED_UNINSTALLED.get(row.module, "**not declared**")
        lines.append(f"| `{row.module}` | {row.distribution} | {verdict} |")
    missing = [row.module for row in rows if not row.installed]
    lines += [
        "",
        (
            "Every guard above is on a dependency the full lane installs, so every guarded "
            "test runs in at least one lane."
            if not missing
            else "Guards the full lane cannot satisfy: "
            + ", ".join(f"`{m}`" for m in missing)
            + "."
        ),
        "",
        LEDGER_END,
    ]
    return "\n".join(lines)


def splice(text: str, block: str) -> str | None:
    """Replace the marked block in ``text``; ``None`` when the markers are absent."""
    start, end = text.find(LEDGER_START), text.find(LEDGER_END)
    if start == -1 or end == -1 or end < start:
        return None
    return text[:start] + block + text[end + len(LEDGER_END) :]


def current_block(text: str) -> str | None:
    """The marked block as it stands in ``text``."""
    start, end = text.find(LEDGER_START), text.find(LEDGER_END)
    if start == -1 or end == -1 or end < start:
        return None
    return text[start : end + len(LEDGER_END)]


def _report(scan: Scan, rows: list[GuardRow], problems: list[str]) -> str:
    out = [
        f"{scan.files} test files scanned",
        f"  importorskip guards:        {sum(row.files for row in rows)} file-guards on "
        f"{len(rows)} imports",
        f"  conditional pytest.skip:    {scan.conditional_skips}",
        f"  module-level pytest.skip:   {scan.module_skips}",
        f"  skipif marks:               {scan.skipif_marks}",
        f"  switched off (skip/xfail):  {len(scan.parked)}",
        "",
        f"{'import':<22} {'distribution':<24} {'full lane':<10} files",
    ]
    for row in rows:
        out.append(
            f"{row.module:<22} {row.distribution:<24} "
            f"{'yes' if row.installed else 'NO':<10} {row.files}"
        )
    out.append("")
    out += [f"VIOLATION: {problem}" for problem in problems] or ["No violations."]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    """Report, update or check the ledger."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--json", action="store_true", help="print the measurement as JSON")
    group.add_argument("--update", metavar="FILE", help="rewrite the marked block in FILE")
    group.add_argument("--check", metavar="FILE", help="exit 1 if FILE is stale or a rule fails")
    args = parser.parse_args(argv)

    if not TESTS.is_dir() or not LOCK.is_file():
        print(f"ERROR: {TESTS} or {LOCK} is missing", file=sys.stderr)
        return 2
    scan = scan_tests(TESTS)
    rows = guard_rows(scan, full_lane_distributions(LOCK))
    problems = violations(scan, rows)
    block = render_block(scan, rows)

    if args.json:
        payload = {
            "files": scan.files,
            "conditional_skips": scan.conditional_skips,
            "module_skips": scan.module_skips,
            "skipif_marks": scan.skipif_marks,
            "parked": [asdict(parked) for parked in scan.parked],
            "guards": [asdict(row) for row in rows],
            "violations": problems,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if not (args.update or args.check):
        print(_report(scan, rows, problems))
        return 0

    target = Path(args.update or args.check)
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as error:
        print(f"ERROR: cannot read {target}: {error}", file=sys.stderr)
        return 2
    existing = current_block(text)
    if existing is None:
        print(f"ERROR: {target} lacks the {LEDGER_START} … {LEDGER_END} markers", file=sys.stderr)
        return 2

    if args.update:
        updated = splice(text, block)
        assert updated is not None  # the markers were found above
        target.write_text(updated, encoding="utf-8")
        print(f"Updated skip ledger in {target}")
        for problem in problems:
            print(f"VIOLATION: {problem}")
        return 0

    failed = False
    for problem in problems:
        failed = True
        print(f"::error title=Skip ledger::{problem}")
    if existing.strip() != block.strip():
        failed = True
        print(
            f"DRIFT: the skip ledger in {target} is stale. Run "
            f"`python scripts/measure_skip_ledger.py --update {target}`."
        )
        print("---- expected ----")
        print(block)
        print("---- current ----")
        print(existing)
    if failed:
        return 1
    print(f"OK: the skip ledger in {target} is up to date and no test is parked or unreachable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
