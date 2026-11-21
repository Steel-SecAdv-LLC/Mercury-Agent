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

The forms followed are listed in ``docs/SKIP_LEDGER.md``. The pass is static
and by name: a mark built by a function call, fetched with ``getattr`` or
imported from another module under a new name is not followed.

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

#: Modules whose skip machinery the ledger follows through import aliases.
_SKIP_MODULES = frozenset({"pytest", "unittest"})

_IMPORTORSKIP = frozenset({"pytest.importorskip", "importorskip"})

#: Calls that switch the running test off, and the kind they park it as.
_IMPERATIVE = {
    "pytest.skip": "skip",
    "pytest.xfail": "xfail",
}

#: Exception classes that do the same when raised.
_RAISED = {
    "pytest.skip.Exception": "skip",
    "pytest.xfail.Exception": "xfail",
    "unittest.SkipTest": "skip",
    "unittest.case.SkipTest": "skip",
}

#: ``unittest`` decorators that skip on a condition, and the literal value of
#: that condition which makes the skip unconditional.
_UNITTEST_CONDITIONAL = {"unittest.skipIf": True, "unittest.skipUnless": False}

#: Methods that attach a mark to a collected item or the running test.
_ATTACHING_METHODS = frozenset({"add_marker", "applymarker"})

#: List methods that add to ``pytestmark`` after it was assigned.
_LIST_GROWERS = frozenset({"append", "extend", "insert"})

#: Marks and decorators that switch a test off without a condition.
_MARKS = {
    "pytest.mark.skip": "skip",
    "pytest.mark.xfail": "xfail",
    "unittest.skip": "skip",
    "unittest.expectedFailure": "xfail",
}

_BRANCHING = (
    ast.If,
    ast.Try,
    ast.ExceptHandler,
    ast.While,
    ast.For,
    ast.AsyncFor,
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
    # The function's own statements only: a ``return`` inside a nested helper
    # does not let the test leave, and used to make an unconditional skip
    # after it count as conditional.
    pending: list[ast.AST] = list(ast.iter_child_nodes(function))
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom)) and node.lineno < line:
            return True
        pending.extend(ast.iter_child_nodes(node))
    return False


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


def _import_aliases(tree: ast.Module) -> dict[str, str]:
    """Local names that stand for ``pytest`` / ``unittest`` or one of their members.

    ``import pytest as pt`` maps ``pt`` to ``pytest``; ``from pytest import
    skip, mark`` maps ``skip`` to ``pytest.skip`` and ``mark`` to
    ``pytest.mark``. Without this, ``@mark.skip`` and a bare ``skip(...)``
    were invisible to the ledger.
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _SKIP_MODULES:
                    aliases[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module in _SKIP_MODULES:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def _resolved(node: ast.AST, aliases: dict[str, str]) -> str | None:
    """:func:`_dotted` with the first segment resolved through the file's imports."""
    name = _dotted(node)
    if name is None:
        return None
    head, _, rest = name.partition(".")
    if head in aliases:
        return f"{aliases[head]}.{rest}" if rest else aliases[head]
    return name


def _literal_condition(call: ast.AST | None) -> bool | None:
    """The truth of a mark's condition when it is a constant.

    Returns:
        ``True`` / ``False`` for a constant condition, ``None`` when the
        condition is computed. A mark with no condition at all (``xfail`` with
        only a reason) is ``True``: nothing can switch it back on.
    """
    if not isinstance(call, ast.Call):
        return True
    condition: ast.AST | None = None
    if call.args and not (
        isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str)
    ):
        condition = call.args[0]
    for keyword in call.keywords:
        if keyword.arg == "condition":
            condition = keyword.value
    if condition is None:
        return True
    if isinstance(condition, ast.Constant):
        return bool(condition.value)
    return None


def _in_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return True
    return False


def _inside_raises(node: ast.AST, parents: dict[ast.AST, ast.AST], aliases: dict[str, str]) -> bool:
    """Whether a skip is the subject of a ``with pytest.raises(...)`` check."""
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        if isinstance(current, (ast.With, ast.AsyncWith)):
            for item in current.items:
                expression = item.context_expr
                callee = expression.func if isinstance(expression, ast.Call) else expression
                if _resolved(callee, aliases) == "pytest.raises":
                    return True
    return False


def _owner(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    """Name of the test, class or module-level target a mark is attached to."""
    current = node
    while current in parents:
        parent = parents[current]
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return parent.name
        if isinstance(parent, (ast.Assign, ast.AnnAssign)):
            targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
            names = [target.id for target in targets if isinstance(target, ast.Name)]
            if names:
                scope = _owner(parent, parents)
                return names[0] if scope == "<module>" else f"{scope}.{names[0]}"
        current = parent
    return "<module>"


def _attachment(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str | None:
    """Where a mark expression is attached, if it is.

    Returns:
        ``"static"`` for a place pytest reads marks from at collection: a
        decorator, ``pytestmark`` assigned or grown at module or class level,
        or the ``marks=`` of ``pytest.param``. ``"dynamic"`` for the argument
        of ``add_marker`` / ``applymarker``, which attaches when that line
        runs. ``None`` for a mark that is only a value.
    """
    current = node
    while current in parents:
        parent = parents[current]
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            decorated = any(current is decorator for decorator in parent.decorator_list)
            return "static" if decorated else None
        if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
            named = any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in targets)
            # A local variable that happens to be called ``pytestmark`` is not
            # read by pytest.
            return "static" if named and not _in_function(parent, parents) else None
        if isinstance(parent, ast.keyword) and parent.arg == "marks":
            return "static"
        if isinstance(parent, ast.Call) and current is not parent.func:
            callee = parent.func
            if isinstance(callee, ast.Attribute):
                if callee.attr in _ATTACHING_METHODS:
                    return "dynamic"
                grows = callee.attr in _LIST_GROWERS and _dotted(callee.value) == "pytestmark"
                if grows:
                    return None if _in_function(parent, parents) else "static"
        current = parent
    return None


def _named_marks(
    tree: ast.Module, aliases: dict[str, str], parents: dict[ast.AST, ast.AST]
) -> dict[str, tuple[str, ast.AST]]:
    """Names bound to a mark that switches a test off without a condition.

    ``parked = pytest.mark.skip(reason="...")`` followed by ``@parked`` (or
    ``marks=parked``, ``pytestmark = parked``) parks the test exactly as the
    inline form does, and was invisible. A name bound under a branch is left
    out: which mark it holds is decided at import.
    """
    named: dict[str, tuple[str, ast.AST]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [t.id for t in targets if isinstance(t, ast.Name) and t.id != "pytestmark"]
        if not names or _under_module_branch(node, parents):
            continue
        value = node.value
        callee = value.func if isinstance(value, ast.Call) else value
        resolved = _resolved(callee, aliases)
        kind: str | None = None
        if resolved in _MARKS and _literal_condition(value) is True:
            kind = _MARKS[resolved]
        elif resolved == "pytest.mark.skipif" and _literal_condition(value) is True:
            kind = "skip"
        if kind is not None:
            for name in names:
                named[name] = (kind, value)
    return named


def _is_raised(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """Whether an exception class reference is the subject of a ``raise``."""
    parent = parents.get(node)
    if isinstance(parent, ast.Call) and parent.func is node:
        parent = parents.get(parent)
    return isinstance(parent, ast.Raise)


def _scan_file(path: Path, rel: str, scan: Scan) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
    tables = _module_string_tables(tree)
    aliases = _import_aliases(tree)
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    named = _named_marks(tree, aliases, parents)

    def park(node: ast.AST, kind: str, reason_source: ast.AST) -> None:
        scan.parked.append(
            Parked(
                rel, getattr(node, "lineno", 0), _owner(node, parents), kind, _reason(reason_source)
            )
        )

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Attribute, ast.Name)):
            continue
        name = _resolved(node, aliases)
        if name is None:
            continue
        holder = parents.get(node)
        # The call this name is the callee of, if any (``pytest.mark.skip(...)``).
        call = holder if isinstance(holder, ast.Call) and holder.func is node else None

        if name in _IMPORTORSKIP and call is not None:
            argument: ast.AST | None = call.args[0] if call.args else None
            for keyword in call.keywords:
                if keyword.arg == "modname":
                    argument = keyword.value
            targets = _guard_targets(argument, tables)
            if targets is None:
                scan.dynamic_guards.append(f"{rel}:{node.lineno}")
            else:
                for target in targets:
                    scan.guards.setdefault(target, set()).add(rel)
        elif _imperative_kind(node, name, call, parents) is not None:
            # ``pytest.skip(...)``, ``self.skipTest(...)``, ``pytest.xfail(...)``
            # and a raised skip exception, inside a test or at import.
            if _inside_raises(node, parents, aliases):
                continue
            function, branched = _enclosing(node, parents)
            if function is None:
                scan.module_skips += 1
            elif branched or _returns_before(function, node.lineno):
                scan.conditional_skips += 1
            else:
                kind = _imperative_kind(node, name, call, parents)
                assert kind is not None
                park(node, kind, call if call is not None else node)
        elif name in _MARKS or name in named:
            if isinstance(node, ast.Name) and name in named and name not in _MARKS:
                if not isinstance(node.ctx, ast.Load):
                    continue
                kind, source = named[name]
            else:
                kind, source = _MARKS[name], call if call is not None else node
                condition = _literal_condition(call)
                if condition is False:
                    continue  # ``xfail(False)``: the mark never applies
                if condition is None:
                    scan.skipif_marks += 1
                    continue
            attachment = _attachment(call or node, parents)
            if attachment is None:
                continue
            function, branched = _enclosing(node, parents)
            if attachment == "dynamic" and (
                function is None or branched or _returns_before(function, node.lineno)
            ):
                # A hook that marks some items under a condition.
                scan.conditional_skips += 1
            elif kind == "skip" and _under_module_branch(node, parents):
                scan.conditional_skips += 1
            else:
                park(node, kind, source)
        elif name == "pytest.mark.skipif" or name in _UNITTEST_CONDITIONAL:
            parks_when = _UNITTEST_CONDITIONAL.get(name, True)
            # Here the first positional argument is always the condition. A
            # string is one pytest evaluates at collection, so it is computed,
            # not constant ("sys.platform == 'win32'").
            condition = None
            subject: ast.AST | None = call.args[0] if call is not None and call.args else None
            if call is not None:
                for keyword in call.keywords:
                    if keyword.arg == "condition":
                        subject = keyword.value
            if isinstance(subject, ast.Constant) and not isinstance(subject.value, str):
                condition = bool(subject.value)
            if condition is parks_when and _attachment(call or node, parents) is not None:
                park(node, "skip", call if call is not None else node)
            else:
                scan.skipif_marks += 1


def _imperative_kind(
    node: ast.AST, name: str, call: ast.Call | None, parents: dict[ast.AST, ast.AST]
) -> str | None:
    """The kind a name parks the running test as when called or raised, if it does."""
    if name in _RAISED:
        return _RAISED[name] if _is_raised(node, parents) else None
    if call is None:
        return None
    if name in _IMPERATIVE:
        return _IMPERATIVE[name]
    if isinstance(node, ast.Attribute) and node.attr == "skipTest":
        return "skip"
    return None


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
