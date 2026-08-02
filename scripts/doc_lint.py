# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Documentation lint: ban claims the code cannot back, require the mission line.

A repository's prose is part of its interface. This gate holds three rules that
kept getting broken by hand:

1. **Banned phrases.** Some words assert a status Mercury does not hold (a
   certification claim), and some encode a framing that has been retired (a
   humanitarian slice presented as the whole mission). Run ``--list-rules`` to
   print the exact table. Passing published test vectors is not certification;
   a humanitarian slice is not the mission. Each ban carries the reason and the
   wording to use instead, printed on failure.
2. **The mission is civilization-first.** Every document that states Mercury's
   mission must contain ``civilization-first``. FINDΩYOU is one deployment of
   that mission, named as such, and never the ceiling.
3. **Enforced means tested.** Every row in ``CAPABILITY_MATRIX.md`` whose
   status says *enforced* must carry a repro command that names a path which
   exists in this repository. A status column is where a reader looks to tell a
   shipped guarantee from an aspiration, so a row that claims enforcement
   without pointing at the thing doing the enforcing is the exact failure this
   gate exists to catch. (A blanket regex over the word "enforced" in every
   document was tried and discarded: it fired on sixty ordinary sentences and
   would have taught readers to ignore it.)

A line ending in ``doc-lint: allow`` is skipped by the banned-phrase rule. It
exists so a document can *name* a forbidden phrase in order to forbid it —
used twice, in the README and SECURITY.md passages that explain why
"certified" is the wrong word. It is a visible, greppable exception, not a
silent one.

Run ``python scripts/doc_lint.py`` to check, ``--list-rules`` to print the rule
table. Exit status is 0 when clean, 1 when a violation is found; CI and
``tests/pillars/test_candor.py`` both call it, so the rules mean the same thing
in both places.

The scan deliberately covers this file's own rule table too — a lint that has to
exempt itself is a lint with a hole — so every pattern below is written to match
prose, not the Python string literals that define it (see ``_is_rule_source``).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Directories scanned for prose. Source docstrings count: a claim in a module
#: docstring reaches a reader through ``help()`` exactly as a README does.
SCAN_DIRS: tuple[str, ...] = ("docs", "src", "scripts", "tools", "benchmarks", "examples")

#: Top-level documents scanned individually.
SCAN_FILES: tuple[str, ...] = (
    "README.md",
    "ARCHITECTURE.md",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "CAPABILITY_MATRIX.md",
    "CODE_OF_CONDUCT.md",
    "DEPRECATION.md",
)

#: Extensions considered prose-bearing.
SCAN_SUFFIXES: frozenset[str] = frozenset({".md", ".py", ".rst", ".txt"})

#: Paths excluded from the banned-phrase scan, with the reason. CHANGELOG.md is
#: a historical record: rewriting what a past release said would be dishonest in
#: a different direction, so history is preserved and only current prose is
#: linted.
EXCLUDED: tuple[tuple[str, str], ...] = (
    ("CHANGELOG.md", "historical record; past wording is preserved, not rewritten"),
    ("docs/CAPABILITY_INVENTORY.md", "machine-generated from source by a script"),
    ("tests/", "test corpora quote banned phrases in order to assert they are banned"),
)


@dataclass(frozen=True)
class BannedPhrase:
    """A phrase that must not appear in current prose.

    Attributes:
        pattern: Case-insensitive regex matched against each line.
        label: Human-readable name of the phrase, for the failure message.
        why: Why the phrase is wrong — the part a reader needs to fix it.
        instead: The wording to use in its place.
    """

    pattern: str
    label: str
    why: str
    instead: str


BANNED: tuple[BannedPhrase, ...] = (
    BannedPhrase(
        pattern=r"survivor[\s‐-―-]first",
        label="survivor-first",
        why=(
            "it presents one humanitarian deployment as the mission, which "
            "understates what Mercury is and overstates what that slice covers"
        ),
        instead=(
            "describe the behaviour concretely (e.g. 'prioritises the people "
            "most exposed'), and frame the humanitarian slice as one deployment "
            "of Mercury's civilization-first mission"
        ),
    ),
    BannedPhrase(
        pattern=r"people[\s‐-―-]first",
        label="People First",
        why="it is a slogan, not a statement anything in the code enforces",
        instead="state the specific control or priority rule the code implements",
    ),
    BannedPhrase(
        pattern=r"FIPS[\s‐-―-]certified",
        label="FIPS-certified",
        why=(
            "Mercury's PQC backend has not entered the CMVP validation "
            "programme; passing published KAT vectors is not certification"
        ),
        instead=(
            "'implements FIPS 203/204/205 and passes the ACVP-Server KAT "
            "vectors in CI (tests/security/test_nist_fips_kat.py); not "
            "CAVP/CMVP validated and not independently audited'"
        ),
    ),
    BannedPhrase(
        pattern=r"NIST[\s‐-―-]validated",
        label="NIST-validated",
        why=("'validated' names a formal NIST programme (CAVP/CMVP) Mercury has not been through"),
        instead="'implements the NIST FIPS 203/204/205 standards' plus the KAT evidence",
    ),
)

#: Documents that state the mission and must therefore name it correctly.
MISSION_DOCUMENTS: tuple[str, ...] = (
    "README.md",
    "docs/index.md",
    "src/omni_mercury_engine/__init__.py",
)

#: The mission phrase, matched case-insensitively.
MISSION_PHRASE = r"civilization[\s‐-―-]first"

#: Trailing marker that exempts one line from the banned-phrase rule. Visible
#: and greppable on purpose: an exception a reader cannot see is a hole.
ALLOW_MARKER = "doc-lint: allow"

#: The claims registry. Rows here are the project's enforced-status claims.
CAPABILITY_MATRIX = "CAPABILITY_MATRIX.md"

#: Status values a matrix row may carry. ``enforced`` is the only one that
#: obliges a repro command pointing at real code.
VALID_STATUSES: frozenset[str] = frozenset(
    {"enforced", "measured", "advisory", "untrained", "aspirational", "removed"}
)


@dataclass(frozen=True)
class Violation:
    """One rule breach at one location."""

    path: str
    line_number: int
    rule: str
    line: str
    guidance: str

    def render(self) -> str:
        """Format the violation as a compiler-style, clickable diagnostic."""
        return (
            f"{self.path}:{self.line_number}: {self.rule}\n"
            f"    {self.line.strip()[:160]}\n"
            f"    -> {self.guidance}"
        )


def _is_excluded(relative: str) -> bool:
    return any(relative.startswith(prefix) or relative == prefix for prefix, _ in EXCLUDED)


def _is_rule_source(path: Path) -> bool:
    """True for this module, whose own source defines the banned patterns.

    The rule table necessarily contains the phrases it bans. Excluding only the
    literal ``BANNED``/``MISSION`` definitions -- not the whole file -- keeps the
    lint honest about its own prose while letting it name what it forbids.
    """
    return path.resolve() == Path(__file__).resolve()


def iter_scanned_files() -> list[Path]:
    """Return every prose-bearing file the lint covers, sorted and deduplicated."""
    found: set[Path] = set()
    for name in SCAN_FILES:
        candidate = REPO_ROOT / name
        if candidate.is_file():
            found.add(candidate)
    for directory in SCAN_DIRS:
        root = REPO_ROOT / directory
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in SCAN_SUFFIXES:
                found.add(path)
    return sorted(
        path
        for path in found
        if not _is_excluded(str(path.relative_to(REPO_ROOT))) and "__pycache__" not in path.parts
    )


def _scan_banned(path: Path, text: str) -> list[Violation]:
    relative = str(path.relative_to(REPO_ROOT))
    rule_source = _is_rule_source(path)
    violations: list[Violation] = []
    for number, line in enumerate(text.splitlines(), start=1):
        # Inside this module the rule table itself quotes the banned phrases;
        # only the regex/label definitions are skipped, never its prose.
        if rule_source and ("pattern=r" in line or "label=" in line):
            continue
        if ALLOW_MARKER in line:
            continue
        for banned in BANNED:
            if re.search(banned.pattern, line, re.IGNORECASE):
                violations.append(
                    Violation(
                        path=relative,
                        line_number=number,
                        rule=f"banned phrase {banned.label!r}: {banned.why}",
                        line=line,
                        guidance=f"use {banned.instead}",
                    )
                )
    return violations


def _scan_mission(path: Path, text: str) -> list[Violation]:
    relative = str(path.relative_to(REPO_ROOT))
    if relative not in MISSION_DOCUMENTS:
        return []
    if re.search(MISSION_PHRASE, text, re.IGNORECASE):
        return []
    return [
        Violation(
            path=relative,
            line_number=1,
            rule="missing mission phrase 'civilization-first'",
            line="(document does not state the mission)",
            guidance=(
                "state the civilization-first mission; FINDOMEGAYOU and other "
                "humanitarian deployments are named as deployments of it, not as it"
            ),
        )
    ]


_PATH_RE = re.compile(r"(?:tests|src|scripts|benchmarks|tools|docs)/[\w./-]+")


def _matrix_rows(text: str) -> list[tuple[int, list[str]]]:
    """Return ``(line_number, cells)`` for each pipe-table data row."""
    rows: list[tuple[int, list[str]]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 2:
            continue
        # Skip header separators (``|---|---|``).
        if all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        rows.append((number, cells))
    return rows


def scan_capability_matrix(text: str) -> list[Violation]:
    """Every 'enforced' claim in the matrix must point at code that exists.

    A row's ``status`` is what a reader uses to separate a shipped guarantee
    from an aspiration. Claiming ``enforced`` without a repro command naming a
    real path makes that column decorative.
    """
    violations: list[Violation] = []
    rows = _matrix_rows(text)
    if not rows:
        return [
            Violation(
                path=CAPABILITY_MATRIX,
                line_number=1,
                rule="claims registry is empty",
                line="(no table rows found)",
                guidance="the matrix must list one row per claim",
            )
        ]
    for number, cells in rows:
        # Only claim rows carry a status: ``claim | ... | repro | status``, with
        # the status in the last cell. Two-column tables (the status vocabulary
        # itself, the removed-claims list) describe statuses rather than
        # asserting them, and linting their prose would fire on every definition.
        if len(cells) < 4:
            continue
        status_cell = cells[-1].lower()
        if "enforced" not in status_cell:
            continue
        joined = " ".join(cells)
        referenced = _PATH_RE.findall(joined)
        existing = [ref for ref in referenced if (REPO_ROOT / ref.rstrip(".,;`")).exists()]
        if not existing:
            violations.append(
                Violation(
                    path=CAPABILITY_MATRIX,
                    line_number=number,
                    rule="row claims 'enforced' with no repro command naming real code",
                    line=joined,
                    guidance=(
                        "add a repro command citing an existing test/module path, "
                        "or change the status to what is true (measured / advisory / "
                        "untrained / aspirational)"
                    ),
                )
            )
    return violations


def run() -> list[Violation]:
    """Run every rule over every scanned file and return the violations found."""
    violations: list[Violation] = []
    for path in iter_scanned_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        violations.extend(_scan_banned(path, text))
        violations.extend(_scan_mission(path, text))
    matrix = REPO_ROOT / CAPABILITY_MATRIX
    if matrix.is_file():
        violations.extend(scan_capability_matrix(matrix.read_text(encoding="utf-8")))
    else:
        violations.append(
            Violation(
                path=CAPABILITY_MATRIX,
                line_number=1,
                rule="claims registry is missing",
                line="(file not found)",
                guidance="every enforced-status claim lives in CAPABILITY_MATRIX.md",
            )
        )
    return violations


def _print_rules() -> None:
    print("Banned phrases:")
    for banned in BANNED:
        print(f"  - {banned.label}: {banned.why}")
        print(f"      instead: {banned.instead}")
    print("\nMission phrase required in:")
    for document in MISSION_DOCUMENTS:
        print(f"  - {document}")
    print("\nExcluded from the banned-phrase scan:")
    for prefix, reason in EXCLUDED:
        print(f"  - {prefix}: {reason}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 when clean, 1 when a violation is found."""
    parser = argparse.ArgumentParser(
        prog="python scripts/doc_lint.py",
        description="Lint documentation prose for unbacked claims and retired framing.",
    )
    parser.add_argument("--list-rules", action="store_true", help="Print the rule table and exit.")
    args = parser.parse_args(argv)

    if args.list_rules:
        _print_rules()
        return 0

    violations = run()
    if not violations:
        print(f"doc-lint: clean ({len(iter_scanned_files())} files scanned)")
        return 0

    print(f"doc-lint: {len(violations)} violation(s)\n", file=sys.stderr)
    for violation in violations:
        print(violation.render(), file=sys.stderr)
        print(file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
