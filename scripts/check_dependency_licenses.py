# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fail when a runtime dependency's license is not known to be AGPL-compatible.

Mercury is AGPL-3.0-or-later, and what it ships -- ``pip install
".[all]"``, the Docker image -- combines it with every dependency in that
closure. This gate walks the runtime requirement closure declared in
``pyproject.toml`` for the requested extras, resolves each installed
distribution's license, and fails on anything it cannot show to be compatible.

A license is resolved from the distribution's own metadata, in order: the PEP
639 ``License-Expression``; a ``License`` field that is a recognised SPDX
expression or alias; the trove classifiers (alternatives, so any compatible
one suffices). Labels that name a family without a version -- "BSD",
"Apache" -- are not guessed at: the distribution's shipped license text
decides (a BSD text with the advertising clause, or an Apache text that is
not version 2.0, is incompatible), and no text means unknown. A distribution
that ships nothing but its own metadata has nothing to license.

Unknown or incompatible licenses fail the gate unless the exact metadata
signature is recorded in ``configs/dependency_license_review.toml``:
``verified`` entries carry the primary evidence the metadata lacks;
``permitted`` entries are covered by an additional permission Mercury's
copyright holder grants under AGPL section 7, in a file the entry names (which
must exist and say so); ``flagged`` entries are known not to be AGPL-compatible
by their metadata, are the owner's decision, and are reported as warnings on
every run -- never silently passed.

Exit status: 0 clean, 1 an unreviewed unknown or incompatible license, 2 the
environment or configuration is unusable (a required distribution is not
installed, an unknown extra, a malformed review file).
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib.metadata as md
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from packaging.licenses import InvalidLicenseExpression, canonicalize_license_expression
from packaging.requirements import Requirement
from packaging.utils import NormalizedName, canonicalize_name

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW = REPO_ROOT / "configs" / "dependency_license_review.toml"

#: SPDX identifiers whose terms permit distributing the combined work under
#: AGPL-3.0-or-later: the FSF's GPLv3-compatible licenses (AGPLv3 section 13
#: makes GPLv3 code combinable; every LGPL version converts to the GPL).
#: GPL-2.0-only, Apache-1.x, BSD-4-Clause, CNRI-Python, EPL and proprietary
#: licenses are deliberately absent.
COMPATIBLE: frozenset[str] = frozenset(
    spdx.lower()
    for spdx in (
        "0BSD",
        "AGPL-3.0-only",
        "AGPL-3.0-or-later",
        "Apache-2.0",
        "Artistic-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "BSL-1.0",
        "CC0-1.0",
        "GPL-2.0-or-later",
        "GPL-3.0-only",
        "GPL-3.0-or-later",
        "HPND",
        "ISC",
        "LGPL-2.0-or-later",
        "LGPL-2.1-only",
        "LGPL-2.1-or-later",
        "LGPL-3.0-only",
        "LGPL-3.0-or-later",
        "MIT",
        "MIT-0",
        "MIT-CMU",
        "MPL-2.0",
        "NCSA",
        "PSF-2.0",
        "Python-2.0",
        "Unicode-3.0",
        "Unicode-DFS-2016",
        "Unlicense",
        "UPL-1.0",
        "X11",
        "Zlib",
    )
)

#: Unversioned family labels, settled by the shipped license text.
BSD_FAMILY = "family:BSD"
APACHE_FAMILY = "family:Apache"
#: Explicitly proprietary.
PROPRIETARY = "LicenseRef-Proprietary"

#: Free-text ``License`` values seen in the wild, mapped to SPDX or a family.
_ALIASES: dict[str, str] = {
    "mit": "MIT",
    "mit license": "MIT",
    "the mit license": "MIT",
    "expat": "MIT",
    "bsd": BSD_FAMILY,
    "bsd license": BSD_FAMILY,
    "bsd 3-clause": "BSD-3-Clause",
    "bsd 3-clause license": "BSD-3-Clause",
    "3-clause bsd license": "BSD-3-Clause",
    "new bsd license": "BSD-3-Clause",
    "bsd 2-clause": "BSD-2-Clause",
    "bsd 2-clause license": "BSD-2-Clause",
    "simplified bsd": "BSD-2-Clause",
    "apache": APACHE_FAMILY,
    "apache license": APACHE_FAMILY,
    "apache software license": APACHE_FAMILY,
    "apache 2": "Apache-2.0",
    "apache 2.0": "Apache-2.0",
    "apache-2": "Apache-2.0",
    "apache 2.0 license": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "apache license, version 2.0": "Apache-2.0",
    "apache license version 2.0": "Apache-2.0",
    "apache software license 2.0": "Apache-2.0",
    "psf": "PSF-2.0",
    "psfl": "PSF-2.0",
    "psf license": "PSF-2.0",
    "python software foundation license": "PSF-2.0",
    "isc license": "ISC",
    "mozilla public license 2.0": "MPL-2.0",
}

#: Trove classifier tails (after the last ``::``), mapped to SPDX or a family.
_CLASSIFIERS: dict[str, str] = {
    "MIT License": "MIT",
    "MIT No Attribution License (MIT-0)": "MIT-0",
    "BSD License": BSD_FAMILY,
    "Apache Software License": APACHE_FAMILY,
    "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "Python Software Foundation License": "PSF-2.0",
    "ISC License (ISCL)": "ISC",
    "The Unlicense (Unlicense)": "Unlicense",
    "zlib/libpng License": "Zlib",
    "Boost Software License 1.0 (BSL-1.0)": "BSL-1.0",
    "Historical Permission Notice and Disclaimer (HPND)": "HPND",
    "Universal Permissive License (UPL)": "UPL-1.0",
    "CC0 1.0 Universal (CC0 1.0) Public Domain Dedication": "CC0-1.0",
    "GNU Lesser General Public License v2 or later (LGPLv2+)": "LGPL-2.0-or-later",
    "GNU Lesser General Public License v3 (LGPLv3)": "LGPL-3.0-only",
    "GNU Lesser General Public License v3 or later (LGPLv3+)": "LGPL-3.0-or-later",
    "GNU General Public License v2 (GPLv2)": "GPL-2.0-only",
    "GNU General Public License v2 or later (GPLv2+)": "GPL-2.0-or-later",
    "GNU General Public License v3 (GPLv3)": "GPL-3.0-only",
    "GNU General Public License v3 or later (GPLv3+)": "GPL-3.0-or-later",
    "GNU Affero General Public License v3": "AGPL-3.0-only",
    "GNU Affero General Public License v3 or later (AGPLv3+)": "AGPL-3.0-or-later",
    "Other/Proprietary License": PROPRIETARY,
}

_LICENSE_FILE = re.compile(r"(^|/)(licen[cs]e|copying|notice)[^/]*$", re.IGNORECASE)
_BSD_ADVERTISING = re.compile(r"all\s+advertising\s+materials\s+mentioning", re.IGNORECASE)
_APACHE_2 = re.compile(r"apache\s+license.{0,80}version\s+2\.0", re.IGNORECASE | re.DOTALL)
_SYMBOL = re.compile(r"\(|\)|[^\s()]+")

COMPATIBLE_VERDICT = "compatible"
INCOMPATIBLE_VERDICT = "incompatible"
UNKNOWN_VERDICT = "unknown"


@dataclass(frozen=True)
class Assessment:
    """The license verdict for one distribution."""

    name: NormalizedName
    version: str
    verdict: str
    signature: str
    detail: str


@dataclass(frozen=True)
class Review:
    """One reviewed entry of the review file."""

    pattern: str
    status: str
    signatures: tuple[str, ...]
    note: str
    permission: str | None = None

    def matches(self, assessment: Assessment) -> bool:
        """Whether this entry covers the distribution's current metadata."""
        return (
            fnmatch.fnmatchcase(assessment.name, self.pattern)
            and assessment.signature in self.signatures
        )


class GateError(Exception):
    """The environment or configuration cannot be checked (exit 2)."""


# --------------------------------------------------------------------------- #
# license resolution
# --------------------------------------------------------------------------- #
def _spdx_compatible(identifier: str) -> bool:
    """Whether one SPDX license identifier is AGPL-compatible."""
    return identifier.lower() in COMPATIBLE


def evaluate_expression(expression: str) -> bool:
    """Whether an SPDX license expression permits AGPL distribution.

    ``AND`` needs every operand compatible, ``OR`` any one; ``WITH`` adds an
    exception, which only ever grants permissions, so the base decides.

    Args:
        expression: A valid SPDX license expression.

    Returns:
        ``True`` when the expression is compatible.

    Raises:
        ValueError: The expression is malformed.
    """
    symbols: list[str] = _SYMBOL.findall(expression)
    position = 0

    def peek() -> str | None:
        return symbols[position] if position < len(symbols) else None

    def take() -> str:
        nonlocal position
        if position >= len(symbols):
            raise ValueError(f"unexpected end of license expression {expression!r}")
        symbol = symbols[position]
        position += 1
        return symbol

    def parse_or() -> bool:
        value = parse_and()
        while peek() == "OR":
            take()
            value = parse_and() or value
        return value

    def parse_and() -> bool:
        value = parse_atom()
        while peek() == "AND":
            take()
            value = parse_atom() and value
        return value

    def parse_atom() -> bool:
        symbol = take()
        if symbol == "(":
            value = parse_or()
            if take() != ")":
                raise ValueError(f"unbalanced parentheses in {expression!r}")
            return value
        if symbol in {")", "AND", "OR", "WITH"}:
            raise ValueError(f"unexpected {symbol!r} in {expression!r}")
        if peek() == "WITH":
            take()
            take()  # the exception id: it only ever adds permissions
        return _spdx_compatible(symbol)

    result = parse_or()
    if position != len(symbols):
        raise ValueError(f"trailing symbols in {expression!r}")
    return result


def _license_texts(dist: md.Distribution) -> list[str]:
    """Return the license and notice texts the distribution ships."""
    texts: list[str] = []
    for path in dist.files or []:
        if _LICENSE_FILE.search(str(path)):
            try:
                raw = Path(str(dist.locate_file(path))).read_bytes()
            except OSError:
                continue
            texts.append(raw.decode("utf-8", errors="replace"))
    return texts


def _ships_no_code(dist: md.Distribution) -> bool:
    """Whether the distribution installs nothing but its own metadata."""
    files = dist.files
    if not files:
        return False
    return all(path.parts and path.parts[0].endswith(".dist-info") for path in files)


def _resolve_family(family: str, dist: md.Distribution) -> tuple[str, str]:
    """Settle an unversioned family label from the shipped license text."""
    texts = _license_texts(dist)
    if not texts:
        return UNKNOWN_VERDICT, f"{family} label and no shipped license text"
    if family == BSD_FAMILY:
        if any(_BSD_ADVERTISING.search(text) for text in texts):
            return INCOMPATIBLE_VERDICT, "BSD text carries the advertising clause (4-clause)"
        return COMPATIBLE_VERDICT, "BSD text without the advertising clause"
    if any(_APACHE_2.search(text) for text in texts):
        return COMPATIBLE_VERDICT, "Apache License, Version 2.0 text"
    return UNKNOWN_VERDICT, "Apache label without an Apache 2.0 text"


def _verdict_for(tokens: Sequence[str], dist: md.Distribution) -> tuple[str, str]:
    """Combine alternative license tokens (any compatible one suffices)."""
    outcomes = []
    for token in tokens:
        if token in (BSD_FAMILY, APACHE_FAMILY):
            outcomes.append(_resolve_family(token, dist))
        elif token == PROPRIETARY:
            outcomes.append((INCOMPATIBLE_VERDICT, "proprietary"))
        else:
            ok = _spdx_compatible(token)
            outcomes.append((COMPATIBLE_VERDICT if ok else INCOMPATIBLE_VERDICT, token))
    for wanted in (COMPATIBLE_VERDICT, UNKNOWN_VERDICT):
        for verdict, detail in outcomes:
            if verdict == wanted:
                return verdict, detail
    return outcomes[0] if outcomes else (UNKNOWN_VERDICT, "no license metadata")


def _as_expression(text: str) -> str | None:
    """Return ``text`` as a canonical SPDX expression, or ``None``."""
    try:
        return str(canonicalize_license_expression(text))
    except InvalidLicenseExpression:
        return None


def assess(dist: md.Distribution) -> Assessment:
    """Resolve one installed distribution's license verdict.

    Args:
        dist: The installed distribution.

    Returns:
        Its :class:`Assessment`.
    """
    meta = dist.metadata
    name = canonicalize_name(meta["Name"])
    version = meta["Version"]

    expression = (meta.get("License-Expression") or "").strip()
    if expression:
        signature = f"License-Expression: {expression}"
        canonical = _as_expression(expression)
        if canonical is None:
            return Assessment(name, version, UNKNOWN_VERDICT, signature, "invalid SPDX expression")
        ok = evaluate_expression(canonical)
        verdict = COMPATIBLE_VERDICT if ok else INCOMPATIBLE_VERDICT
        return Assessment(name, version, verdict, signature, canonical)

    license_field = (meta.get("License") or "").strip()
    first_line = license_field.splitlines()[0].strip() if license_field else ""
    single = bool(first_line) and first_line == license_field and len(first_line) <= 100
    if single and first_line.upper() != "UNKNOWN":
        signature = f"License: {first_line}"
        alias = _ALIASES.get(first_line.lower())
        if alias is not None:
            verdict, detail = _verdict_for([alias], dist)
            return Assessment(name, version, verdict, signature, detail)
        canonical = _as_expression(first_line)
        if canonical is not None:
            ok = evaluate_expression(canonical)
            verdict = COMPATIBLE_VERDICT if ok else INCOMPATIBLE_VERDICT
            return Assessment(name, version, verdict, signature, canonical)

    tails = [
        c.rsplit("::", 1)[-1].strip()
        for c in meta.get_all("Classifier") or []
        if c.startswith("License ::")
    ]
    mapped = [_CLASSIFIERS[t] for t in tails if t in _CLASSIFIERS]
    if mapped:
        signature = "Classifier: " + "; ".join(sorted(tails))
        verdict, detail = _verdict_for(mapped, dist)
        return Assessment(name, version, verdict, signature, detail)

    if _ships_no_code(dist):
        return Assessment(name, version, COMPATIBLE_VERDICT, "none", "ships only its metadata")
    signature = f"License: {first_line}" if single else "none"
    return Assessment(name, version, UNKNOWN_VERDICT, signature, "no recognisable license")


# --------------------------------------------------------------------------- #
# requirement closure
# --------------------------------------------------------------------------- #
def declared_requirements(pyproject: Path, extras: Iterable[str]) -> list[Requirement]:
    """Return the runtime requirements ``pyproject.toml`` declares for ``extras``.

    Self-referencing extras (``mercury-agent-global[ml,api]``) are expanded.

    Raises:
        GateError: An extra is not defined.
    """
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    own = canonicalize_name(project["name"])
    optional = project.get("optional-dependencies", {})
    requirements = [Requirement(raw) for raw in project.get("dependencies", [])]
    pending, seen = list(extras), set()
    while pending:
        extra = pending.pop()
        if extra in seen:
            continue
        seen.add(extra)
        if extra not in optional:
            raise GateError(f"pyproject.toml defines no extra {extra!r}")
        for raw in optional[extra]:
            requirement = Requirement(raw)
            if canonicalize_name(requirement.name) == own:
                pending.extend(requirement.extras)
            else:
                requirements.append(requirement)
    return requirements


def _applies(requirement: Requirement, extras: Iterable[str]) -> bool:
    """Whether a requirement's marker holds here for any of the active extras."""
    if requirement.marker is None:
        return True
    return any(requirement.marker.evaluate({"extra": extra}) for extra in ("", *extras))


def resolve_closure(requirements: Iterable[Requirement]) -> dict[NormalizedName, md.Distribution]:
    """Walk the installed distributions the requirements pull in.

    Raises:
        GateError: A required distribution is not installed.
    """
    closure: dict[NormalizedName, md.Distribution] = {}
    visited: set[tuple[NormalizedName, frozenset[str]]] = set()
    queue: list[tuple[Requirement, frozenset[str]]] = [(r, frozenset()) for r in requirements]
    missing: set[str] = set()
    while queue:
        requirement, parent_extras = queue.pop()
        if not _applies(requirement, parent_extras):
            continue
        name = canonicalize_name(requirement.name)
        key = (name, frozenset(requirement.extras))
        if key in visited:
            continue
        visited.add(key)
        try:
            dist = closure.get(name) or md.distribution(requirement.name)
        except md.PackageNotFoundError:
            missing.add(requirement.name)
            continue
        closure[name] = dist
        for raw in dist.requires or []:
            queue.append((Requirement(raw), frozenset(requirement.extras)))
    if missing:
        raise GateError(
            "required distributions are not installed: "
            + ", ".join(sorted(missing))
            + " -- install the requested extras before running the gate"
        )
    return closure


# --------------------------------------------------------------------------- #
# review file and report
# --------------------------------------------------------------------------- #
def load_reviews(path: Path, root: Path = REPO_ROOT) -> list[Review]:
    """Load the reviewed entries.

    Args:
        path: The review file.
        root: The project root that ``permitted`` entries' permission files
            are relative to.

    Raises:
        GateError: The file is malformed, or a ``permitted`` entry names a
            permission file that is missing or is not a section 7 grant.
    """
    if not path.is_file():
        return []
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    reviews: list[Review] = []
    for index, entry in enumerate(data.get("review", [])):
        try:
            status = entry["status"]
            review = Review(
                pattern=(
                    entry["name"].lower()
                    if "*" in entry["name"]
                    else canonicalize_name(entry["name"])
                ),
                status=status,
                signatures=tuple(entry["signatures"]),
                note=entry["evidence"] if status == "verified" else entry["reason"],
                permission=entry["permission"] if status == "permitted" else None,
            )
        except KeyError as exc:
            raise GateError(f"{path.name}: review entry {index} lacks {exc}") from None
        if review.status not in ("verified", "permitted", "flagged") or not review.signatures:
            raise GateError(
                f"{path.name}: review entry {index} has an invalid status or signatures"
            )
        if review.permission is not None:
            grant = (root / review.permission).resolve()
            inside = grant.is_relative_to(root.resolve())
            if (
                not inside
                or not grant.is_file()
                or "section 7" not in grant.read_text(encoding="utf-8")
            ):
                raise GateError(
                    f"{path.name}: review entry {index} names {review.permission!r}, "
                    "which is not a section 7 permission file in the project"
                )
        reviews.append(review)
    return reviews


def run(extras: Sequence[str], pyproject: Path, review_path: Path, *, verbose: bool) -> int:
    """Run the gate and print its report.

    Returns:
        The exit status.
    """
    closure = resolve_closure(declared_requirements(pyproject, extras))
    own = canonicalize_name(tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["name"])
    reviews = load_reviews(review_path, pyproject.parent)
    assessments = [assess(dist) for name, dist in sorted(closure.items()) if name != own]

    failed: list[Assessment] = []
    flagged: list[tuple[Assessment, Review]] = []
    permitted: list[tuple[Assessment, Review]] = []
    verified: list[tuple[Assessment, Review]] = []
    for assessment in assessments:
        if verbose:
            print(
                f"  {assessment.name} {assessment.version}: {assessment.verdict} "
                f"[{assessment.signature}] {assessment.detail}"
            )
        if assessment.verdict == COMPATIBLE_VERDICT:
            continue
        review = next((r for r in reviews if r.matches(assessment)), None)
        if review is None:
            failed.append(assessment)
        elif review.status == "flagged":
            flagged.append((assessment, review))
        elif review.status == "permitted":
            permitted.append((assessment, review))
        else:
            verified.append((assessment, review))

    reviewed = len(flagged) + len(permitted) + len(verified)
    compatible = len(assessments) - len(failed) - reviewed
    print(
        f"Dependency license gate: {len(assessments)} distributions in the runtime "
        f"closure of [{', '.join(extras) or 'core'}] -- compatible {compatible}, "
        f"verified by review {len(verified)}, permitted by section 7 {len(permitted)}, "
        f"flagged {len(flagged)}, failed {len(failed)}"
    )
    for review in reviews:
        granted = [a for a, r in permitted if r is review]
        if granted:
            print(
                f"permitted (AGPL section 7, {review.permission}) -- {review.pattern}: "
                f"{len(granted)} distribution(s)"
            )
        hits = [a for a, r in flagged if r is review]
        if hits:
            names = ", ".join(f"{a.name} {a.version}" for a in hits)
            print(f"WARNING flagged (owner decision) -- {review.pattern}: {names}")
            print(f"        {review.note}")
    for assessment in failed:
        stale = [r for r in reviews if fnmatch.fnmatchcase(assessment.name, r.pattern)]
        hint = " (a review entry exists but no longer matches its metadata)" if stale else ""
        print(
            f"FAIL {assessment.name} {assessment.version}: {assessment.verdict} "
            f"[{assessment.signature}] {assessment.detail}{hint}"
        )
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--extras",
        default="",
        help="comma-separated extras whose runtime closure to check (e.g. 'all')",
    )
    parser.add_argument("--pyproject", type=Path, default=REPO_ROOT / "pyproject.toml")
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--verbose", action="store_true", help="print every distribution")
    args = parser.parse_args(argv)
    extras = [e.strip() for e in args.extras.split(",") if e.strip()]
    try:
        return run(extras, args.pyproject, args.review, verbose=args.verbose)
    except GateError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
