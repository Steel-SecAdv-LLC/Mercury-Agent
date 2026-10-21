#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Decide, before any upload, whether a built distribution can go to PyPI.

A ``PYPI_TOKEN`` (or a trusted publisher) only authenticates the upload. PyPI
still refuses it when any of the following holds, and each one was true of
this repository's release pipeline when the check was written:

1. **The project name belongs to someone else.** The distribution is named
   ``mercury-agent-global`` because ``mercury-agent`` on PyPI is an unrelated
   project (v0.1.10, a different author and homepage): an upload under that
   name is rejected for lack of ownership, and a user following a bare
   ``pip install mercury-agent`` gets the other project's code. The check
   guards the chosen name the same way against future collisions.
2. **A direct URL dependency in the metadata.** PyPI rejects any
   ``Requires-Dist`` of the form ``name @ url`` ("Can't have direct
   dependency"), extras included. The ``[pqc]`` extra pins AMA Cryptography
   as ``ama-cryptography @ git+https://...``, and AMA has no PyPI release.
3. **Non-distribution files in the upload directory.** The release job writes
   ``SBOM.spdx.json`` and ``provenance.json`` next to the wheel; twine refuses
   an unknown distribution format and the whole upload fails.

Checks 2 and 3 are offline. Check 1 queries the PyPI JSON API and is skipped
with ``--offline``. A name is accepted when PyPI has no project by that name
(the first upload creates it) or when one of the existing project's URLs names
this repository.

Exit status: 0 publishable, 1 not publishable (reasons on stderr, plus GitHub
``::error``/``::warning`` annotations with ``--github``), 2 usage error.

Usage::

    python scripts/check_pypi_publishable.py dist/
    python scripts/check_pypi_publishable.py dist/ --offline --json
    python scripts/check_pypi_publishable.py dist/ --stage pypi-dist/
"""

from __future__ import annotations

import argparse
import email.parser
import json
import re
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The repository the published project must point back to.
DEFAULT_REPOSITORY = "Steel-SecAdv-LLC/Mercury-Agent"
PYPI_JSON = "https://pypi.org/pypi/{name}/json"
_DIST_SUFFIXES = (".whl", ".tar.gz")
#: ``name @ url`` (PEP 508 direct reference), with or without extras.
_DIRECT_REFERENCE = re.compile(r"^\s*[A-Za-z0-9][A-Za-z0-9._-]*(\[[^\]]*\])?\s*@")


@dataclass
class Report:
    """Outcome of the preflight; ``publishable`` is true only without errors."""

    name: str | None = None
    version: str | None = None
    distributions: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    index_checked: bool = False

    @property
    def publishable(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "publishable": self.publishable,
            "name": self.name,
            "version": self.version,
            "distributions": self.distributions,
            "index_checked": self.index_checked,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def normalize(name: str) -> str:
    """PEP 503 normalised project name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def is_distribution(path: Path) -> bool:
    return path.is_file() and path.name.endswith(_DIST_SUFFIXES)


def read_metadata(path: Path) -> email.message.Message:
    """Return the core metadata of a wheel (``METADATA``) or sdist (``PKG-INFO``)."""
    parser = email.parser.Parser()
    if path.name.endswith(".whl"):
        with zipfile.ZipFile(path) as archive:
            members = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
            if len(members) != 1:
                raise ValueError(f"{path.name}: expected one .dist-info/METADATA, found {members}")
            return parser.parsestr(archive.read(members[0]).decode("utf-8"))
    with tarfile.open(path, "r:gz") as archive:
        # The top-level PKG-INFO is the sdist's own; nested ones (e.g. an
        # egg-info copy) carry the same fields, so the shortest path is taken.
        infos = sorted(
            (m for m in archive.getmembers() if m.isfile() and m.name.endswith("/PKG-INFO")),
            key=lambda m: m.name.count("/"),
        )
        if not infos:
            raise ValueError(f"{path.name}: no PKG-INFO")
        handle = archive.extractfile(infos[0])
        if handle is None:
            raise ValueError(f"{path.name}: unreadable PKG-INFO")
        return parser.parsestr(handle.read().decode("utf-8"))


def direct_references(metadata: email.message.Message) -> list[str]:
    return [req for req in metadata.get_all("Requires-Dist") or [] if _DIRECT_REFERENCE.match(req)]


def check_distributions(directory: Path, report: Report) -> list[Path]:
    """Offline checks 2 and 3. Returns the distribution files found."""
    if not directory.is_dir():
        report.errors.append(f"{directory} is not a directory")
        return []
    entries = sorted(p for p in directory.iterdir() if p.is_file())
    dists = [p for p in entries if is_distribution(p)]
    others = [p.name for p in entries if not is_distribution(p)]
    if not dists:
        report.errors.append(f"no wheel or sdist in {directory}")
        return []
    if others:
        # Not fatal when the caller stages the distributions separately
        # (``--stage``); the caller decides by reading the warning.
        report.warnings.append(
            "non-distribution files would be rejected by twine if uploaded from this "
            f"directory: {', '.join(others)} (upload a staged copy holding only wheels/sdists)"
        )
    names: set[str] = set()
    versions: set[str] = set()
    for dist in dists:
        report.distributions.append(dist.name)
        try:
            metadata = read_metadata(dist)
        except (OSError, ValueError, zipfile.BadZipFile, tarfile.TarError) as exc:
            report.errors.append(f"{dist.name}: unreadable metadata ({exc})")
            continue
        names.add(str(metadata.get("Name", "")))
        versions.add(str(metadata.get("Version", "")))
        for req in direct_references(metadata):
            report.errors.append(
                f"{dist.name}: PyPI rejects direct URL dependencies; found "
                f"'Requires-Dist: {req}'. Publish the dependency to an index, or drop it "
                "from the metadata of the PyPI build."
            )
    if len({normalize(n) for n in names}) > 1 or len(versions) > 1:
        report.errors.append(f"distributions disagree on name/version: {names} {versions}")
    report.name = sorted(names)[0] if names else None
    report.version = sorted(versions)[0] if versions else None
    return dists


def _urls_of(info: dict[str, Any]) -> list[str]:
    urls = [str(v) for v in (info.get("project_urls") or {}).values() if v]
    for key in ("home_page", "project_url", "download_url"):
        if info.get(key):
            urls.append(str(info[key]))
    return urls


def check_index(
    report: Report,
    *,
    repository: str,
    fetch: Any = None,
    timeout: float = 20.0,
) -> None:
    """Check 1: the name on PyPI is free, or is already this repository's."""
    if not report.name:
        return
    fetch = fetch or _fetch_json
    url = PYPI_JSON.format(name=normalize(report.name))
    try:
        status, payload = fetch(url, timeout)
    except OSError as exc:
        report.errors.append(f"could not query {url}: {exc}")
        return
    report.index_checked = True
    if status == 404:
        return
    if status != 200 or not isinstance(payload, dict):
        report.errors.append(f"unexpected response from {url}: HTTP {status}")
        return
    info = payload.get("info") or {}
    needle = repository.lower().rstrip("/")
    if any(needle in u.lower() for u in _urls_of(info)):
        releases = payload.get("releases") or {}
        if report.version and report.version in releases:
            report.errors.append(f"{report.name} {report.version} is already on PyPI")
        return
    report.errors.append(
        f"the PyPI name '{info.get('name', report.name)}' belongs to another project "
        f"(latest {info.get('version')!s}, author {info.get('author')!r}, "
        f"urls {_urls_of(info) or 'none'}); none of its URLs name {repository}. "
        "Choose an unclaimed distribution name in pyproject.toml, or obtain ownership."
    )


def _fetch_json(url: str, timeout: float) -> tuple[int, Any]:
    if not url.startswith("https://pypi.org/"):
        raise OSError(f"refusing non-PyPI URL {url!r}")
    # Scheme and host are pinned by the check above.
    request = urllib.request.Request(url, headers={"Accept": "application/json"})  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return int(response.status), json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return int(exc.code), None


def stage(dists: list[Path], target: Path) -> None:
    """Copy only the distributions into ``target`` (created empty)."""
    target.mkdir(parents=True, exist_ok=True)
    for stale in target.iterdir():
        if stale.is_file():
            stale.unlink()
    for dist in dists:
        shutil.copy2(dist, target / dist.name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path, help="directory holding the built dists")
    parser.add_argument("--offline", action="store_true", help="skip the PyPI name check")
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--stage", type=Path, help="copy only wheels/sdists here")
    parser.add_argument("--json", action="store_true", help="machine-readable report")
    parser.add_argument("--github", action="store_true", help="emit Actions annotations")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = Report()
    dists = check_distributions(args.directory, report)
    if not args.offline:
        check_index(report, repository=args.repository)
    if args.stage and dists:
        stage(dists, args.stage)
        # The staged directory holds distributions only, so the
        # non-distribution warning no longer applies to what is uploaded.
        report.warnings = [w for w in report.warnings if "non-distribution" not in w]
    if args.github:
        for message in report.errors:
            print(f"::error title=PyPI preflight::{message}")
        for message in report.warnings:
            print(f"::warning title=PyPI preflight::{message}")
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        verdict = "publishable" if report.publishable else "NOT publishable"
        print(f"{report.name} {report.version}: {verdict}")
        for message in report.errors:
            print(f"  error: {message}", file=sys.stderr)
        for message in report.warnings:
            print(f"  warning: {message}", file=sys.stderr)
    return 0 if report.publishable else 1


if __name__ == "__main__":
    sys.exit(main())
