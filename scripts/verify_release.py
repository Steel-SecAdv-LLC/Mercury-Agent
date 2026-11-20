#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Verify a Mercury release locally: provenance, SBOMs and image against the tag.

The release workflow publishes signed provenance, two SBOMs and a pushed
image. A claim about them is only as good as someone else's ability to check
it without waiting for, or trusting, the workflow. This script is that check,
runnable by anyone against the published release:

1. **Artifacts.** The release carries a wheel and an sdist whose filenames
   name the tag's version.
2. **Self-authored provenance** (``provenance.json``). Its subjects are
   exactly the dist files — no file missing, none extra — each with the sha256
   of the file on disk, and its ``releaseVersion`` is the tag's version.
3. **Artifact SBOM** (``SBOM.spdx.json``). The same per-file checksum and
   version agreement.
4. **Dependency SBOM** (``SBOM-api-closure.cdx.json``). A CycloneDX document
   whose root component is linked and is ``mercury-agent-global`` at the tag's
   version.
5. **Sigstore provenance.** ``gh attestation verify`` on every dist file,
   enforcing this repository, the tag's source ref, and ``release.yml`` as the
   signing workflow.
6. **Image** (with ``--image``). The tagged image resolves to a digest and
   carries a provenance attestation (``docker buildx imagetools inspect``).

Every check reports PASS, FAIL or UNVERIFIED. A check whose tool is missing
(``gh``, ``docker``) is UNVERIFIED, never a pass: a tool that could not run
must not look like a clean result.

Exit status: 0 every requested check passed; 1 at least one check FAILED;
2 nothing failed but at least one check could not be verified.

Which releases can pass: only those cut by a ``release.yml`` that includes
the Sigstore attestation and dependency-SBOM steps (added after v3.0.0).
Every earlier release (v3.0.0, 2.0.0, v1.1) was never signed and carries no
CycloneDX SBOM, so the ``sigstore`` and ``SBOM-api-closure.cdx.json`` checks
FAIL on it — truthfully: the claim does not hold for those artifacts, and the
tool reports that rather than passing them.

A locally rebuilt or substituted artifact fails ``sigstore`` too: its digest
has no attestation (GitHub answers 404), which is the point of the check.

Usage::

    # fetch the release assets and verify everything but the image
    python scripts/verify_release.py v3.1.0 --download
    # verify assets already on disk, plus the pushed image
    python scripts/verify_release.py v3.1.0 --assets ./release-v3.1.0 --image ghcr.io/steel-secadv-llc/mercury-agent
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = "Steel-SecAdv-LLC/Mercury-Agent"
DISTRIBUTION = "mercury-agent-global"
SIGNER_WORKFLOW = f"{REPO}/.github/workflows/release.yml"
_DIST_SUFFIXES = (".whl", ".tar.gz")
_WHEEL_RE = re.compile(r"^(?P<dist>[A-Za-z0-9_.]+)-(?P<version>[^-]+)-.+\.whl$")
_SDIST_RE = re.compile(r"^(?P<dist>[A-Za-z0-9_.]+)-(?P<version>.+)\.tar\.gz$")

PASS, FAIL, UNVERIFIED = "PASS", "FAIL", "UNVERIFIED"

#: ``run(argv) -> (returncode, stdout, stderr)``; injectable for tests.
Runner = Callable[[Sequence[str]], tuple[int, str, str]]


@dataclass(frozen=True)
class CheckResult:
    """One verification outcome."""

    name: str
    status: str
    detail: str


def _run(argv: Sequence[str]) -> tuple[int, str, str]:
    completed = subprocess.run(  # noqa: S603 - argv list, no shell
        list(argv), capture_output=True, text=True, timeout=300, check=False
    )
    return completed.returncode, completed.stdout, completed.stderr


def version_of(tag: str) -> str:
    """The release version a tag names (``v3.1.0`` and ``3.1.0`` both name 3.1.0)."""
    return tag[1:] if tag.startswith("v") else tag


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dist_files(assets: Path) -> list[Path]:
    """The wheel and sdist files among the release assets, sorted."""
    return sorted(p for p in assets.iterdir() if p.is_file() and p.name.endswith(_DIST_SUFFIXES))


def check_artifacts(assets: Path, version: str) -> CheckResult:
    """A wheel and an sdist exist, and both filenames name the tag's version."""
    files = dist_files(assets)
    wheels = [p for p in files if p.name.endswith(".whl")]
    sdists = [p for p in files if p.name.endswith(".tar.gz")]
    if not wheels or not sdists:
        return CheckResult(
            "artifacts",
            FAIL,
            f"expected a wheel and an sdist; found {[p.name for p in files] or 'none'}",
        )
    wrong = []
    for path in files:
        match = (_WHEEL_RE if path.name.endswith(".whl") else _SDIST_RE).match(path.name)
        if match is None or match.group("version") != version:
            wrong.append(path.name)
    if wrong:
        return CheckResult("artifacts", FAIL, f"version is not {version} in: {wrong}")
    return CheckResult(
        "artifacts", PASS, f"{len(wheels)} wheel(s), {len(sdists)} sdist(s) at {version}"
    )


def _load_json(path: Path) -> tuple[Any, str | None]:
    if not path.is_file():
        return None, f"{path.name} is missing"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except ValueError as exc:
        return None, f"{path.name} is not valid JSON ({exc})"


def check_self_authored_provenance(assets: Path, version: str) -> CheckResult:
    """``provenance.json`` subjects equal the dist files, digest for digest."""
    doc, problem = _load_json(assets / "provenance.json")
    if problem:
        return CheckResult("provenance.json", FAIL, problem)
    actual = {p.name: _sha256(p) for p in dist_files(assets)}
    try:
        declared = {s["name"]: s["digest"]["sha256"] for s in doc["subject"]}
        released = doc["predicate"]["buildDefinition"]["externalParameters"]["releaseVersion"]
    except (KeyError, TypeError) as exc:
        return CheckResult("provenance.json", FAIL, f"malformed statement (missing {exc})")
    if set(declared) != set(actual):
        return CheckResult(
            "provenance.json",
            FAIL,
            f"subjects {sorted(declared)} are not the dist files {sorted(actual)}",
        )
    mismatched = [name for name in actual if declared[name] != actual[name]]
    if mismatched:
        return CheckResult("provenance.json", FAIL, f"sha256 mismatch for {mismatched}")
    if released != version:
        return CheckResult("provenance.json", FAIL, f"releaseVersion {released!r} != {version!r}")
    return CheckResult("provenance.json", PASS, f"{len(actual)} subjects match, version {version}")


def check_spdx(assets: Path, version: str) -> CheckResult:
    """``SBOM.spdx.json`` checksums and versions equal the dist files."""
    doc, problem = _load_json(assets / "SBOM.spdx.json")
    if problem:
        return CheckResult("SBOM.spdx.json", FAIL, problem)
    actual = {p.name: _sha256(p) for p in dist_files(assets)}
    try:
        packages = {pkg["name"]: pkg for pkg in doc["packages"]}
        declared = {
            name: next(c["checksumValue"] for c in pkg["checksums"] if c["algorithm"] == "SHA256")
            for name, pkg in packages.items()
        }
    except (KeyError, TypeError, StopIteration) as exc:
        return CheckResult("SBOM.spdx.json", FAIL, f"malformed document ({exc!r})")
    if set(declared) != set(actual):
        return CheckResult(
            "SBOM.spdx.json", FAIL, f"packages {sorted(declared)} are not {sorted(actual)}"
        )
    mismatched = [name for name in actual if declared[name] != actual[name]]
    if mismatched:
        return CheckResult("SBOM.spdx.json", FAIL, f"sha256 mismatch for {mismatched}")
    off = [n for n, pkg in packages.items() if pkg.get("versionInfo") != version]
    if off:
        return CheckResult("SBOM.spdx.json", FAIL, f"versionInfo is not {version} for {off}")
    return CheckResult("SBOM.spdx.json", PASS, f"{len(actual)} packages match, version {version}")


def check_cyclonedx(assets: Path, version: str) -> CheckResult:
    """The dependency SBOM is CycloneDX with a linked, correctly versioned root."""
    doc, problem = _load_json(assets / "SBOM-api-closure.cdx.json")
    if problem:
        return CheckResult("SBOM-api-closure.cdx.json", FAIL, problem)
    if not isinstance(doc, dict) or doc.get("bomFormat") != "CycloneDX":
        return CheckResult("SBOM-api-closure.cdx.json", FAIL, "not a CycloneDX document")
    components = doc.get("components") or []
    if not components:
        return CheckResult("SBOM-api-closure.cdx.json", FAIL, "no components")
    root = (doc.get("metadata") or {}).get("component") or {}
    root_ref = root.get("bom-ref")
    if not root_ref:
        return CheckResult(
            "SBOM-api-closure.cdx.json",
            FAIL,
            f"root component {root.get('name')!r} is unlinked (bom-ref null): the SBOM "
            "does not identify the product",
        )
    linked = [c for c in components if c.get("bom-ref") == root_ref]
    if not linked:
        return CheckResult(
            "SBOM-api-closure.cdx.json", FAIL, f"root bom-ref {root_ref!r} names no component"
        )
    product = linked[0]
    if product.get("name") != DISTRIBUTION or product.get("version") != version:
        return CheckResult(
            "SBOM-api-closure.cdx.json",
            FAIL,
            f"root is {product.get('name')}@{product.get('version')}, "
            f"not {DISTRIBUTION}@{version}",
        )
    return CheckResult(
        "SBOM-api-closure.cdx.json",
        PASS,
        f"{len(components)} components; root {DISTRIBUTION}@{version} linked",
    )


def check_sigstore(assets: Path, tag: str, run: Runner = _run) -> CheckResult:
    """``gh attestation verify`` every dist file against this repo, tag and workflow."""
    if shutil.which("gh") is None:
        return CheckResult("sigstore", UNVERIFIED, "gh is not installed (https://cli.github.com)")
    files = dist_files(assets)
    if not files:
        return CheckResult("sigstore", FAIL, "no dist files to verify")
    failures = []
    for path in files:
        code, _out, err = run(
            [
                "gh",
                "attestation",
                "verify",
                str(path),
                "--repo",
                REPO,
                "--source-ref",
                f"refs/tags/{tag}",
                "--signer-workflow",
                SIGNER_WORKFLOW,
            ]
        )
        if code != 0:
            failures.append(f"{path.name}: {err.strip().splitlines()[-1] if err.strip() else code}")
    if failures:
        return CheckResult("sigstore", FAIL, "; ".join(failures))
    return CheckResult("sigstore", PASS, f"{len(files)} artifacts verified against {tag}")


def check_image(image: str, version: str, run: Runner = _run) -> CheckResult:
    """The tagged image resolves to a digest and carries a provenance attestation."""
    if shutil.which("docker") is None:
        return CheckResult("image", UNVERIFIED, "docker is not installed")
    ref = f"{image}:{version}"
    code, out, err = run(
        ["docker", "buildx", "imagetools", "inspect", ref, "--format", "{{json .}}"]
    )
    if code != 0:
        return CheckResult("image", FAIL, f"cannot inspect {ref}: {err.strip()[:200]}")
    try:
        info = json.loads(out)
    except ValueError:
        return CheckResult("image", FAIL, f"unreadable inspect output for {ref}")
    digest = ((info.get("manifest") or {}).get("digest")) or info.get("digest")
    if not digest:
        return CheckResult("image", FAIL, f"{ref} resolved to no digest")
    if not info.get("provenance"):
        return CheckResult("image", FAIL, f"{ref} ({digest}) carries no provenance attestation")
    return CheckResult("image", PASS, f"{ref} -> {digest}, provenance attached")


def download(tag: str, dest: Path, run: Runner = _run) -> CheckResult:
    """Fetch the release's assets with ``gh release download``."""
    if shutil.which("gh") is None:
        return CheckResult("download", UNVERIFIED, "gh is not installed")
    dest.mkdir(parents=True, exist_ok=True)
    code, _out, err = run(
        ["gh", "release", "download", tag, "--repo", REPO, "--dir", str(dest), "--clobber"]
    )
    if code != 0:
        return CheckResult("download", FAIL, err.strip()[:200] or f"exit {code}")
    return CheckResult("download", PASS, f"assets in {dest}")


def verdict(results: Sequence[CheckResult]) -> int:
    """0 all pass; 1 any FAIL; 2 no FAIL but something UNVERIFIED."""
    if any(r.status == FAIL for r in results):
        return 1
    if any(r.status == UNVERIFIED for r in results):
        return 2
    return 0


def verify(
    tag: str,
    assets: Path,
    *,
    image: str | None = None,
    run: Runner = _run,
) -> list[CheckResult]:
    """Run every applicable check; never stops at the first failure."""
    version = version_of(tag)
    results = [
        check_artifacts(assets, version),
        check_self_authored_provenance(assets, version),
        check_spdx(assets, version),
        check_cyclonedx(assets, version),
        check_sigstore(assets, tag, run),
    ]
    if image:
        results.append(check_image(image, version, run))
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("tag", help="release tag, e.g. v3.1.0")
    parser.add_argument("--assets", type=Path, help="directory holding the release assets")
    parser.add_argument(
        "--download", action="store_true", help="fetch the assets with gh release download first"
    )
    parser.add_argument("--image", help="image repository to verify, e.g. ghcr.io/owner/name")
    args = parser.parse_args(argv)

    assets = args.assets or Path(f"release-{args.tag}")
    results: list[CheckResult] = []
    if args.download:
        fetched = download(args.tag, assets)
        results.append(fetched)
        if fetched.status != PASS:
            for r in results:
                print(f"{r.status:10s} {r.name:28s} {r.detail}")
            return verdict(results)
    if not assets.is_dir():
        print(f"FAIL       assets                       {assets} is not a directory")
        return 1
    results.extend(verify(args.tag, assets, image=args.image))
    for r in results:
        print(f"{r.status:10s} {r.name:28s} {r.detail}")
    code = verdict(results)
    print(
        {0: "release verified", 1: "release verification FAILED", 2: "release NOT fully verified"}[
            code
        ]
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
