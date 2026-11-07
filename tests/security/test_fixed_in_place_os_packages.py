# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The OS packages fixed in place stay fixed, and the scanner exception stays exact.

Debian trixie publishes no fix for seven HIGH CVEs in five library packages
(``zlib1g``, ``libsqlite3-0``, ``libacl1``, ``libsystemd0``, ``libudev1``);
Debian unstable fixes all seven, and each fixed build depends on ``libc6``
alone. The Dockerfile's runtime stage therefore installs those exact builds
from one pinned ``snapshot.debian.org`` timestamp instead of accepting the
CVEs in ``.trivyignore``.

Trivy keys Debian findings on the image's release, and trixie's advisory
carries no fixed version, so it still reports the seven against the fixed
builds. ``trivy-fixed-in-place.rego`` ignores each of them **only at the exact
fixed version**. Measured with the CI-pinned trivy 0.70.0 on 2026-11-06: the
fixed runtime layer scans to 0 CRITICAL/HIGH findings with the ledger and the
policy, and the unmodified trixie layer scans to exactly these eight
package findings with the same two files, so reverting the Dockerfile swap
fails the blocking gate.

That property holds only while three artifacts agree, which is what this
module pins offline:

* every exact ``package=version`` request in the Dockerfile has a policy entry
  at the same version, and every policy entry has such a request (a policy
  entry with no matching install would ignore findings for a version that is
  not shipped; an install with no entry would fail the gate);
* the unstable source is a pinned, signature-verified snapshot at priority 1,
  deleted in the same layer, and the build re-proves each installed version;
* the policy CVEs are not also in the acceptance ledger, and both blocking
  Trivy gates pass the policy alongside the ledger.

It also keeps the one sqlite condition that does not depend on a fix:
CVE-2026-119586 (the ``sqlite3`` shell's ``.recover``) affects the shell
through 3.54.1, so the image must keep not installing the shell.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

import yaml

_REPO = Path(__file__).resolve().parents[2]
_DOCKERFILE = _REPO / "Dockerfile"
_POLICY = _REPO / "trivy-fixed-in-place.rego"
_LEDGER = _REPO / ".trivyignore"
_WORKFLOWS = _REPO / ".github" / "workflows"

#: The snapshot source line the runtime stage writes, with its required options.
_SNAPSHOT_SOURCE = (
    "deb [signed-by=/usr/share/keyrings/debian-archive-keyring.gpg check-valid-until=no] "
    "http://snapshot.debian.org/archive/debian/${DEBIAN_FIX_SNAPSHOT}/ sid main"
)
_SOURCE_FILE = "/etc/apt/sources.list.d/debian-fixed-in-place.list"
_PIN_FILE = "/etc/apt/preferences.d/debian-fixed-in-place"


def _runtime_stage() -> str:
    text = _DOCKERFILE.read_text(encoding="utf-8")
    marker = "AS runtime"
    assert marker in text, "the Dockerfile has no runtime stage"
    return text[text.index(marker) :]


def _runtime_args() -> dict[str, str]:
    return dict(re.findall(r"^ARG\s+([A-Z0-9_]+)=(\S+)\s*$", _runtime_stage(), re.MULTILINE))


def _exact_installs() -> dict[str, str]:
    """``{package: version}`` for every ``"pkg=${VAR}"`` install in the runtime stage."""
    args = _runtime_args()
    installs: dict[str, str] = {}
    for package, var in re.findall(r'"([a-z0-9.+-]+)=\$\{([A-Z0-9_]+)\}"', _runtime_stage()):
        assert var in args, f"{package} is installed at ${{{var}}}, which has no ARG default"
        assert package not in installs, f"{package} is requested twice"
        installs[package] = args[var]
    return installs


def _policy_entries() -> dict[str, tuple[str, frozenset[str]]]:
    """``{package: (version, cves)}`` from the policy's ``fixed_in_place`` map."""
    text = _POLICY.read_text(encoding="utf-8")
    entries: dict[str, tuple[str, frozenset[str]]] = {}
    pattern = re.compile(
        r'"([a-z0-9.+-]+)":\s*\{\s*"version":\s*"([^"]+)",\s*"cves":\s*\{([^}]*)\}',
        re.DOTALL,
    )
    for package, version, body in pattern.findall(text):
        cves = frozenset(re.findall(r'"(CVE-\d{4}-\d+)"', body))
        assert cves, f"policy entry {package} names no CVE"
        entries[package] = (version, cves)
    return entries


def _ledger_cves() -> set[str]:
    return set(
        re.findall(r"^\s*(CVE-\d{4}-\d+)\s", _LEDGER.read_text(encoding="utf-8"), re.MULTILINE)
    )


def test_the_parsers_read_something() -> None:
    """Vacuity guard: an unparseable file must not make every check below pass."""
    assert len(_exact_installs()) >= 5
    assert len(_policy_entries()) >= 5
    assert _ledger_cves()


def test_policy_and_dockerfile_name_the_same_packages_at_the_same_versions() -> None:
    installs = _exact_installs()
    policy = {package: version for package, (version, _cves) in _policy_entries().items()}
    assert policy == installs, (
        "trivy-fixed-in-place.rego and the Dockerfile disagree. A policy version "
        "that is not installed ignores findings for a build the image does not "
        "ship; an installed version the policy does not name fails the gate. "
        f"policy={policy} dockerfile={installs}"
    )


def test_policy_binds_every_exception_to_the_installed_version() -> None:
    """The single ignore rule must compare the exact version and the exact CVE."""
    text = _POLICY.read_text(encoding="utf-8")
    code = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    assert "default ignore := false" in code
    rules = re.findall(r"^ignore\b[^\n]*\{(.*?)^\}", code, re.MULTILINE | re.DOTALL)
    assert len(rules) == 1, f"expected exactly one ignore rule, found {len(rules)}"
    body = rules[0]
    for clause in (
        "entry := fixed_in_place[input.PkgName]",
        "input.InstalledVersion == entry.version",
        "input.VulnerabilityID in entry.cves",
    ):
        assert clause in body, f"the ignore rule no longer requires: {clause}"


def test_policy_cves_are_remediated_not_also_accepted() -> None:
    policy_cves = set().union(*(cves for _version, cves in _policy_entries().values()))
    overlap = sorted(policy_cves & _ledger_cves())
    assert not overlap, (
        f"{overlap} are both fixed in place and accepted in .trivyignore; the "
        "ledger entry would keep the CVE ignored after the fix is reverted"
    )


def test_unstable_source_is_pinned_signed_low_priority_and_removed() -> None:
    stage = _runtime_stage()
    args = _runtime_args()
    assert re.fullmatch(
        r"\d{8}T\d{6}Z", args.get("DEBIAN_FIX_SNAPSHOT", "")
    ), "DEBIAN_FIX_SNAPSHOT must be one snapshot timestamp, so the source never moves"
    assert _SNAPSHOT_SOURCE in stage, "the snapshot source line lost its pin or signature check"
    assert (
        "Pin: release n=sid\\nPin-Priority: 1\\n" in stage
    ), "the unstable source must be pinned at priority 1 so nothing is taken from it implicitly"
    run = next(block for block in stage.split("\nRUN ") if _SOURCE_FILE in block)
    removal = f"rm -f {_SOURCE_FILE} \\\n        {_PIN_FILE}"
    assert removal in run, "the source and its pin must be removed together, in the same layer"
    assert run.index('"zlib1g=${') < run.index(
        removal
    ), "the source must be removed after the install"


def test_build_reproves_every_fixed_version() -> None:
    stage = _runtime_stage()
    args = _runtime_args()
    var_for = {
        package: var for package, var in re.findall(r'"([a-z0-9.+-]+)=\$\{([A-Z0-9_]+)\}"', stage)
    }
    for package in _exact_installs():
        check = f'test "$(dpkg-query -W -f=\'${{Version}}\' {package})" = "${{{var_for[package]}}}"'
        assert check in stage, f"the build does not re-prove the installed {package} version"
        assert var_for[package] in args
    assert f"test ! -e {_SOURCE_FILE}" in stage
    assert f"test ! -e {_PIN_FILE}" in stage


def _trivy_steps() -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(_WORKFLOWS.glob("*.yml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for job_name, job in (doc.get("jobs") or {}).items():
            for step in job.get("steps") or []:
                if "aquasecurity/trivy-action" in str(step.get("uses", "")):
                    found.append((f"{path.name}::{job_name}::{step.get('name')}", step))
    return found


def test_every_blocking_gate_applies_ledger_and_policy_together() -> None:
    blocking = [
        (name, step.get("with") or {})
        for name, step in _trivy_steps()
        if str((step.get("with") or {}).get("ignore-unfixed")).lower() == "false"
    ]
    assert len(blocking) >= 2, f"expected the ci.yml and security.yml gates, found {blocking}"
    for name, inputs in blocking:
        assert inputs.get("trivyignores") == ".trivyignore", name
        assert inputs.get("ignore-policy") == _POLICY.name, (
            f"{name} runs without {_POLICY.name}, so it would fail on the fixed builds "
            "or, worse, invite the CVEs back into the acceptance ledger"
        )


def _apt_package_names(text: str) -> set[str]:
    """Package names an ``apt-get install`` in ``text`` would install, normalised.

    APT accepts qualified names -- ``sqlite3:amd64`` (architecture),
    ``sqlite3=3.46.1-1`` (version), ``sqlite3/trixie`` (release) -- and a quoted
    name installs the same package as a bare one, so qualifiers and quotes are
    stripped before the comparison.
    """
    blocks = re.findall(
        r"apt(?:-get)?\s+install[^&]*?(?=&&|\n\n|RUN |\Z)", text, re.IGNORECASE | re.DOTALL
    )
    installed: set[str] = set()
    for block in blocks:
        body = "\n".join(line.split("#", 1)[0] for line in block.splitlines())
        try:
            tokens = shlex.split(body.replace("\\\n", " "), comments=False, posix=True)
        except ValueError:
            tokens = body.replace("\\", " ").split()
        for token in tokens:
            if token.startswith("-") or token in {"apt-get", "apt", "install", "&&"}:
                continue
            name = re.split(r"[:=/]", token, maxsplit=1)[0]
            if name:
                installed.add(name.lower())
    return installed


def test_qualified_apt_package_names_are_normalised() -> None:
    for qualified in (
        "sqlite3:amd64",
        "sqlite3=3.46.1-1",
        "sqlite3/trixie",
        "SQLite3:arm64=3.46.1",
        '"sqlite3"',
        "'sqlite3=3.46.1-1'",
    ):
        text = f"RUN apt-get install -y --no-install-recommends curl \\\n    {qualified} && true\n"
        assert "sqlite3" in _apt_package_names(text), qualified
    assert _apt_package_names('RUN apt-get install -y "libsqlite3-0=${V}"\n') == {"libsqlite3-0"}


def test_dockerfile_installs_no_sqlite3_shell() -> None:
    """CVE-2026-119586 is in the sqlite3 shell through 3.54.1; the shell stays out."""
    installed = _apt_package_names(_DOCKERFILE.read_text(encoding="utf-8"))
    assert "libsqlite3-0" in installed, "the normaliser no longer sees the fixed library install"
    assert "sqlite3" not in installed, (
        "the Dockerfile installs the sqlite3 command-line shell, which carries "
        "CVE-2026-119586 (.recover command injection) with no fixed version"
    )
