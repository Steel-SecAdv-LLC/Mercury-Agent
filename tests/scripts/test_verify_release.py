# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``scripts/verify_release.py`` checks a release locally, and agrees with the workflow.

The strongest test here runs the release workflow's OWN provenance/SBOM
generator — extracted verbatim from ``.github/workflows/release.yml`` — over a
dist directory that also holds the dependency SBOM, then verifies the result.
That pins two things at once: the generator's subjects are exactly the dist
files (it used to promote every other file in ``dist/`` to a build subject, so
the dependency SBOM would have been listed as an artifact — this test fails on
that prior exclusion-list generator), and the generator and the verifier read
the same shape, so a future change to one that breaks the other fails here.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(_ROOT))

from scripts import verify_release as vr

VERSION = "3.1.0"
TAG = f"v{VERSION}"
WHEEL = f"mercury_agent_global-{VERSION}-py3-none-any.whl"
SDIST = f"mercury_agent_global-{VERSION}.tar.gz"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_dist(assets: Path) -> dict[str, str]:
    assets.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name, payload in ((WHEEL, b"wheel-bytes"), (SDIST, b"sdist-bytes")):
        (assets / name).write_bytes(payload)
        digests[name] = _sha(payload)
    return digests


def _write_cdx(assets: Path, *, linked: bool = True, version: str = VERSION) -> None:
    ref = f"pkg:pypi/mercury-agent-global@{version}"
    doc = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "metadata": {
            "component": {
                "type": "application",
                "bom-ref": ref if linked else None,
                "name": "mercury-agent-global",
            }
        },
        "components": [
            {"bom-ref": ref, "name": "mercury-agent-global", "version": version},
            {"bom-ref": "pkg:pypi/numpy@2.5.4", "name": "numpy", "version": "2.5.4"},
        ],
    }
    (assets / "SBOM-api-closure.cdx.json").write_text(json.dumps(doc))


def _run_workflow_generator(dist: Path) -> None:
    """Execute release.yml's 'Generate package SBOM and provenance' script."""
    workflow = yaml.safe_load((_ROOT / ".github/workflows/release.yml").read_text())
    steps = workflow["jobs"]["build"]["steps"]
    step = next(s for s in steps if s.get("name") == "Generate package SBOM and provenance")
    run = step["run"]
    body = run.split("<<'PY'", 1)[1].rsplit("\nPY", 1)[0]
    env = {
        **os.environ,
        "RELEASE_VERSION": VERSION,
        "GITHUB_REPOSITORY": "Steel-SecAdv-LLC/Mercury-Agent",
        "GITHUB_SHA": "0" * 40,
        "GITHUB_RUN_ID": "1",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_REF": f"refs/tags/{TAG}",
    }
    subprocess.run([sys.executable, "-c", body], cwd=dist.parent, env=env, check=True, timeout=60)


class TestWorkflowGeneratorAgreesWithVerifier:
    def test_generated_records_verify_and_exclude_the_sbom_from_subjects(
        self, tmp_path: Path
    ) -> None:
        dist = tmp_path / "dist"
        _write_dist(dist)
        _write_cdx(dist)  # present in dist/ BEFORE the generator runs, as in CI
        _run_workflow_generator(dist)

        provenance = json.loads((dist / "provenance.json").read_text())
        subjects = {s["name"] for s in provenance["subject"]}
        assert subjects == {
            WHEEL,
            SDIST,
        }, "the generator listed a non-distribution file as a build subject"
        for check in (vr.check_self_authored_provenance, vr.check_spdx):
            result = check(dist, VERSION)
            assert result.status == vr.PASS, result


class TestOfflineChecks:
    @pytest.fixture
    def assets(self, tmp_path: Path) -> Path:
        digests = _write_dist(tmp_path)
        provenance = {
            "subject": [{"name": n, "digest": {"sha256": d}} for n, d in digests.items()],
            "predicate": {"buildDefinition": {"externalParameters": {"releaseVersion": VERSION}}},
        }
        (tmp_path / "provenance.json").write_text(json.dumps(provenance))
        spdx = {
            "packages": [
                {
                    "name": n,
                    "versionInfo": VERSION,
                    "checksums": [{"algorithm": "SHA256", "checksumValue": d}],
                }
                for n, d in digests.items()
            ]
        }
        (tmp_path / "SBOM.spdx.json").write_text(json.dumps(spdx))
        _write_cdx(tmp_path)
        return tmp_path

    def test_a_consistent_release_passes_offline(self, assets: Path) -> None:
        for check in (
            vr.check_artifacts,
            vr.check_self_authored_provenance,
            vr.check_spdx,
            vr.check_cyclonedx,
        ):
            assert check(assets, VERSION).status == vr.PASS

    def test_a_tampered_artifact_fails_both_records(self, assets: Path) -> None:
        (assets / WHEEL).write_bytes(b"tampered")
        assert vr.check_self_authored_provenance(assets, VERSION).status == vr.FAIL
        assert vr.check_spdx(assets, VERSION).status == vr.FAIL

    def test_an_extra_subject_fails(self, assets: Path) -> None:
        doc = json.loads((assets / "provenance.json").read_text())
        doc["subject"].append({"name": "SBOM-api-closure.cdx.json", "digest": {"sha256": "x"}})
        (assets / "provenance.json").write_text(json.dumps(doc))
        result = vr.check_self_authored_provenance(assets, VERSION)
        assert result.status == vr.FAIL
        assert "not the dist files" in result.detail

    def test_a_wrong_tag_fails_every_versioned_check(self, assets: Path) -> None:
        for check in (
            vr.check_artifacts,
            vr.check_self_authored_provenance,
            vr.check_spdx,
            vr.check_cyclonedx,
        ):
            assert check(assets, "9.9.9").status == vr.FAIL

    def test_an_unlinked_sbom_root_fails(self, assets: Path) -> None:
        """The emitter's default root name produced exactly this."""
        _write_cdx(assets, linked=False)
        result = vr.check_cyclonedx(assets, VERSION)
        assert result.status == vr.FAIL
        assert "unlinked" in result.detail

    def test_missing_records_fail_rather_than_pass(self, tmp_path: Path) -> None:
        _write_dist(tmp_path)
        for check in (vr.check_self_authored_provenance, vr.check_spdx, vr.check_cyclonedx):
            result = check(tmp_path, VERSION)
            assert result.status == vr.FAIL
            assert "missing" in result.detail


class TestToolChecks:
    def test_missing_gh_is_unverified_never_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_dist(tmp_path)
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        assert vr.check_sigstore(tmp_path, TAG).status == vr.UNVERIFIED
        assert vr.check_image("ghcr.io/x/y", VERSION).status == vr.UNVERIFIED

    def test_sigstore_enforces_repo_tag_and_workflow(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_dist(tmp_path)
        monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/gh")
        calls: list[list[str]] = []

        def runner(argv: Any) -> tuple[int, str, str]:
            calls.append(list(argv))
            return 0, "", ""

        assert vr.check_sigstore(tmp_path, TAG, runner).status == vr.PASS
        assert len(calls) == 2  # wheel and sdist
        for argv in calls:
            assert argv[:3] == ["gh", "attestation", "verify"]
            assert argv[argv.index("--repo") + 1] == vr.REPO
            assert argv[argv.index("--source-ref") + 1] == f"refs/tags/{TAG}"
            assert argv[argv.index("--signer-workflow") + 1] == vr.SIGNER_WORKFLOW

    def test_a_failed_attestation_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_dist(tmp_path)
        monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/gh")
        result = vr.check_sigstore(tmp_path, TAG, lambda _a: (1, "", "no attestations found"))
        assert result.status == vr.FAIL
        assert "no attestations found" in result.detail

    def test_image_without_provenance_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/docker")
        out = json.dumps({"manifest": {"digest": "sha256:abc"}})
        result = vr.check_image("ghcr.io/x/y", VERSION, lambda _a: (0, out, ""))
        assert result.status == vr.FAIL
        assert "no provenance" in result.detail


class TestVerdict:
    def test_exit_codes(self) -> None:
        ok = vr.CheckResult("a", vr.PASS, "")
        bad = vr.CheckResult("b", vr.FAIL, "")
        unknown = vr.CheckResult("c", vr.UNVERIFIED, "")
        assert vr.verdict([ok, ok]) == 0
        assert vr.verdict([ok, unknown]) == 2
        assert vr.verdict([ok, unknown, bad]) == 1

    def test_tag_forms_name_the_same_version(self) -> None:
        assert vr.version_of("v3.1.0") == vr.version_of("3.1.0") == "3.1.0"
