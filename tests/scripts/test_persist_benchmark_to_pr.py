"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Regression tests for ``scripts/persist_benchmark_to_pr.py``.

The persister talks to the GitHub Git Database API to commit
benchmark artefacts on a feature branch and open / update a PR
into ``main``.  Because the script is exercised only by a CI
workflow against the real API, regressions in the call sequence
or in the new-PR-vs-existing-PR branching can land unnoticed
until ``ci/benchmark-results`` starts producing wrong commits.

These tests stub out ``_api`` with a recording double and drive
the public functions and ``main()`` with synthetic responses.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "persist_benchmark_to_pr.py"


@pytest.fixture
def persister() -> object:
    """Import the persister as a module so we can monkeypatch ``_api``."""
    spec = importlib.util.spec_from_file_location("persister", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["persister"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("persister", None)


class _RecordingApi:
    """Fake ``_api`` that records every call and returns canned responses."""

    def __init__(self, responses: dict[tuple[str, str], Any]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        method: str,
        path: str,
        token: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        self.calls.append({"method": method, "path": path, "payload": payload})
        # Match by (method, path) first, fall back to path-only for GETs.
        key = (method, path)
        if key in self._responses:
            return self._responses[key]
        if path in self._responses:
            return self._responses[path]
        raise AssertionError(f"unmocked API call: {method} {path}")


class TestCreateCommit:
    def test_creates_commit_without_signature_field(self, persister: object) -> None:
        """The persister deliberately does NOT submit a detached
        signature; verification is delegated to GitHub's web-flow
        signing on github-actions[bot] commits.  This test pins the
        behaviour so a future change cannot silently start sending
        signatures (or stop sending them) without an explicit decision."""
        api = _RecordingApi({("POST", "/repos/o/r/git/commits"): {"sha": "feedface" * 5}})
        persister._api = api  # type: ignore[attr-defined]

        sha = persister.create_commit(  # type: ignore[attr-defined]
            "o", "r", "msg", "tree-sha", "parent-sha", "tok"
        )

        assert sha.startswith("feedface")
        assert len(api.calls) == 1
        payload = api.calls[0]["payload"]
        assert payload == {
            "message": "msg",
            "tree": "tree-sha",
            "parents": ["parent-sha"],
        }
        assert "signature" not in payload, (
            "Persister must not include a `signature` field — verification "
            "comes from GitHub's web-flow signing on github-actions[bot] "
            "commits, not from a detached signature in the payload. If you "
            "are intentionally adding signed-commit support, update both "
            "this test and the module docstring."
        )


class TestUpdatePullRequest:
    def test_patches_title_and_body(self, persister: object) -> None:
        api = _RecordingApi({("PATCH", "/repos/o/r/pulls/42"): {"number": 42}})
        persister._api = api  # type: ignore[attr-defined]

        persister.update_pull_request(  # type: ignore[attr-defined]
            "o", "r", 42, "new title", "new body", "tok"
        )

        assert len(api.calls) == 1
        assert api.calls[0]["method"] == "PATCH"
        assert api.calls[0]["path"] == "/repos/o/r/pulls/42"
        assert api.calls[0]["payload"] == {"title": "new title", "body": "new body"}


class TestUpsertBranchRef:
    def test_creates_new_ref_when_absent(self, persister: object) -> None:
        api = _RecordingApi({("POST", "/repos/o/r/git/refs"): {}})
        persister._api = api  # type: ignore[attr-defined]

        persister.upsert_branch_ref("o", "r", "feature", "abc1234", "tok")  # type: ignore[attr-defined]

        assert len(api.calls) == 1
        assert api.calls[0]["payload"] == {
            "ref": "refs/heads/feature",
            "sha": "abc1234",
        }

    def test_force_updates_when_ref_already_exists(
        self, persister: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If ``POST /git/refs`` returns 'Reference already exists', the
        persister must fall back to ``PATCH /git/refs/heads/{branch}``
        with ``force=true``.  Without that, reruns of the workflow
        leave the branch on a stale commit and the PR diverges from
        the actual benchmark results."""

        def fake_api(method: str, path: str, token: str, payload: Any = None) -> Any:
            if method == "POST" and path == "/repos/o/r/git/refs":
                raise RuntimeError("HTTP 422 — Reference already exists")
            if method == "PATCH" and path == "/repos/o/r/git/refs/heads/feature":
                fake_api.patched = payload  # type: ignore[attr-defined]
                return {}
            raise AssertionError(f"unmocked: {method} {path}")

        fake_api.patched = None  # type: ignore[attr-defined]
        monkeypatch.setattr(persister, "_api", fake_api)

        persister.upsert_branch_ref("o", "r", "feature", "abc1234", "tok")  # type: ignore[attr-defined]

        assert fake_api.patched == {"sha": "abc1234", "force": True}, (  # type: ignore[attr-defined]
            "Existing-ref path must force-update with the new commit SHA."
        )


class TestFindOpenPr:
    def test_returns_pr_number_when_match(self, persister: object) -> None:
        api = _RecordingApi(
            {"/repos/o/r/pulls?state=open&base=main&head=o:feature": [{"number": 7}]}
        )
        persister._api = api  # type: ignore[attr-defined]

        n = persister.find_open_pr("o", "r", "feature", "main", "tok")  # type: ignore[attr-defined]
        assert n == 7

    def test_returns_none_when_no_match(self, persister: object) -> None:
        api = _RecordingApi({"/repos/o/r/pulls?state=open&base=main&head=o:feature": []})
        persister._api = api  # type: ignore[attr-defined]

        n = persister.find_open_pr("o", "r", "feature", "main", "tok")  # type: ignore[attr-defined]
        assert n is None
