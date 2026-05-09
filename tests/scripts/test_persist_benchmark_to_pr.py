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

    def test_other_422_reasons_are_reraised(
        self, persister: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GitHub returns HTTP 422 for a *family* of validation failures
        (invalid ref name, bad SHA, ref already exists, ...).  Only the
        'Reference already exists' case may fall through to the PATCH
        fallback; any other 422 must surface so callers see the real
        create-ref failure rather than a silent retry against a ref
        that was never created."""

        def fake_api(method: str, path: str, token: str, payload: Any = None) -> Any:
            if method == "POST" and path == "/repos/o/r/git/refs":
                raise RuntimeError("HTTP 422 — Validation Failed: invalid ref name")
            raise AssertionError(
                f"PATCH must NOT be attempted when 422 is for a "
                f"different reason; got: {method} {path}"
            )

        monkeypatch.setattr(persister, "_api", fake_api)

        with pytest.raises(RuntimeError, match="invalid ref name"):
            persister.upsert_branch_ref(  # type: ignore[attr-defined]
                "o", "r", "feature", "abc1234", "tok"
            )


class TestNoopBranchLookupErrorHandling:
    """Pin the round-5 fix: the no-op-vs-existing-branch safeguard
    must distinguish HTTP 404 (branch genuinely absent) from
    transient API failures (403 / 500 / rate-limit / network).
    Catching every ``RuntimeError`` would silently fall through to
    commit + force-push on transient errors, churning required-check
    approvals on the open bot PR for any blip."""

    def test_404_treated_as_branch_absent(
        self,
        persister: object,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        f = tmp_path / "result.json"
        f.write_text('{"x": 1}\n', encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        same_tree = "TREE-A"

        def fake_api(method: str, path: str, token: str, payload: Any = None) -> Any:
            if method == "GET" and path == "/repos/o/r/git/ref/heads/main":
                return {"object": {"sha": "main-sha"}}
            if method == "GET" and path == "/repos/o/r/git/commits/main-sha":
                return {"sha": "main-sha", "tree": {"sha": "main-tree"}}
            if method == "POST" and path == "/repos/o/r/git/blobs":
                return {"sha": "blob-sha"}
            if method == "POST" and path == "/repos/o/r/git/trees":
                return {"sha": same_tree}
            # Branch genuinely absent: 404 must be swallowed and the
            # script must continue to create the commit + ref normally.
            if method == "GET" and path == "/repos/o/r/git/ref/heads/feature":
                raise RuntimeError(
                    "GitHub API GET .../git/ref/heads/feature failed: HTTP 404 — Not Found"
                )
            if method == "POST" and path == "/repos/o/r/git/commits":
                return {"sha": "new-commit"}
            if method == "POST" and path == "/repos/o/r/git/refs":
                return {}
            if method == "GET" and path == "/repos/o/r/pulls?state=open&base=main&head=o:feature":
                return []
            if method == "POST" and path == "/repos/o/r/pulls":
                return {"number": 7}
            raise AssertionError(f"unexpected: {method} {path}")

        monkeypatch.setattr(persister, "_api", fake_api)
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "p.py",
                "--base",
                "main",
                "--branch",
                "feature",
                "--commit-message",
                "msg",
                "--pr-title",
                "t",
                "--pr-body",
                "b",
                "--files",
                f.name,
            ],
        )

        rc = persister.main()  # type: ignore[attr-defined]
        assert rc == 0

    def test_non_404_lookup_error_is_reraised(
        self,
        persister: object,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Transient 500 / 403 / rate-limit must NOT be silently
        treated as 'branch absent'; the persister must surface them
        so the workflow run fails and the bot PR's approvals stay
        intact."""
        f = tmp_path / "result.json"
        f.write_text('{"x": 1}\n', encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        same_tree = "TREE-B"

        def fake_api(method: str, path: str, token: str, payload: Any = None) -> Any:
            if method == "GET" and path == "/repos/o/r/git/ref/heads/main":
                return {"object": {"sha": "main-sha"}}
            if method == "GET" and path == "/repos/o/r/git/commits/main-sha":
                return {"sha": "main-sha", "tree": {"sha": "main-tree"}}
            if method == "POST" and path == "/repos/o/r/git/blobs":
                return {"sha": "blob-sha"}
            if method == "POST" and path == "/repos/o/r/git/trees":
                return {"sha": same_tree}
            if method == "GET" and path == "/repos/o/r/git/ref/heads/feature":
                raise RuntimeError(
                    "GitHub API GET .../git/ref/heads/feature failed: HTTP 500 — Internal Server Error"
                )
            raise AssertionError(
                f"persister must NOT continue past a 500 on the branch lookup; got: {method} {path}"
            )

        monkeypatch.setattr(persister, "_api", fake_api)
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "p.py",
                "--base",
                "main",
                "--branch",
                "feature",
                "--commit-message",
                "msg",
                "--pr-title",
                "t",
                "--pr-body",
                "b",
                "--files",
                f.name,
            ],
        )

        with pytest.raises(RuntimeError, match="HTTP 500"):
            persister.main()  # type: ignore[attr-defined]


class TestNoopAgainstExistingBranch:
    """Pin the round-3 fix: when the persistence branch already exists and
    its head commit's tree matches what we would create, skip the commit +
    force-push so required-check approvals on the bot PR are preserved."""

    def test_skips_commit_and_force_push_when_existing_tree_matches(
        self,
        persister: object,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Stage a file the persister will read off disk.
        f = tmp_path / "result.json"
        f.write_text('{"x": 1}\n', encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        # The blob/tree/commit SHAs are arbitrary as long as the
        # mock returns the same tree_sha for the new tree AND for
        # the existing branch head's commit tree.
        same_tree = "TREE-MATCH"

        def fake_api(method: str, path: str, token: str, payload: Any = None) -> Any:
            # base ref + base commit tree
            if method == "GET" and path == "/repos/o/r/git/ref/heads/main":
                return {"object": {"sha": "main-sha"}}
            if method == "GET" and path == "/repos/o/r/git/commits/main-sha":
                return {"sha": "main-sha", "tree": {"sha": "main-tree"}}
            # blob create
            if method == "POST" and path == "/repos/o/r/git/blobs":
                return {"sha": "blob-sha"}
            # tree create
            if method == "POST" and path == "/repos/o/r/git/trees":
                return {"sha": same_tree}
            # branch already exists and points at a commit whose tree matches
            if method == "GET" and path == "/repos/o/r/git/ref/heads/feature":
                return {"object": {"sha": "branch-sha"}}
            if method == "GET" and path == "/repos/o/r/git/commits/branch-sha":
                return {"sha": "branch-sha", "tree": {"sha": same_tree}}
            # PR exists; refresh title/body must hit PATCH
            if method == "GET" and path == "/repos/o/r/pulls?state=open&base=main&head=o:feature":
                return [{"number": 99}]
            if method == "PATCH" and path == "/repos/o/r/pulls/99":
                fake_api.patched = payload  # type: ignore[attr-defined]
                return {"number": 99}
            raise AssertionError(f"unexpected API call in no-op-vs-existing path: {method} {path}")

        fake_api.patched = None  # type: ignore[attr-defined]

        monkeypatch.setattr(persister, "_api", fake_api)
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "persist_benchmark_to_pr.py",
                "--base",
                "main",
                "--branch",
                "feature",
                "--commit-message",
                "noop run",
                "--pr-title",
                "fresh title",
                "--pr-body",
                "fresh body",
                "--files",
                f.name,
            ],
        )

        rc = persister.main()  # type: ignore[attr-defined]

        assert rc == 0
        # The PATCH MUST have run with the fresh title/body; if it
        # didn't, the no-op path silently swallowed the metadata refresh
        # and the bot PR would keep stale numbers.
        assert fake_api.patched == {"title": "fresh title", "body": "fresh body"}, (  # type: ignore[attr-defined]
            "No-op-vs-existing path must still PATCH the PR title/body."
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
