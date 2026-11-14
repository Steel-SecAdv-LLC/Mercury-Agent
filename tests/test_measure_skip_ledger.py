# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for ``scripts/measure_skip_ledger.py``, the gate on tests that can never run.

A skipped test reports as a pass, so two kinds of skip hide a test for good: a
guard on a dependency no extra installs (the test skips in every lane), and an
unconditional skip or expected failure. ``gymnasium`` was the first kind: one
test guarded on it, no extra carried it, and the code it covered had never run
in CI. These tests exercise the measurement on fixture trees and fixture
locks; the live-tree gate is a Workflow Hardening step and a pre-commit hook,
as for the codebase-scale block, so a stale page fails in seconds rather than
inside a full test lane.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "measure_skip_ledger.py"

_spec = importlib.util.spec_from_file_location("measure_skip_ledger", SCRIPT)
assert _spec and _spec.loader
msl = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = msl
_spec.loader.exec_module(msl)

_LOCK = """
version = 1

[[package]]
name = "mercury-agent-global"
version = "3.0.0"
dependencies = [{ name = "numpy" }]

[package.optional-dependencies]
all = [{ name = "torch" }, { name = "httpx", extra = ["http2"] }]
dev = [{ name = "pytest-xdist" }, { name = "PyYAML" }]
pqc = [{ name = "ama-cryptography" }]

[[package]]
name = "numpy"
version = "2.4.6"

[[package]]
name = "torch"
version = "2.15.0"
dependencies = [{ name = "sympy", marker = "sys_platform == 'linux'" }]

[[package]]
name = "sympy"
version = "1.14.0"

[[package]]
name = "httpx"
version = "0.28.1"

[package.optional-dependencies]
http2 = [{ name = "h2" }]
socks = [{ name = "socksio" }]

[[package]]
name = "h2"
version = "4.3.0"

[[package]]
name = "socksio"
version = "1.0.0"

[[package]]
name = "pytest-xdist"
version = "3.8.0"

[[package]]
name = "pyyaml"
version = "6.0.3"

[[package]]
name = "ama-cryptography"
version = "5.0.0"

[[package]]
name = "orphan"
version = "1.0.0"
"""


@pytest.fixture()
def no_declarations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start from no declared exception, so a fixture tree is judged on its own."""
    monkeypatch.setattr(msl, "ALLOWED_UNINSTALLED", {})
    monkeypatch.setattr(msl, "ALLOWED_PARKED", {})


def _tree(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "tests"
    for name, body in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body), encoding="utf-8")
    return root


def _lock(tmp_path: Path) -> Path:
    path = tmp_path / "uv.lock"
    path.write_text(_LOCK, encoding="utf-8")
    return path


class TestFullLaneDistributions:
    """What the full lane installs is read from the lock, extras included."""

    def test_follows_dependencies_and_their_extras(self, tmp_path: Path) -> None:
        installed = msl.full_lane_distributions(_lock(tmp_path), ("all", "dev"))
        assert installed == {"numpy", "torch", "sympy", "httpx", "h2", "pytest-xdist", "pyyaml"}

    def test_an_extra_the_lane_does_not_install_is_left_out(self, tmp_path: Path) -> None:
        installed = msl.full_lane_distributions(_lock(tmp_path), ("all", "dev"))
        assert "ama-cryptography" not in installed  # only [pqc] carries it
        assert "socksio" not in installed  # httpx[socks] was never requested
        assert "orphan" not in installed  # in the lock, reached by nothing

    def test_names_are_normalised(self, tmp_path: Path) -> None:
        installed = msl.full_lane_distributions(_lock(tmp_path), ("dev",))
        assert "pyyaml" in installed  # the lock entry is spelled "PyYAML"

    def test_the_declared_lane_is_the_one_ci_installs(self) -> None:
        workflow = yaml.safe_load(
            (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        )
        runs = " ".join(str(s.get("run") or "") for s in workflow["jobs"]["ml-tests"]["steps"])
        command = "uv sync --locked " + " ".join(f"--extra {e}" for e in msl.FULL_LANE_EXTRAS)
        assert command in runs


class TestProvider:
    """An import name resolves to the distribution that provides it."""

    def test_same_name(self) -> None:
        assert msl.provider("torch", {"torch"}) == ("torch", True)

    def test_submodule_and_underscore(self) -> None:
        assert msl.provider("prometheus_client.core", {"prometheus-client"}) == (
            "prometheus-client",
            True,
        )

    def test_aliased_import_names(self) -> None:
        assert msl.provider("yaml", {"pyyaml"}) == ("pyyaml", True)
        assert msl.provider("PIL.Image", {"pillow"}) == ("pillow", True)
        assert msl.provider("cv2", {"opencv-python-headless"}) == ("opencv-python-headless", True)

    def test_first_party_and_standard_library_are_always_present(self) -> None:
        assert msl.provider("omni_mercury_engine.ml.fusion_network", set()) == ("first-party", True)
        assert msl.provider("sqlite3", set()) == ("standard library", True)

    def test_a_distribution_no_extra_installs(self) -> None:
        assert msl.provider("gymnasium", {"torch"}) == ("gymnasium", False)


class TestScan:
    """The static pass tells a parked test from a conditionally skipped one."""

    def test_guards_are_collected_per_import(self, tmp_path: Path) -> None:
        root = _tree(
            tmp_path,
            {
                "test_a.py": """
                    import pytest
                    torch = pytest.importorskip("torch")
                    def test_x():
                        pytest.importorskip("gymnasium", reason="optional")
                """,
                "sub/test_b.py": """
                    import pytest
                    pytest.importorskip("torch.nn")
                """,
            },
        )
        scan = msl.scan_tests(root)
        assert scan.files == 2
        assert scan.guards == {
            "torch": {"tests/test_a.py"},
            "gymnasium": {"tests/test_a.py"},
            "torch.nn": {"tests/sub/test_b.py"},
        }
        rows = {row.module: row for row in msl.guard_rows(scan, {"torch"})}
        assert rows["torch"].files == 2 and rows["torch"].installed
        assert not rows["gymnasium"].installed

    def test_a_table_of_module_names_is_resolved(self, tmp_path: Path) -> None:
        root = _tree(
            tmp_path,
            {"test_a.py": """
                    import pytest
                    _EXTRA: dict[str, str] = {"chronos": "chronos", "moirai": "uni2ts"}
                    def guard(entry):
                        pytest.importorskip(_EXTRA[entry.name])
                """},
        )
        scan = msl.scan_tests(root)
        assert set(scan.guards) == {"chronos", "uni2ts"}
        assert scan.dynamic_guards == []

    def test_a_computed_name_cannot_be_accounted_for(self, tmp_path: Path) -> None:
        root = _tree(
            tmp_path,
            {"test_a.py": """
                    import pytest
                    def guard(name):
                        pytest.importorskip(name)
                """},
        )
        scan = msl.scan_tests(root)
        assert scan.dynamic_guards == ["tests/test_a.py:4"]
        assert any("computed name" in v for v in msl.violations(scan, []))

    def test_unconditional_skips_and_xfails_are_parked(self, tmp_path: Path) -> None:
        root = _tree(
            tmp_path,
            {"test_a.py": """
                    import pytest

                    @pytest.mark.skip(reason="flaky")
                    def test_marked():
                        pass

                    def test_inline():
                        pytest.skip("later")

                    @pytest.mark.xfail(reason="known bug", strict=True)
                    def test_expected_failure():
                        assert False

                    @pytest.mark.skip
                    class TestWhole:
                        def test_inner(self):
                            pass
                """},
        )
        parked = [(p.name, p.kind, p.reason) for p in msl.scan_tests(root).parked]
        assert parked == [
            ("test_marked", "skip", "flaky"),
            ("test_inline", "skip", "later"),
            ("test_expected_failure", "xfail", "known bug"),
            ("TestWhole", "skip", ""),
        ]

    def test_conditional_skips_are_not_parked(self, tmp_path: Path) -> None:
        root = _tree(
            tmp_path,
            {"test_a.py": """
                    import sys
                    import pytest

                    if sys.platform == "win32":
                        pytest.skip("posix only", allow_module_level=True)

                    @pytest.mark.skipif(sys.platform == "win32", reason="posix only")
                    def test_marked():
                        pass

                    def test_branch():
                        if sys.platform == "win32":
                            pytest.skip("posix only")

                    def test_except():
                        try:
                            import nothing
                        except ImportError:
                            pytest.skip("optional")

                    def helper():
                        for seed in range(3):
                            if seed:
                                return seed
                        pytest.skip("no seed qualified")

                    try:
                        import hypothesis
                    except ImportError:
                        @pytest.mark.skip(reason="hypothesis is not installed")
                        def test_placeholder():
                            pass
                """},
        )
        scan = msl.scan_tests(root)
        assert scan.parked == []
        assert scan.module_skips == 1
        assert scan.skipif_marks == 1
        assert scan.conditional_skips == 4


@pytest.mark.usefixtures("no_declarations")
class TestViolations:
    """The gate fails on what hides a test, and on declarations that went stale."""

    def _scan(self, tmp_path: Path, body: str) -> Any:
        return msl.scan_tests(_tree(tmp_path, {"test_a.py": body}))

    def test_a_guard_no_lane_can_satisfy_is_a_violation(self, tmp_path: Path) -> None:
        scan = self._scan(tmp_path, 'import pytest\npytest.importorskip("gymnasium")\n')
        problems = msl.violations(scan, msl.guard_rows(scan, {"torch"}))
        assert len(problems) == 1
        assert "`gymnasium`" in problems[0] and "skip in every lane" in problems[0]

    def test_a_declared_exception_passes_and_prints_its_reason(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(msl, "ALLOWED_UNINSTALLED", {"gymnasium": "run in the nightly lane"})
        scan = self._scan(tmp_path, 'import pytest\npytest.importorskip("gymnasium")\n')
        rows = msl.guard_rows(scan, {"torch"})
        assert msl.violations(scan, rows) == []
        assert "**no** — run in the nightly lane" in msl.render_block(scan, rows)

    def test_a_parked_test_is_a_violation_until_declared(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body = 'import pytest\n@pytest.mark.xfail(reason="bug 12")\ndef test_x():\n    pass\n'
        scan = self._scan(tmp_path, body)
        (problem,) = msl.violations(scan, [])
        assert "tests/test_a.py:2 `test_x` is an expected failure" in problem
        monkeypatch.setattr(msl, "ALLOWED_PARKED", {"tests/test_a.py::test_x": "tracked in gh-12"})
        assert msl.violations(scan, []) == []
        assert "tracked in gh-12" in msl.render_block(scan, [])

    def test_a_declaration_that_matches_nothing_is_stale(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(msl, "ALLOWED_UNINSTALLED", {"gymnasium": "old reason"})
        monkeypatch.setattr(msl, "ALLOWED_PARKED", {"tests/test_a.py::test_gone": "old reason"})
        scan = self._scan(tmp_path, 'import pytest\npytest.importorskip("torch")\n')
        problems = msl.violations(scan, msl.guard_rows(scan, {"torch"}))
        assert sorted(p.split("`")[1] for p in problems) == [
            "gymnasium",
            "tests/test_a.py::test_gone",
        ]
        assert all("remove it" in p for p in problems)


def test_the_live_declarations_name_real_reasons() -> None:
    assert set(msl.ALLOWED_UNINSTALLED) == {"ama_cryptography"}
    assert "build-ama-cryptography" in msl.ALLOWED_UNINSTALLED["ama_cryptography"]
    assert msl.ALLOWED_PARKED == {}


@pytest.mark.usefixtures("no_declarations")
class TestBlock:
    """Rendering is deterministic, and ``--check``/``--update`` round-trip."""

    def _render(self, tmp_path: Path) -> str:
        scan = msl.scan_tests(
            _tree(tmp_path, {"test_a.py": 'import pytest\npytest.importorskip("torch")\n'})
        )
        block: str = msl.render_block(scan, msl.guard_rows(scan, {"torch"}))
        return block

    def test_block_is_marked_and_deterministic(self, tmp_path: Path) -> None:
        first, second = self._render(tmp_path), self._render(tmp_path)
        assert first == second
        assert first.startswith(msl.LEDGER_START) and first.endswith(msl.LEDGER_END)
        assert "| `torch` | torch | yes |" in first
        assert "None. No test carries" in first

    def test_volatile_totals_stay_out_of_the_gated_block(self, tmp_path: Path) -> None:
        """Adding a second file that guards on torch must not change the block."""
        one = self._render(tmp_path / "one")
        scan = msl.scan_tests(
            _tree(
                tmp_path / "two",
                {
                    "test_a.py": 'import pytest\npytest.importorskip("torch")\n',
                    "test_b.py": 'import pytest\npytest.importorskip("torch")\n',
                },
            )
        )
        assert msl.render_block(scan, msl.guard_rows(scan, {"torch"})) == one

    def test_splice_replaces_only_the_marked_block(self) -> None:
        text = f"# Title\n\nintro\n\n{msl.LEDGER_START}\nold\n{msl.LEDGER_END}\n\ntail\n"
        block = f"{msl.LEDGER_START}\nnew\n{msl.LEDGER_END}"
        updated = msl.splice(text, block)
        assert updated == f"# Title\n\nintro\n\n{block}\n\ntail\n"
        assert msl.current_block(updated) == block

    def test_missing_markers_are_an_error(self) -> None:
        assert msl.splice("no markers here", "x") is None
        assert msl.current_block("no markers here") is None

    def _point_at(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> Path:
        monkeypatch.setattr(msl, "TESTS", _tree(tmp_path, {"test_a.py": body}))
        monkeypatch.setattr(msl, "LOCK", _lock(tmp_path))
        page = tmp_path / "LEDGER.md"
        page.write_text(f"# Ledger\n\n{msl.LEDGER_START}\n{msl.LEDGER_END}\n", encoding="utf-8")
        return page

    def test_update_then_check_roundtrip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        page = self._point_at(
            tmp_path, monkeypatch, 'import pytest\npytest.importorskip("torch")\n'
        )
        assert msl.main(["--check", str(page)]) == 1  # the empty block is stale
        assert msl.main(["--update", str(page)]) == 0
        assert msl.main(["--check", str(page)]) == 0

    def test_check_fails_on_a_violation_even_when_the_block_is_current(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        page = self._point_at(
            tmp_path, monkeypatch, 'import pytest\npytest.importorskip("gymnasium")\n'
        )
        assert msl.main(["--update", str(page)]) == 0
        capsys.readouterr()
        assert msl.main(["--check", str(page)]) == 1
        out = capsys.readouterr().out
        assert "::error title=Skip ledger::" in out and "DRIFT" not in out

    def test_a_page_without_markers_is_exit_two(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        page = self._point_at(tmp_path, monkeypatch, "import pytest\n")
        page.write_text("# Ledger\n", encoding="utf-8")
        assert msl.main(["--check", str(page)]) == 2
        assert msl.main(["--update", str(page)]) == 2
