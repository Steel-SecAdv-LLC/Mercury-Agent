"""Smoke tests for ``python -m tools`` dispatcher entry-point.

The dispatcher lives in ``tools/__main__.py`` and registers operator
tools by name in ``_REGISTRY``.  These tests pin three invariants:

* ``python -m tools list`` enumerates the registry (no missing tools,
  no extra entries that don't actually exist).
* ``python -m tools`` (no subcommand) prints usage and exits 2 so a
  bare invocation surfaces an error rather than appearing to succeed.
* ``python -m tools <unknown>`` exits 2 with a "unknown tool" message
  so typos / removed-tool references fail loudly.

The actual subcommand behaviour is covered by per-tool test files
(e.g. ``test_lyapunov_validator.py::TestCli::test_module_dash_m_invocation``);
this file deliberately stays at the dispatcher boundary so the
contract between ``tools.__main__`` and the individual submodules is
not coupled to the submodules' implementation details.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from tools.__main__ import _REGISTRY, _main

if TYPE_CHECKING:
    import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


class TestRegistry:
    def test_registry_has_lyapunov_validator(self) -> None:
        """The lyapunov validator is the canonical registered tool."""
        assert "lyapunov_validator" in _REGISTRY
        module_path, entry = _REGISTRY["lyapunov_validator"]
        assert module_path == "tools.lyapunov_validator"
        assert entry == "_cli"

    def test_registry_entries_are_resolvable(self) -> None:
        """Every registered (module, attr) must actually exist.

        Catches the failure mode where a tool gets renamed or deleted
        but the registry entry is forgotten — the dispatcher would
        then accept the name but fail at resolution time.
        """
        import importlib

        for name, (module_path, attr) in _REGISTRY.items():
            mod = importlib.import_module(module_path)
            assert hasattr(
                mod, attr
            ), f"registry tool {name!r} maps to {module_path}.{attr} which does not exist"


class TestListSubcommand:
    def test_list_returns_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = _main(["list"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "lyapunov_validator" in captured.out

    def test_list_via_subprocess(self) -> None:
        """End-to-end ``python -m tools list`` smoke (no entry-point bypass)."""
        result = subprocess.run(
            [sys.executable, "-m", "tools", "list"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "lyapunov_validator" in result.stdout


class TestNoSubcommand:
    def test_no_args_prints_usage_and_exits_2(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = _main([])
        assert rc == 2
        captured = capsys.readouterr()
        assert "usage:" in captured.err.lower() or "usage:" in captured.out.lower()

    def test_help_flag_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``--help`` is an explicit ask and should return 0."""
        rc = _main(["--help"])
        assert rc == 0


class TestUnknownSubcommand:
    def test_unknown_tool_exits_2(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = _main(["this_tool_does_not_exist"])
        assert rc == 2
        captured = capsys.readouterr()
        # Either the missing-tool message or the usage banner should mention
        # the unknown name; we accept either since the usage path also runs.
        combined = captured.err.lower() + captured.out.lower()
        assert "this_tool_does_not_exist" in combined or "unknown" in combined
