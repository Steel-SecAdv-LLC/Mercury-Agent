# Copyright (C) 2025 Steel Security Advisors LLC
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
        """The lyapunov validator is the canonical top-level registered tool."""
        assert "lyapunov_validator" in _REGISTRY
        module_path, entry = _REGISTRY["lyapunov_validator"]
        assert module_path == "tools.lyapunov_validator"
        assert entry == "_cli"

    def test_registry_has_cryptographic_evidence_tools(self) -> None:
        """The 3 graduated cryptographic evidence tools route through ``main``.

        ``sigma_immutable_verifier``, ``pqc_capability_probe`` and
        ``kat_runner_standalone`` were scaffolded under
        ``omni_mercury_engine.tools.*`` and graduated into the dispatcher
        once their behavioural tests landed in
        ``tests/tools/test_new_tools.py``.  Pin the (module, attr) mapping
        so a silent rename / move would surface as a hard test failure.
        """
        expected = {
            "sigma_immutable_verifier": (
                "omni_mercury_engine.tools.sigma_immutable_verifier",
                "main",
            ),
            "pqc_capability_probe": (
                "omni_mercury_engine.tools.pqc_capability_probe",
                "main",
            ),
            "kat_runner_standalone": (
                "omni_mercury_engine.tools.kat_runner_standalone",
                "main",
            ),
        }
        for name, mapping in expected.items():
            assert name in _REGISTRY, f"{name!r} missing from dispatcher registry"
            assert (
                _REGISTRY[name] == mapping
            ), f"{name!r} registry mapping drifted: {_REGISTRY[name]!r} != {mapping!r}"

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

    def test_registry_entries_are_callable(self) -> None:
        """Every registered entry must resolve to a callable taking ``argv``.

        The dispatcher's ``_resolve_entrypoint`` raises on non-callables,
        but exercising every entry here means a regression at registry
        edit time fails CI rather than at operator dispatch time.
        """
        import importlib

        for name, (module_path, attr) in _REGISTRY.items():
            mod = importlib.import_module(module_path)
            fn = getattr(mod, attr)
            assert callable(fn), f"registry tool {name!r} attr {attr!r} is not callable"


class TestListSubcommand:
    def test_list_returns_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = _main(["list"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "lyapunov_validator" in captured.out

    def test_list_enumerates_every_registered_tool(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The ``list`` subcommand is the operator's source-of-truth — pin it.

        If a new tool is added to ``_REGISTRY`` it MUST appear in
        ``python -m tools list``; if a tool is removed the line MUST
        disappear.  Comparing the captured lines against the registry
        catches both directions.
        """
        rc = _main(["list"])
        assert rc == 0
        captured = capsys.readouterr()
        listed = {line for line in captured.out.splitlines() if line.strip()}
        assert listed == set(_REGISTRY), (
            f"``list`` output drifted from _REGISTRY: extra={listed - set(_REGISTRY)!r}, "
            f"missing={set(_REGISTRY) - listed!r}"
        )

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
        # The three graduated cryptographic evidence tools must also be listed.
        for name in ("sigma_immutable_verifier", "pqc_capability_probe", "kat_runner_standalone"):
            assert name in result.stdout, f"{name!r} missing from ``python -m tools list`` output"


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


class TestDispatchesToGraduatedTools:
    """End-to-end: ``python -m tools <name>`` actually runs the tool.

    These tests ensure the (module, attr) tuples in :data:`_REGISTRY`
    resolve to working callables AND that the dispatcher's argv forwarding
    correctly drops the subcommand and passes the remaining args through
    to the tool.  A registry entry that points at a non-existent attribute
    would fail at dispatch time with an opaque AttributeError; this guard
    surfaces the regression as a clear test failure.
    """

    def test_dispatch_kat_runner_ed25519(self, tmp_path: Path) -> None:
        """``python -m tools kat_runner_standalone`` runs the ed25519 path."""
        import json

        out = tmp_path / "cert.json"
        rc = _main(
            [
                "kat_runner_standalone",
                "--algorithms",
                "ed25519",
                "--output",
                str(out),
            ]
        )
        assert rc == 0, "kat_runner_standalone ed25519 path must always exit 0"
        cert = json.loads(out.read_text())
        assert cert["schema"] == "mercury.tools.kat_runner_standalone/v1"
        assert cert["status"] == "ok"
        assert cert["body"]["summary"]["passed"] == 3

    def test_dispatch_pqc_capability_probe(self, tmp_path: Path) -> None:
        """``python -m tools pqc_capability_probe`` runs and emits a cert."""
        import json

        out = tmp_path / "cert.json"
        rc = _main(["pqc_capability_probe", "--output", str(out)])
        # Without ``--require-real`` the contract is rc==0 for both
        # ``ok`` and ``warn`` (per ``tools._base.emit``).  Allowing
        # ``rc == 1`` here would silently accept a ``fail`` certificate
        # — never the intended outcome for this dispatch smoke.
        assert rc == 0
        cert = json.loads(out.read_text())
        assert cert["schema"] == "mercury.tools.pqc_capability_probe/v1"
        assert cert["status"] in {"ok", "warn"}, (
            f"dispatched probe must not return 'fail' without --require-real "
            f"(got {cert['status']!r})"
        )

    def test_dispatch_sigma_immutable_verifier(self, tmp_path: Path) -> None:
        """``python -m tools sigma_immutable_verifier`` runs end-to-end."""
        import json

        out = tmp_path / "cert.json"
        rc = _main(["sigma_immutable_verifier", "--output", str(out)])
        assert rc == 0
        cert = json.loads(out.read_text())
        assert cert["schema"] == "mercury.tools.sigma_immutable_verifier/v1"
        assert cert["status"] == "ok"
        assert cert["body"]["signatures"]["ml-dsa-65"] == "verified"
