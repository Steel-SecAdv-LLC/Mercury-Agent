# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The observatory CLI (student/professor front door) and the Lean project verifier."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pytest
from click.testing import CliRunner

from omni_mercury_engine.cli import main
from omni_mercury_engine.verifiers.lean_theorem import verify_lean_project


@pytest.fixture
def runner() -> CliRunner:
    """A fresh click runner."""
    return CliRunner()


class TestObservatoryCli:
    """Classroom-notation input and filter flags through `mercury-agent observatory`."""

    def test_solve_human_output(self, runner: CliRunner) -> None:
        """A quadratic solves with exact answers in readable lines."""
        result = runner.invoke(main, ["observatory", "solve", "x^2 - 5x + 6 = 0"])
        assert result.exit_code == 0
        assert "x = 2" in result.output
        assert "x = 3" in result.output
        assert "[exact]" in result.output

    def test_solve_json_output(self, runner: CliRunner) -> None:
        """--json emits the full machine-readable result."""
        result = runner.invoke(main, ["observatory", "solve", "x^2 = 4", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "grounded"
        assert sorted(s["value"] for s in payload["solutions"]) == ["-2", "2"]

    def test_solve_domain_filter_mentions_hidden_roots(self, runner: CliRunner) -> None:
        """The real-domain filter says how many complex solutions it hid."""
        result = runner.invoke(main, ["observatory", "solve", "x^2 + 1 = 0"])
        assert result.exit_code == 0
        assert "2 complex solution(s) hidden" in result.output

    def test_solve_refuses_unsafe_input_with_exit_code(self, runner: CliRunner) -> None:
        """Unsafe input exits 2 with a refusal, not a traceback."""
        result = runner.invoke(main, ["observatory", "solve", "x[0] = 1"])
        assert result.exit_code == 2
        assert "refused:" in result.output

    def test_zeta_filters(self, runner: CliRunner) -> None:
        """--top bounds the pair list; --anomalies-only applies the threshold."""
        result = runner.invoke(main, ["observatory", "zeta", "--zeros", "10", "--top", "2"])
        assert result.exit_code == 0
        assert result.output.count("closest pair:") == 2
        quiet = runner.invoke(
            main,
            ["observatory", "zeta", "--zeros", "10", "--threshold", "0.01", "--anomalies-only"],
        )
        assert quiet.exit_code == 0
        assert "no pairs below threshold" in quiet.output

    def test_goldbach_command(self, runner: CliRunner) -> None:
        """Bands and the empty counterexample hunt are reported."""
        result = runner.invoke(main, ["observatory", "goldbach", "--limit", "500"])
        assert result.exit_code == 0
        assert "comet bands" in result.output
        assert "counterexample candidates (r(n)=0): none" in result.output

    def test_collatz_command_shows_27(self, runner: CliRunner) -> None:
        """The classic 27 record appears with its peak."""
        result = runner.invoke(main, ["observatory", "collatz", "--limit", "100", "--top", "20"])
        assert result.exit_code == 0
        assert "n=27 takes 111 steps" in result.output
        assert "peak 9232" in result.output


class TestVerifyLeanProject:
    """The mathlib-capable project verifier: fail-closed, never fabricated."""

    @staticmethod
    def _project(tmp_path: Path) -> Path:
        (tmp_path / "lakefile.lean").write_text("-- lake project\n", encoding="utf-8")
        return tmp_path

    def test_missing_directory_refused(self, tmp_path: Path) -> None:
        """A nonexistent project directory is caller error, not oracle absence."""
        with pytest.raises(ValueError, match="not a directory"):
            verify_lean_project(tmp_path / "nowhere")

    def test_missing_lakefile_refused(self, tmp_path: Path) -> None:
        """A directory without a lakefile is not a Lean project."""
        with pytest.raises(ValueError, match="lakefile"):
            verify_lean_project(tmp_path)

    def test_no_lake_toolchain_is_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without lake on PATH the verdict is available=False -- no fake pass."""
        monkeypatch.setattr(
            "omni_mercury_engine.verifiers.lean_theorem.shutil.which", lambda name: None
        )
        verdict = verify_lean_project(self._project(tmp_path))
        assert verdict.available is False
        assert verdict.valid is False
        assert "lake toolchain" in verdict.reason
        assert verdict.checker == "lean4+lake"

    def test_successful_build_is_valid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exit code 0 with clean diagnostics means the kernel accepted the project."""
        monkeypatch.setattr(
            "omni_mercury_engine.verifiers.lean_theorem.shutil.which",
            lambda name: "/usr/bin/lake",
        )
        calls: list[list[str]] = []

        def fake_run(argv: list[str], **kwargs: Any) -> SimpleNamespace:
            calls.append(list(argv))
            return SimpleNamespace(returncode=0, stdout="Build completed.", stderr="")

        monkeypatch.setattr("omni_mercury_engine.verifiers.lean_theorem.subprocess.run", fake_run)
        verdict = verify_lean_project(self._project(tmp_path), target="Erdos.Problem316")
        assert verdict.valid is True
        assert verdict.available is True
        assert calls == [["/usr/bin/lake", "build", "Erdos.Problem316"]]

    def test_failed_build_is_refuted_with_diagnostics(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A kernel rejection carries the compiler's own words, truncated."""
        monkeypatch.setattr(
            "omni_mercury_engine.verifiers.lean_theorem.shutil.which",
            lambda name: "/usr/bin/lake",
        )
        monkeypatch.setattr(
            "omni_mercury_engine.verifiers.lean_theorem.subprocess.run",
            lambda argv, **kwargs: SimpleNamespace(
                returncode=1, stdout="", stderr="error: type mismatch at two_plus_two"
            ),
        )
        verdict = verify_lean_project(self._project(tmp_path))
        assert verdict.valid is False
        assert verdict.available is True
        assert "type mismatch" in verdict.reason

    def test_timeout_is_not_a_pass(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A build that times out is reported as such, never as accepted."""
        monkeypatch.setattr(
            "omni_mercury_engine.verifiers.lean_theorem.shutil.which",
            lambda name: "/usr/bin/lake",
        )

        def raise_timeout(argv: list[str], **kwargs: Any) -> SimpleNamespace:
            raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout", 0))

        monkeypatch.setattr(
            "omni_mercury_engine.verifiers.lean_theorem.subprocess.run", raise_timeout
        )
        verdict = verify_lean_project(self._project(tmp_path), timeout=1.0)
        assert verdict.valid is False
        assert verdict.available is True
        assert "timed out" in verdict.reason
