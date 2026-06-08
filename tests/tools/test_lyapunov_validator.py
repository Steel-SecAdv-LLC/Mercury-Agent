# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for :mod:`tools.lyapunov_validator`.

These tests exercise the validator on both *certified-positive* and
*certified-negative* cases.  They are designed to:

* prove that a known-good linear system passes (certified lambda for
  ``A = diag(-0.25, -0.5)``, ``P = I`` is exactly ``0.5``);
* prove that the canonical config ``configs/lyapunov_canonical.yaml``
  remains consistent with the validator (this is the regression guard
  that the rest of the codebase depends on);
* prove that the validator rejects a non-positive-definite ``P``;
* prove that the validator rejects a system whose decay rate is *less*
  than the claimed lambda (false-claim guard);
* exercise the sample-based fallback path;
* exercise the nested ``lyapunov:`` block path so multi-variant ablation
  configs are gated on a single shared certificate;
* exercise the ``python -m tools.lyapunov_validator`` CLI shim.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.lyapunov_validator import (
    _cli,
    _extract_lyapunov_block,
    canonical_lambda_for_linear_system,
    is_positive_definite,
    validate_lyapunov_from_config,
    validate_quadratic,
    validate_samples,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_CFG = _REPO_ROOT / "configs" / "lyapunov_canonical.yaml"


class TestPositiveDefinite:
    def test_identity_is_pd(self) -> None:
        assert is_positive_definite(np.eye(3))

    def test_zero_is_not_pd(self) -> None:
        assert not is_positive_definite(np.zeros((3, 3)))

    def test_negative_definite_is_not_pd(self) -> None:
        assert not is_positive_definite(-np.eye(2))

    def test_non_symmetric_is_not_pd(self) -> None:
        M = np.array([[1.0, 2.0], [0.0, 1.0]])
        assert not is_positive_definite(M)


class TestCanonicalLambda:
    def test_diagonal_decay(self) -> None:
        # A = diag(-a, -b), P = I => Q = -2 diag(a, b), pencil eigvals
        # are {-2a, -2b}; max is -2*min(a, b); certified lambda is
        # 2 * min(a, b).
        A = np.diag([-0.25, -0.5])
        P = np.eye(2)
        assert canonical_lambda_for_linear_system(A, P) == pytest.approx(0.5)

    def test_unstable_system_yields_negative_lambda(self) -> None:
        A = np.array([[1.0, 0.0], [0.0, -1.0]])
        P = np.eye(2)
        lam = canonical_lambda_for_linear_system(A, P)
        assert lam < 0  # certifies no exponential decay


class TestValidateQuadratic:
    def test_canonical_claim_holds(self) -> None:
        A = np.diag([-0.25, -0.5])
        P = np.eye(2)
        ok, details = validate_quadratic(A, P, claimed_lambda=0.25)
        assert ok
        assert details["computed_lambda"] == pytest.approx(0.5)

    def test_overclaim_is_rejected(self) -> None:
        A = np.diag([-0.25, -0.5])
        P = np.eye(2)
        ok, details = validate_quadratic(A, P, claimed_lambda=0.75)
        assert not ok
        assert details["computed_lambda"] == pytest.approx(0.5)

    def test_non_pd_P_is_rejected(self) -> None:
        A = -np.eye(2)
        P = -np.eye(2)  # not PD
        ok, details = validate_quadratic(A, P, claimed_lambda=0.1)
        assert not ok
        assert "not positive definite" in details["error"]

    def test_shape_mismatch_returns_structured_error(self) -> None:
        """A and P with different (but individually square) shapes must
        produce ``(False, {"error": ...})`` rather than raising.  Without
        the explicit shape check, ``A.T @ P + P @ A`` would propagate a
        ``ValueError`` out of ``validate_quadratic``, breaking the
        ``validate_lyapunov_from_config`` non-raising contract.
        """
        A = np.diag([-0.25, -0.5])  # 2x2
        P = np.eye(3)  # 3x3 — incompatible
        ok, details = validate_quadratic(A, P, claimed_lambda=0.1)
        assert not ok
        assert "matching shape" in details["error"]
        assert details["claimed_lambda"] == pytest.approx(0.1)


class TestValidateSamples:
    def test_decay_samples_pass(self) -> None:
        samples = [{"x": [1.0], "V": 1.0, "Vdot": -0.5}]
        ok, details = validate_samples(samples, claimed_lambda=0.25)
        assert ok
        assert details["computed_lambda"] == pytest.approx(0.5)

    def test_worst_ratio_dominates(self) -> None:
        samples = [
            {"V": 1.0, "Vdot": -1.0},  # ratio = 1.0
            {"V": 1.0, "Vdot": -0.1},  # ratio = 0.1 (worst)
        ]
        ok, details = validate_samples(samples, claimed_lambda=0.2)
        assert not ok
        assert details["computed_lambda"] == pytest.approx(0.1)

    def test_empty_samples_rejected(self) -> None:
        ok, details = validate_samples([], claimed_lambda=0.0)
        assert not ok
        assert "no samples" in details["error"]

    def test_nonpositive_V_rejected(self) -> None:
        ok, details = validate_samples([{"V": 0.0, "Vdot": -1.0}], claimed_lambda=0.0)
        assert not ok
        assert "V must be > 0" in details["error"]


class TestValidateFromConfig:
    def test_canonical_config_passes(self) -> None:
        """Regression guard for the canonical lambda used repo-wide."""
        ok, details = validate_lyapunov_from_config(_CANONICAL_CFG)
        assert ok, f"canonical Lyapunov config no longer certifies: {details}"
        assert details["mode"] == "quadratic"
        # Canonical claim is 0.25; certified must remain >= 0.25.
        assert details["computed_lambda"] >= 0.25

    def test_missing_config_returns_error(self, tmp_path: Path) -> None:
        ok, details = validate_lyapunov_from_config(tmp_path / "missing.yaml")
        assert not ok
        assert "not found" in details["error"]

    def test_samples_mode(self, tmp_path: Path) -> None:
        cfg = tmp_path / "samples.yaml"
        cfg.write_text(
            "lambda: 0.25\n"
            "lyapunov_samples:\n"
            "  - {V: 1.0, Vdot: -0.5}\n"
            "  - {V: 2.0, Vdot: -1.0}\n"
        )
        ok, details = validate_lyapunov_from_config(cfg)
        assert ok
        assert details["mode"] == "samples"

    def test_neither_matrices_nor_samples(self, tmp_path: Path) -> None:
        cfg = tmp_path / "bad.yaml"
        cfg.write_text("lambda: 0.25\n")
        ok, details = validate_lyapunov_from_config(cfg)
        assert not ok
        assert "neither" in details["error"]

    def test_missing_lambda_field_rejected(self, tmp_path: Path) -> None:
        """A nested ``lyapunov:`` block without ``lambda`` is a config error.

        Defaulting a missing ``lambda`` to ``0.0`` would silently certify
        any stable system (since the worst-case decay rate is almost
        always ``>= 0``); the gate must refuse to proceed instead.
        """
        cfg = tmp_path / "no_lambda.yaml"
        cfg.write_text(
            "lyapunov:\n" "  A: [[-0.25, 0.0], [0.0, -0.5]]\n" "  P: [[1.0, 0.0], [0.0, 1.0]]\n"
        )
        ok, details = validate_lyapunov_from_config(cfg)
        assert not ok
        assert "missing the required `lambda`" in details["error"]

    def test_zero_lambda_rejected(self, tmp_path: Path) -> None:
        cfg = tmp_path / "zero.yaml"
        cfg.write_text(
            "lambda: 0\n" "A: [[-0.25, 0.0], [0.0, -0.5]]\n" "P: [[1.0, 0.0], [0.0, 1.0]]\n"
        )
        ok, details = validate_lyapunov_from_config(cfg)
        assert not ok
        assert "strictly positive" in details["error"]

    def test_negative_lambda_rejected(self, tmp_path: Path) -> None:
        cfg = tmp_path / "neg.yaml"
        cfg.write_text(
            "lambda: -0.5\n" "A: [[-0.25, 0.0], [0.0, -0.5]]\n" "P: [[1.0, 0.0], [0.0, 1.0]]\n"
        )
        ok, details = validate_lyapunov_from_config(cfg)
        assert not ok
        assert "strictly positive" in details["error"]

    def test_non_numeric_lambda_rejected(self, tmp_path: Path) -> None:
        cfg = tmp_path / "bad_type.yaml"
        cfg.write_text(
            "lambda: not_a_number\n"
            "A: [[-0.25, 0.0], [0.0, -0.5]]\n"
            "P: [[1.0, 0.0], [0.0, 1.0]]\n"
        )
        ok, details = validate_lyapunov_from_config(cfg)
        assert not ok
        assert "not numeric" in details["error"]

    def test_nested_lyapunov_block(self, tmp_path: Path) -> None:
        """A nested ``lyapunov:`` block must drive the same validator path."""
        cfg = tmp_path / "ablation.yaml"
        cfg.write_text(
            "baseline:\n"
            "  model: {input_dim: 25}\n"
            "lyapunov:\n"
            "  A: [[-0.25, 0.0], [0.0, -0.5]]\n"
            "  P: [[1.0, 0.0], [0.0, 1.0]]\n"
            "  lambda: 0.25\n"
        )
        ok, details = validate_lyapunov_from_config(cfg)
        assert ok, details
        assert details["mode"] == "quadratic"

    def test_ablation_config_certifies(self) -> None:
        """``configs/ablation_3r_lyapunov.yaml`` ships a valid nested block."""
        ablation = _REPO_ROOT / "configs" / "ablation_3r_lyapunov.yaml"
        ok, details = validate_lyapunov_from_config(ablation)
        assert ok, f"ablation config no longer certifies: {details}"
        assert details["mode"] == "quadratic"

    def test_unreadable_config_returns_structured_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A config that exists but cannot be read (permission / TOCTOU /
        transient FS error) must surface as ``(False, {"error": ...})``,
        not as an unhandled ``OSError``.  Without the explicit catch the
        ``exists()`` check could pass and ``read_text()`` could still
        raise on the very next syscall.
        """
        cfg = tmp_path / "unreadable.yaml"
        cfg.write_text("lambda: 0.25\nA: [[-0.25]]\nP: [[1.0]]\n")

        # Simulate the read-after-exists race by patching ``Path.read_text``
        # to raise PermissionError just for this specific path.  We monkey-
        # patch at the class level so the validator's call site picks up
        # the patched method.  The signature mirrors ``Path.read_text``
        # exactly (encoding + errors keyword args, ``str`` return) so no
        # ``# type: ignore`` escape hatch is needed at the delegate call.
        original_read_text = Path.read_text

        def fake_read_text(
            self: Path,
            encoding: str | None = None,
            errors: str | None = None,
        ) -> str:
            if self == cfg:
                raise PermissionError(13, "simulated permission denied", str(self))
            return original_read_text(self, encoding, errors)

        monkeypatch.setattr(Path, "read_text", fake_read_text)

        ok, details = validate_lyapunov_from_config(cfg)
        assert not ok
        assert "cannot read config" in details["error"]
        assert "simulated permission denied" in details["error"]


class TestExtractLyapunovBlock:
    def test_flat_form_returns_self(self) -> None:
        cfg = {"A": [[1]], "P": [[1]], "lambda": 0.25}
        assert _extract_lyapunov_block(cfg) is cfg

    def test_nested_form_returned(self) -> None:
        cfg = {"variant": {}, "lyapunov": {"A": [[1]], "P": [[1]], "lambda": 0.25}}
        block = _extract_lyapunov_block(cfg)
        assert block == cfg["lyapunov"]

    def test_neither_form_empty(self) -> None:
        assert _extract_lyapunov_block({"unrelated": 1}) == {}


class TestCli:
    def test_cli_canonical_exit_0(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = _cli([str(_CANONICAL_CFG)])
        assert rc == 0
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["mode"] == "quadratic"
        assert payload["ok"] is True

    def test_cli_bad_config_exit_2(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = _cli([str(tmp_path / "missing.yaml")])
        assert rc == 2
        payload = json.loads(capsys.readouterr().out)
        assert "error" in payload

    def test_cli_overclaim_exit_1(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        cfg = tmp_path / "overclaim.yaml"
        cfg.write_text(
            "A: [[-0.25, 0.0], [0.0, -0.5]]\n" "P: [[1.0, 0.0], [0.0, 1.0]]\n" "lambda: 10.0\n"
        )
        rc = _cli([str(cfg)])
        assert rc == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is False

    def test_module_dash_m_invocation(self) -> None:
        """``python -m tools.lyapunov_validator`` must be a real entrypoint."""
        result = subprocess.run(
            [sys.executable, "-m", "tools.lyapunov_validator", str(_CANONICAL_CFG)],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["mode"] == "quadratic"
        assert payload["ok"] is True
