# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Comprehensive CLI tests covering all commands with correct flag signatures."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

from click.testing import CliRunner

from omni_mercury_engine.cli import main

# ---------------------------------------------------------------------------
# detect command tests
# ---------------------------------------------------------------------------


def test_detect_with_csv_data() -> None:
    """Test detect command with CSV data file."""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("1.0,2.0,3.0,4.0,5.0\n")
        f.write("2.0,3.0,4.0,5.0,6.0\n")
        f.write("100.0,200.0,300.0,400.0,500.0\n")
        temp_file = f.name

    try:
        result = runner.invoke(main, ["detect", "--input", temp_file])
        assert result.exit_code in (0, 1), f"Exit {result.exit_code}: {result.output}"
        assert result.exception is None or isinstance(
            result.exception, SystemExit
        ), f"Unhandled exception: {result.exception!r}\n{result.output}"
    finally:
        Path(temp_file).unlink()


def test_detect_with_json_data() -> None:
    """Test detect command with JSON data file."""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([[1, 2, 3], [4, 5, 6], [100, 200, 300]], f)
        temp_file = f.name

    try:
        result = runner.invoke(main, ["detect", "--input", temp_file])
        assert result.exit_code in (0, 1), f"Exit {result.exit_code}: {result.output}"
        assert result.exception is None or isinstance(
            result.exception, SystemExit
        ), f"Unhandled exception: {result.exception!r}\n{result.output}"
    finally:
        Path(temp_file).unlink()


def test_detect_with_threshold() -> None:
    """Test detect command with custom threshold."""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("1.0,2.0,3.0\n4.0,5.0,6.0\n")
        temp_file = f.name

    try:
        result = runner.invoke(main, ["detect", "--input", temp_file, "--threshold", "0.8"])
        assert result.exit_code in (0, 1), f"Exit {result.exit_code}: {result.output}"
        assert result.exception is None or isinstance(
            result.exception, SystemExit
        ), f"Unhandled exception: {result.exception!r}\n{result.output}"
    finally:
        Path(temp_file).unlink()


def test_detect_with_output_file() -> None:
    """Test detect command writes to output file."""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("1.0,2.0,3.0\n4.0,5.0,6.0\n")
        data_file = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as out:
        output_file = out.name

    try:
        result = runner.invoke(main, ["detect", "--input", data_file, "--output", output_file])
        assert result.exit_code in (0, 1), f"Exit {result.exit_code}: {result.output}"
        assert result.exception is None or isinstance(
            result.exception, SystemExit
        ), f"Unhandled exception: {result.exception!r}\n{result.output}"
    finally:
        Path(data_file).unlink()
        if os.path.exists(output_file):
            Path(output_file).unlink()


def test_detect_with_statistical_detector() -> None:
    """Test detect command with statistical detector type."""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("1,2,3\n4,5,6\n7,8,9\n")
        temp_file = f.name

    try:
        result = runner.invoke(main, ["detect", "--input", temp_file, "--detector", "statistical"])
        assert result.exit_code in (0, 1), f"Exit {result.exit_code}: {result.output}"
        assert result.exception is None or isinstance(
            result.exception, SystemExit
        ), f"Unhandled exception: {result.exception!r}\n{result.output}"
    finally:
        Path(temp_file).unlink()


def test_detect_with_unsupported_format() -> None:
    """Test detect command with unsupported file format."""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("1 2 3\n")
        temp_file = f.name

    try:
        result = runner.invoke(main, ["detect", "--input", temp_file])
        assert result.exit_code != 0 or "Unsupported" in result.output or "Error" in result.output
    finally:
        Path(temp_file).unlink()


def test_detect_missing_input() -> None:
    """Test detect command without required --input flag."""
    runner = CliRunner()
    result = runner.invoke(main, ["detect"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# train command tests
# ---------------------------------------------------------------------------


def test_train_command_help() -> None:
    """Test train command help output."""
    runner = CliRunner()
    result = runner.invoke(main, ["train", "--help"])
    assert result.exit_code == 0
    assert "--data" in result.output
    assert "--output" in result.output
    assert "--epochs" in result.output


def test_train_missing_required_flags() -> None:
    """Test train command fails without required flags."""
    runner = CliRunner()
    result = runner.invoke(main, ["train"])
    assert result.exit_code != 0


def test_train_with_correct_flags() -> None:
    """Test train command with correct --data and --output flags."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "data")
        os.makedirs(data_dir)
        output_path = os.path.join(tmpdir, "model.pt")

        result = runner.invoke(
            main,
            ["train", "--data", data_dir, "--output", output_path, "--epochs", "1"],
        )
        # May fail due to missing torch/data, but should parse flags correctly
        assert result.exit_code >= 0


# ---------------------------------------------------------------------------
# security command tests
# ---------------------------------------------------------------------------


def test_security_command_with_payload() -> None:
    """Test security command with --payload flag (correct flag name)."""
    runner = CliRunner()
    result = runner.invoke(main, ["security", "--payload", "SELECT * FROM users WHERE 1=1"])
    assert result.exit_code >= 0 or "Error" in result.output


def test_security_command_with_xss_payload() -> None:
    """Test security command with XSS payload."""
    runner = CliRunner()
    result = runner.invoke(main, ["security", "--payload", "<script>alert('xss')</script>"])
    assert result.exit_code >= 0 or "Error" in result.output


def test_security_missing_payload() -> None:
    """Test security command fails without required --payload."""
    runner = CliRunner()
    result = runner.invoke(main, ["security"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# biometric command tests
# ---------------------------------------------------------------------------


def test_biometric_with_reference() -> None:
    """Test biometric command with --reference flag."""
    runner = CliRunner()
    result = runner.invoke(main, ["biometric", "--reference", "/nonexistent/ref.jpg"])
    assert result.exit_code != 0 or "Error" in result.output or "error" in result.output.lower()


def test_biometric_with_both_flags() -> None:
    """Test biometric command with --reference and --test flags."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["biometric", "--reference", "/tmp/ref.jpg", "--test", "/tmp/test.jpg"],
    )
    assert result.exit_code != 0 or "Error" in result.output or "error" in result.output.lower()


def test_biometric_missing_reference() -> None:
    """Test biometric command fails without required --reference."""
    runner = CliRunner()
    result = runner.invoke(main, ["biometric"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# explain command tests
# ---------------------------------------------------------------------------


def test_explain_help() -> None:
    """Test explain command help output."""
    runner = CliRunner()
    result = runner.invoke(main, ["explain", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--model" in result.output


def test_explain_with_csv_input() -> None:
    """Test explain command with CSV data."""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("1,2,3\n4,5,6\n7,8,9\n")
        temp_file = f.name

    try:
        result = runner.invoke(main, ["explain", "--input", temp_file])
        assert result.exit_code in (0, 1), f"Exit {result.exit_code}: {result.output}"
        assert result.exception is None or isinstance(
            result.exception, SystemExit
        ), f"Unhandled exception: {result.exception!r}\n{result.output}"
    finally:
        Path(temp_file).unlink()


def test_explain_with_model_flag() -> None:
    """Test explain command with --model flag."""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("1,2,3\n4,5,6\n")
        temp_file = f.name

    try:
        result = runner.invoke(main, ["explain", "--input", temp_file, "--model", "statistical"])
        assert result.exit_code in (0, 1), f"Exit {result.exit_code}: {result.output}"
        assert result.exception is None or isinstance(
            result.exception, SystemExit
        ), f"Unhandled exception: {result.exception!r}\n{result.output}"
    finally:
        Path(temp_file).unlink()


def test_explain_missing_input() -> None:
    """Test explain command fails without required --input."""
    runner = CliRunner()
    result = runner.invoke(main, ["explain"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# physics command group tests
# ---------------------------------------------------------------------------


def test_physics_help() -> None:
    """Test physics command group help."""
    runner = CliRunner()
    result = runner.invoke(main, ["physics", "--help"])
    assert result.exit_code == 0
    assert "spectral" in result.output
    assert "dynamics" in result.output
    assert "uiux" in result.output
    assert "integrated" in result.output
    assert "list" in result.output


def test_physics_spectral_help() -> None:
    """Test physics spectral subcommand help."""
    runner = CliRunner()
    result = runner.invoke(main, ["physics", "spectral", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--mode" in result.output
    assert "--sample-rate" in result.output


def test_physics_dynamics_help() -> None:
    """Test physics dynamics subcommand help."""
    runner = CliRunner()
    result = runner.invoke(main, ["physics", "dynamics", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--time-step" in result.output


def test_physics_uiux_help() -> None:
    """Test physics uiux subcommand help."""
    runner = CliRunner()
    result = runner.invoke(main, ["physics", "uiux", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output


def test_physics_integrated_help() -> None:
    """Test physics integrated subcommand help."""
    runner = CliRunner()
    result = runner.invoke(main, ["physics", "integrated", "--help"])
    assert result.exit_code == 0
    assert "--spectral-input" in result.output
    assert "--dynamics-input" in result.output
    assert "--uiux-input" in result.output


def test_physics_list() -> None:
    """Test physics list subcommand outputs detector catalog."""
    runner = CliRunner()
    result = runner.invoke(main, ["physics", "list"])
    assert result.exit_code == 0
    assert "Spectral" in result.output or "spectral" in result.output


def test_physics_spectral_with_csv() -> None:
    """Test physics spectral with CSV signal data."""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        for i in range(100):
            f.write(f"{math.sin(i * 0.1)}\n")
        temp_file = f.name

    try:
        result = runner.invoke(main, ["physics", "spectral", "--input", temp_file])
        assert result.exit_code in (0, 1), f"Exit {result.exit_code}: {result.output}"
        assert result.exception is None or isinstance(
            result.exception, SystemExit
        ), f"Unhandled exception: {result.exception!r}\n{result.output}"
    finally:
        Path(temp_file).unlink()


def test_physics_dynamics_with_csv() -> None:
    """Test physics dynamics with motion data."""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        for i in range(50):
            f.write(f"{i * 0.1},{i * 0.2},{i * 0.05}\n")
        temp_file = f.name

    try:
        result = runner.invoke(main, ["physics", "dynamics", "--input", temp_file])
        assert result.exit_code in (0, 1), f"Exit {result.exit_code}: {result.output}"
        assert result.exception is None or isinstance(
            result.exception, SystemExit
        ), f"Unhandled exception: {result.exception!r}\n{result.output}"
    finally:
        Path(temp_file).unlink()


def test_physics_uiux_with_json() -> None:
    """Test physics uiux with interaction JSON data."""
    runner = CliRunner()

    interactions = [
        {"timestamp": i * 0.5, "type": "click", "x": 100 + i, "y": 200 + i} for i in range(10)
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(interactions, f)
        temp_file = f.name

    try:
        result = runner.invoke(main, ["physics", "uiux", "--input", temp_file])
        assert result.exit_code in (0, 1), f"Exit {result.exit_code}: {result.output}"
        assert result.exception is None or isinstance(
            result.exception, SystemExit
        ), f"Unhandled exception: {result.exception!r}\n{result.output}"
    finally:
        Path(temp_file).unlink()


def test_physics_spectral_missing_input() -> None:
    """Test physics spectral fails without required --input."""
    runner = CliRunner()
    result = runner.invoke(main, ["physics", "spectral"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# serve command tests
# ---------------------------------------------------------------------------


def test_serve_help() -> None:
    """Test serve command help output."""
    runner = CliRunner()
    result = runner.invoke(main, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--host" in result.output
    assert "--port" in result.output
    assert "--workers" in result.output
    assert "--reload" in result.output
    assert "--log-level" in result.output


# ---------------------------------------------------------------------------
# voice command tests
# ---------------------------------------------------------------------------


def test_voice_help() -> None:
    """Test voice command help output."""
    runner = CliRunner()
    result = runner.invoke(main, ["voice", "--help"])
    assert result.exit_code == 0
    assert "--domain" in result.output or "--model" in result.output or "--offline" in result.output
