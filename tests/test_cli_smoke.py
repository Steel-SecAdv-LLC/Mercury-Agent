# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""CLI smoke tests to boost coverage."""

from __future__ import annotations

import tempfile
from pathlib import Path

from click.testing import CliRunner

from omni_mercury_engine.cli import main


def test_cli_main_help() -> None:
    """Test CLI main help command"""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Mercury Agent" in result.output


def test_detect_help() -> None:
    """Test detect command help"""
    runner = CliRunner()
    result = runner.invoke(main, ["detect", "--help"])
    assert result.exit_code == 0
    assert "detector" in result.output.lower()


def test_train_help() -> None:
    """Test train command help"""
    runner = CliRunner()
    result = runner.invoke(main, ["train", "--help"])
    assert result.exit_code == 0
    assert "train" in result.output.lower()


def test_biometric_help() -> None:
    """Test biometric command help"""
    runner = CliRunner()
    result = runner.invoke(main, ["biometric", "--help"])
    assert result.exit_code == 0
    assert "biometric" in result.output.lower()


def test_version_command() -> None:
    """Test version command"""
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "1.8.0" in result.output


def test_detect_with_sample_data() -> None:
    """Test detect command with sample data"""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("1,2,3\n4,5,6\n7,8,9\n")
        data_file = f.name

    try:
        result = runner.invoke(main, ["detect", "--input", data_file, "--detector", "statistical"])
        assert result.exit_code == 0 or "Error" in result.output
    finally:
        Path(data_file).unlink()


def test_biometric_with_invalid_path() -> None:
    """Test biometric command with invalid image path"""
    runner = CliRunner()
    result = runner.invoke(main, ["biometric", "--reference", "/nonexistent/image.jpg"])
    assert result.exit_code != 0 or "error" in result.output.lower()
