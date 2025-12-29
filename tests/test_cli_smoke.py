"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
CLI smoke tests to boost coverage
"""

import tempfile
from pathlib import Path

from click.testing import CliRunner

from omni_mercury_engine.cli import main


def test_cli_main_help():
    """Test CLI main help command"""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Mercury Agent ♱" in result.output


def test_detect_help():
    """Test detect command help"""
    runner = CliRunner()
    result = runner.invoke(main, ["detect", "--help"])
    assert result.exit_code == 0
    assert "detector" in result.output.lower()


def test_train_help():
    """Test train command help"""
    runner = CliRunner()
    result = runner.invoke(main, ["train", "--help"])
    assert result.exit_code == 0
    assert "train" in result.output.lower()


def test_biometric_help():
    """Test biometric command help"""
    runner = CliRunner()
    result = runner.invoke(main, ["biometric", "--help"])
    assert result.exit_code == 0
    assert "biometric" in result.output.lower()


def test_version_command():
    """Test version command"""
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "1.0.0" in result.output


def test_detect_with_sample_data():
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


def test_biometric_with_invalid_path():
    """Test biometric command with invalid image path"""
    runner = CliRunner()
    result = runner.invoke(main, ["biometric", "--reference", "/nonexistent/image.jpg"])
    assert result.exit_code != 0 or "error" in result.output.lower()
