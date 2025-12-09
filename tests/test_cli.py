"""
OMNI ♱ AVA (O♱A)
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
Test CLI functionality
"""

from click.testing import CliRunner

from omni_anomaly_engine.cli import main


def test_cli_help():
    """Test CLI help command"""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "OMNI ♱ AVA" in result.output


def test_detect_help():
    """Test detect command help"""
    runner = CliRunner()
    result = runner.invoke(main, ["detect", "--help"])
    assert result.exit_code == 0


def test_train_help():
    """Test train command help"""
    runner = CliRunner()
    result = runner.invoke(main, ["train", "--help"])
    assert result.exit_code == 0


def test_biometric_help():
    """Test biometric command help"""
    runner = CliRunner()
    result = runner.invoke(main, ["biometric", "--help"])
    assert result.exit_code == 0


def test_security_help():
    """Test security command help"""
    runner = CliRunner()
    result = runner.invoke(main, ["security", "--help"])
    assert result.exit_code == 0
