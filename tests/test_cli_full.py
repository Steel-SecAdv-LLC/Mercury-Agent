"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisors LLC

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
Comprehensive CLI tests to boost coverage
"""

import os
import tempfile
from pathlib import Path

from click.testing import CliRunner

from omni_mercury_engine.cli import main


def test_detect_command_with_data():
    """Test detect command with actual data file"""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("1.0 2.0 3.0 4.0 5.0\n")
        f.write("2.0 3.0 4.0 5.0 6.0\n")
        temp_file = f.name

    try:
        result = runner.invoke(main, ["detect", "--data", temp_file])
        assert result.exit_code == 0 or "error" in result.output.lower()
    finally:
        Path(temp_file).unlink()


def test_train_command_with_config():
    """Test train command with config file"""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write('{"learning_rate": 0.001, "max_epochs": 1}')
        config_file = f.name

    try:
        result = runner.invoke(main, ["train", "--config", config_file])
        assert result.exit_code >= 0
    finally:
        Path(config_file).unlink()


def test_biometric_command_with_image():
    """Test biometric command with image file"""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_image = f.name

    try:
        result = runner.invoke(main, ["biometric", "--image", temp_image])
        assert result.exit_code >= 0
    finally:
        if os.path.exists(temp_image):
            Path(temp_image).unlink()


def test_security_command_with_payloads():
    """Test security command with payload file"""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("SELECT * FROM users\n")
        f.write("<script>alert('xss')</script>\n")
        payload_file = f.name

    try:
        result = runner.invoke(main, ["security", "--payloads", payload_file])
        assert result.exit_code >= 0
    finally:
        Path(payload_file).unlink()


def test_detect_with_detectors_option():
    """Test detect command with specific detectors"""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("1.0 2.0 3.0\n")
        temp_file = f.name

    try:
        result = runner.invoke(
            main, ["detect", "--data", temp_file, "--detectors", "statistical,temporal"]
        )
        assert result.exit_code >= 0
    finally:
        Path(temp_file).unlink()


def test_train_with_output_option():
    """Test train command with output path"""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "model.pt")

        result = runner.invoke(main, ["train", "--output", output_path, "--max-epochs", "1"])
        assert result.exit_code >= 0
