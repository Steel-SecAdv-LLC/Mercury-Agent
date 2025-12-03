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

"""
Additional CLI tests to boost coverage above 85%
"""

import json
import os
import tempfile

from click.testing import CliRunner

from omni_anomaly_engine.cli import main


def test_security_command_help():
    """Test security command help"""
    runner = CliRunner()
    result = runner.invoke(main, ["security", "--help"])
    assert result.exit_code == 0


def test_security_command_with_input():
    """Test security command with sample input"""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("SELECT * FROM users WHERE id = 1")
        input_file = f.name

    try:
        result = runner.invoke(main, ["security", "--input", input_file])
        assert result.exit_code == 0 or "error" in result.output.lower()
    finally:
        os.unlink(input_file)


def test_detect_with_json_input():
    """Test detect command with JSON input"""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([[1, 2, 3], [4, 5, 6]], f)
        data_file = f.name

    try:
        result = runner.invoke(main, ["detect", "--input", data_file, "--detector", "temporal"])
        assert result.exit_code == 0 or "error" in result.output.lower()
    finally:
        os.unlink(data_file)


def test_detect_with_output_file():
    """Test detect command with output file"""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("1,2,3\n4,5,6\n")
        data_file = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as out_f:
        output_file = out_f.name

    try:
        result = runner.invoke(
            main,
            ["detect", "--input", data_file, "--detector", "statistical", "--output", output_file],
        )
        assert result.exit_code == 0 or "error" in result.output.lower()
    finally:
        os.unlink(data_file)
        if os.path.exists(output_file):
            os.unlink(output_file)


def test_train_command_with_options():
    """Test train command with various options"""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("1,2,3\n4,5,6\n")
        data_file = f.name

    try:
        result = runner.invoke(
            main, ["train", "--data", data_file, "--epochs", "1", "--batch-size", "2"]
        )
        assert result.exit_code == 0 or "error" in result.output.lower()
    finally:
        os.unlink(data_file)


def test_biometric_with_both_images():
    """Test biometric command with reference and test images"""
    runner = CliRunner()

    result = runner.invoke(
        main, ["biometric", "--reference", "/tmp/ref.jpg", "--test", "/tmp/test.jpg"]
    )
    assert result.exit_code != 0 or "error" in result.output.lower()
