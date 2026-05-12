"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

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

------------------------------------------------------------------------

Gate tests for :mod:`omni_mercury_engine.security.safe_exec`.
"""

from __future__ import annotations

import sys

import pytest

from omni_mercury_engine.security.safe_exec import (
    UnsafeSubprocessError,
    python_module,
    safe_exec,
)


class TestArgvValidation:
    def test_string_argv_rejected(self) -> None:
        with pytest.raises(UnsafeSubprocessError, match="sequence of strings"):
            safe_exec("/bin/sh -c whoami")  # type: ignore[arg-type]

    def test_empty_argv_rejected(self) -> None:
        with pytest.raises(UnsafeSubprocessError, match="non-empty"):
            safe_exec([])

    def test_relative_executable_rejected(self) -> None:
        with pytest.raises(UnsafeSubprocessError, match="absolute path"):
            safe_exec(["echo", "hi"])

    def test_nul_byte_in_arg_rejected(self) -> None:
        with pytest.raises(UnsafeSubprocessError, match="NUL byte"):
            safe_exec([sys.executable, "--version\x00; rm -rf /"])

    def test_non_string_arg_rejected(self) -> None:
        with pytest.raises(UnsafeSubprocessError, match="not str"):
            safe_exec([sys.executable, 42])  # type: ignore[list-item]


class TestExecHappyPath:
    def test_runs_python_version(self) -> None:
        result = safe_exec(
            [sys.executable, "-c", "print('safe')"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "safe" in result.stdout


class TestPythonModule:
    def test_bad_module_name_rejected(self) -> None:
        with pytest.raises(UnsafeSubprocessError, match="module name"):
            python_module("evil; rm -rf /")

    def test_module_name_with_dots_accepted(self) -> None:
        # The validation accepts dotted module names (a.b.c). Passing
        # a known module verifies the module-name regex does not
        # reject legitimate paths.
        result = python_module(
            "json.tool",
            args=["--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode in (0, 2)
