# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every ``mercury-agent ...`` example in a command's --help is valid for the installed CLI.

22 example lines in ``serve``, ``voice``, ``stream`` and the four ``physics``
commands told users to run ``mercury ...``; the package installs only
``mercury-agent`` (``[project.scripts]``), so each one failed with "command
not found". This pins that examples use the installed name and that click
accepts every one of them (command path, options, types, choices).
"""

from __future__ import annotations

import re
import shlex
import tomllib
from pathlib import Path

import click
import pytest

from omni_mercury_engine.cli import main

_REPO = Path(__file__).resolve().parents[1]
_CLI = _REPO / "src" / "omni_mercury_engine" / "cli.py"


def _examples() -> list[str]:
    text = _CLI.read_text(encoding="utf-8").replace("\\\\\n", " ")
    found = re.findall(r"(?m)^\s+mercury-agent ([^\n#]+?)\s*(?:#.*)?$", text)
    return [re.sub(r"\s{2,}", " ", e).strip() for e in found]


def test_examples_use_the_installed_binary_name() -> None:
    scripts = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "scripts"
    ]
    assert list(scripts) == ["mercury-agent"]
    stale = re.findall(r"(?m)^\s+mercury (?=[a-z])", _CLI.read_text(encoding="utf-8"))
    assert not stale, f"{len(stale)} docstring example(s) still call 'mercury'"


def test_example_scan_is_not_vacuous() -> None:
    assert len(_examples()) >= 31


@pytest.mark.parametrize("example", _examples())
def test_example_parses(example: str) -> None:
    args = shlex.split(example)
    command: click.Command = main
    path: list[str] = []
    while isinstance(command, click.Group) and args and args[0] in command.commands:
        path.append(args[0])
        command = command.commands[args.pop(0)]
    assert path, f"no command in {example!r}"
    with command.make_context(" ".join(["mercury-agent", *path]), args, resilient_parsing=False):
        pass
