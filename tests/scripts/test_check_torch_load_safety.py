# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the ``check_torch_load_safety`` CI gate.

The gate is the breaker that keeps ``safe_torch_load`` un-bypassable: it must
flag any real ``torch.load(`` call in ``src/`` that is not the sanctioned
wrapper, while never flagging the same text inside a docstring or comment.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_PATH = REPO_ROOT / "scripts" / "check_torch_load_safety.py"

_spec = importlib.util.spec_from_file_location("check_torch_load_safety", GATE_PATH)
assert _spec is not None and _spec.loader is not None
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def _write(tmp: Path, name: str, body: str) -> Path:
    p = tmp / name
    p.write_text(body, encoding="utf-8")
    return p


class TestFindCalls:
    def test_detects_single_line_call(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", "import torch\nx = torch.load('m.pt')\n")
        assert gate.find_torch_load_calls(p) == [2]

    def test_detects_multiline_call(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            "b.py",
            "import torch\nx = torch.load(\n    'm.pt', weights_only=True\n)\n",
        )
        assert gate.find_torch_load_calls(p) == [2]

    def test_ignores_docstring_mention(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            "c.py",
            '"""Explains torch.load(weights_only=True) in prose."""\nY = 1\n',
        )
        assert gate.find_torch_load_calls(p) == []

    def test_ignores_comment_mention(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "d.py", "Y = 1  # never call torch.load(x) directly\n")
        assert gate.find_torch_load_calls(p) == []

    def test_ignores_safe_wrapper_name(self, tmp_path: Path) -> None:
        # safe_torch_load(...) must not be confused with torch.load(...).
        p = _write(tmp_path, "e.py", "x = safe_torch_load('m.pt')\n")
        assert gate.find_torch_load_calls(p) == []

    def test_multiple_calls_all_reported(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            "f.py",
            "import torch\na = torch.load('1.pt')\nb = torch.load('2.pt')\n",
        )
        assert gate.find_torch_load_calls(p) == [2, 3]


class TestScan:
    def test_clean_tree_passes(self, tmp_path: Path) -> None:
        _write(tmp_path, "ok.py", "x = safe_torch_load('m.pt')\n")
        violations, errors = gate.scan(tmp_path)
        assert violations == []
        assert errors == []

    def test_violation_detected(self, tmp_path: Path) -> None:
        _write(tmp_path, "bad.py", "import torch\nx = torch.load('m.pt')\n")
        violations, errors = gate.scan(tmp_path)
        assert errors == []
        assert len(violations) == 1
        assert violations[0][0].endswith("bad.py")
        assert violations[0][1] == 2


class TestRealTree:
    def test_repository_src_is_clean(self) -> None:
        # The live gate on the real source tree must pass: every checkpoint
        # load goes through the wrapper.
        rc = gate.main(["--root", "src"])
        assert rc == 0

    def test_wrapper_is_the_only_allowlisted_file(self) -> None:
        assert frozenset({"src/omni_mercury_engine/security/safe_torch.py"}) == gate.ALLOWLIST
