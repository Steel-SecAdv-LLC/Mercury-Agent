# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the centralized _compat module.

Verifies that:
- All HAS_* flags are boolean
- Convenience groupings are correct logical combinations
- The module is importable without side effects
"""

from __future__ import annotations

from omni_mercury_engine import _compat


class TestCompatFlags:
    """Verify every exported HAS_* flag is a bool."""

    def test_all_flags_are_bool(self) -> None:
        public_flags = [name for name in dir(_compat) if name.startswith("HAS_")]
        assert len(public_flags) > 0, "_compat exports no HAS_* flags"
        for name in public_flags:
            value = getattr(_compat, name)
            assert isinstance(value, bool), f"{name} is {type(value)}, expected bool"

    def test_core_ml_flags_exist(self) -> None:
        assert hasattr(_compat, "HAS_TORCH")
        assert hasattr(_compat, "HAS_TORCHVISION")
        assert hasattr(_compat, "HAS_TIMM")
        assert hasattr(_compat, "HAS_CV2")

    def test_convenience_groupings(self) -> None:
        # HAS_ML_STACK requires only PyTorch (Mercury uses native ML primitives,
        # not sklearn).
        assert _compat.HAS_TORCH == _compat.HAS_ML_STACK
        assert (
            _compat.HAS_TORCH and _compat.HAS_TORCHVISION and _compat.HAS_TIMM
        ) == _compat.HAS_VISUAL_STACK
        assert (
            _compat.HAS_TORCH and _compat.HAS_TRANSFORMERS and _compat.HAS_ACCELERATE
        ) == _compat.HAS_VLM_STACK


class TestCompatNoSideEffects:
    """Verify _compat doesn't import heavy packages."""

    def test_torch_not_in_sys_modules_from_compat(self) -> None:
        """_compat uses find_spec, not import, so torch must not be forced in.

        Run in a fresh interpreter so the check holds even when the test
        session itself (or the [ml] extra) already imported torch.
        """
        import subprocess
        import sys

        code = "import sys; " "import omni_mercury_engine._compat; " "print('torch' in sys.modules)"
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        assert result.stdout.strip() == "False", (
            "importing _compat pulled torch into sys.modules: "
            f"stdout={result.stdout!r} stderr={result.stderr[-500:]!r}"
        )
