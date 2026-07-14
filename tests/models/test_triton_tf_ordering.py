# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Native-library ordering guard: TensorFlow must never precede triton.

TensorFlow and triton (torch's compiler backend) each bundle an LLVM.
Importing triton *after* TensorFlow hard-segfaults the process during
``libtriton`` initialisation (observed: tensorflow 2.21 + triton 3.7.1).
In an ``[all]`` install on Python <= 3.13 the detector registry hit
exactly that sequence — deepface loaded TensorFlow, then a
torchvision-backed detector import reached ``torchvision.ops`` ->
``torch._dynamo`` -> the triton probe -> SIGSEGV.  The cure is
``_compat.preload_triton_before_tensorflow()`` at Mercury's TensorFlow
entry points (``models/biometric*.py``).

These tests run the hazardous sequence in a subprocess (a segfault must
not kill the test runner) and only when the full colliding stack is
actually installed.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

_HAS_COLLIDING_STACK = all(
    importlib.util.find_spec(mod) is not None for mod in ("deepface", "torchvision", "triton")
)

_BIOMETRIC_THEN_TORCHVISION = """
import logging
logging.disable(logging.CRITICAL)
import omni_mercury_engine.models.biometric_advanced  # pulls TensorFlow via deepface
import torchvision.ops  # imports torch._dynamo -> triton probe
print("ordering-ok")
"""


@pytest.mark.skipif(
    not _HAS_COLLIDING_STACK,
    reason="needs deepface + torchvision + triton installed to reproduce the LLVM collision",
)
def test_biometric_then_torchvision_does_not_segfault() -> None:
    """The historical registry-discovery crash sequence must survive."""
    result = subprocess.run(
        [sys.executable, "-c", _BIOMETRIC_THEN_TORCHVISION],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, (
        f"returncode={result.returncode} (negative = killed by signal; -11 = SIGSEGV)\n"
        f"stdout={result.stdout}\nstderr={result.stderr[-2000:]}"
    )
    assert "ordering-ok" in result.stdout


def test_preload_helper_is_exception_safe() -> None:
    """The preload helper must never raise, whatever the environment."""
    from omni_mercury_engine._compat import preload_triton_before_tensorflow

    preload_triton_before_tensorflow()
    preload_triton_before_tensorflow()  # idempotent
