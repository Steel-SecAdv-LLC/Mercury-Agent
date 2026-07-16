# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Torch-free contract for :mod:`omni_mercury_engine.detectors._torch_perf`.

``_torch_perf`` is imported unconditionally by torch-optional detectors such
as ``detectors.acceleration_dynamics`` (which itself guards torch behind
``TORCH_AVAILABLE``).  Historically ``_torch_perf`` imported torch at module
top and called ``torch.get_num_threads()`` unconditionally, so a torch-free
install (no ``[ml]`` extra) crashed on import of that detector — silently
defeating its optional-torch design.  This test pins that ``_torch_perf``
imports and runs (as a no-op) with torch genuinely absent, and therefore
fails loudly if an unconditional torch dependency is ever reintroduced.
"""

from __future__ import annotations

import subprocess
import sys

# Pre-import numpy/scipy so their own optional-torch probes settle while torch
# still imports, then block torch to simulate a torch-free install and import
# ``_torch_perf`` under it.  Uses a MetaPathFinder that raises
# ``ModuleNotFoundError`` (a genuine "not installed"), not the
# ``sys.modules["torch"] = None`` sentinel that trips third-party torch probes.
_TORCH_FREE_PROOF = """
import sys, importlib.abc
import numpy, scipy  # noqa: F401  -- settle optional-torch probes first


class _NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("No module named %r" % name)
        return None


for _m in [m for m in list(sys.modules) if m == "torch" or m.startswith("torch.")]:
    del sys.modules[_m]
sys.meta_path.insert(0, _NoTorch())

from omni_mercury_engine.detectors._torch_perf import (
    TORCH_AVAILABLE,
    single_threaded_torch,
)

assert TORCH_AVAILABLE is False, TORCH_AVAILABLE
assert "torch" not in sys.modules, "torch must not be imported"

ran = False
with single_threaded_torch():
    ran = True
assert ran, "context body must run under the torch-free no-op"
print("torch-free-ok")
"""


def test_torch_perf_imports_and_noops_without_torch() -> None:
    """Import + enter the context manager with torch absent; must not raise."""
    result = subprocess.run(
        [sys.executable, "-c", _TORCH_FREE_PROOF],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "torch-free-ok" in result.stdout
