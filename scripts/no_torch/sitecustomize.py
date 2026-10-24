# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Make torch behave as if it were not installed, for the local CI mirror.

CI's ``pqc-production-check`` lane installs only ``[dev,api]``, so torch is
absent there; ``scripts/run_ci_gates.sh`` reproduces that lane by putting
this directory on ``PYTHONPATH``. ``importlib.util.find_spec("torch")``
returns ``None`` and ``import torch`` raises ``ModuleNotFoundError``,
exactly as on a host without the ``[ml]`` extra.
"""

import sys
from importlib.machinery import PathFinder

_HIDDEN = frozenset({"torch", "torchvision", "torchaudio"})


class _NoTorchPathFinder(PathFinder):
    @classmethod
    def find_spec(cls, name, path=None, target=None):  # type: ignore[no-untyped-def]
        if name.split(".")[0] in _HIDDEN:
            return None
        return PathFinder.find_spec(name, path, target)


sys.meta_path[:] = [_NoTorchPathFinder if f is PathFinder else f for f in sys.meta_path]
