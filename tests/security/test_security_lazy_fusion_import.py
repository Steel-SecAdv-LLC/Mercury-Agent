# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``security.IntelligenceFusionEngine`` is ``None`` only when torch is absent.

Regression: the lazy resolver caught every ``ImportError`` from importing the
fusion module and reported it as "torch not installed", so a missing scipy or
a broken submodule silently disabled fusion instead of failing. Torch-free:
the import is intercepted, so this runs in the no-torch CI lane too.
"""

from __future__ import annotations

import importlib.util
import sys
from importlib.abc import MetaPathFinder
from typing import TYPE_CHECKING

import pytest

from omni_mercury_engine import security

if TYPE_CHECKING:
    from collections.abc import Sequence
    from importlib.machinery import ModuleSpec
    from types import ModuleType

FUSION = "omni_mercury_engine.security.intelligence_fusion"


class _FailingImport(MetaPathFinder):
    """Make importing the fusion module raise ``ModuleNotFoundError(name=missing)``."""

    def __init__(self, missing: str) -> None:
        self.missing = missing

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if fullname == FUSION:
            raise ModuleNotFoundError(f"No module named {self.missing!r}", name=self.missing)
        return None


def _resolve_with_missing(monkeypatch: pytest.MonkeyPatch, missing: str) -> object:
    monkeypatch.delitem(sys.modules, FUSION, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_FailingImport(missing), *sys.meta_path])
    return security.__getattr__("IntelligenceFusionEngine")


@pytest.mark.parametrize("missing", ["torch", "torch.nn"])
def test_missing_torch_resolves_to_none(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    assert _resolve_with_missing(monkeypatch, missing) is None


@pytest.mark.parametrize("missing", ["scipy", "omni_mercury_engine.security.counterintelligence"])
def test_other_missing_modules_are_not_reported_as_missing_torch(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    with pytest.raises(ModuleNotFoundError) as caught:
        _resolve_with_missing(monkeypatch, missing)
    assert caught.value.name == missing


def test_real_resolution_matches_torch_availability() -> None:
    engine_cls = security.__getattr__("IntelligenceFusionEngine")
    if importlib.util.find_spec("torch") is None:
        assert engine_cls is None
    else:
        assert isinstance(engine_cls, type)
        assert engine_cls.__name__ == "IntelligenceFusionEngine"
