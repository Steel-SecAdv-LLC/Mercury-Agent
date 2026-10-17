# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""DeepFace (and the TensorFlow behind it) loads when an image needs it.

Importing it at module level cost 1.8 s of the 2.0 s it took to construct an
engine, in every CLI call and API worker, for a library only image inputs use.
The module-level ``DeepFace`` name stays patchable.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.models import biometric as biometric_module

_PROBE = """
import json, sys, types
import numpy as np
import omni_mercury_engine.engine as engine


def loaded():
    return sorted(m for m in ("deepface", "tensorflow") if m in sys.modules)


model = engine.OmniMercuryEngine().models["biometric"]
after_construction = loaded()
model.extract_features(np.zeros((4, 8)))
after_tabular = loaded()

touched = []
stub = types.ModuleType("deepface")


class _Face:
    @staticmethod
    def represent(*args, **kwargs):
        raise RuntimeError("stub")


def _getattr(name):
    touched.append(name)
    return _Face


stub.__getattr__ = _getattr
sys.modules["deepface"] = stub
model.extract_features(np.zeros((32, 32, 3), dtype=np.uint8))
first_image = len(touched)
model.extract_features(np.zeros((32, 32, 3), dtype=np.uint8))
print(json.dumps([after_construction, after_tabular, first_image, len(touched)]))
"""


def test_deepface_is_imported_only_for_an_image() -> None:
    proc = subprocess.run(
        [sys.executable, "-I", "-c", _PROBE], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    after_construction, after_tabular, first_image, second_image = json.loads(
        proc.stdout.strip().splitlines()[-1]
    )
    assert after_construction == []  # building the engine
    assert after_tabular == []  # a tabular batch
    assert first_image > 0  # the first image input imports it ...
    assert second_image == first_image  # ... and a second does not import it again


@pytest.fixture
def unresolved(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Reset the lazy name and record every triton preload."""
    calls: list[str] = []
    monkeypatch.setattr(biometric_module, "DeepFace", biometric_module._UNRESOLVED)
    monkeypatch.setattr(
        biometric_module, "preload_triton_before_tensorflow", lambda: calls.append("preload")
    )
    return calls


def test_resolves_once_and_preloads_triton_first(
    unresolved: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Stub:
        pass

    imports: list[str] = []
    fake = types.ModuleType("deepface")

    def _module_getattr(name: str) -> Any:
        imports.append(name)
        return _Stub

    fake.__getattr__ = _module_getattr  # type: ignore[method-assign]
    monkeypatch.setitem(sys.modules, "deepface", fake)

    assert biometric_module._deepface() is _Stub
    assert unresolved == ["preload"]
    assert "DeepFace" in imports

    resolved_after_first_call = len(imports)
    assert biometric_module._deepface() is _Stub
    assert len(imports) == resolved_after_first_call  # no second import


@pytest.mark.parametrize("failure", [None, "oserror"])
def test_an_unusable_deepface_resolves_to_none_and_is_not_retried(
    unresolved: list[str], monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    if failure is None:
        monkeypatch.setitem(sys.modules, "deepface", None)  # import raises ImportError
    else:
        fake = types.ModuleType("deepface")

        def _boom(name: str) -> Any:
            raise OSError("CUDA runtime missing")

        fake.__getattr__ = _boom  # type: ignore[method-assign]
        monkeypatch.setitem(sys.modules, "deepface", fake)

    assert biometric_module._deepface() is None
    assert biometric_module._deepface() is None
    assert unresolved == ["preload"]


def test_tabular_features_never_touch_deepface(unresolved: list[str]) -> None:
    model = biometric_module.BiometricAnomalyModel()
    features = model.extract_features(np.random.default_rng(0).normal(size=(5, 12)))
    assert tuple(features.shape) == (5, 128)
    model.predict(np.zeros((3, 9)))
    assert unresolved == []  # nothing resolved, so triton was never preloaded either


def test_patching_the_module_name_still_controls_the_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(biometric_module, "DeepFace", None)
    assert biometric_module._deepface() is None

    class _Stub:
        pass

    monkeypatch.setattr(biometric_module, "DeepFace", _Stub)
    assert biometric_module._deepface() is _Stub


def test_deepface_available_is_still_a_module_flag(
    unresolved: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flag callers could read before the lazy import still answers, and still lies about nothing."""
    monkeypatch.setattr(biometric_module, "DeepFace", None)
    assert biometric_module.DEEPFACE_AVAILABLE is False

    class _Stub:
        pass

    monkeypatch.setattr(biometric_module, "DeepFace", _Stub)
    assert biometric_module.DEEPFACE_AVAILABLE is True
    with pytest.raises(AttributeError, match="no attribute"):
        biometric_module.NOT_A_FLAG  # noqa: B018
