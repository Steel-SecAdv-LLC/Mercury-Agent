"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Tests that every ``from_pretrained`` / ``load_dataset`` callsite is
actually routed through ``HFModelPolicy.validate(...)`` and that the
policy refusal propagates rather than getting swallowed.

The tests do not exercise transformers/datasets themselves; we monkey
patch the network-touching call so we can observe (1) that the
validator is called first, and (2) that an ``UnsafeModelError`` aborts
the load instead of falling back to a stub.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from omni_mercury_engine.security.model_policy import HFModelPolicy, UnsafeModelError


@pytest.fixture(autouse=True)
def _enable_revision_pinning():
    """Run these tests under the production-default revision policy."""
    with patch.dict(os.environ, {"MERCURY_HF_REQUIRE_REVISION": "true"}, clear=False):
        yield


# ---------------------------------------------------------------------------
# datasets/security.py::HuggingFace mirror loop
# ---------------------------------------------------------------------------


def test_security_dataset_loader_calls_policy_validate(monkeypatch):
    """The HuggingFace mirror branch validates each mirror id before load_dataset."""
    pytest.importorskip("datasets", reason="huggingface 'datasets' not installed")
    from omni_mercury_engine.datasets import security as sec_mod

    seen_ids = []

    def fake_validate(ds_id, **kwargs):
        seen_ids.append(ds_id)

    fake_load = MagicMock(side_effect=RuntimeError("network disabled in test"))

    monkeypatch.setattr(HFModelPolicy, "validate_vetted_dataset", fake_validate)
    monkeypatch.setattr(sec_mod, "logger", MagicMock())

    # Inject a fake load_dataset into the lazy import the loader does.
    fake_datasets_module = MagicMock()
    fake_datasets_module.load_dataset = fake_load
    monkeypatch.setitem(__import__("sys").modules, "datasets", fake_datasets_module)

    cfg = sec_mod.DatasetConfig(
        name="cicids2017",
        data_dir="/tmp/_mercury_test_data",
        cache_dir="/tmp/_mercury_test_cache",
        download=True,
    )
    loader = sec_mod.CICIDSLoader(cfg, binary_labels=True)
    # No cache yet, so the HF mirror branch runs.
    loader._download_from_huggingface()

    assert seen_ids, "validate_vetted_dataset was never called"
    assert "bvk/CICIDS-2017" in seen_ids or seen_ids[0].endswith(
        ("CICIDS-2017", "cicids2017")
    )


def test_security_dataset_loader_propagates_unsafe_model_error(monkeypatch):
    """An UnsafeModelError must NOT silently fall through to the next mirror."""
    pytest.importorskip("datasets", reason="huggingface 'datasets' not installed")
    from omni_mercury_engine.datasets import security as sec_mod

    def raising(*_a, **_kw):
        raise UnsafeModelError("simulated policy denial", model_id="attacker/poison")

    fake_load = MagicMock()
    monkeypatch.setattr(HFModelPolicy, "validate_vetted_dataset", raising)

    fake_datasets_module = MagicMock()
    fake_datasets_module.load_dataset = fake_load
    monkeypatch.setitem(__import__("sys").modules, "datasets", fake_datasets_module)

    cfg = sec_mod.DatasetConfig(
        name="cicids2017",
        data_dir="/tmp/_mercury_test_data2",
        cache_dir="/tmp/_mercury_test_cache2",
        download=True,
    )
    loader = sec_mod.CICIDSLoader(cfg, binary_labels=True)
    with pytest.raises(UnsafeModelError):
        loader._download_from_huggingface()
    fake_load.assert_not_called()


# ---------------------------------------------------------------------------
# detectors/vlm/lvlm_backends.py
# ---------------------------------------------------------------------------


def test_qwen2vl_backend_calls_policy(monkeypatch):
    pytest.importorskip("transformers")
    from omni_mercury_engine.detectors.vlm import lvlm_backends

    called = {"validate": False, "from_pretrained": False}

    def fake_validate(model_id, *, revision, trust_remote_code, **_):
        assert model_id == "Qwen/Qwen2-VL-Instruct"
        assert revision == "deadbeef"
        assert trust_remote_code is True
        called["validate"] = True

    monkeypatch.setattr(HFModelPolicy, "validate", fake_validate)
    fake_proc = MagicMock()
    fake_model_cls = MagicMock()

    def from_pretrained_proc(*a, **kw):
        called["from_pretrained"] = True
        return MagicMock()

    fake_proc.from_pretrained = from_pretrained_proc
    fake_model_cls.from_pretrained = MagicMock(return_value=MagicMock())

    fake_tx = MagicMock()
    fake_tx.AutoProcessor = fake_proc
    fake_tx.Qwen2VLForConditionalGeneration = fake_model_cls
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_tx)

    backend = lvlm_backends.Qwen2VLBackend(
        model_name="Qwen/Qwen2-VL-Instruct",
        device="cpu",
        revision="deadbeef",
    )
    backend.initialize()
    assert called["validate"], "HFModelPolicy.validate was not called"
    assert called["from_pretrained"], "from_pretrained never reached"


def test_qwen2vl_backend_blocks_on_policy_refusal(monkeypatch):
    pytest.importorskip("transformers")
    from omni_mercury_engine.detectors.vlm import lvlm_backends

    def raising(*_a, **_kw):
        raise UnsafeModelError("simulated policy denial")

    fake_from_pretrained = MagicMock()
    monkeypatch.setattr(HFModelPolicy, "validate", raising)

    fake_tx = MagicMock()
    fake_tx.AutoProcessor = MagicMock(from_pretrained=fake_from_pretrained)
    fake_tx.Qwen2VLForConditionalGeneration = MagicMock(from_pretrained=fake_from_pretrained)
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_tx)

    backend = lvlm_backends.Qwen2VLBackend(
        model_name="attacker/evil-model",
        device="cpu",
        revision=None,
    )
    with pytest.raises(UnsafeModelError):
        backend.initialize()
    fake_from_pretrained.assert_not_called()


# ---------------------------------------------------------------------------
# models/foundation/llm_adapter.py
# ---------------------------------------------------------------------------


def test_llm_adapter_calls_policy(monkeypatch):
    """``HuggingFaceLocalAdapter._load_model`` validates before from_pretrained."""
    pytest.importorskip("transformers")
    from omni_mercury_engine.models.foundation import llm_adapter

    called = {"validate": False, "tokenizer": False, "model": False}

    def fake_validate(model_id, *, revision, trust_remote_code, **_):
        assert model_id == "meta-llama/Llama-3.2-3B"
        assert revision == "abc123"
        called["validate"] = True

    monkeypatch.setattr(HFModelPolicy, "validate", fake_validate)

    fake_tx = MagicMock()
    fake_tok_cls = MagicMock()
    fake_tok_cls.from_pretrained = MagicMock(return_value=MagicMock())
    fake_model_cls = MagicMock()
    fake_model_cls.from_pretrained = MagicMock(return_value=MagicMock())

    def tok_ret(*a, **kw):
        called["tokenizer"] = True
        return MagicMock()

    def model_ret(*a, **kw):
        called["model"] = True
        return MagicMock()

    fake_tok_cls.from_pretrained.side_effect = tok_ret
    fake_model_cls.from_pretrained.side_effect = model_ret

    fake_tx.AutoTokenizer = fake_tok_cls
    fake_tx.AutoModelForCausalLM = fake_model_cls
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_tx)

    cfg = llm_adapter.LLMConfig(
        provider=llm_adapter.LLMProvider.HUGGINGFACE,
        model_name="meta-llama/Llama-3.2-3B",
        revision="abc123",
    )
    adapter = llm_adapter.HuggingFaceLocalAdapter(cfg)
    adapter._is_available = True
    adapter._load_model()
    assert called["validate"]
    assert called["tokenizer"]
    assert called["model"]
