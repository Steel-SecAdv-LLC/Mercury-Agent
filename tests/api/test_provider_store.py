# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The provider store keeps keys sealed, refuses to keep them unsealed, and survives restarts.

A key stored for a provider is sealed under the engine's stable at-rest key
with the provider name bound in; without a stable key the store refuses the
key with the reason instead of writing it in the clear. Endpoints and models
are plain fields. The file is owner-only and rewritten atomically, a second
store instance at the same path reads what the first wrote, the selection
names the provider the voice generates with, deleting a provider clears a
selection that named it, and ``config_for`` hands the voice an explicit
configuration so no key ever enters the environment.
"""

from __future__ import annotations

import json
import os
import stat
from typing import TYPE_CHECKING

import pytest

from omni_mercury_engine.api import provider_store as module
from omni_mercury_engine.api.provider_store import ProviderStore, ProviderStoreError

if TYPE_CHECKING:
    from pathlib import Path

STABLE_KEY = "7f" * 32


@pytest.fixture
def stable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MERCURY_DATA_ENC_KEY", STABLE_KEY)
    monkeypatch.delenv("AMA_MASTER_SEED", raising=False)


@pytest.fixture
def unstable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MERCURY_DATA_ENC_KEY", raising=False)
    monkeypatch.delenv("AMA_MASTER_SEED", raising=False)


def test_a_key_is_sealed_at_rest_and_opens_again(stable: None, tmp_path: Path) -> None:
    store = ProviderStore(tmp_path / "providers.json")
    record = store.set("groq", api_key="gsk-test-not-real-0123456789", model="llama-3.3-70b")
    assert record["has_key"] is True and record["model"] == "llama-3.3-70b"
    raw = (tmp_path / "providers.json").read_text(encoding="utf-8")
    assert "gsk-test-not-real" not in raw
    assert json.loads(raw)["providers"]["groq"]["api_key"].startswith("enc$v1$")
    assert store.key("groq") == "gsk-test-not-real-0123456789"
    again = ProviderStore(tmp_path / "providers.json")
    assert again.key("groq") == "gsk-test-not-real-0123456789"
    assert again.config_for("groq") == {
        "api_key": "gsk-test-not-real-0123456789",
        "base_url": None,
        "model": "llama-3.3-70b",
    }


@pytest.mark.skipif(
    os.name != "posix",
    reason="0o600 is a POSIX mode; Windows chmod sets no ACL, and the store "
    "logs that the restriction is not enforced there (secrets stay sealed)",
)
def test_the_file_is_owner_only(stable: None, tmp_path: Path) -> None:
    store = ProviderStore(tmp_path / "nested" / "providers.json")
    store.set("mistral", model="mistral-large-latest")
    mode = stat.S_IMODE((tmp_path / "nested" / "providers.json").stat().st_mode)
    assert mode == 0o600


def test_without_a_stable_key_a_secret_is_refused_but_fields_are_kept(
    unstable: None, tmp_path: Path
) -> None:
    store = ProviderStore(tmp_path / "providers.json")
    ok, why = store.can_store_keys()
    assert ok is False and "MERCURY_DATA_ENC_KEY" in why
    with pytest.raises(ProviderStoreError, match="MERCURY_DATA_ENC_KEY"):
        store.set("openai", api_key="sk-test-not-real")
    assert not (tmp_path / "providers.json").exists()
    record = store.set("openai", model="gpt-4o")
    assert record["has_key"] is False and record["model"] == "gpt-4o"


def test_a_sealed_key_bound_to_another_provider_does_not_open(stable: None, tmp_path: Path) -> None:
    store = ProviderStore(tmp_path / "providers.json")
    store.set("openai", api_key="sk-test-not-real")
    data = json.loads((tmp_path / "providers.json").read_text(encoding="utf-8"))
    data["providers"]["xai"] = dict(data["providers"]["openai"])
    (tmp_path / "providers.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ProviderStoreError, match="cannot be opened"):
        store.key("xai")


def test_empty_strings_clear_fields_and_none_leaves_them(stable: None, tmp_path: Path) -> None:
    store = ProviderStore(tmp_path / "providers.json")
    store.set("azure_openai", api_key="k", base_url="https://res.openai.azure.com", model="dep")
    store.set("azure_openai", model="dep2")
    record = store.get("azure_openai")
    assert record is not None
    assert record == {
        "has_key": True,
        "base_url": "https://res.openai.azure.com",
        "model": "dep2",
        "updated_at": record["updated_at"],
    }
    store.set("azure_openai", api_key="", base_url="")
    record = store.get("azure_openai")
    assert record is not None
    assert record["has_key"] is False and record["base_url"] is None and record["model"] == "dep2"


def test_selection_persists_and_is_cleared_with_its_provider(stable: None, tmp_path: Path) -> None:
    store = ProviderStore(tmp_path / "providers.json")
    assert store.selection() is None
    store.set("anthropic", api_key="k", model="claude-sonnet-4-5")
    store.select("anthropic", "claude-sonnet-4-5")
    assert ProviderStore(tmp_path / "providers.json").selection() == {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
    }
    assert store.delete("anthropic") is True
    assert store.selection() is None and store.get("anthropic") is None
    assert store.delete("anthropic") is False
    store.select(None)
    assert store.selection() is None


def test_unknown_providers_are_refused(stable: None, tmp_path: Path) -> None:
    store = ProviderStore(tmp_path / "providers.json")
    with pytest.raises(ProviderStoreError, match="unknown provider"):
        store.set("not-a-provider", model="x")
    with pytest.raises(ProviderStoreError, match="unknown provider"):
        store.select("not-a-provider")


def test_the_default_path_follows_xdg_state_and_the_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(module.ENV_VAR, raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    assert module.default_store_path() == tmp_path / "state" / "mercury-agent" / "providers.json"
    monkeypatch.setenv(module.ENV_VAR, str(tmp_path / "custom.json"))
    assert module.default_store_path() == tmp_path / "custom.json"
    assert module.get_provider_store().path == tmp_path / "custom.json"


def test_a_corrupt_file_is_reported_not_silently_emptied(stable: None, tmp_path: Path) -> None:
    path = tmp_path / "providers.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ProviderStoreError, match="unreadable"):
        ProviderStore(path).get("openai")
