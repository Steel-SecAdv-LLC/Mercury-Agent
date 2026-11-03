# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The provider store: keys, endpoints and models an operator connects from the console.

Before this store, linking a language-model vendor to a served engine meant
editing the engine host's environment and restarting it. The store lets an
administrator do it from the console or the API: ``PUT
/api/v1/voice/backends/{provider}`` stores a key, an endpoint and a model
for a provider; ``POST /api/v1/voice/backends/select`` names the provider the
voice generates prose with; both survive a restart.

What is stored, and how:

* The key is sealed with :class:`~omni_mercury_engine.api.secret_sealer.SecretSealer`
  (AES-256-GCM through the AMA backend, the provider name bound in as AAD),
  under the same stable key the account store uses (``MERCURY_DATA_ENC_KEY``
  or ``AMA_MASTER_SEED``). With no stable key configured the store refuses
  to keep a key at all, with the reason, rather than writing it in the
  clear: a key readable by anyone who can read the file would be worse than
  the environment it replaces.
* The endpoint and the model are plain fields; neither is a secret.
* The file (``MERCURY_PROVIDER_STORE_PATH``, default
  ``$XDG_STATE_HOME/mercury-agent/providers.json``) is written atomically
  with owner-only permissions, inventoried by ``mercury-agent decouple
  status`` and removed by ``decouple purge-work``.
* Rotating the at-rest key: ``scripts/reseal_totp_secrets.py`` re-seals the
  stored provider keys in the same run as the TOTP secrets
  (:meth:`ProviderStore.reseal`), so one rotation covers every secret sealed
  under ``MERCURY_DATA_ENC_KEY``.

The store never puts a key into the process environment: the voice and the
connection probe build an explicit ``LLMConfig`` from it, so a key stored
here reaches exactly the adapter it was stored for, and removing it removes
it everywhere at once. The environment stays the other, equal way to
configure a provider; the backends report says which one is in effect.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omni_mercury_engine.api.secret_sealer import (
    SealedSecretError,
    SecretSealer,
    build_secret_sealer,
)
from omni_mercury_engine.models.llm_registry import PROVIDER_CATALOG

logger = logging.getLogger(__name__)

__all__ = [
    "ENV_VAR",
    "ProviderRecord",
    "ProviderResealReport",
    "ProviderStore",
    "ProviderStoreError",
    "default_store_path",
    "get_provider_store",
]

#: The environment variable naming the store file.
ENV_VAR = "MERCURY_PROVIDER_STORE_PATH"
_VERSION = 1


class ProviderStoreError(RuntimeError):
    """The store cannot do what was asked, with the operator-facing reason."""


def default_store_path() -> Path:
    """``MERCURY_PROVIDER_STORE_PATH``, else ``$XDG_STATE_HOME/mercury-agent/providers.json``."""
    explicit = os.environ.get(ENV_VAR, "").strip()
    if explicit:
        return Path(explicit).expanduser()
    xdg_state = os.environ.get("XDG_STATE_HOME", "").strip()
    root = Path(xdg_state).expanduser() if xdg_state else Path.home() / ".local" / "state"
    return root / "mercury-agent" / "providers.json"


class ProviderRecord(dict[str, Any]):
    """One provider's stored fields: ``has_key``, ``base_url``, ``model``, ``updated_at``."""


@dataclass
class ProviderResealReport:
    """Outcome of re-sealing the stored provider keys under a new at-rest key."""

    with_key: int = 0
    resealed: int = 0
    sealed_plaintext: int = 0
    already_current: int = 0
    #: Providers whose key opened under neither key (left untouched, never overwritten).
    failed: list[str] = field(default_factory=list)
    dry_run: bool = False

    @property
    def changed(self) -> int:
        """How many keys this pass rewrote (or would, under ``dry_run``)."""
        return self.resealed + self.sealed_plaintext


class ProviderStore:
    """A small JSON store of sealed provider keys, endpoints, models and the selection."""

    def __init__(self, path: Path | None = None, sealer: SecretSealer | None = None) -> None:
        """Bind the store to ``path`` (default from the environment) and an optional sealer."""
        self.path = path or default_store_path()
        self._sealer = sealer
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ file
    def _read(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"version": _VERSION, "providers": {}, "selection": None}
        except (OSError, ValueError) as exc:
            raise ProviderStoreError(f"provider store at {self.path} is unreadable: {exc}") from exc
        if not isinstance(raw, dict) or not isinstance(raw.get("providers"), dict):
            raise ProviderStoreError(f"provider store at {self.path} has an unexpected shape")
        raw.setdefault("selection", None)
        return raw

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=".providers-", suffix=".json", dir=str(self.path.parent)
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
            tmp.chmod(0o600)
            tmp.replace(self.path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    # ---------------------------------------------------------------- sealer
    def sealer(self) -> SecretSealer:
        """The at-rest sealer; a non-stable key refuses to store a secret."""
        if self._sealer is None:
            self._sealer = build_secret_sealer(store_is_durable=True)
        return self._sealer

    def can_store_keys(self) -> tuple[bool, str]:
        """Whether a key can be kept across restarts, and why not when it cannot."""
        sealer = self.sealer()
        if getattr(sealer, "key_is_stable", True):
            return True, ""
        return (
            False,
            "no stable at-rest key is configured on this engine (set MERCURY_DATA_ENC_KEY, "
            "python scripts/generate_secret_key.py, or AMA_MASTER_SEED); a provider key "
            "would otherwise be stored unsealed, so the store refuses it. Endpoints and "
            "models can still be stored, and a key can still be set in the environment.",
        )

    # ------------------------------------------------------------------ api
    @staticmethod
    def _check_provider(provider: str) -> None:
        if provider not in PROVIDER_CATALOG:
            raise ProviderStoreError(
                f"unknown provider {provider!r}; one of {', '.join(sorted(PROVIDER_CATALOG))}"
            )

    def get(self, provider: str) -> ProviderRecord | None:
        """The stored fields of ``provider`` without the key, or ``None``."""
        self._check_provider(provider)
        entry = self._read()["providers"].get(provider)
        if not entry:
            return None
        return ProviderRecord(
            has_key=bool(entry.get("api_key")),
            base_url=entry.get("base_url") or None,
            model=entry.get("model") or None,
            updated_at=entry.get("updated_at"),
        )

    def all(self) -> dict[str, ProviderRecord]:
        """Every stored provider's fields, keys omitted."""
        data = self._read()
        out: dict[str, ProviderRecord] = {}
        for provider, entry in data["providers"].items():
            if provider in PROVIDER_CATALOG:
                out[provider] = ProviderRecord(
                    has_key=bool(entry.get("api_key")),
                    base_url=entry.get("base_url") or None,
                    model=entry.get("model") or None,
                    updated_at=entry.get("updated_at"),
                )
        return out

    def key(self, provider: str) -> str | None:
        """The stored key of ``provider``, unsealed, or ``None``."""
        self._check_provider(provider)
        entry = self._read()["providers"].get(provider) or {}
        sealed = entry.get("api_key")
        if not sealed:
            return None
        try:
            return self.sealer().unseal(str(sealed), aad=f"provider:{provider}")
        except SealedSecretError as exc:
            raise ProviderStoreError(
                f"the stored key for {provider} cannot be opened with this engine's at-rest key "
                f"({exc}); store it again"
            ) from exc

    def set(
        self,
        provider: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        now: float | None = None,
    ) -> ProviderRecord:
        """Store the fields given; ``None`` leaves a field alone, ``""`` clears it."""
        import time

        self._check_provider(provider)
        with self._lock:
            data = self._read()
            entry = dict(data["providers"].get(provider) or {})
            if api_key is not None:
                if api_key == "":
                    entry.pop("api_key", None)
                else:
                    ok, why = self.can_store_keys()
                    if not ok:
                        raise ProviderStoreError(why)
                    entry["api_key"] = self.sealer().seal(api_key, aad=f"provider:{provider}")
            if base_url is not None:
                if base_url == "":
                    entry.pop("base_url", None)
                else:
                    entry["base_url"] = base_url
            if model is not None:
                if model == "":
                    entry.pop("model", None)
                else:
                    entry["model"] = model
            entry["updated_at"] = now if now is not None else time.time()
            data["providers"][provider] = entry
            self._write(data)
        logger.info(
            "provider store: %s updated (%s)",
            provider,
            ", ".join(k for k in ("api_key", "base_url", "model") if k in entry),
        )
        record = self.get(provider)
        assert record is not None
        return record

    def delete(self, provider: str) -> bool:
        """Forget everything stored for ``provider``; clear the selection if it named it."""
        self._check_provider(provider)
        with self._lock:
            data = self._read()
            removed = data["providers"].pop(provider, None) is not None
            selection = data.get("selection") or {}
            if selection.get("provider") == provider:
                data["selection"] = None
            if removed or selection.get("provider") == provider:
                self._write(data)
        if removed:
            logger.info("provider store: %s removed", provider)
        return removed

    def selection(self) -> dict[str, Any] | None:
        """The provider and model the voice should generate with, or ``None``."""
        selection = self._read().get("selection")
        if not isinstance(selection, dict) or not selection.get("provider"):
            return None
        return {"provider": str(selection["provider"]), "model": selection.get("model") or None}

    def select(self, provider: str | None, model: str | None = None) -> None:
        """Remember the voice's provider (``None`` returns to the built-in templates)."""
        if provider is not None:
            self._check_provider(provider)
        with self._lock:
            data = self._read()
            data["selection"] = {"provider": provider, "model": model or None} if provider else None
            self._write(data)
        logger.info("provider store: voice selection %s", provider or "cleared")

    def reseal(
        self,
        *,
        new_sealer: SecretSealer,
        old_sealer: SecretSealer | None = None,
        dry_run: bool = False,
    ) -> ProviderResealReport:
        """Re-seal every stored key under ``new_sealer``, opening old ones with ``old_sealer``.

        Rotating ``MERCURY_DATA_ENC_KEY`` without this would leave every stored
        provider key unopenable, since a sealed value opens only under the key
        it was sealed with. Each key is classified as the TOTP rotation does:
        already sealed under the new key (left alone, so a re-run is
        idempotent), sealed under the old key (opened and re-sealed, bound to
        the same ``provider:<name>`` AAD), not sealed at all (a hand-edited
        file; sealed under the new key) or openable under neither (reported,
        never overwritten). The file is rewritten once, atomically, and only
        when something changed.
        """
        report = ProviderResealReport(dry_run=dry_run)
        with self._lock:
            data = self._read()
            changed = False
            for provider, entry in data["providers"].items():
                stored = entry.get("api_key")
                if not stored:
                    continue
                report.with_key += 1
                aad = f"provider:{provider}"
                if not SecretSealer.is_sealed(str(stored)):
                    entry["api_key"] = new_sealer.seal(str(stored), aad=aad)
                    report.sealed_plaintext += 1
                    changed = True
                    continue
                try:
                    new_sealer.unseal(str(stored), aad=aad)
                    report.already_current += 1
                    continue
                except SealedSecretError:
                    pass
                if old_sealer is None:
                    report.failed.append(provider)
                    continue
                try:
                    plaintext = old_sealer.unseal(str(stored), aad=aad)
                except SealedSecretError:
                    report.failed.append(provider)
                    continue
                entry["api_key"] = new_sealer.seal(plaintext, aad=aad)
                report.resealed += 1
                changed = True
            if changed and not dry_run:
                self._write(data)
        if report.changed and not dry_run:
            logger.info(
                "provider store: %d key(s) re-sealed under the new at-rest key", report.changed
            )
        return report

    def config_for(self, provider: str) -> dict[str, Any]:
        """``api_key``, ``base_url`` and ``model`` for an ``LLMConfig``: the store, else nothing.

        The environment is the adapter's own fallback for each field, so a
        missing value here leaves the adapter to read its variable.
        """
        record = self.get(provider)
        return {
            "api_key": self.key(provider) if record and record["has_key"] else None,
            "base_url": record["base_url"] if record else None,
            "model": record["model"] if record else None,
        }


_store: ProviderStore | None = None
_store_lock = threading.Lock()


def get_provider_store() -> ProviderStore:
    """The process-wide store at the configured path (re-read on every access)."""
    global _store
    path = default_store_path()
    with _store_lock:
        if _store is None or _store.path != path:
            _store = ProviderStore(path)
        return _store
