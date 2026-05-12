"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.

------------------------------------------------------------------------

HuggingFace model / dataset policy gate.

Centralises every ``from_pretrained`` and ``datasets.load_dataset``
call in Mercury Agent through a single helper that enforces:

* **Identifier shape** -- model IDs must match the ``namespace/name``
  pattern that HuggingFace Hub itself uses, or be an absolute local
  path.  Bare names like ``"bert"`` (which would resolve to a
  community-namespace model with no provenance) are rejected.
* **Revision pinning** -- every remote load must specify a non-empty
  ``revision``.  Without this, supply-chain attacks against the
  default branch ``main`` can swap weights from under us (CWE-494).
* **trust_remote_code stays off** -- ``trust_remote_code=True``
  executes arbitrary code from the repo and is only permitted when
  the model ID is in the explicit allowlist passed to the helper.

Every B615 ``from_pretrained`` call in ``src/`` now goes through
:class:`SafeHFLoader`; the single bandit-suppression for B615 lives
on the underlying ``from_pretrained`` call inside this module.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


# A HuggingFace Hub model id has the shape ``namespace/name`` where
# both components match this character set.  See:
# https://huggingface.co/docs/hub/repositories-naming
_HF_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+$")


class UnsafeModelError(ValueError):
    """An attempted HuggingFace load violated the policy gate."""


def _is_local_path(model_id: str) -> bool:
    """Return True if ``model_id`` is an absolute local filesystem path.

    Relative paths (``./foo``, ``../foo``) are *not* treated as local
    -- they would make resolution depend on the current working
    directory, which is exactly the kind of ambient state we refuse
    to load weights from.  An operator who really wants to load from
    disk must pass an absolute path.
    """
    return model_id.startswith("/")


class HFModelPolicy:
    """
    Static policy gate for HuggingFace model / dataset loads.

    Used internally by :class:`SafeHFLoader`.  Exposed as a public
    class so callers can run ``HFModelPolicy.validate(...)`` at
    config-validation time (e.g. at engine startup) before any
    network IO.
    """

    @classmethod
    def validate(
        cls,
        model_id: str,
        *,
        revision: str | None,
        allowlist: Iterable[str] | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        """Enforce identifier shape, revision pinning, and trust_remote_code policy.

        Args:
            model_id: HuggingFace Hub identifier or an absolute
                local path.
            revision: Revision (commit SHA, tag, or branch).
                Required for Hub IDs; ignored for local paths.
            allowlist: If supplied, the model id must appear here
                for the load to proceed.  Use for downstream
                callers that want to restrict which HF repos a
                given subsystem may touch (e.g. VLM detectors
                allowlist their model id at the top of the file).
            trust_remote_code: If True, the model id must be in
                ``allowlist`` (so the operator has explicitly
                acknowledged that this repo is permitted to ship
                executable code in its config).

        Raises:
            UnsafeModelError: any gate failed.
        """
        if not isinstance(model_id, str) or not model_id:
            raise UnsafeModelError("HFModelPolicy: model_id must be a non-empty string.")

        is_local = _is_local_path(model_id)
        if not is_local and not _HF_ID_RE.match(model_id):
            raise UnsafeModelError(
                f"HFModelPolicy: model_id '{model_id}' does not match "
                "the HuggingFace Hub namespace/name pattern, and is not "
                "an absolute local path. Refusing to resolve to a "
                "community-namespace default."
            )

        # Hub IDs must be revision-pinned. Local paths are fine
        # without a revision -- they don't go through the Hub
        # resolver at all.
        if not is_local and (revision is None or not str(revision).strip()):
            raise UnsafeModelError(
                f"HFModelPolicy: model_id '{model_id}' requires a non-empty "
                "revision (commit SHA preferred). Unpinned loads expose "
                "Mercury Agent to supply-chain swaps on the default branch."
            )

        if allowlist is not None:
            allow_set = {entry.strip() for entry in allowlist if entry}
            if model_id not in allow_set:
                raise UnsafeModelError(
                    f"HFModelPolicy: model_id '{model_id}' is not in the "
                    f"caller's allowlist {sorted(allow_set)}."
                )

        if trust_remote_code and allowlist is None:
            raise UnsafeModelError(
                "HFModelPolicy: trust_remote_code=True requires an explicit "
                "allowlist. Refusing to execute arbitrary repo code without "
                "operator opt-in."
            )


class SafeHFLoader:
    """
    The single from_pretrained / load_dataset entry point.

    Each method:

    1. Calls :meth:`HFModelPolicy.validate`.
    2. Calls the underlying ``transformers`` / ``datasets``
       function inside this helper.  The bandit suppression for
       B615 lives here, exactly once.

    Callers never reference ``transformers.*.from_pretrained`` or
    ``datasets.load_dataset`` directly -- they import this class.
    """

    @classmethod
    def load_model(
        cls,
        cls_: Any,
        model_id: str,
        *,
        revision: str | None,
        allowlist: Iterable[str] | None = None,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Load a transformers model class via its ``from_pretrained``.

        Args:
            cls_: The transformers class (e.g. ``AutoModelForCausalLM``,
                ``BlipForConditionalGeneration``).
            model_id: HuggingFace Hub id or local path.
            revision: Revision to pin to (Hub ids only).
            allowlist: Optional id allowlist for this caller.
            trust_remote_code: Forwarded to the underlying call.
                Requires ``allowlist`` to be set.
            **kwargs: Forwarded verbatim to ``from_pretrained``.
        """
        HFModelPolicy.validate(
            model_id,
            revision=revision,
            allowlist=allowlist,
            trust_remote_code=trust_remote_code,
        )
        effective_revision = revision if not _is_local_path(model_id) else None
        # B615: the call is gated by HFModelPolicy.validate above.
        # This is the only annotated from_pretrained in src/.
        return cls_.from_pretrained(  # nosec B615
            model_id,
            revision=effective_revision,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    @classmethod
    def load_processor(
        cls,
        cls_: Any,
        model_id: str,
        *,
        revision: str | None,
        allowlist: Iterable[str] | None = None,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Load a transformers processor (BlipProcessor, AutoProcessor, ...)."""
        return cls.load_model(
            cls_,
            model_id,
            revision=revision,
            allowlist=allowlist,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    @classmethod
    def load_tokenizer(
        cls,
        cls_: Any,
        model_id: str,
        *,
        revision: str | None,
        allowlist: Iterable[str] | None = None,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Load a transformers tokenizer (AutoTokenizer, BertTokenizer, ...)."""
        return cls.load_model(
            cls_,
            model_id,
            revision=revision,
            allowlist=allowlist,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    @classmethod
    def load_dataset(
        cls,
        dataset_id: str,
        *,
        allowlist: Iterable[str],
        revision: str | None,
        **kwargs: Any,
    ) -> Any:
        """
        Load a ``datasets`` dataset under the same gates.

        The ``allowlist`` is required for dataset loads: there is
        no equivalent of "trusted huggingface-internal datasets",
        so every dataset id must be acknowledged by the caller.

        Args:
            dataset_id: HuggingFace Hub dataset id (``namespace/name``).
            allowlist: Required iterable of permitted dataset ids.
            revision: Required revision to pin to.
            **kwargs: Forwarded verbatim to ``datasets.load_dataset``.
        """
        HFModelPolicy.validate(
            dataset_id,
            revision=revision,
            allowlist=allowlist,
            trust_remote_code=False,
        )
        # Lazy-imported so importing this module does not require the
        # ``datasets`` package to be installed.
        from datasets import load_dataset as _hf_load_dataset

        return _hf_load_dataset(
            dataset_id,
            revision=revision,
            **kwargs,
        )


__all__ = [
    "HFModelPolicy",
    "SafeHFLoader",
    "UnsafeModelError",
]
