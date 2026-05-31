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

# An immutable revision is a 40-character lowercase git commit SHA.
# Branch names ("main", "develop") and tag names are NOT immutable --
# the upstream repo owner can rotate them with no detectable change to
# the consumer. The whole point of the revision pin is to prevent that
# rotation from silently swapping the model / dataset weights under us;
# accepting a branch name is a security regression dressed up as a pin.
# HuggingFace Hub exposes the SHA via ``HfApi.list_repo_commits()`` or
# the UI; operators MUST resolve to a SHA before pinning here.
_HF_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class UnsafeModelError(ValueError):
    """An attempted HuggingFace load violated the policy gate."""


def _is_local_path(model_id: str) -> bool:
    r"""Return True if ``model_id`` is an absolute local filesystem path.

    Uses :meth:`pathlib.PurePosixPath.is_absolute` and
    :meth:`pathlib.PureWindowsPath.is_absolute` so the check is
    cross-platform: POSIX absolute paths (``/opt/models/foo``),
    Windows absolute paths (``C:\models\foo``), and UNC paths
    (``\\server\share\foo``) are all detected as local.  Relative
    paths (``./foo``, ``../foo``, ``foo``) are *not* treated as local
    -- they would make resolution depend on the current working
    directory, which is exactly the kind of ambient state we refuse
    to load weights from.  An operator who really wants to load from
    disk must pass an absolute path.

    A HuggingFace Hub id is the shape ``namespace/name`` and is never
    absolute on either POSIX or Windows under this rule, so this
    helper cleanly partitions the input space.
    """
    from pathlib import PurePosixPath, PureWindowsPath

    return PurePosixPath(model_id).is_absolute() or PureWindowsPath(model_id).is_absolute()


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
            revision: 40-character lowercase git commit SHA. Required
                for Hub IDs; ignored for local paths. Branch names
                and tag names are NOT accepted -- HuggingFace Hub
                does not enforce tag immutability and any moving
                reference defeats the supply-chain guarantee this
                gate provides (CWE-494). Resolve a branch or tag to
                a SHA via ``HfApi.list_repo_commits(repo_id,
                revision=ref)[0].commit_id`` before passing it here.
            allowlist: If supplied, the model id must appear here
                for the load to proceed.  Use for downstream
                callers that want to restrict which HF repos a
                given subsystem may touch (e.g. VLM detectors
                allowlist their model id at the top of the file).
                Absolute local paths bypass this allowlist (they
                are operator-trusted by their absolute form); the
                allowlist is only consulted for Hub identifiers.
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

        # Hub IDs must be pinned to an *immutable* revision. A 40-char
        # lowercase commit SHA is the only string we accept here --
        # branch names ("main", "develop") are mutable references the
        # upstream owner can rotate at will, and tag names are also
        # mutable on HuggingFace Hub (no signed-tag enforcement). The
        # whole purpose of the pin is to defeat that rotation, so
        # accepting "main" would defeat the gate it claims to enforce.
        # Local paths are exempt: they never go through the Hub
        # resolver and are operator-trusted by their absolute form.
        if not is_local:
            if revision is None or not str(revision).strip():
                raise UnsafeModelError(
                    f"HFModelPolicy: model_id '{model_id}' requires a 40-char "
                    "git commit SHA. Pass the immutable SHA, not a branch or "
                    "tag -- branches/tags are mutable references the upstream "
                    "owner can rotate, exposing Mercury Agent to supply-chain "
                    "swaps (CWE-494)."
                )
            rev = str(revision).strip()
            if not _HF_SHA_RE.match(rev):
                raise UnsafeModelError(
                    f"HFModelPolicy: revision '{rev}' for '{model_id}' is not "
                    "a 40-char lowercase commit SHA. Branch names and tag "
                    "names are mutable on HuggingFace Hub; the upstream owner "
                    "can rotate them and silently swap the resolved weights. "
                    "Resolve the branch/tag to a SHA via "
                    "``HfApi.list_repo_commits(repo_id, revision=branch)[0].commit_id`` "
                    "or the HuggingFace web UI, and pin that SHA here."
                )

        # Local paths skip the Hub allowlist entirely. The allowlist
        # exists to constrain which *upstream* repos a subsystem may
        # touch; an absolute local path is by construction not an
        # upstream repo. Without this carve-out the documented
        # ``/opt/models/foo`` escape hatch would be unreachable for any
        # caller that also passes a Hub-id allowlist (BLIP, Chronos,
        # LVLM all do).
        if not is_local and allowlist is not None:
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
        # ``cls_`` arrives as a parameter, so bandit's B615 static
        # check cannot match this call site (and a suppression
        # marker here would warn as unused). The equivalent
        # enforcement happens dynamically in
        # ``HFModelPolicy.validate`` above: revision pinning,
        # identifier shape, allowlist, trust_remote_code.
        return cls_.from_pretrained(
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

        return _hf_load_dataset(  # nosec B615 - HFModelPolicy.validate enforces SHA pinning.
            dataset_id,
            revision=revision,
            **kwargs,
        )


__all__ = [
    "HFModelPolicy",
    "SafeHFLoader",
    "UnsafeModelError",
]
