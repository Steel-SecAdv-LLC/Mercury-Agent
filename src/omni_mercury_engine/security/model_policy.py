"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Centralized Hugging Face model / dataset load policy.

The Hugging Face Hub (and any third-party model hub reachable from
``from_pretrained``) is, from Mercury Agent's perspective, an
*untrusted* code distribution surface — a malicious or compromised
repository can publish weights that, when loaded with the default
``transformers`` settings, will execute attacker-controlled Python via
the legacy pickle deserialiser, ``trust_remote_code=True``, or a custom
``modeling.py``.

This module is the single chokepoint every ``from_pretrained`` /
``load_dataset`` call in Mercury Agent flows through.  It enforces, by
construction (not by review):

1. **Identifier shape** — the model/dataset id must match the canonical
   ``namespace/name`` (or single-segment) pattern and resolve to a
   namespace this codebase has explicitly trusted.  Anything else
   raises :class:`UnsafeModelError`.
2. **Revision pinning** — when ``MERCURY_HF_REQUIRE_REVISION`` is on
   (the production default) the call site **must** pass a non-empty
   ``revision``.  Tag/branch revisions are normalised to commit SHAs by
   the caller; this module only enforces presence.
3. **trust_remote_code allow-list** — ``trust_remote_code=True`` is
   permitted **only** when the namespace is in
   :data:`REMOTE_CODE_ALLOWED_NAMESPACES`.  Anywhere else, passing
   ``trust_remote_code=True`` is a hard ``UnsafeModelError``.
4. **Local path opt-out** — operator-managed local checkpoints
   (``/abs/path``, ``./relative``, ``../relative``) are accepted with no
   network policy applied, because the operator has already vouched
   for the file system path.

There is no environment opt-out of the policy itself.  The only
environment switch is ``MERCURY_HF_REQUIRE_REVISION``, which exists so
unit tests and offline development environments can drop revision
pinning while keeping the namespace and remote-code gates intact.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

__all__ = [
    "HFModelPolicy",
    "REMOTE_CODE_ALLOWED_NAMESPACES",
    "TRUSTED_DATASET_NAMESPACES",
    "TRUSTED_NAMESPACES",
    "UnsafeModelError",
]

logger = logging.getLogger(__name__)


class UnsafeModelError(RuntimeError):
    """Raised when a Hugging Face load violates Mercury Agent model policy."""

    def __init__(
        self,
        message: str,
        *,
        model_id: str | None = None,
        revision: str | None = None,
    ) -> None:
        super().__init__(message)
        self.model_id = model_id
        self.revision = revision


# ---------------------------------------------------------------------------
# Trusted namespaces
# ---------------------------------------------------------------------------
# Namespaces on the Hugging Face Hub that Mercury Agent's code references
# directly (default model strings on adapter configs, dataset mirrors used
# by the security loaders).  Adding to this set is a security review
# decision: every entry has been vetted as a maintained, published
# project from a known-good organization or research group.
TRUSTED_NAMESPACES: frozenset[str] = frozenset(
    {
        # Major foundation-model publishers
        "amazon",
        "facebook",
        "google",
        "meta-llama",
        "microsoft",
        "mistralai",
        "openai-community",
        "Salesforce",
        "stabilityai",
        # Vision-language model families used by detectors/vlm/*
        "Qwen",
        "llava-hf",
        "openbmb",
        "OpenGVLab",
        # Time-series foundation models
        "huggingface",
    }
)

# Subset of namespaces whose repositories are allowed to publish models
# requiring ``trust_remote_code=True``.  These are projects that legitimately
# ship a custom modeling.py (e.g. MiniCPM-V) and whose maintainers we
# trust to publish unsigned code.  Every other namespace is rejected
# outright if trust_remote_code=True is requested.
REMOTE_CODE_ALLOWED_NAMESPACES: frozenset[str] = frozenset(
    {
        "openbmb",  # MiniCPM-V ships custom modeling.py
        "OpenGVLab",  # InternVL family ships custom code
        "Qwen",  # some Qwen2-VL revisions ship custom code
    }
)

# Namespaces we allow as published *dataset* mirrors of public research
# corpora.  Community mirrors live here, and an entry in this set is
# narrower than TRUSTED_NAMESPACES: presence here means "we accept this
# user as a CICIDS / NSL-KDD / etc. mirror", not "we accept arbitrary
# code from this user".
TRUSTED_DATASET_NAMESPACES: frozenset[str] = frozenset(
    {
        "bvk",  # bvk/CICIDS-2017
        "Riccorl",  # Riccorl/CIC-IDS-2017
        "tcabanski",  # tcabanski/cicids2017
        "huggingface",
        "google",
        "facebook",
        "Salesforce",
    }
)

# Canonical ``namespace/name`` (or single-segment) pattern.  Matches the
# Hugging Face id rules: 1-64 chars per segment, ASCII letters/digits/
# ``_.-``.  This is intentionally a hard restriction — anything outside
# this shape is rejected before we touch the network.
_HF_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}(/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?$")


def _is_local_path(model_id: str) -> bool:
    """Return True if ``model_id`` is an operator-managed local path."""
    return (
        model_id.startswith("/")
        or model_id.startswith("./")
        or model_id.startswith("../")
        or os.path.sep in model_id
        and os.path.isabs(model_id)
    )


def _require_revision_active() -> bool:
    """Read MERCURY_HF_REQUIRE_REVISION, defaulting to True."""
    raw = os.environ.get("MERCURY_HF_REQUIRE_REVISION", "true")
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


class HFModelPolicy:
    """
    Single chokepoint for every Hugging Face ``from_pretrained`` /
    ``load_dataset`` invocation in Mercury Agent.

    Usage
    -----

    ::

        from omni_mercury_engine.security.model_policy import HFModelPolicy

        HFModelPolicy.validate(model_id, revision=revision, trust_remote_code=False)
        model = AutoModel.from_pretrained(model_id, revision=revision)

    The validator must run on the line immediately above the
    ``from_pretrained`` call.  The exception type ``UnsafeModelError``
    is a ``RuntimeError`` subclass so existing exception handlers that
    catch ``Exception`` continue to abort the load — there is no
    silent-fallback path through the policy.
    """

    @staticmethod
    def _split(model_id: str) -> tuple[str, str]:
        """Return ``(namespace, name)`` or raise ``UnsafeModelError``.

        Single-segment ids (``"bert-base-uncased"`` legacy style) get
        the synthetic namespace ``"_legacy"`` so they never accidentally
        land in any trusted set.
        """
        if not isinstance(model_id, str) or not model_id:
            raise UnsafeModelError(
                "HFModelPolicy: model id must be a non-empty string",
                model_id=str(model_id),
            )
        if not _HF_ID_PATTERN.match(model_id):
            raise UnsafeModelError(
                f"HFModelPolicy: malformed model id {model_id!r}",
                model_id=model_id,
            )
        if "/" in model_id:
            ns, name = model_id.split("/", 1)
            return ns, name
        return "_legacy", model_id

    @classmethod
    def validate(
        cls,
        model_id: str,
        *,
        revision: str | None = None,
        trust_remote_code: bool = False,
        allow_namespaces: frozenset[str] | None = None,
    ) -> None:
        # Fail fast on non-string input; downstream string ops would
        # otherwise TypeError before the policy logic runs.
        if not isinstance(model_id, str) or not model_id:
            raise UnsafeModelError(
                "HFModelPolicy: model id must be a non-empty string",
                model_id=str(model_id),
            )
        """Validate a planned HuggingFace ``from_pretrained`` call.

        Args:
            model_id: The HuggingFace model identifier (``"namespace/name"``)
                or absolute / relative local filesystem path.
            revision: The pinned revision (commit SHA or tag) the caller
                will pass to ``from_pretrained``.  Required when
                ``MERCURY_HF_REQUIRE_REVISION`` is on (production
                default).
            trust_remote_code: Whether the caller will pass
                ``trust_remote_code=True``.  Permitted only when the
                namespace is in :data:`REMOTE_CODE_ALLOWED_NAMESPACES`.
            allow_namespaces: Override the trusted-namespace set for
                this call.  Used by ``validate_vetted_dataset`` to
                accept the dataset namespaces without widening the
                model-loading allow-list.

        Raises:
            UnsafeModelError: On any policy violation.  The caller must
                not catch this exception to silently fall back; the
                whole point of the policy is that the load aborts.
        """
        if _is_local_path(model_id):
            # Local paths bypass the network policy.  The operator vouched
            # for the path; trust_remote_code is still meaningful (the
            # local file may contain remote_code) so re-validate it.
            if trust_remote_code:
                logger.warning(
                    "HFModelPolicy: trust_remote_code=True on local path %r "
                    "(no namespace check available; relying on operator "
                    "control of the file system)",
                    model_id,
                )
            return

        namespace, name = cls._split(model_id)

        permitted = allow_namespaces if allow_namespaces is not None else TRUSTED_NAMESPACES
        if namespace not in permitted:
            raise UnsafeModelError(
                f"HFModelPolicy: namespace {namespace!r} is not in the trusted "
                f"set for this load. id={model_id!r}, name={name!r}. "
                f"Trusted: {sorted(permitted)}",
                model_id=model_id,
                revision=revision,
            )

        if _require_revision_active():
            if not revision or not str(revision).strip():
                raise UnsafeModelError(
                    "HFModelPolicy: revision pinning is mandatory "
                    "(MERCURY_HF_REQUIRE_REVISION=true). "
                    f"Pass a commit SHA or tag to from_pretrained for {model_id!r}.",
                    model_id=model_id,
                    revision=revision,
                )

        if trust_remote_code:
            if namespace not in REMOTE_CODE_ALLOWED_NAMESPACES:
                raise UnsafeModelError(
                    f"HFModelPolicy: trust_remote_code=True is not permitted for "
                    f"namespace {namespace!r}. Allowed namespaces: "
                    f"{sorted(REMOTE_CODE_ALLOWED_NAMESPACES)}.",
                    model_id=model_id,
                    revision=revision,
                )

    @classmethod
    def validate_vetted_dataset(
        cls,
        dataset_id: str,
        *,
        revision: str | None = None,
    ) -> None:
        """Validate a ``datasets.load_dataset`` call.

        Mirror of :meth:`validate`, but checks the dataset namespace
        allow-list rather than the model namespace allow-list.
        ``trust_remote_code`` is not exposed because ``load_dataset``
        with a custom loader script is a separate, harder-to-defend
        surface that Mercury Agent does not currently use.
        """
        cls.validate(
            dataset_id,
            revision=revision,
            trust_remote_code=False,
            allow_namespaces=TRUSTED_DATASET_NAMESPACES,
        )

    @classmethod
    def assert_namespace_trusted(cls, model_id: str) -> tuple[str, str]:
        """Convenience: split + namespace-check without revision policy.

        Useful when the caller needs the parsed ``(namespace, name)`` for
        downstream logging *and* the cheaper structural validation.
        Callers that go on to load weights must still use :meth:`validate`.
        """
        if _is_local_path(model_id):
            # Local path: parse opaquely as ``(parent_dir, basename)`` so the
            # caller still gets a useful 2-tuple for logging.  No namespace
            # check applies — the operator vouched for the filesystem path.
            parent = os.path.dirname(model_id) or "."
            base = os.path.basename(model_id) or model_id
            return parent, base
        ns, name = cls._split(model_id)
        if ns not in TRUSTED_NAMESPACES:
            raise UnsafeModelError(
                f"HFModelPolicy: namespace {ns!r} not in trusted set",
                model_id=model_id,
            )
        return ns, name

    @classmethod
    def policy_summary(cls) -> dict[str, Any]:
        """Return a debuggable snapshot of the active policy."""
        return {
            "require_revision": _require_revision_active(),
            "trusted_namespaces": sorted(TRUSTED_NAMESPACES),
            "trusted_dataset_namespaces": sorted(TRUSTED_DATASET_NAMESPACES),
            "remote_code_allowed_namespaces": sorted(REMOTE_CODE_ALLOWED_NAMESPACES),
        }
