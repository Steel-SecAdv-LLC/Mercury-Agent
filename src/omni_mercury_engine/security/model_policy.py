"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Model-policy gate for HuggingFace Hub loads (Bandit B615).

HuggingFace ``from_pretrained`` is a remote-code surface: it pulls
artifacts from the Hub, can run arbitrary Python when
``trust_remote_code=True`` is set, and silently follows whatever
revision the Hub currently serves unless an explicit ``revision`` is
pinned. Bandit's B615 finding warns about exactly that risk.

This module centralises the policy that authorises ``from_pretrained``
calls so individual call-sites only need a single guard call and a
nosec annotation can clearly reference the policy:

    HFModelPolicy.validate(model_id, revision=rev, trust_remote_code=False)
    model = AutoModel.from_pretrained(model_id, revision=rev)  # nosec B615

Policy enforced:

1. **Local paths** (absolute or relative, or anything with a path
   separator) are accepted without further checks — operators may
   load models off-disk.
2. **Remote model IDs** must be ``<namespace>/<name>`` with the
   namespace in :data:`TRUSTED_NAMESPACES`. The full ID is
   double-checked against a permissive identifier regex so the value
   cannot smuggle shell/SQL/path characters into downstream tooling.
3. **Revision pinning** is mandatory for remote IDs in production
   (``MERCURY_HF_REQUIRE_REVISION=true``, the default for non-test
   environments). A revision is accepted as either a 7+ char hex
   commit SHA or a tag-like identifier (``[A-Za-z0-9._-]+``).
4. **trust_remote_code** is rejected unless the namespace appears in
   :data:`REMOTE_CODE_ALLOWED_NAMESPACES`. This stops adversaries from
   substituting a new namespace and getting arbitrary code execution.

If the policy rejects a load, :class:`UnsafeModelError` is raised so
the call-site fails closed rather than degrading silently.
"""

from __future__ import annotations

import os
import re
from typing import Final

__all__ = [
    "REMOTE_CODE_ALLOWED_NAMESPACES",
    "TRUSTED_NAMESPACES",
    "HFModelPolicy",
    "UnsafeModelError",
]


class UnsafeModelError(ValueError):
    """Raised when an HF model load violates :class:`HFModelPolicy`."""


# Allowlist of HuggingFace namespaces (the part before ``/``) that
# Mercury is willing to load weights from. These are first-party
# publishers (model vendors, well-known research orgs). Operators who
# need additional namespaces should extend this list explicitly so the
# review history records the decision.
TRUSTED_NAMESPACES: Final[frozenset[str]] = frozenset(
    {
        # Llama family
        "meta-llama",
        "NousResearch",
        # Mistral
        "mistralai",
        # Microsoft (Phi, BERT, BLIP, etc.)
        "microsoft",
        # Salesforce (BLIP, CodeGen)
        "Salesforce",
        # Qwen / Alibaba
        "Qwen",
        # MiniCPM / OpenBMB
        "openbmb",
        # LLaVA
        "llava-hf",
        "liuhaotian",
        # InternVL / OpenGVLab
        "OpenGVLab",
        # Google (T5, Gemma, ViT, etc.)
        "google",
        # OpenAI (CLIP weights, Whisper)
        "openai",
        # BigScience (BLOOM)
        "bigscience",
        # EleutherAI
        "EleutherAI",
        # HuggingFace canonical orgs
        "HuggingFaceH4",
        "HuggingFaceM4",
        "huggingface",
        # Sentence-Transformers
        "sentence-transformers",
        # Anthropic-trained open weights
        "Anthropic",
        # Stability AI
        "stabilityai",
        # TII (Falcon)
        "tiiuae",
        # 01-ai (Yi)
        "01-ai",
        # DeepSeek
        "deepseek-ai",
    }
)

# Namespaces whose published code Mercury will execute via
# ``trust_remote_code=True``. This is a tighter subset of the trusted
# namespaces — most legitimate research code is published by the same
# orgs that ship the weights.
REMOTE_CODE_ALLOWED_NAMESPACES: Final[frozenset[str]] = frozenset(
    {
        "openbmb",  # MiniCPM-V relies on custom forward
        "Qwen",
        "OpenGVLab",
        "01-ai",
        "deepseek-ai",
    }
)

# Permissive HF model-id regex: namespace/name, where each segment is
# made of [A-Za-z0-9._-]. Matches the constraint the Hub enforces.
_MODEL_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
)

# Revisions: 7+ char hex commit SHA preferred, but tag-like values are
# permitted because not every release publishes a frozen SHA.
_REVISION_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

_ENV_REQUIRE_REVISION: Final[str] = "MERCURY_HF_REQUIRE_REVISION"
_ENV_EXTRA_NAMESPACES: Final[str] = "MERCURY_HF_EXTRA_NAMESPACES"


def _is_local_path(model_id: str) -> bool:
    """Return True if model_id refers to a local filesystem path.

    HuggingFace model ids use ``/`` as a namespace separator, so a bare
    ``/`` cannot disambiguate a local path from a remote id. We therefore
    only treat the value as a local path when it starts with an
    unambiguous filesystem prefix (``/``, ``./``, ``../``, ``~``, a
    Windows drive letter) or it actually exists on disk.
    """
    if not model_id:
        return False
    if model_id.startswith(("/", "./", "../", "~")):
        return True
    if len(model_id) >= 2 and model_id[1] == ":":  # Windows drive letter
        return True
    if os.sep != "/" and os.sep in model_id:
        # Native separator on non-POSIX (e.g. backslash on Windows).
        return True
    if os.path.exists(model_id):
        return True
    return False


def _extra_namespaces() -> frozenset[str]:
    """Read operator-extended namespaces from the environment."""
    raw = os.environ.get(_ENV_EXTRA_NAMESPACES, "").strip()
    if not raw:
        return frozenset()
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def _require_revision() -> bool:
    """Return whether revision pinning is required in this environment."""
    raw = os.environ.get(_ENV_REQUIRE_REVISION)
    if raw is None:
        # Default: pinning required unless running under pytest.
        return "PYTEST_CURRENT_TEST" not in os.environ
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class HFModelPolicy:
    """Gate ``HuggingFace.from_pretrained`` calls (Bandit B615).

    Usage:

        from omni_mercury_engine.security.model_policy import HFModelPolicy

        HFModelPolicy.validate(model_id, revision=rev, trust_remote_code=False)
        AutoModel.from_pretrained(model_id, revision=rev)  # nosec B615
    """

    @staticmethod
    def is_local(model_id: str) -> bool:
        """Public helper — does this identifier refer to a local path?"""
        return _is_local_path(model_id)

    @classmethod
    def validate(
        cls,
        model_id: str,
        *,
        revision: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        """Authorise an HF load or raise :class:`UnsafeModelError`.

        Args:
            model_id: HuggingFace repo id (``"org/name"``) or local path.
            revision: Pinned commit SHA / tag / branch.
            trust_remote_code: Whether the caller intends to set
                ``trust_remote_code=True`` on ``from_pretrained``.

        Raises:
            UnsafeModelError: If the load is not allowed by policy.
        """
        if not isinstance(model_id, str) or not model_id:
            raise UnsafeModelError("HF model id must be a non-empty string")

        if _is_local_path(model_id):
            # Local files are operator-managed; integrity is the
            # filesystem's responsibility, not Hub policy.
            return

        if not _MODEL_ID_RE.match(model_id):
            raise UnsafeModelError(
                f"HF model id {model_id!r} is malformed; expected '<namespace>/<name>'"
            )

        namespace = model_id.split("/", 1)[0]
        allowlist = TRUSTED_NAMESPACES | _extra_namespaces()
        if namespace not in allowlist:
            raise UnsafeModelError(
                f"HF namespace {namespace!r} not in trusted allowlist. "
                f"Extend via {_ENV_EXTRA_NAMESPACES} or edit "
                "omni_mercury_engine.security.model_policy.TRUSTED_NAMESPACES."
            )

        if revision is not None:
            if not isinstance(revision, str) or not _REVISION_RE.match(revision):
                raise UnsafeModelError(
                    f"HF revision {revision!r} is malformed; expected SHA or tag"
                )
        elif _require_revision():
            raise UnsafeModelError(
                f"HF model {model_id!r} requested without a pinned revision; "
                f"set revision to a commit SHA, or set {_ENV_REQUIRE_REVISION}=false "
                "for development."
            )

        if trust_remote_code and namespace not in REMOTE_CODE_ALLOWED_NAMESPACES:
            raise UnsafeModelError(
                f"trust_remote_code=True is not authorised for namespace "
                f"{namespace!r}. Edit REMOTE_CODE_ALLOWED_NAMESPACES to opt in."
            )

    @classmethod
    def validate_vetted_dataset(
        cls,
        dataset_id: str,
        *,
        allowlist: tuple[str, ...],
        revision: str | None = None,
    ) -> None:
        """Authorise an HF *datasets* load against a code-constant allowlist.

        Dataset mirrors are often hosted under individual user namespaces
        that we cannot blanket-trust the way we do model publishers.
        Instead we require:

        1. The caller passes a tuple of dataset IDs that are class
           constants (and therefore went through code review).
        2. ``dataset_id`` is checked to be a member of that tuple at
           runtime — defence-in-depth against a future code change that
           lets caller input flow into the load.
        3. The revision (if any) matches the same hardened regex used
           for model revisions.

        Raises :class:`UnsafeModelError` on failure.
        """
        if not isinstance(dataset_id, str) or not dataset_id:
            raise UnsafeModelError("HF dataset id must be a non-empty string")

        if dataset_id not in allowlist:
            raise UnsafeModelError(
                f"HF dataset id {dataset_id!r} is not in the caller's "
                "vetted allowlist; refusing to load."
            )

        if not _MODEL_ID_RE.match(dataset_id):
            raise UnsafeModelError(
                f"HF dataset id {dataset_id!r} is malformed; expected "
                "'<namespace>/<name>'."
            )

        if revision is not None and not _REVISION_RE.match(revision):
            raise UnsafeModelError(
                f"HF dataset revision {revision!r} is malformed; expected SHA or tag."
            )
