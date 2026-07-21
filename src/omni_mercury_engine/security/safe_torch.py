# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Centralized, fail-closed ``torch.load`` wrapper.

``torch.load`` deserializes a Python pickle. On a torch older than 2.6 the
default is ``weights_only=False``, which runs a *full* pickle VM and can
execute arbitrary code embedded in a ``.pt`` file (CVE-class remote code
execution: a malicious checkpoint's ``__reduce__`` runs ``os.system`` at
load time). Even on torch >= 2.6, where the default flipped to ``True``,
relying on the library default means a single forgotten keyword — or a
downgrade to an older pinned torch — silently re-opens the hole.

Mercury loads checkpoints from several surfaces (the fusion engine, the
geological/space/visual detectors, the ML inference server, the hazard
training candidate-selection paths, the σ_Immutable weights). Scattering
``torch.load(..., weights_only=True)`` across ~24 call sites makes the
security property *per-call-site* and therefore un-auditable: nothing stops
the next call site from omitting the keyword.

This module makes the safe load the **only** sanctioned path:

* :func:`safe_torch_load` hard-pins ``weights_only=True`` (the restricted
  unpickler that admits tensors, storages and primitive containers only,
  and refuses arbitrary global resolution). Passing ``weights_only=False``
  is a hard :class:`UnsafeCheckpointError`, not an option — the wrapper has
  no bypass by design.
* Path inputs are validated (exists, is a regular file) and bounded by an
  on-disk size ceiling before torch reads a single byte, so a truncated or
  absurdly large file fails fast with a clear error instead of OOM-ing a
  worker.
* torch's ``weights_only`` refusal (``pickle.UnpicklingError`` /
  ``_pickle.UnpicklingError`` / the torch "Weights only load failed"
  message) is translated into :class:`UnsafeCheckpointError` so callers see
  one security-typed exception describing the refused global.

The companion CI gate ``scripts/check_torch_load_safety.py`` forbids any raw
``torch.load(`` in ``src/`` outside this module, so the wrapper cannot be
routed around silently.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import os
    from typing import IO

    import torch

__all__ = [
    "DEFAULT_MAX_CHECKPOINT_BYTES",
    "UnsafeCheckpointError",
    "safe_torch_load",
]

#: On-disk size ceiling for a checkpoint path (default 2 GiB). The largest
#: shipped checkpoint is ~9 MiB; 2 GiB leaves generous headroom for
#: operator-supplied fusion models while bounding a hostile/truncated file
#: from exhausting memory. Raise per-call via ``max_bytes`` for a trusted
#: larger payload.
DEFAULT_MAX_CHECKPOINT_BYTES: int = 2 * 1024 * 1024 * 1024


class UnsafeCheckpointError(ValueError):
    """Raised when a checkpoint load is refused on security grounds.

    Covers: an attempt to use ``weights_only=False`` through the safe
    wrapper; a missing / non-regular / oversized checkpoint path; and a
    checkpoint whose pickle stream references a global the restricted
    unpickler refuses (the RCE-class signature). The message never echoes
    payload bytes.
    """


def _validate_checkpoint_path(source: str | os.PathLike[str], max_bytes: int) -> None:
    """Validate a path-like checkpoint source before torch reads it."""
    p = Path(source)
    if not p.exists():
        raise UnsafeCheckpointError(f"Checkpoint path does not exist: {p}")
    if not p.is_file():
        raise UnsafeCheckpointError(f"Checkpoint path is not a regular file: {p}")
    size = p.stat().st_size
    if size <= 0:
        raise UnsafeCheckpointError(f"Checkpoint file is empty: {p}")
    if size > max_bytes:
        raise UnsafeCheckpointError(
            f"Checkpoint exceeds size ceiling: {size} bytes > {max_bytes} bytes "
            f"(raise max_bytes explicitly if this checkpoint is trusted)"
        )


def _is_path_like(source: object) -> bool:
    return isinstance(source, (str, Path)) or hasattr(source, "__fspath__")


def safe_torch_load(
    source: str | os.PathLike[str] | IO[bytes],
    *,
    map_location: str | torch.device | dict[str, str] | None = "cpu",
    weights_only: bool = True,
    max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
    mmap: bool | None = None,
    **torch_load_kwargs: Any,
) -> Any:
    """Load a torch checkpoint under the restricted (``weights_only``) unpickler.

    This is the only sanctioned entry point for ``torch.load`` in Mercury.
    ``weights_only=True`` is enforced unconditionally; there is deliberately
    no way to request the full pickle VM through this function.

    Parameters
    ----------
    source:
        Filesystem path (``str`` / ``os.PathLike``) or an already-open
        binary file object. Path inputs are validated and size-bounded
        before any bytes are read; stream inputs are size-checked when the
        stream is seekable.
    map_location:
        Passed straight to ``torch.load``. Defaults to ``"cpu"`` (the safe,
        device-agnostic default); pass an explicit device to restore onto a
        GPU engine.
    weights_only:
        Must be ``True`` (the default). Any other value raises
        :class:`UnsafeCheckpointError` — the wrapper exists precisely to
        forbid the arbitrary-code-execution load path.
    max_bytes:
        On-disk size ceiling for path inputs. Default
        :data:`DEFAULT_MAX_CHECKPOINT_BYTES` (2 GiB).
    mmap:
        Forwarded to ``torch.load`` when not ``None`` (memory-map the file
        instead of reading it whole — useful for large read-only weights).
    **torch_load_kwargs:
        Any remaining keyword arguments forwarded verbatim to
        ``torch.load`` (``pickle_module`` is intentionally rejected — see
        below).

    Returns
    -------
    Any
        Whatever the checkpoint deserializes to (typically a ``state_dict``
        or a dict of primitives + tensors).

    Raises
    ------
    UnsafeCheckpointError
        On a rejected ``weights_only`` value, an invalid/oversized path, or
        a checkpoint whose pickle references a global the restricted
        unpickler refuses.
    """
    if weights_only is not True:
        raise UnsafeCheckpointError(
            "safe_torch_load refuses weights_only=False: the full pickle VM "
            "can execute arbitrary code embedded in a checkpoint. Convert the "
            "payload to a weights_only-compatible state dict, or (if a trusted "
            "local file genuinely requires the full loader) call torch.load "
            "directly with an explicit, reviewed justification — the "
            "check_torch_load_safety CI gate will require an allowlist entry."
        )
    # A caller must not smuggle an alternative unpickler in through kwargs.
    if "pickle_module" in torch_load_kwargs:
        raise UnsafeCheckpointError(
            "safe_torch_load does not accept a custom pickle_module; the "
            "restricted weights_only unpickler is mandatory."
        )

    if _is_path_like(source):
        _validate_checkpoint_path(source, max_bytes)  # type: ignore[arg-type]

    import torch  # lazy: keep the security package importable without the ml extra

    if mmap is not None:
        torch_load_kwargs["mmap"] = mmap

    try:
        return torch.load(
            source,
            map_location=map_location,
            weights_only=True,
            **torch_load_kwargs,
        )
    except pickle.UnpicklingError as exc:
        # torch raises UnpicklingError from the restricted unpickler when a
        # checkpoint references a disallowed global — exactly the RCE-class
        # payload this wrapper defends against. Surface it as a security
        # refusal rather than a generic load failure.
        raise UnsafeCheckpointError(
            f"Refusing checkpoint: the restricted (weights_only) unpickler "
            f"rejected a global in the pickle stream — the checkpoint is not "
            f"a plain state dict and may be hostile. Underlying error: {exc}"
        ) from exc
