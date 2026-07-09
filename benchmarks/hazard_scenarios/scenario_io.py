# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared IO + integrity helpers for the hazard-guard scenario sets.

Hashing policy
--------------
JSON scenario files are written canonically (sorted keys, fixed separators,
trailing newline), so their **file bytes** hash reproducibly and
``sha256_file`` is used directly.

NPZ files are zip containers whose entry metadata embeds timestamps, so their
file bytes are *not* reproducible across regenerations even when every array
is bit-identical. ``sha256_npz_content`` therefore hashes the **canonical
array content** -- for each key in sorted order: the key, the dtype
descriptor, the shape, and the raw C-order array bytes. Any change to any
array (or to the set of keys) changes the hash; a mere re-zip does not.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCENARIO_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = SCENARIO_DIR / "manifest.json"


def write_canonical_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as canonical JSON (sorted keys, stable separators).

    Args:
        path: Destination file path.
        payload: JSON-serialisable mapping.
    """
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, separators=(",", ": ")) + "\n")


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's raw bytes.

    Args:
        path: File to hash.

    Returns:
        Hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_npz_content(path: Path) -> str:
    """SHA-256 over the canonical array content of an NPZ file.

    Hashes, for each key in sorted order: the key name, dtype descriptor,
    shape, and raw C-contiguous array bytes. Container (zip) metadata such as
    entry timestamps is deliberately excluded -- see the module docstring.

    Args:
        path: NPZ file to hash.

    Returns:
        Hex digest of the canonical content.

    Raises:
        ValueError: If the NPZ contains an object-dtype array (would require
            pickle, which is neither canonical nor safe).
    """
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as bundle:
        for key in sorted(bundle.files):
            arr = np.ascontiguousarray(bundle[key])
            if arr.dtype == object:  # pragma: no cover - np.load already rejects
                raise ValueError(f"{path.name}:{key} has object dtype")
            digest.update(key.encode())
            digest.update(str(arr.dtype.str).encode())
            digest.update(str(arr.shape).encode())
            digest.update(arr.tobytes())
    return digest.hexdigest()


def hash_scenario_file(path: Path) -> str:
    """Hash a scenario file with the policy appropriate to its format.

    Args:
        path: ``.json`` (file-byte hash) or ``.npz`` (content hash) file.

    Returns:
        Hex digest.

    Raises:
        ValueError: For unsupported file extensions.
    """
    if path.suffix == ".json":
        return sha256_file(path)
    if path.suffix == ".npz":
        return sha256_npz_content(path)
    raise ValueError(f"unsupported scenario file type: {path.name}")


def load_manifest() -> dict[str, Any]:
    """Load the committed scenario manifest.

    Returns:
        Parsed manifest mapping.

    Raises:
        FileNotFoundError: If the manifest has not been generated/committed.
    """
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"scenario manifest missing: {MANIFEST_PATH} "
            "(run benchmarks/hazard_scenarios/generate_scenarios.py)"
        )
    manifest: dict[str, Any] = json.loads(MANIFEST_PATH.read_text())
    return manifest


def verify_file_against_manifest(name: str) -> Path:
    """Verify a scenario file's hash against the committed manifest.

    Args:
        name: File name relative to the scenario directory.

    Returns:
        The verified file path.

    Raises:
        FileNotFoundError: If the file or its manifest entry is missing.
        ValueError: If the hash does not match the manifest (tampered or
            regenerated without updating the manifest + baseline).
    """
    manifest = load_manifest()
    entry = manifest.get("files", {}).get(name)
    if entry is None:
        raise FileNotFoundError(f"{name} has no manifest entry")
    path = SCENARIO_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"scenario file missing: {path}")
    actual = hash_scenario_file(path)
    if actual != entry["sha256"]:
        raise ValueError(
            f"{name}: sha256 mismatch (manifest {entry['sha256'][:12]}..., "
            f"actual {actual[:12]}...); scenario sets are pinned -- regenerate "
            "via generate_scenarios.py and re-pin the guard baseline if this "
            "change is intentional"
        )
    return path
