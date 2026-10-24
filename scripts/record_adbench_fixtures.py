#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Record ADBench tabular fixtures (tests/fixtures/adbench) with content hashes.

Downloads each named dataset from the ADBench repository (MIT; Han et al.,
NeurIPS 2022), checks it is a labelled NPZ (``X`` float matrix, ``y`` binary),
and writes ``<name>.npz`` (``X``, ``y``, compressed) next to a
``PROVENANCE.json`` holding the source URL and the SHA-256 of the served file
and of the recorded arrays, so a fixture cannot drift unnoticed.

Usage::

    python scripts/record_adbench_fixtures.py
"""

from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import requests

BASE = "https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical/"
FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "adbench"
#: name -> file in the ADBench catalogue (index_name.npz)
DATASETS = {
    "cardio": "6_cardio.npz",
    "breastw": "4_breastw.npz",
    "wbc": "42_WBC.npz",
    "thyroid": "38_thyroid.npz",
}


def _arrays_sha256(x: np.ndarray, y: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(x, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(y, dtype="<i8").tobytes())
    return digest.hexdigest()


def main() -> None:
    """Fetch, validate and write every dataset, then the provenance record."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    provenance_path = FIXTURES / "PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    for name, filename in DATASETS.items():
        url = BASE + filename
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        with np.load(io.BytesIO(response.content), allow_pickle=False) as archive:
            x = np.asarray(archive["X"], dtype=np.float64)
            y = np.asarray(archive["y"]).astype(np.int64).ravel()
        if x.ndim != 2 or y.shape != (x.shape[0],) or not set(np.unique(y)) <= {0, 1}:
            raise ValueError(f"{filename}: not a labelled binary NPZ ({x.shape}, {y.shape})")
        if not np.isfinite(x).all():
            raise ValueError(f"{filename}: non-finite features")
        np.savez_compressed(FIXTURES / f"{name}.npz", X=x, y=y)
        provenance[name] = {
            "url": url,
            "served_sha256": hashlib.sha256(response.content).hexdigest(),
            "arrays_sha256": _arrays_sha256(x, y),
            "n_samples": int(x.shape[0]),
            "n_features": int(x.shape[1]),
            "n_anomalies": int(y.sum()),
            "recorded_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        print(f"{name}: {x.shape}, {int(y.sum())} anomalies")
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
