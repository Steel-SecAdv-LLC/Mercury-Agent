#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Record scalp-EEG test fixtures (tests/fixtures/eeg) from the CHB-MIT database on PhysioNet.

Downloads two recordings of patient chb01 that each contain one annotated
seizure, reads channel ``FP1-F7`` (256 Hz) from the EDF file with a small
reader (the EDF header and 16-bit records are plain binary), and writes for
each recording ``tests/fixtures/eeg/<record>.npz`` with float32 ``background``
(microvolts, the 300 s ending 120 s before seizure onset) and ``seizure`` (the
annotated seizure, onset to end).  Seizure times are the ones in the
database's ``chb01-summary.txt``: ``chb01_03`` 2996-3036 s, ``chb01_04``
1467-1494 s.  ``PROVENANCE.json`` records the URLs and the SHA-256 of every
served file and stored array.

CHB-MIT Scalp EEG Database 1.0.0 (Guttag 2010, doi:10.13026/C2K01R) is distributed
on PhysioNet (Goldberger et al. 2000) under the Open Data Commons Attribution
License v1.0.

Usage::

    python scripts/record_eeg_fixtures.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import requests

BASE = "https://physionet.org/files/chbmit/1.0.0/chb01/"
FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "eeg"
#: record -> (seizure start s, seizure end s), from chb01-summary.txt
SEIZURES = {"chb01_03": (2996, 3036), "chb01_04": (1467, 1494)}
CHANNEL = "FP1-F7"
RATE = 256
BACKGROUND_SECONDS = 300
GAP_SECONDS = 120


def read_channel(raw: bytes, label: str) -> np.ndarray:
    """One channel of an EDF file, in the file's physical unit (microvolts here)."""
    n_signals = int(raw[252:256])
    n_records = int(raw[236:244])
    cursor = 256
    labels = [
        raw[cursor + 16 * i : cursor + 16 * (i + 1)].decode().strip() for i in range(n_signals)
    ]
    cursor += 16 * n_signals + 80 * n_signals + 8 * n_signals  # label, transducer, dimension

    def fields(width: int) -> list[float]:
        nonlocal cursor
        values = [
            float(raw[cursor + width * i : cursor + width * (i + 1)]) for i in range(n_signals)
        ]
        cursor += width * n_signals
        return values

    phys_min, phys_max, dig_min, dig_max = fields(8), fields(8), fields(8), fields(8)
    cursor += 80 * n_signals  # prefiltering
    per_record = [int(v) for v in fields(8)]
    cursor += 32 * n_signals  # reserved
    index = labels.index(label)
    stride = sum(per_record)
    samples = np.frombuffer(raw, dtype="<i2", offset=cursor, count=n_records * stride)
    records = samples.reshape(n_records, stride)
    start = sum(per_record[:index])
    digital = records[:, start : start + per_record[index]].reshape(-1).astype(np.float64)
    gain = (phys_max[index] - phys_min[index]) / (dig_max[index] - dig_min[index])
    return (digital - dig_min[index]) * gain + phys_min[index]


def main() -> None:
    """Download, cut the segments, and write the fixtures."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, dict[str, object]] = {}
    for record, (onset, end) in SEIZURES.items():
        url = f"{BASE}{record}.edf"
        response = requests.get(url, timeout=900)
        response.raise_for_status()
        signal = read_channel(response.content, CHANNEL)
        seizure = signal[onset * RATE : end * RATE].astype(np.float32)
        stop = (onset - GAP_SECONDS) * RATE
        background = signal[stop - BACKGROUND_SECONDS * RATE : stop].astype(np.float32)
        np.savez_compressed(FIXTURES / f"{record}.npz", background=background, seizure=seizure)
        provenance[record] = {
            "url": url,
            "channel": CHANNEL,
            "sampling_rate_hz": RATE,
            "seizure_seconds": [onset, end],
            "served_sha256": hashlib.sha256(response.content).hexdigest(),
            "background_sha256": hashlib.sha256(background.tobytes()).hexdigest(),
            "seizure_sha256": hashlib.sha256(seizure.tobytes()).hexdigest(),
        }
    (FIXTURES / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "description": (
                    "Single-channel (FP1-F7, 256 Hz, microvolts) scalp EEG from two "
                    "recordings of CHB-MIT patient chb01: 300 s of background ending "
                    "120 s before each annotated seizure, and the seizure itself."
                ),
                "source": "https://physionet.org/content/chbmit/1.0.0/",
                "license": "Open Data Commons Attribution License v1.0",
                "citation": (
                    "Guttag, J. (2010), CHB-MIT Scalp EEG Database, version 1.0.0, PhysioNet, "
                    "doi:10.13026/C2K01R; Goldberger et al. (2000), PhysioBank, PhysioToolkit, "
                    "and PhysioNet, Circulation 101(23), doi:10.1161/01.CIR.101.23.e215"
                ),
                "recorder": "scripts/record_eeg_fixtures.py",
                "recorded_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "files": provenance,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {FIXTURES}")


if __name__ == "__main__":
    main()
