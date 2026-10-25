#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Record bearing-vibration test fixtures (tests/fixtures/cwru) from the CWRU Bearing Data Center.

Downloads five 12 kHz drive-end accelerometer recordings and writes the leading
segment of each as ``tests/fixtures/cwru/<name>.npz`` (float32 ``x``, in g):

* ``normal_0hp`` (file 97) and ``normal_1hp`` (file 98): healthy bearing at 0 and 1 hp;
* ``inner_race`` (105), ``ball`` (118), ``outer_race`` (130): 0.007-inch faults at 0 hp.

``PROVENANCE.json`` records each URL, the SHA-256 of the served ``.mat`` file and
of the stored array, and the number of samples kept.  The data are published by
the Case Western Reserve University Bearing Data Center; cite Smith & Randall
(2015), Mech. Syst. Signal Process. 64-65, 100-131, doi:10.1016/j.ymssp.2015.04.021.

Usage::

    python scripts/record_cwru_fixtures.py
"""

from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import requests
from scipy.io import loadmat

BASE = "https://engineering.case.edu/sites/default/files/"
FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "cwru"
#: name -> (file number, windows of 2048 samples kept)
FILES = {
    "normal_0hp": (97, 60),
    "normal_1hp": (98, 60),
    "inner_race": (105, 20),
    "ball": (118, 20),
    "outer_race": (130, 20),
}
WINDOW = 2048


def main() -> None:
    """Download, verify the drive-end channel, and write the fixtures."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, dict[str, object]] = {}
    for name, (number, windows) in FILES.items():
        url = f"{BASE}{number}.mat"
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        served = response.content
        record = loadmat(io.BytesIO(served))
        key = f"X{number:03d}_DE_time"
        signal = np.asarray(record[key], dtype=np.float64).reshape(-1)
        kept = np.asarray(signal[: windows * WINDOW], dtype=np.float32)
        if kept.size != windows * WINDOW:
            raise RuntimeError(f"{url}: {signal.size} samples, need {windows * WINDOW}")
        np.savez_compressed(FIXTURES / f"{name}.npz", x=kept)
        provenance[name] = {
            "url": url,
            "channel": key,
            "served_sha256": hashlib.sha256(served).hexdigest(),
            "array_sha256": hashlib.sha256(kept.tobytes()).hexdigest(),
            "samples_in_file": int(signal.size),
            "samples_kept": int(kept.size),
        }
    (FIXTURES / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "description": (
                    "Leading segments of five 12 kHz drive-end accelerometer recordings "
                    "(g) from the Case Western Reserve University Bearing Data Center: "
                    "healthy at 0 and 1 hp, and 0.007-inch inner-race, ball and "
                    "outer-race faults at 0 hp."
                ),
                "source": "https://engineering.case.edu/bearingdatacenter",
                "citation": (
                    "Smith & Randall (2015), Rolling element bearing diagnostics using the "
                    "Case Western Reserve University data: a benchmark study, Mechanical "
                    "Systems and Signal Processing 64-65, 100-131, "
                    "doi:10.1016/j.ymssp.2015.04.021"
                ),
                "recorder": "scripts/record_cwru_fixtures.py",
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
