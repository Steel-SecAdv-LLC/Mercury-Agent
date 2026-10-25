#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Record a log-template-key fixture (tests/fixtures/loghub) from LogHub's HDFS sample.

Downloads ``HDFS/HDFS_2k.log_structured.csv`` -- 2000 lines of a real Hadoop
HDFS log, already parsed into event templates by Drain -- and writes the
``EventId`` column as an integer key sequence to
``tests/fixtures/loghub/hdfs_templates.npz``:

* ``keys``: ``(2000,)`` int64, the template key of each line in file order;
* ``event_ids``: the ``EventId`` strings, so ``keys[i]`` is
  ``event_ids.tolist().index(EventId of line i)``.

That key sequence is the input ``DeepLogSequenceDetector`` takes.  The lines are
not anomaly-labelled -- LogHub's block-level labels ship with the full
11-million-line ``HDFS_1`` release, not with this sample -- so it supports a
normal-regime false-alarm measurement and an injected-unseen-template check,
not an AUROC.

``PROVENANCE.json`` records the URL, the SHA-256 of the served CSV and of the
stored arrays, the line count and the vocabulary size.  The data are published
by LogPAI; cite He, Zhu, He & Lyu (2020), "Loghub: A Large Collection of System
Log Datasets towards Automated Log Analytics", arXiv:2008.06448.

Usage::

    python scripts/record_hdfs_log_fixtures.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import requests

URL = "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/" "HDFS_2k.log_structured.csv"
FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "loghub"


def main() -> None:
    """Download the structured sample and write the key-sequence fixture."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    response = requests.get(URL, timeout=120)
    response.raise_for_status()
    served = response.content

    rows = list(csv.DictReader(served.decode("utf-8").splitlines()))
    if not rows or "EventId" not in rows[0]:
        raise SystemExit(f"{URL} has no EventId column; LogHub's layout changed")
    # Sort the vocabulary by the numeric part of E<n> so the key order is the
    # file's own template numbering, not dictionary order.
    event_ids = sorted({row["EventId"] for row in rows}, key=lambda e: int(e[1:]))
    key_of = {event: index for index, event in enumerate(event_ids)}
    keys = np.array([key_of[row["EventId"]] for row in rows], dtype=np.int64)

    path = FIXTURES / "hdfs_templates.npz"
    np.savez_compressed(path, keys=keys, event_ids=np.array(event_ids))

    provenance = {
        "description": (
            "Drain-parsed event-template keys of 2000 real Hadoop HDFS log lines "
            "(LogHub HDFS_2k sample), in file order. Not anomaly-labelled: "
            "LogHub's block-level labels ship with the full HDFS_1 release."
        ),
        "source": "https://github.com/logpai/loghub",
        "citation": (
            "He, Zhu, He & Lyu (2020), Loghub: A Large Collection of System Log "
            "Datasets towards Automated Log Analytics, arXiv:2008.06448"
        ),
        "license": "LogHub datasets are released for research use; see the LogHub repository.",
        "recorder": "scripts/record_hdfs_log_fixtures.py",
        "recorded_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": {
            "hdfs_templates": {
                "url": URL,
                "served_sha256": hashlib.sha256(served).hexdigest(),
                "keys_sha256": hashlib.sha256(keys.tobytes()).hexdigest(),
                "lines": int(keys.size),
                "vocabulary": len(event_ids),
                "event_ids": event_ids,
            }
        },
    }
    (FIXTURES / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {path} ({keys.size} keys, {len(event_ids)} templates)")


if __name__ == "__main__":
    main()
