#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Record NAB test fixtures (tests/fixtures/nab), and reference scores from NAB itself.

Downloads, at a pinned NAB commit, four real data files (one per category of
four), the labels file and NAB's own scorer (``nab/sweeper.py``), then writes:

* ``data/<category>/<file>.csv.gz``: each served file byte-verbatim,
  gzip-compressed (some NAB files have CRLF line ends, which the repository's
  whitespace hooks would otherwise rewrite);
* ``labels/combined_windows.json``: NAB's windows for those four files, the
  entries copied unchanged;
* ``reference_scores.json``: raw scores computed by NAB's ``Sweeper`` for
  seeded alarm patterns on each file under all three profiles, which
  ``tests/evaluation/test_nab_scoring.py`` holds ``evaluation.nab_scoring`` to;
* ``PROVENANCE.json``: URLs and SHA-256 of every downloaded file.

NAB is MIT-licensed (Copyright 2014-2024 Numenta Inc.). The recorder executes
NAB's ``sweeper.py`` (standard library only) to produce the reference values.

Usage::

    python scripts/record_nab_fixtures.py
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests

NAB_COMMIT = "ea702d75cc2258d9d7dd35ca8e5e2539d71f3140"
BASE = f"https://raw.githubusercontent.com/numenta/NAB/{NAB_COMMIT}/"
FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "nab"
FILES = (
    "realAWSCloudwatch/iio_us-east-1_i-a2eb1cd9_NetworkIn.csv",
    "realAdExchange/exchange-3_cpc_results.csv",
    "realKnownCause/rogue_agent_key_hold.csv",
    "realTraffic/speed_7578.csv",
)
#: Alarm patterns per file and profile: none, all, and seeded random densities.
DENSITIES = (0.002, 0.01, 0.05, 0.2)


def _get(path: str) -> bytes:
    url = BASE + path
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def _load_sweeper(source: bytes) -> Any:
    with tempfile.TemporaryDirectory() as tmp:
        file = Path(tmp) / "sweeper.py"
        file.write_bytes(source)
        spec = importlib.util.spec_from_file_location("nab_sweeper", file)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def main() -> None:
    """Download, write the fixtures, and record NAB's own scores."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from omni_mercury_engine.evaluation.nab_scoring import PROFILES, load_nab_corpus

    provenance: dict[str, Any] = {}
    sweeper_source = _get("nab/sweeper.py")
    labels_source = _get("labels/combined_windows.json")
    provenance["nab/sweeper.py"] = hashlib.sha256(sweeper_source).hexdigest()
    provenance["labels/combined_windows.json"] = hashlib.sha256(labels_source).hexdigest()
    labels = json.loads(labels_source)
    for name in FILES:
        raw = _get(f"data/{name}")
        provenance[f"data/{name}"] = hashlib.sha256(raw).hexdigest()
        out = FIXTURES / "data" / f"{name}.gz"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(gzip.compress(raw, mtime=0))
    (FIXTURES / "labels").mkdir(parents=True, exist_ok=True)
    subset = {name: labels[name] for name in FILES}
    (FIXTURES / "labels" / "combined_windows.json").write_text(
        json.dumps(subset, indent=2) + "\n", encoding="utf-8"
    )

    nab = _load_sweeper(sweeper_source)
    rng = np.random.default_rng(0)
    reference: dict[str, Any] = {}
    for stream in load_nab_corpus(FIXTURES, sorted({n.split("/")[0] for n in FILES})):
        timestamps = stream.timestamps.astype("datetime64[us]").tolist()
        limits = [(timestamps[a], timestamps[b]) for a, b in stream.windows]
        n = len(timestamps)
        patterns = [[], list(range(n))]
        for density in DENSITIES:
            patterns.append(np.flatnonzero(rng.random(n) < density).tolist())
        entry: dict[str, Any] = {"alarm_patterns": patterns, "raw_scores": {}}
        for profile, cost in PROFILES.items():
            sweeper = nab.Sweeper(
                probationPercent=0.15,
                costMatrix={"tpWeight": cost.tp, "fpWeight": cost.fp, "fnWeight": cost.fn},
            )
            scores = []
            for alarms in patterns:
                values = np.zeros(n)
                values[alarms] = 1.0
                _, row = sweeper.scoreDataSet(timestamps, values.tolist(), limits, stream.name, 1.0)
                scores.append(row.score)
            entry["raw_scores"][profile] = scores
        reference[stream.name] = entry
    (FIXTURES / "reference_scores.json").write_text(json.dumps(reference) + "\n", encoding="utf-8")
    (FIXTURES / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "description": (
                    "Four real NAB data files (byte-verbatim, gzip-compressed), their "
                    "windows from NAB's labels/combined_windows.json, and raw scores "
                    "computed by NAB's own Sweeper (nab/sweeper.py) for seeded alarm "
                    "patterns under the three NAB profiles."
                ),
                "source": f"https://github.com/numenta/NAB at commit {NAB_COMMIT}",
                "license": "MIT, Copyright 2014-2024 Numenta Inc.",
                "recorder": "scripts/record_nab_fixtures.py",
                "recorded_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "sha256_of_served_files": provenance,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {FIXTURES}")


if __name__ == "__main__":
    main()
