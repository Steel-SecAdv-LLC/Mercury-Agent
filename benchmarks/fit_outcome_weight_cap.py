#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fit ``counterintelligence.MAX_OUTCOME_WEIGHT`` on real adjudicated outcomes.

Default data: LIAR (Wang, 2017, "Liar, Liar Pants on Fire", ACL), 12,836
PolitiFact rulings, each attributed to the speaker who made the claim. It has
the structure the source-reliability posterior models: a source makes claims,
an independent adjudicator later rules them true or false. PolitiFact
statement ids increase over time, so ordering by id gives each speaker's
chronology. Rulings map to outcomes as: ``true``/``mostly-true`` confirmed;
``false``/``pants-fire``/``barely-true`` refuted; ``half-true`` dropped as
neither.

It is a proxy: intelligence sources are not politicians. An operator with
their own adjudicated outcomes refits on them with ``--histories`` (JSON:
source id to a list of booleans, oldest first).

Usage::

    python benchmarks/fit_outcome_weight_cap.py                 # LIAR, downloaded
    python benchmarks/fit_outcome_weight_cap.py --liar-zip PATH # LIAR, local copy
    python benchmarks/fit_outcome_weight_cap.py --histories outcomes.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import sys
import tempfile
import time
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from omni_mercury_engine.security.counterintelligence import (
    MAX_OUTCOME_WEIGHT,
    CounterintelligenceConfig,
    fit_outcome_weight_cap,
)

RESULTS_PATH = REPO_ROOT / "benchmarks" / "results" / "outcome_weight_fit.json"
LIAR_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
#: SHA-256 of the archive the committed result was measured on.
LIAR_SHA256 = "611c1addad919743dde15822b87a60bfb760d8f85597f25289e34621800654c7"
CONFIRMED = frozenset({"true", "mostly-true"})
REFUTED = frozenset({"false", "pants-fire", "barely-true"})


def _liar_bytes(local: Path | None) -> bytes:
    if local is not None:
        data = local.read_bytes()
    else:
        # Fixed https:// URL, verified by digest below.
        with urllib.request.urlopen(LIAR_URL, timeout=60) as response:
            data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != LIAR_SHA256:
        raise SystemExit(f"LIAR archive digest {digest} != pinned {LIAR_SHA256}")
    return data


def liar_histories(archive: bytes) -> dict[str, list[bool]]:
    """Speaker to chronological outcomes, from the LIAR archive bytes."""
    rows: list[tuple[int, str, str]] = []
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        for name in ("train.tsv", "valid.tsv", "test.tsv"):
            text = bundle.read(name).decode("utf-8")
            for record in csv.reader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE):
                if len(record) >= 5 and record[0].endswith(".json"):
                    rows.append((int(record[0].split(".")[0]), record[1], record[4]))
    histories: dict[str, list[bool]] = defaultdict(list)
    for _, label, speaker in sorted(rows):
        if label in CONFIRMED:
            histories[speaker].append(True)
        elif label in REFUTED:
            histories[speaker].append(False)
    return dict(histories)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--liar-zip", type=Path, default=None, help="local LIAR archive")
    source.add_argument("--histories", type=Path, default=None, help="your own outcomes (JSON)")
    parser.add_argument("--n-boot", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = parser.parse_args(argv)

    started = time.time()
    if args.histories is not None:
        histories = json.loads(args.histories.read_text(encoding="utf-8"))
        dataset: dict[str, Any] = {"name": "operator histories", "path": str(args.histories)}
    else:
        histories = liar_histories(_liar_bytes(args.liar_zip))
        dataset = {
            "name": "LIAR (Wang, 2017), PolitiFact rulings by speaker",
            "url": LIAR_URL,
            "sha256": LIAR_SHA256,
            "mapping": {
                "confirmed": sorted(CONFIRMED),
                "refuted": sorted(REFUTED),
                "dropped": ["half-true"],
            },
        }
    prior_strength = CounterintelligenceConfig().prior_strength
    fit = fit_outcome_weight_cap(
        histories, prior_strength=prior_strength, n_boot=args.n_boot, seed=args.seed
    )
    low = fit.break_even_ci[0] if fit.break_even_ci else fit.break_even_weight
    recommended = float(np.floor(low * 2.0) / 2.0)
    result = {
        "dataset": dataset,
        "fit": fit.as_dict(),
        "recommended_max_outcome_weight": recommended,
        "rule": "lower bound of the bootstrap interval of the break-even weight, rounded down to 0.5",
        "module_max_outcome_weight": MAX_OUTCOME_WEIGHT,
        "seconds": round(time.time() - started, 2),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.out.parent, delete=False, suffix=".tmp") as tmp:
        tmp.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    Path(tmp.name).replace(args.out)
    print(
        f"sources={fit.n_sources} outcomes={fit.n_outcomes} optimal_weight={fit.optimal_weight:.3g} "
        f"break_even={fit.break_even_weight:.3f} CI={fit.break_even_ci} -> cap {recommended:g} "
        f"(module: {MAX_OUTCOME_WEIGHT:g})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
