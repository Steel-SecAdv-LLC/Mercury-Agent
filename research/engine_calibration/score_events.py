# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Score every cached event once and write ``scores.pkl`` for ``measure_cached.py``.

``measure_cached.py`` reads ``scores.pkl`` from the calibration workspace, and
nothing in the repository wrote it: the published hazard table could not be
rebuilt from a checkout. This is the step that was missing.

The rule, which the committed ``results.json`` already recorded (NSL-KDD is
``n=20000, subsampled=true``) and ``research/hazard_calibration/measure.py``
states for the same events:

* one ``MercuryAnomalyDetector`` pass per event: fitted unlabelled on the
  event's own rows (``fit(X, unlabeled_reference=True)``) and run on them
  (``detect(X)["scores"]``);
* an event with more than 20,000 rows is read at 20,000 evenly spaced rows
  (``numpy.linspace(0, n - 1, 20000)`` as integers), the same rows every run;
* an event whose cached labels hold one class is not scored.

Nothing is random: the same cache gives the same ``scores.pkl``.

The whole table, from an empty workspace (network access to the sources; the
loaders that need a key say so and are skipped)::

    export MERCURY_CALIBRATION_CACHE=/some/empty/dir
    python research/engine_calibration/build_cache.py      # events -> cache/*.npz, manifest.json
    python research/engine_calibration/score_events.py     # cache/*.npz -> scores.pkl
    python research/engine_calibration/measure_cached.py   # scores.pkl -> results.json

``build_cache.py`` fetches through the shipped loaders. On a host that can
only reach the sources through ``HTTPS_PROXY``, see ``_proxy_fallback.py``:
the table of 2026-11-20 needed it for two hosts, and it lives in these
research scripts, not in the loaders.
"""

from __future__ import annotations

import pickle
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from _workspace import cache_archives, workspace_file

#: An event with more rows than this is read at this many evenly spaced rows.
ROW_LIMIT = 20_000


def evenly_spaced(n_rows: int, limit: int = ROW_LIMIT) -> np.ndarray[Any, Any]:
    """The row indices an event of ``n_rows`` is read at: all of them up to ``limit``."""
    if n_rows <= limit:
        return np.arange(n_rows)
    return np.linspace(0, n_rows - 1, limit).astype(int)


def score_archive(path: str) -> tuple[tuple[str, str], dict[str, Any] | None]:
    """Score one ``{domain}__{event}.npz`` archive.

    Returns:
        ``((domain, event), {"scores", "y", "subsampled", "rows"})``, or
        ``((domain, event), None)`` when the labels hold one class.
    """
    from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

    name = Path(path).name
    domain = name.split("__")[0]
    event = name[len(domain) + 2 : -len(".npz")]
    with np.load(path) as held:
        rows, labels = held["X"], held["y"].astype(int)
    if rows.ndim == 1:
        rows = rows.reshape(-1, 1)
    if labels.size == 0 or labels.min() == labels.max():
        return (domain, event), None
    keep = evenly_spaced(len(labels))
    subsampled = len(keep) < len(labels)
    total = len(labels)
    rows, labels = rows[keep], labels[keep]
    detector = MercuryAnomalyDetector()
    detector.fit(rows, unlabeled_reference=True)
    scores = np.asarray(detector.detect(rows)["scores"], dtype=float).reshape(-1)
    return (domain, event), {
        "scores": scores,
        "y": labels,
        "subsampled": subsampled,
        "rows": total,
    }


def main() -> int:
    """Score the workspace's archives and write ``scores.pkl``."""
    warnings.filterwarnings("ignore")
    scored: dict[tuple[str, str], dict[str, Any]] = {}
    for path in cache_archives():
        started = time.time()
        key, result = score_archive(path)
        if result is None:
            print(f"{key[0]}/{key[1]} not scored: one class only", flush=True)
            continue
        scored[key] = result
        print(
            f"{key[0]}/{key[1]} n={len(result['y'])} of {result['rows']} "
            f"pos={int(result['y'].sum())} ({time.time() - started:.1f}s)",
            flush=True,
        )
    out = workspace_file("scores.pkl")
    with open(out, "wb") as handle:
        pickle.dump(scored, handle)
    print(f"wrote {out}: {len(scored)} events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
