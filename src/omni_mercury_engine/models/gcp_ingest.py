# Copyright (C) 2025 Steel Security Advisors LLC
"""Global Consciousness Project (GCP) ingestion + pre-registered statistics (WS-D).

The GCP archive is ~20 years of synchronised hardware-RNG trials (~65 nodes,
each summing 200 bits/second; expected mean 100, variance 50 under the null).
This module provides:

* :func:`fetch_egg_stream` -- ingestion targeting the documented raw-stream
  endpoint (``noosphere.princeton.edu`` ``eggdatareq``). It returns a structured
  result with provenance and an explicit ``reachable`` flag, so an unreachable
  archive is reported honestly rather than silently faked.
* the **pre-registered** statistics (:func:`egg_sums_to_z`,
  :func:`network_variance`, :func:`stouffer_z`) fixed in
  ``docs/PARAPSYCH_PREREGISTRATION.md`` *before* any analysis -- this dataset's
  documented failure mode is post-hoc analytic flexibility.
* :func:`synthetic_null_streams` -- a clearly-labelled true-random generator for
  validating the statistics/encoder plumbing under the null.

This is a pure signal-processing / anomaly task. **No claim is made that
"psi" is real.** A faithful null is the expected and scientifically valid
outcome; the data decides.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from omni_mercury_engine.datasets.base import http_get_with_retry

# Documented GCP raw-stream endpoint (per-second egg sums). The archive host is
# frequently offline; callers must handle ``reachable=False``.
EGGDATAREQ_URL = (
    "https://noosphere.princeton.edu/cgi-bin/eggdatareq.pl"
    "?z=1&year={year}&month={month}&day={day}&stime={stime}&etime={etime}"
)
BITS_PER_TRIAL = 200  # each egg sums 200 fair bits/second
NULL_MEAN = BITS_PER_TRIAL / 2.0  # 100
NULL_STD = math.sqrt(BITS_PER_TRIAL / 4.0)  # sqrt(50)


@dataclass
class IngestResult:
    """Outcome of a GCP ingestion attempt (honest about reachability)."""

    reachable: bool
    reason: str
    egg_sums: np.ndarray[Any, Any] | None = None  # (n_seconds, n_eggs)
    provenance: dict[str, Any] = field(default_factory=dict)


def fetch_egg_stream(
    year: int, month: int, day: int, stime: str = "00:00:00", etime: str = "00:05:00"
) -> IngestResult:
    """Attempt to fetch a real GCP egg-sum stream; report reachability honestly.

    Returns ``reachable=False`` with a reason (not an exception) when the
    archive cannot be reached or the host is not on the trusted allowlist, so
    downstream code can fall back to a clearly-labelled synthetic null without
    ever presenting fabricated data as real.
    """
    url = EGGDATAREQ_URL.format(
        year=year, month=f"{month:02d}", day=f"{day:02d}", stime=stime, etime=etime
    )
    provenance: dict[str, Any] = {
        "url": url,
        "source": "Global Consciousness Project (Princeton/ICRL)",
        "format": "per-second per-egg 200-bit sums",
    }
    try:
        raw = http_get_with_retry(url, timeout=30, retries=2)
    except Exception as e:  # unreachable, untrusted host, timeout, etc.
        return IngestResult(
            reachable=False,
            reason=f"{type(e).__name__}: {str(e)[:160]}",
            provenance=provenance,
        )
    try:
        egg_sums = _parse_eggdatareq_csv(raw.decode(errors="replace"))
    except Exception as e:
        return IngestResult(
            reachable=False, reason=f"parse_error: {str(e)[:160]}", provenance=provenance
        )
    provenance["n_seconds"], provenance["n_eggs"] = egg_sums.shape
    return IngestResult(reachable=True, reason="ok", egg_sums=egg_sums, provenance=provenance)


def _parse_eggdatareq_csv(text: str) -> np.ndarray[Any, Any]:
    """Parse the eggdatareq CSV (rows=seconds, cols=eggs; first cols are meta)."""
    rows: list[list[float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0].isalpha() or line.startswith("#"):
            continue
        parts = line.split(",")
        vals = []
        for p in parts[2:]:  # skip leading timestamp/index columns
            p = p.strip()
            if p in ("", "*", "NaN"):
                vals.append(np.nan)
            else:
                vals.append(float(p))
        if vals:
            rows.append(vals)
    if not rows:
        raise ValueError("no numeric rows parsed")
    width = max(len(r) for r in rows)
    arr = np.full((len(rows), width), np.nan)
    for i, r in enumerate(rows):
        arr[i, : len(r)] = r
    return arr


# ---------------------------------------------------------------------------
# Pre-registered statistics (fixed BEFORE analysis; see pre-registration doc)
# ---------------------------------------------------------------------------


def egg_sums_to_z(egg_sums: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Standardise per-second egg sums to z-scores under the fair-coin null."""
    return (egg_sums - NULL_MEAN) / NULL_STD


def network_variance(z: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """GCP network variance: per-second sum of squared egg z-scores.

    Under the null this is chi-square with df = (#eggs that second), so the
    expected value each second equals the egg count. Returns one value/second.
    """
    return np.nansum(z**2, axis=1)


def stouffer_z(z: np.ndarray[Any, Any]) -> float:
    """Stouffer's Z combining all egg-seconds: sum(z)/sqrt(N). ~N(0,1) under null."""
    flat = z[~np.isnan(z)]
    if flat.size == 0:
        return 0.0
    return float(flat.sum() / math.sqrt(flat.size))


def synthetic_null_streams(n_seconds: int, n_eggs: int, seed: int) -> np.ndarray[Any, Any]:
    """Clearly-SYNTHETIC true-random egg sums (Binomial(200, 0.5)). NOT real GCP.

    For validating the statistics/encoder plumbing under a known null; can never
    lift quarantine.
    """
    rng = np.random.RandomState(seed)
    return rng.binomial(BITS_PER_TRIAL, 0.5, size=(n_seconds, n_eggs)).astype(np.float64)


__all__ = [
    "BITS_PER_TRIAL",
    "NULL_MEAN",
    "NULL_STD",
    "IngestResult",
    "egg_sums_to_z",
    "fetch_egg_stream",
    "network_variance",
    "stouffer_z",
    "synthetic_null_streams",
]
