# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-event default-detector score cache.

Fitting ``MercuryAnomalyDetector`` (5-fold unsupervised adaptive weighting +
oracle) is the expensive step.  The detector's *scoring* is unsupervised — the
conformal operating point (Item 4) and reliability-weighted fusion (Item 3)
only change the threshold / the component-combination, not the component scores
themselves.  So we fit + ``detect()`` once per event, cache every score array
plus the fitted info-geometry manifold, and let the corrections operate on the
cached arrays.  This makes a full-suite before/after pass seconds, not minutes.

Cache key includes the row cap + seed so subsample variants don't collide.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector
from research.governed_fusion.label_provenance import event_is_external_label
from research.governed_fusion.suite import EventData, stratified_subsample

_CACHE_DIR = os.environ.get("GF_CACHE_DIR", "/home/user/gf_cache")


@dataclass(frozen=True)
class EventScores:
    """Cached default-detector outputs for one (subsampled) event."""

    domain: str
    event_id: str
    y: np.ndarray[Any, Any]
    combined: np.ndarray[Any, Any]
    verdict: np.ndarray[Any, Any]
    threshold: float
    resonance: np.ndarray[Any, Any]
    kinematic: np.ndarray[Any, Any]
    info_geo: np.ndarray[Any, Any]
    ig_mean: np.ndarray[Any, Any]
    ig_cov_inv: np.ndarray[Any, Any]

    def components(self) -> np.ndarray[Any, Any]:
        """``(n, 3)`` matrix [resonance, kinematic, info_geo]."""
        return np.column_stack([self.resonance, self.kinematic, self.info_geo])


def _scores_path(
    domain: str,
    event_id: str,
    cap: int,
    seed: int,
    *,
    cache_dir: str = _CACHE_DIR,
) -> str:
    # The protocol is part of the key: scores of the externally labelled
    # events from the fit-in-place protocol must not be read back as these.
    protocol = ":normal-reference" if event_is_external_label(domain, event_id) else ""
    key = hashlib.sha256(f"{domain}:{event_id}:{cap}:{seed}{protocol}".encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"scores_{domain}_{event_id}_{cap}_{key}.npz")


def scored_rows(
    ev: EventData, *, cap: int = 6000, seed: int = 42
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], bool]:
    """The rows the detector is fitted on and the rows it scores, for one event.

    Two protocols, by what the event's labels allow:

    * **Self-labelled event.** The detector is fitted on the capped event and
      scores the same rows: an unlabelled reference, declared as such.
    * **Externally labelled event.** Its labels are audited ground truth, so a
      reference known to be normal exists: every other normal row of the
      capped event. The detector is fitted on that and scores the remaining
      rows, which is the protocol of ``benchmarks/mercury_benchmark.py`` and
      the detector's own contract. Fitting on the whole event measured what
      the detector does with a reference that is half attacks (``nsl_kdd``:
      48%), not whether it finds the attacks.

    Returns:
        ``(reference, scored, labels_of_scored, in_place)``.
    """
    x_sub, y_sub = stratified_subsample(ev.X, ev.y, cap, seed=seed)
    y_sub = np.asarray(y_sub, dtype=int).reshape(-1)
    if not event_is_external_label(ev.domain, ev.event_id):
        return x_sub, x_sub, y_sub, True
    reference = np.flatnonzero(y_sub == 0)[::2]
    held_out = np.setdiff1d(np.arange(len(y_sub)), reference)
    return x_sub[reference], x_sub[held_out], y_sub[held_out], False


def event_scores(
    ev: EventData,
    *,
    cap: int = 6000,
    seed: int = 42,
    cache_dir: str = _CACHE_DIR,
) -> EventScores:
    """Fit the default detector once on the (subsampled) event; cache outputs."""
    path = _scores_path(ev.domain, ev.event_id, cap, seed, cache_dir=cache_dir)
    if os.path.exists(path):
        with np.load(path) as z:
            return EventScores(
                ev.domain,
                ev.event_id,
                z["y"],
                z["combined"],
                z["verdict"],
                float(z["threshold"]),
                z["resonance"],
                z["kinematic"],
                z["info_geo"],
                z["ig_mean"],
                z["ig_cov_inv"],
            )

    reference, X, y, in_place = scored_rows(ev, cap=cap, seed=seed)
    det = MercuryAnomalyDetector().fit(reference, unlabeled_reference=in_place)
    r = det.detect(X)
    es = EventScores(
        domain=ev.domain,
        event_id=ev.event_id,
        y=np.asarray(y, dtype=int).reshape(-1),
        combined=np.asarray(r["scores"], dtype=np.float64).reshape(-1),
        verdict=np.asarray(r["is_anomaly"], dtype=int).reshape(-1),
        threshold=float(r["threshold"]),
        resonance=np.asarray(r["resonance_scores"], dtype=np.float64).reshape(-1),
        kinematic=np.asarray(r["kinematic_scores"], dtype=np.float64).reshape(-1),
        info_geo=np.asarray(r["info_geometry_scores"], dtype=np.float64).reshape(-1),
        ig_mean=np.asarray(det._ig_mean, dtype=np.float64).reshape(-1),
        ig_cov_inv=np.asarray(det._ig_cov_inv, dtype=np.float64),
    )
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(
        path,
        y=es.y,
        combined=es.combined,
        verdict=es.verdict,
        threshold=np.asarray(es.threshold),
        resonance=es.resonance,
        kinematic=es.kinematic,
        info_geo=es.info_geo,
        ig_mean=es.ig_mean,
        ig_cov_inv=es.ig_cov_inv,
    )
    return es
