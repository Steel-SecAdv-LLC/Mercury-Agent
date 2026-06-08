"""Reachable real-domain measurement suite for the governed fusion substrate.

This is the single shared measurement driver used by every governed-fusion
correction script (conformal operating point, adversarial survivability,
reliability-weighted fusion).  It builds ``(X, y)`` per event from the **real**
live-API loaders (USGS / NOAA / FEMA / CISA / WHO), fits the **real**
``MercuryAnomalyDetector``, and scores with ``ml.mercury_ml`` (no scikit-learn).

Reachable suite (9 domains / 30 events).  ``pandemic/ebola_2014`` falls back to
synthetic despite ``MERCURY_ALLOW_SYNTHETIC=0`` and is excluded -> 29 real
events.  Unreachable domains (wildfire, flood, volcanic, landslide, financial,
sepsis) are *not* synthesised and never enter the suite.

Determinism: the loaders fetch historical events whose payloads are stable;
``MercuryAnomalyDetector`` is deterministic after ``fit()``.  Built ``(X, y)``
arrays are cached under ``$GF_CACHE_DIR`` (default ``/home/user/gf_cache``) so
iteration does not re-hit the network; deleting the cache regenerates from the
live loaders.
"""

from __future__ import annotations

import hashlib
import importlib
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

# Reachable domains: domain -> (loader module, loader class, max events to pull).
REACHABLE: dict[str, tuple[str, str, int]] = {
    "earthquake": ("earthquake_loader", "EarthquakeLoader", 5),
    "tsunami": ("tsunami_loader", "TsunamiLoader", 3),
    "tornado": ("tornado_loader", "TornadoLoader", 2),
    "marine": ("marine_loader", "MarineLoader", 2),
    "hurricane": ("hurricane_loader", "HurricaneLoader", 4),
    "energy": ("energy_loader", "EnergyLoader", 3),
    "fema": ("fema_loader", "FEMALoader", 3),
    "network_security": ("network_security_loader", "NetworkSecurityLoader", 2),
    "pandemic": ("pandemic_loader", "PandemicLoader", 6),
}

# Known synthetic-fallback events excluded from official numbers (stated, never
# averaged in).  ``ebola_2014`` fell back to synthetic despite the flag.
EXCLUDED_EVENTS: frozenset[str] = frozenset({"ebola_2014"})

_CACHE_DIR = os.environ.get("GF_CACHE_DIR", "/home/user/gf_cache")


@dataclass(frozen=True)
class EventData:
    """A single real event: engineered features and binary ground truth."""

    domain: str
    event_id: str
    X: np.ndarray[Any, Any]
    y: np.ndarray[Any, Any]

    @property
    def n_pos(self) -> int:
        return int(np.sum(self.y == 1))

    @property
    def usable(self) -> bool:
        """At least one of each class and >= 8 rows (enough to fit + split)."""
        return self.X.shape[0] >= 8 and 0 < self.n_pos < self.X.shape[0]


def stratified_subsample(
    X: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
    cap: int,
    *,
    seed: int = 42,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Seeded, class-stratified row subsample (no-op when ``len(X) <= cap``).

    Used to keep very large events (nsl_kdd 148k, mpox 63k, fema 10k)
    tractable for iteration.  Stratification preserves each class's prevalence
    so AUROC/AUPRC stay representative; the seed makes it reproducible.
    """
    n = len(X)
    if cap <= 0 or n <= cap:
        return X, y
    rng = np.random.RandomState(seed)
    keep: list[np.ndarray[Any, Any]] = []
    classes = np.unique(y)
    for c in classes:
        ci = np.where(y == c)[0]
        k = max(1, round(cap * len(ci) / n))
        keep.append(rng.choice(ci, size=min(k, len(ci)), replace=False))
    idx = np.concatenate(keep)
    rng.shuffle(idx)
    return X[idx], y[idx]


def _cache_path(domain: str, event_id: str) -> str:
    key = hashlib.sha256(f"{domain}:{event_id}".encode()).hexdigest()[:16]
    return os.path.join(_CACHE_DIR, f"{domain}_{event_id}_{key}.npz")


def _load_event(domain: str, mod: str, cls: str, event_id: str) -> EventData | None:
    """Build one event's (X, y), using the npz cache when present."""
    path = _cache_path(domain, event_id)
    if os.path.exists(path):
        with np.load(path) as z:
            return EventData(domain, event_id, z["X"], z["y"])

    loader_mod = importlib.import_module(f"omni_mercury_engine.loaders.{mod}")
    loader = getattr(loader_mod, cls)()
    raw = loader.fetch_historical(event_id)
    feats = np.asarray(loader.engineer_features(raw), dtype=np.float64)
    y = np.asarray(loader.get_ground_truth(event_id)).astype(int).reshape(-1)
    if feats.ndim == 1:
        feats = feats.reshape(-1, 1)
    m = min(len(feats), len(y))
    feats, y = feats[:m], y[:m]
    # Drop non-finite rows so every consumer sees clean real data.
    finite = np.isfinite(feats).all(axis=1)
    feats, y = feats[finite], y[finite]
    os.makedirs(_CACHE_DIR, exist_ok=True)
    np.savez(path, X=feats, y=y)
    return EventData(domain, event_id, feats, y)


def build_suite(*, verbose: bool = False) -> list[EventData]:
    """Build the reachable suite of usable real events (excludes synthetic)."""
    events: list[EventData] = []
    for domain, (mod, cls, max_ev) in REACHABLE.items():
        try:
            loader_mod = importlib.import_module(f"omni_mercury_engine.loaders.{mod}")
            loader = getattr(loader_mod, cls)()
            listing = loader.list_events()
        except Exception as exc:  # pragma: no cover - loader/import guard
            if verbose:
                print(f"{domain}: LOADER_ERR {str(exc)[:90]}", flush=True)
            continue
        taken = 0
        for ev in listing:
            if taken >= max_ev:
                break
            event_id = ev.get("event_id") if isinstance(ev, dict) else str(ev)
            if event_id in EXCLUDED_EVENTS:
                if verbose:
                    print(f"{domain} {event_id}: EXCLUDED (synthetic fallback)", flush=True)
                continue
            try:
                data = _load_event(domain, mod, cls, event_id)
            except Exception as exc:
                if verbose:
                    print(f"{domain} {event_id}: EVENT_ERR {str(exc)[:80]}", flush=True)
                continue
            if data is None or not data.usable:
                if verbose:
                    print(f"{domain} {event_id}: skip(unusable)", flush=True)
                continue
            events.append(data)
            taken += 1
            if verbose:
                print(
                    f"{domain} {event_id}: X={data.X.shape} pos={data.n_pos}",
                    flush=True,
                )
    return events
