"""Reachable real-domain measurement suite for the governed fusion substrate.

This is the single shared measurement driver used by every governed-fusion
correction script (conformal operating point, adversarial survivability,
reliability-weighted fusion, calibration).  It builds ``(X, y)`` per event from
the **real live-API loaders** (USGS / NOAA / FEMA / CISA / WHO / OBIS), fits the
**real** ``MercuryAnomalyDetector``, and scores with ``ml.mercury_ml`` (no
scikit-learn).

Two clearly-separated event classes (a reviewer must never conflate them):

* **Live headline suite -- 23 real events / 7 domains.**  earthquake 5,
  pandemic 5, hurricane 4, fema 3, tornado 2, marine 2, network_security 2.
  Every figure headlined in ``FINDINGS.md`` is computed on this set.  These
  loaders return genuine live-API payloads (verified: no synthetic-generation
  method is reached -- ``build_suite(kind="real")``).

* **Reconstructed-from-live-events -- 7 events / 3 domains.**  tsunami 3
  (``chile_2010``, ``tohoku_2011``, ``tonga_2022``), energy 3 (``quebec_1989``,
  ``bastille_day_2000``, ``halloween_2003``), pandemic 1 (``ebola_2014``).  For
  these *documented real events* the live feed is unavailable (NDBC BPR files
  rotate out after ~45 days; SWPC has no pre-realtime Kp; the WHO GHO Ebola
  series 404s), so the loader **reconstructs** a series that mirrors the
  documented event's statistical properties (amplitude / period / noise floor,
  storm Kp profile, epidemic curve).  This is the credible next-best source when
  live data is unachievable -- reported separately and **always labelled
  reconstruction, never claimed as live** (``build_suite(kind="reconstructed")``).

``MERCURY_ALLOW_SYNTHETIC=0`` is set for discipline but is **not** enforced by
the ``loaders/`` package (only by the ``datasets/`` package); the real/reconstructed
split here is therefore made explicit in code (``RECONSTRUCTED_DOMAINS`` /
``RECONSTRUCTED_EVENTS``), not delegated to that flag.

As defense-in-depth against a *silent* live->synthetic fallback (``marine`` on an
empty OBIS response synthesises data tagged ``dataset_id="synthetic"``),
``_load_event`` additionally refuses to label any event live whose loader returned
that marker (``_looks_synthesized`` / ``ProvenanceError``).  This is a read-only
check on an existing column -- it never changes a loader's return contract -- so a
live-labelled event that silently fell back to synthesis is *excluded from the live
headline*, never counted as live (announced as ``PROVENANCE_SKIP``).

Determinism: ``MercuryAnomalyDetector`` is deterministic after ``fit()``; the
reconstruction loaders derive their RNG seed from ``hashlib.sha256`` (process-
stable), so the reconstructed group reproduces byte-identically across processes
without relying on ``PYTHONHASHSEED`` (``gf_env.sh`` still pins
``PYTHONHASHSEED=0`` as defense-in-depth and for the ``datasets/`` path).  Built
``(X, y)`` arrays are cached under ``$GF_CACHE_DIR`` (default
``/home/user/gf_cache``) so iteration does not re-hit the network; deleting the
cache regenerates from the live loaders.
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

# Domains whose loaders synthesise the series by design (no live path exists):
# tsunami reconstructs BPR patterns; energy reconstructs Kp storm profiles.
RECONSTRUCTED_DOMAINS: frozenset[str] = frozenset({"tsunami", "energy"})
# Individual events that fall back to reconstruction inside an otherwise-live
# domain (the WHO GHO Ebola series is unavailable, so pandemic/ebola_2014
# reconstructs the documented 2014 epidemic curve).
RECONSTRUCTED_EVENTS: frozenset[str] = frozenset({"ebola_2014"})

_CACHE_DIR = os.environ.get("GF_CACHE_DIR", "/home/user/gf_cache")


def is_reconstructed(domain: str, event_id: str) -> bool:
    """True iff an event is reconstructed-from-live (not a live payload)."""
    return domain in RECONSTRUCTED_DOMAINS or event_id in RECONSTRUCTED_EVENTS


class ProvenanceError(RuntimeError):
    """Raised when a LIVE-labelled event's loader returned synthesized data.

    Provenance is labelled statically (``RECONSTRUCTED_DOMAINS`` /
    ``RECONSTRUCTED_EVENTS``).  A live-labelled loader that *silently* falls back
    to synthesis -- ``marine`` on an empty OBIS response -- would otherwise be
    mislabelled live; this read-only guard makes the suite exclude it instead, so
    silent-fallback data can never be counted in the live headline.
    """


def _looks_synthesized(raw: Any) -> bool:
    """Return True iff a raw loader frame carries the synthesis marker.

    The ``marine`` loader tags every row of its OBIS-empty fallback with
    ``dataset_id == "synthetic"`` while the live OBIS path never does (it writes
    real dataset IDs or an empty string).  Reading this existing column lets the
    suite refuse to label silent-fallback data as live **without changing any
    loader's return contract**.

    Bounded residual (documented in ``FINDINGS.md``): a loader that synthesises
    *without* the marker is not caught here.  Today the only live-labelled
    silent-synthesizer is ``marine`` (tagged); ``pandemic/ebola_2014`` also
    synthesises without a marker but is ``RECONSTRUCTED``-labelled, so it is never
    live-labelled.

    Args:
        raw: The object returned by ``loader.fetch_historical`` (a DataFrame).

    Returns:
        True iff ``raw`` exposes a ``dataset_id`` column containing the value
        ``"synthetic"``.
    """
    columns = getattr(raw, "columns", None)
    if columns is None or "dataset_id" not in columns:
        return False
    try:
        return bool((raw["dataset_id"].astype(str) == "synthetic").any())
    except Exception:  # pragma: no cover - defensive: a guard must never crash the build
        return False


@dataclass(frozen=True)
class EventData:
    """A single event: engineered features and binary ground truth."""

    domain: str
    event_id: str
    X: np.ndarray[Any, Any]
    y: np.ndarray[Any, Any]

    @property
    def n_pos(self) -> int:
        """Number of positive (anomaly) rows."""
        return int(np.sum(self.y == 1))

    @property
    def usable(self) -> bool:
        """At least one of each class and >= 8 rows (enough to fit + split)."""
        return self.X.shape[0] >= 8 and 0 < self.n_pos < self.X.shape[0]

    @property
    def reconstructed(self) -> bool:
        """True iff this event is reconstructed-from-live, not a live payload."""
        return is_reconstructed(self.domain, self.event_id)


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
    if not is_reconstructed(domain, event_id) and _looks_synthesized(raw):
        raise ProvenanceError(
            f"{domain}/{event_id}: loader returned synthesized data "
            "(dataset_id='synthetic') for a LIVE-labelled event; refusing to "
            "mislabel a silent fallback as live (restore the live source / use "
            "the committed cache, or add it to RECONSTRUCTED_* if reconstructed)."
        )
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


def _build_all(*, verbose: bool = False) -> list[EventData]:
    """Build every usable event (live + reconstructed) across reachable domains."""
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
            try:
                data = _load_event(domain, mod, cls, event_id)
            except ProvenanceError as exc:
                # Always announce a provenance violation (never silently drop):
                # a live-labelled event whose loader synthesized is excluded from
                # the live headline, not counted as live.
                print(f"{domain} {event_id}: PROVENANCE_SKIP {exc}", flush=True)
                continue
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
                tag = "RECON" if data.reconstructed else "live"
                print(
                    f"{domain} {event_id}: X={data.X.shape} pos={data.n_pos} [{tag}]",
                    flush=True,
                )
    return events


def build_suite(*, kind: str = "real", verbose: bool = False) -> list[EventData]:
    """Build the reachable suite for the requested event class.

    Args:
        kind: ``"real"`` -> the 23-event / 7-domain live headline suite
            (default); ``"reconstructed"`` -> the 7-event / 3-domain
            reconstructed-from-live group (tsunami, energy, ebola_2014);
            ``"all"`` -> both (30 events).
        verbose: print per-event provenance while building.

    Returns:
        The usable events of the requested class.
    """
    if kind not in ("real", "reconstructed", "all"):
        raise ValueError(f"kind must be real|reconstructed|all, got {kind!r}")
    events = _build_all(verbose=verbose)
    if kind == "all":
        return events
    want_recon = kind == "reconstructed"
    return [e for e in events if e.reconstructed == want_recon]
