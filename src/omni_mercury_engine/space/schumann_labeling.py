# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Weak-supervision labels for the Schumann sub-net (WS-C).

The prior session claimed no labels could be constructed for the Schumann/ELF
sub-net. That is false: independent, **public-domain** geophysical catalogs
physically perturb the ionosphere -- and therefore the Schumann-resonance
parameters -- on documented timescales, so they can be used as *weak* labels:

* **NOAA SWPC planetary Kp index** -- NOAA G1 or stronger (Kp 5- and up, via
  :func:`~omni_mercury_engine.evaluation.hazard_metrics.noaa_g_scale`) marks a
  geomagnetic storm.
* **NOAA SWPC GOES X-ray long band (0.1-0.8 nm)** -- ``>= 1e-5 W/m^2`` is an
  M-class flare, ``>= 1e-4`` an X-class flare. Flares drive prompt Sudden
  Ionospheric Disturbances (SID) in the D-region.

Both feeds are US-Government public domain (no copyright; freely redistributable)
and reachable without authentication via ``services.swpc.noaa.gov`` (already on
the project's trusted-endpoint allowlist).

These are **proxy / event-coincidence** labels, NOT hand-annotated Schumann
ground truth. ``label_noise_disclosure()`` states the failure modes explicitly.
Nothing here lifts the Schumann sub-net out of quarantine on its own -- that
requires an openly-licensed *real* ELF corpus the proxy labels can be applied to
(see ``docs/SCHUMANN_PREREGISTRATION.md``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from omni_mercury_engine.datasets.base import http_get_with_retry
from omni_mercury_engine.evaluation.hazard_metrics import noaa_g_scale, parse_kp

# Public-domain NOAA SWPC endpoints (no auth; host is trusted-allowlisted).
KP_URL = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
XRAY_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"

# A-priori thresholds (documented, fixed before evaluation). The storm
# threshold is the NOAA G1 onset, Kp 5- (14/3 ~ 4.667): classification goes
# through hazard_metrics.noaa_g_scale, never a literal "Kp >= 5.0" (which
# would depend on the feed's integer-filing convention).
KP_STORM_THRESHOLD = 14.0 / 3.0  # NOAA G1 onset, Kp 5- (documentation; see noaa_g_scale)
FLARE_M_CLASS = 1e-5  # W/m^2, GOES long band
FLARE_X_CLASS = 1e-4

# Documented physical lag windows (how long after the driver the ionospheric /
# Schumann response is expected). Flares: prompt SID, minutes -> ~1 h. Storms:
# multi-hour storm-time response; widen slightly on both sides.
FLARE_LAG = timedelta(minutes=60)
STORM_LEAD = timedelta(hours=1)
STORM_LAG = timedelta(hours=3)


def _parse_time(s: str) -> datetime:
    s = s.replace("Z", "").strip()
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


def _row_kp(row: dict[str, Any]) -> float | None:
    """The most precise Kp a SWPC 1-minute feed row carries, or ``None``.

    Preference order: the thirds string (``"5M"`` = 5- = 4.667), then
    ``estimated_kp`` (already a thirds float), then ``kp_index`` -- SWPC's
    filing integer, under which every third of 5 (5-, 5o, 5+) files as 5, so
    old snapshots that carry only it still classify to the same G level.
    """
    for key in ("kp", "estimated_kp", "kp_index"):
        value = row.get(key)
        if value is None:
            continue
        try:
            return parse_kp(value)
        except ValueError:
            continue
    return None


@dataclass
class EventWindow:
    """A positive (anomaly-expected) interval coincident with a driver."""

    start: datetime
    end: datetime
    driver: str  # "geomagnetic_storm" | "M_flare" | "X_flare"
    magnitude: float

    def contains(self, t: datetime) -> bool:
        """Return whether ``t`` falls inside this event window."""
        return self.start <= t <= self.end


@dataclass
class LabelCatalog:
    """Weak-label catalog + full provenance."""

    windows: list[EventWindow]
    provenance: dict[str, Any] = field(default_factory=dict)

    def label(self, t: datetime) -> int:
        """1 if ``t`` falls in any driver window (with lag), else 0 (quiet)."""
        return int(any(w.contains(t) for w in self.windows))

    def positive_fraction(self, times: list[datetime]) -> float:
        """Return the fraction of timestamps covered by positive windows."""
        if not times:
            return 0.0
        return sum(self.label(t) for t in times) / len(times)


def _fetch(url: str) -> tuple[bytes, Any]:
    raw = http_get_with_retry(url, timeout=30, retries=3)
    return raw, json.loads(raw.decode())


def fetch_catalogs(
    *,
    kp_json: Any | None = None,
    xray_json: Any | None = None,
) -> LabelCatalog:
    """Fetch NOAA Kp + GOES X-ray and derive driver windows with provenance.

    ``kp_json`` / ``xray_json`` may be supplied directly (e.g. a committed
    snapshot) to make labeling reproducible and offline-testable; otherwise the
    live public-domain feeds are fetched.
    """
    provenance: dict[str, Any] = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "sources": [],
        "thresholds": {
            "kp_storm": KP_STORM_THRESHOLD,
            "flare_m_class_wm2": FLARE_M_CLASS,
            "flare_x_class_wm2": FLARE_X_CLASS,
        },
        "lag_windows": {
            "flare_minutes": FLARE_LAG.total_seconds() / 60,
            "storm_lead_hours": STORM_LEAD.total_seconds() / 3600,
            "storm_lag_hours": STORM_LAG.total_seconds() / 3600,
        },
        "label_kind": "proxy_event_coincidence",
        "license": "US Government public domain (NOAA SWPC)",
    }

    if kp_json is None:
        raw, kp_json = _fetch(KP_URL)
        provenance["sources"].append(
            {"url": KP_URL, "sha256": hashlib.sha256(raw).hexdigest(), "rows": len(kp_json)}
        )
    if xray_json is None:
        raw, xray_json = _fetch(XRAY_URL)
        provenance["sources"].append(
            {"url": XRAY_URL, "sha256": hashlib.sha256(raw).hexdigest(), "rows": len(xray_json)}
        )

    windows: list[EventWindow] = []

    # --- Geomagnetic storms (NOAA G1 or stronger) ---
    for row in kp_json:
        kp = _row_kp(row)
        if kp is None:
            continue
        if noaa_g_scale(kp) >= 1:
            t = _parse_time(row["time_tag"])
            windows.append(EventWindow(t - STORM_LEAD, t + STORM_LAG, "geomagnetic_storm", kp))

    # --- Solar flares (GOES long band) ---
    for row in xray_json:
        if row.get("energy") != "0.1-0.8nm":
            continue
        flux = row.get("flux")
        if flux is None or float(flux) < FLARE_M_CLASS:
            continue
        t = _parse_time(row["time_tag"])
        driver = "X_flare" if float(flux) >= FLARE_X_CLASS else "M_flare"
        windows.append(EventWindow(t, t + FLARE_LAG, driver, float(flux)))

    provenance["n_windows"] = len(windows)
    provenance["n_storm_windows"] = sum(1 for w in windows if w.driver == "geomagnetic_storm")
    provenance["n_flare_windows"] = sum(1 for w in windows if w.driver.endswith("flare"))
    return LabelCatalog(windows=windows, provenance=provenance)


def label_noise_disclosure() -> dict[str, str]:
    """Explicit, quantifiable failure modes of these proxy labels.

    Transparency contract: these are event-coincidence labels, not verified Schumann
    anomalies. A model scored against them is scored against a *noisy* target.
    """
    return {
        "false_positive": (
            "Not every flare/storm produces a measurable Schumann perturbation; "
            "some positive-labelled windows carry no real ELF anomaly."
        ),
        "false_negative": (
            "Schumann anomalies have other drivers (local lightning/Q-bursts, "
            "instrumental, seismo-EM); genuine anomalies in quiet windows are "
            "mislabelled negative."
        ),
        "timing": (
            "Lag windows are documented approximations, not per-event ground "
            "truth; window edges are inherently fuzzy."
        ),
        "implication": (
            "Reported metrics are upper-bounded by this label noise. Treat as "
            "weak supervision; do NOT present as hand-annotated ground truth."
        ),
    }


__all__ = [
    "FLARE_M_CLASS",
    "FLARE_X_CLASS",
    "KP_STORM_THRESHOLD",
    "EventWindow",
    "LabelCatalog",
    "fetch_catalogs",
    "label_noise_disclosure",
]
