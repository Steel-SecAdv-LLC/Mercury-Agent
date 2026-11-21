#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measure the watch's time thresholds on the records of what they describe.

Three values decide whom the watch tells and when, and none can be reasoned
out; each is a property of the thing measured:

* how soon after a flood clears a return is still the same flood
  (``RiverGaugeFeed.same_episode``);
* the same for a geomagnetic storm (``SpaceWeatherFeed.same_episode``);
* how long an earthquake stays news (``WatchStore.OCCURRENCE_STANDS``), and
  how often its felt extent is worth reading again.

This reads the records and writes what it found to ``evidence.json`` beside
it; ``tests/research/test_watch_thresholds.py`` holds the values to it.

* **River gauges.** The observed stage series (about thirty days at fifteen
  minutes) of every NWS gauge at or above action stage and of a random sample
  of the rest, from the National Water Prediction Service, with each gauge's
  own flood stages. For every time a gauge dropped below minor flood stage
  and came back: how long it was below, and whether it stayed at or above
  action stage (the river still up) throughout.
* **Geomagnetic storms.** The definitive Kp index since 1932 from GFZ
  Potsdam. For every time Kp dropped below G1 (by the library's own
  ``noaa_g_scale``: Kp 5, which begins at "5-") and came back: how many
  three-hour intervals it was below, and whether it stayed at Kp 4
  ("active", from "4-") throughout.
* **Earthquakes.** Every version of USGS's ShakeMap and "Did You Feel It?"
  products for the earthquakes of magnitude 4.5 and above in a window that
  have either: when the first ShakeMap came, and what share of an
  earthquake's felt reports had arrived by each hour.

Usage::

    python research/watch_thresholds/measure.py                 # print the tables
    python research/watch_thresholds/measure.py --collect DIR   # read the sources, rewrite evidence.json
    python research/watch_thresholds/measure.py --from-raw DIR  # rebuild from what --collect saved

``--collect`` makes a few thousand read-only requests to public services and
takes some minutes.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence.json"

KP_URL = "https://kp.gfz.de/app/files/Kp_ap_since_1932.txt"
NWPS = "https://api.water.noaa.gov/nwps/v1/gauges"
USGS = "https://earthquake.usgs.gov/fdsnws/event/1/query"
HEADERS = {"User-Agent": "mercury-agent research/watch_thresholds"}

#: Hours at which the share of an earthquake's felt reports is taken.
REPORT_HOURS = (1, 3, 6, 12, 24, 48, 72, 168)
#: Kp counts in thirds: "4-" is 3.67. NOAA calls Kp 4 "active"; G1 is Kp 5,
#: which begins at "5-" (4.67). The storm test below is the library's own.
ACTIVE = 4.0 - 1.0 / 3.0
_EPS = 1e-6
_MISSING = (None, -999, -9999)


# --------------------------------------------------------------------------- #
# Collecting
# --------------------------------------------------------------------------- #


def _get(url: str, **params: Any) -> Any:
    import requests

    response = requests.get(url, params=params or None, headers=HEADERS, timeout=120)
    response.raise_for_status()
    return response


def collect(raw: Path, sample: int, start: str, end: str) -> None:
    """Read the three sources into ``raw``."""
    from omni_mercury_engine.loaders import FloodLoader

    raw.mkdir(parents=True, exist_ok=True)
    (raw / "kp.txt").write_text(_get(KP_URL).text, encoding="utf-8")

    status = FloodLoader(cache_dir=raw / "cache", max_retries=2).fetch_flood_status()
    up = status[status["category"].isin(["action", "minor", "moderate", "major"])]
    rest = status[status["category"] == "no_flooding"]["gauge_id"].tolist()
    random.seed(20261117)
    chosen = [*up["gauge_id"].tolist(), *random.sample(rest, min(sample, len(rest)))]

    def gauge(gauge_id: str) -> dict[str, Any] | None:
        stages = (_get(f"{NWPS}/{gauge_id}").json().get("flood") or {}).get("categories") or {}
        minor = (stages.get("minor") or {}).get("stage")
        if minor in _MISSING:
            return None
        action = (stages.get("action") or {}).get("stage")
        observed = _get(f"{NWPS}/{gauge_id}/stageflow/observed").json().get("data") or []
        return {
            "id": gauge_id,
            "minor": minor,
            "action": None if action in _MISSING else action,
            "points": [
                (p["validTime"], p["primary"]) for p in observed if p.get("primary") not in _MISSING
            ],
        }

    with (
        (raw / "gauges.jsonl").open("w", encoding="utf-8") as out,
        ThreadPoolExecutor(max_workers=6) as pool,
    ):
        for row in pool.map(gauge, chosen):
            if row is not None:
                out.write(json.dumps(row) + "\n")

    listed = _get(
        USGS, format="geojson", starttime=start, endtime=end, minmagnitude=4.5, orderby="time"
    ).json()["features"]
    have = [f for f in listed if {"shakemap", "dyfi"} & set(f["properties"]["types"].split(","))]

    def versions(feature: dict[str, Any]) -> dict[str, Any]:
        detail = _get(USGS, format="geojson", eventid=feature["id"], includesuperseded="true")
        products = detail.json()["properties"].get("products") or {}

        def live(name: str) -> list[dict[str, Any]]:
            return [p for p in products.get(name, []) if p.get("status") != "DELETE"]

        return {
            "id": feature["id"],
            "mag": feature["properties"]["mag"],
            "time": feature["properties"]["time"],
            "shakemap": sorted(int(p["updateTime"]) for p in live("shakemap")),
            "dyfi": sorted(
                (int(p["updateTime"]), int(p["properties"].get("num-responses") or 0))
                for p in live("dyfi")
            ),
        }

    with (
        (raw / "usgs.jsonl").open("w", encoding="utf-8") as out,
        ThreadPoolExecutor(max_workers=4) as pool,
    ):
        for row in pool.map(versions, have):
            out.write(json.dumps(row) + "\n")
    (raw / "window.json").write_text(
        json.dumps({"start": start, "end": end, "listed": len(listed)}), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# Measuring
# --------------------------------------------------------------------------- #


def _runs(above: list[bool]) -> list[tuple[int, int]]:
    """Each maximal run of ``True`` as ``(first, last)``."""
    runs, start = [], None
    for index, flag in enumerate(above):
        if flag and start is None:
            start = index
        if start is not None and (not flag or index == len(above) - 1):
            runs.append((start, index if flag else index - 1))
            start = None
    return runs


def measure_gauges(raw: Path) -> dict[str, Any]:
    """Every time a gauge dropped below minor flood stage and came back."""
    gauges = readings = flooded = runs_total = 0
    returns: list[list[Any]] = []
    for line in (raw / "gauges.jsonl").read_text(encoding="utf-8").splitlines():
        gauge = json.loads(line)
        points = [(datetime.fromisoformat(t.replace("Z", "+00:00")), v) for t, v in gauge["points"]]
        if len(points) <= 50:
            continue
        gauges += 1
        readings += len(points)
        runs = _runs([value >= gauge["minor"] for _, value in points])
        if not runs:
            continue
        flooded += 1
        runs_total += len(runs)
        action = gauge["action"]
        graded = action is not None and action < gauge["minor"]
        for (_, last), (first, _) in pairwise(runs):
            below = points[last + 1 : first]
            hours = (points[first][0] - below[0][0]).total_seconds() / 3600.0
            stayed = (min(v for _, v in below) >= action) if graded else None
            returns.append([round(hours, 2), stayed])
    return {
        "source": "NOAA National Water Prediction Service: /gauges/{id} flood stages and "
        "/gauges/{id}/stageflow/observed, every gauge at action stage or above and a random "
        "sample of the others",
        "gauges": gauges,
        "readings": readings,
        "gauges_that_flooded": flooded,
        "flood_runs": runs_total,
        "returns": sorted(returns, key=lambda r: r[0]),
    }


def measure_kp(raw: Path) -> dict[str, Any]:
    """Every time Kp dropped below G1 and came back, by intervals below."""
    values: list[float] = []
    first = last = ""
    for line in (raw / "kp.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if float(parts[7]) < 0:
            continue
        values.append(float(parts[7]))
        last = "-".join(parts[:3])
        first = first or last
    from omni_mercury_engine.evaluation.hazard_metrics import noaa_g_scale

    runs = _runs([noaa_g_scale(min(max(kp, 0.0), 9.0)) >= 1 for kp in values])
    by_gap: dict[int, list[int]] = {}
    for (_, end), (start, _) in pairwise(runs):
        gap = start - end - 1
        entry = by_gap.setdefault(gap, [0, 0])
        entry[0] += 1
        entry[1] += min(values[end + 1 : start]) >= ACTIVE - _EPS
    return {
        "source": f"GFZ Potsdam definitive Kp, {KP_URL}",
        "first": first,
        "last": last,
        "intervals": len(values),
        "storm_runs": len(runs),
        "returns_by_intervals_below": {str(g): by_gap[g] for g in sorted(by_gap)},
    }


def measure_earthquakes(raw: Path) -> dict[str, Any]:
    """When the first ShakeMap came, and how fast the felt reports arrived."""
    hour = 3600.0 * 1000.0
    events = [
        json.loads(line) for line in (raw / "usgs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    first_map = sorted(
        round((e["shakemap"][0] - e["time"]) / hour, 3) for e in events if e["shakemap"]
    )
    shares = []
    for event in events:
        final = event["dyfi"][-1][1] if event["dyfi"] else 0
        if final < 10:
            continue
        shares.append(
            [
                round(
                    max([n for at, n in event["dyfi"] if (at - event["time"]) / hour <= cut] or [0])
                    / final,
                    3,
                )
                for cut in REPORT_HOURS
            ]
        )
    window = json.loads((raw / "window.json").read_text(encoding="utf-8"))
    return {
        "source": f"USGS ComCat, {USGS} with includesuperseded=true, magnitude 4.5 and above "
        f"from {window['start']} to {window['end']}",
        "listed": window["listed"],
        "with_a_product": len(events),
        "first_shakemap_hours": first_map,
        "report_hours": list(REPORT_HOURS),
        "report_shares": shares,
    }


def build(raw: Path) -> dict[str, Any]:
    """The evidence from what ``collect`` saved."""
    return {
        "measured": datetime.now(UTC).strftime("%Y-%m-%d"),
        "river_gauges": measure_gauges(raw),
        "geomagnetic": measure_kp(raw),
        "earthquakes": measure_earthquakes(raw),
    }


# --------------------------------------------------------------------------- #
# Reading the evidence
# --------------------------------------------------------------------------- #


def episodes_told(returns: list[list[Any]], runs: int, window_hours: float) -> int:
    """How many of ``runs`` are told when returns within the window are merged."""
    return runs - sum(1 for hours, _ in returns if hours <= window_hours)


def share_with_reports(shares: list[list[float]], hours: list[int], at: int, least: float) -> float:
    """The share of earthquakes that had ``least`` of their reports by ``at`` hours."""
    column = hours.index(at)
    return sum(1 for row in shares if row[column] >= least) / len(shares)


def report(evidence: dict[str, Any]) -> str:
    """The evidence as the tables the thresholds were read from."""
    from omni_mercury_engine.watch.feeds import RiverGaugeFeed, SpaceWeatherFeed
    from omni_mercury_engine.watch.store import WatchStore

    lines = [f"Measured {evidence['measured']}", ""]
    river = evidence["river_gauges"]
    returns, runs = river["returns"], river["flood_runs"]
    window = RiverGaugeFeed.same_episode or timedelta(0)
    lines += [
        f"River gauges: {river['gauges']:,} gauges, {river['readings']:,} readings; "
        f"{river['gauges_that_flooded']} reached minor flood stage in {runs} runs; "
        f"{len(returns)} returns.",
        "  hours below flood stage   returns   still at action stage",
    ]
    for low, high in ((0, 3), (3, 6), (6, 12), (12, 26), (26, 30), (30, 10**6)):
        band = [r for r in returns if low < r[0] <= high]
        graded = [r for r in band if r[1] is not None]
        still = f"{100 * sum(r[1] for r in graded) / len(graded):.0f}%" if graded else "-"
        label = f"{low}-{high}" if high < 10**5 else f"over {low}"
        lines.append(f"  {label:>23}   {len(band):7d}   {still:>20}")
    hours = window.total_seconds() / 3600.0
    lines += [
        f"  window {hours:g} h: {runs} runs are {episodes_told(returns, runs, hours)} episodes.",
        "",
    ]
    kp = evidence["geomagnetic"]
    lines.append(
        f"Geomagnetic storms: Kp {kp['first']} to {kp['last']}, {kp['intervals']:,} intervals, "
        f"{kp['storm_runs']:,} runs at G1 or above."
    )
    lines.append("  intervals below G1   returns   stayed at Kp 4")
    for gap in ("1", "2", "3", "4"):
        count, stayed = kp["returns_by_intervals_below"][gap]
        lines.append(f"  {gap:>18}   {count:7d}   {100 * stayed / count:13.0f}%")
    storm = SpaceWeatherFeed.same_episode or timedelta(0)
    lines += [f"  window {storm.total_seconds() / 3600.0:g} h covers two intervals.", ""]
    quakes = evidence["earthquakes"]
    maps, shares = quakes["first_shakemap_hours"], quakes["report_shares"]
    lines.append(
        f"Earthquakes: {quakes['with_a_product']} of {quakes['listed']:,} had a ShakeMap or "
        f"felt reports; {len(maps)} ShakeMaps, {len(shares)} with ten or more reports."
    )
    lines.append(
        f"  first ShakeMap: median {60 * statistics.median(maps):.0f} min; "
        f"{100 * sum(m <= 1 for m in maps) / len(maps):.0f}% within 1 h, "
        f"{100 * sum(m <= 3 for m in maps) / len(maps):.0f}% within 3 h."
    )
    lines.append("  by hour   median share of reports   earthquakes with 90% of theirs")
    for column, cut in enumerate(quakes["report_hours"]):
        median = statistics.median(row[column] for row in shares)
        most = share_with_reports(shares, quakes["report_hours"], cut, 0.9)
        lines.append(f"  {cut:7d}   {100 * median:22.0f}%   {100 * most:29.0f}%")
    stands = WatchStore.OCCURRENCE_STANDS.total_seconds() / 3600.0
    lines.append(f"  an earthquake stands {stands:g} h.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Print the tables, or collect and rebuild the evidence."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--collect", type=Path, help="read the sources into this directory")
    parser.add_argument("--from-raw", type=Path, help="rebuild from a --collect directory")
    parser.add_argument("--sample", type=int, default=1500, help="gauges sampled, beside those up")
    parser.add_argument("--start", default="2026-09-01")
    parser.add_argument("--end", default="2026-11-16")
    args = parser.parse_args(argv)
    raw = args.collect or args.from_raw
    if args.collect:
        collect(args.collect, args.sample, args.start, args.end)
    if raw:
        EVIDENCE.write_text(json.dumps(build(raw), separators=(",", ":")) + "\n", encoding="utf-8")
    print(report(json.loads(EVIDENCE.read_text(encoding="utf-8"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
