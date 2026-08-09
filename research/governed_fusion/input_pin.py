# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Verify that a governed-fusion measurement ran on the pinned inputs.

``build_manifest.py`` records a SHA-256 of every event's ``(X, y)`` -- but until
now nothing ever *checked* one. The manifest was a record, not a gate, and that
is the reason the suite's per-event AUROC drift has stood unexplained.

The concrete symptom: the external-label headline is the mean of two events, and
a 2026-08-04 refit in a fresh environment moved them ``nsl_kdd 0.679 -> 0.728``
and ``batadal 0.862 -> 0.889`` -- a headline swing of ``0.770 -> 0.809``. Two
explanations fit that observation equally well:

* the **inputs** changed (a loader returned different rows on a different day), or
* the **code or environment** changed (a library version, a numerical path).

Nothing in the repository could tell those apart, so neither could be ruled out
and the number could not be improved: any gain smaller than the drift is
unfalsifiable. This module separates them. Re-hash the built ``(X, y)`` and
compare against the pin:

* hashes **match** and the metric moved  -> the movement is code/environment, and
  is a real finding worth chasing.
* hashes **differ**                      -> the inputs moved; the metric is not
  comparable and must not be published or promoted against.

That is the whole contribution: it converts a silent ±0.05 into an attributable
signal. It does not by itself make the number stable -- it makes the number's
instability *diagnosable*, which is the prerequisite for fixing it.

Determinism note: the detector itself is not the problem. ``MercuryAnomalyDetector``
was measured bit-identical across processes and across ``OMP_NUM_THREADS=1/4`` on
fixed input, and ``stratified_subsample`` is seeded. The remaining degrees of
freedom are the loaders and the environment, which is what this checks.

CLI::

    python research/governed_fusion/input_pin.py            # verify the live suite
    python research/governed_fusion/input_pin.py --kind reconstructed
    python research/governed_fusion/input_pin.py --check    # non-zero exit on drift
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

_GF_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = _GF_DIR / "manifest.json"

#: Row cap and seed the manifest pins its ``capped`` digest under. Kept here so
#: the writer and the checker cannot drift apart -- see ``build_manifest.CAP``.
CAP = 6000
SEED = 42


class PinStatus(StrEnum):
    """Per-event verdict."""

    MATCH = "match"  # rebuilt (X, y) hashes to the pinned digest
    DRIFT = "drift"  # rebuilt successfully, but the content differs
    UNPINNED = "unpinned"  # event is in the suite with no manifest entry
    UNREACHABLE = "unreachable"  # event could not be rebuilt (loader/network)


def sha256_xy(X: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> str:
    """Deterministic, shape-tagged SHA-256 over canonical ``(X, y)`` bytes.

    Canonicalizes dtype and memory layout first, so an array that is
    semantically identical but C- vs F-ordered, or float32 vs float64, still
    hashes the same. Shapes are folded in so a reshape cannot collide.

    This is the single source of truth for the digest: ``build_manifest`` writes
    it and this module checks it, from the same function, so the writer and the
    checker cannot disagree about what "the same data" means.
    """
    h = hashlib.sha256()
    xc = np.ascontiguousarray(X, dtype=np.float64)
    yc = np.ascontiguousarray(y, dtype=np.int64).reshape(-1)
    h.update(b"X" + str(xc.shape).encode())
    h.update(xc.tobytes())
    h.update(b"y" + str(yc.shape).encode())
    h.update(yc.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class EventPin:
    """One event's verification result."""

    domain: str
    event_id: str
    status: PinStatus
    pinned: str | None = None
    observed: str | None = None
    detail: str = ""

    @property
    def key(self) -> str:
        """``domain/event_id``."""
        return f"{self.domain}/{self.event_id}"

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary."""
        return {
            "domain": self.domain,
            "event_id": self.event_id,
            "status": str(self.status),
            "pinned": self.pinned,
            "observed": self.observed,
            "detail": self.detail,
        }


@dataclass
class PinReport:
    """Suite-level verification result."""

    kind: str
    events: list[EventPin] = field(default_factory=list)

    def by_status(self, status: PinStatus) -> list[EventPin]:
        """Events with the given verdict."""
        return [e for e in self.events if e.status is status]

    @property
    def drifted(self) -> list[EventPin]:
        """Events whose content no longer matches the pin."""
        return self.by_status(PinStatus.DRIFT)

    @property
    def ok(self) -> bool:
        """True when nothing drifted and nothing was unpinned.

        ``UNREACHABLE`` is deliberately *not* a failure: an upstream being down
        is an availability fact, not evidence that the data changed. It is
        reported so a partial verification is never mistaken for a full one.
        """
        return not self.drifted and not self.by_status(PinStatus.UNPINNED)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary."""
        return {
            "kind": self.kind,
            "n_events": len(self.events),
            "counts": {str(s): len(self.by_status(s)) for s in PinStatus if self.by_status(s)},
            "ok": self.ok,
            "events": [e.to_dict() for e in self.events],
        }


def load_pins(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, dict[str, Any]]:
    """Return ``{"domain/event_id": entry}`` for every event in the manifest."""
    with manifest_path.open(encoding="utf-8") as fh:
        manifest = json.load(fh)
    pins: dict[str, dict[str, Any]] = {}
    for group in ("real", "reconstructed"):
        for entry in manifest.get(group, []) or []:
            pins[f"{entry['domain']}/{entry['event_id']}"] = entry
    return pins


def verify_suite(
    kind: str = "real",
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    only: Iterable[str] | None = None,
) -> PinReport:
    """Rebuild each event in the suite and compare its digest to the pin.

    Args:
        kind: ``"real"`` (live headline) or ``"reconstructed"``.
        manifest_path: Manifest carrying the pinned digests.
        only: Optional ``domain/event_id`` filter -- useful for verifying just
            the external-label events the fitness signal actually reads.

    Returns:
        A :class:`PinReport`. Building an event can hit the network, so an
        unreachable upstream is recorded rather than raised.
    """
    from research.governed_fusion.suite import build_suite

    pins = load_pins(manifest_path)
    wanted = set(only) if only is not None else None
    report = PinReport(kind=kind)

    try:
        events = build_suite(kind=kind)
    except Exception as exc:  # pragma: no cover - environment dependent
        report.events.append(
            EventPin("", "", PinStatus.UNREACHABLE, detail=f"suite build failed: {exc}")
        )
        return report

    for ev in events:
        key = f"{ev.domain}/{ev.event_id}"
        if wanted is not None and key not in wanted:
            continue
        entry = pins.get(key)
        if entry is None:
            report.events.append(
                EventPin(ev.domain, ev.event_id, PinStatus.UNPINNED, detail="no manifest entry")
            )
            continue
        try:
            observed = sha256_xy(ev.X, ev.y)
        except Exception as exc:  # pragma: no cover - defensive
            report.events.append(
                EventPin(ev.domain, ev.event_id, PinStatus.UNREACHABLE, detail=str(exc))
            )
            continue
        pinned = str(entry.get("sha256_xy", ""))
        status = PinStatus.MATCH if observed == pinned else PinStatus.DRIFT
        detail = ""
        if status is PinStatus.DRIFT:
            detail = (
                f"rows {entry.get('n_rows')} -> {int(ev.X.shape[0])}, "
                f"pos {entry.get('n_pos')} -> {int(ev.n_pos)}"
            )
        report.events.append(EventPin(ev.domain, ev.event_id, status, pinned, observed, detail))
    return report


def external_label_keys(manifest_path: Path = DEFAULT_MANIFEST) -> list[str]:
    """The ``domain/event_id`` keys the fitness signal reads (external labels).

    These are the only events whose metric is claimed as skill, so they are the
    ones whose pins matter most: a drift here invalidates the headline itself,
    not merely a per-domain row.
    """
    return sorted(
        key for key, entry in load_pins(manifest_path).items() if entry.get("external_label")
    )


def external_label_count_from_entries(manifest: Mapping[str, object]) -> int | None:
    """Count external-label events by walking the per-event entries.

    ``None`` when the manifest carries no entry lists at all (some callers pass
    a summary-only manifest), which is distinct from a genuine count of zero.
    """
    seen_any = False
    n = 0
    for group in ("real", "reconstructed"):
        entries = manifest.get(group)
        if not isinstance(entries, list):
            continue
        seen_any = True
        n += sum(1 for e in entries if isinstance(e, dict) and e.get("external_label"))
    return n if seen_any else None


def external_label_count_from_summary(manifest: Mapping[str, object]) -> int | None:
    """Count external-label events from ``provenance_summary``; ``None`` if absent.

    This is the field ``promotion_gate._external_label_count`` reads.
    """
    summary = manifest.get("provenance_summary")
    if not isinstance(summary, dict):
        return None
    real = summary.get("real")
    if not isinstance(real, dict):
        return None
    bucket = real.get("external_label")
    if not isinstance(bucket, dict):
        return None
    n_events = bucket.get("n_events")
    if not isinstance(n_events, (int, float, str)):
        return None
    try:
        return int(n_events)
    except (TypeError, ValueError):
        return None


def external_label_count(manifest: Mapping[str, object]) -> int:
    """Number of external-label events this manifest pins.

    Prefers the per-event entries (the ground truth) and falls back to the
    ``provenance_summary`` rollup when a caller passes a summary-only manifest.
    """
    from_entries = external_label_count_from_entries(manifest)
    if from_entries is not None:
        return from_entries
    return external_label_count_from_summary(manifest) or 0


def manifest_consistency_reasons(manifest: Mapping[str, object]) -> list[str]:
    """Reasons the manifest disagrees with itself about its own fitness set.

    The manifest states the external-label count twice: once as a rollup in
    ``provenance_summary`` and once implicitly in the per-event entries. The
    promotion gate reads the rollup; the input pin reads the entries. Nothing
    previously required them to agree, so a manifest could pin one event set and
    *report* another, and every downstream number would silently describe a
    different suite than the one named. Cheap to check, so it is checked.
    """
    entries = external_label_count_from_entries(manifest)
    summary = external_label_count_from_summary(manifest)
    if entries is None or summary is None:
        return []
    if entries != summary:
        return [
            f"manifest disagrees with itself: provenance_summary reports {summary} "
            f"external-label events but {entries} entries carry external_label=true"
        ]
    return []


def verify_pinned_results(
    results: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
) -> list[str]:
    """Return reasons a results blob must not be compared against this manifest.

    Cheap, offline, no rebuild: checks that a results document was measured over
    the same external-label event set the manifest pins. A candidate measured
    over a different set is not comparable to the baseline however good its
    numbers look -- the gate would be reading a change in *data* as a change in
    *capability*, which is precisely the error that makes an improvement claim
    unfalsifiable.

    Deliberately silent when the candidate declares nothing: this function
    reports contradictions, and absence of a declaration is handled by the
    gate's own required-field checks rather than duplicated here.
    """
    reasons = manifest_consistency_reasons(manifest)
    declared = results.get("external_label_events")
    if declared is None:
        return reasons
    if not isinstance(declared, (int, float, str)):
        reasons.append(f"external_label_events is not an integer: {declared!r}")
        return reasons
    try:
        declared_n = int(declared)
    except (TypeError, ValueError):
        reasons.append(f"external_label_events is not an integer: {declared!r}")
        return reasons
    expected = external_label_count(manifest)
    if declared_n != expected:
        reasons.append(
            f"measured over {declared_n} external-label events but the manifest pins "
            f"{expected}; a metric from a different event set is not comparable to "
            "the baseline"
        )
    return reasons


def _print_report(report: PinReport) -> None:
    print(f"\n==== input pin verification: {report.kind} suite ====")
    for ev in sorted(report.events, key=lambda e: e.key):
        mark = {
            PinStatus.MATCH: "OK   ",
            PinStatus.DRIFT: "DRIFT",
            PinStatus.UNPINNED: "UNPIN",
            PinStatus.UNREACHABLE: "UNREA",
        }[ev.status]
        line = f"  {mark} {ev.key}"
        if ev.status is PinStatus.DRIFT:
            line += f"  pinned={ev.pinned[:12] if ev.pinned else '?'} observed={ev.observed[:12] if ev.observed else '?'}  {ev.detail}"
        elif ev.detail:
            line += f"  {ev.detail}"
        print(line)
    counts = report.to_dict()["counts"]
    print(f"  -- {counts}  ok={report.ok}")
    if report.drifted:
        print(
            "\n  Inputs drifted. Any metric measured now is NOT comparable to the\n"
            "  committed figures: re-pin with build_manifest.py and re-measure, or\n"
            "  restore the pinned data. Do not publish or promote across a drift."
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", default="real", choices=["real", "reconstructed"])
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument(
        "--external-only",
        action="store_true",
        help="verify only the external-label events the fitness signal reads",
    )
    ap.add_argument("--json", type=Path, help="write the report as JSON")
    ap.add_argument("--check", action="store_true", help="exit non-zero on drift")
    args = ap.parse_args(argv)

    only = external_label_keys(args.manifest) if args.external_only else None
    report = verify_suite(args.kind, manifest_path=args.manifest, only=only)
    _print_report(report)
    if args.json:
        args.json.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0 if (report.ok or not args.check) else 1


if __name__ == "__main__":
    raise SystemExit(main())
