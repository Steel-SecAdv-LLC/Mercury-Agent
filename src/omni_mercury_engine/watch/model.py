# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What the watch produces: findings and feed statuses."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime

__all__ = ["FeedStatus", "Finding", "Outcome", "Severity"]


class Severity(IntEnum):
    """How much a finding asks of the reader."""

    NOTABLE = 1  #: worth knowing
    SIGNIFICANT = 2  #: worth acting on if it concerns you
    SEVERE = 3  #: rare and consequential

    @property
    def label(self) -> str:
        """Lower-case name, as shown and stored."""
        return self.name.lower()


@dataclass(frozen=True)
class Finding:
    """One thing a feed showed that is worth a person's attention.

    Attributes:
        feed: Name of the feed that produced it.
        kind: What sort of thing it is (``"earthquake"``, ``"geomagnetic_storm"``).
        subject: What it is about: an event id, a station, a gauge.
        severity: How much it asks of the reader.
        headline: One line, plain language.
        detail: The account: what was measured, against what, and its limits.
        observed_at: When the reading that produced it was taken (UTC).
        source: Who measured it, as a person would cite it.
        basis: The rule that made it a finding.
        value: The measured quantity, when there is one.
        unit: Unit of ``value``.
        latitude: Where, when the feed says.
        longitude: Where, when the feed says.
        ongoing: Whether the condition can persist across pulls (a storm) or
            is a single occurrence (an earthquake).
        reference: The source's own identifier for the record behind the
            finding (an observatory notice), when it has one.
        action: What a person there should do, in the words of the agency
            responsible for that hazard.
        action_source: Whose guidance ``action`` is.
        area: Where it applies, as the source describes it, when it gives
            no coordinates (the counties of a warning).
        place_ids: Subscribers' places the source itself says it applies
            to (a warning in effect at that point).
    """

    feed: str
    kind: str
    subject: str
    severity: Severity
    headline: str
    detail: str
    observed_at: datetime
    source: str
    basis: str
    value: float | None = None
    unit: str = ""
    latitude: float | None = None
    longitude: float | None = None
    ongoing: bool = False
    reference: str = ""
    action: str = ""
    action_source: str = ""
    area: str = ""
    place_ids: tuple[int, ...] = ()

    @property
    def key(self) -> str:
        """Identity of the thing found: the same key is the same finding."""
        raw = f"{self.feed}|{self.kind}|{self.subject}"
        return hashlib.sha256(raw.encode()).hexdigest()[:20]

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON types."""
        data = asdict(self)
        data["severity"] = self.severity.label
        data["observed_at"] = self.observed_at.isoformat()
        data["key"] = self.key
        return data


@dataclass(frozen=True)
class Outcome:
    """What a later, authoritative record says about a finding.

    Attributes:
        verdict: ``"confirmed"`` (the record agrees), ``"revised"`` (it
            agrees something happened and changes the particulars),
            ``"withdrawn"`` (the source no longer lists it) or
            ``"unconfirmed"`` (the record that would confirm it does not, so
            far).
        note: One sentence saying what the record shows.
        record: Which record was consulted, as a person would cite it.
        final: Whether the record is settled, so the finding need not be
            checked again.
    """

    verdict: str
    note: str
    record: str
    final: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON types."""
        return asdict(self)


@dataclass
class FeedStatus:
    """What happened when a feed was last read.

    Attributes:
        feed: Feed name.
        title: What the feed is, for a person.
        state: ``"ok"``, ``"stale"`` (answered, but its newest reading is
            old), ``"down"`` (did not answer) or ``"waiting"`` (not read yet).
        checked_at: When it was read (UTC).
        latest_reading: Time of the newest reading it returned (UTC).
        readings: Rows it returned.
        subjects: Distinct stations, gauges or events among them.
        new_findings: Findings first reported by this read.
        open_findings: Findings still standing after it.
        summary: One line saying what was seen, including "nothing notable".
        error: Why it is down, when it is.
    """

    feed: str
    title: str
    state: str = "waiting"
    checked_at: datetime | None = None
    latest_reading: datetime | None = None
    readings: int = 0
    subjects: int = 0
    new_findings: int = 0
    open_findings: int = 0
    summary: str = "not read yet"
    error: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON types."""
        data = asdict(self)
        for name in ("checked_at", "latest_reading"):
            value = data[name]
            data[name] = value.isoformat() if value is not None else None
        return data
