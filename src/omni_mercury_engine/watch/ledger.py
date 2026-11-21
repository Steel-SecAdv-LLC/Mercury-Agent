# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The forecast ledger: what was forecast, written down before; what happened, after.

A forecast that is recorded once its outcome is known is not a forecast. The
ledger is where the watch commits to one beforehand, and it holds the watch
to the order:

* A forecast is **issued** with the window it is about. The ledger refuses
  one whose window has begun, and a second for the same thing.
* It is **resolved** with what happened. The ledger refuses that before the
  window has ended, for a forecast it does not hold, and twice.
* Nothing is changed or removed. The table refuses UPDATE and DELETE, and
  each entry carries the hash of the one before it, so an entry altered,
  dropped or put in later breaks every hash after it (:func:`verify`).

What the chain proves is the order of the entries and that none was changed
since. It does not by itself prove *when* an entry was written: whoever
holds the file could write a whole new chain. That is what the head is for.
The head (the last entry's hash) is served at ``GET /api/v1/watch/forecasts``
and printed by ``mercury-agent forecasts verify``; anyone who notes it down
before a day can check afterwards that the ledger they are shown still
contains that head with the forecast before it. With
``MERCURY_FORECAST_SIGNING_KEY`` set (a key file from
:mod:`~omni_mercury_engine.security.signing_keys`), each entry names the
public key and is signed with ML-DSA-65 as well, which says who wrote it.

:func:`score` is the account: over every forecast that has resolved, the
Brier score and the log score, the same for each reference forecast that was
written down with it, the skill against each, how often what was forecast at
each probability happened, and how many days were not forecast at all.
"""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import closing
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from omni_mercury_engine.watch.store import WatchStore

__all__ = [
    "GENESIS",
    "SIGNING_KEY_ENV",
    "ForecastLedger",
    "LedgerError",
    "entry_hash",
    "score",
    "verify",
]

#: The "previous hash" of the first entry.
GENESIS = "0" * 64
#: Names the ML-DSA-65 key file entries are signed with, when set.
SIGNING_KEY_ENV = "MERCURY_FORECAST_SIGNING_KEY"
_SCHEMA = "mercury.forecast-ledger/v1"
#: A probability is scored no nearer to 0 or 1 than this in the log score,
#: where a forecast of exactly 0 for something that happened would be infinite.
_LOG_FLOOR = 1e-6


class LedgerError(ValueError):
    """The ledger refused an entry: it would not be a forecast made beforehand."""


def _canonical(body: Mapping[str, Any]) -> str:
    return json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)


def entry_hash(body: Mapping[str, Any]) -> str:
    """The hash of an entry: SHA-256 of its canonical JSON, which names the entry before it."""
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


class ForecastLedger:
    """The append-only record of forecasts and their outcomes, in the watch's store."""

    def __init__(self, store: WatchStore, *, signer: tuple[bytes, bytes] | None = None) -> None:
        """Open the ledger in ``store``.

        Args:
            store: The watch's store; the ledger is a table in it.
            signer: An ML-DSA-65 ``(public key, secret key)`` to sign each
                new entry with. Reading needs none.
        """
        self.store = store
        self._signer = signer

    @classmethod
    def from_env(cls, store: WatchStore) -> ForecastLedger:
        """The ledger as this deployment writes it: signed if :data:`SIGNING_KEY_ENV` names a key.

        Raises:
            omni_mercury_engine.security.signing_keys.SigningKeyError: If the
                key file named is missing, readable by others or mismatched.
        """
        from omni_mercury_engine.security.signing_keys import keypair_from_env

        return cls(store, signer=keypair_from_env(SIGNING_KEY_ENV))

    # ------------------------------------------------------------------ writing

    def issue(
        self,
        forecast: str,
        *,
        target: str,
        window: tuple[datetime, datetime],
        probability: float,
        now: datetime,
        references: Mapping[str, float] | None = None,
        basis: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Commit to a forecast.

        Args:
            forecast: Its name, unique in the ledger.
            target: What kind of thing is forecast.
            window: When it is about: from, and until.
            probability: The probability that it happens in the window.
            now: When it is issued.
            references: Other forecasts of the same thing, fixed now, that
                this one is to be scored against.
            basis: What it was made from: enough for someone else to make it again.

        Returns:
            The entry as recorded.

        Raises:
            LedgerError: If the window has begun, the forecast is already
                held, or a probability is not one.
        """
        start, end = window
        if not start < end:
            raise LedgerError("a forecast's window ends after it begins")
        if now >= start:
            raise LedgerError(
                f"{forecast}: its window began at {start.isoformat()}; a forecast is made before"
            )
        given = {"forecast": probability, **(references or {})}
        for name, value in given.items():
            if not (isinstance(value, float) and 0.0 <= value <= 1.0):
                raise LedgerError(f"{forecast}: the {name} probability {value!r} is not one")
        return self._append(
            forecast,
            "issued",
            {
                "target": target,
                "window": [start.isoformat(), end.isoformat()],
                "probability": probability,
                "references": dict(references or {}),
                "basis": dict(basis or {}),
            },
            now,
        )

    def resolve(
        self,
        forecast: str,
        *,
        outcome: bool,
        now: datetime,
        observed: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record what happened in a forecast's window.

        Raises:
            LedgerError: If the ledger holds no such forecast, its window has
                not ended, or it is already resolved.
        """
        issued = self.state().get(forecast)
        if issued is None:
            raise LedgerError(f"{forecast}: no such forecast was issued")
        if now < datetime.fromisoformat(issued["window"][1]):
            raise LedgerError(f"{forecast}: its window has not ended")
        return self._append(
            forecast, "resolved", {"outcome": bool(outcome), "observed": dict(observed or {})}, now
        )

    def _append(
        self, forecast: str, kind: str, content: dict[str, Any], now: datetime
    ) -> dict[str, Any]:
        import sqlite3

        with (
            self.store._lock,
            closing(self.store._connect()) as connection,
            connection,
        ):
            # One writer at a time, across processes: the hash of the last
            # entry is read and the next one written in the same transaction.
            connection.execute("BEGIN IMMEDIATE")
            last = connection.execute(
                "SELECT seq, hash FROM forecast_ledger ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            body = {
                "schema": _SCHEMA,
                "seq": (int(last["seq"]) if last else 0) + 1,
                "prev": str(last["hash"]) if last else GENESIS,
                "forecast": forecast,
                "kind": kind,
                "at": now.isoformat(),
                **content,
            }
            if self._signer is not None:
                body["signer"] = self._signer[0].hex()  # inside what is hashed and signed
            digest = entry_hash(body)
            signature = ""
            if self._signer is not None:
                from omni_mercury_engine.security.crypto_api import MLDSAProvider

                signature = MLDSAProvider().sign(bytes.fromhex(digest), self._signer[1]).hex()
            try:
                connection.execute(
                    "INSERT INTO forecast_ledger (seq, forecast, kind, body, hash, signature) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (body["seq"], forecast, kind, _canonical(body), digest, signature),
                )
            except sqlite3.IntegrityError as exc:
                raise LedgerError(f"{forecast}: already {kind}") from exc
        return {**body, "hash": digest, "signature": signature}

    # ------------------------------------------------------------------ reading

    def entries(self) -> list[dict[str, Any]]:
        """Every entry, oldest first, each with its ``hash`` and ``signature``."""
        with closing(self.store._connect()) as connection:
            rows = connection.execute(
                "SELECT body, hash, signature FROM forecast_ledger ORDER BY seq"
            ).fetchall()
        return [
            {**json.loads(row["body"]), "hash": row["hash"], "signature": row["signature"]}
            for row in rows
        ]

    def head(self) -> str:
        """The last entry's hash: what to note down to hold the ledger to later."""
        with closing(self.store._connect()) as connection:
            row = connection.execute(
                "SELECT hash FROM forecast_ledger ORDER BY seq DESC LIMIT 1"
            ).fetchone()
        return str(row["hash"]) if row else GENESIS

    def state(self, target: str | None = None) -> dict[str, dict[str, Any]]:
        """Each forecast as it stands: what was issued, and its outcome or ``None``."""
        return _state(self.entries(), target)


def _state(
    entries: Iterable[Mapping[str, Any]], target: str | None = None
) -> dict[str, dict[str, Any]]:
    held: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if entry["kind"] == "issued":
            if target is None or entry["target"] == target:
                held[entry["forecast"]] = {
                    "target": entry["target"],
                    "window": entry["window"],
                    "probability": entry["probability"],
                    "references": entry["references"],
                    "issued_at": entry["at"],
                    "outcome": None,
                    "resolved_at": None,
                }
        elif entry["forecast"] in held:
            held[entry["forecast"]].update(outcome=entry["outcome"], resolved_at=entry["at"])
    return held


def verify(entries: list[dict[str, Any]], *, public_key: str = "") -> list[str]:
    """Everything wrong with a ledger as given. An empty list is a ledger that holds.

    Checks that each entry hashes to its ``hash`` and names the one before
    it, that entries are numbered in order from 1, that every forecast was
    issued before its window began and at most once, and that every outcome
    is of a forecast issued earlier in the ledger, recorded once, after its
    window ended. With ``public_key``, that every entry is signed by it.

    Args:
        entries: The ledger, oldest first (:meth:`ForecastLedger.entries`,
            or the lines of an export).
        public_key: Hex ML-DSA-65 public key every entry must be signed with.
    """
    wrong: list[str] = []
    previous = GENESIS
    windows: dict[str, list[str]] = {}
    resolved: set[str] = set()
    provider = None
    if public_key:
        from omni_mercury_engine.security.crypto_api import MLDSAProvider

        provider = MLDSAProvider()
    for position, entry in enumerate(entries, start=1):
        body = {key: value for key, value in entry.items() if key not in ("hash", "signature")}
        name = f"entry {position} ({body.get('forecast')}, {body.get('kind')})"
        if body.get("seq") != position:
            wrong.append(f"{name}: numbered {body.get('seq')}, where {position} belongs")
        if body.get("prev") != previous:
            wrong.append(f"{name}: does not follow the entry before it")
        digest = entry_hash(body)
        if entry.get("hash") != digest:
            wrong.append(f"{name}: its contents do not match its hash")
        previous = digest
        if provider is not None:
            try:
                signed = provider.verify(
                    bytes.fromhex(digest),
                    bytes.fromhex(str(entry.get("signature") or "")),
                    bytes.fromhex(public_key),
                )
            except Exception:
                signed = False
            if not signed or body.get("signer") != public_key.lower():
                wrong.append(f"{name}: is not signed with the key given")
        try:
            at = datetime.fromisoformat(str(body["at"]))
            forecast = str(body["forecast"])
            if body["kind"] == "issued":
                if forecast in windows:
                    wrong.append(f"{name}: issued a second time")
                elif at >= datetime.fromisoformat(body["window"][0]):
                    wrong.append(f"{name}: issued after its window began")
                windows.setdefault(forecast, body["window"])
            elif body["kind"] == "resolved":
                if forecast not in windows:
                    wrong.append(f"{name}: resolves a forecast that was not issued before it")
                elif forecast in resolved:
                    wrong.append(f"{name}: resolved a second time")
                elif at < datetime.fromisoformat(windows[forecast][1]):
                    wrong.append(f"{name}: resolved before its window ended")
                resolved.add(forecast)
            else:
                wrong.append(f"{name}: is of no kind the ledger holds")
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            wrong.append(f"{name}: is malformed ({type(exc).__name__})")
    return wrong


def _brier(pairs: list[tuple[float, bool]]) -> float:
    return sum((p - float(o)) ** 2 for p, o in pairs) / len(pairs)


def _log_score(pairs: list[tuple[float, bool]]) -> float:
    total = 0.0
    for p, o in pairs:
        clipped = min(max(p, _LOG_FLOOR), 1.0 - _LOG_FLOOR)
        total -= math.log(clipped if o else 1.0 - clipped)
    return total / len(pairs)


def score(
    entries: list[dict[str, Any]], target: str, *, until: date | None = None
) -> dict[str, Any]:
    """The account of the forecasts of ``target``: how they did, against what, and what is missing.

    Lower is better for the Brier score (the mean squared difference between
    the probability and what happened, 0 to 1) and for the log score. Skill
    against a reference is ``1 - brier / reference brier``: above 0 the
    forecast did better than the reference, below 0 worse. ``difference``
    is the mean of ``reference error - forecast error`` over the resolved
    forecasts and ``standard_error`` its standard error taken as if the days
    were independent, which neighbouring days are not: it is a floor on the
    uncertainty, not the uncertainty.

    Args:
        entries: The ledger, oldest first.
        target: Which forecasts.
        until: The last day whose issue window has closed, to count days
            with no forecast up to; without it they are counted to the last
            forecast issued.

    Returns:
        ``issued``, ``resolved``, ``open``, ``missed`` (days between the
        first forecast and ``until`` that have none), ``observed_rate``,
        ``brier``, ``log_score``, ``references`` (each with its ``brier``,
        ``log_score``, ``skill``, ``difference`` and ``standard_error``)
        and ``reliability``. The scores are ``None`` until a forecast has
        resolved.
    """
    held = _state(entries, target)
    done = [entry for entry in held.values() if entry["outcome"] is not None]
    days = sorted(date.fromisoformat(entry["window"][0][:10]) for entry in held.values())
    missed: list[str] = []
    if days:
        have, day, last = set(days), days[0], max(days[-1], until or days[-1])
        while day <= last:
            if day not in have:
                missed.append(day.isoformat())
            day += timedelta(days=1)
    account: dict[str, Any] = {
        "target": target,
        "issued": len(held),
        "resolved": len(done),
        "open": len(held) - len(done),
        "missed": missed,
        "observed_rate": None,
        "brier": None,
        "log_score": None,
        "references": {},
        "reliability": [],
    }
    if not done:
        return account
    pairs = [(float(entry["probability"]), bool(entry["outcome"])) for entry in done]
    account.update(
        observed_rate=sum(o for _, o in pairs) / len(pairs),
        brier=_brier(pairs),
        log_score=_log_score(pairs),
    )
    for name in sorted({name for entry in done for name in entry["references"]}):
        both = [entry for entry in done if name in entry["references"]]
        ours = [(float(entry["probability"]), bool(entry["outcome"])) for entry in both]
        theirs = [(float(entry["references"][name]), bool(entry["outcome"])) for entry in both]
        gains = [
            (r - float(o)) ** 2 - (p - float(o)) ** 2
            for (p, o), (r, _) in zip(ours, theirs, strict=True)
        ]
        mean = sum(gains) / len(gains)
        spread = (
            math.sqrt(sum((gain - mean) ** 2 for gain in gains) / (len(gains) - 1) / len(gains))
            if len(gains) > 1
            else None
        )
        reference = _brier(theirs)
        account["references"][name] = {
            "forecasts": len(both),
            "brier": reference,
            "log_score": _log_score(theirs),
            "skill": 1.0 - _brier(ours) / reference if reference > 0.0 else None,
            "difference": mean,
            "standard_error": spread,
        }
    for tenth in range(10):
        low, high = tenth / 10.0, (tenth + 1) / 10.0
        inside = [(p, o) for p, o in pairs if low <= p < high or (tenth == 9 and p == 1.0)]
        if inside:
            account["reliability"].append(
                {
                    "from": low,
                    "to": high,
                    "forecasts": len(inside),
                    "mean_probability": sum(p for p, _ in inside) / len(inside),
                    "observed_rate": sum(o for _, o in inside) / len(inside),
                }
            )
    return account
