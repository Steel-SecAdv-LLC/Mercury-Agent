# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A forecast is written down before its day, and held to what happened after.

The ledger is what makes it one: it refuses a forecast of a day that has
begun, an outcome before the day has ended, and any change to either. These
tests issue and resolve forecasts over real days of the Kp record (the one
``research/forecast_kp`` fits the model on, and GFZ's own answer as it was
read on 2026-11-18), and then try to cheat the ledger every way they can.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from omni_mercury_engine.cli import main
from omni_mercury_engine.watch import (
    Watcher,
    WatchStore,
    forecast as forecast_module,
)
from omni_mercury_engine.watch.forecast import (
    FEATURES,
    TARGET,
    Forecaster,
    features,
    load_model,
    parse_gfz_json,
    storm_day,
)
from omni_mercury_engine.watch.ledger import (
    GENESIS,
    ForecastLedger,
    LedgerError,
    entry_hash,
    score,
    verify,
)
from research.forecast_kp.measure import read_record

DATA = Path(__file__).parent / "data"
#: The record the model was fitted on: every UT day's eight Kp values since 1932.
RECORD, _ = read_record()
#: A week of it that holds storm days and quiet ones.
START = date(2026, 3, 18)


def _at(day: date, hour: int = 0, minute: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=UTC)


def _published(now: datetime) -> dict[date, dict[str, list[Any]]]:
    """The record as GFZ would have published it by ``now``: no interval that has not ended."""
    days: dict[date, dict[str, list[Any]]] = {}
    for day, values in RECORD.items():
        if day > now.date():
            continue
        over = 8 if day < now.date() else now.hour // 3
        days[day] = {
            "kp": [value if n < over else None for n, value in enumerate(values)],
            "status": ["def" if n < over else "" for n in range(8)],
        }
    return days


@pytest.fixture(autouse=True)
def _forecasting_as_a_deployment_has_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """The suite turns forecasting off so that no other test reads the network; not here."""
    monkeypatch.delenv("MERCURY_WATCH_FORECASTS", raising=False)


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture()
def store(tmp_path: Path) -> WatchStore:
    return WatchStore(tmp_path / "watch")


@pytest.fixture()
def clock() -> Clock:
    return Clock(_at(START, 12, 30))


@pytest.fixture()
def forecaster(store: WatchStore, clock: Clock) -> Forecaster:
    return Forecaster(
        ForecastLedger(store), fetch=lambda first, last: _published(clock.now), clock=clock
    )


def _run_days(forecaster: Forecaster, clock: Clock, days: int) -> None:
    """Run the forecaster as the watch does, every ten minutes, for ``days`` days."""
    end = clock.now + timedelta(days=days)
    while clock.now < end:
        forecaster.run()
        clock.now += timedelta(minutes=10)


# --------------------------------------------------------------------------- #
# The ledger holds the watch to the order
# --------------------------------------------------------------------------- #


class TestTheLedger:
    WINDOW = (_at(START + timedelta(days=1)), _at(START + timedelta(days=2)))

    def _issue(
        self, ledger: ForecastLedger, now: datetime, name: str = "kp-g1:x"
    ) -> dict[str, Any]:
        return ledger.issue(
            name, target=TARGET, window=self.WINDOW, probability=0.25, now=now,
            references={"climatology": 0.2},
        )  # fmt: skip

    def test_a_forecast_of_a_day_that_has_begun_is_refused(self, store: WatchStore) -> None:
        ledger = ForecastLedger(store)

        for late in (self.WINDOW[0], self.WINDOW[0] + timedelta(seconds=1), self.WINDOW[1]):
            with pytest.raises(LedgerError, match="a forecast is made before"):
                self._issue(ledger, late)
        assert ledger.entries() == [] and ledger.head() == GENESIS

        issued = self._issue(ledger, self.WINDOW[0] - timedelta(seconds=1))
        assert issued["seq"] == 1 and issued["prev"] == GENESIS
        assert (
            ledger.head()
            == issued["hash"]
            == entry_hash(
                {key: value for key, value in issued.items() if key not in ("hash", "signature")}
            )
        )

    def test_it_is_issued_once_and_resolved_once_and_only_after_its_day(
        self, store: WatchStore
    ) -> None:
        ledger = ForecastLedger(store)
        self._issue(ledger, _at(START, 13))

        with pytest.raises(LedgerError, match="already issued"):
            self._issue(ledger, _at(START, 14))
        with pytest.raises(LedgerError, match="has not ended"):
            ledger.resolve("kp-g1:x", outcome=True, now=self.WINDOW[1] - timedelta(seconds=1))
        with pytest.raises(LedgerError, match="no such forecast"):
            ledger.resolve("kp-g1:y", outcome=True, now=self.WINDOW[1])
        ledger.resolve("kp-g1:x", outcome=True, now=self.WINDOW[1])
        with pytest.raises(LedgerError, match="already resolved"):
            ledger.resolve("kp-g1:x", outcome=False, now=self.WINDOW[1] + timedelta(hours=1))

        assert [entry["kind"] for entry in ledger.entries()] == ["issued", "resolved"]
        assert ledger.state()["kp-g1:x"]["outcome"] is True
        assert verify(ledger.entries()) == []

    @pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan"), 1, "0.5", None])
    def test_a_probability_is_a_number_from_nought_to_one(
        self, store: WatchStore, probability: Any
    ) -> None:
        with pytest.raises(LedgerError, match="is not one"):
            ForecastLedger(store).issue(
                "kp-g1:x", target=TARGET, window=self.WINDOW, probability=probability,
                now=_at(START),
            )  # fmt: skip

    def test_nothing_in_it_can_be_changed_or_removed(self, store: WatchStore) -> None:
        self._issue(ForecastLedger(store), _at(START, 13))

        for statement in (
            "UPDATE forecast_ledger SET body = replace(body, '0.25', '0.95')",
            "DELETE FROM forecast_ledger",
        ):
            with (
                sqlite3.connect(store.path) as connection,
                pytest.raises(sqlite3.DatabaseError, match="append-only"),
            ):
                connection.execute(statement)
        assert ForecastLedger(store).entries()[0]["probability"] == 0.25


class TestVerify:
    def _ledger(self, store: WatchStore) -> list[dict[str, Any]]:
        ledger = ForecastLedger(store)
        for n in range(3):
            day = START + timedelta(days=n + 1)
            ledger.issue(
                f"kp-g1:{day}", target=TARGET, window=(_at(day), _at(day + timedelta(days=1))),
                probability=0.1 * (n + 1), now=_at(day - timedelta(days=1), 13),
            )  # fmt: skip
        ledger.resolve(
            f"kp-g1:{START + timedelta(days=1)}",
            outcome=False,
            now=_at(START, 0) + timedelta(days=2, hours=3),
        )
        return ledger.entries()

    def test_a_ledger_as_written_holds(self, store: WatchStore) -> None:
        assert verify(self._ledger(store)) == []

    def test_a_changed_probability_is_seen(self, store: WatchStore) -> None:
        entries = self._ledger(store)
        entries[1]["probability"] = 0.9

        wrong = verify(entries)

        assert any("entry 2" in line and "do not match its hash" in line for line in wrong)
        assert any("entry 3" in line and "does not follow" in line for line in wrong)

    def test_a_dropped_entry_and_one_put_in_later_are_seen(self, store: WatchStore) -> None:
        entries = self._ledger(store)

        assert any("does not follow" in line for line in verify(entries[:1] + entries[2:]))
        assert any("numbered" in line for line in verify(entries[1:]))

    def test_a_forecast_written_in_after_its_day_is_seen_even_when_rehashed(
        self, store: WatchStore
    ) -> None:
        """Someone with the file writes a perfect forecast afterwards and redoes every hash."""
        forged, previous = [], GENESIS
        day = START + timedelta(days=1)
        for seq, (kind, at, more) in enumerate(
            (
                ("issued", _at(day, 9), {"target": TARGET, "window": [_at(day).isoformat(), _at(day + timedelta(days=1)).isoformat()], "probability": 1.0, "references": {}, "basis": {}}),
                ("resolved", _at(day + timedelta(days=1), 3), {"outcome": True, "observed": {}}),
            ),
            start=1,
        ):  # fmt: skip
            body = {"schema": "mercury.forecast-ledger/v1", "seq": seq, "prev": previous,
                    "forecast": f"kp-g1:{day}", "kind": kind, "at": at.isoformat(), **more}  # fmt: skip
            previous = entry_hash(body)
            forged.append({**body, "hash": previous, "signature": ""})

        assert verify(forged) == [
            "entry 1 (kp-g1:2026-03-19, issued): issued after its window began"
        ]

    def test_a_whole_chain_rewritten_is_caught_by_the_head_that_was_noted(
        self, store: WatchStore, tmp_path: Path
    ) -> None:
        """What the chain cannot show by itself, and what noting the head down is for."""
        entries = self._ledger(store)
        noted = entries[-1]["hash"]
        rewritten, previous = [], GENESIS
        for entry in entries:
            body = {k: v for k, v in entry.items() if k not in ("hash", "signature")}
            body["prev"] = previous
            if body["kind"] == "issued":
                body["probability"] = 0.01  # the one that resolved "no storm" now looks prescient
            previous = entry_hash(body)
            rewritten.append({**body, "hash": previous, "signature": ""})
        export = tmp_path / "ledger.jsonl"
        export.write_text("".join(json.dumps(entry) + "\n" for entry in rewritten))
        runner = CliRunner()

        assert verify(rewritten) == []  # it holds together: the chain alone cannot tell
        plain = runner.invoke(main, ["forecasts", "verify", "--file", str(export)])
        assert plain.exit_code == 0
        held = runner.invoke(
            main, ["forecasts", "verify", "--file", str(export), "--contains", noted]
        )
        assert held.exit_code == 1 and "is not in this ledger" in held.output

    def test_entries_are_signed_with_the_deployments_key_when_it_has_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("ama_cryptography")
        from omni_mercury_engine.security.signing_keys import generate_keyfile
        from omni_mercury_engine.watch.ledger import SIGNING_KEY_ENV

        public = generate_keyfile(tmp_path / "forecast.key")
        other = generate_keyfile(tmp_path / "other.key")
        monkeypatch.setenv(SIGNING_KEY_ENV, str(tmp_path / "forecast.key"))
        store = WatchStore(tmp_path / "watch")
        day = START + timedelta(days=1)
        ForecastLedger.from_env(store).issue(
            f"kp-g1:{day}", target=TARGET, window=(_at(day), _at(day + timedelta(days=1))),
            probability=0.3, now=_at(START, 13),
        )  # fmt: skip
        entries = ForecastLedger(store).entries()

        assert entries[0]["signer"] == public and entries[0]["signature"]
        assert verify(entries, public_key=public) == []
        assert "not signed with the key given" in verify(entries, public_key=other)[0]
        unsigned = [{**entries[0], "signature": ""}]
        assert "not signed with the key given" in verify(unsigned, public_key=public)[0]


# --------------------------------------------------------------------------- #
# The score
# --------------------------------------------------------------------------- #


class TestScore:
    def test_it_is_the_arithmetic_it_says_it_is(self, store: WatchStore) -> None:
        ledger = ForecastLedger(store)
        told = ((0.8, True), (0.2, False), (0.6, False))
        for n, (probability, _) in enumerate(told):
            day = START + timedelta(days=n + 1)
            ledger.issue(
                f"{TARGET}:{day}", target=TARGET, window=(_at(day), _at(day + timedelta(days=1))),
                probability=probability, references={"climatology": 0.2},
                now=_at(day - timedelta(days=1), 13),
            )  # fmt: skip
        for n, (_, outcome) in enumerate(told):
            day = START + timedelta(days=n + 1)
            ledger.resolve(f"{TARGET}:{day}", outcome=outcome, now=_at(day + timedelta(days=1), 3))

        account = score(ledger.entries(), TARGET)

        assert (account["issued"], account["resolved"], account["open"]) == (3, 3, 0)
        assert account["brier"] == pytest.approx((0.04 + 0.04 + 0.36) / 3)
        assert account["observed_rate"] == pytest.approx(1 / 3)
        climatology = account["references"]["climatology"]
        assert climatology["brier"] == pytest.approx((0.64 + 0.04 + 0.04) / 3)
        assert climatology["skill"] == pytest.approx(1 - (0.44 / 3) / (0.72 / 3))
        assert climatology["difference"] == pytest.approx((0.60 + 0.0 - 0.32) / 3)
        assert [(row["from"], row["forecasts"]) for row in account["reliability"]] == [
            (0.2, 1), (0.6, 1), (0.8, 1),
        ]  # fmt: skip

    def test_a_day_with_no_forecast_is_counted_not_left_out(self, store: WatchStore) -> None:
        ledger = ForecastLedger(store)
        for n in (1, 2, 5):
            day = START + timedelta(days=n)
            ledger.issue(
                f"{TARGET}:{day}", target=TARGET, window=(_at(day), _at(day + timedelta(days=1))),
                probability=0.2, now=_at(day - timedelta(days=1), 13),
            )  # fmt: skip

        account = score(ledger.entries(), TARGET, until=START + timedelta(days=6))

        assert account["missed"] == [
            str(START + timedelta(days=3)), str(START + timedelta(days=4)), str(START + timedelta(days=6)),
        ]  # fmt: skip
        assert account["brier"] is None and account["open"] == 3

    def test_an_empty_ledger_has_no_score_to_show(self) -> None:
        account = score([], TARGET)

        assert account["issued"] == 0 and account["brier"] is None and account["missed"] == []


# --------------------------------------------------------------------------- #
# The forecast itself
# --------------------------------------------------------------------------- #


class TestTheForecast:
    def test_it_reads_nothing_from_noon_the_day_before_onwards(self) -> None:
        target = START + timedelta(days=1)
        made_from = features(RECORD, target)
        assert made_from is not None and len(made_from) == len(FEATURES)

        later = {day: list(values) for day, values in RECORD.items() if day <= target}
        later[target] = [9.0] * 8
        later[START][4:] = [9.0] * 4  # the afternoon before
        assert features(later, target) == made_from

        earlier = {day: list(values) for day, values in later.items()}
        earlier[START][3] = 9.0  # the interval that ends at the cut-off
        changed = features(earlier, target)
        assert changed is not None and changed != made_from and changed[0] == 9.0

    def test_a_gap_in_the_record_is_no_forecast(self) -> None:
        target = START + timedelta(days=1)
        for missing in (START, target - timedelta(days=28), target - timedelta(days=10)):
            gapped = dict(RECORD)
            gapped[missing] = [None] * 8
            assert features(gapped, target) is None
        without = {
            day: values for day, values in RECORD.items() if day != target - timedelta(days=5)
        }
        assert features(without, target) is None

    def test_the_model_that_ships_is_the_one_measured(self) -> None:
        model = load_model()
        evidence = json.loads(
            (
                Path(forecast_module.__file__).parents[3]
                / "research"
                / "forecast_kp"
                / "evidence.json"
            ).read_text()
        )

        assert model.version == evidence["model"] == "kp-g1/2025"
        assert (
            0.0
            < model.persistence["after_quiet"]
            < model.climatology
            < model.persistence["after_storm"]
            < 1.0
        )
        quiet = features(RECORD, date(2026, 1, 5))
        assert quiet is not None
        assert model.references(quiet)["climatology"] == model.climatology
        with pytest.raises(ValueError, match="weighs 9 features"):
            model.probability(quiet[:-1])

    def test_gfz_answers_in_the_shape_this_reads_and_agrees_with_its_own_file(self) -> None:
        """The service's answer on 2026-11-18, against the file the record was packed from."""
        answer = json.loads((DATA / "gfz_kp_2026-10-18_2026-11-18.json").read_text())

        read = parse_gfz_json(answer)

        assert min(read) == date(2026, 10, 18) and max(read) == date(2026, 11, 18)
        assert read[date(2026, 10, 18)]["status"] == ["def"] * 8
        assert read[date(2026, 11, 18)]["status"][-1] == "pre"
        target = date(2026, 11, 16)
        from_service = features({day: got["kp"] for day, got in read.items()}, target)
        from_file = features(RECORD, target)
        assert from_service is not None and from_file is not None
        assert from_service == pytest.approx(from_file, abs=2e-3)
        assert storm_day(read[target]["kp"]) == storm_day(RECORD[target])

    @pytest.mark.parametrize(
        "broken",
        [
            {},
            {"datetime": ["2026-11-18T00:00:00Z"], "Kp": []},
            {"datetime": ["2026-11-18T01:00:00Z"], "Kp": [1.0]},
        ],
    )
    def test_an_answer_of_another_shape_is_refused(self, broken: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="GFZ"):
            parse_gfz_json(broken)

    def test_a_value_that_is_not_a_kp_is_a_gap(self) -> None:
        read = parse_gfz_json(
            {"datetime": ["2026-11-18T00:00:00Z", "2026-11-18T03:00:00Z", "2026-11-18T06:00:00Z"],
             "Kp": [-1.0, None, 2.333]}
        )  # fmt: skip

        assert read[date(2026, 11, 18)]["kp"][:3] == [None, None, 2.333]


class TestTheForecaster:
    def test_it_issues_before_each_day_and_resolves_after_with_what_happened(
        self, forecaster: Forecaster, clock: Clock
    ) -> None:
        _run_days(forecaster, clock, days=7)

        entries = forecaster.ledger.entries()
        state = forecaster.ledger.state(TARGET)
        assert verify(entries) == []
        assert len(state) == 7
        model = load_model()
        for name, entry in state.items():
            day = date.fromisoformat(name.partition(":")[2])
            issued = datetime.fromisoformat(entry["issued_at"])
            assert _at(day - timedelta(days=1), 12, 30) <= issued < _at(day)
            made_from = features(RECORD, day)
            assert made_from is not None
            assert entry["probability"] == pytest.approx(model.probability(made_from))
            assert entry["references"] == model.references(made_from)
            if entry["outcome"] is not None:
                assert entry["outcome"] is storm_day(RECORD[day])
                assert datetime.fromisoformat(entry["resolved_at"]) >= _at(
                    day + timedelta(days=1), 3
                )
        account = score(entries, TARGET, until=clock.now.date())
        assert account["resolved"] == 6 and account["open"] == 1 and account["missed"] == []
        basis = entries[0]["basis"]
        assert basis["model"] == model.version and sorted(basis["features"]) == sorted(FEATURES)
        assert basis["cutoff"] == _at(START, 12).isoformat()

    def test_a_day_it_could_not_forecast_in_time_stays_without_one(
        self, store: WatchStore, clock: Clock
    ) -> None:
        """The watch was down through a day's window: that day is missed, not made up."""
        forecaster = Forecaster(
            ForecastLedger(store), fetch=lambda first, last: _published(clock.now), clock=clock
        )
        _run_days(forecaster, clock, days=1)
        clock.now += timedelta(days=2)  # down for two days
        _run_days(forecaster, clock, days=2)

        account = score(forecaster.ledger.entries(), TARGET, until=clock.now.date())

        assert account["missed"] == [str(START + timedelta(days=n)) for n in (2, 3)]
        assert verify(forecaster.ledger.entries()) == []

    def test_before_half_past_twelve_there_is_nothing_to_issue_and_nothing_is_read(
        self, store: WatchStore
    ) -> None:
        clock = Clock(_at(START, 12, 20))
        forecaster = Forecaster(
            ForecastLedger(store), fetch=lambda first, last: pytest.fail("read"), clock=clock
        )

        assert forecaster.run() == {"issued": 0, "resolved": 0}

    def test_a_record_not_yet_complete_to_noon_is_waited_for(
        self, store: WatchStore, clock: Clock, caplog: pytest.LogCaptureFixture
    ) -> None:
        late = {"by": timedelta(hours=4)}  # GFZ is four hours behind
        forecaster = Forecaster(
            ForecastLedger(store),
            fetch=lambda first, last: _published(clock.now - late["by"]),
            clock=clock,
        )

        assert forecaster.run() == {"issued": 0, "resolved": 0}
        clock.now += timedelta(hours=4)
        assert forecaster.run() == {"issued": 1, "resolved": 0}

    def test_a_record_that_cannot_be_read_is_nothing_made_up_and_tried_again(
        self, store: WatchStore, clock: Clock
    ) -> None:
        answers: list[Any] = [OSError("no route"), None]

        def fetch(first: date, last: date) -> dict[date, dict[str, list[Any]]]:
            answer = answers.pop(0)
            if answer is not None:
                raise answer
            return _published(clock.now)

        forecaster = Forecaster(ForecastLedger(store), fetch=fetch, clock=clock)

        assert forecaster.run() == {"issued": 0, "resolved": 0}
        assert forecaster.run() == {"issued": 0, "resolved": 0}  # not again within ten minutes
        clock.now += timedelta(minutes=10)
        assert forecaster.run() == {"issued": 1, "resolved": 0}

    def test_two_watches_of_one_store_make_one_forecast(
        self, store: WatchStore, clock: Clock
    ) -> None:
        def fetch(first: date, last: date) -> dict[date, dict[str, list[Any]]]:
            return _published(clock.now)

        first = Forecaster(ForecastLedger(store), fetch=fetch, clock=clock)
        second = Forecaster(ForecastLedger(store), fetch=fetch, clock=clock)

        assert first.run("a") == {"issued": 1, "resolved": 0}
        assert second.run("b") == {"issued": 0, "resolved": 0}
        assert len(ForecastLedger(store).entries()) == 1

    def test_it_is_on_unless_switched_off(
        self, forecaster: Forecaster, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert Forecaster.enabled()
        monkeypatch.setenv("MERCURY_WATCH_FORECASTS", "false")

        assert not Forecaster.enabled()
        assert forecaster.run() == {"issued": 0, "resolved": 0}
        assert forecaster.ledger.entries() == []

    def test_a_forecast_gfz_never_publishes_the_day_of_is_left_open_after_sixty_days(
        self, store: WatchStore, clock: Clock
    ) -> None:
        reads: list[date] = []

        def fetch(first: date, last: date) -> dict[date, dict[str, list[Any]]]:
            reads.append(first)
            record = _published(clock.now)
            record.pop(START + timedelta(days=1), None)  # that day never appears
            return record

        forecaster = Forecaster(ForecastLedger(store), fetch=fetch, clock=clock)
        forecaster.run()
        clock.now += timedelta(days=62)
        clock.now = clock.now.replace(hour=6, minute=0)  # before any forecast can be issued
        reads.clear()

        assert forecaster.run() == {"issued": 0, "resolved": 0}
        assert reads == []  # no longer asked about
        assert score(forecaster.ledger.entries(), TARGET)["open"] == 1


# --------------------------------------------------------------------------- #
# The watch, the route, the commands
# --------------------------------------------------------------------------- #


class TestSurfaces:
    def test_the_watch_makes_the_forecast_on_its_own(
        self, store: WatchStore, clock: Clock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(forecast_module, "fetch_kp", lambda first, last: _published(clock.now))
        watcher = Watcher(store, feeds=[], clock=clock)

        watcher._pass(force=False)

        (entry,) = ForecastLedger(store).entries()
        assert entry["forecast"] == f"{TARGET}:{START + timedelta(days=1)}"

    def test_a_forecast_pass_that_fails_does_not_stop_the_watch(
        self, store: WatchStore, clock: Clock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MERCURY_FORECAST_SIGNING_KEY", "/nonexistent/key")
        watcher = Watcher(store, feeds=[], clock=clock)

        assert watcher._pass(force=False) == []

    def test_a_forecast_key_that_cannot_be_read_is_said_once_not_every_pass(
        self,
        store: WatchStore,
        clock: Clock,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Was: the same error logged on every 30-second tick, for as long as the key stayed wrong."""
        monkeypatch.setenv("MERCURY_FORECAST_SIGNING_KEY", "/nonexistent/key")
        watcher = Watcher(store, feeds=[], clock=clock)

        with caplog.at_level("WARNING"):
            for _ in range(5):
                clock.now += timedelta(seconds=30)
                watcher._pass(force=False)

        said = [r for r in caplog.records if "forecast" in r.getMessage()]
        assert len(said) == 1, [r.getMessage() for r in said]
        assert watcher._forecast_unavailable is True

    def test_the_route_serves_the_head_the_score_and_the_forecasts(
        self, forecaster: Forecaster, clock: Clock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("fastapi")
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from omni_mercury_engine.api.routes import watch as watch_routes

        monkeypatch.setenv("MERCURY_WATCH_DIR", str(forecaster.ledger.store.directory))
        _run_days(forecaster, clock, days=4)
        app = FastAPI()
        app.include_router(watch_routes.router)

        shown = TestClient(app).get("/api/v1/watch/forecasts?limit=2").json()

        assert shown["head"] == forecaster.ledger.head() != GENESIS
        assert shown["model"] == {"version": "kp-g1/2025", "trained_through": "2025-12-31"}
        assert len(shown["forecasts"]) == 2
        assert shown["forecasts"][0]["forecast"] > shown["forecasts"][1]["forecast"]  # newest first
        assert shown["score"]["resolved"] == 3 and shown["score"]["brier"] is not None
        assert set(shown["score"]["references"]) == {"climatology", "persistence"}

    def test_with_no_store_the_route_says_what_it_would_forecast(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("fastapi")
        from omni_mercury_engine.api.routes.watch import forecasts

        monkeypatch.setenv("MERCURY_WATCH_DIR", str(tmp_path / "none"))

        shown = forecasts()

        assert shown["head"] == GENESIS and shown["forecasts"] == []
        assert shown["score"]["issued"] == 0 and "G1" in shown["what"]

    def test_the_commands_show_export_and_verify(
        self, forecaster: Forecaster, clock: Clock, tmp_path: Path
    ) -> None:
        _run_days(forecaster, clock, days=4)
        state = ["--state-dir", str(forecaster.ledger.store.directory)]
        runner = CliRunner()

        shown = runner.invoke(main, ["forecasts", "show", *state])
        assert shown.exit_code == 0, shown.output
        # The days since are counted as missed: these forecasts are of March.
        assert (
            "3 resolved of 4 issued; " in shown.output and "day(s) with no forecast" in shown.output
        )
        assert "against persistence" in shown.output and "says little yet" in shown.output
        assert forecaster.ledger.head() in shown.output

        export = tmp_path / "ledger.jsonl"
        assert (
            runner.invoke(main, ["forecasts", "export", "--output", str(export), *state]).exit_code
            == 0
        )
        held = runner.invoke(main, ["forecasts", "verify", "--file", str(export)])
        assert held.exit_code == 0 and "7 entries hold" in held.output

        lines = export.read_text().splitlines()
        export.write_text(
            "\n".join([lines[0].replace('"probability": 0.', '"probability": 0.9'), *lines[1:]])
        )
        broken = runner.invoke(main, ["forecasts", "verify", "--file", str(export)])
        assert broken.exit_code == 1 and "do not match its hash" in broken.output

        as_json = runner.invoke(main, ["forecasts", "show", "--json", *state])
        assert json.loads(as_json.output)["score"]["resolved"] == 3

    def test_an_empty_ledger_says_when_the_first_forecast_comes(self, tmp_path: Path) -> None:
        shown = CliRunner().invoke(main, ["forecasts", "show", "--state-dir", str(tmp_path / "w")])

        assert shown.exit_code == 0 and "No forecast has been issued" in shown.output
