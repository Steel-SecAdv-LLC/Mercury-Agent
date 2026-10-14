# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for scripts/gcn_kafka_keepalive.py (GCN credential keep-alive probe).

Covers the three things the script has to get right: credentials are resolved
without ever landing in a repository file or on stdout, a failed broker probe is
classified honestly (a rejected credential is FAIL, an outage is UNREACH), and
the state ledger records enough to answer "when did this last connect?" across
runs. No test touches the network -- both probes are injected.
"""

from __future__ import annotations

import importlib.util
import json
import stat
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_MOD = REPO_ROOT / "scripts" / "gcn_kafka_keepalive.py"
_spec = importlib.util.spec_from_file_location("gcn_kafka_keepalive", _MOD)
assert _spec is not None and _spec.loader is not None
keepalive = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = keepalive
_spec.loader.exec_module(keepalive)

#: Synthetic stand-ins. Deliberately low-entropy and self-describing so no
#: secret scanner has to decide whether this file leaks a real credential.
_CLIENT_ID = "gcn-fixture-client-id"
_STANDIN = "gcn-fixture-value-not-a-credential"


def _creds(**overrides: str) -> Any:
    fields = {
        "client_id": _CLIENT_ID,
        "client_secret": _STANDIN,
        "domain": "gcn.nasa.gov",
        "bootstrap_servers": "kafka.gcn.nasa.gov:9092",
    }
    fields.update(overrides)
    return keepalive.Credentials(**fields)


class _FakeMetadata:
    def __init__(self, topics: dict[str, object]) -> None:
        self.topics = topics


class _FakeConsumer:
    """Stands in for confluent_kafka.Consumer; records whether it was closed."""

    def __init__(self, config: dict[str, Any], *, error: Exception | None = None) -> None:
        self.config = config
        self.error = error
        self.closed = False

    def list_topics(self, timeout: float = 0.0) -> _FakeMetadata:
        if self.error is not None:
            raise self.error
        return _FakeMetadata({"gcn.notices.icecube.lvk_nu_track_search": object()})

    def close(self) -> None:
        self.closed = True


def _factory(error: Exception | None = None) -> Any:
    built: list[_FakeConsumer] = []

    def build(config: dict[str, Any]) -> _FakeConsumer:
        consumer = _FakeConsumer(config, error=error)
        built.append(consumer)
        return consumer

    build.built = built  # type: ignore[attr-defined]
    return build


# ---------------------------------------------------------------------------
# Credential resolution
# ---------------------------------------------------------------------------


def test_env_file_must_be_owner_only(tmp_path: Path) -> None:
    """A world-readable credentials file is a configuration error, not a warning."""
    path = tmp_path / "gcn-kafka.env"
    path.write_text(f"{keepalive._ENV_CLIENT_ID}={_CLIENT_ID}\n", encoding="utf-8")
    path.chmod(0o644)
    with pytest.raises(keepalive.ConfigError, match="chmod 600"):
        keepalive.parse_env_file(path)


def test_env_file_parses_quotes_exports_and_comments(tmp_path: Path) -> None:
    path = tmp_path / "gcn-kafka.env"
    path.write_text(
        "# comment\n"
        "\n"
        f'export {keepalive._ENV_CLIENT_ID}="{_CLIENT_ID}"\n'
        f"{keepalive._ENV_CLIENT_SECRET}='{_STANDIN}'\n"
        "NOT_AN_ASSIGNMENT\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    values = keepalive.parse_env_file(path)
    assert values == {keepalive._ENV_CLIENT_ID: _CLIENT_ID, keepalive._ENV_CLIENT_SECRET: _STANDIN}


def test_missing_file_is_empty_not_an_error(tmp_path: Path) -> None:
    assert keepalive.parse_env_file(tmp_path / "absent.env") == {}


def test_environment_wins_over_credentials_file(tmp_path: Path) -> None:
    """A CI-injected secret must not be shadowed by a stale file on the runner."""
    path = tmp_path / "gcn-kafka.env"
    path.write_text(
        f"{keepalive._ENV_CLIENT_ID}=file-id\n{keepalive._ENV_CLIENT_SECRET}=file-secret\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    creds = keepalive.load_credentials(
        credentials_file=path,
        environ={keepalive._ENV_CLIENT_ID: _CLIENT_ID, keepalive._ENV_CLIENT_SECRET: _STANDIN},
    )
    assert (creds.client_id, creds.client_secret) == (_CLIENT_ID, _STANDIN)


def test_credentials_file_fills_gaps(tmp_path: Path) -> None:
    path = tmp_path / "gcn-kafka.env"
    path.write_text(
        f"{keepalive._ENV_CLIENT_ID}={_CLIENT_ID}\n{keepalive._ENV_CLIENT_SECRET}={_STANDIN}\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    creds = keepalive.load_credentials(credentials_file=path, environ={})
    assert creds.client_id == _CLIENT_ID
    assert creds.bootstrap_servers == "kafka.gcn.nasa.gov:9092"


def test_missing_credentials_names_both_variables(tmp_path: Path) -> None:
    with pytest.raises(keepalive.ConfigError) as excinfo:
        keepalive.load_credentials(credentials_file=tmp_path / "absent.env", environ={})
    message = str(excinfo.value)
    assert keepalive._ENV_CLIENT_ID in message
    assert keepalive._ENV_CLIENT_SECRET in message
    assert "gcn.nasa.gov/quickstart" in message


def test_unknown_domain_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(keepalive.ConfigError, match="unknown GCN domain"):
        keepalive.load_credentials(
            credentials_file=tmp_path / "absent.env",
            domain="kafka.example.com",
            environ={keepalive._ENV_CLIENT_ID: _CLIENT_ID, keepalive._ENV_CLIENT_SECRET: _STANDIN},
        )


@pytest.mark.parametrize(
    ("domain", "broker"),
    [
        ("gcn.nasa.gov", "kafka.gcn.nasa.gov:9092"),
        ("test.gcn.nasa.gov", "kafka.test.gcn.nasa.gov:9092"),
        ("dev.gcn.nasa.gov", "kafka.dev.gcn.nasa.gov:9092"),
    ],
)
def test_broker_and_token_endpoint_derive_from_domain(
    tmp_path: Path, domain: str, broker: str
) -> None:
    creds = keepalive.load_credentials(
        credentials_file=tmp_path / "absent.env",
        domain=domain,
        environ={keepalive._ENV_CLIENT_ID: _CLIENT_ID, keepalive._ENV_CLIENT_SECRET: _STANDIN},
    )
    assert creds.bootstrap_servers == broker
    assert creds.token_endpoint == f"https://auth.{domain}/oauth2/token"


def test_fingerprint_is_stable_and_not_the_client_id() -> None:
    fingerprint = _creds().fingerprint
    assert fingerprint == _creds().fingerprint
    assert len(fingerprint) == 12
    assert _CLIENT_ID not in fingerprint


def test_redact_strips_secret_and_client_id() -> None:
    text = _creds().redact(f"failed for {_CLIENT_ID} using {_STANDIN}")
    assert _STANDIN not in text
    assert _CLIENT_ID not in text


def test_consumer_config_is_gcn_oauthbearer() -> None:
    config = keepalive.consumer_config(_creds())
    assert config["security.protocol"] == "SASL_SSL"
    assert config["sasl.mechanisms"] == "OAUTHBEARER"
    assert config["sasl.oauthbearer.method"] == "oidc"
    assert config["sasl.oauthbearer.token.endpoint.url"] == "https://auth.gcn.nasa.gov/oauth2/token"
    assert config["bootstrap.servers"] == "kafka.gcn.nasa.gov:9092"


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def test_missing_client_library_is_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """No confluent-kafka means no probe -- reported, never silently skipped."""
    monkeypatch.setitem(sys.modules, "confluent_kafka", None)
    with pytest.raises(keepalive.ConfigError, match="pip install gcn-kafka"):
        keepalive._default_consumer_factory({})


def test_probe_broker_success_closes_the_consumer() -> None:
    factory = _factory()
    ok, detail, topics = keepalive.probe_broker(_creds(), factory=factory)
    assert ok is True
    assert topics == 1
    assert "1 topic" in detail
    assert factory.built[0].closed is True


def test_probe_broker_failure_redacts_and_still_closes() -> None:
    factory = _factory(error=RuntimeError(f"SASL handshake failed for {_STANDIN}"))
    ok, detail, topics = keepalive.probe_broker(_creds(), factory=factory)
    assert ok is False
    assert topics == 0
    assert _STANDIN not in detail
    assert factory.built[0].closed is True


def test_alive_verdict_records_topic_count() -> None:
    result = keepalive.probe(
        _creds(), factory=_factory(), token_probe=lambda creds: (True, "unused")
    )
    assert result.verdict == keepalive.ALIVE
    assert result.topic_count == 1
    assert result.broker == "kafka.gcn.nasa.gov:9092"


def test_rejected_credential_is_fail() -> None:
    result = keepalive.probe(
        _creds(),
        factory=_factory(error=RuntimeError("Local: Authentication failure")),
        token_probe=lambda creds: (False, "token endpoint rejected the credential (HTTP 401)"),
    )
    assert result.verdict == keepalive.FAIL


def test_valid_credential_with_dead_broker_is_unreach() -> None:
    """A token exchange proves the credential, not that GCN saw a broker connection."""
    result = keepalive.probe(
        _creds(),
        factory=_factory(error=RuntimeError("Local: Broker transport failure")),
        token_probe=lambda creds: (True, "token endpoint issued an access token"),
    )
    assert result.verdict == keepalive.UNREACH


def test_inconclusive_token_probe_falls_back_to_the_broker_error_text() -> None:
    auth_shaped = keepalive.probe(
        _creds(),
        factory=_factory(error=RuntimeError("SASL OAUTHBEARER: unauthorized")),
        token_probe=lambda creds: (None, "token endpoint unreachable"),
    )
    transport_shaped = keepalive.probe(
        _creds(),
        factory=_factory(error=RuntimeError("Local: Broker transport failure")),
        token_probe=lambda creds: (None, "token endpoint unreachable"),
    )
    assert auth_shaped.verdict == keepalive.FAIL
    assert transport_shaped.verdict == keepalive.UNREACH


# ---------------------------------------------------------------------------
# State ledger
# ---------------------------------------------------------------------------


def test_state_roundtrip_is_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "state.json"
    keepalive.write_state(path, {"last_verdict": keepalive.ALIVE})
    assert keepalive.read_state(path) == {"last_verdict": keepalive.ALIVE}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_corrupt_state_reads_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("{not json", encoding="utf-8")
    assert keepalive.read_state(path) == {}
    assert keepalive.read_state(tmp_path / "absent.json") == {}


def test_age_days_handles_naive_missing_and_bad_timestamps() -> None:
    now = datetime(2026, 1, 31, tzinfo=UTC)
    assert keepalive.age_days((now - timedelta(days=3)).isoformat(), now=now) == 3.0
    assert keepalive.age_days("2026-01-28T00:00:00", now=now) == 3.0
    assert keepalive.age_days(None) is None
    assert keepalive.age_days("not-a-timestamp") is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _args(**overrides: Any) -> Any:
    parser = keepalive.build_parser()
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_check_only_reports_stale_without_connecting(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    old = (datetime.now(UTC) - timedelta(days=25)).isoformat()
    keepalive.write_state(path, {"last_success_utc": old, "broker": "kafka.gcn.nasa.gov:9092"})
    result = keepalive.run(_args(check_only=True, state_file=str(path), max_age_days=14))
    assert result.verdict == keepalive.STALE
    assert result.age_days is not None and result.age_days > 14


def test_check_only_reports_fresh(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    recent = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    keepalive.write_state(path, {"last_success_utc": recent})
    result = keepalive.run(_args(check_only=True, state_file=str(path), max_age_days=14))
    assert result.verdict == keepalive.ALIVE


def test_check_only_with_no_history_is_stale(tmp_path: Path) -> None:
    result = keepalive.run(_args(check_only=True, state_file=str(tmp_path / "absent.json")))
    assert result.verdict == keepalive.STALE
    assert result.last_success is None


def test_successful_run_records_the_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(keepalive._ENV_CLIENT_ID, _CLIENT_ID)
    monkeypatch.setenv(keepalive._ENV_CLIENT_SECRET, _STANDIN)
    monkeypatch.setenv(keepalive._ENV_CREDENTIALS_FILE, str(tmp_path / "absent.env"))
    monkeypatch.setattr(
        keepalive,
        "probe",
        lambda creds, timeout=0.0: keepalive.Result(
            verdict=keepalive.ALIVE,
            detail="ok",
            broker=creds.bootstrap_servers,
            fingerprint=creds.fingerprint,
            topic_count=7,
        ),
    )
    state_file = tmp_path / "state.json"
    result = keepalive.run(_args(state_file=str(state_file)))
    assert result.verdict == keepalive.ALIVE
    written = json.loads(state_file.read_text(encoding="utf-8"))
    assert written["last_verdict"] == keepalive.ALIVE
    assert written["last_success_utc"] == result.last_success
    assert written["client_id_fingerprint"] == _creds().fingerprint
    assert _CLIENT_ID not in state_file.read_text(encoding="utf-8")
    assert _STANDIN not in state_file.read_text(encoding="utf-8")


def test_failed_run_keeps_the_previous_success_and_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(keepalive._ENV_CLIENT_ID, _CLIENT_ID)
    monkeypatch.setenv(keepalive._ENV_CLIENT_SECRET, _STANDIN)
    monkeypatch.setenv(keepalive._ENV_CREDENTIALS_FILE, str(tmp_path / "absent.env"))
    state_file = tmp_path / "state.json"
    old = (datetime.now(UTC) - timedelta(days=20)).isoformat()
    keepalive.write_state(state_file, {"last_success_utc": old})
    monkeypatch.setattr(
        keepalive,
        "probe",
        lambda creds, timeout=0.0: keepalive.Result(verdict=keepalive.UNREACH, detail="down"),
    )
    result = keepalive.run(_args(state_file=str(state_file), max_age_days=14))
    assert result.verdict == keepalive.UNREACH
    assert result.last_success == old
    assert any("30 days" in warning for warning in result.warnings)
    assert json.loads(state_file.read_text(encoding="utf-8"))["last_success_utc"] == old


def test_main_missing_credentials_exits_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(keepalive._ENV_CLIENT_ID, raising=False)
    monkeypatch.delenv(keepalive._ENV_CLIENT_SECRET, raising=False)
    monkeypatch.setenv(keepalive._ENV_CREDENTIALS_FILE, str(tmp_path / "absent.env"))
    code = keepalive.main(["--state-file", str(tmp_path / "state.json")])
    assert code == keepalive._EXIT_CONFIG
    assert "CONFIG" in capsys.readouterr().out


def test_main_prints_fingerprint_never_the_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(keepalive._ENV_CLIENT_ID, _CLIENT_ID)
    monkeypatch.setenv(keepalive._ENV_CLIENT_SECRET, _STANDIN)
    monkeypatch.setenv(keepalive._ENV_CREDENTIALS_FILE, str(tmp_path / "absent.env"))
    monkeypatch.setattr(
        keepalive,
        "probe",
        lambda creds, timeout=0.0: keepalive.Result(
            verdict=keepalive.ALIVE,
            detail="ok",
            broker=creds.bootstrap_servers,
            fingerprint=creds.fingerprint,
        ),
    )
    code = keepalive.main(["--state-file", str(tmp_path / "state.json")])
    out = capsys.readouterr().out
    assert code == 0
    assert _STANDIN not in out
    assert _CLIENT_ID not in out
    assert _creds().fingerprint in out


def test_main_json_mode_is_machine_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    keepalive.write_state(
        tmp_path / "state.json",
        {"last_success_utc": (datetime.now(UTC) - timedelta(days=1)).isoformat()},
    )
    code = keepalive.main(["--check-only", "--json", "--state-file", str(tmp_path / "state.json")])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["verdict"] == keepalive.ALIVE
    assert "client_id_fingerprint" in payload


def test_exit_codes_are_distinct() -> None:
    """A scheduler must be able to tell a dead credential from a dead network."""
    assert keepalive._EXIT_CODES[keepalive.ALIVE] == 0
    assert keepalive._EXIT_CODES[keepalive.FAIL] == 1
    assert keepalive._EXIT_CODES[keepalive.UNREACH] == 2
    assert keepalive._EXIT_CONFIG == 3
