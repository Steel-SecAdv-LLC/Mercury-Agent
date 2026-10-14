#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep-alive probe for the GCN Kafka client credentials behind the MCP server.

NASA's General Coordinates Network (GCN) disables a Kafka client credential
that has not been used to connect to a broker for 30 days, and warns by email
nine days before it goes. A disabled credential cannot be re-enabled -- a new
one has to be issued and every consumer reconfigured. An MCP client that only
attaches the Kafka server occasionally is exactly the usage pattern that walks
into that deadline.

This script is the scheduled counterpart to that interactive use: it makes ONE
authenticated metadata request against the GCN broker, which is what GCN counts
as "use", and records the timestamp in a small state file so an operator can
answer "when did this credential last actually connect?" without guessing.

Secrets are read from the environment (or from a credentials file the script
refuses to read unless it is owner-only), never from a repository file, never
echoed, and never written to the state file -- the client id is recorded as a
truncated SHA-256 fingerprint so one state file can be told from another
without carrying the identifier itself.

Verdicts
========

``ALIVE``
    The broker accepted the credential and returned cluster metadata. The
    30-day inactivity clock is reset as of this run.
``FAIL``
    The credential was rejected (disabled, revoked, or wrong). Actionable:
    issue a new one at https://gcn.nasa.gov/quickstart and update the secret
    store.
``UNREACH``
    Neither the broker nor the token endpoint could be reached. This says
    nothing about the credential, so it is reported and not counted against it
    -- but it also means the clock was NOT reset, so a stale-state warning
    still fires.
``STALE``
    ``--check-only``: no connection was attempted, and the last recorded
    success is older than ``--max-age-days``.

Exit status: 0 ``ALIVE`` (or a fresh ``--check-only``), 1 ``FAIL`` or a stale
``--check-only``, 2 ``UNREACH``, 3 configuration/dependency problem.

Usage::

    python scripts/gcn_kafka_keepalive.py                  # probe + record
    python scripts/gcn_kafka_keepalive.py --check-only     # read state only
    python scripts/gcn_kafka_keepalive.py --json           # machine-readable

Requires ``confluent-kafka`` (or ``gcn-kafka``, which wraps it) for the broker
probe; the token pre-flight and the state ledger are stdlib-only.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

#: GCN's three Kafka deployments. ``gcn-kafka`` derives the broker and the
#: OpenID Connect token endpoint from the domain the same way, so a caller
#: that knows the domain never has to hand-write either host.
_DOMAINS: tuple[str, ...] = ("gcn.nasa.gov", "test.gcn.nasa.gov", "dev.gcn.nasa.gov")

#: Default location of the owner-only credentials file, kept OUTSIDE the
#: repository so no checkout, backup, or AI transcript of this project can
#: contain the secret.
_DEFAULT_CREDENTIALS_FILE = "~/.config/mercury/gcn-kafka.env"

#: Default state ledger. XDG state dir: this is machine-local runtime history,
#: not configuration, and it is safe to lose (the next run re-establishes it).
_DEFAULT_STATE_FILE = "~/.local/state/mercury/gcn-kafka-keepalive.json"

#: Warn when the last successful connection is older than this. GCN disables at
#: 30 days and emails at 21. The shipped schedules run daily, so 14 days with no
#: success means the schedule itself has stopped working -- surfaced with half
#: the window still left, and a week before GCN's own warning email.
_DEFAULT_MAX_AGE_DAYS = 14

#: Broker metadata timeout (seconds). A metadata round trip that has not
#: completed in 30s is an outage, not slowness.
_DEFAULT_TIMEOUT = 30.0

#: Env var names. Only the secret is sensitive, but both are resolved through
#: the same path so the credentials file stays a single unit.
_ENV_CLIENT_ID = "GCN_KAFKA_CLIENT_ID"
_ENV_CLIENT_SECRET = "GCN_KAFKA_CLIENT_SECRET"  # noqa: S105 - a var NAME, not a secret
_ENV_DOMAIN = "GCN_KAFKA_DOMAIN"
_ENV_BOOTSTRAP = "GCN_KAFKA_BOOTSTRAP_SERVERS"
_ENV_CREDENTIALS_FILE = "GCN_KAFKA_CREDENTIALS_FILE"
_ENV_STATE_FILE = "GCN_KAFKA_STATE_FILE"

#: Substrings that mark a librdkafka error as "the broker rejected who you are"
#: rather than "the broker never answered". librdkafka reports authentication
#: outcomes as free text, so this classification is heuristic -- which is why a
#: rejection is confirmed against the token endpoint before it is reported as
#: ``FAIL`` (see :func:`probe`).
_AUTH_ERROR_MARKERS: tuple[str, ...] = (
    "authentication",
    "sasl",
    "oauthbearer",
    "unauthorized",
    "invalid_client",
    "token",
    "401",
    "403",
)

ALIVE = "ALIVE"
FAIL = "FAIL"
UNREACH = "UNREACH"
STALE = "STALE"

_EXIT_CODES = {ALIVE: 0, FAIL: 1, UNREACH: 2, STALE: 1}
_EXIT_CONFIG = 3


class ConfigError(RuntimeError):
    """The run cannot start: credentials missing, unreadable, or unsafe."""


@dataclass(frozen=True)
class Credentials:
    """A GCN client credential pair and the deployment it belongs to."""

    client_id: str
    client_secret: str
    domain: str
    bootstrap_servers: str

    @property
    def token_endpoint(self) -> str:
        """OpenID Connect token endpoint for this deployment."""
        return f"https://auth.{self.domain}/oauth2/token"

    @property
    def fingerprint(self) -> str:
        """Non-secret, stable handle for the client id (first 12 hex of SHA-256)."""
        return hashlib.sha256(self.client_id.encode("utf-8")).hexdigest()[:12]

    def redact(self, text: str) -> str:
        """Strip the secret (and the client id) out of *text* before it is printed."""
        for value in (self.client_secret, self.client_id):
            if len(value) >= 8:
                text = text.replace(value, "***")
        return text


@dataclass
class Result:
    """Outcome of one keep-alive run."""

    verdict: str
    detail: str
    broker: str = ""
    fingerprint: str = ""
    topic_count: int | None = None
    last_success: str | None = None
    age_days: float | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Render the result as a JSON-serialisable mapping."""
        return {
            "verdict": self.verdict,
            "detail": self.detail,
            "broker": self.broker,
            "client_id_fingerprint": self.fingerprint,
            "topic_count": self.topic_count,
            "last_success_utc": self.last_success,
            "age_days": self.age_days,
            "warnings": list(self.warnings),
        }


def _now() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Credential resolution
# ---------------------------------------------------------------------------


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a ``KEY=VALUE`` credentials file, refusing group/world-readable ones.

    The file holds a live secret, so a mode that lets any other account on the
    host read it is a configuration error, not a warning to be logged and
    ignored: the whole point of keeping the secret out of the repository is
    lost if it sits at 0644 in the operator's home directory.
    """
    if not path.exists():
        return {}
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise ConfigError(
            f"credentials file {path} is mode {mode:04o}; it holds a live secret. "
            f"Run: chmod 600 {path}"
        )
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        values[key.strip()] = value.strip().strip("'\"")
    return values


def load_credentials(
    *,
    credentials_file: Path | None = None,
    domain: str | None = None,
    bootstrap_servers: str | None = None,
    environ: dict[str, str] | None = None,
) -> Credentials:
    """Resolve credentials from the environment, then the credentials file.

    The environment wins: a CI job injecting a secret must not be silently
    overridden by a stale file left in the runner's home directory.
    """
    env = dict(os.environ if environ is None else environ)
    path = (
        credentials_file
        or Path(env.get(_ENV_CREDENTIALS_FILE, _DEFAULT_CREDENTIALS_FILE)).expanduser()
    )
    from_file = parse_env_file(path)

    def resolve(name: str) -> str:
        return (env.get(name) or from_file.get(name) or "").strip()

    client_id = resolve(_ENV_CLIENT_ID)
    client_secret = resolve(_ENV_CLIENT_SECRET)
    missing = [
        name
        for name, value in ((_ENV_CLIENT_ID, client_id), (_ENV_CLIENT_SECRET, client_secret))
        if not value
    ]
    if missing:
        raise ConfigError(
            f"missing {', '.join(missing)}. Set them in the environment or in {path} "
            f"(chmod 600). Issue credentials at https://gcn.nasa.gov/quickstart"
        )

    resolved_domain = (domain or resolve(_ENV_DOMAIN) or _DOMAINS[0]).strip()
    if resolved_domain not in _DOMAINS:
        raise ConfigError(f"unknown GCN domain {resolved_domain!r}; expected one of {_DOMAINS}")
    servers = (
        bootstrap_servers or resolve(_ENV_BOOTSTRAP) or f"kafka.{resolved_domain}:9092"
    ).strip()
    return Credentials(
        client_id=client_id,
        client_secret=client_secret,
        domain=resolved_domain,
        bootstrap_servers=servers,
    )


def consumer_config(creds: Credentials, group_id: str = "mercury-gcn-keepalive") -> dict[str, Any]:
    """librdkafka configuration for a GCN SASL_SSL/OAUTHBEARER connection.

    This is the same shape ``gcn-kafka`` builds internally; it is spelled out
    here so the probe works with a plain ``confluent-kafka`` install and so the
    identical block can be pasted into an MCP server's own config.
    """
    return {
        "bootstrap.servers": creds.bootstrap_servers,
        "group.id": group_id,
        "security.protocol": "SASL_SSL",
        "sasl.mechanisms": "OAUTHBEARER",
        "sasl.oauthbearer.method": "oidc",
        "sasl.oauthbearer.client.id": creds.client_id,
        "sasl.oauthbearer.client.secret": creds.client_secret,
        "sasl.oauthbearer.token.endpoint.url": creds.token_endpoint,
    }


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def _default_consumer_factory(config: dict[str, Any]) -> Any:
    """Build a confluent-kafka Consumer, preferring the gcn-kafka wrapper."""
    try:
        from confluent_kafka import Consumer
    except ImportError as exc:
        raise ConfigError(
            "confluent-kafka is required for the broker probe. "
            "Install with: pip install gcn-kafka"
        ) from exc
    return Consumer(config)


def probe_broker(
    creds: Credentials,
    *,
    timeout: float = _DEFAULT_TIMEOUT,
    factory: Any = None,
) -> tuple[bool, str, int]:
    """Make one authenticated metadata request. Returns ``(ok, detail, topics)``.

    A metadata request is the cheapest exchange that still requires the broker
    to authenticate the client, which is precisely what GCN counts as use.
    """
    build = factory or _default_consumer_factory
    consumer = build(consumer_config(creds))
    try:
        metadata = consumer.list_topics(timeout=timeout)
        topics = getattr(metadata, "topics", {}) or {}
        return True, f"broker returned metadata for {len(topics)} topic(s)", len(topics)
    except Exception as exc:
        return False, creds.redact(f"{type(exc).__name__}: {exc}"), 0
    finally:
        close = getattr(consumer, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def probe_token(creds: Credentials, *, timeout: float = 15.0) -> tuple[bool | None, str]:
    """Ask the OIDC token endpoint whether the credential itself is still valid.

    Returns ``(True, ...)`` when a token is issued, ``(False, ...)`` when the
    credential is rejected, and ``(None, ...)`` when the endpoint could not
    answer -- inconclusive, never counted against the credential. This runs
    only to classify a failed broker probe: librdkafka's error text alone
    cannot reliably tell a disabled credential from a network outage.
    """
    body = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode("ascii")
    basic = base64.b64encode(f"{creds.client_id}:{creds.client_secret}".encode()).decode("ascii")
    request = urllib.request.Request(  # noqa: S310 - fixed https:// GCN endpoint
        creds.token_endpoint,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8") or "{}")
        if payload.get("access_token"):
            return True, "token endpoint issued an access token"
        return None, "token endpoint answered without a token"
    except urllib.error.HTTPError as exc:
        if exc.code in (400, 401, 403):
            return False, f"token endpoint rejected the credential (HTTP {exc.code})"
        return None, f"token endpoint returned HTTP {exc.code}"
    except Exception as exc:
        return None, creds.redact(f"token endpoint unreachable: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# State ledger
# ---------------------------------------------------------------------------


def read_state(path: Path) -> dict[str, Any]:
    """Read the state ledger, treating a missing or corrupt file as empty."""
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def write_state(path: Path, state: dict[str, Any]) -> None:
    """Write the state ledger owner-only, creating parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.chmod(0o600)
    tmp.replace(path)


def age_days(timestamp: str | None, *, now: datetime | None = None) -> float | None:
    """Days since *timestamp* (ISO-8601 UTC), or ``None`` if it is absent/unparsable."""
    if not timestamp:
        return None
    try:
        when = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    delta = (now or _now()) - when
    return round(delta / timedelta(days=1), 3)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def probe(
    creds: Credentials,
    *,
    timeout: float = _DEFAULT_TIMEOUT,
    factory: Any = None,
    token_probe: Any = None,
) -> Result:
    """Run the broker probe and classify the outcome."""
    ok, detail, topics = probe_broker(creds, timeout=timeout, factory=factory)
    if ok:
        return Result(
            verdict=ALIVE,
            detail=detail,
            broker=creds.bootstrap_servers,
            fingerprint=creds.fingerprint,
            topic_count=topics,
        )

    looks_like_auth = any(marker in detail.lower() for marker in _AUTH_ERROR_MARKERS)
    verify = token_probe or probe_token
    valid, token_detail = verify(creds)
    if valid is False:
        verdict = FAIL
    elif valid is True:
        # The credential is good but the broker would not talk: an outage, and
        # the token exchange alone does not reset GCN's broker-side clock.
        verdict = UNREACH
    else:
        verdict = FAIL if looks_like_auth else UNREACH
    return Result(
        verdict=verdict,
        detail=f"{detail} | {token_detail}",
        broker=creds.bootstrap_servers,
        fingerprint=creds.fingerprint,
    )


def run(args: argparse.Namespace) -> Result:
    """Execute the requested mode and return its result."""
    state_path = Path(
        args.state_file or os.environ.get(_ENV_STATE_FILE) or _DEFAULT_STATE_FILE
    ).expanduser()
    state = read_state(state_path)
    last_success = state.get("last_success_utc")

    if args.check_only:
        age = age_days(last_success)
        fresh = age is not None and age <= args.max_age_days
        return Result(
            verdict=ALIVE if fresh else STALE,
            detail=(
                f"last successful connection {age} day(s) ago"
                if age is not None
                else f"no successful connection recorded in {state_path}"
            ),
            broker=str(state.get("broker", "")),
            fingerprint=str(state.get("client_id_fingerprint", "")),
            last_success=last_success,
            age_days=age,
        )

    creds = load_credentials(
        credentials_file=(
            Path(args.credentials_file).expanduser() if args.credentials_file else None
        ),
        domain=args.domain,
        bootstrap_servers=args.bootstrap_servers,
    )
    result = probe(creds, timeout=args.timeout)
    started = _now().isoformat()
    if result.verdict == ALIVE:
        last_success = started
    result.last_success = last_success
    result.age_days = age_days(last_success)
    if result.age_days is not None and result.age_days > args.max_age_days:
        result.warnings.append(
            f"last successful connection was {result.age_days} day(s) ago; GCN disables a "
            f"credential unused for 30 days"
        )
    if last_success is None:
        result.warnings.append("no successful connection has ever been recorded")

    if not args.no_state:
        write_state(
            state_path,
            {
                "broker": creds.bootstrap_servers,
                "client_id_fingerprint": creds.fingerprint,
                "domain": creds.domain,
                "last_attempt_utc": started,
                "last_success_utc": last_success,
                "last_verdict": result.verdict,
            },
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Keep GCN Kafka client credentials alive by connecting to the broker.",
    )
    parser.add_argument("--domain", choices=_DOMAINS, help="GCN deployment (default: production)")
    parser.add_argument("--bootstrap-servers", help="Override the derived broker host:port")
    parser.add_argument(
        "--credentials-file",
        help=(
            "Owner-only KEY=VALUE file holding the credentials "
            f"(default: {_DEFAULT_CREDENTIALS_FILE})"
        ),
    )
    parser.add_argument(
        "--state-file", help=f"Where to record run history (default: {_DEFAULT_STATE_FILE})"
    )
    parser.add_argument(
        "--max-age-days",
        type=float,
        default=_DEFAULT_MAX_AGE_DAYS,
        help=f"Warn when the last success is older than this (default: {_DEFAULT_MAX_AGE_DAYS})",
    )
    parser.add_argument(
        "--timeout", type=float, default=_DEFAULT_TIMEOUT, help="Broker metadata timeout in seconds"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Report the recorded state without connecting to the broker",
    )
    parser.add_argument("--no-state", action="store_true", help="Do not write the state file")
    parser.add_argument("--json", action="store_true", help="Emit the result as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. See module docstring for exit-status semantics."""
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except ConfigError as exc:
        message = f"CONFIG: {exc}"
        print(json.dumps({"verdict": "CONFIG", "detail": str(exc)}) if args.json else message)
        return _EXIT_CONFIG

    if args.json:
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    else:
        print(f"{result.verdict}: {result.detail}")
        if result.broker:
            print(f"  broker      {result.broker}")
        if result.fingerprint:
            print(f"  client id   sha256:{result.fingerprint}")
        if result.last_success:
            print(f"  last ALIVE  {result.last_success} ({result.age_days} day(s) ago)")
        for warning in result.warnings:
            print(f"  WARNING     {warning}")
    return _EXIT_CODES[result.verdict]


if __name__ == "__main__":
    sys.exit(main())
