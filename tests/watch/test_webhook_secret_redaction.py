# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A secret-bearing webhook URL never reaches a log, an error, or the store.

Incoming-webhook URLs (Slack, Discord, Teams, generic) are bearer
credentials: whoever holds the URL can post, and the secret rides in the URL
*path*.  The generic redactors strip credential-named query values and
userinfo, and scrub the values of env vars whose NAMES look like credentials
— but ``MERCURY_WATCH_WEBHOOK_URL`` matched none of those names, so a failed
delivery's ``"... for url: https://hooks.../<secret>"`` reached the
SafeHTTPClient debug log, the re-raised ``HTTPError`` text and the stored
delivery error verbatim.

Pinned here, at three layers, with a deliberately secret-bearing URL:

* the root cause — the env-value pass now scrubs ``*WEBHOOK*`` variables
  (fails on the prior name pattern);
* the transport layer — SafeHTTPClient's canonical scrub of an
  ``HTTPError`` message removes the secret;
* the delivery layer — ``Notifier._attempt`` keeps the secret out of the
  stored delivery error and out of EVERY log record at every level, even
  when handed an exception text no lower layer redacted.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

_SECRET = "zQ9mT7vK2pX4sECRETwebhookTOKEN81"
_SECRET_URL = f"https://hooks.example.invalid/services/T0000000/B0000000/{_SECRET}"


@pytest.fixture
def webhook_env(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("MERCURY_WATCH_WEBHOOK_URL", _SECRET_URL)
    return _SECRET_URL


class TestRootCause:
    def test_webhook_named_vars_are_credential_shaped(self) -> None:
        from omni_mercury_engine.security.redaction import _ENV_SECRET_NAME_RE

        for name in (
            "MERCURY_WATCH_WEBHOOK_URL",
            "SLACK_WEBHOOK_URL",
            "ALERT_WEBHOOK",
        ):
            assert _ENV_SECRET_NAME_RE.search(name), name

    def test_env_pass_scrubs_the_webhook_value(self, webhook_env: str) -> None:
        """Fails on the prior pattern, where no alternative matched the name."""
        from omni_mercury_engine.security.redaction import redact_env_secrets

        text = f"404 Client Error: Not Found for url: {webhook_env}"
        scrubbed = redact_env_secrets(text)
        assert _SECRET not in scrubbed
        assert "404 Client Error" in scrubbed  # the diagnosis survives


class TestTransportLayer:
    def test_safe_http_canonical_scrub_removes_the_secret(self, webhook_env: str) -> None:
        """The exact transform SafeHTTPClient applies to a re-raised HTTPError."""
        from omni_mercury_engine.security.redaction import redact_text
        from omni_mercury_engine.security.safe_http import _redact_full

        raw = f"404 Client Error: Not Found for url: {webhook_env}"
        assert _SECRET not in _redact_full(redact_text(raw))


class _RecordingStore:
    def __init__(self) -> None:
        self.deliveries: list[dict[str, Any]] = []

    def record_delivery(self, *args: Any, **kwargs: Any) -> None:
        self.deliveries.append({"args": args, **kwargs})


class TestDeliveryLayer:
    def _attempt_with_failing_send(self, message: str) -> _RecordingStore:
        from omni_mercury_engine.watch.notify import Notifier

        store = _RecordingStore()

        class _Self:
            pass

        stand_in = _Self()
        stand_in.store = store  # type: ignore[attr-defined]
        stand_in._clock = lambda: 0.0  # type: ignore[attr-defined]

        def failing_send() -> None:
            raise RuntimeError(message)

        done = {"sent": 0, "failed": 0}
        Notifier._attempt(stand_in, "finding-1", 2, 7, "webhook", done, failing_send)  # type: ignore[arg-type]
        assert done == {"sent": 0, "failed": 1}
        return store

    def test_unredacted_exception_text_never_reaches_store_or_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Worst case: handed raw text no lower layer scrubbed (and with no
        env var set, so the env pass cannot help) — the delivery layer's own
        URL-path stripping must still keep the secret out of everything."""
        caplog.set_level(logging.DEBUG)
        store = self._attempt_with_failing_send(
            f"404 Client Error: Not Found for url: {_SECRET_URL}"
        )
        stored_error = store.deliveries[0]["error"]
        assert stored_error is not None
        assert _SECRET not in stored_error
        assert "hooks.example.invalid" in stored_error  # host kept for diagnosis
        for record in caplog.records:
            assert _SECRET not in record.getMessage(), record.name
            assert _SECRET not in str(record.args), record.name

    def test_secret_absent_with_the_env_configured_too(
        self, webhook_env: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG)
        store = self._attempt_with_failing_send(f"ConnectionError: {webhook_env} refused")
        assert _SECRET not in str(store.deliveries[0])
        assert all(_SECRET not in r.getMessage() for r in caplog.records)
