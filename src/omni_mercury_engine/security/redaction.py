# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Canonical credential-redaction primitives for exception text and logs.

Mercury composes keyed upstream URLs two ways: credentials ride in
query parameters (NASA ``api_key``, AirNow ``API_KEY``, EIA
``api_key``, OpenWeatherMap ``appid``, Alpha Vantage ``apikey``) or in
a path segment (the NASA FIRMS MAP key).  Transport libraries embed the
fully-composed request URL in exception text -- ``httpx.HTTPStatusError``
renders ``"... for url 'https://host/path?api_key=SECRET'"`` and
requests/urllib3 do the equivalent -- so any exception message, chained
``__cause__``/``__context__``, or log line that carries a composed URL
is a credential-disclosure surface.

This module is the one shared implementation those surfaces call.  It
is deliberately **stdlib-only** (``re`` + ``urllib.parse``) with no
intra-package imports, so the exception layer (``datasets/exceptions``),
the egress gate (``security/safe_http``), the live-source transport
(``data_sources/base``), and operator scripts can all use it without
import-weight or cycle concerns.

Two complementary strategies, used together as defence in depth:

* **Structural** (:func:`redact_url`, :func:`redact_text`): recognise
  credential-bearing *positions* -- URL userinfo and query parameters
  whose names are credential-shaped -- and replace the values.  Works
  with no knowledge of the secret's value, so it also protects
  credentials the process never held (e.g. a signed URL echoed by an
  upstream in a redirect ``Location``).

* **Value-based** (:func:`redact_secrets`): replace known secret
  *values* wherever they appear, including URL-encoded forms.  This is
  what catches path-segment keys (FIRMS) that structural redaction
  cannot recognise, and secrets echoed into response bodies.

Neither strategy destroys diagnostics: scheme, host, path,
non-credential parameters, HTTP status and remediation hints all
survive, so ``test_error_carries_url_and_remediation``-style contracts
(the *path* names the artifact being fetched) keep holding.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Replacement token for a structurally-redacted value.
REDACTED = "<redacted>"

#: Query-parameter names (lowercase, underscore-normalised) treated as
#: credential-bearing.  Matching is case-insensitive and ``-``/``_``
#: insensitive, so ``API_KEY``, ``api-key`` and ``ApiKey`` all match
#: ``api_key``.  The list is deliberately broad: over-redacting a benign
#: parameter costs one diagnostic detail, while under-redacting leaks a
#: live credential into every rendered traceback.  Real Mercury
#: upstreams covered: NASA / EIA (``api_key``), AirNow (``API_KEY``),
#: Alpha Vantage (``apikey``), OpenWeatherMap (``appid``), OAuth token
#: flows (``access_token`` / ``refresh_token`` / ``client_secret``),
#: signed URLs (``signature`` / ``sig``), FIRMS when passed as a query
#: (``map_key``).
CREDENTIAL_QUERY_PARAMS: frozenset[str] = frozenset(
    {
        "access_key",
        "access_token",
        "accesskey",
        "api_key",
        "apikey",
        "app_id",
        "app_key",
        "appid",
        "auth",
        "auth_token",
        "authorization",
        "bearer",
        "client_secret",
        "credential",
        "credentials",
        "id_token",
        "key",
        "map_key",
        "mapkey",
        "passwd",
        "password",
        "private_key",
        "pwd",
        "refresh_token",
        "secret",
        "session_id",
        "sessionid",
        "sid",
        "sig",
        "signature",
        "token",
        "x_api_key",
    }
)

#: Below this length a "secret" is degenerate: it cannot hold enough
#: entropy to be a real credential, and blind substring replacement of a
#: 1-3 character value corrupts arbitrary text (every ``key=`` would
#: lose its ``k``).  Structural redaction still protects short values in
#: URLs -- :func:`redact_url` replaces by *position*, not value -- so
#: skipping them here does not reopen the query-parameter leak the old
#: 8-character floor left for short keys.
_MIN_SECRET_LEN = 4

#: At and above this length a secret value is distinctive enough for a
#: plain substring replacement (the historical ``live_data_smoke``
#: behaviour).  Values between :data:`_MIN_SECRET_LEN` and this floor
#: are replaced only at non-alphanumeric boundaries so a 4-7 character
#: value cannot silently mangle unrelated words that merely contain it.
_BOUNDARY_SECRET_LEN = 8

#: Upper bound on the scheme this pattern will scan before giving up on a
#: candidate start position.  RFC 3986 places no length limit on a scheme,
#: but the longest IANA-registered one is 31 characters, so 40 admits every
#: real scheme with room to spare.  The bound is what makes the scan
#: **linear**: unbounded, ``[a-zA-Z0-9+.-]*`` re-scans to end-of-string from
#: every one of ``n`` start positions before failing to find ``://``, which
#: is quadratic in the input length.  That input is attacker-influenced --
#: this module exists to scrub *upstream* response bodies and transport
#: exception text -- so the quadratic form is a denial-of-service surface on
#: the sanitizer itself.  Measured on a scheme-shaped string with no ``://``:
#: 128 KB took 13.4 s unbounded and 11.7 ms bounded.
_MAX_SCHEME_LEN = 40

#: ``<`` and ``>`` terminate a URL/query match so markup is never
#: swallowed -- EXCEPT the literal :data:`REDACTED` token, which must be
#: admitted mid-match or redaction would not be idempotent: on a second
#: pass over already-redacted text the match would stop at ``<redacted>``,
#: see a credential parameter with an apparently-empty value, and stamp a
#: second token beside the first.  Layers deliberately stack (a scrubbed
#: transport message flows into ``DataSourceUnavailableError`` and again
#: through script-level redaction), so idempotency is load-bearing.
_URL_IN_TEXT_RE = re.compile(
    rf"[a-zA-Z][a-zA-Z0-9+.-]{{0,{_MAX_SCHEME_LEN - 1}}}://(?:<redacted>|[^\s'\"<>`\\])+"
)

#: A query string appearing WITHOUT a scheme-qualified URL around it.
#: urllib3 renders transport failures as ``"HTTPSConnectionPool(host=...):
#: Max retries exceeded with url: /query?apikey=SECRET (Caused by ...)"``
#: -- path + query, no ``scheme://`` -- so a URL-shaped regex alone would
#: miss exactly the shape ``requests`` produces on every connection
#: failure.  Measured against the real message shape before being written.
#: Admits the ``<redacted>`` token for the same idempotency reason as
#: :data:`_URL_IN_TEXT_RE`.
_BARE_QUERY_RE = re.compile(r"\?(?:<redacted>|[^\s'\"<>`])+")


def _param_is_credential(name: str) -> bool:
    """Whether query parameter ``name`` is credential-shaped."""
    return name.strip().lower().replace("-", "_") in CREDENTIAL_QUERY_PARAMS


def _redact_query(query: str) -> str:
    """Redact credential-named parameter values inside a query string.

    Leaves the string byte-identical when no credential-shaped parameter
    is present, so prose that merely contains a ``?`` is never mangled
    and non-credential queries keep their original encoding.
    """
    pairs = parse_qsl(query, keep_blank_values=True)
    if not any(_param_is_credential(name) for name, _ in pairs):
        return query
    redacted_pairs = [
        (name, REDACTED if _param_is_credential(name) else value) for name, value in pairs
    ]
    # ``safe="<>"`` keeps the REDACTED token readable instead of
    # percent-escaping its angle brackets (also makes re-redaction
    # idempotent).
    return urlencode(redacted_pairs, safe="<>")


def redact_url(url: str) -> str:
    """Return ``url`` with credential-bearing components replaced.

    Redacts, structurally (no knowledge of any secret value needed):

    * **userinfo** -- ``https://user:pass@host/`` becomes
      ``https://<redacted>@host/`` (the password is the secret; the
      whole userinfo goes because usernames are frequently API keys);
    * **credential query parameters** -- any parameter whose name
      matches :data:`CREDENTIAL_QUERY_PARAMS` case- and
      ``-``/``_``-insensitively keeps its name but has its value
      replaced with :data:`REDACTED`.

    Everything else -- scheme, host, port, path, non-credential
    parameters, fragment -- survives verbatim so the result stays a
    useful diagnostic.  Relative references (``/path?key=x``) are
    handled; an unparseable value degrades to the constant
    ``"<unparseable-url:redacted>"`` rather than passing through
    unredacted (fail closed: if we cannot prove it is safe we do not
    emit it).

    Args:
        url: Absolute or relative URL, possibly credentialed.

    Returns:
        The URL with credential positions redacted.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<unparseable-url:redacted>"

    netloc = parts.netloc
    if "@" in netloc:
        netloc = f"{REDACTED}@{netloc.rpartition('@')[2]}"

    return urlunsplit(
        (parts.scheme, netloc, parts.path, _redact_query(parts.query), parts.fragment)
    )


def redact_text(text: str) -> str:
    """Structurally redact every URL that appears inside free text.

    Two passes:

    * ``scheme://...`` substrings (any scheme, so ``http``/``https``
      /``ftp`` and friends are all covered) get :func:`redact_url`;
    * bare ``?name=value`` query strings with no scheme around them get
      their credential-named parameter values redacted -- urllib3
      renders connection failures as ``"... Max retries exceeded with
      url: /query?apikey=SECRET (Caused by ...)"``, path + query with
      no scheme, so the URL pass alone would miss the exact shape
      ``requests`` produces on every connection failure.

    Use this on text you did not compose -- transport-exception strings,
    upstream response bodies, re-wrapped loader messages -- where a
    composed URL may be embedded anywhere.  Prose containing a ``?``
    with no credential-shaped parameter after it passes through
    byte-identical.

    Args:
        text: Arbitrary diagnostic text.

    Returns:
        ``text`` with every embedded URL's credential positions redacted.
    """
    if not text:
        return text
    text = _URL_IN_TEXT_RE.sub(lambda match: redact_url(match.group(0)), text)
    return _BARE_QUERY_RE.sub(lambda match: f"?{_redact_query(match.group(0)[1:])}", text)


def redact_secrets(text: str, labels: Sequence[str], secret_values: Sequence[str | None]) -> str:
    """Replace known secret values wherever they appear in ``text``.

    The complement of the structural functions: when the caller *holds*
    the credential (a loader's configured ``api_key``, the key env vars
    a CI script exports) every occurrence of the value -- and of its
    URL-encoded form, which is how a key with special characters
    actually appears inside a composed URL -- is replaced with
    ``<LABEL:redacted>`` so the reader learns *which* credential was
    scrubbed without learning the credential.

    Labels and secret values travel as PARALLEL SEQUENCES, deliberately
    not as one mapping or as ``(label, value)`` pairs: the replacement
    token -- the only part of the output this function composes -- is
    built exclusively from the ``labels`` channel, which never holds
    secret material, so "no secret value reaches the output" holds by
    construction and is visible to dataflow analysis.  Collapsing the
    two channels back into one container re-couples them: every read
    out of a secret-holding container carries the secret's taint, which
    puts label extraction -- and therefore the composed output -- back
    on the tainted path at every log site downstream of this function.

    Values shorter than 4 characters are skipped: they cannot be real
    credentials and replacing them would corrupt arbitrary text (see
    :data:`_MIN_SECRET_LEN`).  Values of 4-7 characters are replaced
    only at non-alphanumeric boundaries; 8 characters and up are
    replaced everywhere, including as substrings (a real key embedded in
    a FIRMS path segment has no word boundary on either side when
    concatenated into surrounding text).

    Args:
        text: Text that may contain the secret values.
        labels: Human-readable label per secret (typically the env var
            or config-field name); position-matched with
            ``secret_values``.
        secret_values: The secret value per label; ``None`` / empty /
            whitespace-only values are ignored.

    Returns:
        ``text`` with every known secret value redacted.

    Raises:
        TypeError: If ``labels`` or ``secret_values`` is a single
            string -- a ``str`` is itself a ``Sequence[str]``, and
            iterating one silently redacts per *character*.
        ValueError: If the two sequences differ in length -- silent
            zip-truncation would drop a value and let it through
            unredacted.
    """
    if isinstance(labels, str) or isinstance(secret_values, str):
        raise TypeError("labels and secret_values must be sequences, not a single string")
    if len(labels) != len(secret_values):
        raise ValueError(
            f"labels ({len(labels)}) and secret_values ({len(secret_values)}) "
            "must be position-matched"
        )
    if not text:
        return text
    for index, label in enumerate(labels):
        value = (secret_values[index] or "").strip()
        if len(value) < _MIN_SECRET_LEN:
            continue
        replacement = f"<{label}:redacted>"
        candidates = {value}
        encoded = quote(value, safe="")
        if encoded != value:
            candidates.add(encoded)
        for candidate in candidates:
            if len(candidate) >= _BOUNDARY_SECRET_LEN:
                text = text.replace(candidate, replacement)
            else:
                text = re.sub(
                    rf"(?<![A-Za-z0-9]){re.escape(candidate)}(?![A-Za-z0-9])",
                    replacement,
                    text,
                )
    return text


#: Environment-variable NAMES treated as credential-holding for
#: :func:`redact_env_secrets`.  Matches a credential-shaped token
#: (``API_KEY`` / ``TOKEN`` / ``SECRET`` / ``PASSWORD`` / ``MAP_KEY`` /
#: ``CREDENTIAL`` / ``AUTH`` / ``BEARER`` / ``PRIVATE_KEY``) bounded by
#: ``_`` or the string edge, so ``NASA_FIRMS_MAP_KEY`` and
#: ``FHIR_BEARER_TOKEN`` match while ``AUTHOR`` or ``TOKENIZER_PATH``
#: (token bounded by neither edge nor underscore) do not.
_ENV_SECRET_NAME_RE = re.compile(
    r"(?:^|_)(?:API_?KEY|APIKEY|TOKEN|SECRET|PASSWORD|PASSWD|MAP_?KEY"
    r"|CREDENTIALS?|AUTH|BEARER|PRIVATE_?KEY|SIGNATURE)(?:$|_)",
    re.IGNORECASE,
)


def redact_env_secrets(text: str) -> str:
    """Scrub the value of every credential-shaped environment variable.

    The structural functions cannot recognise a credential that rides in
    a URL *path segment* (NASA FIRMS embeds the MAP key that way) -- only
    knowledge of the value catches those.  This process *holds* its own
    credentials in the environment, so scanning ``os.environ`` for
    credential-shaped names (``*_API_KEY``, ``*_TOKEN``, ``*_SECRET``,
    ``*MAP_KEY*``, ...) and value-redacting each is the general form of
    the per-script env-var lists: it needs no per-callsite registry and
    automatically covers keys added later.  Values shorter than 4
    characters are skipped by :func:`redact_secrets`'s degeneracy floor.

    Args:
        text: Arbitrary diagnostic text.

    Returns:
        ``text`` with every configured credential value replaced by
        ``<VAR_NAME:redacted>``.
    """
    if not text:
        return text
    # Names first, values looked up per name: the two channels stay
    # separate from the origin (see the redact_secrets docstring).
    labels = [name for name in os.environ if _ENV_SECRET_NAME_RE.search(name)]
    return redact_secrets(text, labels, [os.environ.get(name) for name in labels])


__all__ = [
    "CREDENTIAL_QUERY_PARAMS",
    "REDACTED",
    "redact_env_secrets",
    "redact_secrets",
    "redact_text",
    "redact_url",
]
