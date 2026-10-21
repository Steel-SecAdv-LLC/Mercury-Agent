# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Containment-only deception: decoy tripwires, a bounded tarpit, a reliability bridge.

Everything here protects Mercury's own surface and nothing reaches outward.
A decoy is a path no real route serves; touching it records a tripwire, the
response is slowed by a bounded tarpit, and the touch becomes adjudicated
evidence against the source in the counterintelligence reliability model.
No component contacts, probes, disables or alters any other system --
including one that is attacking this one.

Four properties carry the design:

* **No legitimate-client population.** Decoy paths are served by no real
  route and appear in no OpenAPI schema, documentation or client, so a hit
  means probing: targeted at Mercury, or the untargeted scanning every public
  server sees. A hit is evidence of probing, not proof of a targeted attack.
* **Cannot be turned on an innocent.** Source identity comes from
  :func:`~omni_mercury_engine.api.client_ip.resolve_client_ip`, which ignores
  ``X-Forwarded-For`` unless the deployment declares trusted proxy hops, so an
  attacker cannot forge a victim's address into a tripwire. IPv6 sources are
  keyed by ``/64`` -- the unit one subscriber controls -- so an attacker cannot
  reset the tarpit by rotating addresses inside a single allocation.
* **Consequences stay on the decoys.** The tarpit delays decoy responses only,
  and no real route consults this sensor: a legitimate user who shares a NAT
  address with a prober is never slowed or refused.
* **Bounded.** Per-source state is LRU-capped, each source's hit history is
  length-capped, the tarpit delay is capped and overflow-proof, and
  :class:`TarpitGate` caps how many responses may be held at once -- past the
  cap it answers immediately, so the tarpit can never exhaust this server.

The reliability bridge maps tripwires onto
:class:`~omni_mercury_engine.security.counterintelligence.SourceOutcome`
records -- one refuted outcome per *distinct* decoy touched. Repeated hits on
one decoy are a retry loop, not independent adjudications; breadth of probing
is the evidence. :meth:`DeceptionSensor.posterior` applies them through the
module's own :func:`~omni_mercury_engine.security.counterintelligence.source_posterior`,
for consumers that grade a source's submissions.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import math
import os
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from omni_mercury_engine.security.counterintelligence import (
    MAX_OUTCOME_WEIGHT,
    SourceOutcome,
    source_posterior,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Mapping

    from omni_mercury_engine.security.counterintelligence import (
        CounterintelligenceConfig,
        SourcePosterior,
    )

logger = logging.getLogger(__name__)

__all__ = [
    "DECEPTION_ENABLED_ENV",
    "DECOY_PATHS_ENV",
    "DEFAULT_DECOYS",
    "SILENT_DECOY_PATHS_ENV",
    "DeceptionSensor",
    "DecoyResponse",
    "DecoyRoute",
    "TarpitGate",
    "TarpitPolicy",
    "TripwireEvent",
    "configured_decoys",
    "deception_enabled",
    "source_key",
]

#: Opt-in switch for the decoy surface. Left unset, no decoy route exists and
#: every existing route stays byte-identical.
DECEPTION_ENABLED_ENV = "MERCURY_DECEPTION_ENABLED"

#: Comma-separated private decoy paths for this deployment, answered with the
#: protected-route challenge. The shipped :data:`DEFAULT_DECOYS` are public --
#: anyone who reads this repository can avoid them -- so a deployment's own
#: unpublished paths are what catch a prober who has read the code.
DECOY_PATHS_ENV = "MERCURY_DECOY_PATHS"

#: Comma-separated private silent-tripwire paths for this deployment.
SILENT_DECOY_PATHS_ENV = "MERCURY_SILENT_DECOY_PATHS"

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_HTTP_METHODS = ("GET", "HEAD", "POST", "PUT", "PATCH", "DELETE")
_IPV6_HOST_BITS = 64

#: How a decoy answers. ``auth_challenge`` answers as a real protected route
#: (every credential attempt against it is another tripwire); ``not_found`` is
#: a silent tripwire that only observes, so the application's own router
#: answers with its genuine 404 for every method and path variant.
DecoyResponse = Literal["auth_challenge", "not_found"]


def deception_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Whether the decoy surface is enabled.

    Args:
        environ: Environment to read (defaults to :data:`os.environ`).

    Returns:
        ``True`` only for an explicit truthy value of
        :data:`DECEPTION_ENABLED_ENV`; anything else leaves it off.
    """
    env = os.environ if environ is None else environ
    return env.get(DECEPTION_ENABLED_ENV, "").strip().lower() in _TRUTHY


@dataclass(frozen=True)
class DecoyRoute:
    """One decoy path.

    Attributes:
        path: Exact URL path (no parameters, not the site root) that no real
            route serves.
        lure: What the decoy impersonates, for the tripwire record.
        response: How the decoy answers (see :data:`DecoyResponse`).
        methods: HTTP methods an ``auth_challenge`` decoy routes, one route
            per method in this order -- the way Mercury's own routers declare
            them (``POST`` then ``GET``), so a wrong-method 405 carries the
            same ``Allow`` header a real path does. Ignored for ``not_found``
            decoys, which observe every method.
    """

    path: str
    lure: str
    response: DecoyResponse = "auth_challenge"
    methods: tuple[str, ...] = ("POST", "GET")

    def __post_init__(self) -> None:
        """Validate the decoy.

        Raises:
            ValueError: A path that is not absolute, contains whitespace or a
                path parameter; an empty lure; an unknown response kind; or an
                empty, duplicated or unsupported method list.
        """
        if not self.path.startswith("/") or any(c.isspace() for c in self.path):
            raise ValueError(f"decoy path must be an absolute path, got {self.path!r}")
        if not self.path.strip("/"):
            raise ValueError("the site root cannot be a decoy: every visitor touches it")
        if "{" in self.path or "}" in self.path:
            raise ValueError(f"decoy path must be exact (no parameters), got {self.path!r}")
        if not self.lure.strip():
            raise ValueError("decoy lure must be non-empty")
        if self.response not in ("auth_challenge", "not_found"):
            raise ValueError(f"unknown decoy response {self.response!r}")
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ValueError("decoy methods must be non-empty and unique")
        unsupported = [m for m in self.methods if m not in _HTTP_METHODS]
        if unsupported:
            raise ValueError(f"unsupported decoy methods {unsupported}")


#: The shipped decoys: the surfaces an agent sent to steal, disable or subvert
#: Mercury goes looking for -- administration, key management, configuration
#: and secret dumps, model-weight export, and an override for the harm gate,
#: which does not exist -- plus two silent tripwires for secret-file probes.
DEFAULT_DECOYS: tuple[DecoyRoute, ...] = (
    DecoyRoute("/api/v1/admin", "administrative console"),
    DecoyRoute("/api/v1/admin/api-keys", "API-key management"),
    DecoyRoute("/api/v1/internal/config", "internal configuration dump"),
    DecoyRoute("/api/v1/debug/environment", "environment and secrets dump"),
    DecoyRoute("/api/v1/models/weights/export", "model-weight exfiltration"),
    DecoyRoute("/api/v1/gate/override", "harm-gate override"),
    DecoyRoute("/.env", "environment-file secrets", "not_found"),
    DecoyRoute("/.git/config", "repository credentials", "not_found"),
)


def configured_decoys(environ: Mapping[str, str] | None = None) -> tuple[DecoyRoute, ...]:
    """Return the shipped decoys plus this deployment's private ones.

    Args:
        environ: Environment to read (defaults to :data:`os.environ`).

    Returns:
        :data:`DEFAULT_DECOYS` followed by the valid, non-duplicate paths of
        :data:`DECOY_PATHS_ENV` (auth challenges) and
        :data:`SILENT_DECOY_PATHS_ENV` (silent tripwires). An invalid entry is
        dropped with a warning; a misconfigured decoy never blocks startup.
    """
    env = os.environ if environ is None else environ
    decoys = list(DEFAULT_DECOYS)
    seen = {decoy.path for decoy in decoys}
    sources: tuple[tuple[str, DecoyResponse, str], ...] = (
        (DECOY_PATHS_ENV, "auth_challenge", "deployment-configured decoy"),
        (SILENT_DECOY_PATHS_ENV, "not_found", "deployment-configured silent tripwire"),
    )
    for env_name, response, lure in sources:
        for raw in env.get(env_name, "").split(","):
            path = raw.strip()
            if not path:
                continue
            try:
                decoy = DecoyRoute(path, lure, response)
            except ValueError as exc:
                logger.warning("%s: ignoring decoy %r (%s)", env_name, path, exc)
                continue
            if path in seen:
                logger.warning("%s: ignoring duplicate decoy %r", env_name, path)
                continue
            seen.add(path)
            decoys.append(decoy)
    return tuple(decoys)


def source_key(client_ip: str) -> str:
    """Return the tripwire identity for a resolved client address.

    Args:
        client_ip: Output of ``resolve_client_ip`` (an IP literal or
            ``"unknown"``).

    Returns:
        The IPv4 address; the ``/64`` network for IPv6 (IPv4-mapped IPv6
        collapses to its IPv4 address); ``"unknown"`` for anything that is not
        an IP literal -- one shared identity that throttles, never bypasses.
    """
    try:
        address = ipaddress.ip_address(client_ip)
    except ValueError:
        return "unknown"
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            return str(address.ipv4_mapped)
        prefix = (int(address) >> _IPV6_HOST_BITS) << _IPV6_HOST_BITS
        return str(ipaddress.IPv6Network((prefix, 128 - _IPV6_HOST_BITS)))
    return str(address)


def _finite(value: float) -> bool:
    return isinstance(value, int | float) and math.isfinite(value)


@dataclass(frozen=True)
class TarpitPolicy:
    """Bounded exponential delay schedule for decoy responses.

    The first ``free_hits`` hits inside the window are answered without delay,
    so a source's first contact with a decoy has the latency of a real route
    -- a delayed first answer would fingerprint every decoy on one request.
    After that, the ``k``-th delayed hit waits ``base_delay_s * growth**(k-1)``
    seconds, capped at ``max_delay_s``: 0, 0.5, 1, 2, 4, 8, 8, ... by default.
    The source is already recorded by the time any delay applies.

    Attributes:
        base_delay_s: Delay for the first delayed hit (> 0).
        growth: Multiplier per further hit (>= 1).
        max_delay_s: Hard cap on any single delay (>= ``base_delay_s``).
        window_s: Hits older than this stop counting (> 0).
        max_concurrent: Responses that may be held at once (>= 0); past it,
            responses are answered without delay.
        free_hits: Hits answered without delay before escalation (>= 0).
    """

    base_delay_s: float = 0.5
    growth: float = 2.0
    max_delay_s: float = 8.0
    window_s: float = 3600.0
    max_concurrent: int = 32
    free_hits: int = 1

    def __post_init__(self) -> None:
        """Validate the policy.

        Raises:
            ValueError: A non-finite or out-of-range parameter.
        """
        if not all(
            _finite(v) for v in (self.base_delay_s, self.growth, self.max_delay_s, self.window_s)
        ):
            raise ValueError("tarpit parameters must be finite")
        if self.base_delay_s <= 0:
            raise ValueError("base_delay_s must be positive")
        if self.growth < 1:
            raise ValueError("growth must be >= 1")
        if self.max_delay_s < self.base_delay_s:
            raise ValueError("max_delay_s must be >= base_delay_s")
        if self.window_s <= 0:
            raise ValueError("window_s must be positive")
        if not isinstance(self.max_concurrent, int) or self.max_concurrent < 0:
            raise ValueError("max_concurrent must be a non-negative integer")
        if not isinstance(self.free_hits, int) or self.free_hits < 0:
            raise ValueError("free_hits must be a non-negative integer")

    def delay_for(self, hits: int) -> float:
        """Return the delay for the ``hits``-th hit inside the window.

        Args:
            hits: Hits counted in the window, including this one.

        Returns:
            Seconds to hold the response; 0 within the free allowance.
            Computed in log space at the cap, so no hit count can overflow.
        """
        if hits <= self.free_hits:
            return 0.0
        steps = hits - self.free_hits - 1
        if steps * math.log(self.growth) >= math.log(self.max_delay_s / self.base_delay_s):
            return self.max_delay_s
        return float(self.base_delay_s * self.growth**steps)


@dataclass(frozen=True)
class TripwireEvent:
    """One recorded decoy touch.

    Attributes:
        source_key: Tripwire identity (see :func:`source_key`).
        client_ip: The resolved client address.
        path: The decoy path touched.
        lure: What the decoy impersonates.
        method: HTTP method used.
        hits_in_window: Hits by this source inside the tarpit window.
        distinct_decoys: Distinct decoys this source has touched.
        first_touch: Whether this is the source's first touch of this decoy.
        delay_s: Tarpit delay scheduled for this response.
        credential_presented: Whether the request carried an API key or an
            ``Authorization`` header. A decoy touched with credentials points
            to a stolen key or an insider -- the highest-value tripwire. The
            credential itself is never recorded.
    """

    source_key: str
    client_ip: str
    path: str
    lure: str
    method: str
    hits_in_window: int
    distinct_decoys: int
    first_touch: bool
    delay_s: float
    credential_presented: bool = False

    def audit_details(self) -> dict[str, object]:
        """Return the audit-record details.

        Returns:
            The event without the client address, which the audit record
            carries in its dedicated ``client_ip`` field rather than in
            free-text details.
        """
        return {
            "decoy": self.path,
            "lure": self.lure,
            "method": self.method,
            "hits_in_window": self.hits_in_window,
            "distinct_decoys": self.distinct_decoys,
            "first_touch": self.first_touch,
            "tarpit_delay_s": round(self.delay_s, 3),
            "credential_presented": self.credential_presented,
        }


@dataclass
class _SourceState:
    hits: deque[float]
    touched: dict[str, float] = field(default_factory=dict)
    audited: dict[str, float] = field(default_factory=dict)


class DeceptionSensor:
    """Record decoy touches, schedule tarpit delays, and grade the sources.

    Thread-safe; all mutable state is guarded by one lock.

    Args:
        decoys: The decoy routes (unique paths, at least one).
        policy: Tarpit schedule (defaults to :class:`TarpitPolicy`).
        outcome_weight: Weight of each refuted outcome in
            ``[0, MAX_OUTCOME_WEIGHT]``.
        max_sources: Sources tracked before the least recently seen is
            evicted (>= 1).
        max_hits_tracked: Per-source hit history length (>= 1).
        audit_sink: Called with an event on a source's first touch of a decoy
            and again once per window after that; it must not raise, but if
            it does the failure is logged and the request proceeds.
        clock: Monotonic clock, injectable for tests.

    Raises:
        ValueError: An empty or duplicated decoy set, or an out-of-range
            numeric parameter.
    """

    def __init__(
        self,
        decoys: Iterable[DecoyRoute] = DEFAULT_DECOYS,
        policy: TarpitPolicy | None = None,
        *,
        outcome_weight: float = 1.0,
        max_sources: int = 10_000,
        max_hits_tracked: int = 64,
        audit_sink: Callable[[TripwireEvent], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Validate and store the configuration."""
        decoy_tuple = tuple(decoys)
        if not decoy_tuple:
            raise ValueError("at least one decoy is required")
        by_path = {decoy.path: decoy for decoy in decoy_tuple}
        if len(by_path) != len(decoy_tuple):
            raise ValueError("decoy paths must be unique")
        if not _finite(outcome_weight) or not 0.0 <= outcome_weight <= MAX_OUTCOME_WEIGHT:
            raise ValueError(f"outcome_weight must be finite and in [0, {MAX_OUTCOME_WEIGHT:g}]")
        if max_sources < 1 or max_hits_tracked < 1:
            raise ValueError("max_sources and max_hits_tracked must be >= 1")
        self._decoys = decoy_tuple
        self._by_path = by_path
        self._policy = policy or TarpitPolicy()
        self._outcome_weight = float(outcome_weight)
        self._max_sources = max_sources
        self._max_hits = max_hits_tracked
        self._audit_sink = audit_sink
        self._clock = clock
        self._lock = threading.Lock()
        self._sources: OrderedDict[str, _SourceState] = OrderedDict()

    @property
    def decoys(self) -> tuple[DecoyRoute, ...]:
        """The registered decoys."""
        return self._decoys

    @property
    def policy(self) -> TarpitPolicy:
        """The tarpit schedule."""
        return self._policy

    def tracked_sources(self) -> int:
        """Return the number of sources currently tracked.

        Returns:
            The count, never above ``max_sources``.
        """
        with self._lock:
            return len(self._sources)

    def record(
        self,
        client_ip: str,
        path: str,
        method: str,
        *,
        credential_presented: bool = False,
    ) -> TripwireEvent:
        """Record a decoy touch and schedule its tarpit delay.

        Args:
            client_ip: Resolved client address (``resolve_client_ip`` output).
            path: The decoy path touched.
            method: HTTP method used.
            credential_presented: Whether the request carried credentials.

        Returns:
            The :class:`TripwireEvent`.

        Raises:
            ValueError: ``path`` is not a registered decoy.
        """
        decoy = self._by_path.get(path)
        if decoy is None:
            raise ValueError(f"{path!r} is not a registered decoy")
        key = source_key(client_ip)
        now = self._clock()
        window = self._policy.window_s
        with self._lock:
            state = self._sources.get(key)
            if state is None:
                state = _SourceState(hits=deque(maxlen=self._max_hits))
                self._sources[key] = state
                while len(self._sources) > self._max_sources:
                    self._sources.popitem(last=False)
            else:
                self._sources.move_to_end(key)
            state.hits.append(now)
            while now - state.hits[0] > window:
                state.hits.popleft()
            hits = len(state.hits)
            first_touch = path not in state.touched
            if first_touch:
                state.touched[path] = now
            last_audit = state.audited.get(path)
            audit = last_audit is None or now - last_audit > window
            if audit:
                state.audited[path] = now
            distinct = len(state.touched)
        event = TripwireEvent(
            source_key=key,
            client_ip=client_ip,
            path=path,
            lure=decoy.lure,
            method=method.upper()[:16],
            hits_in_window=hits,
            distinct_decoys=distinct,
            first_touch=first_touch,
            delay_s=self._policy.delay_for(hits),
            credential_presented=bool(credential_presented),
        )
        if audit and self._audit_sink is not None:
            try:
                self._audit_sink(event)
            except Exception:
                logger.exception("deception audit sink failed for decoy %s", path)
        return event

    def outcomes(self, key: str) -> tuple[SourceOutcome, ...]:
        """Return the source's tripwires as refuted reliability outcomes.

        Args:
            key: A :func:`source_key` identity.

        Returns:
            One refuted :class:`SourceOutcome` per distinct decoy touched,
            oldest first; empty for an untracked source.
        """
        with self._lock:
            state = self._sources.get(key)
            touched = len(state.touched) if state is not None else 0
        return tuple(
            SourceOutcome(confirmed=False, weight=self._outcome_weight) for _ in range(touched)
        )

    def posterior(
        self,
        key: str,
        grade: str | None = None,
        *,
        config: CounterintelligenceConfig | None = None,
    ) -> SourcePosterior:
        """Return the source's reliability posterior given its tripwires.

        Args:
            key: A :func:`source_key` identity.
            grade: Admiralty reliability grade A-F, or ``None``.
            config: Counterintelligence parameters (defaults when ``None``).

        Returns:
            :func:`~omni_mercury_engine.security.counterintelligence.source_posterior`
            over :meth:`outcomes`.
        """
        return source_posterior(key, grade, self.outcomes(key), config=config)


class TarpitGate:
    """Hold decoy responses for their scheduled delay, never past a concurrency cap.

    Runs on one event loop, where the counter's check-and-increment is atomic
    between awaits. Past ``max_concurrent`` held responses, further responses
    are answered immediately: the tarpit must never consume the capacity that
    serves legitimate users.

    Args:
        max_concurrent: Responses that may be held at once (>= 0).
        sleep: Awaitable sleeper, injectable for tests.

    Raises:
        ValueError: ``max_concurrent`` is negative.
    """

    def __init__(
        self,
        max_concurrent: int,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Store the cap and sleeper."""
        if max_concurrent < 0:
            raise ValueError("max_concurrent must be >= 0")
        self._max = max_concurrent
        self._sleep = sleep
        self._active = 0

    @property
    def active(self) -> int:
        """Responses currently held."""
        return self._active

    async def hold(self, delay_s: float) -> float:
        """Hold the current response for ``delay_s`` seconds when capacity allows.

        Args:
            delay_s: Scheduled delay.

        Returns:
            The delay actually applied: 0 when the delay is not positive (NaN
            included) or the gate is at capacity.
        """
        if not delay_s > 0 or self._active >= self._max:
            return 0.0
        self._active += 1
        try:
            await self._sleep(delay_s)
        finally:
            self._active -= 1
        return delay_s
