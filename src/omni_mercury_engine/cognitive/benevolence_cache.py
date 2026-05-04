# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""LRU-bounded benevolence-decision cache.

Wraps :class:`omni_mercury_engine.cognitive.ethical_bounding.BenevolenceScorer`
so that repeated identical ``enforce(action, context)`` calls return the
already-computed :class:`EthicalScore` from a bounded LRU cache instead of
re-running the full scoring pipeline.

Three invariants are pinned by ``tests/ethical/test_benevolence_cache.py``:

1. **Ruleset-version invalidation.** Every cache key is prefixed with
   :attr:`omni_mercury_engine.core.centralized_constants.centralized_constants.ETHICAL.RULESET_VERSION`.
   When that version is bumped (or monkey-patched in tests), keys produced
   under the new version cannot collide with old entries, and the cache also
   actively purges stale-version entries on the first lookup after a bump.

2. **Identical input is a cache hit.** Two ``enforce`` calls with bytewise
   equal canonicalised payloads return the same :class:`EthicalScore`
   instance and increment ``stats["hits"]``.

3. **Violations are never cached.** When the underlying scorer raises
   :class:`EthicalConstraintViolationError`, the wrapper re-raises and does
   *not* store anything for that key — every subsequent call with the same
   input recomputes, so a transient ruleset/data fix that flips an action
   from impermissible back to permissible is observed immediately.

The wrapper exposes the same call signature as ``BenevolenceScorer.enforce``,
so it is a drop-in replacement at the boundary call site.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from typing import Any

from omni_mercury_engine.cognitive.ethical_bounding import (
    BenevolenceScorer,
    EthicalConstraintViolationError,
    EthicalScore,
)

# Imported as a module so a deployment / test that swaps the ETHICAL
# singleton (frozen dataclass) is observed by the cache on the next lookup.
from omni_mercury_engine.core import centralized_constants

logger = logging.getLogger(__name__)


DEFAULT_CACHE_CAPACITY: int = 1024


def _canonicalise(action: str, context: dict[str, Any]) -> str:
    """Return a deterministic JSON serialisation of ``(action, context)``."""
    return json.dumps(
        {"action": action, "context": context},
        sort_keys=True,
        default=str,  # accept datetime / Path / UUID etc. without crashing
        separators=(",", ":"),
    )


def _hash_payload(payload: str) -> str:
    """Stable, collision-resistant 128-bit SHA3-256 (FIPS 202) digest.

    Truncated to 16 bytes (128 bits) for cache-key compactness; the
    full SHA3-256 algorithm is the same one pinned as the content-hash
    standard by Mercury's AMA Cryptography surface
    (``security/crypto_api.py::CryptoPackageConfig.hash_algorithm``),
    so cache keys hash-align with on-the-wire AMA payload hashes for
    the same canonicalised input.
    """
    return hashlib.sha3_256(payload.encode("utf-8")).hexdigest()[:32]


class CachedBenevolenceScorer:
    """Thread-safe LRU wrapper around :meth:`BenevolenceScorer.enforce`."""

    def __init__(
        self,
        scorer: BenevolenceScorer | None = None,
        capacity: int = DEFAULT_CACHE_CAPACITY,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")

        self._scorer = scorer if scorer is not None else BenevolenceScorer()
        self._capacity = capacity
        self._cache: OrderedDict[tuple[int, str], EthicalScore] = OrderedDict()
        self._lock = threading.Lock()

        self._hits = 0
        self._misses = 0
        self._violations_uncached = 0
        # Track the last ruleset version we saw so we can purge stale-version
        # entries when the constant is bumped (e.g. by an operator deploying
        # a new ruleset, or by a test monkey-patching the value).
        self._last_seen_ruleset_version: int = centralized_constants.ETHICAL.RULESET_VERSION

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def underlying_scorer(self) -> BenevolenceScorer:
        return self._scorer

    @property
    def benevolence_threshold(self) -> float:
        """Pass-through for callers that introspect the gate threshold."""
        return self._scorer.benevolence_threshold

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "violations_uncached": self._violations_uncached,
                "size": len(self._cache),
                "capacity": self._capacity,
                "ruleset_version": self._last_seen_ruleset_version,
            }

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Drop every cached decision."""
        with self._lock:
            self._cache.clear()

    def _purge_stale_version_entries_locked(self, current_version: int) -> None:
        """Drop entries whose key prefix doesn't match the current version.

        Called under ``self._lock``.
        """
        if current_version == self._last_seen_ruleset_version and all(
            k[0] == current_version for k in self._cache
        ):
            return
        before = len(self._cache)
        self._cache = OrderedDict((k, v) for k, v in self._cache.items() if k[0] == current_version)
        purged = before - len(self._cache)
        if purged:
            logger.info(
                "Benevolence cache: purged %d stale entries on ruleset version "
                "transition %d → %d",
                purged,
                self._last_seen_ruleset_version,
                current_version,
            )
        self._last_seen_ruleset_version = current_version

    def _make_key(
        self, action: str, context: dict[str, Any], ruleset_version: int
    ) -> tuple[int, str]:
        return (ruleset_version, _hash_payload(_canonicalise(action, context)))

    # ------------------------------------------------------------------
    # Drop-in scorer surface
    # ------------------------------------------------------------------
    def enforce(self, action: str, context: dict[str, Any]) -> EthicalScore:
        """Cache-aware ``enforce``: hit returns stored score; miss runs scorer.

        Violations propagate :class:`EthicalConstraintViolationError` and are
        never inserted into the cache.
        """
        # Read the version once per call so a concurrent bump can't split a
        # single call between two version regimes.
        current_version = centralized_constants.ETHICAL.RULESET_VERSION
        key = self._make_key(action, context, current_version)

        with self._lock:
            self._purge_stale_version_entries_locked(current_version)
            cached = self._cache.get(key)
            if cached is not None:
                # LRU touch.
                self._cache.move_to_end(key)
                self._hits += 1
                return cached

        # Cache miss: call the underlying scorer outside the lock so a slow
        # scoring run can't serialise other lookups.
        try:
            result = self._scorer.enforce(action, context)
        except EthicalConstraintViolationError:
            with self._lock:
                self._violations_uncached += 1
            raise

        with self._lock:
            # Re-check under lock in case a concurrent caller already inserted.
            if key not in self._cache:
                self._cache[key] = result
                self._cache.move_to_end(key)
                if len(self._cache) > self._capacity:
                    self._cache.popitem(last=False)
            self._misses += 1
        return result

    def score_action(self, action: str, context: dict[str, Any]) -> EthicalScore:
        """Pass-through to the underlying scorer (advisory, not cached).

        ``score_action`` is the advisory variant that returns
        ``is_permissible=False`` rather than raising. Caching it would
        require a second key dimension (permissible vs not) and offers
        little payoff because callers that want enforcement use
        :meth:`enforce`. Pass-through keeps the wrapper a strict superset
        of the BenevolenceScorer surface without surprise semantics.
        """
        return self._scorer.score_action(action, context)


__all__ = [
    "DEFAULT_CACHE_CAPACITY",
    "CachedBenevolenceScorer",
]
