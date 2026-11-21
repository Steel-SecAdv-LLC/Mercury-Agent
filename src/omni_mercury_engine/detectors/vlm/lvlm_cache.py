# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""LVLM Backend Cache with Pre-warming and Resource Management.

Provides:
- Singleton model cache to avoid redundant loading
- Pre-warming for latency-critical applications
- Memory management with LRU eviction
- Thread-safe concurrent access
- Health monitoring and auto-recovery

Ownership: the cache SHARES backends, it does not own them.  ``get`` hands
the same backend object to every caller, and detectors keep that reference
for their lifetime.  Evicting an entry therefore only drops the cache's own
reference -- the backend object is never modified -- and the model's memory
is reclaimed by the garbage collector when the last holder lets go.  (An
eviction used to delete ``model`` / ``processor`` from the shared object,
which broke every detector still holding it.)

Performance characteristics:
- First load: Full model initialization time
- Subsequent uses: Near-zero latency (cached reference)
- Pre-warmed: Immediate availability
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

from .lvlm_backends import LVLMBackend, get_lvlm_backend

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class ModelState(Enum):
    """State of a cached model."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    WARMING = "warming"
    ERROR = "error"
    EVICTED = "evicted"


@dataclass
class CachedModel:
    """Container for a cached LVLM backend."""

    model_type: str
    model_name: str
    backend: LVLMBackend | None = None
    state: ModelState = ModelState.UNLOADED
    load_time: float = 0.0
    last_used: float = 0.0
    use_count: int = 0
    error_count: int = 0
    last_error: str | None = None
    memory_bytes: int = 0
    warmup_complete: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class CacheStatistics:
    """Statistics for the model cache."""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    errors: int = 0
    total_load_time: float = 0.0
    models_loaded: int = 0
    memory_used_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests


class LVLMBackendCache:
    """Thread-safe singleton cache for LVLM backends.

    Features:
    - LRU eviction when memory limit exceeded
    - Background pre-warming
    - Health monitoring
    - Automatic error recovery

    ``max_models`` / ``max_memory_gb`` bound what the CACHE keeps alive.  An
    evicted backend that a detector still holds stays loaded (and usable)
    until that detector releases it; ``memory_used_bytes`` counts only the
    entries the cache itself still references.
    """

    _instance: LVLMBackendCache | None = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> LVLMBackendCache:
        """Singleton pattern implementation."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(
        self,
        max_memory_gb: float = 16.0,
        max_models: int = 3,
        warmup_timeout: float = 300.0,
        health_check_interval: float = 60.0,
    ):
        """Initialize the cache.

        Args:
            max_memory_gb: Maximum GPU memory to use (approximate)
            max_models: Maximum number of models to keep loaded
            warmup_timeout: Timeout for model warmup in seconds
            health_check_interval: Interval between health checks
        """
        if getattr(self, "_initialized", False):
            return

        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        self.max_models = max_models
        self.warmup_timeout = warmup_timeout
        self.health_check_interval = health_check_interval

        self._cache: dict[str, CachedModel] = {}
        self._cache_lock = threading.RLock()
        self._stats = CacheStatistics()

        # Background thread pool for warmup and health checks
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="lvlm_cache")
        self._warmup_futures: dict[str, Future[bool]] = {}

        # Health monitoring
        self._health_check_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()

        # Callbacks
        self._on_load_callbacks: list[Callable[[str, float], None]] = []
        self._on_evict_callbacks: list[Callable[[str], None]] = []
        self._on_error_callbacks: list[Callable[[str, str], None]] = []

        # Register cleanup
        atexit.register(self.shutdown)

        self._initialized = True
        logger.info(
            f"LVLMBackendCache initialized: max_memory={max_memory_gb}GB, "
            f"max_models={max_models}"
        )

    @classmethod
    def get_instance(cls) -> LVLMBackendCache:
        """Get the singleton cache instance."""
        return cls()

    def _cache_key(
        self,
        model_type: str,
        model_name: str | None = None,
        revision: str | None = None,
    ) -> str:
        """Generate cache key for a model.

        ``revision`` is part of the security-relevant identity of the
        backend: two callers asking for the same ``model_name`` at
        different commit SHAs MUST receive different backend instances,
        otherwise an LRU hit on the older entry silently defeats the
        caller's pin. We render ``None`` as the literal ``"none"``
        rather than the empty string so a local-path load (revision
        ``None``) and a Hub load that someone accidentally tried to
        pin to ``""`` do not collide.

        The ``revision`` rendering uses an explicit ``is None`` check
        rather than ``revision or 'none'`` so an empty-string revision
        (which ``HFModelPolicy.validate`` will reject downstream, but
        which can transit through kwargs unobserved) does NOT collide
        with the legitimate no-revision / local-path key.
        """
        rev_token = "none" if revision is None else revision
        return f"{model_type}:{model_name or model_type}@{rev_token}"

    def get(
        self,
        model_type: str,
        model_name: str | None = None,
        device: str = "cuda",
        **kwargs: Any,
    ) -> LVLMBackend:
        """Get a cached LVLM backend, loading if necessary.

        Args:
            model_type: Type of LVLM backend
            model_name: HuggingFace model identifier
            device: Computation device
            **kwargs: Additional backend arguments

        Returns:
            Initialized LVLM backend

        Raises:
            RuntimeError: If model loading fails
        """
        # ``revision`` is part of the cache identity (see ``_cache_key``).
        # Inspect kwargs without consuming so it still threads through
        # to ``get_lvlm_backend`` -> the concrete backend constructor.
        revision = kwargs.get("revision")
        key = self._cache_key(model_type, model_name, revision)

        loading: CachedModel | None = None
        with self._cache_lock:
            self._stats.total_requests += 1

            if key in self._cache:
                cached = self._cache[key]

                if cached.state == ModelState.READY and cached.backend is not None:
                    self._stats.cache_hits += 1
                    cached.last_used = time.time()
                    cached.use_count += 1
                    logger.debug(f"Cache hit for {key}")
                    return cached.backend

                if cached.state == ModelState.LOADING:
                    self._stats.cache_misses += 1
                    loading = cached

                elif cached.state == ModelState.ERROR:
                    # Retry loading
                    self._stats.cache_misses += 1
                    cached.state = ModelState.UNLOADED

            if loading is None:
                self._stats.cache_misses += 1

        if loading is not None:
            # Wait for the load in progress -- with the cache lock RELEASED.
            # The loading thread publishes its entry under that lock; waiting
            # for it while holding the lock (as this did) deadlocked both
            # threads, e.g. a ``prewarm`` racing the first ``get``.
            return self._wait_for_load(loading, key)

        # Load model (outside lock to allow concurrent loads).  Pass
        # ``revision`` through so the loader and cache key agree.
        return self._load_model(model_type, model_name, device, **kwargs)

    def touch(self, backend: Any) -> bool:
        """Record a use of ``backend`` so LRU eviction follows use, not fetch.

        ``get`` refreshes an entry only when a backend is fetched, but a
        detector fetches once and then calls its backend directly for as
        long as it lives -- so the busiest detector's entry looked like the
        least recently used one.  Holders call this each time they use the
        backend.

        Args:
            backend: A backend object previously returned by :meth:`get`.

        Returns:
            ``True`` if the cache still holds ``backend`` and refreshed its
            entry; ``False`` for an evicted, dedicated or foreign object
            (which is fine: there is nothing to refresh).
        """
        with self._cache_lock:
            for cached in self._cache.values():
                if cached.backend is backend and cached.state == ModelState.READY:
                    cached.last_used = time.time()
                    cached.use_count += 1
                    return True
        return False

    def _wait_for_load(self, cached: CachedModel, key: str) -> LVLMBackend:
        """Wait for a model that's currently loading."""
        with cached._lock:
            # Double-check state
            if cached.state == ModelState.READY and cached.backend is not None:
                return cached.backend

            # Wait for loading
            start_time = time.time()
            while cached.state == ModelState.LOADING:
                if time.time() - start_time > self.warmup_timeout:
                    raise RuntimeError(f"Timeout waiting for model {key} to load")
                time.sleep(0.1)

            if cached.state == ModelState.READY and cached.backend is not None:
                cached.last_used = time.time()
                cached.use_count += 1
                return cached.backend

            raise RuntimeError(f"Model {key} failed to load: {cached.last_error}")

    def _load_model(
        self,
        model_type: str,
        model_name: str | None,
        device: str,
        **kwargs: Any,
    ) -> LVLMBackend:
        """Load a model into the cache."""
        key = self._cache_key(model_type, model_name, kwargs.get("revision"))

        # Create cache entry
        with self._cache_lock:
            if key not in self._cache:
                self._cache[key] = CachedModel(
                    model_type=model_type,
                    model_name=model_name or model_type,
                )
            cached = self._cache[key]

        with cached._lock:
            if cached.state == ModelState.READY and cached.backend is not None:
                return cached.backend

            cached.state = ModelState.LOADING
            start_time = time.time()

            try:
                # Check memory and evict if needed
                self._ensure_memory_available()

                # Create and initialize backend
                backend = get_lvlm_backend(
                    model_type=model_type,
                    model_name=model_name,
                    device=device,
                    **kwargs,
                )
                backend.initialize()

                # Publish the entry and its statistics in one step under the
                # cache lock.  Evictions run under that lock and only touch
                # READY entries, so they never wait on the lock of an entry
                # that is still loading (which this thread holds).
                memory_bytes = self._estimate_memory(backend)
                with self._cache_lock:
                    cached.backend = backend
                    cached.load_time = time.time() - start_time
                    cached.last_used = time.time()
                    cached.use_count = 1
                    cached.error_count = 0
                    cached.memory_bytes = memory_bytes
                    cached.state = ModelState.READY
                    self._stats.total_load_time += cached.load_time
                    self._stats.models_loaded += 1
                    self._stats.memory_used_bytes += cached.memory_bytes

                # Notify callbacks
                for callback in self._on_load_callbacks:
                    try:
                        callback(key, cached.load_time)
                    except Exception as e:
                        logger.warning(f"Load callback error: {e}")

                logger.info(
                    f"Loaded model {key} in {cached.load_time:.2f}s "
                    f"(~{cached.memory_bytes / 1024**3:.1f}GB)"
                )

                return backend

            except Exception as e:
                cached.state = ModelState.ERROR
                cached.error_count += 1
                cached.last_error = str(e)

                self._stats.errors += 1

                for error_callback in self._on_error_callbacks:
                    try:
                        error_callback(key, str(e))
                    except Exception as cb_e:
                        logger.warning(f"Error callback error: {cb_e}")

                logger.error(f"Failed to load model {key}: {e}")
                if isinstance(e, NotImplementedError):
                    # The anti-mock contract (Phase 2 audit cure): a mock
                    # backend's hard-fail must keep its type through the
                    # cache so callers pinned on NotImplementedError see it.
                    raise
                raise RuntimeError(f"Failed to load model {key}: {e}") from e

    def _estimate_memory(self, backend: LVLMBackend) -> int:
        """Estimate GPU memory used by a backend."""
        if backend.model is None:
            return 0

        try:
            # Try to get actual memory from model parameters
            if hasattr(backend.model, "parameters"):
                param_bytes = sum(p.numel() * p.element_size() for p in backend.model.parameters())
                # Estimate optimizer states and activations
                return int(param_bytes * 2.5)

            # Fallback estimates based on model type
            model_name = backend.model_name.lower()
            if "7b" in model_name:
                return 14 * 1024**3  # ~14GB for 7B model
            elif "13b" in model_name:
                return 26 * 1024**3
            elif "70b" in model_name:
                return 140 * 1024**3
            else:
                return 8 * 1024**3  # Default 8GB

        except Exception as e:
            logger.debug(f"Failed to estimate model memory, using default 8GB: {e}")
            return 8 * 1024**3

    def _ensure_memory_available(self) -> None:
        """Ensure sufficient memory is available, evicting if necessary."""
        with self._cache_lock:
            # Check model count
            loaded_models = [c for c in self._cache.values() if c.state == ModelState.READY]

            while len(loaded_models) >= self.max_models:
                self._evict_lru()
                loaded_models = [c for c in self._cache.values() if c.state == ModelState.READY]

            # Check memory
            while self._stats.memory_used_bytes > self.max_memory_bytes:
                if not self._evict_lru():
                    break

    def _release_entry(self, key: str, cached: CachedModel, notify: bool = False) -> bool:
        """Drop the cache's reference to a READY entry.

        Only the cache's reference goes.  The backend object is left exactly
        as it is: detectors that fetched it keep a working backend, and its
        memory is reclaimed when the last of them lets go.  The caller holds
        ``_cache_lock``.

        Args:
            key: The entry's cache key.
            cached: The entry.
            notify: Whether to run the ``on_evict`` callbacks (LRU evictions
                do; manual ``evict`` / ``clear`` never did).

        Returns:
            ``True`` if the entry was READY and has been released.
        """
        # Checked before the entry's own lock is taken: an entry that is
        # still loading holds that lock for the whole load (and then wants
        # the cache lock, held here), so it must be skipped, not waited on.
        if cached.state != ModelState.READY:
            return False
        with cached._lock:
            if cached.state != ModelState.READY:
                return False

            cached.state = ModelState.EVICTED
            self._stats.memory_used_bytes -= cached.memory_bytes
            self._stats.evictions += 1
            cached.backend = None

        # Return cached GPU blocks to the driver if this was the last holder.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for callback in self._on_evict_callbacks if notify else ():
            try:
                callback(key)
            except Exception as e:
                logger.warning(f"Evict callback error: {e}")
        return True

    def _evict_lru(self) -> bool:
        """Evict the least recently used model (see :meth:`_release_entry`)."""
        with self._cache_lock:
            # Find LRU model
            lru_key = None
            lru_time = float("inf")

            for key, cached in self._cache.items():
                if cached.state == ModelState.READY and cached.last_used < lru_time:
                    lru_time = cached.last_used
                    lru_key = key

            if lru_key is None:
                return False

            if not self._release_entry(lru_key, self._cache[lru_key], notify=True):
                return False
            logger.info(f"Evicted model {lru_key}")
            return True

    def prewarm(
        self,
        model_type: str,
        model_name: str | None = None,
        device: str = "cuda",
        warmup_input: Any | None = None,
        **kwargs: Any,
    ) -> Future[bool]:
        """Pre-warm a model in the background.

        Args:
            model_type: Type of LVLM backend
            model_name: HuggingFace model identifier
            device: Computation device
            warmup_input: Optional input for warmup inference
            **kwargs: Additional backend arguments

        Returns:
            Future that resolves to True when warmup is complete
        """
        key = self._cache_key(model_type, model_name, kwargs.get("revision"))

        # Check if already warming or loaded
        with self._cache_lock:
            if key in self._warmup_futures:
                return self._warmup_futures[key]

            if key in self._cache:
                cached = self._cache[key]
                if cached.state == ModelState.READY:
                    # Already loaded, just do warmup inference
                    future: Future[bool] = Future()
                    future.set_result(True)
                    return future

        # Submit warmup task
        future = self._executor.submit(
            self._prewarm_task,
            model_type,
            model_name,
            device,
            warmup_input,
            kwargs,
        )

        with self._cache_lock:
            self._warmup_futures[key] = future

        return future

    def _prewarm_task(
        self,
        model_type: str,
        model_name: str | None,
        device: str,
        warmup_input: Any | None,
        kwargs: dict[str, Any],
    ) -> bool:
        """Background task for pre-warming a model."""
        key = self._cache_key(model_type, model_name, kwargs.get("revision"))

        try:
            # Load model
            backend = self.get(
                model_type=model_type,
                model_name=model_name,
                device=device,
                **kwargs,
            )

            # Perform warmup inference if input provided
            if warmup_input is not None:
                try:
                    from PIL import Image

                    # Create dummy input if needed
                    if warmup_input == "auto":
                        dummy_image = Image.new("RGB", (224, 224), color="gray")
                        warmup_input = [dummy_image]

                    backend.generate(warmup_input, "Describe this image briefly.")
                    logger.info(f"Warmup inference completed for {key}")
                except Exception as e:
                    logger.warning(f"Warmup inference failed for {key}: {e}")

            # Mark warmup complete
            with self._cache_lock:
                if key in self._cache:
                    self._cache[key].warmup_complete = True
                self._warmup_futures.pop(key, None)

            return True

        except Exception as e:
            logger.error(f"Prewarm failed for {key}: {e}")
            with self._cache_lock:
                self._warmup_futures.pop(key, None)
            return False

    def is_loaded(
        self,
        model_type: str,
        model_name: str | None = None,
        revision: str | None = None,
    ) -> bool:
        """Check if a model is currently loaded."""
        key = self._cache_key(model_type, model_name, revision)
        with self._cache_lock:
            if key in self._cache:
                return self._cache[key].state == ModelState.READY
            return False

    def is_warming(
        self,
        model_type: str,
        model_name: str | None = None,
        revision: str | None = None,
    ) -> bool:
        """Check if a model is currently pre-warming."""
        key = self._cache_key(model_type, model_name, revision)
        with self._cache_lock:
            return key in self._warmup_futures

    def get_stats(self) -> CacheStatistics:
        """Get cache statistics."""
        with self._cache_lock:
            return CacheStatistics(
                total_requests=self._stats.total_requests,
                cache_hits=self._stats.cache_hits,
                cache_misses=self._stats.cache_misses,
                evictions=self._stats.evictions,
                errors=self._stats.errors,
                total_load_time=self._stats.total_load_time,
                models_loaded=self._stats.models_loaded,
                memory_used_bytes=self._stats.memory_used_bytes,
            )

    def get_model_info(
        self,
        model_type: str,
        model_name: str | None = None,
        revision: str | None = None,
    ) -> dict[str, Any] | None:
        """Get information about a cached model."""
        key = self._cache_key(model_type, model_name, revision)
        with self._cache_lock:
            if key not in self._cache:
                return None

            cached = self._cache[key]
            return {
                "model_type": cached.model_type,
                "model_name": cached.model_name,
                "state": cached.state.value,
                "load_time": cached.load_time,
                "last_used": cached.last_used,
                "use_count": cached.use_count,
                "error_count": cached.error_count,
                "memory_gb": cached.memory_bytes / 1024**3,
                "warmup_complete": cached.warmup_complete,
            }

    def list_models(self) -> list[dict[str, Any]]:
        """List all cached models."""
        with self._cache_lock:
            return [
                {
                    "key": key,
                    "state": cached.state.value,
                    "use_count": cached.use_count,
                    "memory_gb": cached.memory_bytes / 1024**3,
                }
                for key, cached in self._cache.items()
            ]

    def evict(
        self,
        model_type: str,
        model_name: str | None = None,
        revision: str | None = None,
    ) -> bool:
        """Manually evict a model from the cache.

        Like LRU eviction this only drops the cache's reference: a detector
        still holding the backend keeps a working one.
        """
        key = self._cache_key(model_type, model_name, revision)

        # Same lock order as ``_evict_lru`` (cache lock, then the entry's).
        with self._cache_lock:
            if key not in self._cache:
                return False
            if not self._release_entry(key, self._cache[key]):
                return False

        logger.info(f"Manually evicted model {key}")
        return True

    def clear(self) -> None:
        """Clear all cached models (dropping the cache's references only)."""
        with self._cache_lock:
            for key, cached in list(self._cache.items()):
                self._release_entry(key, cached)
            self._cache.clear()
            self._stats = CacheStatistics()

        try:
            logger.info("Cleared all cached models")
        except (ValueError, OSError):
            pass  # stderr may be closed during interpreter shutdown

    def on_load(self, callback: Callable[[str, float], None]) -> None:
        """Register a callback for model load events."""
        self._on_load_callbacks.append(callback)

    def on_evict(self, callback: Callable[[str], None]) -> None:
        """Register a callback for model eviction events."""
        self._on_evict_callbacks.append(callback)

    def on_error(self, callback: Callable[[str, str], None]) -> None:
        """Register a callback for error events."""
        self._on_error_callbacks.append(callback)

    def start_health_monitoring(self) -> None:
        """Start background health monitoring thread."""
        if self._health_check_thread is not None:
            return

        self._shutdown_event.clear()
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            name="lvlm_cache_health",
            daemon=True,
        )
        self._health_check_thread.start()
        logger.info("Started health monitoring thread")

    def _health_check_loop(self) -> None:
        """Background loop for health checks."""
        while not self._shutdown_event.wait(self.health_check_interval):
            try:
                self._perform_health_check()
            except Exception as e:
                logger.error(f"Health check error: {e}")

    def _perform_health_check(self) -> None:
        """Perform health check on all loaded models."""
        with self._cache_lock:
            loaded = [
                (key, cached)
                for key, cached in self._cache.items()
                if cached.state == ModelState.READY
            ]

        for key, cached in loaded:
            try:
                # Simple check: verify model is still accessible
                if cached.backend is None or cached.backend.model is None:
                    logger.warning(f"Model {key} appears to be in invalid state")
                    with cached._lock:
                        cached.state = ModelState.ERROR
                        cached.last_error = "Model reference lost"
            except Exception as e:
                logger.warning(f"Health check failed for {key}: {e}")

    def shutdown(self) -> None:
        """Shutdown the cache and release all resources."""
        try:
            logger.info("Shutting down LVLMBackendCache")
        except (ValueError, OSError):
            pass  # stderr may be closed during interpreter shutdown

        # Stop health monitoring
        self._shutdown_event.set()
        if self._health_check_thread is not None:
            self._health_check_thread.join(timeout=5.0)
            self._health_check_thread = None

        # Cancel pending warmup tasks
        with self._cache_lock:
            for future in self._warmup_futures.values():
                future.cancel()
            self._warmup_futures.clear()

        # Shutdown executor
        self._executor.shutdown(wait=False)

        # Clear cache
        self.clear()


# Convenience functions
def get_cached_backend(
    model_type: str,
    model_name: str | None = None,
    device: str = "cuda",
    **kwargs: Any,
) -> LVLMBackend:
    """Get a cached LVLM backend (convenience function).

    Args:
        model_type: Type of LVLM backend
        model_name: HuggingFace model identifier
        device: Computation device
        **kwargs: Additional backend arguments

    Returns:
        Initialized LVLM backend
    """
    cache = LVLMBackendCache.get_instance()
    return cache.get(model_type, model_name, device, **kwargs)


def touch_cached_backend(backend: Any) -> bool:
    """Record a use of a cached backend (see :meth:`LVLMBackendCache.touch`).

    A no-op returning ``False`` when no cache exists yet or ``backend`` is
    not one of its entries, so callers can touch whatever backend they hold
    -- cached, dedicated or a test double -- without checking.

    Args:
        backend: The backend object about to be used.

    Returns:
        Whether a cache entry was refreshed.
    """
    cache = LVLMBackendCache._instance
    if cache is None or not getattr(cache, "_initialized", False):
        return False
    return cache.touch(backend)


def get_frozen_cached_backend(
    model_type: str,
    model_name: str | None = None,
    device: str = "cuda",
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    revision: str | None = None,
) -> LVLMBackend:
    """Reuse a cached backend only when its frozen configuration matches.

    The cache key covers ``model_type:model_name@revision`` but not the
    device or the generation settings, so a plain cache hit could hand a
    detector a backend built with another instance's ``max_new_tokens`` /
    ``temperature`` / device — leaking configuration across instances.
    This wrapper reuses the cached backend only when every one of those
    frozen attributes matches the request exactly; otherwise it builds and
    initializes a dedicated, uncached backend with the requested settings.

    Args:
        model_type: Type of LVLM backend.
        model_name: HuggingFace model identifier (``None`` for the
            backend's shipped default).
        device: Computation device.
        max_new_tokens: Maximum generation tokens the caller requires.
        temperature: Sampling temperature the caller requires.
        revision: Pinned revision, part of the cache identity.

    Returns:
        An initialized LVLM backend honoring the requested configuration.
    """
    backend = get_cached_backend(
        model_type,
        model_name,
        device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        revision=revision,
    )
    matches = (
        (model_name is None or backend.model_name == model_name)
        and backend.device == torch.device(device)
        and backend.max_new_tokens == max_new_tokens
        and backend.temperature == temperature
        and backend.revision == revision
    )
    if matches:
        return backend

    dedicated = get_lvlm_backend(
        model_type=model_type,
        model_name=model_name,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        revision=revision,
    )
    dedicated.initialize()
    return dedicated


def prewarm_backend(
    model_type: str,
    model_name: str | None = None,
    device: str = "cuda",
    warmup_input: Any | None = "auto",
    **kwargs: Any,
) -> Future[bool]:
    """Pre-warm an LVLM backend (convenience function).

    Args:
        model_type: Type of LVLM backend
        model_name: HuggingFace model identifier
        device: Computation device
        warmup_input: Input for warmup inference ("auto" for dummy input)
        **kwargs: Additional backend arguments

    Returns:
        Future that resolves when warmup is complete
    """
    cache = LVLMBackendCache.get_instance()
    return cache.prewarm(model_type, model_name, device, warmup_input, **kwargs)
