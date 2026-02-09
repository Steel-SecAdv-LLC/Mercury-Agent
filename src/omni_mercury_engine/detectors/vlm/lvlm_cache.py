"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
LVLM Backend Cache with Pre-warming and Resource Management.

Provides:
- Singleton model cache to avoid redundant loading
- Pre-warming for latency-critical applications
- Memory management with LRU eviction
- Thread-safe concurrent access
- Health monitoring and auto-recovery

Performance characteristics:
- First load: Full model initialization time
- Subsequent uses: Near-zero latency (cached reference)
- Pre-warmed: Immediate availability
"""

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
    """
    Thread-safe singleton cache for LVLM backends.

    Features:
    - LRU eviction when memory limit exceeded
    - Background pre-warming
    - Health monitoring
    - Automatic error recovery
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
        """
        Initialize the cache.

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

    def _cache_key(self, model_type: str, model_name: str | None = None) -> str:
        """Generate cache key for a model."""
        return f"{model_type}:{model_name or model_type}"

    def get(
        self,
        model_type: str,
        model_name: str | None = None,
        device: str = "cuda",
        **kwargs: Any,
    ) -> LVLMBackend:
        """
        Get a cached LVLM backend, loading if necessary.

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
        key = self._cache_key(model_type, model_name)

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
                    # Wait for loading to complete
                    self._stats.cache_misses += 1
                    return self._wait_for_load(cached, key)

                if cached.state == ModelState.ERROR:
                    # Retry loading
                    self._stats.cache_misses += 1
                    cached.state = ModelState.UNLOADED

            self._stats.cache_misses += 1

        # Load model (outside lock to allow concurrent loads)
        return self._load_model(model_type, model_name, device, **kwargs)

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
        key = self._cache_key(model_type, model_name)

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

                # Update cache entry
                cached.backend = backend
                cached.state = ModelState.READY
                cached.load_time = time.time() - start_time
                cached.last_used = time.time()
                cached.use_count = 1
                cached.error_count = 0
                cached.memory_bytes = self._estimate_memory(backend)

                # Update statistics
                with self._cache_lock:
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

    def _evict_lru(self) -> bool:
        """Evict the least recently used model."""
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

            cached = self._cache[lru_key]

            with cached._lock:
                if cached.state != ModelState.READY:
                    return False

                # Evict
                cached.state = ModelState.EVICTED
                self._stats.memory_used_bytes -= cached.memory_bytes
                self._stats.evictions += 1

                # Clear model reference
                if cached.backend is not None:
                    try:
                        del cached.backend.model
                        del cached.backend.processor
                    except Exception as e:
                        logger.debug(
                            f"Failed to delete model/processor during eviction for {lru_key}: {e}"
                        )
                    cached.backend = None

                # Force garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                for callback in self._on_evict_callbacks:
                    try:
                        callback(lru_key)
                    except Exception as e:
                        logger.warning(f"Evict callback error: {e}")

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
        """
        Pre-warm a model in the background.

        Args:
            model_type: Type of LVLM backend
            model_name: HuggingFace model identifier
            device: Computation device
            warmup_input: Optional input for warmup inference
            **kwargs: Additional backend arguments

        Returns:
            Future that resolves to True when warmup is complete
        """
        key = self._cache_key(model_type, model_name)

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
        key = self._cache_key(model_type, model_name)

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

    def is_loaded(self, model_type: str, model_name: str | None = None) -> bool:
        """Check if a model is currently loaded."""
        key = self._cache_key(model_type, model_name)
        with self._cache_lock:
            if key in self._cache:
                return self._cache[key].state == ModelState.READY
            return False

    def is_warming(self, model_type: str, model_name: str | None = None) -> bool:
        """Check if a model is currently pre-warming."""
        key = self._cache_key(model_type, model_name)
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
        self, model_type: str, model_name: str | None = None
    ) -> dict[str, Any] | None:
        """Get information about a cached model."""
        key = self._cache_key(model_type, model_name)
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

    def evict(self, model_type: str, model_name: str | None = None) -> bool:
        """Manually evict a model from the cache."""
        key = self._cache_key(model_type, model_name)

        with self._cache_lock:
            if key not in self._cache:
                return False

            cached = self._cache[key]

        with cached._lock:
            if cached.state != ModelState.READY:
                return False

            cached.state = ModelState.EVICTED

            with self._cache_lock:
                self._stats.memory_used_bytes -= cached.memory_bytes
                self._stats.evictions += 1

            if cached.backend is not None:
                try:
                    del cached.backend.model
                    del cached.backend.processor
                except Exception as e:
                    logger.debug(
                        f"Failed to delete model/processor during manual eviction for {key}: {e}"
                    )
                cached.backend = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Manually evicted model {key}")
            return True

    def clear(self) -> None:
        """Clear all cached models."""
        with self._cache_lock:
            keys = list(self._cache.keys())

        for key in keys:
            parts = key.split(":", 1)
            model_type = parts[0]
            model_name = parts[1] if len(parts) > 1 else None
            self.evict(model_type, model_name)

        with self._cache_lock:
            self._cache.clear()
            self._stats = CacheStatistics()

        logger.info("Cleared all cached models")

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
        logger.info("Shutting down LVLMBackendCache")

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
    """
    Get a cached LVLM backend (convenience function).

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


def prewarm_backend(
    model_type: str,
    model_name: str | None = None,
    device: str = "cuda",
    warmup_input: Any | None = "auto",
    **kwargs: Any,
) -> Future[bool]:
    """
    Pre-warm an LVLM backend (convenience function).

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
