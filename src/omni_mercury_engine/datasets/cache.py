"""
Mercury Agent - Filesystem cache for downloaded datasets.

Copyright (C) 2025 Steel Security Advisors LLC
License: GPL-3.0+

Provides a simple filesystem cache to avoid re-downloading datasets
on repeated test runs. Respects MERCURY_DATASET_CACHE env var for
custom cache locations.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DatasetCache:
    """Simple filesystem cache for downloaded datasets."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        if cache_dir is None:
            env_dir = os.getenv("MERCURY_DATASET_CACHE")
            if env_dir:
                cache_dir = Path(env_dir)
            else:
                cache_dir = Path.home() / ".mercury" / "datasets"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, dataset_name: str, source: str = "default") -> Path | None:
        """Check if dataset is cached. Returns path if found, None otherwise."""
        cache_file = self.cache_dir / f"{dataset_name}_{source}.npz"
        if cache_file.exists():
            logger.info("Cache hit: %s from %s", dataset_name, source)
            return cache_file
        return None

    def set(
        self,
        dataset_name: str,
        data: dict[str, np.ndarray[Any, Any]],
        source: str = "default",
    ) -> Path:
        """Save dataset to cache. Returns the cache file path."""
        cache_file = self.cache_dir / f"{dataset_name}_{source}.npz"
        arrays: dict[str, Any] = dict(data)
        np.savez_compressed(str(cache_file), **arrays)
        logger.info("Cached: %s to %s", dataset_name, cache_file)
        return cache_file

    def clear(self) -> None:
        """Clear all cached datasets."""
        shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Cache cleared: %s", self.cache_dir)

    def list_cached(self) -> list[str]:
        """List all cached dataset files."""
        return [f.name for f in self.cache_dir.glob("*.npz")]
