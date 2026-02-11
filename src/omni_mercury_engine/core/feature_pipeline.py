"""
Mercury Agent - Feature Extraction Pipeline
Copyright (C) 2025 Steel Security Advisors LLC

This module provides enhanced feature extraction capabilities including:
- Feature standardization with multiple scaling strategies
- Feature selection using mutual information and SHAP
- Feature imputation for failed detectors
- Feature versioning with schema validation
- Feature caching with Redis backend support

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Available feature scaling strategies."""

    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    MAXABS = "maxabs"
    NONE = "none"


@dataclass
class FeatureSchema:
    """Schema definition for feature validation."""

    name: str
    version: str
    n_features: int
    feature_names: list[str] = field(default_factory=list)
    dtypes: list[str] = field(default_factory=list)
    min_values: list[float] = field(default_factory=list)
    max_values: list[float] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert schema to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "dtypes": self.dtypes,
            "min_values": self.min_values,
            "max_values": self.max_values,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureSchema":
        """Create schema from dictionary."""
        return cls(
            name=data["name"],
            version=data["version"],
            n_features=data["n_features"],
            feature_names=data.get("feature_names", []),
            dtypes=data.get("dtypes", []),
            min_values=data.get("min_values", []),
            max_values=data.get("max_values", []),
            created_at=data.get("created_at", datetime.now(UTC).isoformat()),
        )


@dataclass
class FeatureExtractionResult:
    """Result from feature extraction with metadata."""

    features: np.ndarray
    detector_name: str
    extraction_time_ms: float
    success: bool
    error_message: str | None = None
    imputed: bool = False
    schema_version: str = "1.0.0"
    cache_hit: bool = False


class FeatureStandardizer:
    """
    Feature standardization pipeline with multiple scaling strategies.

    Supports StandardScaler, MinMaxScaler, RobustScaler, and MaxAbsScaler
    with automatic fitting and transformation.
    """

    def __init__(self, strategy: ScalingStrategy = ScalingStrategy.STANDARD):
        """
        Initialize the feature standardizer.

        Args:
            strategy: Scaling strategy to use (standard, minmax, robust, maxabs, none)
        """
        self.strategy = strategy
        self._fitted = False

        # Statistics for different scaling strategies
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._median: np.ndarray | None = None
        self._iqr: np.ndarray | None = None
        self._max_abs: np.ndarray | None = None

        logger.info(f"FeatureStandardizer initialized with strategy: {strategy.value}")

    def fit(self, X: np.ndarray) -> "FeatureStandardizer":
        """
        Fit the standardizer on training data.

        Args:
            X: Training data of shape (n_samples, n_features)

        Returns:
            self for method chaining
        """
        X = np.asarray(X)

        if self.strategy == ScalingStrategy.STANDARD:
            self._mean = np.mean(X, axis=0)
            self._std = np.std(X, axis=0) + 1e-8

        elif self.strategy == ScalingStrategy.MINMAX:
            self._min = np.min(X, axis=0)
            self._max = np.max(X, axis=0)
            # Avoid division by zero
            self._max = np.where(self._max == self._min, self._min + 1e-8, self._max)

        elif self.strategy == ScalingStrategy.ROBUST:
            self._median = np.median(X, axis=0)
            q75 = np.percentile(X, 75, axis=0)
            q25 = np.percentile(X, 25, axis=0)
            self._iqr = q75 - q25 + 1e-8

        elif self.strategy == ScalingStrategy.MAXABS:
            self._max_abs = np.max(np.abs(X), axis=0) + 1e-8

        self._fitted = True
        logger.debug(f"FeatureStandardizer fitted on {X.shape[0]} samples")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using the fitted standardizer.

        Args:
            X: Data to transform of shape (n_samples, n_features)

        Returns:
            Transformed features
        """
        if not self._fitted:
            raise ValueError("FeatureStandardizer must be fitted before transform")

        X = np.asarray(X)

        if self.strategy == ScalingStrategy.STANDARD:
            if self._mean is None or self._std is None:
                raise ValueError("FeatureStandardizer: _mean/_std not set for STANDARD strategy")
            return np.asarray((X - self._mean) / self._std)  # type: ignore[no-any-return, unused-ignore]

        elif self.strategy == ScalingStrategy.MINMAX:
            if self._min is None or self._max is None:
                raise ValueError("FeatureStandardizer: _min/_max not set for MINMAX strategy")
            return np.asarray((X - self._min) / (self._max - self._min))  # type: ignore[no-any-return, unused-ignore]

        elif self.strategy == ScalingStrategy.ROBUST:
            if self._median is None or self._iqr is None:
                raise ValueError("FeatureStandardizer: _median/_iqr not set for ROBUST strategy")
            return np.asarray((X - self._median) / self._iqr)  # type: ignore[no-any-return, unused-ignore]

        elif self.strategy == ScalingStrategy.MAXABS:
            if self._max_abs is None:
                raise ValueError("FeatureStandardizer: _max_abs not set for MAXABS strategy")
            return np.asarray(X / self._max_abs)  # type: ignore[no-any-return, unused-ignore]

        return X  # NONE strategy

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform to original scale.

        Args:
            X: Transformed data

        Returns:
            Data in original scale
        """
        if not self._fitted:
            raise ValueError("FeatureStandardizer must be fitted before inverse_transform")

        X = np.asarray(X)

        if self.strategy == ScalingStrategy.STANDARD:
            if self._mean is None or self._std is None:
                raise ValueError("FeatureStandardizer: _mean/_std not set for STANDARD strategy")
            return np.asarray(X * self._std + self._mean)  # type: ignore[no-any-return, unused-ignore]

        elif self.strategy == ScalingStrategy.MINMAX:
            if self._min is None or self._max is None:
                raise ValueError("FeatureStandardizer: _min/_max not set for MINMAX strategy")
            return np.asarray(X * (self._max - self._min) + self._min)  # type: ignore[no-any-return, unused-ignore]

        elif self.strategy == ScalingStrategy.ROBUST:
            if self._median is None or self._iqr is None:
                raise ValueError("FeatureStandardizer: _median/_iqr not set for ROBUST strategy")
            return np.asarray(X * self._iqr + self._median)  # type: ignore[no-any-return, unused-ignore]

        elif self.strategy == ScalingStrategy.MAXABS:
            if self._max_abs is None:
                raise ValueError("FeatureStandardizer: _max_abs not set for MAXABS strategy")
            return np.asarray(X * self._max_abs)  # type: ignore[no-any-return, unused-ignore]

        return X


class FeatureSelector:
    """
    Feature selection using mutual information and importance scoring.

    Supports mutual information-based selection and can integrate with
    SHAP for model-agnostic feature importance.
    """

    def __init__(
        self,
        n_features_to_select: int | None = None,
        selection_ratio: float = 0.5,
        method: str = "mutual_info",
    ):
        """
        Initialize the feature selector.

        Args:
            n_features_to_select: Number of features to select (overrides ratio)
            selection_ratio: Ratio of features to keep (0.0-1.0)
            method: Selection method ('mutual_info', 'variance', 'correlation')
        """
        self.n_features_to_select = n_features_to_select
        self.selection_ratio = selection_ratio
        self.method = method

        self._fitted = False
        self._feature_scores: np.ndarray | None = None
        self._selected_indices: np.ndarray | None = None
        self._n_original_features: int = 0

        logger.info(f"FeatureSelector initialized with method: {method}")

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "FeatureSelector":
        """
        Fit the feature selector.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target labels (required for mutual_info method)

        Returns:
            self for method chaining
        """
        X = np.asarray(X)
        self._n_original_features = X.shape[1]

        # Determine number of features to select
        if self.n_features_to_select is not None:
            n_select = min(self.n_features_to_select, X.shape[1])
        else:
            n_select = max(1, int(X.shape[1] * self.selection_ratio))

        # Compute feature scores based on method
        if self.method == "mutual_info" and y is not None:
            self._feature_scores = self._compute_mutual_info(X, y)
        elif self.method == "variance":
            self._feature_scores = np.var(X, axis=0)
        elif self.method == "correlation":
            self._feature_scores = self._compute_correlation_scores(X)
        else:
            # Default to variance if no labels provided
            self._feature_scores = np.var(X, axis=0)

        # Select top features
        self._selected_indices = np.argsort(self._feature_scores)[-n_select:]
        self._selected_indices = np.sort(self._selected_indices)

        self._fitted = True
        logger.debug(f"FeatureSelector selected {n_select}/{X.shape[1]} features")
        return self

    def _compute_mutual_info(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute mutual information scores for each feature."""
        try:
            from sklearn.feature_selection import mutual_info_classif

            # Handle continuous targets by discretizing
            if len(np.unique(y)) > 10:
                y_discrete = np.digitize(y, np.percentile(y, [25, 50, 75]))
            else:
                y_discrete = y

            scores = mutual_info_classif(X, y_discrete, random_state=42)
            return np.asarray(scores)  # type: ignore[no-any-return, unused-ignore]
        except ImportError:
            logger.warning("sklearn not available, falling back to variance")
            return np.asarray(np.var(X, axis=0))  # type: ignore[no-any-return, unused-ignore]

    def _compute_correlation_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute correlation-based scores (inverse of mean correlation)."""
        corr_matrix = np.corrcoef(X.T)
        # Handle NaN values
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        # Score is inverse of mean absolute correlation (prefer less correlated)
        mean_corr = np.mean(np.abs(corr_matrix), axis=1)
        return np.asarray(1.0 / (mean_corr + 1e-8))  # type: ignore[no-any-return, unused-ignore]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features by selecting the fitted subset.

        Args:
            X: Feature matrix to transform

        Returns:
            Selected features
        """
        if not self._fitted:
            raise ValueError("FeatureSelector must be fitted before transform")

        X = np.asarray(X)
        return X[:, self._selected_indices]

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_feature_importance(self) -> dict[int, float]:
        """Get feature importance scores."""
        if self._feature_scores is None:
            return {}
        return {i: float(score) for i, score in enumerate(self._feature_scores)}

    def get_selected_indices(self) -> np.ndarray:
        """Get indices of selected features."""
        if self._selected_indices is None:
            return np.array([])
        return self._selected_indices.copy()


class FeatureImputer:
    """
    Feature imputation for failed detectors using historical patterns.

    Tracks historical feature statistics and imputes missing features
    when detectors fail.
    """

    def __init__(self, strategy: str = "mean", max_history: int = 1000):
        """
        Initialize the feature imputer.

        Args:
            strategy: Imputation strategy ('mean', 'median', 'zero', 'last')
            max_history: Maximum number of historical samples to track
        """
        self.strategy = strategy
        self.max_history = max_history

        # Historical statistics per detector
        self._detector_history: dict[str, list[np.ndarray]] = {}
        self._detector_stats: dict[str, dict[str, np.ndarray]] = {}

        logger.info(f"FeatureImputer initialized with strategy: {strategy}")

    def update_history(self, detector_name: str, features: np.ndarray) -> None:
        """
        Update historical features for a detector.

        Args:
            detector_name: Name of the detector
            features: Successfully extracted features
        """
        if detector_name not in self._detector_history:
            self._detector_history[detector_name] = []

        self._detector_history[detector_name].append(features.copy())

        # Limit history size
        if len(self._detector_history[detector_name]) > self.max_history:
            self._detector_history[detector_name] = self._detector_history[detector_name][
                -self.max_history :
            ]

        # Update statistics
        history = np.array(self._detector_history[detector_name])
        self._detector_stats[detector_name] = {
            "mean": np.mean(history, axis=0),
            "median": np.median(history, axis=0),
            "std": np.std(history, axis=0),
            "last": history[-1],
        }

    def impute(self, detector_name: str, n_features: int) -> np.ndarray:
        """
        Impute features for a failed detector.

        Args:
            detector_name: Name of the failed detector
            n_features: Number of features to impute

        Returns:
            Imputed feature vector
        """
        if detector_name in self._detector_stats:
            stats = self._detector_stats[detector_name]

            if self.strategy == "mean":
                return stats["mean"]
            elif self.strategy == "median":
                return stats["median"]
            elif self.strategy == "last":
                return stats["last"]

        # Default to zeros if no history
        logger.warning(f"No history for detector {detector_name}, using zeros")
        return np.zeros(n_features)

    def has_history(self, detector_name: str) -> bool:
        """Check if historical data exists for a detector."""
        return detector_name in self._detector_stats


class FeatureVersionManager:
    """
    Feature versioning with schema validation.

    Tracks feature schema versions and validates extracted features
    against expected schemas.
    """

    def __init__(self) -> None:
        """Initialize the version manager."""
        self._schemas: dict[str, FeatureSchema] = {}
        self._version_history: dict[str, list[str]] = {}

        logger.info("FeatureVersionManager initialized")

    def register_schema(self, schema: FeatureSchema) -> None:
        """
        Register a feature schema.

        Args:
            schema: Feature schema to register
        """
        key = f"{schema.name}:{schema.version}"
        self._schemas[key] = schema

        if schema.name not in self._version_history:
            self._version_history[schema.name] = []
        self._version_history[schema.name].append(schema.version)

        logger.info(f"Registered schema: {key}")

    def get_schema(self, name: str, version: str | None = None) -> FeatureSchema | None:
        """
        Get a feature schema.

        Args:
            name: Schema name
            version: Schema version (latest if None)

        Returns:
            Feature schema or None if not found
        """
        if version is None:
            # Get latest version
            if self._version_history.get(name):
                version = self._version_history[name][-1]
            else:
                return None

        key = f"{name}:{version}"
        return self._schemas.get(key)

    def validate_features(
        self, features: np.ndarray, schema_name: str, schema_version: str | None = None
    ) -> tuple[bool, list[str]]:
        """
        Validate features against a schema.

        Args:
            features: Feature array to validate
            schema_name: Name of the schema
            schema_version: Version of the schema

        Returns:
            Tuple of (is_valid, list of validation errors)
        """
        schema = self.get_schema(schema_name, schema_version)
        if schema is None:
            return True, []  # No schema to validate against

        errors = []

        # Check feature count
        if features.shape[-1] != schema.n_features:
            errors.append(
                f"Feature count mismatch: expected {schema.n_features}, got {features.shape[-1]}"
            )

        # Check value ranges if defined
        if schema.min_values and schema.max_values:
            for i, (min_val, max_val) in enumerate(zip(schema.min_values, schema.max_values)):
                if i < features.shape[-1]:
                    feat_min = np.min(features[..., i])
                    feat_max = np.max(features[..., i])
                    if feat_min < min_val - 1e-6 or feat_max > max_val + 1e-6:
                        errors.append(
                            f"Feature {i} out of range: [{feat_min}, {feat_max}] "
                            f"not in [{min_val}, {max_val}]"
                        )

        is_valid = len(errors) == 0
        if not is_valid:
            logger.warning(f"Schema validation failed for {schema_name}: {errors}")

        return is_valid, errors

    def create_schema_from_features(
        self, name: str, version: str, features: np.ndarray, feature_names: list[str] | None = None
    ) -> FeatureSchema:
        """
        Create a schema from sample features.

        Args:
            name: Schema name
            version: Schema version
            features: Sample features to derive schema from
            feature_names: Optional feature names

        Returns:
            Created feature schema
        """
        n_features = features.shape[-1]

        schema = FeatureSchema(
            name=name,
            version=version,
            n_features=n_features,
            feature_names=feature_names or [f"feature_{i}" for i in range(n_features)],
            dtypes=[str(features.dtype)] * n_features,
            min_values=(
                np.min(features, axis=0).tolist()
                if features.ndim > 1
                else [float(np.min(features))]
            ),
            max_values=(
                np.max(features, axis=0).tolist()
                if features.ndim > 1
                else [float(np.max(features))]
            ),
        )

        self.register_schema(schema)
        return schema


class CacheBackend(Protocol):
    """Protocol for cache backends."""

    def get(self, key: str) -> bytes | None:
        """Get value from cache."""
        ...

    def set(self, key: str, value: bytes, ttl: int | None = None) -> bool:
        """Set value in cache with optional TTL."""
        ...

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        ...

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        ...


class InMemoryCache:
    """Simple in-memory cache implementation."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[bytes, float | None]] = {}

    def get(self, key: str) -> bytes | None:
        """Get value from cache."""
        if key not in self._cache:
            return None

        value, expiry = self._cache[key]
        if expiry is not None and time.time() > expiry:
            del self._cache[key]
            return None

        return value

    def set(self, key: str, value: bytes, ttl: int | None = None) -> bool:
        """Set value in cache with optional TTL."""
        expiry = time.time() + ttl if ttl else None
        self._cache[key] = (value, expiry)
        return True

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self.get(key) is not None


class RedisCache:
    """Redis cache backend implementation."""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        """
        Initialize Redis cache.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
        """
        self._client = None
        self._host = host
        self._port = port
        self._db = db

        try:
            import redis

            self._client = redis.Redis(host=host, port=port, db=db)
            self._client.ping()
            logger.info(f"Redis cache connected: {host}:{port}")
        except ImportError:
            logger.warning("redis package not available, using in-memory fallback")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using in-memory fallback")

    def get(self, key: str) -> bytes | None:
        """Get value from cache."""
        if self._client is None:
            return None
        try:
            result: bytes | None = self._client.get(key)  # type: ignore[assignment, unused-ignore]
            return result
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    def set(self, key: str, value: bytes, ttl: int | None = None) -> bool:
        """Set value in cache with optional TTL."""
        if self._client is None:
            return False
        try:
            if ttl:
                result: bool = self._client.setex(key, ttl, value)  # type: ignore[assignment, unused-ignore]
                return result
            result = self._client.set(key, value)  # type: ignore[assignment, unused-ignore]
            return result
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if self._client is None:
            return False
        try:
            deleted: int = self._client.delete(key)  # type: ignore[assignment, unused-ignore]
            return deleted > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if self._client is None:
            return False
        try:
            count: int = self._client.exists(key)  # type: ignore[assignment, unused-ignore]
            return count > 0
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False


class FeatureStore:
    """
    Feature store with caching support.

    Provides caching for extracted features with configurable backends
    (in-memory or Redis) and TTL support.
    """

    def __init__(
        self,
        backend: str = "memory",
        ttl: int = 3600,
        redis_host: str = "localhost",
        redis_port: int = 6379,
    ):
        """
        Initialize the feature store.

        Args:
            backend: Cache backend ('memory' or 'redis')
            ttl: Default TTL in seconds
            redis_host: Redis host (if using redis backend)
            redis_port: Redis port (if using redis backend)
        """
        self.ttl = ttl
        self._backend_type = backend

        if backend == "redis":
            self._cache: CacheBackend = RedisCache(host=redis_host, port=redis_port)
        else:
            self._cache = InMemoryCache()

        # Metrics
        self._hits = 0
        self._misses = 0
        self._stores = 0

        logger.info(f"FeatureStore initialized with {backend} backend, TTL={ttl}s")

    def _generate_key(self, detector_name: str, data_hash: str) -> str:
        """Generate cache key from detector name and data hash."""
        return f"features:{detector_name}:{data_hash}"

    def _hash_data(self, data: np.ndarray) -> str:
        """Generate hash of input data."""
        return hashlib.sha3_256(data.tobytes()).hexdigest()[:16]

    def get(self, detector_name: str, data: np.ndarray) -> np.ndarray | None:
        """
        Get cached features.

        Args:
            detector_name: Name of the detector
            data: Input data (used for cache key)

        Returns:
            Cached features or None if not found
        """
        data_hash = self._hash_data(data)
        key = self._generate_key(detector_name, data_hash)

        cached = self._cache.get(key)
        if cached is not None:
            self._hits += 1
            try:
                return np.frombuffer(cached, dtype=np.float64)
            except Exception as e:
                logger.error(f"Failed to deserialize cached features: {e}")
                return None

        self._misses += 1
        return None

    def store(
        self, detector_name: str, data: np.ndarray, features: np.ndarray, ttl: int | None = None
    ) -> bool:
        """
        Store features in cache.

        Args:
            detector_name: Name of the detector
            data: Input data (used for cache key)
            features: Features to cache
            ttl: Optional TTL override

        Returns:
            True if stored successfully
        """
        data_hash = self._hash_data(data)
        key = self._generate_key(detector_name, data_hash)

        try:
            value = features.astype(np.float64).tobytes()
            success = self._cache.set(key, value, ttl or self.ttl)
            if success:
                self._stores += 1
            return success
        except Exception as e:
            logger.error(f"Failed to store features: {e}")
            return False

    def invalidate(self, detector_name: str, data: np.ndarray) -> bool:
        """Invalidate cached features."""
        data_hash = self._hash_data(data)
        key = self._generate_key(detector_name, data_hash)
        return self._cache.delete(key)

    def get_metrics(self) -> dict[str, Any]:
        """Get cache performance metrics."""
        total = self._hits + self._misses
        hit_ratio = self._hits / total if total > 0 else 0.0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "stores": self._stores,
            "hit_ratio": hit_ratio,
            "backend": self._backend_type,
            "ttl": self.ttl,
        }


class FeaturePipeline:
    """
    Complete feature extraction pipeline combining all components.

    Integrates standardization, selection, imputation, versioning, and caching
    into a unified pipeline.
    """

    def __init__(
        self,
        scaling_strategy: ScalingStrategy = ScalingStrategy.STANDARD,
        selection_ratio: float = 1.0,
        selection_method: str = "mutual_info",
        imputation_strategy: str = "mean",
        cache_backend: str = "memory",
        cache_ttl: int = 3600,
    ):
        """
        Initialize the feature pipeline.

        Args:
            scaling_strategy: Feature scaling strategy
            selection_ratio: Ratio of features to select (1.0 = all)
            selection_method: Feature selection method
            imputation_strategy: Strategy for imputing missing features
            cache_backend: Cache backend type
            cache_ttl: Cache TTL in seconds
        """
        self.standardizer = FeatureStandardizer(strategy=scaling_strategy)
        self.selector = (
            FeatureSelector(selection_ratio=selection_ratio, method=selection_method)
            if selection_ratio < 1.0
            else None
        )
        self.imputer = FeatureImputer(strategy=imputation_strategy)
        self.version_manager = FeatureVersionManager()
        self.feature_store = FeatureStore(backend=cache_backend, ttl=cache_ttl)

        self._fitted = False

        logger.info(
            f"FeaturePipeline initialized: scaling={scaling_strategy.value}, "
            f"selection_ratio={selection_ratio}, cache={cache_backend}"
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "FeaturePipeline":
        """
        Fit the pipeline on training data.

        Args:
            X: Training features
            y: Training labels (optional)

        Returns:
            self for method chaining
        """
        # Fit standardizer
        self.standardizer.fit(X)

        # Standardize before selection
        X_scaled = self.standardizer.transform(X)

        # Fit selector if enabled
        if self.selector is not None:
            self.selector.fit(X_scaled, y)

        self._fitted = True
        return self

    def transform(
        self,
        X: np.ndarray,
        detector_name: str = "default",
        use_cache: bool = True,
    ) -> FeatureExtractionResult:
        """
        Transform features through the pipeline.

        Args:
            X: Input features
            detector_name: Name of the detector (for caching)
            use_cache: Whether to use caching

        Returns:
            Feature extraction result
        """
        start_time = time.perf_counter()
        cache_hit = False

        # Check cache first
        if use_cache:
            cached = self.feature_store.get(detector_name, X)
            if cached is not None:
                cache_hit = True
                return FeatureExtractionResult(
                    features=cached,
                    detector_name=detector_name,
                    extraction_time_ms=(time.perf_counter() - start_time) * 1000,
                    success=True,
                    cache_hit=True,
                )

        try:
            # Standardize
            X_scaled = self.standardizer.transform(X)

            # Select features if enabled
            if self.selector is not None:
                X_selected = self.selector.transform(X_scaled)
            else:
                X_selected = X_scaled

            # Update imputer history
            if X_selected.ndim == 1:
                self.imputer.update_history(detector_name, X_selected)
            else:
                for row in X_selected:
                    self.imputer.update_history(detector_name, row)

            # Cache result
            if use_cache:
                self.feature_store.store(detector_name, X, X_selected)

            return FeatureExtractionResult(
                features=X_selected,
                detector_name=detector_name,
                extraction_time_ms=(time.perf_counter() - start_time) * 1000,
                success=True,
                cache_hit=cache_hit,
            )

        except Exception as e:
            logger.error(f"Feature extraction failed for {detector_name}: {e}")

            # Try imputation
            n_features = X.shape[-1]
            if self.selector is not None:
                n_features = len(self.selector.get_selected_indices())

            imputed = self.imputer.impute(detector_name, n_features)

            return FeatureExtractionResult(
                features=imputed,
                detector_name=detector_name,
                extraction_time_ms=(time.perf_counter() - start_time) * 1000,
                success=False,
                error_message=str(e),
                imputed=True,
            )

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray | None = None, detector_name: str = "default"
    ) -> FeatureExtractionResult:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X, detector_name)

    def get_metrics(self) -> dict[str, Any]:
        """Get pipeline metrics."""
        return {
            "cache_metrics": self.feature_store.get_metrics(),
            "feature_importance": (self.selector.get_feature_importance() if self.selector else {}),
            "fitted": self._fitted,
        }
