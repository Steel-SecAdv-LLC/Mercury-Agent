"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisors LLC

Standardized dataset metadata for all loaders.

Every dataset object returned by a loader must carry metadata including
data_source, source_url, sha256, record_count, anomaly_ratio, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class LoaderDatasetMetadata:
    """Metadata that every loader must attach to its returned data.

    Attributes:
        data_source: One of "live", "cached", "synthetic".
        source_url: The URL from which data was fetched.
        download_timestamp: ISO 8601 timestamp of download.
        record_count: Number of records/samples in the dataset.
        feature_count: Number of features per sample.
        anomaly_count: Number of anomaly-labeled samples.
        anomaly_ratio: Fraction of samples labeled as anomaly.
        sha256: SHA-256 hex digest of the raw downloaded file.
        loader_name: Identifier of the loader class.
        loader_version: Version string for the loader.
        labels_available: Whether ground truth labels are present.
        label_source: One of "ground_truth", "expert_annotated", "statistical", "none".
    """

    data_source: str  # "live", "cached", "synthetic"
    source_url: str
    download_timestamp: str  # ISO 8601
    record_count: int
    feature_count: int
    anomaly_count: int
    anomaly_ratio: float
    sha256: str
    loader_name: str
    loader_version: str = "1.0.0"
    labels_available: bool = True
    label_source: str = "ground_truth"


@dataclass
class LoaderDataset:
    """Standardized dataset returned by every loader.

    Attributes:
        X: Feature matrix of shape (n_samples, n_features).
        y: Label vector of shape (n_samples,), 0=normal, 1=anomaly.
        metadata: Loader metadata with provenance information.
        feature_names: Optional list of feature name strings.
        timestamps: Optional array of timestamps.
        split: One of "train", "test", "full".
    """

    X: NDArray[np.float64]
    y: NDArray[np.int32]
    metadata: LoaderDatasetMetadata
    feature_names: Optional[list[str]] = None
    timestamps: Optional[NDArray[np.float64]] = None
    split: str = "full"


__all__ = [
    "LoaderDataset",
    "LoaderDatasetMetadata",
]
