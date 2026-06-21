# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""ADRepository Dataset Loaders - REAL Anomaly Detection Datasets.

This module provides loaders for the ADRepository collection of real-world
anomaly detection datasets. These are REAL datasets with REAL anomalies,
used in academic benchmarks.

Repository: https://github.com/GuansongPang/ADRepository-Anomaly-detection-datasets
Paper: Pang et al., "Deep Learning for Anomaly Detection: A Review",
       ACM Computing Surveys, 2021.

Datasets include:
- Tabular: fraud, backdoor, campaign, thyroid, donors, census, celeba
- Time Series: SMD, SWAT, DSADS, Epilepsy
- Graph: Multiple graph-level anomaly detection datasets
"""

from __future__ import annotations

import logging
import tarfile
import zipfile
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


class DatasetMetadata(TypedDict):
    """Type definition for dataset metadata."""

    samples: int
    features: int
    anomaly_ratio: float
    domain: str
    description: str
    url: str
    file: str


class ODDSDatasetInfo(TypedDict, total=False):
    """Type definition for ODDS dataset info."""

    url: str
    format: str
    requires_auth: bool
    instructions: str


from .base import DatasetConfig, DatasetLoader, DatasetRegistry, safe_urlretrieve
from .exceptions import DataSourceUnavailableError, check_synthetic_allowed

logger = logging.getLogger(__name__)


def _safe_extract_tar(tf: tarfile.TarFile, dest: Path) -> None:
    """Extract a tar archive with the stdlib ``data`` filter (tar-slip safe).

    Tar archives from external mirrors (unlike the pure-numpy ADBench ``.npz``
    files) can carry absolute paths, ``..`` escapes, symlinks and device nodes.
    Python's ``data`` extraction filter (3.12; backported to the 3.9-3.11
    security releases) rejects exactly those — raising rather than writing
    outside ``dest``. It is the stdlib-sanctioned, fail-loud safe extraction for
    untrusted tar archives, so no hand-rolled member validation is needed.
    """
    tf.extractall(dest, filter="data")


def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract a zip archive, refusing path-traversal members (zip-slip guard).

    ``zipfile`` already sanitises member paths on extraction; as defence in depth
    we additionally resolve every member against ``dest``, refuse anything that
    would escape, and extract members **individually** (never ``extractall``) for
    the ``.zip`` archives from external mirrors.
    """
    dest = dest.resolve()
    for name in zf.namelist():
        target = (dest / name).resolve()
        if target != dest and dest not in target.parents:
            raise RuntimeError(
                f"Refusing zip member {name!r}: it escapes the extraction directory."
            )
        zf.extract(name, dest)


# =============================================================================
# ADRepository Dataset Metadata
# =============================================================================

ADREPOSITORY_DATASETS: dict[str, DatasetMetadata] = {
    # Tabular datasets (DevNet collection)
    "fraud": {
        "samples": 284807,
        "features": 29,
        "anomaly_ratio": 0.00173,
        "domain": "finance",
        "description": "Credit card fraud detection dataset",
        "url": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        "file": "creditcardfraud_normalised.tar.xz",
    },
    "backdoor": {
        "samples": 95329,
        "features": 196,
        "anomaly_ratio": 0.0244,
        "domain": "security",
        "description": "Network intrusion backdoor detection",
        "url": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        "file": "UNSW_NB15_traintest_backdoor.tar.xz",
    },
    "campaign": {
        "samples": 41188,
        "features": 62,
        "anomaly_ratio": 0.1127,
        "domain": "marketing",
        "description": "Bank marketing campaign success prediction",
        "url": "https://archive.ics.uci.edu/ml/datasets/bank+marketing",
        "file": "bank-additional-full_normalised.csv",
    },
    "thyroid": {
        "samples": 7200,
        "features": 21,
        "anomaly_ratio": 0.0244,
        "domain": "medical",
        "description": "Thyroid disease detection",
        "url": "https://archive.ics.uci.edu/ml/datasets/thyroid+disease",
        "file": "annthyroid_21feat_normalised.csv",
    },
    "donors": {
        "samples": 619326,
        "features": 10,
        "anomaly_ratio": 0.059,
        "domain": "nonprofit",
        "description": "KDD Cup 2014 donor prediction",
        "url": "https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose",
        "file": "KDD2014_donors_10feat_nomissing_normalised.csv",
    },
    "census": {
        "samples": 299285,
        "features": 500,
        "anomaly_ratio": 0.06,
        "domain": "demographics",
        "description": "US Census income prediction (high-income as anomaly)",
        "url": "https://archive.ics.uci.edu/ml/datasets/census+income",
        "file": "census-income-full-mixed-binarized.tar.xz",
    },
    "celeba": {
        "samples": 202599,
        "features": 39,
        "anomaly_ratio": 0.0227,
        "domain": "vision",
        "description": "Celebrity face attributes (bald as anomaly)",
        "url": "http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html",
        "file": "celeba_baldvsnonbald_normalised.csv",
    },
    # Time series datasets
    "smd": {
        "samples": 708405,
        "features": 38,
        "anomaly_ratio": 0.042,
        "domain": "infrastructure",
        "description": "Server Machine Dataset - real server metrics",
        "url": "https://github.com/NetManAIOps/OmniAnomaly",
        "file": "SMD.zip",
    },
    "swat": {
        "samples": 496800,
        "features": 51,
        "anomaly_ratio": 0.112,
        "domain": "industrial",
        "description": "Secure Water Treatment testbed",
        "url": "https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/",
        "file": "SWaT.zip",
    },
    "dsads": {
        "samples": 9120,
        "features": 405,
        "anomaly_ratio": 0.066,
        "domain": "activity",
        "description": "Daily and Sports Activities Dataset",
        "url": "https://archive.ics.uci.edu/ml/datasets/daily+and+sports+activities",
        "file": "DSADS.npz",
    },
    "epilepsy": {
        "samples": 11500,
        "features": 178,
        "anomaly_ratio": 0.2,
        "domain": "medical",
        "description": "Epileptic seizure detection from EEG",
        "url": "https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition",
        "file": "epilepsy.npz",
    },
}


class ADRepositoryLoader(DatasetLoader):
    """Loader for ADRepository real-world anomaly detection datasets.

    This loader fetches REAL datasets from the ADRepository collection,
    which are standard benchmarks used in academic anomaly detection research.

    Reference:
        Pang, G., Shen, C., Cao, L., & Hengel, A. V. D. (2021).
        Deep learning for anomaly detection: A review.
        ACM Computing Surveys (CSUR), 54(2), 1-38.

    Repository:
        https://github.com/GuansongPang/ADRepository-Anomaly-detection-datasets
    """

    DATASET_NAME = "adrepository"
    DATASET_URL = "https://github.com/GuansongPang/ADRepository-Anomaly-detection-datasets"
    LICENSE = "Various (see individual datasets)"
    CITATION = """Pang G, Shen C, Cao L, Hengel AVD. Deep learning for anomaly detection:
    A review. ACM Computing Surveys. 2021;54(2):1-38."""
    REQUIRES_CREDENTIALS = False

    # Canonical raw mirror. The ADRepository dataset corpus moved
    # ``GuansongPang`` -> ``mala-lab``; the old ``github.com/.../raw/main/``
    # form issues a 301 to ``raw.githubusercontent.com`` that the SSRF-safe
    # HTTP client refuses to follow by design, which is what silently routed
    # this loader to synthetic data. Pin directly to ``raw.githubusercontent``
    # (already on ``TrustedEndpoints.TRUSTED_DOMAINS``) so no redirect is
    # involved. The DevNet tabular sets live under ``numerical data/DevNet
    # datasets/`` with spaces in the path — the loader URL-encodes them.
    MIRROR_BASE = (
        "https://raw.githubusercontent.com/mala-lab/"
        "ADRepository-Anomaly-detection-datasets/main/"
    )
    DEVNET_FOLDER = "numerical data/DevNet datasets"

    # Tabular DevNet sets the mala-lab mirror actually serves (verified
    # reachable). The time-series sets (smd/swat/dsads/epilepsy) are NOT in
    # this mirror — they are served by their dedicated loaders (SMDLoader,
    # SWaTLoader, ...), so this loader fails loud for them rather than
    # silently fabricating data.
    _DEVNET_TABULAR = frozenset(
        {"fraud", "backdoor", "campaign", "thyroid", "donors", "census", "celeba"}
    )

    # ODDS (Outlier Detection DataSets) - Stony Brook University
    # These are VERIFIED working URLs for real anomaly detection datasets
    # Reference: https://odds.cs.stonybrook.edu/
    ODDS_URLS: dict[str, ODDSDatasetInfo] = {
        "thyroid": {
            "url": "https://odds.cs.stonybrook.edu/wp-content/uploads/2016/04/thyroid.mat",
            "format": "mat",
        },
        "smtp": {
            "url": "https://odds.cs.stonybrook.edu/wp-content/uploads/2016/04/smtp.mat",
            "format": "mat",
        },
        "satellite": {
            "url": "https://odds.cs.stonybrook.edu/wp-content/uploads/2016/04/satellite.mat",
            "format": "mat",
        },
        "pendigits": {
            "url": "https://odds.cs.stonybrook.edu/wp-content/uploads/2016/04/pendigits.mat",
            "format": "mat",
        },
        "mammography": {
            "url": "https://odds.cs.stonybrook.edu/wp-content/uploads/2016/04/mammography.mat",
            "format": "mat",
        },
        "shuttle": {
            "url": "https://odds.cs.stonybrook.edu/wp-content/uploads/2016/04/shuttle.mat",
            "format": "mat",
        },
        "fraud": {
            "url": "kaggle://mlg-ulb/creditcardfraud/creditcard.csv",
            "format": "csv",
            "requires_auth": True,
            "instructions": "Run: kaggle datasets download -d mlg-ulb/creditcardfraud",
        },
    }

    def __init__(self, config: DatasetConfig, dataset_name: str = "thyroid") -> None:
        """Initialize ADRepository loader.

        Args:
            config: Dataset configuration
            dataset_name: Name of dataset to load (e.g., 'fraud', 'thyroid', 'smd')
        """
        super().__init__(config)
        self.dataset_name = dataset_name.lower()

        if self.dataset_name not in ADREPOSITORY_DATASETS:
            available = ", ".join(ADREPOSITORY_DATASETS.keys())
            raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")

        self.dataset_info = ADREPOSITORY_DATASETS[self.dataset_name]
        self._features: np.ndarray[Any, Any] | None = None
        self._raw_labels: np.ndarray[Any, Any] | None = None
        self._is_real_data = False

        logger.info(
            f"ADRepositoryLoader initialized for '{dataset_name}' "
            f"({self.dataset_info['samples']} samples, "
            f"{self.dataset_info['features']} features, "
            f"{self.dataset_info['anomaly_ratio']*100:.2f}% anomalies)"
        )

    @property
    def is_real_data(self) -> bool:
        """Return True if real data was loaded (not synthetic fallback)."""
        return self._is_real_data

    def download(self) -> bool:
        """Download the dataset from its canonical real-data mirror.

        Real data only by default. If the real fetch fails, the synthetic
        path is *attempted* but :meth:`_create_synthetic_fallback` self-gates
        on ``MERCURY_ALLOW_SYNTHETIC`` and re-raises
        :class:`DataSourceUnavailableError` when synthetic is forbidden (the
        default) — so this method never silently degrades to fabricated data.

        Returns:
            True when real (or, only when explicitly permitted, synthetic)
            data was obtained.

        Raises:
            DataSourceUnavailableError: the real source was unreachable and
                ``MERCURY_ALLOW_SYNTHETIC`` is not set.
        """
        try:
            return self._download_from_repository()
        except Exception as e:
            logger.warning("ADRepository %s: real download failed: %s", self.dataset_name, e)
            return self._create_synthetic_fallback(reason=f"download failed: {e}")

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load raw data from files (implements abstract method)."""
        dataset_dir = self.data_path / self.dataset_name
        filename = self.dataset_info["file"]
        local_path = dataset_dir / filename

        # Check for ODDS-downloaded .mat file first
        if self.dataset_name in self.ODDS_URLS:
            odds_info = self.ODDS_URLS[self.dataset_name]
            odds_path = dataset_dir / f"{self.dataset_name}.{odds_info['format']}"
            if odds_path.exists():
                self._load_from_file(odds_path)
                if self._features is not None and self._raw_labels is not None:
                    return self._features, self._raw_labels

        if not local_path.exists():
            self.download()

        # Check again for ODDS file after download
        if self.dataset_name in self.ODDS_URLS:
            odds_info = self.ODDS_URLS[self.dataset_name]
            odds_path = dataset_dir / f"{self.dataset_name}.{odds_info['format']}"
            if odds_path.exists():
                self._load_from_file(odds_path)
                if self._features is not None and self._raw_labels is not None:
                    return self._features, self._raw_labels

        if local_path.exists():
            self._load_from_file(local_path)

        if self._features is None:
            # Real load produced nothing usable. Gate before fabricating.
            self._create_synthetic_fallback(reason="download succeeded but no features were parsed")

        # Type guard for mypy - at this point both should be set
        if self._features is None or self._raw_labels is None:
            raise RuntimeError("Failed to load dataset features and labels")

        return self._features, self._raw_labels

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply preprocessing (implements abstract method)."""
        # Basic normalization - zero mean, unit variance
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True) + 1e-8
        return np.asarray((data - mean) / std)  # type: ignore[no-any-return, unused-ignore]

    def _download_from_repository(self) -> bool:
        """Fetch the dataset from its single canonical real-data source.

        DevNet tabular sets come from the mala-lab ``raw.githubusercontent``
        mirror — one source per dataset, so a given name always resolves to the
        same composition (no count-ambiguous multi-mirror racing). Any failure
        propagates so the caller applies the synthetic policy gate; this method
        never substitutes data on its own.

        Raises:
            DataSourceUnavailableError: no canonical mirror serves this set
                (e.g. the time-series sets, which have their own loaders), or
                the set requires credentials.
        """
        from urllib.parse import quote

        dataset_dir = self.data_path / self.dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        filename = self.dataset_info["file"]
        local_path = dataset_dir / filename
        if local_path.exists():
            logger.info("ADRepository %s already cached at %s", self.dataset_name, local_path)
            self._is_real_data = True
            return True

        if self.dataset_name in self._DEVNET_TABULAR:
            # ``quote`` encodes the spaces in "numerical data/DevNet datasets"
            # and any spaces in the filename; the slash structure is preserved
            # because we encode the folder and filename segments separately.
            url = f"{self.MIRROR_BASE}{quote(self.DEVNET_FOLDER)}/{quote(filename)}"
            logger.info("Downloading ADRepository %s from %s", self.dataset_name, url)
            safe_urlretrieve(url, str(local_path))
            self._is_real_data = True
            logger.info(
                "Successfully downloaded ADRepository %s to %s", self.dataset_name, local_path
            )
            return True

        # ODDS-only sets (host now allowlisted in TrustedEndpoints). None of the
        # currently-registered datasets are ODDS-canonical, but keep the path so
        # a future .mat-only set works without re-plumbing.
        if self.dataset_name in self.ODDS_URLS:
            odds_info = self.ODDS_URLS[self.dataset_name]
            if odds_info.get("requires_auth"):
                raise DataSourceUnavailableError(
                    loader_name=f"ADRepository-{self.dataset_name}",
                    source_url=odds_info.get("url", ""),
                    reason=odds_info.get("instructions", "authentication required"),
                )
            odds_path = dataset_dir / f"{self.dataset_name}.{odds_info['format']}"
            logger.info(
                "Downloading ADRepository %s from ODDS: %s", self.dataset_name, odds_info["url"]
            )
            safe_urlretrieve(odds_info["url"], str(odds_path))
            self._is_real_data = True
            return True

        raise DataSourceUnavailableError(
            loader_name=f"ADRepository-{self.dataset_name}",
            reason=(
                "no canonical real-data mirror for this set; the time-series sets "
                "(smd/swat/dsads/epilepsy) are served by their dedicated loaders "
                "(SMDLoader, SWaTLoader, SMAPMSLLoader)."
            ),
        )

    def _create_synthetic_fallback(self, reason: str = "real data unavailable") -> bool:
        """Create a synthetic approximation — only when policy permits it.

        This is the single synthetic chokepoint for the loader. It self-gates
        on ``MERCURY_ALLOW_SYNTHETIC`` via
        :func:`~omni_mercury_engine.datasets.exceptions.check_synthetic_allowed`,
        which raises :class:`DataSourceUnavailableError` when synthetic is
        forbidden (the default). Every internal call site routes through here,
        so no production path can fabricate data without the explicit opt-in —
        this is the closure for the silent-synthetic foot-gun. The metadata is
        marked ``is_real_data=False`` so a permitted synthetic run can never be
        mistaken for real signal.

        Args:
            reason: Human-readable explanation of why real data was
                unavailable, surfaced in logs and in the raised error.
        """
        # Fail loud unless the deployment explicitly opted into synthetic.
        check_synthetic_allowed(f"ADRepository-{self.dataset_name}", reason)

        logger.warning(
            "ADRepository %s: returning SYNTHETIC approximation "
            "(MERCURY_ALLOW_SYNTHETIC=1). Results do NOT reflect real-world "
            "performance. Reason: %s",
            self.dataset_name,
            reason,
        )

        rng = np.random.default_rng(self.config.random_seed)

        info = self.dataset_info
        n_samples = min(info["samples"], self.config.max_samples or info["samples"])
        n_features = info["features"]
        anomaly_ratio = info["anomaly_ratio"]

        n_anomalies = int(n_samples * anomaly_ratio)
        n_normal = n_samples - n_anomalies

        # Generate normal samples
        normal_data = rng.standard_normal((n_normal, n_features))

        # Generate anomalies (shifted distribution)
        anomaly_data = rng.standard_normal((n_anomalies, n_features)) * 2 + 3

        # Combine
        self._features = np.vstack([normal_data, anomaly_data]).astype(np.float32)
        self._raw_labels = np.array([0] * n_normal + [1] * n_anomalies, dtype=np.int64)

        # Shuffle
        perm = rng.permutation(n_samples)
        self._features = self._features[perm]
        self._raw_labels = self._raw_labels[perm]

        self._is_real_data = False
        return True

    def load_data(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load dataset features and labels.

        This is the main entry point for loading ADRepository datasets.
        Use this instead of the base class load() for simpler access.

        Returns:
            Tuple of (features, labels) numpy arrays.
        """
        if self._features is not None and self._raw_labels is not None:
            return self._features, self._raw_labels

        return self._load_raw()

    def _load_mat_file(self, path: Path) -> None:
        """Load MATLAB .mat file from ODDS repository."""
        from scipy.io import loadmat

        data = loadmat(str(path))
        self._features = data["X"].astype(np.float32)
        self._raw_labels = data["y"].ravel().astype(np.int64)
        self._is_real_data = True
        logger.info(f"Loaded .mat file from {path.name} (real_data=True)")

    def _load_from_file(self, path: Path) -> None:
        """Load data from downloaded file."""
        suffix = path.suffix.lower()

        try:
            if suffix == ".mat":
                self._load_mat_file(path)

            elif suffix == ".npz":
                # External dataset files MUST round-trip through
                # allow_pickle=False. A ValueError means the file
                # contains pickled objects; refuse rather than
                # silently executing arbitrary code from an external
                # mirror. The operator can use tools/migrate_pkl.py
                # offline to convert a trusted legacy artefact.
                #
                # ``np.load`` is lazy for ``.npz``: it returns an
                # ``NpzFile`` and raises only when a member is
                # materialised. The member reads MUST sit inside the
                # same try so a pickle-backed ``X`` / ``y`` array
                # surfaces as the operator-actionable RuntimeError
                # instead of leaking a raw ``ValueError`` past this
                # block.
                try:
                    data = np.load(path, allow_pickle=False)
                    x_arr = data["X"]
                    y_arr = data["y"]
                except ValueError as exc:
                    raise RuntimeError(
                        f"Refusing to load .npz '{path}' that requires "
                        "allow_pickle=True. External dataset archives must "
                        "be pure numpy; convert offline via tools/migrate_pkl.py "
                        "if you trust the source."
                    ) from exc
                self._features = x_arr.astype(np.float32)
                self._raw_labels = y_arr.astype(np.int64)
                self._is_real_data = True

            elif suffix == ".csv":
                import pandas as pd

                df = pd.read_csv(path)

                # Assume last column is label
                self._features = df.iloc[:, :-1].values.astype(np.float32)
                self._raw_labels = df.iloc[:, -1].values.astype(np.int64)
                self._is_real_data = True

            elif suffix in (".xz", ".tar"):
                # The DevNet mirror ships several sets as a single CSV inside a
                # ``.tar.xz``. Extract with the tar-slip guard, then load the
                # CSV exactly like the plain-``.csv`` path (last column=label).
                import pandas as pd

                extract_dir = path.parent / f"{path.name}_extracted"
                extract_dir.mkdir(parents=True, exist_ok=True)
                with tarfile.open(path, "r:*") as tf:
                    _safe_extract_tar(tf, extract_dir)

                csv_members = sorted(extract_dir.rglob("*.csv"))
                if not csv_members:
                    raise RuntimeError(f"No CSV member found inside archive '{path.name}'.")
                df = pd.read_csv(csv_members[0])
                self._features = df.iloc[:, :-1].values.astype(np.float32)
                self._raw_labels = df.iloc[:, -1].values.astype(np.int64)
                self._is_real_data = True

            elif suffix == ".zip":
                # Extract and load (zip-slip guarded)
                extract_dir = path.parent / path.stem
                with zipfile.ZipFile(path, "r") as zf:
                    _safe_extract_zip(zf, extract_dir)

                # Find npz or csv files
                for f in extract_dir.rglob("*.npz"):
                    # External archives must round-trip via
                    # allow_pickle=False; legacy artefacts that need
                    # pickle must be converted offline.  ``np.load``
                    # is lazy for ``.npz`` so member reads MUST live
                    # inside the same try block as ``np.load`` itself
                    # -- a pickle-backed ``X`` / ``y`` array only
                    # raises when materialised, and we want that to
                    # surface as the same operator-actionable
                    # RuntimeError as the eager-failure case above.
                    try:
                        data = np.load(f, allow_pickle=False)
                        if "X" in data and "y" in data:
                            x_arr = data["X"]
                            y_arr = data["y"]
                        else:
                            x_arr = None
                            y_arr = None
                    except ValueError as exc:
                        raise RuntimeError(
                            f"Refusing to load .npz '{f}' that requires "
                            "allow_pickle=True. External dataset archives "
                            "must be pure numpy; convert offline via "
                            "tools/migrate_pkl.py if you trust the source."
                        ) from exc
                    if x_arr is not None and y_arr is not None:
                        self._features = x_arr.astype(np.float32)
                        self._raw_labels = y_arr.astype(np.int64)
                        self._is_real_data = True
                        break

            # Apply max_samples limit
            if (
                self._features is not None
                and self._raw_labels is not None
                and self.config.max_samples
            ):
                n = min(len(self._features), self.config.max_samples)
                self._features = self._features[:n]
                self._raw_labels = self._raw_labels[:n]

            if self._features is not None:
                logger.info(
                    f"Loaded {len(self._features)} samples from {path.name} "
                    f"(real_data={self._is_real_data})"
                )

        except RuntimeError:
            # Operator-actionable refusal (legacy pickle in external
            # .npz). Re-raise so the synthetic-fallback path below
            # cannot mask the security gate by silently downgrading the
            # load to generated data. ``RuntimeError`` is the type
            # ``_load_from_file`` raises for pickle refusal; if a
            # future code path uses a different exception, add it
            # here too.
            raise
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            self._create_synthetic_fallback()

    def get_metadata(self) -> dict[str, Any]:
        """Get dataset metadata."""
        info = self.dataset_info
        return {
            "name": self.dataset_name,
            "source": "ADRepository",
            "samples": info["samples"],
            "features": info["features"],
            "anomaly_ratio": info["anomaly_ratio"],
            "domain": info["domain"],
            "description": info["description"],
            "original_url": info["url"],
            "is_real_data": self._is_real_data,
            # Explicit, positively-named flag so callers that check
            # ``meta["synthetic"]`` (the harness convention) see the honest
            # value rather than having to invert ``is_real_data``.
            "synthetic": not self._is_real_data,
            "citation": self.CITATION,
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about loaded data."""
        if self._features is None:
            self.load_data()

        # Type guards for mypy - load_data() ensures these are not None
        if self._features is None or self._raw_labels is None:
            raise RuntimeError("Failed to load data")

        return {
            "n_samples": len(self._features),
            "n_features": self._features.shape[1],
            "n_anomalies": int(self._raw_labels.sum()),
            "anomaly_ratio": float(self._raw_labels.mean()),
            "feature_mean": float(self._features.mean()),
            "feature_std": float(self._features.std()),
            "is_real_data": self._is_real_data,
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def list_available_datasets() -> dict[str, DatasetMetadata]:
    """List all available ADRepository datasets with metadata."""
    return ADREPOSITORY_DATASETS.copy()


def load_dataset(
    name: str,
    data_dir: str = "./data/adrepository",
    max_samples: int | None = None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], dict[str, Any]]:
    """Convenience function to load an ADRepository dataset.

    Args:
        name: Dataset name (e.g., 'fraud', 'thyroid', 'smd')
        data_dir: Directory to store downloaded data
        max_samples: Maximum samples to load

    Returns:
        Tuple of (features, labels, metadata)

    Example:
        >>> X, y, meta = load_dataset('thyroid')
        >>> print(f"Loaded {len(X)} samples, {meta['anomaly_ratio']*100:.1f}% anomalies")
    """
    config = DatasetConfig(
        name=name,
        data_dir=data_dir,
        max_samples=max_samples,
    )

    loader = ADRepositoryLoader(config, dataset_name=name)
    X, y = loader.load_data()
    metadata = loader.get_metadata()

    return X, y, metadata


# Register datasets
for dataset_name in ADREPOSITORY_DATASETS:

    def _make_loader(dn: str) -> DatasetLoader:
        def _factory(cfg: DatasetConfig) -> DatasetLoader:
            return ADRepositoryLoader(cfg, dataset_name=dn)

        return _factory  # type: ignore[return-value]

    DatasetRegistry.register(f"adrepository-{dataset_name}", _make_loader(dataset_name))  # type: ignore[arg-type]
