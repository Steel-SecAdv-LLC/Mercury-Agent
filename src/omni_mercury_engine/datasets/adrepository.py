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
import shutil
import tarfile
import zipfile
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
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


from .base import DatasetConfig, DatasetLoader, DatasetRegistry, safe_urlretrieve
from .exceptions import DataSourceUnavailableError, check_synthetic_allowed

logger = logging.getLogger(__name__)


def _tar_supports_data_filter() -> bool:
    """Whether this interpreter's ``tarfile`` exposes the PEP 706 ``data`` filter.

    ``tarfile.data_filter`` was added in lock-step with the ``filter=`` keyword
    on ``TarFile.extract``/``extractall`` (CPython 3.12, backported to the
    3.9-3.11 *security* releases, i.e. 3.11.4+). This is the single
    feature-detection seam for the extraction path, so tests can simulate a
    pre-3.11.4 interpreter by patching this one function **without** deleting the
    stdlib attribute: on CPython 3.14 the ``data`` filter is the PEP 706 default
    that ``TarFile.extract`` looks up *by that global name*, so deleting it
    breaks extraction itself (``NameError``) and models no real interpreter.
    """
    return hasattr(tarfile, "data_filter")


def _safe_extract_tar(tf: tarfile.TarFile, dest: Path) -> None:
    """Extract a tar archive, rejecting tar-slip members (tar-slip safe).

    Tar archives from external mirrors (unlike the pure-numpy ADBench ``.npz``
    files) can carry absolute paths, ``..`` escapes, symlinks and device nodes.
    Python's ``data`` extraction filter rejects exactly those — but the
    ``filter=`` parameter only exists on CPython 3.12+ and the 3.9-3.11
    *security* releases (3.11.4+). On the 3.11.0-3.11.3 patch releases this
    project still supports (``requires-python = ">=3.11"``) it is absent, so
    passing ``filter="data"`` there raises ``TypeError`` — which the caller's
    broad ``except`` would turn into a silent synthetic-data fallback. We
    therefore feature-detect the filter (via ``_tar_supports_data_filter``) and
    fall back to an explicit member guard that enforces the same invariants when
    it is missing.
    """
    if _tar_supports_data_filter():
        tf.extractall(dest, filter="data")
        return
    _extract_tar_members_guarded(tf, dest)


def _extract_tar_members_guarded(tf: tarfile.TarFile, dest: Path) -> None:
    """Tar-slip-safe extraction for interpreters predating the ``data`` filter.

    Enforces the same invariants the stdlib ``data`` filter would: reject
    sym/hard links and device/special members outright, and refuse any member
    whose resolved path escapes ``dest``. Members are extracted individually
    (never a bare ``extractall``) and only after they pass the guard, so a
    malicious archive cannot write outside ``dest`` even on old interpreters.
    """
    dest = dest.resolve()
    # If the runtime actually exposes the ``filter=`` kwarg, pass ``data``
    # explicitly per member instead of relying on ``TarFile``'s default-filter
    # resolution. On CPython 3.14 a bare ``extract()`` resolves the default by
    # looking up the module-global ``data_filter`` *by name*, so an explicit
    # kwarg keeps any path that reaches this guard on a modern interpreter
    # deterministic and free of that global dependency. On the genuine
    # pre-3.11.4 releases this guard targets, the kwarg does not exist (and no
    # default-filter machinery references that global), so we call plain
    # ``extract()`` there. ``hasattr`` (not ``_tar_supports_data_filter``) is
    # the real runtime capability: the latter is patched in tests to force this
    # fallback branch, but the per-member call must still match the interpreter.
    pass_data_filter = hasattr(tarfile, "data_filter")
    for member in tf.getmembers():
        if member.issym() or member.islnk():
            raise RuntimeError(f"Refusing tar member {member.name!r}: links are not allowed.")
        if not (member.isfile() or member.isdir()):
            raise RuntimeError(
                f"Refusing tar member {member.name!r}: only regular files and "
                "directories may be extracted."
            )
        target = (dest / member.name).resolve()
        if target != dest and dest not in target.parents:
            raise RuntimeError(
                f"Refusing tar member {member.name!r}: it escapes the extraction directory."
            )
        if pass_data_filter:
            tf.extract(member, dest, filter="data")
        else:
            tf.extract(member, dest)


def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path) -> None:
    r"""Extract a zip archive, refusing path-traversal members (zip-slip guard).

    Do **not** assume ``zipfile`` sanitises member paths for you: a member named
    ``../x`` or with an absolute path can write outside the destination
    (zip-slip). This helper *is* the sanitiser — it resolves every member
    against ``dest``, refuses anything that would escape, and extracts members
    **individually** (never ``extractall``). Keep this guard even if a future
    stdlib adds its own, and never replace it with a bare ``extractall``.

    The escape check runs against a separator-normalised name: ZIP entries may
    carry **backslash** separators, which ``zipfile.extract`` honours as path
    separators on Windows but POSIX ``Path`` treats as an ordinary filename
    character. A ``..\escape.csv`` member would therefore slip past a raw-name
    check on POSIX yet escape ``dest`` on Windows. Normalising ``\`` -> ``/``
    for the *check* refuses it on every platform; ``zf.extract`` still receives
    the original member name.
    """
    dest = dest.resolve()
    for name in zf.namelist():
        check_name = name.replace("\\", "/")
        target = (dest / check_name).resolve()
        if target != dest and dest not in target.parents:
            raise RuntimeError(
                f"Refusing zip member {name!r}: it escapes the extraction directory."
            )
        zf.extract(name, dest)


def _fresh_extract_dir(base: Path) -> Path:
    """Return ``base`` emptied of any prior extraction, then (re)created.

    Re-using a persistent extraction directory across runs is a stale-data
    hazard: the callers locate the member to load via ``rglob('*.csv')`` and
    take the first match, so a leftover CSV from a previously-extracted archive
    with a different layout could be picked up and silently loaded as the wrong
    dataset. Clearing first guarantees the directory reflects only the archive
    about to be extracted.
    """
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)
    return base


def _select_single_csv(members: list[Path], archive_name: str) -> Path:
    """Return the sole CSV member of an archive, or fail loud otherwise.

    The ``.tar(.xz)`` DevNet sets each ship a *single* normalised CSV. Silently
    taking the first of several (``sorted(...)[0]``) could load the wrong split —
    e.g. a train/test bundle whose name implies multiple members — and change the
    dataset composition with no signal. Refuse rather than guess: a deterministic
    loud failure is recoverable; a silently-wrong split is not.
    """
    if not members:
        raise RuntimeError(f"No CSV member found inside archive {archive_name!r}.")
    if len(members) > 1:
        names = ", ".join(sorted(m.name for m in members))
        raise RuntimeError(
            f"Archive {archive_name!r} contains multiple CSV members ({names}); "
            "refusing to silently pick one (it could load the wrong split). Add an "
            "explicit member-selection rule for this dataset to load it."
        )
    return members[0]


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
        # Constructed convention (DSADS has no native anomaly labels): default
        # anomaly = activity 19 (basketball) -> 480/9120 = 0.0526. See DSADSLoader.
        "anomaly_ratio": 0.0526,
        "domain": "activity",
        "description": (
            "Daily and Sports Activities (UCI 256): 9120 segments x 405 per-channel "
            "stats; real sensor data, documented constructed anomaly labels"
        ),
        "url": "https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities",
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
    # reachable end-to-end). The time-series sets (smd/swat/dsads/epilepsy) are
    # NOT in this mirror; they are handled in ``_load_raw`` — delegated to a
    # dedicated loader where one exists (_TIMESERIES_DELEGATES) or failed loud
    # with a named closing step otherwise (_TIMESERIES_NO_LOADER). Either way
    # this loader never fabricates them.
    _DEVNET_TABULAR = frozenset(
        {"fraud", "backdoor", "campaign", "thyroid", "donors", "census", "celeba"}
    )

    # Time-series sets are NOT in the DevNet tabular mirror. Those with a
    # dedicated Mercury loader are delegated to it (the loader owns the real
    # fetch and fails loud when it can't); the mapping is name -> (module,
    # class). The legacy ODDS (Stony Brook) path was removed: the host is dead
    # (503) and no registered dataset name ever routed through it.
    _TIMESERIES_DELEGATES: dict[str, tuple[str, str]] = {
        "smd": ("omni_mercury_engine.datasets.timeseries", "SMDLoader"),
        "swat": ("omni_mercury_engine.datasets.industrial", "SWaTLoader"),
        # DSADS: real UCI-256 inertial-sensor data; DSADSLoader fetches it and
        # constructs a documented anomaly convention (DSADS has no native labels).
        "dsads": ("omni_mercury_engine.datasets.timeseries", "DSADSLoader"),
        # Epilepsy: EpilepsyLoader reconstructs the canonical 11500×178 form from
        # the official Bonn EEG sets (Andrzejak et al. 2001). The UPF source is
        # Cloudflare-gated, so the data is supplied via preprocessing['bonn_dir'];
        # without it the loader fails loud (never fabricates, never uses a mirror).
        "epilepsy": ("omni_mercury_engine.datasets.timeseries", "EpilepsyLoader"),
    }

    # Time-series sets with a documented real upstream but no dedicated Mercury
    # loader yet. Currently empty — smd/swat/dsads/epilepsy are all delegated
    # above. Retained as the explicit, gated chokepoint for any future such set:
    # it fails loud (naming the source) rather than silently fabricating.
    _TIMESERIES_NO_LOADER: dict[str, str] = {}

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
        """Load raw data (implements abstract method).

        Time-series sets are never served by the DevNet tabular mirror: those
        with a dedicated Mercury loader (smd, swat) are delegated to it, and
        those without one yet (dsads, epilepsy) fail loud naming the real
        upstream and the exact closing step. Only the tabular DevNet sets flow
        through the download + file-parse path below.
        """
        if self.dataset_name in self._TIMESERIES_DELEGATES:
            return self._load_via_delegate()
        if self.dataset_name in self._TIMESERIES_NO_LOADER:
            # No dedicated Mercury loader yet → route through the single gated
            # chokepoint (raises by default, naming the upstream + closing step;
            # only fabricates, marked synthetic, under MERCURY_ALLOW_SYNTHETIC).
            self._create_synthetic_fallback(
                reason=(
                    f"'{self.dataset_name}' is a time-series set with no dedicated "
                    f"Mercury loader yet (it is not in the DevNet tabular mirror). "
                    f"Real source: {self._TIMESERIES_NO_LOADER[self.dataset_name]}. "
                    f"Closing step: add a dedicated loader (cf. SMDLoader) and register "
                    f"it in ADRepositoryLoader._TIMESERIES_DELEGATES."
                )
            )
            if self._features is None or self._raw_labels is None:
                raise RuntimeError("Failed to load dataset features and labels")
            return self._features, self._raw_labels

        dataset_dir = self.data_path / self.dataset_name
        filename = self.dataset_info["file"]
        local_path = dataset_dir / filename

        if not local_path.exists():
            self.download()

        if local_path.exists():
            self._load_from_file(local_path)

        if self._features is None:
            # Real load produced nothing usable. Gate before fabricating.
            self._create_synthetic_fallback(reason="download succeeded but no features were parsed")

        # Type guard for mypy - at this point both should be set
        if self._features is None or self._raw_labels is None:
            raise RuntimeError("Failed to load dataset features and labels")

        return self._features, self._raw_labels

    def _load_via_delegate(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load a time-series set through its dedicated loader (real data only).

        The DevNet mirror serves only tabular sets, so smd/swat are routed to
        their own loaders (SMDLoader, SWaTLoader). This fails loud — never
        fabricates — whenever the dedicated loader cannot produce real data,
        including the credentialed case (SWaT requires iTrust registration),
        which is detected up front so we never invoke a loader that cannot
        succeed.
        """
        import importlib

        module_name, cls_name = self._TIMESERIES_DELEGATES[self.dataset_name]
        loader_cls = getattr(importlib.import_module(module_name), cls_name)

        # Single gated chokepoint for every failure mode: ``_create_synthetic_fallback``
        # raises by default (fail loud) and only fabricates — clearly marked
        # synthetic — under MERCURY_ALLOW_SYNTHETIC.
        if getattr(loader_cls, "REQUIRES_CREDENTIALS", False) and not self.config.credentials_path:
            # Detected up front so we never invoke a loader that cannot succeed.
            self._create_synthetic_fallback(
                reason=(
                    f"'{self.dataset_name}' is served by {cls_name}, which requires credentials "
                    f"({getattr(loader_cls, 'DATASET_URL', 'the data provider')}); set "
                    f"DatasetConfig.credentials_path to the registered download."
                )
            )
        else:
            delegate = loader_cls(self.config)
            try:
                delegate.download()
                features, labels = delegate._load_raw()
            except Exception as exc:  # the dedicated loader couldn't get real data
                self._create_synthetic_fallback(
                    reason=(
                        f"delegated to {cls_name}, which could not load real data: "
                        f"{type(exc).__name__}: {exc}"
                    )
                )
            else:
                features = np.asarray(features, dtype=np.float32)
                labels = np.asarray(labels).ravel().astype(np.int64)
                if self.config.max_samples:
                    n = min(len(features), self.config.max_samples)
                    features, labels = features[:n], labels[:n]
                self._features = features
                self._raw_labels = labels
                self._is_real_data = True
                logger.info(
                    "ADRepository %s: loaded %d real samples via %s (real_data=True)",
                    self.dataset_name,
                    len(features),
                    cls_name,
                )

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
            DataSourceUnavailableError: a name reaches here without a DevNet
                fetch path (time-series sets are resolved earlier in _load_raw).
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

        # All time-series sets are resolved in ``_load_raw`` (delegated to a
        # dedicated loader, or failed loud with a closing step) before
        # ``download`` is ever reached, and every other registered name is a
        # DevNet tabular set. Reaching here means a name was added to
        # ADREPOSITORY_DATASETS without a fetch path — fail loud, never fabricate.
        raise DataSourceUnavailableError(
            loader_name=f"ADRepository-{self.dataset_name}",
            reason=(
                "no canonical real-data mirror for this set. Tabular DevNet sets "
                "load from the mala-lab mirror; time-series sets are delegated to "
                "their dedicated loaders in _load_raw."
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

    def _load_from_file(self, path: Path) -> None:
        """Load data from downloaded file."""
        suffix = path.suffix.lower()

        try:
            if suffix == ".npz":
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

                # ``nrows`` bounds memory on large sets (e.g. census 299k x 500)
                # without changing the result: the post-read cap below keeps the
                # same first ``max_samples`` rows, but this avoids materialising
                # the whole CSV first. ``None`` (no cap) reads everything.
                df = pd.read_csv(path, nrows=self.config.max_samples)

                # Assume last column is label
                self._features = df.iloc[:, :-1].values.astype(np.float32)
                self._raw_labels = df.iloc[:, -1].values.astype(np.int64)
                self._is_real_data = True

            elif suffix in (".xz", ".tar"):
                # The DevNet mirror ships several sets as a single CSV inside a
                # ``.tar.xz``. Extract with the tar-slip guard, then load the
                # CSV exactly like the plain-``.csv`` path (last column=label).
                import pandas as pd

                extract_dir = _fresh_extract_dir(path.parent / f"{path.name}_extracted")
                with tarfile.open(path, "r:*") as tf:
                    _safe_extract_tar(tf, extract_dir)

                csv_members = sorted(extract_dir.rglob("*.csv"))
                csv_path = _select_single_csv(csv_members, path.name)
                df = pd.read_csv(csv_path, nrows=self.config.max_samples)
                self._features = df.iloc[:, :-1].values.astype(np.float32)
                self._raw_labels = df.iloc[:, -1].values.astype(np.int64)
                self._is_real_data = True

            elif suffix == ".zip":
                # Extract and load (zip-slip guarded)
                extract_dir = _fresh_extract_dir(path.parent / path.stem)
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
            # ``meta["synthetic"]`` (the harness convention) see the transparent
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

    def _make_loader(dn: str) -> Callable[[DatasetConfig], DatasetLoader]:
        def _factory(cfg: DatasetConfig) -> DatasetLoader:
            return ADRepositoryLoader(cfg, dataset_name=dn)

        return _factory

    DatasetRegistry.register(f"adrepository-{dataset_name}", _make_loader(dataset_name))
