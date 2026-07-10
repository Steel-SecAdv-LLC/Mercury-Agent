# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Train the SeismicWaveAnalyzer on real STEAD waveforms (SeisBench mirror).

Data source (hook ``seismic_wave``, ``EarthquakeDetector.load_neural_weights``):

* **STEAD** -- the STanford EArthquake Dataset (Mousavi et al. 2019, IEEE
  Access), 1,265,657 labeled 60 s three-component 100 Hz traces
  (1,030,231 ``earthquake_local`` + 235,426 ``noise``), served by the public
  SeisBench mirror at ``https://seisbench.gfz.de/mirror/datasets/stead/``
  (fallback mirror: ``hifis-storage.desy.de``). ``metadata.csv``
  (402,560,190 bytes) is downloaded whole and sha256-pinned;
  ``waveforms.hdf5`` (91,127,786,704 bytes) is **never** downloaded whole --
  a seekable HTTP-Range adapter streams only the selected trace subset
  through ``h5py`` (see :class:`BlockCachedRangeReader`).

Verified layout of the mirror's HDF5 (probed 2026-07-10, this pipeline
re-asserts all of it at fetch time and fails loud on any change):

* ``/data/bucket<N>`` datasets of shape ``(n_traces, 3, 6000)`` float32,
  **contiguous and uncompressed** (``chunks=None``), addressed by
  ``trace_name`` strings like ``bucket682$774,:3,:6000``.
* ``/data_format``: ``component_order=b"ZNE"``, ``dimension_order=b"CW"``,
  ``sampling_rate=100``. The vertical (Z) component is therefore **index 0**
  -- NOT index 2 as in the original STEAD distribution (E,N,Z order); the
  SeisBench conversion reorders components. Empirically confirmed on real
  cataloged events (e.g. ``bucket890$227``, M2.7: P-onset/pre-noise ratio
  37.5 on comp0 vs 23.3/19.3 on comps1-2; S onset strongest on the
  horizontals) -- the impulsive P arrival rides the vertical, so comp 0 = Z.

Task: classify earthquake vs noise from the single Z trace the public
detector API consumes, reproducing the detector's **exact** preprocessing
(``scipy.signal.spectrogram`` with ``nperseg=min(256, n//4)``,
``noverlap=min(128, n//8)``, then ``log10(Sxx + 1e-10)`` and per-spectrogram
z-normalization -- see :func:`detector_spectrogram`). Normalization is per
sample only; no cross-sample statistics exist, so nothing can leak from
validation/test years. The merit gate compares the trained network against
the detector's deterministic STA/LTA + band-resonance physics fallback
*through the public* ``predict_earthquake`` *API* on identical held-out
waveforms.

Temporal split (never random -- station deployments and catalog density
autocorrelate across years): train <=2015, validation 2016, test 2017+
(2017, 2018 and 2020; the archive has no 2019 traces). Magnitude head
targets are ``(source_magnitude - 2) / 4`` clamped to [0, 1] to match the
detector's fixed ``mag * 4 + 2`` inference scaling; the documented
consequence is a magnitude floor of 2.0 / ceiling of 6.0 at inference
(58% of STEAD magnitudes are below M2, so magnitude_mae carries that floor
bias -- recorded in the evaluation extras; magnitude is a SECONDARY metric).
Truncated 30 s window variants (first 3000 samples, only when the real
P pick is inside and the real S pick is beyond the cut -- no padding, no
fabrication) teach the s_wave head that P-without-S exists; matching noise
truncations keep window length uninformative for the classifier.
"""

from __future__ import annotations

import io
import json
import logging
import time
from bisect import bisect_right
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

import numpy as np
import torch
from scipy import signal as _scipy_signal

from omni_mercury_engine.datasets.base import http_get_with_retry
from omni_mercury_engine.ml.hazard_training.common import (
    EvaluationOutcome,
    PipelineContext,
    TemporalSplit,
    binary_auc,
    cached_fetch,
    candidate_paths,
    save_candidate,
    save_evaluation,
    seed_everything,
    sha256_file,
    ship_checkpoint,
)

logger = logging.getLogger(__name__)

HOOK_NAME = "seismic_stead"
CHECKPOINT_NAME = "seismic_stead"
FEATURE_SPEC_VERSION = "seismic-stead-v1"

MIRRORS = (
    "https://seisbench.gfz.de/mirror/datasets/stead",
    "https://hifis-storage.desy.de/Helmholtz/HelmholtzAI/SeisBench/datasets/stead",
)
METADATA_FILENAME = "metadata.csv"
WAVEFORMS_FILENAME = "waveforms.hdf5"

#: Upstream sizes, verified 2026-07-10 on both mirrors. fetch() re-checks.
EXPECTED_METADATA_BYTES = 402_560_190
EXPECTED_WAVEFORMS_BYTES = 91_127_786_704

SAMPLING_RATE_HZ = 100.0
WINDOW_SAMPLES = 6000
N_COMPONENTS = 3
#: Z (vertical) is component 0 on the SeisBench mirror (ZNE order, see module
#: docstring for the data_format + impulsive-P verification).
Z_COMPONENT_INDEX = 0
_ROW_BYTES = N_COMPONENTS * WINDOW_SAMPLES * 4  # float32
_Z_BYTES = WINDOW_SAMPLES * 4

#: Truncated-window variant: first TRUNC_SAMPLES of the trace, admissible only
#: when the real P pick is comfortably inside and the real S pick beyond it.
TRUNC_SAMPLES = 3000
_TRUNC_P_MAX = 2800.0
_TRUNC_S_MIN = 3200.0

SPLIT = TemporalSplit(
    train_years=tuple(range(1984, 2016)),
    val_years=(2016,),
    test_years=(2017, 2018, 2019, 2020),
)

#: Balanced per-class subset targets (traces per class per split). Totals
#: 48,000 traces = 48,000 x 24,000 Z bytes ~= 1.1 GiB streamed, safely under
#: the ~4 GiB transfer budget.
SUBSET_TARGETS = {"train": 15_000, "val": 4_000, "test": 5_000}

_METADATA_COLUMNS = [
    "trace_name",
    "trace_category",
    "trace_start_time",
    "source_magnitude",
    "trace_p_arrival_sample",
    "trace_s_arrival_sample",
    "trace_snr_db",
]


class BlockCachedRangeReader(io.RawIOBase):
    """Read-only seekable file over a ``fetch_range(start, end)`` callable.

    Serves ``h5py`` (or any consumer of the file protocol) from two layers:

    * an LRU cache of fixed-size **blocks**, each fetched with one
      block-aligned range request (metadata walks, fallback reads);
    * optional **preloaded ranges** (exact byte extents registered via
      :meth:`add_preload`) so bulk waveform reads planned ahead of time are
      served without per-read round trips.

    The transport is injected, so tests exercise the block math against a
    local file with zero network. Correctness never depends on the preload
    planner: a read missing every preloaded range simply falls back to block
    fetches (counted in ``preload_misses`` so cost regressions are visible).
    """

    def __init__(
        self,
        fetch_range: Callable[[int, int], bytes],
        size: int,
        *,
        block_size: int = 256 * 1024,
        max_cached_blocks: int = 64,
    ) -> None:
        """Initialize the reader.

        Args:
            fetch_range: Callable returning the inclusive byte range
                ``[start, end]`` of the underlying resource.
            size: Total size of the resource in bytes.
            block_size: Bytes per block-aligned fetch.
            max_cached_blocks: LRU capacity in blocks.
        """
        super().__init__()
        if size <= 0 or block_size <= 0 or max_cached_blocks <= 0:
            raise ValueError("size, block_size and max_cached_blocks must be positive")
        self._fetch_range = fetch_range
        self._size = size
        self._block_size = block_size
        self._max_cached_blocks = max_cached_blocks
        self._blocks: OrderedDict[int, bytes] = OrderedDict()
        self._preload_starts: list[int] = []
        self._preloads: dict[int, bytes] = {}
        self._pos = 0
        self.block_fetches = 0
        self.preload_misses = 0

    # -- file protocol -------------------------------------------------------

    def readable(self) -> bool:
        """Report that the stream supports reads (h5py checks this)."""
        return True

    def seekable(self) -> bool:
        """Report that the stream supports seeking (h5py checks this)."""
        return True

    def writable(self) -> bool:
        """Report the stream as read-only."""
        return False

    def tell(self) -> int:
        """Return the current byte position."""
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Move the position; supports SET/CUR/END like a regular file."""
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError(f"unsupported whence {whence}")
        if self._pos < 0:
            raise ValueError("negative seek position")
        return self._pos

    def read(self, size: int = -1) -> bytes:
        """Read up to ``size`` bytes (all remaining bytes when negative)."""
        n = self._size - self._pos if size is None or size < 0 else size
        n = max(0, min(n, self._size - self._pos))
        out = bytearray()
        while n > 0:
            buf, off, avail = self._source_for(self._pos)
            take = min(n, avail)
            out += buf[off : off + take]
            self._pos += take
            n -= take
        return bytes(out)

    def readinto(self, b: Any) -> int:
        """Fill ``b`` from the current position (h5py's fileobj driver path)."""
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    # -- preload layer -------------------------------------------------------

    def add_preload(self, start: int, data: bytes) -> None:
        """Register an exact byte extent so reads inside it skip the network."""
        self._preloads[start] = data
        idx = bisect_right(self._preload_starts, start)
        self._preload_starts.insert(idx, start)

    def clear_preloads(self) -> None:
        """Drop all preloaded extents (frees memory between planned batches)."""
        self._preloads.clear()
        self._preload_starts.clear()

    # -- internals -----------------------------------------------------------

    def _source_for(self, pos: int) -> tuple[bytes, int, int]:
        """Return ``(buffer, offset, available)`` covering ``pos``."""
        i = bisect_right(self._preload_starts, pos) - 1
        if i >= 0:
            start = self._preload_starts[i]
            buf = self._preloads[start]
            if pos < start + len(buf):
                off = pos - start
                return buf, off, len(buf) - off
        if self._preload_starts:
            # A read outside every preloaded extent while preloads are active
            # means the planner missed a range; correctness is preserved by
            # the block fallback, only transfer cost grows.
            self.preload_misses += 1
        block_idx, off = divmod(pos, self._block_size)
        blk = self._get_block(block_idx)
        return blk, off, len(blk) - off

    def _get_block(self, block_idx: int) -> bytes:
        """Fetch (or reuse) the block-aligned extent containing ``block_idx``."""
        blk = self._blocks.get(block_idx)
        if blk is not None:
            self._blocks.move_to_end(block_idx)
            return blk
        start = block_idx * self._block_size
        end = min(start + self._block_size, self._size) - 1
        blk = self._fetch_range(start, end)
        expected = end - start + 1
        if len(blk) != expected:
            raise RuntimeError(
                f"range fetch returned {len(blk)} bytes for [{start}, {end}] "
                f"(expected {expected}); the server ignored the Range header -- "
                "refusing to continue (a full-body response would be 91 GB)"
            )
        self.block_fetches += 1
        self._blocks[block_idx] = blk
        while len(self._blocks) > self._max_cached_blocks:
            self._blocks.popitem(last=False)
        return blk


class _HttpRangeFetcher:
    """Thread-safe HTTP Range transport with mirror failover + byte counting.

    Every byte of ``waveforms.hdf5`` this pipeline touches flows through
    ``http_get_with_retry`` (SafeHTTPClient allowlist, HTTPS-only) via this
    class, so ``bytes_fetched`` is the exact streamed-transfer figure that
    lands in the fetch manifest.
    """

    def __init__(self, filename: str, *, timeout: float = 120.0) -> None:
        """Initialize with the mirror-relative filename to stream."""
        self._filename = filename
        self._timeout = timeout
        self._mirror_idx = 0
        self._lock = Lock()
        self.bytes_fetched = 0
        self.calls = 0

    @property
    def url(self) -> str:
        """Current mirror URL for the file."""
        return f"{MIRRORS[self._mirror_idx]}/{self._filename}"

    def __call__(self, start: int, end: int) -> bytes:
        """Fetch inclusive byte range ``[start, end]``, failing over once."""
        headers = {"Range": f"bytes={start}-{end}"}
        try:
            body = http_get_with_retry(self.url, headers=headers, timeout=self._timeout)
        except Exception:
            if self._mirror_idx + 1 >= len(MIRRORS):
                raise
            logger.warning(
                "range fetch failed on %s; failing over to mirror %s",
                self.url,
                MIRRORS[self._mirror_idx + 1],
            )
            with self._lock:
                self._mirror_idx = min(self._mirror_idx + 1, len(MIRRORS) - 1)
            body = http_get_with_retry(self.url, headers=headers, timeout=self._timeout)
        with self._lock:
            self.bytes_fetched += len(body)
            self.calls += 1
        return body


def detector_spectrogram(trace: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Reproduce ``EarthquakeDetector.predict_earthquake`` preprocessing exactly.

    Keeping this in one helper guarantees train/serve parity for the tensors
    the network is fitted on; the evaluation stage does NOT use it -- it goes
    through the public detector API, which recomputes the identical pipeline.

    Args:
        trace: 1-D waveform (any length >= 4 samples; STEAD Z traces are
            6000 samples @ 100 Hz).

    Returns:
        Per-spectrogram z-normalized ``log10`` spectrogram, float32
        ``[freq_bins, time_bins]`` (129 x 45 for a 6000-sample trace).
    """
    trace = np.asarray(trace, dtype=np.float64)
    _f, _t, sxx = _scipy_signal.spectrogram(
        trace,
        fs=SAMPLING_RATE_HZ,
        nperseg=min(256, len(trace) // 4),
        noverlap=min(128, len(trace) // 8),
    )
    sxx_log = np.log10(sxx + 1e-10)
    sxx_norm = (sxx_log - sxx_log.mean()) / (sxx_log.std() + 1e-10)
    return sxx_norm.astype(np.float32)


# ---------------------------------------------------------------------------
# fetch stage
# ---------------------------------------------------------------------------


def _stead_dir(ctx: PipelineContext) -> Path:
    """On-disk cache directory for the STEAD subset."""
    return ctx.data_dir / "stead"


def _subset_path(ctx: PipelineContext, split: str) -> Path:
    """Path of the cached per-split subset npz (seed-keyed)."""
    return _stead_dir(ctx) / f"subset_seed{ctx.seed}_{split}.npz"


def _manifest_path(ctx: PipelineContext) -> Path:
    """Path of the fetch manifest (seed-keyed, like the subset files)."""
    return _stead_dir(ctx) / f"manifest_seed{ctx.seed}.json"


def _load_metadata(ctx: PipelineContext) -> Any:
    """Load and validate the STEAD metadata CSV (columns this pipeline uses).

    Returns:
        DataFrame with ``year`` (int) and ``snr_db_mean`` (float, NaN for the
        noise class) columns added.

    Raises:
        RuntimeError: On any malformed ``trace_start_time`` / ``trace_name``
            or an unexpected label vocabulary -- refusing to guess.
    """
    import pandas as pd

    path = _stead_dir(ctx) / METADATA_FILENAME
    df = pd.read_csv(path, usecols=_METADATA_COLUMNS, low_memory=False)

    categories = set(df["trace_category"].unique())
    if categories != {"earthquake_local", "noise"}:
        raise RuntimeError(
            f"unexpected trace_category vocabulary {sorted(categories)}; "
            "expected exactly {'earthquake_local', 'noise'} -- refusing to relabel"
        )
    year_str = df["trace_start_time"].astype(str).str[:4]
    if not year_str.str.fullmatch(r"\d{4}").all():
        bad = df.loc[~year_str.str.fullmatch(r"\d{4}"), "trace_start_time"].head(3).tolist()
        raise RuntimeError(f"unparseable trace_start_time values (e.g. {bad}); refusing to guess")
    df["year"] = year_str.astype(int)
    if not df["trace_name"].str.fullmatch(r"bucket\d+\$\d+,:3,:6000").all():
        raise RuntimeError("trace_name format changed upstream; refusing to parse waveform refs")

    def _snr_mean(raw: Any) -> float:
        if not isinstance(raw, str):
            return float("nan")
        try:
            vals = [float(v) for v in raw.strip("[] ").split()]
        except ValueError:
            return float("nan")
        return float(np.mean(vals)) if vals else float("nan")

    df["snr_db_mean"] = df["trace_snr_db"].map(_snr_mean)
    return df


def _select_subset(df: Any, rng: np.random.Generator, limit: int | None) -> dict[str, Any]:
    """Seeded, balanced per-split trace selection with quality filters.

    Earthquake rows must carry a finite P pick and a finite source magnitude
    (in this archive that is 100% of them -- asserted, not assumed). No SNR
    filtering: cherry-picking high-SNR positives would inflate the evaluation,
    so the selection is a uniform seeded sample and the SNR distribution is
    reported in the manifest instead.

    Args:
        df: Metadata frame from :func:`_load_metadata`.
        rng: Seeded generator (selection order is fixed: per split in
            train/val/test order, earthquake class before noise).
        limit: Optional per-class-per-split cap (``ctx.limit_samples``).

    Returns:
        Mapping split name -> selected metadata frame (fetch order: sorted by
        bucket then row for read locality).

    Raises:
        RuntimeError: If any split-class pool is empty (temporal split no
            longer matches the archive -- fail loud, never pad).
    """
    is_eq = df["trace_category"] == "earthquake_local"
    quality = (~is_eq) | (
        np.isfinite(df["trace_p_arrival_sample"].astype(float))
        & np.isfinite(df["source_magnitude"].astype(float))
    )
    dropped = int((~quality).sum())
    if dropped:
        logger.info("quality filter dropped %d earthquake rows (no P pick/magnitude)", dropped)
    df = df[quality]

    masks = dict(zip(("train", "val", "test"), SPLIT.masks(df["year"].to_numpy()), strict=True))
    out: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        target = SUBSET_TARGETS[split]
        if limit is not None:
            target = min(target, limit)
        parts = []
        for want_eq in (True, False):
            pool = df[masks[split] & (is_eq == want_eq)]
            label = "earthquake_local" if want_eq else "noise"
            if len(pool) == 0:
                raise RuntimeError(
                    f"no {label} traces in the {split} years "
                    f"{SPLIT.__getattribute__(split + '_years')}; cannot build an honest split"
                )
            take = min(target, len(pool))
            if take < target:
                logger.warning(
                    "%s/%s pool has only %d traces (target %d); shrinking honestly",
                    split,
                    label,
                    len(pool),
                    target,
                )
            picked = pool.iloc[np.sort(rng.choice(len(pool), size=take, replace=False))]
            parts.append(picked)
        import pandas as pd

        sel = pd.concat(parts, ignore_index=True)
        name_parts = sel["trace_name"].str.split("$", expand=True)
        sel["bucket"] = name_parts[0]
        sel["row"] = name_parts[1].str.split(",").str[0].astype(int)
        out[split] = sel.sort_values(["bucket", "row"], kind="mergesort").reset_index(drop=True)
    return out


def _coalesce_ranges(
    ranges: list[tuple[int, int]], *, max_gap: int = 128 * 1024
) -> list[tuple[int, int]]:
    """Merge sorted inclusive byte ranges whose gap is at most ``max_gap``."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start - merged[-1][1] - 1 <= max_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _assert_stead_layout(h5file: Any) -> None:
    """Fail loud unless the mirror's data_format matches the verified layout."""
    fmt = h5file["data_format"]
    component_order = bytes(fmt["component_order"][()])
    dimension_order = bytes(fmt["dimension_order"][()])
    sampling_rate = float(fmt["sampling_rate"][()])
    if component_order != b"ZNE" or dimension_order != b"CW" or sampling_rate != 100.0:
        raise RuntimeError(
            f"STEAD mirror layout changed: component_order={component_order!r}, "
            f"dimension_order={dimension_order!r}, sampling_rate={sampling_rate}; "
            "expected ZNE/CW/100.0 -- the Z-component index would be wrong, refusing"
        )


def _bucket_dataset(data_group: Any, bucket: str) -> tuple[Any, int]:
    """Return ``(dataset, contiguous byte offset)`` for a bucket, validated."""
    ds = data_group[bucket]
    if ds.shape[1:] != (N_COMPONENTS, WINDOW_SAMPLES) or str(ds.dtype) != "float32":
        raise RuntimeError(
            f"{bucket}: shape {ds.shape} dtype {ds.dtype}; expected (*, 3, 6000) float32"
        )
    if ds.chunks is not None or ds.compression is not None:
        raise RuntimeError(
            f"{bucket} is chunked/compressed; the contiguous-offset prefetch "
            "planner no longer applies -- refusing to blow the transfer budget"
        )
    offset = ds.id.get_offset()
    if offset is None:
        raise RuntimeError(f"{bucket}: h5py reports no contiguous data offset")
    return ds, int(offset)


def _stream_split(
    subset: Any,
    data_group: Any,
    reader: BlockCachedRangeReader,
    fetcher: _HttpRangeFetcher,
    *,
    batch_traces: int = 4096,
    prefetch_workers: int = 24,
) -> np.ndarray[Any, Any]:
    """Stream the Z component for every trace of one split, in fetch order.

    Reads are planned from each bucket's contiguous data offset, coalesced,
    prefetched in parallel, then served to ``h5py`` from the preload layer --
    ``h5py`` still performs all format parsing, so a planner bug can only
    cost bytes, never corrupt data.

    Returns:
        Float32 array ``[n, 6000]`` aligned with ``subset`` rows.

    Raises:
        RuntimeError: On any non-finite sample in a fetched trace (real
            STEAD traces are finite; NaN would mean transport corruption).
    """
    n = len(subset)
    z = np.empty((n, WINDOW_SAMPLES), dtype=np.float32)
    datasets: dict[str, tuple[Any, int]] = {}
    t0 = time.monotonic()
    for batch_start in range(0, n, batch_traces):
        batch = subset.iloc[batch_start : batch_start + batch_traces]
        ranges: list[tuple[int, int]] = []
        for bucket, row in zip(batch["bucket"], batch["row"], strict=True):
            if bucket not in datasets:
                datasets[bucket] = _bucket_dataset(data_group, bucket)
            _, offset = datasets[bucket]
            start = offset + int(row) * _ROW_BYTES
            ranges.append((start, start + _Z_BYTES - 1))
        coalesced = _coalesce_ranges(ranges)
        with ThreadPoolExecutor(max_workers=prefetch_workers) as pool:
            futures = [(s, pool.submit(fetcher, s, e)) for s, e in coalesced]
            for start, fut in futures:
                reader.add_preload(start, fut.result())
        for i, (bucket, row) in enumerate(
            zip(batch["bucket"], batch["row"], strict=True), start=batch_start
        ):
            ds, _ = datasets[bucket]
            trace = ds[int(row), Z_COMPONENT_INDEX, :]
            if not np.all(np.isfinite(trace)):
                raise RuntimeError(
                    f"non-finite samples in {bucket}${row}; refusing corrupted waveform"
                )
            z[i] = trace
        reader.clear_preloads()
        done = min(batch_start + batch_traces, n)
        logger.info(
            "streamed %d/%d traces (%.1f MB fetched, %d calls, %.0fs elapsed)",
            done,
            n,
            fetcher.bytes_fetched / 1e6,
            fetcher.calls,
            time.monotonic() - t0,
        )
    if reader.preload_misses:
        logger.warning(
            "preload planner missed %d reads (served via block fallback; "
            "data is correct, transfer cost was higher than planned)",
            reader.preload_misses,
        )
    return z


def _verify_streamed_traces(
    subset: Any,
    z: np.ndarray[Any, Any],
    rng: np.random.Generator,
    *,
    n_check: int = 8,
) -> int:
    """Re-read a seeded sample of traces through a fresh session and compare.

    The preload planner only redirects transport; this check proves the bytes
    it served decode to the identical waveforms a plain (no-preload) h5py
    session reads. Any mismatch is a hard failure.

    Returns:
        Number of traces verified.
    """
    import h5py  # type: ignore[import-untyped]  # mypy flags first import site only

    idx = rng.choice(len(subset), size=min(n_check, len(subset)), replace=False)
    fetcher = _HttpRangeFetcher(WAVEFORMS_FILENAME)
    reader = BlockCachedRangeReader(fetcher, EXPECTED_WAVEFORMS_BYTES)
    with h5py.File(reader, "r") as h5file:
        data_group = h5file["data"]
        for i in idx:
            row = subset.iloc[int(i)]
            fresh = data_group[row["bucket"]][int(row["row"]), Z_COMPONENT_INDEX, :]
            if not np.array_equal(fresh, z[int(i)]):
                raise RuntimeError(
                    f"verification mismatch for {row['trace_name']}: independently "
                    "re-read waveform differs from the streamed subset -- aborting"
                )
    logger.info(
        "verified %d traces byte-identical via fresh no-preload session (%.1f MB)",
        len(idx),
        fetcher.bytes_fetched / 1e6,
    )
    return len(idx)


def _snr_summary(values: np.ndarray[Any, Any]) -> dict[str, float]:
    """Percentile summary of a (possibly NaN-laden) SNR array in dB."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": 0.0}
    pct = np.percentile(finite, [5, 25, 50, 75, 95])
    return {
        "n": float(finite.size),
        "mean": float(finite.mean()),
        "p5": float(pct[0]),
        "p25": float(pct[1]),
        "p50": float(pct[2]),
        "p75": float(pct[3]),
        "p95": float(pct[4]),
    }


def fetch(ctx: PipelineContext) -> dict[str, Any]:
    """Download STEAD metadata and stream the selected waveform subset.

    Stages: cache ``metadata.csv`` whole (sha256-pinned, size-checked);
    select the seeded balanced subset per temporal split; stream only those
    traces' Z components out of the 91 GB ``waveforms.hdf5`` via HTTP Range
    requests; independently re-verify a seeded sample; write per-split npz
    caches plus a manifest with full provenance (URLs, hashes, byte counts,
    year/SNR distributions).

    Returns:
        The fetch manifest (also written to disk).
    """
    import h5py

    stead_dir = _stead_dir(ctx)
    manifest_file = _manifest_path(ctx)
    if manifest_file.exists() and all(
        _subset_path(ctx, s).exists() for s in ("train", "val", "test")
    ):
        logger.info("fetch cache hit: %s", manifest_file)
        return dict(json.loads(manifest_file.read_text()))

    meta_path = cached_fetch(f"{MIRRORS[0]}/{METADATA_FILENAME}", stead_dir / METADATA_FILENAME)
    if meta_path.stat().st_size != EXPECTED_METADATA_BYTES:
        raise RuntimeError(
            f"metadata.csv is {meta_path.stat().st_size} bytes; expected "
            f"{EXPECTED_METADATA_BYTES} -- upstream changed, refusing to proceed blindly"
        )
    meta_sha = sha256_file(meta_path)

    df = _load_metadata(ctx)
    year_table = {
        str(year): {
            "earthquake_local": int(((df["year"] == year) & eq_mask).sum()),
            "noise": int(((df["year"] == year) & ~eq_mask).sum()),
        }
        for eq_mask in (df["trace_category"] == "earthquake_local",)
        for year in sorted(df["year"].unique())
    }

    rng = seed_everything(ctx.seed)
    subsets = _select_subset(df, rng, ctx.limit_samples)
    del df

    fetcher = _HttpRangeFetcher(WAVEFORMS_FILENAME)
    sig = fetcher(0, 7)
    if sig != b"\x89HDF\r\n\x1a\n":
        raise RuntimeError("waveforms.hdf5 signature mismatch; not an HDF5 file")
    if len(fetcher(EXPECTED_WAVEFORMS_BYTES - 1, EXPECTED_WAVEFORMS_BYTES - 1)) != 1:
        raise RuntimeError("waveforms.hdf5 shorter than the pinned upstream size")

    reader = BlockCachedRangeReader(fetcher, EXPECTED_WAVEFORMS_BYTES)
    subset_records: dict[str, Any] = {}
    trace_lists: dict[str, list[str]] = {}
    snr_report: dict[str, dict[str, float]] = {}
    with h5py.File(reader, "r") as h5file:
        _assert_stead_layout(h5file)
        data_group = h5file["data"]
        for split, subset in subsets.items():
            logger.info("streaming %s split: %d traces", split, len(subset))
            z = _stream_split(subset, data_group, reader, fetcher)
            n_verified = _verify_streamed_traces(subset, z, rng)
            is_eq = (subset["trace_category"] == "earthquake_local").to_numpy()
            path = _subset_path(ctx, split)
            np.savez_compressed(
                path,
                z=z,
                label=is_eq.astype(np.uint8),
                year=subset["year"].to_numpy(np.int32),
                mag=subset["source_magnitude"].to_numpy(np.float32),
                p_sample=subset["trace_p_arrival_sample"].to_numpy(np.float32),
                s_sample=subset["trace_s_arrival_sample"].to_numpy(np.float32),
                snr_db=subset["snr_db_mean"].to_numpy(np.float32),
                trace_name=subset["trace_name"].to_numpy(str),
            )
            trace_lists[split] = subset["trace_name"].tolist()
            snr_report[split] = _snr_summary(subset["snr_db_mean"].to_numpy(np.float64))
            subset_records[split] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "n_earthquake": int(is_eq.sum()),
                "n_noise": int((~is_eq).sum()),
                "years": sorted(int(y) for y in set(subset["year"])),
                "n_independently_verified": n_verified,
            }

    trace_list_path = stead_dir / f"trace_names_seed{ctx.seed}.json"
    trace_list_path.write_text(json.dumps(trace_lists, indent=0, sort_keys=True))

    manifest = {
        "hook": HOOK_NAME,
        "seed": ctx.seed,
        "sources": [
            {
                "url": f"{MIRRORS[0]}/{METADATA_FILENAME}",
                "sha256": meta_sha,
                "bytes": EXPECTED_METADATA_BYTES,
                "description": "STEAD metadata (SeisBench GFZ mirror), all 1,265,657 rows",
            },
            {
                "url": fetcher.url,
                "sha256": "streamed-subset:see subset_files",
                "upstream_bytes": EXPECTED_WAVEFORMS_BYTES,
                "description": (
                    "STEAD waveforms.hdf5 -- Z-component subset streamed via HTTP "
                    "Range requests; per-split npz sha256 digests under subset_files, "
                    "exact trace list in trace_names file"
                ),
            },
        ],
        "subset_files": subset_records,
        "trace_list": {"path": str(trace_list_path), "sha256": sha256_file(trace_list_path)},
        "component": {"order": "ZNE", "dimension_order": "CW", "used_index": Z_COMPONENT_INDEX},
        "bytes_streamed_hdf5": fetcher.bytes_fetched,
        "http_range_calls": fetcher.calls,
        "snr_db_mean_by_split": snr_report,
        "year_distribution_full_archive": year_table,
        "split_years": {
            "train": list(SPLIT.train_years),
            "val": list(SPLIT.val_years),
            "test": list(SPLIT.test_years),
        },
    }
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    logger.info(
        "fetch complete: %.2f GB streamed in %d range calls; manifest %s",
        fetcher.bytes_fetched / 1e9,
        fetcher.calls,
        manifest_file,
    )
    return manifest


# ---------------------------------------------------------------------------
# build stage
# ---------------------------------------------------------------------------


@dataclass
class SpectrogramGroup:
    """One same-shape batch group of spectrograms with aligned labels.

    Attributes:
        spec: Float32 spectrograms ``[n, freq, time]``.
        label: 1.0 earthquake / 0.0 noise.
        mag_target: ``clip((magnitude - 2) / 4, 0, 1)``; NaN for noise (the
            magnitude loss is masked to positives).
        p_label: 1.0 when a real P pick lies inside the window.
        s_label: 1.0 when a real S pick lies inside the window.
    """

    spec: np.ndarray[Any, Any]
    label: np.ndarray[Any, Any]
    mag_target: np.ndarray[Any, Any]
    p_label: np.ndarray[Any, Any]
    s_label: np.ndarray[Any, Any]

    @property
    def n(self) -> int:
        """Number of samples in the group."""
        return int(self.spec.shape[0])


@dataclass
class SeismicDataset:
    """Training-side dataset: precomputed spectrograms for train and val.

    The test split is deliberately absent: :func:`evaluate` consumes raw
    held-out waveforms through the public detector API, never precomputed
    tensors.
    """

    train_full: SpectrogramGroup
    train_trunc: SpectrogramGroup
    val: SpectrogramGroup


def _load_subset(ctx: PipelineContext, split: str) -> dict[str, np.ndarray[Any, Any]]:
    """Load one split's cached subset npz, failing loud when absent."""
    path = _subset_path(ctx, split)
    if not path.exists():
        raise FileNotFoundError(f"missing subset cache {path}; run the --fetch stage first")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _mag_target(mag: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Magnitude head target: ``(M - 2) / 4`` clamped to [0, 1] (see module doc)."""
    return np.clip((mag - 2.0) / 4.0, 0.0, 1.0).astype(np.float32)


def _group_from_arrays(
    arrays: dict[str, np.ndarray[Any, Any]],
    idx: np.ndarray[Any, Any],
    *,
    window: int,
) -> SpectrogramGroup:
    """Build a :class:`SpectrogramGroup` for ``idx`` rows cut to ``window`` samples.

    Labels are derived from the REAL pick columns: a pick counts as present
    only when its sample index falls inside the window actually fed to the
    spectrogram. Noise rows have NaN picks and label 0 for both heads.
    """
    z = arrays["z"][idx, :window]
    label = arrays["label"][idx].astype(np.float32)
    p_sample = arrays["p_sample"][idx]
    s_sample = arrays["s_sample"][idx]
    p_label = (np.nan_to_num(p_sample, nan=np.inf) < window).astype(np.float32)
    s_label = (np.nan_to_num(s_sample, nan=np.inf) < window).astype(np.float32)
    mag_target = np.where(label > 0.5, _mag_target(arrays["mag"][idx]), np.float32(np.nan)).astype(
        np.float32
    )
    if len(idx) == 0:
        # Empty group (e.g. no truncation-eligible rows in a tiny run): keep
        # the correct spectrogram shape via a shape probe -- no sample rows.
        shape = detector_spectrogram(np.zeros(window, dtype=np.float32)).shape
        spec = np.zeros((0, *shape), dtype=np.float32)
    else:
        spec = np.stack([detector_spectrogram(trace) for trace in z])
    return SpectrogramGroup(
        spec=spec, label=label, mag_target=mag_target, p_label=p_label, s_label=s_label
    )


def build_dataset(ctx: PipelineContext) -> SeismicDataset:
    """Assemble train/val spectrogram groups from the cached subsets.

    Preprocessing is byte-for-byte the detector's own pipeline
    (:func:`detector_spectrogram`); the only cross-sample choice is the
    seeded truncated-window assignment for the train split, drawn from
    ``ctx.seed`` so the build is reproducible. Spectrograms are cached to
    disk keyed by seed.

    Returns:
        The training-side dataset.
    """
    cache = _stead_dir(ctx) / f"spectrograms_seed{ctx.seed}.npz"
    if cache.exists():
        logger.info("spectrogram cache hit: %s", cache)
        with np.load(cache, allow_pickle=False) as data:
            return SeismicDataset(
                train_full=SpectrogramGroup(
                    *(data[f"train_full_{k}"] for k in ("spec", "label", "mag", "p", "s"))
                ),
                train_trunc=SpectrogramGroup(
                    *(data[f"train_trunc_{k}"] for k in ("spec", "label", "mag", "p", "s"))
                ),
                val=SpectrogramGroup(
                    *(data[f"val_{k}"] for k in ("spec", "label", "mag", "p", "s"))
                ),
            )

    train = _load_subset(ctx, "train")
    val = _load_subset(ctx, "val")
    rng = np.random.default_rng([ctx.seed, 1])  # independent of the fetch stream

    is_eq = train["label"].astype(bool)
    p_sample = train["p_sample"]
    s_sample = train["s_sample"]
    eligible_eq = np.flatnonzero(
        is_eq
        & (np.nan_to_num(p_sample, nan=np.inf) <= _TRUNC_P_MAX)
        & (np.nan_to_num(s_sample, nan=-np.inf) >= _TRUNC_S_MIN)
    )
    trunc_eq = rng.choice(eligible_eq, size=len(eligible_eq) // 2, replace=False)
    noise_idx = np.flatnonzero(~is_eq)
    trunc_noise = rng.choice(noise_idx, size=min(len(trunc_eq), len(noise_idx)), replace=False)
    trunc_idx = np.sort(np.concatenate([trunc_eq, trunc_noise]))
    full_idx = np.setdiff1d(np.arange(len(is_eq)), trunc_idx)
    logger.info(
        "train truncation: %d eligible eq, %d eq + %d noise truncated to %d samples",
        len(eligible_eq),
        len(trunc_eq),
        len(trunc_noise),
        TRUNC_SAMPLES,
    )

    ds = SeismicDataset(
        train_full=_group_from_arrays(train, full_idx, window=WINDOW_SAMPLES),
        train_trunc=_group_from_arrays(train, trunc_idx, window=TRUNC_SAMPLES),
        val=_group_from_arrays(val, np.arange(len(val["label"])), window=WINDOW_SAMPLES),
    )
    np.savez(
        cache,
        **{
            f"{name}_{k}": getattr(group, attr)
            for name, group in (
                ("train_full", ds.train_full),
                ("train_trunc", ds.train_trunc),
                ("val", ds.val),
            )
            for k, attr in (
                ("spec", "spec"),
                ("label", "label"),
                ("mag", "mag_target"),
                ("p", "p_label"),
                ("s", "s_label"),
            )
        },
    )
    logger.info(
        "built spectrograms: train_full=%d train_trunc=%d val=%d",
        ds.train_full.n,
        ds.train_trunc.n,
        ds.val.n,
    )
    return ds


# ---------------------------------------------------------------------------
# train stage
# ---------------------------------------------------------------------------


def _group_tensors(group: SpectrogramGroup) -> dict[str, torch.Tensor]:
    """Torch views of a group (spectrograms get the CNN channel dim)."""
    return {
        "spec": torch.from_numpy(group.spec).unsqueeze(1),
        "label": torch.from_numpy(group.label),
        "mag": torch.from_numpy(group.mag_target),
        "p": torch.from_numpy(group.p_label),
        "s": torch.from_numpy(group.s_label),
    }


def _val_auc(model: Any, val: dict[str, torch.Tensor], batch_size: int = 256) -> float:
    """Classifier AUC on the validation split (model-level, full windows)."""
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, val["spec"].shape[0], batch_size):
            eq_prob, _, _, _ = model(val["spec"][start : start + batch_size])
            scores.append(eq_prob.reshape(-1).numpy())
    return binary_auc(val["label"].numpy(), np.concatenate(scores))


def train(ctx: PipelineContext) -> dict[str, Any]:
    """Train the detector's own SeismicWaveAnalyzer with early stopping.

    Losses: BCE on the earthquake classifier, BCE on the P/S-pick-presence
    heads, MSE on the magnitude head masked to positives. Batches are drawn
    within a spectrogram-shape group (full 60 s vs truncated 30 s windows)
    and the batch order is shuffled across groups each epoch. Early stopping
    maximizes validation AUC (patience 6) under the ``ctx.max_epochs`` cap.

    Returns:
        Training record (epochs run, best validation AUC, sample counts).
    """
    from omni_mercury_engine.detectors.geological.disaster_detectors import SeismicWaveAnalyzer

    rng = seed_everything(ctx.seed)
    ds = build_dataset(ctx)
    groups = [_group_tensors(g) for g in (ds.train_full, ds.train_trunc) if g.n > 0]
    val = _group_tensors(ds.val)

    model = SeismicWaveAnalyzer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCELoss()
    batch_size = 64
    n_train = sum(int(g["spec"].shape[0]) for g in groups)
    pos_frac = float(sum(float(g["label"].sum()) for g in groups) / max(1, n_train))
    logger.info(
        "training on %d spectrograms (%.1f%% earthquake) in %d shape groups; val=%d",
        n_train,
        100 * pos_frac,
        len(groups),
        int(val["spec"].shape[0]),
    )

    best_auc = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    patience, bad_epochs = 6, 0
    epochs_run = 0
    for epoch in range(ctx.max_epochs):
        epochs_run = epoch + 1
        model.train()
        batches: list[tuple[int, torch.Tensor]] = []
        for gi, g in enumerate(groups):
            perm = torch.from_numpy(rng.permutation(int(g["spec"].shape[0])))
            batches.extend((gi, perm[s : s + batch_size]) for s in range(0, len(perm), batch_size))
        order = rng.permutation(len(batches))
        epoch_loss = 0.0
        for bi in order:
            gi, idx = batches[int(bi)]
            g = groups[gi]
            eq_prob, mag, p_prob, s_prob = model(g["spec"][idx])
            eq_prob = eq_prob.reshape(-1).clamp(1e-6, 1 - 1e-6)
            p_prob = p_prob.reshape(-1).clamp(1e-6, 1 - 1e-6)
            s_prob = s_prob.reshape(-1).clamp(1e-6, 1 - 1e-6)
            loss = (
                bce(eq_prob, g["label"][idx]) + bce(p_prob, g["p"][idx]) + bce(s_prob, g["s"][idx])
            )
            mag_true = g["mag"][idx]
            pos = torch.isfinite(mag_true)
            if bool(pos.any()):
                loss = loss + torch.nn.functional.mse_loss(mag.reshape(-1)[pos], mag_true[pos])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * int(idx.shape[0])

        auc = _val_auc(model, val)
        logger.info("epoch %d: train loss %.4f, val AUC %.5f", epoch + 1, epoch_loss / n_train, auc)
        if auc > best_auc + 1e-5:
            best_auc = auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info("early stop at epoch %d (patience %d)", epoch + 1, patience)
                break

    if best_state is None or not np.isfinite(best_auc):
        raise RuntimeError("training produced no finite validation AUC; refusing to save")
    model.load_state_dict(best_state)

    record = {
        "seed": ctx.seed,
        "epochs_run": epochs_run,
        "best_val_auc": float(best_auc),
        "n_train": n_train,
        "n_train_truncated": ds.train_trunc.n,
        "n_val": ds.val.n,
        "train_years": list(SPLIT.train_years),
        "val_years": list(SPLIT.val_years),
        "train_earthquake_fraction": pos_frac,
    }
    payload: dict[str, Any] = {
        "seismic_analyzer": model.state_dict(),
        "feature_spec": FEATURE_SPEC_VERSION,
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "window_samples": WINDOW_SAMPLES,
        "component": "Z",
    }
    save_candidate(ctx.data_dir, HOOK_NAME, payload, record)
    return record


# ---------------------------------------------------------------------------
# evaluate stage
# ---------------------------------------------------------------------------

_EVAL_DETECTORS: dict[str, Any] = {}


def _eval_worker_init(candidate_path: str) -> None:
    """Per-process initializer: build one physics and one learned detector."""
    torch.set_num_threads(1)
    from omni_mercury_engine.detectors.geological.disaster_detectors import EarthquakeDetector

    physics = EarthquakeDetector()
    learned = EarthquakeDetector()
    learned.load_neural_weights(candidate_path)
    _EVAL_DETECTORS["physics"] = physics
    _EVAL_DETECTORS["learned"] = learned


def _eval_worker(chunk: np.ndarray[Any, Any]) -> dict[str, list[float]]:
    """Run a chunk of held-out waveforms through both detector paths.

    Returns:
        Per-path lists (confidence, detected, p/s flags, magnitude) aligned
        with the chunk order; magnitude is NaN where the path abstains.
    """
    out: dict[str, list[float]] = {
        f"{label}_{key}": []
        for label in ("physics", "learned")
        for key in ("conf", "det", "p", "s", "mag")
    }
    for trace in chunk:
        for label in ("physics", "learned"):
            result = _EVAL_DETECTORS[label].predict_earthquake(trace)
            out[f"{label}_conf"].append(float(result.confidence))
            out[f"{label}_det"].append(float(result.earthquake_detected))
            out[f"{label}_p"].append(float(result.p_wave_detected))
            out[f"{label}_s"].append(float(result.s_wave_detected))
            mag = result.estimated_magnitude
            out[f"{label}_mag"].append(float("nan") if mag is None else float(mag))
    return out


def evaluate(ctx: PipelineContext) -> EvaluationOutcome:
    """Compare learned vs physics through the public detector API.

    Both paths receive the *identical* held-out Z waveforms for every test
    trace (years 2017+). Physics is a fresh :class:`EarthquakeDetector` with
    no weights (STA/LTA trigger + band resonance, magnitude abstained);
    learned is an identical detector after ``load_neural_weights`` on the
    candidate checkpoint. Primary metric: classification AUC of the public
    ``confidence`` field (higher is better).

    Returns:
        The evaluation outcome (persisted next to the candidate).
    """
    import multiprocessing as mp

    cand_path, _ = candidate_paths(ctx.data_dir, HOOK_NAME)
    if not cand_path.exists():
        raise FileNotFoundError(f"no candidate checkpoint at {cand_path}; run --train first")

    test = _load_subset(ctx, "test")
    n_test = len(test["label"])
    if ctx.limit_samples is not None:
        n_test = min(n_test, ctx.limit_samples)
    z = test["z"][:n_test]
    label = test["label"][:n_test].astype(np.float64)
    mag_true = test["mag"][:n_test].astype(np.float64)
    p_present = (np.nan_to_num(test["p_sample"][:n_test], nan=np.inf) < WINDOW_SAMPLES).astype(
        np.float64
    )
    s_present = (np.nan_to_num(test["s_sample"][:n_test], nan=np.inf) < WINDOW_SAMPLES).astype(
        np.float64
    )

    chunks = [z[start : start + 64] for start in range(0, n_test, 64)]
    t0 = time.monotonic()
    n_workers = min(4, mp.cpu_count())
    # fork (not spawn/forkserver): those two re-import the caller's __main__,
    # which breaks REPL/heredoc callers. Fork-after-OpenMP hangs are avoided
    # because each child pins torch.set_num_threads(1) before its first torch
    # op, so no OpenMP parallel region is ever entered in the children.
    mp_ctx = mp.get_context("fork")
    with mp_ctx.Pool(
        processes=n_workers, initializer=_eval_worker_init, initargs=(str(cand_path),)
    ) as pool:
        results = pool.map(_eval_worker, chunks)
    collected = {
        key: np.asarray([v for r in results for v in r[key]], dtype=np.float64)
        for key in results[0]
    }
    logger.info(
        "evaluated %d held-out traces through predict_earthquake x2 paths in %.0fs",
        n_test,
        time.monotonic() - t0,
    )

    def _metrics(path_label: str) -> dict[str, float]:
        conf = collected[f"{path_label}_conf"]
        det = collected[f"{path_label}_det"]
        mag_est = collected[f"{path_label}_mag"]
        is_eq = label == 1.0
        mag_err = np.abs(mag_est[is_eq] - mag_true[is_eq])
        finite_err = mag_err[np.isfinite(mag_err)]
        return {
            "auc": binary_auc(label, conf),
            "recall_at_0.96": float(det[is_eq].mean()),
            "false_alarm_rate_at_0.96": float(det[~is_eq].mean()),
            "p_wave_accuracy": float((collected[f"{path_label}_p"] == p_present).mean()),
            "s_wave_accuracy": float((collected[f"{path_label}_s"] == s_present).mean()),
            "magnitude_mae": (float(finite_err.mean()) if finite_err.size else float("nan")),
        }

    manifest_file = _manifest_path(ctx)
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
    test_years = sorted({int(y) for y in test["year"][:n_test]})
    outcome = EvaluationOutcome(
        hook=HOOK_NAME,
        primary_metric="auc",
        higher_is_better=True,
        learned=_metrics("learned"),
        physics=_metrics("physics"),
        n_test_samples=int(n_test),
        test_years=SPLIT.test_years,
        extras={
            "n_test_earthquake": int(label.sum()),
            "n_test_noise": int((1.0 - label).sum()),
            "test_years_observed": test_years,
            "test_earthquake_snr_db_mean": _snr_summary(
                test["snr_db"][:n_test][label == 1.0].astype(np.float64)
            ),
            "bytes_streamed_hdf5": manifest.get("bytes_streamed_hdf5"),
            "physics_magnitude_abstention": (
                "the physics fallback emits estimated_magnitude=None by design "
                "(an uncalibrated single station has no honest Richter estimate), "
                "so physics magnitude_mae is NaN; magnitude_mae is a SECONDARY "
                "metric and does not enter the merit gate"
            ),
            "magnitude_scale_note": (
                "learned magnitudes are clamped to [2, 6] by the detector's "
                "mag*4+2 scaling of the [0,1]-trained head; 58% of STEAD "
                "magnitudes are below M2, so the learned MAE carries that floor"
            ),
            "comparison": (
                "identical held-out STEAD Z waveforms through "
                "EarthquakeDetector.predict_earthquake, physics fallback vs "
                "loaded candidate checkpoint"
            ),
        },
    )
    save_evaluation(ctx.data_dir, outcome)
    logger.info(
        "evaluation: learned AUC %.5f vs physics %.5f on %d held-out traces (%s)",
        outcome.learned["auc"],
        outcome.physics["auc"],
        outcome.n_test_samples,
        "LEARNED WINS" if outcome.learned_beats_physics else "PHYSICS WINS",
    )
    return outcome


# ---------------------------------------------------------------------------
# ship stage
# ---------------------------------------------------------------------------


def ship(ctx: PipelineContext) -> tuple[Any, Any]:
    """Promote the candidate through the merit gate (may refuse loudly)."""
    from omni_mercury_engine.ml.hazard_training.common import load_evaluation

    outcome = load_evaluation(ctx.data_dir, HOOK_NAME)
    manifest_file = _manifest_path(ctx)
    if not manifest_file.exists():
        raise FileNotFoundError(f"missing fetch manifest {manifest_file}; run --fetch first")
    manifest = json.loads(manifest_file.read_text())
    data_sources = [dict(src) for src in manifest["sources"]]
    for split, record in manifest["subset_files"].items():
        data_sources.append(
            {
                "url": f"{manifest['sources'][1]['url']}#subset-{split}",
                "sha256": record["sha256"],
                "description": (
                    f"STEAD {split} subset npz ({record['n_earthquake']} earthquake + "
                    f"{record['n_noise']} noise Z traces, years {record['years']})"
                ),
            }
        )
    data_sources.append(
        {
            "url": f"{manifest['sources'][1]['url']}#trace-names",
            "sha256": manifest["trace_list"]["sha256"],
            "description": "exact trace_name list per split (subset provenance)",
        }
    )
    return ship_checkpoint(
        hook=HOOK_NAME,
        checkpoint_name=CHECKPOINT_NAME,
        data_dir=ctx.data_dir,
        outcome=outcome,
        data_sources=data_sources,
        seed=ctx.seed,
        out_dir=ctx.ship_dir,
    )
