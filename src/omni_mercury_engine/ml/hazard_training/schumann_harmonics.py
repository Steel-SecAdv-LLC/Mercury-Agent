# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Train the SchumannHarmonicAnalyzer on real Sierra Nevada ELF station data.

Data sources (hook ``schumann_harmonics``,
``SchumannResonanceDetector.load_neural_weights``):

* **Sierra Nevada ELF station corpus** (Zenodo records 6348691 / 6348773 /
  6348838 / 6348930; CC-BY-4.0; Salinas et al., Computers & Geosciences
  165:105148, 2022) -- raw little-endian int16 ADC records from the Sierra
  Nevada (Spain) ELF observatory, one ~hour-long file per sensor at
  fs = 256 Hz (921,600 samples). The year archives are 26-28 GB each, so this
  pipeline never downloads a whole ZIP: it random-accesses members over HTTP
  ranged GETs (ZIP64-aware end-of-central-directory parsing, per-member raw
  deflate decompression) and transfers only the byte prefixes it needs.
  Sensor 0 (NS magnetic component) is used; the published station docs report
  an approximately flat response over ~6-25 Hz.
* **GFZ Potsdam definitive Kp** (``https://kp.gfz.de/app/json/``) -- the
  external, real, non-circular label source: an hour is geomagnetically
  DISTURBED when its 3-hour Kp bin is >= 5 and QUIET when <= 2; intermediate
  hours (2 < Kp < 5) are excluded from train and eval and the exclusion is
  reported (clean-contrast first detector).

The simulated BGS client (``data_sources/geomagnetic.py``) is NEVER training
data -- this pipeline reads only the measured Sierra Nevada corpus.

Train/serve parity: the detector's public API turns an ELF signal into the
neural input via ``_compute_power_spectrum`` (full-signal FFT, one-sided
``|fft|**2``, max-normalized) followed by ``power_spectrum[:512]`` and a
float32 cast. :func:`compute_detector_spectrum` re-derives that byte-for-byte,
and the parity is unit-tested against the detector.

Window-length derivation (fs = 256 Hz, 512 spectrum bins): the detector feeds
the FIRST 512 one-sided FFT bins to the network, which span
``512 * fs / n`` Hz for an n-sample window. Requiring coverage of all five
Schumann modes (7.8 / 14.3 / 20.8 / 27.3 / 33.8 Hz) and the detector's
5-40 Hz physics band means ``512 * 256 / n >= 40`` => ``n <= 3276.8``;
choosing ``n = 3072`` (= 3 * 2**10, a 12.0 s window, fast FFT length) gives
bin spacing ``df = 256 / 3072 = 1/12 Hz`` and a covered span of
0..42.67 Hz -- all modes plus margin, with resolution ~10x finer than the
detector's 0.81 Hz frequency-anomaly threshold.

Temporal split (never random; Kp autocorrelates over days and the corpus
spans a solar-cycle descent): train 2013-2014, validation 2015 (contains the
2015-03-17 St Patrick's Day G4 and 2015-06-22 G4 storms), test 2016-2017Feb.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import re
import struct
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.security.safe_torch import safe_torch_load

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

import numpy as np
import torch
from scipy.fft import fft, fftfreq

from omni_mercury_engine.datasets.base import http_get_with_retry
from omni_mercury_engine.ml.hazard_training.common import (
    EvaluationOutcome,
    PipelineContext,
    TemporalSplit,
    binary_auc,
    brier_score,
    cached_fetch,
    candidate_paths,
    save_candidate,
    save_evaluation,
    seed_everything,
    sha256_file,
    ship_checkpoint,
)

logger = logging.getLogger(__name__)

HOOK_NAME = "schumann_harmonics"
CHECKPOINT_NAME = "schumann_sierra_nevada"
FEATURE_SPEC_VERSION = "schumann-sn-v1"

#: Station sampling rate (Hz). The per-file ``*_info.txt`` sidecars state a
#: sampling period of 3906 usec (nominal 1/256 s); verified during fetch.
FS_HZ = 256.0
#: Samples per analysis window (12.0 s at 256 Hz) -- see module docstring.
WINDOW_SAMPLES = 3072
#: Neural input width: the detector feeds ``power_spectrum[:512]``.
SPECTRUM_BINS = 512
#: Windows extracted per hour file, spaced WINDOW_STRIDE samples apart.
WINDOWS_PER_HOUR = 12
#: Stride between window starts (64 s), so 12 windows cover the first
#: ~12.2 minutes of each hour file (prefix-only ranged fetches).
WINDOW_STRIDE = 16384
#: Raw int16 samples (and bytes) needed from each hour file.
PREFIX_SAMPLES = (WINDOWS_PER_HOUR - 1) * WINDOW_STRIDE + WINDOW_SAMPLES
PREFIX_BYTES = PREFIX_SAMPLES * 2
#: Expected uncompressed size of a complete hour file (921,600 int16).
HOUR_FILE_BYTES = 921_600 * 2

#: Group-fetch coalescing: request setup on these ranged GETs costs seconds
#: (~15 s under Zenodo load) while the stream then runs at ~6 MiB/s, so
#: reading through gaps between wanted members is often cheaper than
#: re-requesting. These tolerances keep the measured whole-plan transfer at
#: ~2.5 GiB, inside the 6 GiB budget; spans are capped to bound the duration
#: of any single request.
GROUP_GAP_BYTES = 8 * (1 << 20)
GROUP_SPAN_BYTES = 64 * (1 << 20)

#: Kp class thresholds: disturbed >= 5, quiet <= 2; 2 < Kp < 5 excluded.
KP_DISTURBED = 5.0
KP_QUIET = 2.0
#: Validation recall floor for operating-point selection: a threshold that
#: detects almost nothing can be technically non-regressing when physics
#: itself detects little on validation, yet operationally useless
#: (see :func:`_select_operating_point`; mirrors the tsunami/solar policy).
OPERATING_POINT_RECALL_FLOOR = 0.5
#: Solar's ratified 20% FAR headroom against val->test distribution shift.
OPERATING_POINT_FAR_HEADROOM = 0.8
#: Hour-level deployed rule (both paths): an hour is flagged disturbed when
#: at least this fraction of its 12 windows report ``anomaly_detected``.
HOUR_VOTE_FRACTION = 0.5

#: Documented storm names for the per-event hit tables (matched on the UTC
#: date an event's first sampled hour falls on, +/- 1 day). Only widely
#: documented events are named; everything else reports its peak Kp and the
#: NOAA G-scale derived from it.
NAMED_STORMS: dict[str, str] = {
    "2015-03-17": "St Patrick's Day storm (G4)",
    "2015-06-22": "22-23 June 2015 storm (G4)",
}
#: Quiet hours sampled per disturbed hour (stratified matching).
QUIET_RATIO = 1.5
#: Hard ceiling on total HTTP bytes fetched by this pipeline.
MAX_FETCH_BYTES = 6 * (1 << 30)

#: Station local time approximated as UTC+1 (Spain standard time, fixed;
#: DST deliberately ignored so the stratification is deterministic).
LOCAL_UTC_OFFSET_HOURS = 1

SPLIT = TemporalSplit(train_years=(2013, 2014), val_years=(2015,), test_years=(2016, 2017))

#: Corpus era covered by the archives: 2013-03 .. 2017-02.
ERA_START = _dt.datetime(2013, 3, 1, tzinfo=_dt.UTC)
ERA_END = _dt.datetime(2017, 3, 1, tzinfo=_dt.UTC)

#: Year archives (Zenodo API file-content URLs; Range-supported).
ZIP_ARCHIVES: dict[str, str] = {
    "2014": "https://zenodo.org/api/records/6348691/files/2014.zip/content",
    "2015": "https://zenodo.org/api/records/6348773/files/2015.zip/content",
    "2016": "https://zenodo.org/api/records/6348838/files/2016.zip/content",
    "2013_2017": "https://zenodo.org/api/records/6348930/files/2013_2017.zip/content",
}

KP_URL_TEMPLATE = (
    "https://kp.gfz.de/app/json/?start={year}-01-01T00:00:00Z"
    "&end={year}-12-31T21:00:00Z&index=Kp"
)

ANOMALY_CLASSES = ("normal", "amplitude", "frequency", "combined")

#: Deterministic sub-class rule for disturbed hours (see :func:`derive_class_labels`).
CLASS_RULE = (
    "quiet (Kp<=2) -> normal; disturbed (Kp>=5) sub-typed against the "
    "train-years quiet climatology (mean/std over train quiet hours of "
    "hour-mean log10 band power 5-40 Hz and hour-mean 6-10 Hz spectral "
    "centroid, raw periodogram |fft|^2/n of demeaned windows): "
    "z_amp=|dlog_bp|/sigma, z_freq=|dcentroid|/sigma; both>2 -> combined, "
    "z_freq>2 -> frequency, z_amp>2 -> amplitude, else the larger z decides "
    "(amplitude on ties); 2<Kp<5 hours are excluded entirely."
)

_MEMBER_RE = re.compile(r"^\d{4}/\d{4}/smplGRTU1_sensor_(?P<sensor>[01])_(?P<ts>\d{10})$")

_EOCD_SIG = b"PK\x05\x06"
_EOCD64_LOC_SIG = b"PK\x06\x07"
_EOCD64_SIG = 0x06064B50
_CDH_SIG = b"PK\x01\x02"
_LFH_SIG = 0x04034B50


# ---------------------------------------------------------------------------
# Remote-ZIP random access over HTTP ranged GETs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZipMember:
    """One central-directory entry of a (possibly remote) ZIP archive."""

    name: str
    header_offset: int
    compressed_size: int
    uncompressed_size: int
    method: int
    crc32: int


class RemoteZipReader:
    """Random access into a ZIP archive through a byte-range fetcher.

    The archive is never downloaded whole: the end-of-central-directory
    record (ZIP64-aware -- the year archives exceed 4 GB) locates the central
    directory, which is fetched in one ranged GET; each member is then read
    with per-member ranged GETs plus raw-deflate decompression
    (``zlib, wbits=-15``) or stored-copy.

    Args:
        fetch_range: Callable mapping an HTTP Range *spec* (``"start-end"``
            inclusive, or a suffix ``"-N"`` for the last N bytes) to the
            returned bytes. Production wires this to
            :func:`omni_mercury_engine.datasets.base.http_get_with_retry`
            with a ``Range`` header; tests wire it to a local file.
    """

    def __init__(self, fetch_range: Callable[[str], bytes]) -> None:
        """Store the range fetcher; the directory is loaded lazily."""
        self._fetch = fetch_range
        self._members: dict[str, ZipMember] | None = None

    @property
    def members(self) -> dict[str, ZipMember]:
        """Member table (name -> :class:`ZipMember`), loading it on first use."""
        if self._members is None:
            self._members = self._load_directory()
        return self._members

    def _load_directory(self) -> dict[str, ZipMember]:
        """Fetch and parse the (ZIP64-aware) central directory."""
        # EOCD lives in the last 22..22+65535 bytes; 128 KiB covers it and,
        # for these archives (empty comments), the ZIP64 locator too. A
        # suffix range longer than the file returns the whole file (RFC 9110).
        tail = self._fetch("-131072")
        eocd_pos = tail.rfind(_EOCD_SIG)
        if eocd_pos < 0:
            raise RuntimeError("no end-of-central-directory record in ZIP tail")
        _sig, _dn, _cdd, _nd, n_entries, cd_size, cd_offset, _clen = struct.unpack(
            "<IHHHHIIH", tail[eocd_pos : eocd_pos + 22]
        )
        loc_pos = tail.rfind(_EOCD64_LOC_SIG, 0, eocd_pos)
        needs_zip64 = 0xFFFFFFFF in (cd_size, cd_offset) or n_entries == 0xFFFF
        if loc_pos >= 0:
            _lsig, _ldisk, eocd64_offset, _lnd = struct.unpack(
                "<IIQI", tail[loc_pos : loc_pos + 20]
            )
            rec = self._fetch(f"{eocd64_offset}-{eocd64_offset + 55}")
            sig64, _rsz, _vm, _vn, _d1, _d2, _n1, n_total64, cd_size64, cd_offset64 = struct.unpack(
                "<IQHHIIQQQQ", rec[:56]
            )
            if sig64 != _EOCD64_SIG:
                raise RuntimeError(f"bad ZIP64 EOCD signature {sig64:#x}")
            n_entries, cd_size, cd_offset = int(n_total64), int(cd_size64), int(cd_offset64)
        elif needs_zip64:
            raise RuntimeError("ZIP64 sizes flagged but no ZIP64 locator found")

        cd = self._fetch(f"{cd_offset}-{cd_offset + cd_size - 1}")
        if len(cd) != cd_size:
            raise RuntimeError(f"central directory truncated: {len(cd)} != {cd_size}")
        members: dict[str, ZipMember] = {}
        off = 0
        while off + 46 <= len(cd):
            if cd[off : off + 4] != _CDH_SIG:
                raise RuntimeError(f"bad central-directory signature at offset {off}")
            (
                _sig,
                _vm,
                _vn,
                _flags,
                method,
                _mt,
                _md,
                crc,
                csize,
                usize,
                nlen,
                xlen,
                clen,
                _ds,
                _ia,
                _ea,
                lho,
            ) = struct.unpack("<IHHHHHHIIIHHHHHII", cd[off : off + 46])
            name = cd[off + 46 : off + 46 + nlen].decode("utf-8")
            extra = cd[off + 46 + nlen : off + 46 + nlen + xlen]
            usize, csize, lho = self._apply_zip64_extra(extra, usize, csize, lho)
            if not name.endswith("/"):
                members[name] = ZipMember(
                    name=name,
                    header_offset=int(lho),
                    compressed_size=int(csize),
                    uncompressed_size=int(usize),
                    method=int(method),
                    crc32=int(crc),
                )
            off += 46 + nlen + xlen + clen
        if n_entries and len(members) == 0:
            raise RuntimeError("central directory parsed to zero file members")
        return members

    @staticmethod
    def _apply_zip64_extra(extra: bytes, usize: int, csize: int, lho: int) -> tuple[int, int, int]:
        """Resolve 0xFFFFFFFF size/offset fields from the ZIP64 extra field."""
        off = 0
        while off + 4 <= len(extra):
            hid, hsz = struct.unpack("<HH", extra[off : off + 4])
            data = extra[off + 4 : off + 4 + hsz]
            if hid == 0x0001:
                pos = 0
                if usize == 0xFFFFFFFF:
                    usize = struct.unpack("<Q", data[pos : pos + 8])[0]
                    pos += 8
                if csize == 0xFFFFFFFF:
                    csize = struct.unpack("<Q", data[pos : pos + 8])[0]
                    pos += 8
                if lho == 0xFFFFFFFF:
                    lho = struct.unpack("<Q", data[pos : pos + 8])[0]
                    pos += 8
            off += 4 + hsz
        return usize, csize, lho

    def _data_range(self, member: ZipMember, comp_bytes: int) -> tuple[bytes, int]:
        """Fetch the local header plus ``comp_bytes`` of compressed payload.

        Returns:
            Tuple of (fetched blob, offset of the payload inside the blob).

        Raises:
            RuntimeError: Bad local-header signature, or the local header's
                name/extra exceed the fetched headroom (would corrupt reads).
        """
        # Local-header name/extra lengths can differ from the central entry;
        # 512 bytes of headroom covers the ~40-char names + ZIP64 extras here.
        headroom = 30 + len(member.name.encode("utf-8")) + 512
        start = member.header_offset
        end = start + headroom + comp_bytes - 1
        blob = self._fetch(f"{start}-{end}")
        if len(blob) < 30:
            raise RuntimeError(f"{member.name}: local header fetch too short")
        sig, _v, _f, _m, _t, _d, _crc, _cs, _us, nlen, xlen = struct.unpack(
            "<IHHHHHIIIHH", blob[:30]
        )
        if sig != _LFH_SIG:
            raise RuntimeError(f"{member.name}: bad local-file-header signature {sig:#x}")
        data_start = 30 + nlen + xlen
        if data_start > headroom:
            raise RuntimeError(f"{member.name}: local header larger than fetched headroom")
        return blob, data_start

    def read_member(self, name: str, *, verify_crc: bool = True) -> bytes:
        """Read one member fully, verifying size (and optionally CRC-32).

        Raises:
            KeyError: Unknown member name.
            RuntimeError: Unsupported compression method, size mismatch, or
                CRC mismatch -- corrupted data must never train silently.
        """
        member = self.members[name]
        blob, data_start = self._data_range(member, member.compressed_size)
        comp = blob[data_start : data_start + member.compressed_size]
        if len(comp) != member.compressed_size:
            raise RuntimeError(f"{name}: short compressed payload fetch")
        if member.method == 0:
            raw = comp
        elif member.method == 8:
            raw = zlib.decompressobj(-15).decompress(comp)
        else:
            raise RuntimeError(f"{name}: unsupported compression method {member.method}")
        if len(raw) != member.uncompressed_size:
            raise RuntimeError(
                f"{name}: decompressed {len(raw)} bytes, expected {member.uncompressed_size}"
            )
        if verify_crc and zlib.crc32(raw) != member.crc32:
            raise RuntimeError(f"{name}: CRC-32 mismatch after decompression")
        return raw

    def read_member_prefix(self, name: str, n_bytes: int) -> bytes:
        """Read the first ``n_bytes`` of a member's uncompressed payload.

        For stored members this is an exact ranged read; for deflate members
        a compressed prefix is fetched and inflated incrementally (raw
        deflate streams cannot be entered mid-stream), growing the fetch if
        the first estimate under-shoots.
        """
        member = self.members[name]
        n_bytes = min(n_bytes, member.uncompressed_size)
        if member.method == 0:
            blob, data_start = self._data_range(member, n_bytes)
            out = blob[data_start : data_start + n_bytes]
            if len(out) != n_bytes:
                raise RuntimeError(f"{name}: short stored-prefix fetch")
            return out
        if member.method != 8:
            raise RuntimeError(f"{name}: unsupported compression method {member.method}")
        # int16 ELF noise compresses to ~0.85x here; n_bytes + 64 KiB of
        # compressed input covers the needed output in one GET essentially
        # always, with an iterative top-up as the loud fallback.
        comp_need = min(member.compressed_size, n_bytes + 65536)
        blob, data_start = self._data_range(member, comp_need)
        comp = blob[data_start : data_start + comp_need]
        decomp = zlib.decompressobj(-15)
        out = decomp.decompress(comp, n_bytes)
        fetched = len(comp)
        while len(out) < n_bytes and fetched < member.compressed_size:
            step = min(member.compressed_size - fetched, 262144)
            abs_start = member.header_offset + data_start + fetched
            more = self._fetch(f"{abs_start}-{abs_start + step - 1}")
            fetched += len(more)
            out += decomp.decompress(decomp.unconsumed_tail + more, n_bytes - len(out))
        if len(out) < n_bytes:
            raise RuntimeError(f"{name}: could not inflate {n_bytes} prefix bytes")
        return out[:n_bytes]

    def read_member_prefix_from_blob(
        self, name: str, blob: bytes, blob_start: int, n_bytes: int
    ) -> bytes:
        """Extract a member's uncompressed prefix out of an already-fetched blob.

        Used by the coalesced group fetch: ``blob`` is one ranged GET that
        starts at absolute archive offset ``blob_start`` and spans several
        consecutive members (time-to-first-byte dominates ranged GETs here,
        so one request per storm run beats one per hour file).

        Raises:
            RuntimeError: The member's local header or enough compressed
                payload is not inside the blob.
        """
        member = self.members[name]
        off = member.header_offset - blob_start
        if off < 0 or off + 30 > len(blob):
            raise RuntimeError(f"{name}: local header outside the group blob")
        sig, _v, _f, _m, _t, _d, _crc, _cs, _us, nlen, xlen = struct.unpack(
            "<IHHHHHIIIHH", blob[off : off + 30]
        )
        if sig != _LFH_SIG:
            raise RuntimeError(f"{name}: bad local-file-header signature {sig:#x}")
        data_off = off + 30 + nlen + xlen
        n_bytes = min(n_bytes, member.uncompressed_size)
        comp = blob[data_off : data_off + member.compressed_size]
        if member.method == 0:
            if len(comp) < n_bytes:
                raise RuntimeError(f"{name}: stored payload not fully inside the group blob")
            return comp[:n_bytes]
        if member.method != 8:
            raise RuntimeError(f"{name}: unsupported compression method {member.method}")
        out = zlib.decompressobj(-15).decompress(comp, n_bytes)
        if len(out) < n_bytes:
            raise RuntimeError(f"{name}: group blob held too little compressed payload")
        return out

    def read_group_prefixes(self, names: list[str], n_bytes: int) -> dict[str, bytes]:
        """Read several consecutive members' prefixes with one ranged GET.

        Fetches the span from the first member's local header through the
        last member's needed compressed prefix, then extracts each member
        from the blob; any member the blob turns out not to cover falls back
        to an individual :meth:`read_member_prefix` (loud in bytes, never in
        correctness).
        """
        ordered = sorted(names, key=lambda n: self.members[n].header_offset)
        first = self.members[ordered[0]]
        last = self.members[ordered[-1]]
        headroom = 30 + len(last.name.encode("utf-8")) + 512
        comp_need = min(last.compressed_size, n_bytes + 65536)
        end = last.header_offset + headroom + comp_need
        blob = self._fetch(f"{first.header_offset}-{end - 1}")
        out: dict[str, bytes] = {}
        for name in ordered:
            try:
                out[name] = self.read_member_prefix_from_blob(
                    name, blob, first.header_offset, n_bytes
                )
            except RuntimeError:
                out[name] = self.read_member_prefix(name, n_bytes)
        return out


class _RangeFetcher:
    """Thread-safe HTTP range fetcher with a hard total-byte budget."""

    def __init__(self, budget_bytes: int = MAX_FETCH_BYTES) -> None:
        """Initialize with the byte budget shared across all archives."""
        self._lock = threading.Lock()
        self.bytes_fetched = 0
        self.requests_made = 0
        self._budget = budget_bytes

    def for_url(self, url: str) -> Callable[[str], bytes]:
        """Bind the fetcher to one archive URL, returning a range callable."""

        def _fetch(spec: str) -> bytes:
            with self._lock:
                if self.bytes_fetched > self._budget:
                    raise RuntimeError(
                        f"fetch budget exceeded: {self.bytes_fetched} bytes > "
                        f"{self._budget}; refusing to keep downloading"
                    )
            body = http_get_with_retry(
                url, headers={"Range": f"bytes={spec}"}, timeout=600.0, retries=4
            )
            if not body:
                raise RuntimeError(f"empty ranged response for bytes={spec} from {url}")
            with self._lock:
                self.bytes_fetched += len(body)
                self.requests_made += 1
            return body

        return _fetch


# ---------------------------------------------------------------------------
# Detector-parity feature computation
# ---------------------------------------------------------------------------


def condition_signal(window: np.ndarray) -> np.ndarray:
    """Demean a raw int16 ADC window into the float signal both paths consume.

    The station records are raw ADC counts with a DC offset; the detector's
    spectrum is max-normalized, so an un-removed DC bin would dominate the
    normalization and crush every Schumann line toward zero. Subtracting the
    window mean (and nothing else -- the response is ~flat over 6-25 Hz and
    the max-normalization removes absolute scale) is the ONLY conditioning
    applied, and it is applied identically to the physics and learned paths:
    the demeaned float64 window is exactly the ``elf_signal`` handed to
    ``detect_resonance_anomaly`` at evaluation time.
    """
    x = np.asarray(window, dtype=np.float64)
    return x - x.mean()


def compute_detector_spectrum(signal: np.ndarray) -> np.ndarray:
    """Re-derive the detector's neural input byte-for-byte (train/serve parity).

    Mirrors ``SchumannResonanceDetector._compute_power_spectrum`` (full-signal
    ``scipy.fft.fft``, one-sided ``|fft|**2`` over the first ``n//2`` bins,
    division by the max over those bins) followed by the
    ``power_spectrum[:512]`` slice and float32 cast that
    ``detect_resonance_anomaly`` applies before the network. Unit-tested for
    exact equality against the detector's own computation.

    Args:
        signal: Conditioned (demeaned) time-domain window.

    Returns:
        float32 array of the first 512 normalized one-sided power bins.
    """
    n = len(signal)
    yf = fft(signal)
    power = np.abs(yf[: n // 2]) ** 2
    power = power / np.max(power)
    return power[:SPECTRUM_BINS].astype(np.float32)


def raw_band_features(signal: np.ndarray) -> tuple[float, float]:
    """Physical-scale spectral features for the deterministic class rule.

    Computed from the raw (NOT max-normalized) periodogram ``|fft|^2 / n`` of
    the demeaned window so band power keeps absolute ADC-count units -- the
    max-normalized detector spectrum is scale-free and cannot measure an
    amplitude anomaly.

    Returns:
        Tuple of (log10 mean band power over 5-40 Hz, power-weighted spectral
        centroid in Hz over the 6-10 Hz fundamental band).
    """
    n = len(signal)
    p = (np.abs(fft(signal)[: n // 2]) ** 2) / n
    freqs = fftfreq(n, 1.0 / FS_HZ)[: n // 2]
    band = (freqs >= 5.0) & (freqs <= 40.0)
    log_bp = float(np.log10(np.mean(p[band]) + 1e-12))
    fund = (freqs >= 6.0) & (freqs <= 10.0)
    denom = float(np.sum(p[fund]))
    centroid = float(np.sum(freqs[fund] * p[fund]) / denom) if denom > 0 else 7.83
    return log_bp, centroid


# ---------------------------------------------------------------------------
# Kp labels and the hour inventory
# ---------------------------------------------------------------------------


def _fetch_kp(ctx: PipelineContext) -> dict[int, float]:
    """Fetch definitive GFZ Kp for every era year; return 3-h-bin -> Kp.

    Bins are keyed by ``unix_seconds // 10800``. Non-definitive bins are
    refused (fail loud) -- provisional labels must not train a checkpoint.
    """
    kp_by_bin: dict[int, float] = {}
    for year in range(ERA_START.year, ERA_END.year + 1):
        url = KP_URL_TEMPLATE.format(year=year)
        path = cached_fetch(url, ctx.data_dir / "schumann" / f"kp_{year}.json")
        payload = json.loads(path.read_text())
        for t_iso, kp, status in zip(
            payload["datetime"], payload["Kp"], payload["status"], strict=True
        ):
            ts = _dt.datetime.fromisoformat(t_iso.replace("Z", "+00:00"))
            if not (ERA_START.year <= ts.year <= ERA_END.year):
                continue
            if status != "def":
                raise RuntimeError(
                    f"GFZ Kp bin {t_iso} has status {status!r}, not definitive; "
                    "refusing to train on provisional labels"
                )
            kp_by_bin[int(ts.timestamp()) // 10800] = float(kp)
    if not kp_by_bin:
        raise RuntimeError("GFZ Kp fetch produced zero definitive bins")
    return kp_by_bin


@dataclass(frozen=True)
class HourRecord:
    """One selected station hour file and its external Kp label."""

    archive: str
    member: str
    t0_iso: str
    year: int
    kp: float
    disturbed: bool


def _parse_member_time(name: str) -> _dt.datetime | None:
    """Parse a sensor-0 member name into its UTC start time (or None).

    Filenames encode YYMMDDHHMM of the first sample, UTC (verified against
    the ``*_info.txt`` sidecars during fetch); seconds are truncated so the
    true start is up to 60 s later.
    """
    m = _MEMBER_RE.match(name)
    if m is None or m.group("sensor") != "0":
        return None
    ts = m.group("ts")
    try:
        return _dt.datetime(
            2000 + int(ts[0:2]),
            int(ts[2:4]),
            int(ts[4:6]),
            int(ts[6:8]),
            int(ts[8:10]),
            tzinfo=_dt.UTC,
        )
    except ValueError:
        return None


def _label_hour(t0: _dt.datetime, kp_by_bin: dict[int, float]) -> tuple[str, float]:
    """Classify one hour file against Kp, spanning every bin its windows touch.

    Files start at arbitrary wall-clock minutes, so the 12 windows (first
    ~716 s, plus 62 s slack for the truncated filename seconds) can cross a
    3-hour Kp bin boundary; the label must hold in EVERY bin touched.

    Returns:
        Tuple of (label, max Kp over touched bins) where label is one of
        ``disturbed`` / ``quiet`` / ``intermediate`` / ``mixed`` /
        ``missing_kp``.
    """
    start = int(t0.timestamp())
    end = start + (WINDOWS_PER_HOUR - 1) * (WINDOW_STRIDE // 256) + WINDOW_SAMPLES // 256 + 62
    bins = range(start // 10800, end // 10800 + 1)
    kps = [kp_by_bin.get(b) for b in bins]
    if any(k is None for k in kps):
        return "missing_kp", float("nan")
    vals = [float(k) for k in kps if k is not None]
    if all(k >= KP_DISTURBED for k in vals):
        return "disturbed", max(vals)
    if all(k <= KP_QUIET for k in vals):
        return "quiet", max(vals)
    if all(k > KP_QUIET for k in vals) or all(k < KP_DISTURBED for k in vals):
        return "intermediate", max(vals)
    return "mixed", max(vals)


def _stratum(t0: _dt.datetime) -> tuple[int, int, int]:
    """Sampling stratum: (year, month, 3-h local-time bucket, UTC+1 fixed)."""
    local_hour = (t0.hour + LOCAL_UTC_OFFSET_HOURS) % 24
    return t0.year, t0.month, local_hour // 3


def _load_member_index(
    ctx: PipelineContext, fetcher: _RangeFetcher
) -> dict[str, dict[str, ZipMember]]:
    """Load (and disk-cache) the member table of every year archive."""
    index: dict[str, dict[str, ZipMember]] = {}
    for key, url in ZIP_ARCHIVES.items():
        cache = ctx.data_dir / "schumann" / f"members_{key}.json"
        if cache.exists():
            raw = json.loads(cache.read_text())
            index[key] = {name: ZipMember(name=name, **fields) for name, fields in raw.items()}
            continue
        reader = RemoteZipReader(fetcher.for_url(url))
        members = reader.members
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps(
                {
                    name: {
                        "header_offset": m.header_offset,
                        "compressed_size": m.compressed_size,
                        "uncompressed_size": m.uncompressed_size,
                        "method": m.method,
                        "crc32": m.crc32,
                    }
                    for name, m in members.items()
                },
                sort_keys=True,
            )
        )
        index[key] = members
        logger.info("archive %s: %d members indexed", key, len(members))
    return index


def build_sampling_plan(
    ctx: PipelineContext,
    member_index: dict[str, dict[str, ZipMember]],
    kp_by_bin: dict[int, float],
) -> dict[str, Any]:
    """Select hours: ALL disturbed + ~1.5x stratified quiet, seeded.

    Strata are (year, month, 3-h local-time bucket) -- month + local time as
    specified, plus year so the by-year temporal split keeps its class ratio.
    Exclusions (intermediate Kp, mixed-bin, missing Kp, wrong file size) are
    counted and reported in the manifest.
    """
    rng = np.random.default_rng(ctx.seed)
    disturbed: list[HourRecord] = []
    quiet_by_stratum: dict[tuple[int, int, int], list[HourRecord]] = {}
    exclusions = {
        "intermediate_kp": 0,
        "mixed_bin": 0,
        "missing_kp": 0,
        "bad_size": 0,
        "out_of_era": 0,
    }
    n_quiet_total = 0
    for archive, members in sorted(member_index.items()):
        for name in sorted(members):
            t0 = _parse_member_time(name)
            if t0 is None:
                continue
            if not (ERA_START <= t0 < ERA_END):
                exclusions["out_of_era"] += 1
                continue
            if members[name].uncompressed_size != HOUR_FILE_BYTES:
                exclusions["bad_size"] += 1
                continue
            label, kp = _label_hour(t0, kp_by_bin)
            rec = HourRecord(
                archive=archive,
                member=name,
                t0_iso=t0.isoformat(),
                year=t0.year,
                kp=kp,
                disturbed=label == "disturbed",
            )
            if label == "disturbed":
                disturbed.append(rec)
            elif label == "quiet":
                n_quiet_total += 1
                quiet_by_stratum.setdefault(_stratum(t0), []).append(rec)
            elif label == "intermediate":
                exclusions["intermediate_kp"] += 1
            elif label == "mixed":
                exclusions["mixed_bin"] += 1
            else:
                exclusions["missing_kp"] += 1
    if not disturbed:
        raise RuntimeError("no disturbed (Kp>=5) hours found in the corpus era")

    # Stratified quiet matching: round(1.5 * disturbed count) per stratum.
    disturbed_by_stratum: dict[tuple[int, int, int], int] = {}
    for rec in disturbed:
        s = _stratum(_dt.datetime.fromisoformat(rec.t0_iso))
        disturbed_by_stratum[s] = disturbed_by_stratum.get(s, 0) + 1
    quiet: list[HourRecord] = []
    shortfall = 0
    for s, n_dist in sorted(disturbed_by_stratum.items()):
        pool = quiet_by_stratum.get(s, [])
        want = round(QUIET_RATIO * n_dist)
        take = min(want, len(pool))
        shortfall += want - take
        if take:
            picks = rng.choice(len(pool), size=take, replace=False)
            quiet.extend(pool[int(i)] for i in sorted(picks))

    plan = {
        "seed": ctx.seed,
        "n_disturbed": len(disturbed),
        "n_quiet": len(quiet),
        "n_quiet_available": n_quiet_total,
        "quiet_shortfall": shortfall,
        "exclusions": exclusions,
        "hours": [rec.__dict__ for rec in [*disturbed, *quiet]],
    }
    logger.info(
        "sampling plan: %d disturbed + %d quiet hours (quiet shortfall %d); " "excluded: %s",
        len(disturbed),
        len(quiet),
        shortfall,
        exclusions,
    )
    return plan


# ---------------------------------------------------------------------------
# Stage 1: fetch
# ---------------------------------------------------------------------------


def _hour_cache_path(ctx: PipelineContext, member: str) -> Path:
    """On-disk cache path for one hour's extracted windows."""
    return ctx.data_dir / "schumann" / "hours" / (member.replace("/", "__") + ".npz")


def _coalesce_fetch_groups(
    recs: list[dict[str, Any]], members: dict[str, ZipMember]
) -> list[list[dict[str, Any]]]:
    """Group one archive's pending hours into consecutive-member fetch runs.

    Hours whose members sit within :data:`GROUP_GAP_BYTES` of the running
    span are coalesced into one ranged GET (storm hours are consecutive
    archive members); spans are capped at :data:`GROUP_SPAN_BYTES`.
    """
    ordered = sorted(recs, key=lambda r: members[r["member"]].header_offset)
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    span_start = span_end = 0
    for rec in ordered:
        m = members[rec["member"]]
        need_end = (
            m.header_offset
            + 30
            + len(m.name.encode("utf-8"))
            + 512
            + min(m.compressed_size, PREFIX_BYTES + 65536)
        )
        if current and (
            m.header_offset - span_end > GROUP_GAP_BYTES or need_end - span_start > GROUP_SPAN_BYTES
        ):
            groups.append(current)
            current = []
        if not current:
            span_start = m.header_offset
        current.append(rec)
        span_end = need_end
    if current:
        groups.append(current)
    return groups


def _verify_one_member(reader: RemoteZipReader, archive: str) -> dict[str, Any]:
    """End-to-end verify one full data member (size + CRC) before bulk use."""
    name = next(
        n
        for n in sorted(reader.members)
        if _parse_member_time(n) is not None
        and reader.members[n].uncompressed_size == HOUR_FILE_BYTES
    )
    raw = reader.read_member(name, verify_crc=True)  # raises on size/CRC mismatch
    return {"archive": archive, "member": name, "bytes": len(raw), "crc_ok": True}


def _parse_info_text(text: str) -> dict[str, Any]:
    """Parse a ``*_info.txt`` sidecar (sampling period, first-sample UTC)."""
    period = re.search(r"sampling period \(usec\):\s*([0-9.]+)", text)
    stamp = re.search(
        r"1st sample timestamp:\s*(\d{2})-(\d{2})-(\d{4}) (\d{2}):(\d{2}):(\d{2})\.\d+ UTC",
        text,
    )
    nsamp = re.search(r"number of samples:\s*(\d+)", text)
    if not (period and stamp and nsamp):
        raise RuntimeError(f"unparseable info sidecar: {text[:120]!r}")
    day, month, year, hh, mm, ss = (int(g) for g in stamp.groups())
    return {
        "sampling_period_usec": float(period.group(1)),
        "first_sample_utc": _dt.datetime(year, month, day, hh, mm, ss, tzinfo=_dt.UTC),
        "n_samples": int(nsamp.group(1)),
    }


def _check_info_sidecar(reader: RemoteZipReader, member: str) -> None:
    """Cross-check one hour's info sidecar: fs, sample count, UTC filename time.

    Raises:
        RuntimeError: Sampling period is not the nominal 1/256 s, the sample
            count is wrong, or the first-sample UTC disagrees with the
            filename time by more than the truncated-seconds minute.
    """
    info_name = member + "_info.txt"
    if info_name not in reader.members:
        raise RuntimeError(f"missing info sidecar {info_name}")
    info = _parse_info_text(reader.read_member(info_name).decode("utf-8", "replace"))
    if abs(info["sampling_period_usec"] - 1e6 / FS_HZ) > 1.0:
        raise RuntimeError(
            f"{member}: sampling period {info['sampling_period_usec']} usec != nominal "
            f"{1e6 / FS_HZ:.2f}; the fs=256 Hz assumption is wrong -- refusing"
        )
    if info["n_samples"] != HOUR_FILE_BYTES // 2:
        raise RuntimeError(f"{member}: info reports {info['n_samples']} samples")
    t_name = _parse_member_time(member)
    if t_name is None:
        raise RuntimeError(f"{member}: unparseable member time")
    delta = (info["first_sample_utc"] - t_name).total_seconds()
    if not (0.0 <= delta < 61.0):
        raise RuntimeError(
            f"{member}: first-sample UTC {info['first_sample_utc']} is {delta:.1f}s from "
            "the filename time; filenames are not UTC-aligned -- refusing"
        )


def fetch(ctx: PipelineContext) -> dict[str, Any]:
    """Fetch labels, archive indexes, and the sampled hours' window prefixes.

    All transfers go through :func:`http_get_with_retry` (allowlisted hosts,
    HTTPS-only) under a hard 6 GiB budget (:data:`MAX_FETCH_BYTES`). Every
    selected hour's first :data:`PREFIX_SAMPLES` samples are cached as int16
    ``.npz`` files; re-running resumes from the cache.

    Returns:
        Manifest with data-source provenance, class counts, exclusion counts,
        verification results, and total bytes fetched.
    """
    data_dir = ctx.data_dir / "schumann"
    data_dir.mkdir(parents=True, exist_ok=True)
    fetcher = _RangeFetcher()
    kp_by_bin = _fetch_kp(ctx)
    member_index = _load_member_index(ctx, fetcher)

    readers = {key: RemoteZipReader(fetcher.for_url(url)) for key, url in ZIP_ARCHIVES.items()}
    for key, reader in readers.items():
        reader._members = member_index[key]  # reuse the cached directory

    plan_path = data_dir / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        plan = build_sampling_plan(ctx, member_index, kp_by_bin)
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True))

    verifications = []
    verify_path = data_dir / "verify.json"
    if verify_path.exists():
        verifications = json.loads(verify_path.read_text())
    else:
        for key, reader in sorted(readers.items()):
            verifications.append(_verify_one_member(reader, key))
            _check_info_sidecar(reader, verifications[-1]["member"])
        verify_path.write_text(json.dumps(verifications, indent=2))
        logger.info("end-to-end member verification passed for all archives")

    hours: list[dict[str, Any]] = plan["hours"]
    if ctx.limit_samples is not None:
        hours = hours[: ctx.limit_samples]

    def _save_hour(rec: dict[str, Any], raw: bytes) -> None:
        cache = _hour_cache_path(ctx, rec["member"])
        samples = np.frombuffer(raw, dtype="<i2")
        windows = np.stack(
            [
                samples[k * WINDOW_STRIDE : k * WINDOW_STRIDE + WINDOW_SAMPLES]
                for k in range(WINDOWS_PER_HOUR)
            ]
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".part")
        # Write through a file object: np.savez_compressed silently appends
        # ".npz" to bare paths, which would orphan the temp file.
        with tmp.open("wb") as fh:
            np.savez_compressed(
                fh,
                windows=windows.astype(np.int16),
                t0_iso=np.array(rec["t0_iso"]),
                kp=np.array(rec["kp"], dtype=np.float64),
                disturbed=np.array(rec["disturbed"]),
                member=np.array(rec["member"]),
            )
        tmp.replace(cache)

    pending = [rec for rec in hours if not _hour_cache_path(ctx, rec["member"]).exists()]
    groups: list[list[dict[str, Any]]] = []
    for key in sorted(ZIP_ARCHIVES):
        groups.extend(
            _coalesce_fetch_groups(
                [rec for rec in pending if rec["archive"] == key], member_index[key]
            )
        )

    def _fetch_group(group: list[dict[str, Any]]) -> int:
        reader = readers[group[0]["archive"]]
        prefixes = reader.read_group_prefixes([rec["member"] for rec in group], PREFIX_BYTES)
        for rec in group:
            _save_hour(rec, prefixes[rec["member"]])
        return len(group)

    n_new = 0
    with ThreadPoolExecutor(max_workers=12) as pool:
        for gi, n_in_group in enumerate(pool.map(_fetch_group, groups)):
            n_new += n_in_group
            if (gi + 1) % 25 == 0:
                logger.info(
                    "hour fetch progress: %d/%d hours in %d/%d groups (%.1f MiB transferred)",
                    n_new,
                    len(pending),
                    gi + 1,
                    len(groups),
                    fetcher.bytes_fetched / (1 << 20),
                )

    # Spot-check info sidecars for a deterministic subset of fetched hours.
    info_checks = 0
    info_check_path = data_dir / "info_checks.json"
    if not info_check_path.exists():
        for rec in hours[::40]:
            _check_info_sidecar(readers[rec["archive"]], rec["member"])
            info_checks += 1
        info_check_path.write_text(json.dumps({"hours_checked": info_checks}))

    sources: list[dict[str, Any]] = [
        {
            "url": url,
            "sha256": sha256_file(ctx.data_dir / "schumann" / f"members_{key}.json"),
            "description": (
                f"Sierra Nevada ELF station year archive {key} (Zenodo, CC-BY-4.0; "
                "Salinas et al. 2022, Comput. Geosci. 165:105148); ranged-GET member "
                "access, sha256 is of the parsed member index"
            ),
        }
        for key, url in sorted(ZIP_ARCHIVES.items())
    ]
    for year in range(ERA_START.year, ERA_END.year + 1):
        path = ctx.data_dir / "schumann" / f"kp_{year}.json"
        sources.append(
            {
                "url": KP_URL_TEMPLATE.format(year=year),
                "sha256": sha256_file(path),
                "description": f"GFZ Potsdam definitive Kp index, {year} (labels)",
            }
        )

    manifest = {
        "hook": HOOK_NAME,
        "sources": sources,
        "plan": {k: v for k, v in plan.items() if k != "hours"},
        "n_hours_selected": len(hours),
        "n_hours_fetched_this_run": n_new,
        "verification": verifications,
        "bytes_fetched_this_run": fetcher.bytes_fetched,
        "requests_this_run": fetcher.requests_made,
        "window": {
            "fs_hz": FS_HZ,
            "window_samples": WINDOW_SAMPLES,
            "windows_per_hour": WINDOWS_PER_HOUR,
            "window_stride_samples": WINDOW_STRIDE,
            "prefix_bytes": PREFIX_BYTES,
        },
    }
    manifest_path = data_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    logger.info(
        "fetch complete: %d hours cached (%d new), %.1f MiB / %d requests this run",
        len(hours),
        n_new,
        fetcher.bytes_fetched / (1 << 20),
        fetcher.requests_made,
    )
    return manifest


# ---------------------------------------------------------------------------
# Stage 2: build
# ---------------------------------------------------------------------------


@dataclass
class SchumannDataset:
    """Hour-level dataset: detector-parity spectra plus derived labels.

    Attributes:
        spectra: float32 ``[n_hours, WINDOWS_PER_HOUR, 512]`` detector-parity
            window spectra.
        windows_int16: int16 ``[n_hours, WINDOWS_PER_HOUR, WINDOW_SAMPLES]``
            raw ADC windows; :func:`condition_signal` of a window yields the
            exact public-API input (kept int16 to bound memory).
        class_label: int64 ``[n_hours]`` into :data:`ANOMALY_CLASSES`.
        disturbed: float32 ``[n_hours]`` (Kp>=5 -> 1, Kp<=2 -> 0).
        years: int64 ``[n_hours]`` (from the file UTC start time).
        kp: float64 ``[n_hours]`` max Kp over the touched bins.
        t0_iso: hour start times (ISO-8601 UTC).
        climatology: train-quiet climatology used by the class rule.
        class_counts: per-split class counts (reported into extras).
    """

    spectra: np.ndarray
    windows_int16: np.ndarray
    class_label: np.ndarray
    disturbed: np.ndarray
    years: np.ndarray
    kp: np.ndarray
    t0_iso: list[str]
    climatology: dict[str, float]
    class_counts: dict[str, dict[str, int]]


def derive_class_labels(
    log_bp: np.ndarray,
    centroid: np.ndarray,
    disturbed: np.ndarray,
    train_quiet_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply the deterministic 4-class rule (see :data:`CLASS_RULE`).

    Args:
        log_bp: Hour-mean log10 band power (5-40 Hz), physical scale.
        centroid: Hour-mean 6-10 Hz spectral centroid (Hz).
        disturbed: Boolean-ish array, 1 where Kp>=5.
        train_quiet_mask: Hours that are BOTH quiet and in the train years --
            the only rows the climatology statistics may touch.

    Returns:
        Tuple of (class labels int64, climatology dict).

    Raises:
        RuntimeError: No train-quiet hours, or a degenerate (zero-variance)
            climatology -- the rule would be meaningless.
    """
    if int(train_quiet_mask.sum()) < 10:
        raise RuntimeError(
            f"only {int(train_quiet_mask.sum())} train-year quiet hours; "
            "cannot build a quiet climatology"
        )
    mu_bp = float(np.mean(log_bp[train_quiet_mask]))
    sd_bp = float(np.std(log_bp[train_quiet_mask]))
    mu_c = float(np.mean(centroid[train_quiet_mask]))
    sd_c = float(np.std(centroid[train_quiet_mask]))
    if sd_bp < 1e-9 or sd_c < 1e-9:
        raise RuntimeError("degenerate quiet climatology (zero variance)")
    labels = np.zeros(len(log_bp), dtype=np.int64)
    z_amp = np.abs(log_bp - mu_bp) / sd_bp
    z_freq = np.abs(centroid - mu_c) / sd_c
    for i in np.flatnonzero(disturbed > 0.5):
        if z_amp[i] > 2.0 and z_freq[i] > 2.0:
            labels[i] = 3
        elif z_freq[i] > 2.0:
            labels[i] = 2
        elif z_amp[i] > 2.0:
            labels[i] = 1
        else:
            labels[i] = 1 if z_amp[i] >= z_freq[i] else 2
    climatology = {
        "log_band_power_mean": mu_bp,
        "log_band_power_std": sd_bp,
        "centroid_hz_mean": mu_c,
        "centroid_hz_std": sd_c,
        "n_train_quiet_hours": int(train_quiet_mask.sum()),
    }
    return labels, climatology


def build_dataset(ctx: PipelineContext) -> SchumannDataset:
    """Assemble the hour-level dataset from the fetched window caches.

    Spectra are the detector-parity features; the class rule's climatology is
    computed from TRAIN-year quiet hours only (no leakage), then applied as a
    fixed rule to every split.
    """
    data_dir = ctx.data_dir / "schumann"
    plan_path = data_dir / "plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"missing sampling plan {plan_path}; run the fetch stage first")
    hours: list[dict[str, Any]] = json.loads(plan_path.read_text())["hours"]
    if ctx.limit_samples is not None:
        hours = hours[: ctx.limit_samples]

    spectra, raw_windows, disturbed, years, kp, t0_iso = [], [], [], [], [], []
    log_bp, centroid = [], []
    missing = 0
    for rec in hours:
        cache = _hour_cache_path(ctx, rec["member"])
        if not cache.exists():
            missing += 1
            continue
        with np.load(cache) as npz:
            windows = npz["windows"]
        conditioned = np.stack([condition_signal(w) for w in windows])
        spectra.append(np.stack([compute_detector_spectrum(s) for s in conditioned]))
        raw_windows.append(windows.astype(np.int16))
        feats = [raw_band_features(s) for s in conditioned]
        log_bp.append(float(np.mean([f[0] for f in feats])))
        centroid.append(float(np.mean([f[1] for f in feats])))
        disturbed.append(1.0 if rec["disturbed"] else 0.0)
        years.append(rec["year"])
        kp.append(rec["kp"])
        t0_iso.append(rec["t0_iso"])
    if missing:
        raise RuntimeError(
            f"{missing} selected hours have no cached windows; re-run the fetch stage"
        )
    if not spectra:
        raise RuntimeError("no cached hours found; run the fetch stage first")

    years_arr = np.asarray(years, dtype=np.int64)
    disturbed_arr = np.asarray(disturbed, dtype=np.float32)
    train_mask, _, _ = SPLIT.masks(years_arr)
    labels, climatology = derive_class_labels(
        np.asarray(log_bp),
        np.asarray(centroid),
        disturbed_arr,
        train_mask & (disturbed_arr < 0.5),
    )

    class_counts: dict[str, dict[str, int]] = {}
    masks = dict(zip(("train", "val", "test"), SPLIT.masks(years_arr), strict=True))
    for split_name, mask in masks.items():
        class_counts[split_name] = {
            cls: int(np.sum(labels[mask] == i)) for i, cls in enumerate(ANOMALY_CLASSES)
        }
    logger.info("dataset: %d hours; class counts per split: %s", len(labels), class_counts)

    return SchumannDataset(
        spectra=np.stack(spectra),
        windows_int16=np.stack(raw_windows),
        class_label=labels,
        disturbed=disturbed_arr,
        years=years_arr,
        kp=np.asarray(kp, dtype=np.float64),
        t0_iso=t0_iso,
        climatology=climatology,
        class_counts=class_counts,
    )


# ---------------------------------------------------------------------------
# Stage 3: train
# ---------------------------------------------------------------------------


def _hour_scores(model: Any, spectra: np.ndarray, batch_hours: int = 64) -> np.ndarray:
    """Deployed hour-level disturbance scores: mean sigmoid-confidence over windows."""
    model.eval()
    n_hours = spectra.shape[0]
    out = np.zeros(n_hours, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, n_hours, batch_hours):
            chunk = spectra[start : start + batch_hours]
            flat = torch.from_numpy(chunk.reshape(-1, 1, SPECTRUM_BINS))
            _, conf = model(flat)
            out[start : start + chunk.shape[0]] = (
                conf.squeeze(-1).reshape(chunk.shape[0], WINDOWS_PER_HOUR).mean(dim=1).numpy()
            )
    return out


def _window_confidences(model: Any, spectra: np.ndarray, batch_hours: int = 64) -> np.ndarray:
    """Per-window sigmoid confidences, shape ``[n_hours, WINDOWS_PER_HOUR]``."""
    model.eval()
    n_hours = spectra.shape[0]
    out = np.zeros((n_hours, WINDOWS_PER_HOUR), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, n_hours, batch_hours):
            chunk = spectra[start : start + batch_hours]
            flat = torch.from_numpy(chunk.reshape(-1, 1, SPECTRUM_BINS))
            _, conf = model(flat)
            out[start : start + chunk.shape[0]] = (
                conf.squeeze(-1).reshape(chunk.shape[0], WINDOWS_PER_HOUR).numpy()
            )
    return out


def _select_operating_point(
    model: Any, ds: SchumannDataset, val_mask: np.ndarray
) -> dict[str, Any]:
    """Choose the learned path's decision threshold on the VALIDATION years.

    Policy (mirroring the ratified solar-storm / tsunami operating-point
    machinery): the deployed hour-level rule for BOTH paths flags an hour
    when >= :data:`HOUR_VOTE_FRACTION` of its 12 windows report
    ``anomaly_detected``. Physics' window decision is its deterministic
    spectral flags, measured here through the public
    ``detect_resonance_anomaly`` API on the validation hours. The learned
    window decision is ``confidence > tau``; a candidate ``tau`` is feasible
    when the learned hour-level validation recall reaches at least
    ``max(physics validation recall, OPERATING_POINT_RECALL_FLOOR)`` (the
    floor keeps a technically-non-regressing but operationally useless
    threshold from being selected) AND the validation false-alarm rate stays
    at or below :data:`OPERATING_POINT_FAR_HEADROOM` ``* physics validation
    FAR`` (solar's ratified 20% headroom against val->test shift). Among
    feasible thresholds the one maximizing validation CSI wins (ties ->
    higher tau, fewer alarms). If no threshold meets both targets, the
    FAR-feasible threshold with the best recall is recorded instead
    (``recall_floor_met`` False) -- the ship gate, not this selection, is
    the final arbiter. The chosen ``tau`` ships in the checkpoint payload as
    ``operating_point`` and ``SchumannResonanceDetector.load_neural_weights``
    validates and applies it to the learned path's ``anomaly_detected``
    DECISION only; the weights, confidence estimate, and anomaly_type are
    never touched, and test years are never consulted.

    Args:
        model: Trained SchumannHarmonicAnalyzer (never modified).
        ds: The built hour-level dataset.
        val_mask: Boolean validation-year mask over ``ds`` rows.

    Returns:
        Operating-point record (policy, detection threshold, validation
        recall/FAR/CSI for the learned rule and physics).

    Raises:
        RuntimeError: Validation lacks a class, or no threshold satisfies
            even the FAR ceiling.
    """
    from omni_mercury_engine.space.schumann_resonance import SchumannResonanceDetector

    val_idx = np.flatnonzero(val_mask)
    y_val = ds.disturbed[val_idx] > 0.5
    if not y_val.any() or y_val.all():
        raise RuntimeError(
            "validation years contain a single class; cannot select an operating point honestly"
        )

    physics_det = SchumannResonanceDetector(sampling_rate=FS_HZ)
    logging.getLogger("omni_mercury_engine.space.schumann_resonance").setLevel(logging.WARNING)
    physics_hour = np.zeros(val_idx.size, dtype=bool)
    for row, i in enumerate(val_idx):
        flags = [
            bool(
                physics_det.detect_resonance_anomaly(
                    condition_signal(ds.windows_int16[i, w])
                ).anomaly_detected
            )
            for w in range(WINDOWS_PER_HOUR)
        ]
        physics_hour[row] = float(np.mean(flags)) >= HOUR_VOTE_FRACTION
    physics_recall = float(physics_hour[y_val].mean())
    physics_far = float(physics_hour[~y_val].mean())

    conf = _window_confidences(model, ds.spectra[val_idx])

    recall_floor = max(physics_recall, OPERATING_POINT_RECALL_FLOOR)
    far_ceiling = OPERATING_POINT_FAR_HEADROOM * physics_far

    def _learned_metrics(tau: float) -> tuple[float, float, float]:
        flagged = np.mean(conf > tau, axis=1) >= HOUR_VOTE_FRACTION
        tp = float(np.sum(flagged & y_val))
        fn = float(np.sum(~flagged & y_val))
        fp = float(np.sum(flagged & ~y_val))
        recall = tp / max(tp + fn, 1.0)
        far = fp / max(float(np.sum(~y_val)), 1.0)
        csi = tp / max(tp + fn + fp, 1.0)
        return recall, far, csi

    taus = np.unique(np.quantile(conf.ravel(), np.linspace(0.0, 1.0, 513)))
    # The detector validates 0 < tau < 1 on load; degenerate endpoints are
    # not deployable decision rules.
    taus = taus[(taus > 0.0) & (taus < 1.0)]
    best: dict[str, Any] | None = None
    fallback: dict[str, Any] | None = None
    for tau in taus:
        recall, far, csi = _learned_metrics(float(tau))
        entry = {
            "detection_threshold": float(tau),
            "val_recall": recall,
            "val_far": far,
            "val_csi": csi,
        }
        # Fallback if no threshold satisfies both targets: the most
        # conservative feasible-on-FAR point with the best recall (a
        # recall-maximizing fallback that blows the FAR ceiling would be
        # selecting a point the ship gate is guaranteed to refuse).
        if far <= far_ceiling and (fallback is None or recall > fallback["val_recall"]):
            fallback = entry
        if (
            recall >= recall_floor
            and far <= far_ceiling
            and (
                best is None
                or csi > best["val_csi"]
                or (csi == best["val_csi"] and tau > best["detection_threshold"])
            )
        ):
            best = entry
    floor_met = best is not None
    chosen = best if best is not None else fallback
    if chosen is None:
        raise RuntimeError(
            "no detection threshold satisfies even the FAR ceiling on validation; "
            "the confidence head is not usable for detection decisions -- refusing "
            "to record a doomed operating point"
        )
    record = {
        **chosen,
        "policy": (
            "learned window decision is confidence>tau, hour flagged when >= "
            f"{HOUR_VOTE_FRACTION} of its windows fire; tau maximizes val CSI subject "
            f"to val recall >= max(physics val recall, {OPERATING_POINT_RECALL_FLOOR}) "
            f"AND val FAR <= {OPERATING_POINT_FAR_HEADROOM} * physics val FAR"
        ),
        "recall_floor": recall_floor,
        "recall_floor_met": floor_met,
        "far_ceiling": far_ceiling,
        "val_recall_physics": physics_recall,
        "val_far_physics": physics_far,
    }
    logger.info("operating point selected: %s", json.dumps(record, sort_keys=True))
    return record


def train(ctx: PipelineContext) -> dict[str, Any]:
    """Train the detector's SchumannHarmonicAnalyzer on the window spectra.

    Losses are wired to the architecture's actual heads: cross-entropy on the
    4-class ``anomaly_classifier`` logits and BCE-with-logits on the
    pre-sigmoid ``confidence_logits`` (disturbed=1 / quiet=0). The LSTM
    temporal path is trained with an auxiliary loss over each hour's window
    sequence (mirroring ``_process_temporal_history``: last 10 spectra,
    first 103 bins) so loading the checkpoint does not enable an untrained
    path. Early stopping monitors the deployed hour-level validation AUC
    (mean window sigmoid-confidence), patience 6.

    The checkpoint payload carries a validation-selected ``operating_point``
    (see :func:`_select_operating_point`): the learned path's deployed
    ``anomaly_detected`` decision is ``confidence > tau``, selected on the
    VALIDATION years only against the same recall/FAR targets the ship gate
    enforces, with solar's ratified FAR headroom. The weights are untouched
    (selection is post-hoc on validation outputs) and test years are never
    consulted; ``SchumannResonanceDetector.load_neural_weights`` validates
    and applies it, decision only.

    Returns:
        Training record (epochs, best validation AUC, split sizes,
        operating-point record).
    """
    from omni_mercury_engine.space.schumann_resonance import SchumannHarmonicAnalyzer

    torch.set_num_threads(2)  # shared box
    rng = seed_everything(ctx.seed)
    ds = build_dataset(ctx)
    train_mask, val_mask, _ = SPLIT.masks(ds.years)
    if not train_mask.any() or not val_mask.any():
        raise RuntimeError("train or validation years empty; cannot train")

    n_win = WINDOWS_PER_HOUR
    x_train = torch.from_numpy(ds.spectra[train_mask].reshape(-1, 1, SPECTRUM_BINS))
    y_cls_train = torch.from_numpy(np.repeat(ds.class_label[train_mask], n_win))
    y_dist_train = torch.from_numpy(np.repeat(ds.disturbed[train_mask], n_win))
    # Temporal-path inputs: per hour, the last 10 window spectra's first 103
    # bins (exactly what _process_temporal_history feeds the LSTM).
    seq_train = torch.from_numpy(ds.spectra[train_mask][:, -10:, :103].copy())
    cls_hour_train = torch.from_numpy(ds.class_label[train_mask])
    dist_hour_train = torch.from_numpy(ds.disturbed[train_mask])

    model = SchumannHarmonicAnalyzer(spectrum_size=SPECTRUM_BINS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = torch.nn.CrossEntropyLoss()
    bce = torch.nn.BCEWithLogitsLoss()

    logger.info(
        "training on %d windows (%d hours, %.1f%% disturbed), validating on %d hours",
        x_train.shape[0],
        int(train_mask.sum()),
        100 * float(ds.disturbed[train_mask].mean()),
        int(val_mask.sum()),
    )

    batch_size = 64
    best_val_auc = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    patience, bad_epochs = 6, 0
    epochs_run = 0
    n_hours_train = int(train_mask.sum())

    for epoch in range(ctx.max_epochs):
        epochs_run = epoch + 1
        model.train()
        perm = torch.from_numpy(rng.permutation(x_train.shape[0]))
        epoch_loss = 0.0
        for start in range(0, x_train.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            if idx.shape[0] < 2:
                continue  # BatchNorm needs >1 sample
            features = model._features(x_train[idx])
            logits = model.anomaly_classifier(features)
            conf_logit = model.confidence_head[0](features).squeeze(-1)
            loss = ce(logits, y_cls_train[idx]) + bce(conf_logit, y_dist_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * idx.shape[0]

        # Auxiliary temporal-path pass (LSTM features -> same heads).
        hperm = torch.from_numpy(rng.permutation(n_hours_train))
        for start in range(0, n_hours_train, batch_size):
            idx = hperm[start : start + batch_size]
            if idx.shape[0] < 2:
                continue
            lstm_out, _ = model.lstm(seq_train[idx])
            features = lstm_out[:, -1, :]
            logits = model.anomaly_classifier(features)
            conf_logit = model.confidence_head[0](features).squeeze(-1)
            loss = ce(logits, cls_hour_train[idx]) + bce(conf_logit, dist_hour_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_scores = _hour_scores(model, ds.spectra[val_mask])
        val_auc = binary_auc(ds.disturbed[val_mask], val_scores)
        logger.info(
            "epoch %d: train loss %.4f, val disturbed AUC %.4f",
            epoch + 1,
            epoch_loss / x_train.shape[0],
            val_auc,
        )
        if np.isfinite(val_auc) and val_auc > best_val_auc + 1e-4:
            best_val_auc = float(val_auc)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info("early stop at epoch %d (patience %d)", epoch + 1, patience)
                break

    if best_state is None:
        raise RuntimeError("training produced no finite validation AUC; refusing to save")
    model.load_state_dict(best_state)
    operating_point = _select_operating_point(model, ds, val_mask)

    record = {
        "seed": ctx.seed,
        "epochs_run": epochs_run,
        "best_val_disturbed_auc": best_val_auc,
        "n_train_hours": int(train_mask.sum()),
        "n_val_hours": int(val_mask.sum()),
        "train_years": list(SPLIT.train_years),
        "val_years": list(SPLIT.val_years),
        "class_counts": ds.class_counts,
        "climatology": ds.climatology,
        "operating_point": operating_point,
    }
    payload: dict[str, Any] = {
        "harmonic_analyzer": model.state_dict(),
        "feature_spec": FEATURE_SPEC_VERSION,
        "fs_hz": FS_HZ,
        "window_samples": WINDOW_SAMPLES,
        "spectrum_bins": SPECTRUM_BINS,
        "station": "Sierra Nevada ELF (Zenodo 6348691/6348773/6348838/6348930)",
        "class_rule": CLASS_RULE,
        "operating_point": operating_point,
    }
    save_candidate(ctx.data_dir, HOOK_NAME, payload, record)
    return record


# ---------------------------------------------------------------------------
# Stage 4: evaluate
# ---------------------------------------------------------------------------


def _majority_type(types: list[str]) -> str:
    """Deterministic hour-level type: majority vote, ties to the lower class index."""
    counts = np.zeros(len(ANOMALY_CLASSES), dtype=np.int64)
    for t in types:
        counts[ANOMALY_CLASSES.index(t)] += 1
    return ANOMALY_CLASSES[int(np.argmax(counts))]


def _storm_events(t0_iso: list[str], kp: np.ndarray, mask: np.ndarray) -> list[dict[str, Any]]:
    """Group disturbed hours (within ``mask``) into storm events (>12 h gaps)."""
    idx = [i for i in np.flatnonzero(mask)]
    idx.sort(key=lambda i: t0_iso[i])
    events: list[dict[str, Any]] = []
    current: list[int] = []
    last_time: _dt.datetime | None = None
    for i in idx:
        t = _dt.datetime.fromisoformat(t0_iso[i])
        if last_time is not None and (t - last_time).total_seconds() > 12 * 3600:
            events.append({"hours": current})
            current = []
        current.append(int(i))
        last_time = t
    if current:
        events.append({"hours": current})
    for ev in events:
        peaks = [float(kp[i]) for i in ev["hours"]]
        ev["start_utc"] = t0_iso[ev["hours"][0]]
        ev["peak_kp"] = max(peaks)
        ev["n_hours"] = len(ev["hours"])
    return events


def _noaa_g_scale(peak_kp: float) -> str:
    """NOAA G-scale label from a Kp value (G1=Kp5 .. G5=Kp9).

    Fractional (thirds) Kp is rounded to the nearest integer per the SWPC
    convention, so Kp 8- (7.667) reads G4 -- matching how documented storms
    (e.g. the 2015 G4 pair) are named.
    """
    kp_int = int(np.floor(peak_kp + 0.5))
    if kp_int >= 9:
        return "G5"
    return {5: "G1", 6: "G2", 7: "G3", 8: "G4"}.get(kp_int, "sub-G1")


def _storm_name(start_iso: str) -> str | None:
    """Documented storm name for an event starting near a NAMED_STORMS date."""
    start = _dt.datetime.fromisoformat(start_iso)
    for date_str, name in NAMED_STORMS.items():
        anchor = _dt.datetime.fromisoformat(f"{date_str}T00:00:00+00:00")
        if abs((start - anchor).total_seconds()) <= 2 * 86400:
            return name
    return None


def _storm_hit_table(
    events: list[dict[str, Any]],
    runs: dict[str, dict[str, np.ndarray]],
    row_of: dict[int, int],
) -> list[dict[str, Any]]:
    """Per-storm-event hit table under each path's deployed hour decision."""
    table = []
    for ev in events:
        rows = [row_of[i] for i in ev["hours"]]
        entry: dict[str, Any] = {
            "name": _storm_name(ev["start_utc"]),
            "start_utc": ev["start_utc"],
            # Peak over the hours THIS pipeline sampled -- the storm's true
            # peak can be higher (mixed-bin hours are excluded from sampling).
            "peak_kp_sampled": ev["peak_kp"],
            "noaa_scale_sampled": _noaa_g_scale(ev["peak_kp"]),
            "n_hours_sampled": ev["n_hours"],
        }
        for label in ("physics", "learned"):
            dec = runs[label]["decision"][rows]
            entry[f"{label}_hit"] = bool(dec.any())
            entry[f"{label}_hour_fraction"] = float(dec.mean())
        table.append(entry)
    return table


def evaluate(ctx: PipelineContext) -> EvaluationOutcome:
    """Compare learned vs physics through the public detector API.

    Both paths receive the identical held-out cases: every test-year hour's
    12 conditioned windows, each passed to
    ``SchumannResonanceDetector.detect_resonance_anomaly``. Physics is the
    untrained detector's deterministic FFT assessment; learned is the same
    detector after ``load_neural_weights`` on the candidate checkpoint (which
    installs the checkpoint's validation-selected operating point, so the
    learned window decision is ``confidence > tau``). Hour-level scores
    average the 12 window outputs so window extraction does not inflate the
    test size (no pseudo-replication); each path's hour decision is >= 50%
    of its windows reporting ``anomaly_detected`` under its OWN deployed
    rule. Secondary non-regression constraints: disturbed-detection recall
    at the deployed rule (higher), false-alarm rate at the deployed rule
    (lower), and hour-level anomaly-type agreement (higher).

    Extras additionally carry a seeded 1000-resample bootstrap 95% CI on
    the hour-level AUC difference, the model parameter count, the median
    single-window inference latency (100 runs), the per-storm hit table on
    the held-out test years, and a clearly-labeled validation-storm
    diagnostic table covering the named 2015 G4 storms (early stopping and
    operating-point context only -- never the gate's metrics).

    Returns:
        Evaluation outcome (primary metric: disturbed_auc, higher is better).
    """
    import time

    from omni_mercury_engine.space.schumann_resonance import SchumannResonanceDetector

    torch.set_num_threads(2)  # shared box
    ds = build_dataset(ctx)
    _, val_mask, test_mask = SPLIT.masks(ds.years)
    test_idx = np.flatnonzero(test_mask)
    if test_idx.size == 0:
        raise RuntimeError("no test hours found; cannot evaluate")

    cand_path, _ = candidate_paths(ctx.data_dir, HOOK_NAME)
    if not cand_path.exists():
        raise FileNotFoundError(f"no candidate checkpoint at {cand_path}; run --train first")

    physics_det = SchumannResonanceDetector(sampling_rate=FS_HZ)
    learned_det = SchumannResonanceDetector(sampling_rate=FS_HZ)
    learned_det.load_neural_weights(str(cand_path))
    payload = safe_torch_load(cand_path, map_location="cpu")
    operating_point = payload.get("operating_point")
    # Tens of thousands of public-API calls follow; the detector logs one INFO
    # line per call, which would swamp the evaluation log.
    logging.getLogger("omni_mercury_engine.space.schumann_resonance").setLevel(logging.WARNING)

    def _run_hours(idx: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
        """Run both detectors over every window of the given hours."""
        out: dict[str, dict[str, Any]] = {
            label: {"conf": [], "type": [], "decision": []} for label in ("physics", "learned")
        }
        for i in idx:
            per: dict[str, dict[str, list[Any]]] = {
                label: {"conf": [], "type": [], "flag": []} for label in out
            }
            for w in range(WINDOWS_PER_HOUR):
                signal = condition_signal(ds.windows_int16[i, w])
                for label, det in (("physics", physics_det), ("learned", learned_det)):
                    res = det.detect_resonance_anomaly(signal)
                    per[label]["conf"].append(float(res.confidence))
                    per[label]["type"].append(res.anomaly_type)
                    # Each path's own deployed window decision: physics keeps
                    # the deterministic spectral flags; the learned path (with
                    # the loaded operating point) decides confidence > tau.
                    per[label]["flag"].append(bool(res.anomaly_detected))
            for label in out:
                out[label]["conf"].append(float(np.mean(per[label]["conf"])))
                out[label]["type"].append(_majority_type(per[label]["type"]))
                out[label]["decision"].append(
                    float(np.mean(per[label]["flag"])) >= HOUR_VOTE_FRACTION
                )
        return {
            label: {
                "conf": np.asarray(v["conf"], dtype=np.float64),
                "type": np.asarray(v["type"], dtype=object),
                "decision": np.asarray(v["decision"], dtype=bool),
            }
            for label, v in out.items()
        }

    runs = _run_hours(test_idx)
    disturbed = ds.disturbed[test_idx].astype(bool)
    labels_true = ds.class_label[test_idx]

    def _metrics(label: str) -> dict[str, float]:
        conf = runs[label]["conf"]
        decision = runs[label]["decision"]
        types = runs[label]["type"]
        pred_cls = np.array([ANOMALY_CLASSES.index(t) for t in types])
        n_dist = max(int(disturbed.sum()), 1)
        n_quiet = max(int((~disturbed).sum()), 1)
        return {
            "disturbed_auc": binary_auc(disturbed.astype(float), conf),
            "disturbed_recall_op": float(np.sum(decision & disturbed)) / n_dist,
            "false_alarm_rate_op": float(np.sum(decision & ~disturbed)) / n_quiet,
            "four_class_accuracy": float(np.mean(pred_cls == labels_true)),
            "confidence_brier": brier_score(disturbed.astype(float), conf),
        }

    # Per-storm hit table on the held-out test years.
    events = _storm_events(ds.t0_iso, ds.kp, test_mask & (ds.disturbed > 0.5))
    storm_table = _storm_hit_table(events, runs, {int(g): k for k, g in enumerate(test_idx)})

    # Validation-storm DIAGNOSTIC table (named 2015 G4 storms live in the
    # validation year by construction of the temporal split). These hours
    # informed early stopping and the operating point only -- they are
    # reported for context and never enter the gate's metrics.
    val_dist_idx = np.flatnonzero(val_mask & (ds.disturbed > 0.5))
    val_runs = _run_hours(val_dist_idx)
    val_events = _storm_events(ds.t0_iso, ds.kp, val_mask & (ds.disturbed > 0.5))
    val_storm_table = _storm_hit_table(
        val_events, val_runs, {int(g): k for k, g in enumerate(val_dist_idx)}
    )

    # Seeded bootstrap 95% CI on the hour-level AUC difference (learned -
    # physics) over the test hours: 1000 resamples with replacement;
    # resamples that lose a class are skipped and counted.
    rng = np.random.default_rng(ctx.seed)
    n_resamples = 1000
    diffs: list[float] = []
    for _ in range(n_resamples):
        rs = rng.integers(0, test_idx.size, size=test_idx.size)
        auc_l = binary_auc(disturbed[rs].astype(float), runs["learned"]["conf"][rs])
        auc_p = binary_auc(disturbed[rs].astype(float), runs["physics"]["conf"][rs])
        if np.isfinite(auc_l) and np.isfinite(auc_p):
            diffs.append(float(auc_l - auc_p))
    diffs_arr = np.asarray(diffs)
    auc_diff_ci = {
        "n_resamples": n_resamples,
        "n_valid": int(diffs_arr.size),
        "seed": ctx.seed,
        "mean": float(diffs_arr.mean()),
        "ci95_low": float(np.percentile(diffs_arr, 2.5)),
        "ci95_high": float(np.percentile(diffs_arr, 97.5)),
    }

    # Median single-window inference latency through the public API (both
    # detectors are warm from the evaluation loop above).
    def _median_latency_ms(det: Any) -> float:
        signal = condition_signal(ds.windows_int16[test_idx[0], 0])
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            det.detect_resonance_anomaly(signal)
            times.append((time.perf_counter() - t0) * 1000.0)
        return float(np.median(times))

    outcome = EvaluationOutcome(
        hook=HOOK_NAME,
        primary_metric="disturbed_auc",
        higher_is_better=True,
        learned=_metrics("learned"),
        physics=_metrics("physics"),
        n_test_samples=int(test_idx.size),
        test_years=SPLIT.test_years,
        extras={
            "comparison": (
                "identical held-out Sierra Nevada hours (12 conditioned windows each) "
                "through SchumannResonanceDetector.detect_resonance_anomaly at "
                "fs=256 Hz; physics = untrained deterministic FFT assessment, learned "
                "= same detector after load_neural_weights(candidate)"
            ),
            "decision_rule": (
                "hour flagged disturbed when >=50% of its 12 windows report "
                "anomaly_detected under the path's OWN deployed rule -- physics: "
                "deterministic spectral flags; learned: confidence > tau from the "
                "checkpoint's validation-selected operating point; hour score = mean "
                "window confidence; hour type = majority vote (ties to the lower "
                "class index)"
            ),
            "operating_point": operating_point,
            "window_derivation": (
                "fs=256 Hz; neural input = first 512 one-sided FFT bins spanning "
                "512*fs/n Hz; covering the 5-40 Hz Schumann band requires n <= 3276; "
                "n=3072 (12.0 s) gives df=1/12 Hz and 0-42.67 Hz coverage of all five "
                "modes (7.8/14.3/20.8/27.3/33.8 Hz); 12 windows/hour, 64 s apart, from "
                "the first ~12.2 min of each hour file (prefix-only ranged fetches)"
            ),
            "calibration_note": (
                "signals are raw ADC counts (station response ~flat 6-25 Hz per the "
                "published docs); the only conditioning is per-window mean removal, "
                "applied identically to both paths -- the detector max-normalizes its "
                "spectrum so absolute gain cancels, but an unremoved DC offset would "
                "dominate that normalization; the class rule's amplitude feature "
                "deliberately uses the raw |fft|^2/n periodogram to retain absolute "
                "scale"
            ),
            "class_rule": CLASS_RULE,
            "class_counts": ds.class_counts,
            "climatology": ds.climatology,
            "test_disturbed_fraction": float(disturbed.mean()),
            "per_storm_hits_test_years": storm_table,
            "per_storm_hits_validation_diagnostic": val_storm_table,
            "named_validation_storms_note": (
                "the 2015-03-17 St Patrick's Day G4 and 2015-06-22 G4 storms fall in "
                "the 2015 validation year by construction of the temporal split; they "
                "informed early stopping and operating-point selection only, never "
                "the merit-gate metrics -- their hit table above is diagnostic"
            ),
            "temporal_path_note": (
                "the LSTM temporal path was trained with an auxiliary loss over each "
                "hour's window-spectrum sequence; the deployed/evaluated decision is "
                "the spectrum-only path (temporal_history=None)"
            ),
            "auc_diff_bootstrap_ci95": auc_diff_ci,
            "model_parameter_count": int(
                sum(p.numel() for p in learned_det.harmonic_analyzer.parameters())
            ),
            "median_inference_latency_ms": {
                "learned": _median_latency_ms(learned_det),
                "physics": _median_latency_ms(physics_det),
                "n_runs": 100,
            },
        },
        constraints=[
            {
                "metric": "disturbed_recall_op",
                "higher_is_better": True,
                "description": (
                    "disturbed-hour detection recall at each path's deployed decision "
                    "rule must not regress below physics"
                ),
            },
            {
                "metric": "false_alarm_rate_op",
                "higher_is_better": False,
                "description": (
                    "quiet-hour false-alarm rate at each path's deployed decision "
                    "rule must not exceed physics"
                ),
            },
            {
                "metric": "four_class_accuracy",
                "higher_is_better": True,
                "description": (
                    "hour-level anomaly-type agreement with the derived labels must "
                    "not regress below the physics assessment"
                ),
            },
        ],
    )
    save_evaluation(ctx.data_dir, outcome)
    logger.info(
        "evaluation: learned disturbed AUC %.4f vs physics %.4f on %d held-out hours (%s)",
        outcome.learned["disturbed_auc"],
        outcome.physics["disturbed_auc"],
        outcome.n_test_samples,
        "LEARNED WINS" if outcome.learned_beats_physics else "PHYSICS WINS",
    )
    return outcome


# ---------------------------------------------------------------------------
# Stage 5: ship
# ---------------------------------------------------------------------------


def ship(ctx: PipelineContext) -> tuple[Any, Any]:
    """Promote the candidate through the merit gate (may refuse loudly)."""
    from omni_mercury_engine.ml.hazard_training.common import load_evaluation

    outcome = load_evaluation(ctx.data_dir, HOOK_NAME)
    manifest_path = ctx.data_dir / "schumann" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing fetch manifest {manifest_path}; run --fetch first")
    manifest = json.loads(manifest_path.read_text())
    return ship_checkpoint(
        hook=HOOK_NAME,
        checkpoint_name=CHECKPOINT_NAME,
        data_dir=ctx.data_dir,
        outcome=outcome,
        data_sources=manifest["sources"],
        seed=ctx.seed,
        out_dir=ctx.ship_dir,
    )
