# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the STEAD seismic-wave training pipeline (fully offline).

Fixture provenance (``tests/fixtures/hazard_training/seismic/stead_z_fixture.npz``):
four REAL traces (Z component, 6000 samples @ 100 Hz) from the STanford
EArthquake Dataset (STEAD; Mousavi et al. 2019, CC-BY-4.0), streamed on
2026-07-10 from the SeisBench mirror
``https://seisbench.gfz.de/mirror/datasets/stead/waveforms.hdf5`` (via HTTP
Range requests; labels/picks from the matching ``metadata.csv``,
sha256 ``9b9007406ebfef8c182060c8bb4266d29bbc433985f91f7e2dc476c8aca08efe``).
The exact ``trace_name`` of each fixture row is stored inside the npz and is
part of the pipeline's held-out TEST split (years 2017+): two
``earthquake_local`` traces and two ``noise`` traces. Nothing here is
synthesized; the fixture is a byte-exact subset of the upstream archive.

No network: pipeline-stage tests run against the fixture and (when present)
the shipped ``seismic_stead`` checkpoint; the Range-adapter tests exercise
the block math against a local temporary file through an injected fetch
callable.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

from omni_mercury_engine.ml.hazard_training.seismic_wave import (
    SPLIT,
    TRUNC_SAMPLES,
    WINDOW_SAMPLES,
    BlockCachedRangeReader,
    detector_spectrogram,
)
from omni_mercury_engine.models.checkpoint_paths import shipped_checkpoint_path

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "hazard_training"
    / "seismic"
    / "stead_z_fixture.npz"
)


@pytest.fixture(scope="module")
def stead_fixture() -> dict[str, np.ndarray]:
    """Load the real-STEAD fixture (2 earthquake + 2 noise Z traces)."""
    with np.load(FIXTURE, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


class _CountingLocalFetcher:
    """Injected transport: serves ranges from local bytes and records calls."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.calls: list[tuple[int, int]] = []

    def __call__(self, start: int, end: int) -> bytes:
        self.calls.append((start, end))
        return self.payload[start : end + 1]


class TestBlockCachedRangeReader:
    """Block math, LRU, preload, and h5py interop -- zero network."""

    def _payload(self, n: int = 100_000) -> bytes:
        # Deterministic non-repeating byte pattern (transport test data, not
        # training data): position-dependent so any offset bug shows up.
        return bytes((i * 31 + (i >> 8)) % 256 for i in range(n))

    def test_read_seek_tell_match_source_bytes(self) -> None:
        payload = self._payload()
        fetcher = _CountingLocalFetcher(payload)
        reader = BlockCachedRangeReader(fetcher, len(payload), block_size=4096)
        assert reader.read(10) == payload[:10]
        assert reader.tell() == 10
        reader.seek(4090)  # crosses a block boundary
        assert reader.read(20) == payload[4090:4110]
        reader.seek(-16, 2)  # SEEK_END
        assert reader.read() == payload[-16:]
        reader.seek(0)
        assert reader.read(0) == b""

    def test_fetches_are_block_aligned(self) -> None:
        payload = self._payload()
        fetcher = _CountingLocalFetcher(payload)
        reader = BlockCachedRangeReader(fetcher, len(payload), block_size=4096)
        reader.seek(5000)
        reader.read(9000)  # spans blocks 1..3
        for start, end in fetcher.calls:
            assert start % 4096 == 0
            assert end == min(start + 4096, len(payload)) - 1
        assert [s // 4096 for s, _ in fetcher.calls] == [1, 2, 3]

    def test_lru_eviction_and_reuse(self) -> None:
        payload = self._payload()
        fetcher = _CountingLocalFetcher(payload)
        reader = BlockCachedRangeReader(fetcher, len(payload), block_size=4096, max_cached_blocks=2)
        reader.seek(0)
        reader.read(1)  # block 0
        reader.read(1)  # cached
        assert len(fetcher.calls) == 1
        reader.seek(3 * 4096)
        reader.read(1)  # block 3
        reader.seek(6 * 4096)
        reader.read(1)  # block 6 -> evicts block 0
        reader.seek(0)
        reader.read(1)  # block 0 refetched
        assert [s // 4096 for s, _ in fetcher.calls] == [0, 3, 6, 0]

    def test_preload_serves_without_transport(self) -> None:
        payload = self._payload()
        fetcher = _CountingLocalFetcher(payload)
        reader = BlockCachedRangeReader(fetcher, len(payload), block_size=4096)
        reader.add_preload(500, payload[500:900])
        reader.seek(510)
        assert reader.read(100) == payload[510:610]
        assert fetcher.calls == []
        assert reader.preload_misses == 0
        reader.clear_preloads()
        reader.seek(510)
        assert reader.read(100) == payload[510:610]
        assert len(fetcher.calls) == 1  # block fallback after clearing

    def test_short_range_response_fails_loud(self) -> None:
        payload = self._payload()

        def bad_fetch(start: int, end: int) -> bytes:
            return payload[start:end]  # one byte short

        reader = BlockCachedRangeReader(bad_fetch, len(payload), block_size=4096)
        with pytest.raises(RuntimeError, match="ignored the Range header"):
            reader.read(1)

    def test_h5py_reads_through_reader(self, tmp_path: Path) -> None:
        """h5py must decode identical values through the adapter as directly."""
        path = tmp_path / "local.h5"
        values = np.arange(10 * 3 * 20, dtype=np.float32).reshape(10, 3, 20)
        with h5py.File(path, "w") as fh:
            fh.create_dataset("data/bucket0", data=values)
        payload = path.read_bytes()
        fetcher = _CountingLocalFetcher(payload)
        reader = BlockCachedRangeReader(fetcher, len(payload), block_size=1024)
        with h5py.File(reader, "r") as fh:
            ds = fh["data/bucket0"]
            assert ds.shape == (10, 3, 20)
            assert np.array_equal(ds[7, 0, :], values[7, 0, :])
            assert np.array_equal(ds[2], values[2])
        assert len(fetcher.calls) > 0  # everything flowed through the adapter


class TestDetectorSpectrogramParity:
    """The training-side preprocessing is the detector's, byte for byte."""

    def test_deterministic(self, stead_fixture: dict[str, np.ndarray]) -> None:
        trace = stead_fixture["z"][0]
        a = detector_spectrogram(trace)
        b = detector_spectrogram(trace)
        assert np.array_equal(a, b)

    def test_matches_detector_preprocessing(self, stead_fixture: dict[str, np.ndarray]) -> None:
        """Independent replication of predict_earthquake's exact pipeline."""
        from scipy import signal

        trace = stead_fixture["z"][0].astype(np.float64)
        _f, _t, sxx = signal.spectrogram(
            trace,
            fs=100.0,
            nperseg=min(256, len(trace) // 4),
            noverlap=min(128, len(trace) // 8),
        )
        sxx_log = np.log10(sxx + 1e-10)
        expected = ((sxx_log - sxx_log.mean()) / (sxx_log.std() + 1e-10)).astype(np.float32)
        assert np.array_equal(detector_spectrogram(trace), expected)

    def test_shapes_for_full_and_truncated_windows(
        self, stead_fixture: dict[str, np.ndarray]
    ) -> None:
        full = detector_spectrogram(stead_fixture["z"][0])
        trunc = detector_spectrogram(stead_fixture["z"][0][:TRUNC_SAMPLES])
        assert full.shape == (129, 45)
        assert trunc.shape == (129, 22)
        # z-normalization is per spectrogram: mean ~0, std ~1.
        assert abs(float(full.mean())) < 1e-4
        assert abs(float(full.std()) - 1.0) < 1e-3


class TestTemporalSplitEnforcement:
    """Train < val < test by year, matching the documented STEAD split."""

    def test_split_years_ordered_and_documented(self) -> None:
        assert max(SPLIT.train_years) == 2015
        assert SPLIT.val_years == (2016,)
        assert min(SPLIT.test_years) == 2017
        assert max(SPLIT.train_years) < min(SPLIT.val_years) < min(SPLIT.test_years)

    def test_masks_disjoint_on_fixture_years(self, stead_fixture: dict[str, np.ndarray]) -> None:
        years = stead_fixture["year"]
        train, val, test = SPLIT.masks(years)
        assert not np.any(train & val) and not np.any(val & test) and not np.any(train & test)
        # The fixture is drawn from the held-out TEST split only.
        assert bool(np.all(test))


class TestFixtureIntegrity:
    """The committed fixture is small, real, and self-describing."""

    def test_fixture_is_small_and_balanced(self, stead_fixture: dict[str, np.ndarray]) -> None:
        assert FIXTURE.stat().st_size < 300_000
        assert stead_fixture["z"].shape == (4, WINDOW_SAMPLES)
        assert str(stead_fixture["z"].dtype) == "float32"
        assert np.all(np.isfinite(stead_fixture["z"]))
        assert int(stead_fixture["label"].sum()) == 2  # 2 eq + 2 noise
        for name in stead_fixture["trace_name"]:
            assert str(name).startswith("bucket") and "$" in str(name)

    def test_earthquake_rows_carry_real_picks(self, stead_fixture: dict[str, np.ndarray]) -> None:
        is_eq = stead_fixture["label"].astype(bool)
        assert np.all(np.isfinite(stead_fixture["p_sample"][is_eq]))
        assert np.all(np.isfinite(stead_fixture["mag"][is_eq]))
        # Noise rows abstain from picks/magnitudes (NaN), never fake zeros.
        assert not np.any(np.isfinite(stead_fixture["p_sample"][~is_eq]))


class TestDetectorCheckpointWiring:
    """load_neural_weights: explicit path mechanics + shipped default."""

    def test_explicit_path_load_sets_trained_flag(self, tmp_path: Path) -> None:
        from omni_mercury_engine.detectors.geological.disaster_detectors import (
            EarthquakeDetector,
            SeismicWaveAnalyzer,
        )

        ckpt = tmp_path / "cand.pt"
        torch.save({"seismic_analyzer": SeismicWaveAnalyzer().state_dict()}, ckpt)
        detector = EarthquakeDetector()
        assert detector._neural_trained is False
        detector.load_neural_weights(str(ckpt))
        assert detector._neural_trained is True

    def test_corrupt_checkpoint_fails_loud(self, tmp_path: Path) -> None:
        from omni_mercury_engine.detectors.geological.disaster_detectors import (
            EarthquakeDetector,
        )

        bad = tmp_path / "bad.pt"
        bad.write_bytes(b"not a checkpoint")
        detector = EarthquakeDetector()
        with pytest.raises((pickle.UnpicklingError, RuntimeError)):
            detector.load_neural_weights(str(bad))
        assert detector._neural_trained is False

    def test_shipped_default_differs_from_physics_on_real_waveforms(
        self, stead_fixture: dict[str, np.ndarray]
    ) -> None:
        """Differential test: physics vs shipped checkpoint on real STEAD data.

        Skips cleanly when no ``seismic_stead`` checkpoint has been shipped
        (the merit gate may legitimately refuse; the physics fallback then
        remains in charge and this wiring test has nothing to verify).
        """
        if not shipped_checkpoint_path("seismic_stead").exists():
            pytest.skip("no shipped seismic_stead checkpoint (merit gate may have refused)")
        from omni_mercury_engine.detectors.geological.disaster_detectors import (
            EarthquakeDetector,
        )

        physics = EarthquakeDetector()
        learned = EarthquakeDetector()
        learned.load_neural_weights()  # no path -> shipped default
        assert physics._neural_trained is False
        assert learned._neural_trained is True

        eq_indices = np.flatnonzero(stead_fixture["label"] == 1)
        confidences: dict[str, list[float]] = {"physics": [], "learned": []}
        for i in eq_indices:
            trace = stead_fixture["z"][int(i)]
            res_p = physics.predict_earthquake(trace)
            res_l = learned.predict_earthquake(trace)
            confidences["physics"].append(res_p.confidence)
            confidences["learned"].append(res_l.confidence)
            # Physics abstains from magnitude; the learned path estimates one.
            assert res_p.estimated_magnitude is None
            assert res_l.estimated_magnitude is not None
            assert np.isfinite(res_l.estimated_magnitude)
        assert confidences["physics"] != confidences["learned"]
