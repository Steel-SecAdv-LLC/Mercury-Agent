# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Offline tests for the Schumann-harmonics checkpoint-training pipeline.

No network: the remote-ZIP reader is exercised against a local ZIP built
in-test from committed REAL station bytes, and the spectrum-parity tests run
on the same committed excerpt.

Fixture provenance (``tests/fixtures/hazard_training/schumann/``): the two
``.npz`` files are verbatim window extracts (first 12 x 3072 int16 samples,
64 s apart) from Sierra Nevada ELF station hour files
``smplGRTU1_sensor_0_*`` (NS component, fs = 256 Hz, raw little-endian int16
ADC counts) inside the Zenodo year archives (records 6348691 / 6348773 /
6348838 / 6348930; CC-BY-4.0; Salinas et al. 2022, Computers & Geosciences
165:105148), fetched by the training pipeline's ranged-GET reader. One hour
is geomagnetically quiet (GFZ definitive Kp <= 2), one disturbed (Kp >= 5);
both fall in the held-out test years. See ``provenance.json`` next to the
fixtures for the exact member names, times, and Kp values.
"""

from __future__ import annotations

import io
import json
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from omni_mercury_engine.ml.hazard_training.schumann_harmonics import (
    ANOMALY_CLASSES,
    FS_HZ,
    PREFIX_BYTES,
    PREFIX_SAMPLES,
    SPECTRUM_BINS,
    SPLIT,
    WINDOW_SAMPLES,
    WINDOW_STRIDE,
    WINDOWS_PER_HOUR,
    RemoteZipReader,
    _label_hour,
    _majority_type,
    _parse_member_time,
    compute_detector_spectrum,
    condition_signal,
    derive_class_labels,
)
from omni_mercury_engine.models.checkpoint_paths import shipped_checkpoint_path
from omni_mercury_engine.space.schumann_resonance import (
    SchumannHarmonicAnalyzer,
    SchumannResonanceDetector,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "hazard_training" / "schumann"


def _load_fixture(name: str) -> dict[str, np.ndarray]:
    path = FIXTURE_DIR / name
    if not path.exists():
        pytest.skip(f"fixture {path} not committed")
    with np.load(path) as npz:
        return {k: npz[k] for k in npz.files}


@pytest.fixture(scope="module")
def quiet_hour() -> dict[str, np.ndarray]:
    return _load_fixture("hour_quiet_2016.npz")


@pytest.fixture(scope="module")
def disturbed_hour() -> dict[str, np.ndarray]:
    return _load_fixture("hour_disturbed_2016.npz")


class TestFixtureIntegrity:
    def test_windows_shape_and_dtype(
        self, quiet_hour: dict[str, np.ndarray], disturbed_hour: dict[str, np.ndarray]
    ) -> None:
        for fx in (quiet_hour, disturbed_hour):
            assert fx["windows"].shape == (WINDOWS_PER_HOUR, WINDOW_SAMPLES)
            assert fx["windows"].dtype == np.int16

    def test_labels_and_era(
        self, quiet_hour: dict[str, np.ndarray], disturbed_hour: dict[str, np.ndarray]
    ) -> None:
        assert not bool(quiet_hour["disturbed"])
        assert bool(disturbed_hour["disturbed"])
        assert float(quiet_hour["kp"]) <= 2.0
        assert float(disturbed_hour["kp"]) >= 5.0
        for fx in (quiet_hour, disturbed_hour):
            year = int(str(fx["t0_iso"])[:4])
            assert year in SPLIT.test_years  # committed excerpts are held-out hours

    def test_provenance_sidecar_present(self) -> None:
        path = FIXTURE_DIR / "provenance.json"
        if not path.exists():
            pytest.skip("fixtures not committed")
        prov = json.loads(path.read_text())
        assert "zenodo" in json.dumps(prov).lower()
        assert len(prov["hours"]) == 2


class TestRemoteZipReader:
    """Ranged-access ZIP reader against a local archive of real fixture bytes."""

    @pytest.fixture(scope="class")
    def real_bytes(self) -> bytes:
        fx = _load_fixture("hour_quiet_2016.npz")
        return fx["windows"].astype("<i2").tobytes()

    @pytest.fixture(scope="class")
    def local_zip(self, real_bytes: bytes) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(
                zipfile.ZipInfo("2016/1601/smplGRTU1_sensor_0_1601010203"),
                real_bytes,
                compress_type=zipfile.ZIP_DEFLATED,
            )
            zf.writestr(
                zipfile.ZipInfo("2016/1601/smplGRTU1_sensor_0_1601010203_info.txt"),
                b"sampling period (usec): 3906.000000\n",
                compress_type=zipfile.ZIP_STORED,
            )
        return buf.getvalue()

    @staticmethod
    def _reader(blob: bytes) -> RemoteZipReader:
        def fetch_range(spec: str) -> bytes:
            if spec.startswith("-"):
                n = int(spec[1:])
                return blob[-n:] if n < len(blob) else blob
            start_s, end_s = spec.split("-")
            return blob[int(start_s) : min(int(end_s) + 1, len(blob))]

        return RemoteZipReader(fetch_range)

    def test_directory_and_full_read(self, local_zip: bytes, real_bytes: bytes) -> None:
        reader = self._reader(local_zip)
        name = "2016/1601/smplGRTU1_sensor_0_1601010203"
        assert set(reader.members) == {name, name + "_info.txt"}
        member = reader.members[name]
        assert member.method == 8  # deflate
        assert member.uncompressed_size == len(real_bytes)
        assert reader.read_member(name) == real_bytes  # size + CRC verified inside

    def test_prefix_reads_deflate_and_stored(self, local_zip: bytes, real_bytes: bytes) -> None:
        reader = self._reader(local_zip)
        name = "2016/1601/smplGRTU1_sensor_0_1601010203"
        for n in (WINDOW_SAMPLES * 2, 1, len(real_bytes)):
            assert reader.read_member_prefix(name, n) == real_bytes[:n]
        # Stored member: exact ranged read, no inflation.
        assert reader.read_member_prefix(name + "_info.txt", 15) == b"sampling period"

    def test_group_prefixes_match_individual_reads(
        self, local_zip: bytes, real_bytes: bytes
    ) -> None:
        reader = self._reader(local_zip)
        name = "2016/1601/smplGRTU1_sensor_0_1601010203"
        info = name + "_info.txt"
        n = WINDOW_SAMPLES * 4
        grouped = reader.read_group_prefixes([info, name], n)
        assert grouped[name] == real_bytes[:n]
        assert grouped[info] == reader.read_member_prefix(info, n)

    def test_corrupt_crc_fails_loud(self, local_zip: bytes) -> None:
        name = "2016/1601/smplGRTU1_sensor_0_1601010203"
        clean = self._reader(local_zip)
        member = clean.members[name]
        # Flip one byte inside the member's compressed payload.
        broken = bytearray(local_zip)
        broken[member.header_offset + 200] ^= 0xFF
        reader = self._reader(bytes(broken))
        with pytest.raises((RuntimeError, zlib.error)):
            reader.read_member(name)

    def test_int16_decode_round_trip(
        self, real_bytes: bytes, quiet_hour: dict[str, np.ndarray]
    ) -> None:
        decoded = np.frombuffer(real_bytes, dtype="<i2").reshape(WINDOWS_PER_HOUR, WINDOW_SAMPLES)
        assert np.array_equal(decoded, quiet_hour["windows"])


class TestSpectrumParity:
    """Training features must be byte-identical to the detector's preprocessing."""

    def test_window_derivation_constants(self) -> None:
        # First 512 one-sided bins of an n-sample window span 512*fs/n Hz;
        # they must cover all five Schumann modes and the 5-40 Hz physics band.
        covered_hz = SPECTRUM_BINS * FS_HZ / WINDOW_SAMPLES
        assert covered_hz >= 40.0
        for mode_hz in (7.83, 14.3, 20.8, 27.3, 33.8):
            assert mode_hz < covered_hz
        assert WINDOW_SAMPLES // 2 >= SPECTRUM_BINS  # spectrum long enough to slice
        assert PREFIX_SAMPLES == (WINDOWS_PER_HOUR - 1) * WINDOW_STRIDE + WINDOW_SAMPLES
        assert PREFIX_BYTES == 2 * PREFIX_SAMPLES

    def test_parity_with_detector_private_pipeline(
        self, disturbed_hour: dict[str, np.ndarray]
    ) -> None:
        det = SchumannResonanceDetector(sampling_rate=FS_HZ)
        for window in disturbed_hour["windows"][:4]:
            signal = condition_signal(window)
            ours = compute_detector_spectrum(signal)
            power, _freqs = det._compute_power_spectrum(signal)
            theirs = torch.tensor(power[:SPECTRUM_BINS], dtype=torch.float32)
            assert torch.equal(torch.from_numpy(ours), theirs)

    def test_parity_through_public_api_diagnostics(self, quiet_hour: dict[str, np.ndarray]) -> None:
        det = SchumannResonanceDetector(sampling_rate=FS_HZ, keep_diagnostics=True)
        signal = condition_signal(quiet_hour["windows"][0])
        result = det.detect_resonance_anomaly(signal)
        assert result.diagnostics is not None
        api_spectrum = result.diagnostics.arrays["power_spectrum"]
        ours = compute_detector_spectrum(signal)
        assert np.array_equal(ours, api_spectrum[:SPECTRUM_BINS].astype(np.float32))

    def test_conditioning_is_demean_only(self, quiet_hour: dict[str, np.ndarray]) -> None:
        window = quiet_hour["windows"][0]
        signal = condition_signal(window)
        assert signal.dtype == np.float64
        assert abs(float(signal.mean())) < 1e-9
        assert np.allclose(signal, window.astype(np.float64) - window.astype(np.float64).mean())


class TestSplitEnforcement:
    def test_split_years(self) -> None:
        assert SPLIT.train_years == (2013, 2014)
        assert SPLIT.val_years == (2015,)
        assert SPLIT.test_years == (2016, 2017)

    def test_climatology_uses_train_quiet_only(self) -> None:
        rng = np.random.default_rng(7)
        n = 40
        log_bp = rng.normal(0.0, 1.0, n)
        centroid = rng.normal(7.8, 0.1, n)
        disturbed = np.zeros(n)
        disturbed[30:] = 1.0
        train_quiet = np.zeros(n, dtype=bool)
        train_quiet[:20] = True
        labels_a, clim_a = derive_class_labels(log_bp, centroid, disturbed, train_quiet)
        # Perturbing rows OUTSIDE the train-quiet mask that are not disturbed
        # must leave the climatology untouched (no leakage from val/test).
        log_bp2 = log_bp.copy()
        log_bp2[20:30] += 100.0
        _labels_b, clim_b = derive_class_labels(log_bp2, centroid, disturbed, train_quiet)
        assert clim_a == clim_b
        assert np.array_equal(labels_a[:20], np.zeros(20))  # quiet rows stay "normal"

    def test_refuses_without_train_quiet_hours(self) -> None:
        n = 12
        with pytest.raises(RuntimeError, match="quiet"):
            derive_class_labels(np.zeros(n), np.full(n, 7.8), np.ones(n), np.zeros(n, dtype=bool))


class TestLabelRuleDeterminism:
    def _base(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # 20 train-quiet hours with known stats, then 4 disturbed probes.
        log_bp = np.concatenate([np.linspace(-1, 1, 20), np.zeros(4)])
        centroid = np.concatenate([np.linspace(7.7, 7.9, 20), np.full(4, 7.8)])
        disturbed = np.concatenate([np.zeros(20), np.ones(4)])
        train_quiet = np.concatenate([np.ones(20, dtype=bool), np.zeros(4, dtype=bool)])
        return log_bp, centroid, disturbed, train_quiet

    def test_subclass_rule(self) -> None:
        log_bp, centroid, disturbed, train_quiet = self._base()
        sd_bp = float(np.std(log_bp[:20]))
        sd_c = float(np.std(centroid[:20]))
        log_bp[20] += 3.0 * sd_bp  # amplitude only
        centroid[21] += 3.0 * sd_c  # frequency only
        log_bp[22] += 3.0 * sd_bp
        centroid[22] += 3.0 * sd_c  # combined
        log_bp[23] += 0.5 * sd_bp
        centroid[23] += 0.2 * sd_c  # below both cuts -> larger z (amplitude)
        labels, _ = derive_class_labels(log_bp, centroid, disturbed, train_quiet)
        assert labels[20] == ANOMALY_CLASSES.index("amplitude")
        assert labels[21] == ANOMALY_CLASSES.index("frequency")
        assert labels[22] == ANOMALY_CLASSES.index("combined")
        assert labels[23] == ANOMALY_CLASSES.index("amplitude")
        labels_again, _ = derive_class_labels(log_bp, centroid, disturbed, train_quiet)
        assert np.array_equal(labels, labels_again)  # deterministic

    def test_kp_hour_labeling(self) -> None:
        t0 = _parse_member_time("2016/1601/smplGRTU1_sensor_0_1601010203")
        assert t0 is not None and t0.year == 2016
        b = int(t0.timestamp()) // 10800

        def near(kp: float) -> dict[int, float]:
            return dict.fromkeys((b - 1, b, b + 1), kp)

        assert _label_hour(t0, near(6.0))[0] == "disturbed"
        assert _label_hour(t0, near(1.667))[0] == "quiet"
        assert _label_hour(t0, near(3.0))[0] == "intermediate"
        assert _label_hour(t0, {})[0] == "missing_kp"
        # Sensor 1 (EW) members are never selected -- sensor 0 only.
        assert _parse_member_time("2016/1601/smplGRTU1_sensor_1_1601010203") is None

    def test_majority_type_tie_breaks_low_index(self) -> None:
        assert _majority_type(["amplitude", "frequency"]) == "amplitude"
        assert _majority_type(["frequency", "frequency", "normal"]) == "frequency"
        assert _majority_type(["combined"]) == "combined"


class TestLoadNeuralWeightsCompat:
    """Bare state_dict (legacy), bare path, wrapped payload path all load."""

    def _fresh_state(self) -> dict[str, torch.Tensor]:
        torch.manual_seed(11)
        state: dict[str, torch.Tensor] = SchumannHarmonicAnalyzer(
            spectrum_size=SPECTRUM_BINS
        ).state_dict()
        return state

    def test_bare_in_memory(self) -> None:
        det = SchumannResonanceDetector(sampling_rate=FS_HZ)
        assert not det._neural_trained
        det.load_neural_weights(self._fresh_state())
        assert det._neural_trained

    def test_bare_path(self, tmp_path: Path) -> None:
        path = tmp_path / "bare.pt"
        torch.save(self._fresh_state(), path)
        det = SchumannResonanceDetector(sampling_rate=FS_HZ)
        det.load_neural_weights(str(path))
        assert det._neural_trained

    def test_wrapped_payload_path(self, tmp_path: Path) -> None:
        path = tmp_path / "wrapped.pt"
        torch.save(
            {
                "harmonic_analyzer": self._fresh_state(),
                "feature_spec": "schumann-sn-v1",
                "fs_hz": 256.0,
            },
            path,
        )
        det = SchumannResonanceDetector(sampling_rate=FS_HZ)
        det.load_neural_weights(str(path))
        assert det._neural_trained

    def test_default_load_requires_shipped_checkpoint(self) -> None:
        det = SchumannResonanceDetector(sampling_rate=FS_HZ)
        if shipped_checkpoint_path("schumann_sierra_nevada").exists():
            det.load_neural_weights()
            assert det._neural_trained
        else:
            with pytest.raises(FileNotFoundError, match="schumann_sierra_nevada"):
                det.load_neural_weights()


class TestOperatingPointConsumption:
    """The checkpoint's ratified threshold governs the learned decision.

    Mirrors the solar-storm/tsunami dual-rule operating-point tests: the
    validation-selected tau carried by the checkpoint is part of the deployed
    rule, so loading it must (a) validate it before any state mutates,
    (b) apply it to the learned path's ``anomaly_detected`` decision without
    touching the confidence estimate or anomaly_type, and (c) leave the
    historical physics-flag decision in charge for payloads that predate the
    convention (and for bare state_dicts).
    """

    @staticmethod
    def _write_payload(path: Path, operating_point: dict[str, float] | None) -> Path:
        torch.manual_seed(11)
        payload: dict[str, object] = {
            "harmonic_analyzer": SchumannHarmonicAnalyzer(spectrum_size=SPECTRUM_BINS).state_dict(),
            "feature_spec": "schumann-sn-v1",
        }
        if operating_point is not None:
            payload["operating_point"] = operating_point
        torch.save(payload, path)
        return path

    def test_operating_point_drives_learned_decision(
        self, tmp_path: Path, disturbed_hour: dict[str, np.ndarray]
    ) -> None:
        """tau just below/above the emitted confidence must flip the decision."""
        det = SchumannResonanceDetector(sampling_rate=FS_HZ)
        det.load_neural_weights(
            str(self._write_payload(tmp_path / "op.pt", {"detection_threshold": 0.5}))
        )
        assert det._operating_point == {"detection_threshold": 0.5}

        signal = condition_signal(disturbed_hour["windows"][0])
        base = det.detect_resonance_anomaly(signal)
        conf = float(base.confidence)
        assert 0.0 < conf < 1.0  # sigmoid head keeps confidence interior

        det._operating_point = {"detection_threshold": max(conf - 1e-6, 1e-9)}
        below = det.detect_resonance_anomaly(signal)
        assert below.anomaly_detected is True
        assert below.confidence == pytest.approx(conf)
        assert below.anomaly_type == base.anomaly_type

        det._operating_point = {"detection_threshold": conf + (1.0 - conf) / 2.0}
        above = det.detect_resonance_anomaly(signal)
        assert above.anomaly_detected is False
        assert above.confidence == pytest.approx(
            conf
        ), "the operating point changes the DECISION, never the confidence estimate"
        assert above.anomaly_type == base.anomaly_type

    def test_operating_point_threshold_validated_on_load(self, tmp_path: Path) -> None:
        """A payload carrying a nonsensical tau must refuse before mutating."""
        det = SchumannResonanceDetector(sampling_rate=FS_HZ)
        for bad_tau in (1.5, 0.0, float("nan")):
            path = self._write_payload(
                tmp_path / f"bad_{bad_tau}.pt", {"detection_threshold": bad_tau}
            )
            with pytest.raises(ValueError, match=r"not a\s+probability"):
                det.load_neural_weights(str(path))
            assert det._neural_trained is False  # validated BEFORE any state mutated
            assert det._operating_point is None

    def test_payload_without_operating_point_keeps_physics_decision(
        self, tmp_path: Path, quiet_hour: dict[str, np.ndarray]
    ) -> None:
        """Pre-convention payloads and bare state_dicts keep the old decision."""
        physics = SchumannResonanceDetector(sampling_rate=FS_HZ)
        learned = SchumannResonanceDetector(sampling_rate=FS_HZ)
        learned.load_neural_weights(str(self._write_payload(tmp_path / "old.pt", None)))
        assert learned._neural_trained is True
        assert learned._operating_point is None
        signal = condition_signal(quiet_hour["windows"][0])
        res_l = learned.detect_resonance_anomaly(signal)
        res_p = physics.detect_resonance_anomaly(signal)
        # anomaly_detected stays the deterministic physics flags on both paths.
        assert res_l.anomaly_detected == res_p.anomaly_detected


@pytest.mark.skipif(
    not shipped_checkpoint_path("schumann_sierra_nevada").exists(),
    reason="schumann_sierra_nevada checkpoint not shipped",
)
class TestDifferentialPhysicsVsShipped:
    """The shipped checkpoint must behave differently from (and alongside) physics."""

    def test_shipped_payload_contract(self) -> None:
        from omni_mercury_engine.models.checkpoint_paths import load_shipped_checkpoint

        payload, provenance = load_shipped_checkpoint("schumann_sierra_nevada")
        assert payload["feature_spec"] == "schumann-sn-v1"
        assert payload["fs_hz"] == FS_HZ
        assert payload["window_samples"] == WINDOW_SAMPLES
        assert payload["spectrum_bins"] == SPECTRUM_BINS
        assert "class_rule" in payload and "Sierra Nevada" in payload["station"]
        assert 0.0 < float(payload["operating_point"]["detection_threshold"]) < 1.0
        assert provenance is not None
        assert provenance["evaluation"]["learned_beats_physics"] is True

    def test_differential_on_real_windows(
        self, quiet_hour: dict[str, np.ndarray], disturbed_hour: dict[str, np.ndarray]
    ) -> None:
        physics = SchumannResonanceDetector(sampling_rate=FS_HZ)
        learned = SchumannResonanceDetector(sampling_rate=FS_HZ)
        learned.load_neural_weights()
        diffs = 0
        for fx in (quiet_hour, disturbed_hour):
            for window in fx["windows"][:6]:
                signal = condition_signal(window)
                res_p = physics.detect_resonance_anomaly(signal)
                res_l = learned.detect_resonance_anomaly(signal)
                assert res_l.anomaly_type in ANOMALY_CLASSES
                assert 0.0 <= res_l.confidence <= 1.0
                # Repeatability of the learned path (eval mode, no dropout).
                res_l2 = learned.detect_resonance_anomaly(signal)
                assert res_l2.confidence == res_l.confidence
                assert res_l2.anomaly_type == res_l.anomaly_type
                diffs += int(res_l.confidence != res_p.confidence)
        assert diffs > 0  # learned confidences come from the network, not the physics formula
