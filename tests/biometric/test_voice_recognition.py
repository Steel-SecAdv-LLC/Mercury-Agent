# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic unit tests for the voice recognition module.

Exercises the full public API of
``omni_mercury_engine.biometric.voice_recognition`` -- preprocessing,
MFCC/pitch/energy feature extraction, speaker embeddings, matching,
liveness detection, voice-activity detection and the top-level
``VoiceRecognizer`` orchestration -- using only seeded ``numpy`` synthetic
audio (no torch, no network, no wall-clock).
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.biometric.voice_recognition import (
    AudioPreprocessor,
    EnergyExtractor,
    MFCCExtractor,
    PitchExtractor,
    SpeakerEmbedding,
    VoiceActivityDetector,
    VoiceFeatures,
    VoiceLivenessDetector,
    VoiceLivenessResult,
    VoiceMatcher,
    VoiceMatchResult,
    VoiceRecognizer,
)

SR = 16000
SEED = 20250721


def make_voice(f0: float = 150.0, seed: int = 0, dur: float = 1.0) -> np.ndarray:
    """Build a deterministic, voiced-sounding synthetic signal.

    Sum of the first five harmonics of ``f0`` plus a little seeded noise.
    Produces a strongly periodic waveform so the autocorrelation pitch
    tracker reports a stable fundamental.
    """
    n = int(SR * dur)
    tt = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    sig = np.zeros(n)
    for k in range(1, 6):
        sig += (1.0 / k) * np.sin(2 * np.pi * f0 * k * tt)
    sig += 0.05 * rng.standard_normal(n)
    return sig


def make_noise(seed: int = 0, dur: float = 1.0) -> np.ndarray:
    """Build deterministic broadband white noise."""
    n = int(SR * dur)
    return np.random.default_rng(seed).standard_normal(n)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
class TestResultDataclasses:
    def test_voice_features_fields(self) -> None:
        vf = VoiceFeatures(
            mfcc=np.zeros((3, 13)),
            delta_mfcc=np.zeros((3, 13)),
            delta2_mfcc=np.zeros((3, 13)),
            embedding=np.zeros(256),
            pitch_contour=np.zeros(3),
            energy_contour=np.zeros(3),
            duration=1.5,
            sample_rate=SR,
            quality_score=0.8,
        )
        assert vf.duration == 1.5
        assert vf.sample_rate == SR
        assert vf.mfcc.shape == (3, 13)

    def test_voice_match_result_fields(self) -> None:
        r = VoiceMatchResult(
            similarity_score=0.9,
            is_match=True,
            confidence=1.0,
            embedding_distance=0.1,
            mfcc_distance=0.2,
        )
        assert r.is_match is True
        assert r.similarity_score == 0.9

    def test_voice_liveness_result_default_details(self) -> None:
        r = VoiceLivenessResult(
            is_live=True,
            confidence=0.6,
            replay_score=0.7,
            synthetic_score=0.5,
            channel_score=0.6,
        )
        # ``details`` defaults to an empty dict via default_factory.
        assert r.details == {}
        assert r.is_live is True


# ---------------------------------------------------------------------------
# AudioPreprocessor
# ---------------------------------------------------------------------------
class TestAudioPreprocessor:
    def test_frame_geometry(self) -> None:
        pre = AudioPreprocessor(SR)
        # 25 ms @ 16 kHz = 400 samples, 10 ms shift = 160 samples.
        assert pre.frame_length == 400
        assert pre.frame_shift == 160

    def test_preprocess_normalizes_and_dtype(self) -> None:
        pre = AudioPreprocessor(SR)
        sig = make_voice(150.0, 1) * 500.0  # large amplitude
        out = pre.preprocess(sig)
        assert out.dtype == np.float64
        assert out.ndim == 1
        assert out.shape == sig.shape
        # first sample is untouched by pre-emphasis (emphasized[0] = signal[0])
        norm = sig / (np.max(np.abs(sig)) + 1e-8)
        assert out[0] == pytest.approx(norm[0])
        # interior samples follow the pre-emphasis recurrence
        assert out[5] == pytest.approx(norm[5] - 0.97 * norm[4])

    def test_preprocess_stereo_is_downmixed(self) -> None:
        pre = AudioPreprocessor(SR)
        mono = make_voice(150.0, 2)
        stereo = np.stack([mono, mono * 0.5], axis=1)
        out = pre.preprocess(stereo)
        assert out.ndim == 1
        assert out.shape[0] == mono.shape[0]

    def test_preprocess_all_zero_signal(self) -> None:
        pre = AudioPreprocessor(SR)
        out = pre.preprocess(np.zeros(1000))
        assert np.allclose(out, 0.0)

    def test_frame_shapes_and_window(self) -> None:
        pre = AudioPreprocessor(SR)
        sig = pre.preprocess(make_voice(150.0, 3))
        frames = pre.frame(sig)
        n_expected = 1 + (len(sig) - pre.frame_length) // pre.frame_shift
        assert frames.shape == (n_expected, pre.frame_length)
        # Hamming window tapers frame edges toward zero relative to centre.
        assert np.max(np.abs(frames[:, 0])) < np.max(np.abs(frames[:, 200]))

    def test_frame_short_signal_is_padded_to_one_frame(self) -> None:
        pre = AudioPreprocessor(SR)
        short = pre.preprocess(np.ones(100))
        frames = pre.frame(short)
        # n_frames clamped to at least 1; frame is zero-padded past the tail.
        assert frames.shape == (1, pre.frame_length)
        assert np.all(frames[0, 100:] == 0.0)


# ---------------------------------------------------------------------------
# MFCCExtractor
# ---------------------------------------------------------------------------
class TestMFCCExtractor:
    def test_matrix_shapes(self) -> None:
        mf = MFCCExtractor(SR, n_mfcc=13, n_fft=512, n_mels=40)
        assert mf._mel_filterbank.shape == (40, 512 // 2 + 1)
        assert mf._dct_matrix.shape == (13, 40)

    def test_extract_shape(self) -> None:
        pre = AudioPreprocessor(SR)
        frames = pre.frame(pre.preprocess(make_voice(150.0, 4)))
        mf = MFCCExtractor(SR)
        mfcc = mf.extract(frames)
        assert mfcc.shape == (frames.shape[0], 13)
        assert np.all(np.isfinite(mfcc))

    def test_deltas_shape_and_constant_is_zero(self) -> None:
        mf = MFCCExtractor(SR)
        feats = np.tile(np.arange(13, dtype=float), (20, 1))  # constant over time
        deltas = mf.compute_deltas(feats)
        assert deltas.shape == feats.shape
        # Derivative of a time-constant feature is exactly zero everywhere.
        assert np.allclose(deltas, 0.0)

    def test_deltas_ramp_is_nonzero(self) -> None:
        mf = MFCCExtractor(SR)
        ramp = np.outer(np.arange(30, dtype=float), np.ones(13))  # rises over time
        deltas = mf.compute_deltas(ramp)
        # Interior of a linear ramp has a constant positive slope.
        assert np.all(deltas[5:25] > 0)

    def test_mel_hz_roundtrip(self) -> None:
        mf = MFCCExtractor(SR)
        hz = np.array([100.0, 1000.0, 4000.0])
        back = mf._mel_to_hz(mf._hz_to_mel(hz))
        assert np.allclose(back, hz, rtol=1e-6)


# ---------------------------------------------------------------------------
# PitchExtractor
# ---------------------------------------------------------------------------
class TestPitchExtractor:
    def test_lag_bounds(self) -> None:
        pe = PitchExtractor(SR, min_pitch=50.0, max_pitch=500.0)
        assert pe._min_lag == int(SR / 500.0)
        assert pe._max_lag == int(SR / 50.0)

    def test_voiced_signal_tracks_fundamental(self) -> None:
        pre = AudioPreprocessor(SR)
        frames = pre.frame(pre.preprocess(make_voice(150.0, 5)))
        pe = PitchExtractor(SR)
        pitch = pe.extract(frames)
        assert pitch.shape == (frames.shape[0],)
        voiced = pitch[pitch > 0]
        assert voiced.size > 0
        assert 135.0 < float(np.mean(voiced)) < 165.0

    def test_white_noise_is_unvoiced(self) -> None:
        # A single noisy frame has weak periodicity -> pitch reported as 0.
        pe = PitchExtractor(SR)
        frame = np.random.default_rng(7).standard_normal(400)
        assert pe._estimate_pitch(frame) == 0.0

    def test_short_frame_returns_zero(self) -> None:
        # Frame shorter than min_lag leaves the autocorrelation window empty.
        pe = PitchExtractor(SR)
        assert pe._estimate_pitch(np.ones(10)) == 0.0


# ---------------------------------------------------------------------------
# EnergyExtractor
# ---------------------------------------------------------------------------
class TestEnergyExtractor:
    def test_log_energy_shape_and_order(self) -> None:
        ee = EnergyExtractor()
        loud = np.ones((1, 400)) * 2.0
        quiet = np.ones((1, 400)) * 0.5
        frames = np.concatenate([loud, quiet], axis=0)
        energy = ee.extract(frames)
        assert energy.shape == (2,)
        assert energy[0] > energy[1]

    def test_zero_frame_floored(self) -> None:
        ee = EnergyExtractor(floor=1e-10)
        energy = ee.extract(np.zeros((1, 400)))
        # log(floor) rather than log(0) == -inf
        assert energy[0] == pytest.approx(np.log(1e-10))


# ---------------------------------------------------------------------------
# SpeakerEmbedding
# ---------------------------------------------------------------------------
class TestSpeakerEmbedding:
    def _features(self, seed: int) -> np.ndarray:
        return np.random.default_rng(seed).standard_normal((40, 39))

    def test_embedding_shape_and_unit_norm(self) -> None:
        se = SpeakerEmbedding(input_dim=39, embedding_dim=256)
        emb = se.generate(self._features(11))
        assert emb.shape == (256,)
        assert float(np.linalg.norm(emb)) == pytest.approx(1.0, abs=1e-6)

    def test_default_seed_is_deterministic(self) -> None:
        feats = self._features(12)
        a = SpeakerEmbedding().generate(feats)
        b = SpeakerEmbedding().generate(feats)
        assert np.array_equal(a, b)

    def test_different_seed_changes_weights(self) -> None:
        feats = self._features(13)
        a = SpeakerEmbedding(seed=1).generate(feats)
        b = SpeakerEmbedding(seed=2).generate(feats)
        assert not np.allclose(a, b)

    def test_zero_features_yield_zero_embedding(self) -> None:
        # ReLU of zeros stays zero -> pooled vector is zero -> norm==0 branch
        # leaves the embedding unnormalised (all zeros) rather than dividing.
        se = SpeakerEmbedding()
        emb = se.generate(np.zeros((10, 39)))
        assert np.allclose(emb, 0.0)


# ---------------------------------------------------------------------------
# VoiceMatcher
# ---------------------------------------------------------------------------
class TestVoiceMatcher:
    def _features(self, f0: float, seed: int) -> VoiceFeatures:
        return VoiceRecognizer(SR).extract_features(make_voice(f0, seed))

    def test_self_match_is_perfect(self) -> None:
        feats = self._features(150.0, 21)
        res = VoiceMatcher(0.7).match(feats, feats)
        assert res.similarity_score == pytest.approx(1.0)
        assert res.is_match is True
        assert res.confidence == pytest.approx(1.0)
        assert res.embedding_distance == pytest.approx(0.0, abs=1e-9)
        assert res.mfcc_distance == pytest.approx(0.0, abs=1e-9)

    def test_match_confidence_formula_when_matched(self) -> None:
        feats = self._features(150.0, 22)
        thr = 0.5
        res = VoiceMatcher(thr).match(feats, feats)
        assert res.is_match is True
        assert res.confidence == pytest.approx(min(1.0, res.similarity_score / thr))

    def test_non_match_confidence_is_half_similarity(self) -> None:
        probe = self._features(240.0, 23)
        enrolled = self._features(150.0, 24)
        # A near-impossible threshold forces the non-match branch.
        res = VoiceMatcher(0.999).match(probe, enrolled)
        assert res.is_match is False
        assert res.confidence == pytest.approx(res.similarity_score * 0.5)
        assert 0.0 <= res.similarity_score <= 1.0

    def test_cosine_similarity_zero_vector(self) -> None:
        m = VoiceMatcher()
        assert m._cosine_similarity(np.zeros(5), np.ones(5)) == 0.0

    def test_cosine_similarity_identical(self) -> None:
        m = VoiceMatcher()
        v = np.array([1.0, 2.0, 3.0])
        assert m._cosine_similarity(v, v) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# VoiceLivenessDetector
# ---------------------------------------------------------------------------
class TestVoiceLivenessDetector:
    def test_empty_samples_returns_not_live(self) -> None:
        r = VoiceLivenessDetector().detect([])
        assert r.is_live is False
        assert r.confidence == 0.0
        assert r.details["error"] == "No audio samples provided"

    def test_confidence_is_mean_of_three_scores(self) -> None:
        r = VoiceLivenessDetector().detect([make_voice(150.0, 31)], SR)
        expected = (r.replay_score + r.synthetic_score + r.channel_score) / 3.0
        assert r.confidence == pytest.approx(expected)
        assert set(r.details) == {"replay_live", "synthetic_live", "channel_live"}

    def test_broadband_noise_is_classified_live(self) -> None:
        # Broadband noise has high spectral flux + broad spectrum, tripping
        # all three "live" sub-decisions with default thresholds.
        r = VoiceLivenessDetector().detect([make_noise(32)], SR)
        assert bool(r.is_live) is True
        assert r.replay_score > 0.5
        assert r.synthetic_score > 0.5
        assert r.channel_score > 0.5

    def test_short_audio_replay_neutral(self) -> None:
        ld = VoiceLivenessDetector()
        # Too few frames for spectral flux -> neutral 0.5.
        assert ld._detect_replay(make_voice(150.0, 33, dur=0.05), SR) == 0.5

    def test_constant_signal_low_scores(self) -> None:
        ld = VoiceLivenessDetector()
        const = np.ones(SR) * 0.5
        assert ld._detect_replay(const, SR) == pytest.approx(0.2)
        assert ld._detect_synthetic(const, SR) == pytest.approx(0.3)
        assert ld._analyze_channel(const, SR) == pytest.approx(0.3)

    def test_synthetic_empty_quefrency_returns_neutral(self) -> None:
        ld = VoiceLivenessDetector()
        # A 20-sample clip yields a cepstrum too short for the quefrency band.
        assert ld._detect_synthetic(np.random.default_rng(3).standard_normal(20), SR) == 0.5

    def test_narrowband_low_tone_channel_low(self) -> None:
        ld = VoiceLivenessDetector()
        tt = np.arange(SR) / SR
        low_tone = np.sin(2 * np.pi * 120.0 * tt)
        # Energy confined to the low band -> mid_ratio < 0.2 -> floored at 0.3.
        assert ld._analyze_channel(low_tone, SR) == pytest.approx(0.3)

    def test_narrowband_mid_tone_channel_low(self) -> None:
        ld = VoiceLivenessDetector()
        tt = np.arange(SR) / SR
        mid_tone = np.sin(2 * np.pi * 1000.0 * tt)
        # Strong mid band but negligible high band -> high_ratio < 0.05 -> 0.3.
        assert ld._analyze_channel(mid_tone, SR) == pytest.approx(0.3)

    def test_custom_thresholds_make_tonal_live(self) -> None:
        ld = VoiceLivenessDetector(
            replay_threshold=0.1,
            synthetic_threshold=0.1,
            channel_threshold=0.1,
        )
        tt = np.arange(SR) / SR
        sig = np.sin(2 * np.pi * 150.0 * tt) + 0.1 * make_noise(34)
        r = ld.detect([sig], SR)
        assert bool(r.is_live) is True


# ---------------------------------------------------------------------------
# VoiceActivityDetector
# ---------------------------------------------------------------------------
class TestVoiceActivityDetector:
    def test_detect_returns_bool_array(self) -> None:
        vad = VoiceActivityDetector(sample_rate=SR)
        v = vad.detect(make_voice(150.0, 41))
        assert v.dtype == bool
        assert v.ndim == 1
        assert v.any()

    def test_short_audio_returns_single_true(self) -> None:
        vad = VoiceActivityDetector(sample_rate=SR)
        out = vad.detect(np.ones(100))
        assert out.shape == (1,)
        assert bool(out[0]) is True

    def test_adaptive_threshold_with_mixed_silence(self) -> None:
        # Signal that is mostly silent with one central burst exercises the
        # adaptive-noise-floor branch (silence frames > 10%).
        vad = VoiceActivityDetector(sample_rate=SR)
        sig = np.zeros(SR)
        tt = np.arange(2000) / SR
        for k in range(1, 5):
            sig[4000:6000] += np.sin(2 * np.pi * 150.0 * k * tt)
        v = vad.detect(sig)
        active = int(np.sum(v))
        assert 0 < active < v.shape[0]

    def test_speech_segments_full_voice(self) -> None:
        vad = VoiceActivityDetector(sample_rate=SR)
        segs = vad.get_speech_segments(make_voice(150.0, 42))
        assert len(segs) == 1
        start, end = segs[0]
        assert start == 0
        assert end == SR  # trailing in-speech segment ends at len(audio)

    def test_speech_segments_all_silence_empty(self) -> None:
        vad = VoiceActivityDetector(sample_rate=SR)
        assert vad.get_speech_segments(np.zeros(SR)) == []

    def test_speech_segment_burst_then_silence(self) -> None:
        vad = VoiceActivityDetector(sample_rate=SR)
        sig = np.zeros(SR)
        tt = np.arange(2000) / SR
        for k in range(1, 5):
            sig[4000:6000] += np.sin(2 * np.pi * 150.0 * k * tt)
        segs = vad.get_speech_segments(sig, min_duration_ms=50.0)
        assert len(segs) == 1
        start, end = segs[0]
        assert start < end <= SR

    def test_speech_segments_min_duration_filters_out(self) -> None:
        vad = VoiceActivityDetector(sample_rate=SR)
        sig = np.zeros(SR)
        tt = np.arange(2000) / SR
        for k in range(1, 5):
            sig[4000:6000] += np.sin(2 * np.pi * 150.0 * k * tt)
        # An absurd minimum duration filters every candidate segment away.
        assert vad.get_speech_segments(sig, min_duration_ms=100000.0) == []

    def test_trailing_segment_too_short_filtered(self) -> None:
        # Speech that runs right up to the end but is shorter than the minimum
        # duration exercises the trailing "in_speech at EOF but too short" path.
        vad = VoiceActivityDetector(sample_rate=SR)
        sig = np.zeros(SR)
        tail = np.arange(1500) / SR
        for k in range(1, 5):
            sig[SR - 1500 :] += np.sin(2 * np.pi * 150.0 * k * tail)
        assert bool(vad.detect(sig)[-1]) is True
        assert vad.get_speech_segments(sig, min_duration_ms=100000.0) == []


# ---------------------------------------------------------------------------
# VoiceRecognizer
# ---------------------------------------------------------------------------
class TestVoiceRecognizer:
    def test_extract_features_shapes(self) -> None:
        rec = VoiceRecognizer(SR)
        feats = rec.extract_features(make_voice(150.0, 51))
        assert isinstance(feats, VoiceFeatures)
        assert feats.mfcc.shape[1] == 13
        assert feats.delta_mfcc.shape == feats.mfcc.shape
        assert feats.delta2_mfcc.shape == feats.mfcc.shape
        assert feats.embedding.shape == (256,)
        assert feats.pitch_contour.shape[0] == feats.mfcc.shape[0]
        assert feats.energy_contour.shape[0] == feats.mfcc.shape[0]
        assert feats.duration == pytest.approx(1.0)
        assert feats.sample_rate == SR
        assert 0.0 <= feats.quality_score <= 1.0

    def test_quality_score_low_for_mostly_silence(self) -> None:
        rec = VoiceRecognizer(SR)
        sig = np.zeros(SR)
        sig[8000:8100] = 1.0  # negligible speech content
        feats = rec.extract_features(sig)
        assert feats.quality_score == pytest.approx(0.2)

    def test_verify_liveness_required_blocks_non_live(self) -> None:
        rec = VoiceRecognizer(SR, liveness_required=True)
        enrolled = rec.extract_features(make_voice(150.0, 52))
        # A clean tonal probe fails liveness -> match forced to non-match.
        match, liveness = rec.verify(make_voice(150.0, 52), enrolled)
        assert liveness is not None
        assert liveness.is_live is False
        assert match.is_match is False
        assert match.similarity_score == 0.0
        assert match.confidence == 0.0

    def test_verify_extends_liveness_samples(self) -> None:
        rec = VoiceRecognizer(SR, liveness_required=True)
        enrolled = rec.extract_features(make_voice(150.0, 53))
        extra = [make_noise(54)]
        match, liveness = rec.verify(make_voice(150.0, 53), enrolled, extra)
        assert liveness is not None
        assert isinstance(match, VoiceMatchResult)

    def test_verify_live_probe_preserves_match(self) -> None:
        # Broadband noise passes liveness, so a genuine self-match survives the
        # liveness gate rather than being zeroed out.
        rec = VoiceRecognizer(SR, liveness_required=True)
        noise = make_noise(56)
        enrolled = rec.extract_features(noise)
        match, liveness = rec.verify(noise, enrolled)
        assert liveness is not None
        assert bool(liveness.is_live) is True
        assert match.similarity_score == pytest.approx(1.0)
        assert match.is_match is True

    def test_verify_without_liveness_returns_none(self) -> None:
        rec = VoiceRecognizer(SR, liveness_required=False)
        enrolled = rec.extract_features(make_voice(150.0, 55))
        match, liveness = rec.verify(make_voice(150.0, 55), enrolled)
        assert liveness is None
        # Same probe/enrolment -> perfect self-match survives (no liveness gate).
        assert match.is_match is True
        assert match.similarity_score == pytest.approx(1.0)
