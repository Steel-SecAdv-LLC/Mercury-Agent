# Copyright (C) 2025 Steel Security Advisors LLC
"""Voice Recognition Module for Mercury Agent Biometric System.

Implements speaker verification using MFCC features and embedding-based
matching with replay attack detection for liveness.

References:
- Reynolds et al. (2000): Speaker Verification Using Adapted Gaussian Mixture Models
- Variani et al. (2014): Deep Neural Networks for Small Footprint Text-Dependent Speaker Verification
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VoiceFeatures:
    """Extracted voice features."""

    mfcc: np.ndarray[Any, Any]
    delta_mfcc: np.ndarray[Any, Any]
    delta2_mfcc: np.ndarray[Any, Any]
    embedding: np.ndarray[Any, Any]
    pitch_contour: np.ndarray[Any, Any]
    energy_contour: np.ndarray[Any, Any]
    duration: float
    sample_rate: int
    quality_score: float


@dataclass
class VoiceMatchResult:
    """Result of voice matching."""

    similarity_score: float
    is_match: bool
    confidence: float
    embedding_distance: float
    mfcc_distance: float


@dataclass
class VoiceLivenessResult:
    """Result of voice liveness detection."""

    is_live: bool
    confidence: float
    replay_score: float
    synthetic_score: float
    channel_score: float
    details: dict[str, Any] = field(default_factory=dict)


class AudioPreprocessor:
    """Preprocess audio for voice recognition.

    Includes pre-emphasis, framing, windowing, and normalization.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length_ms: float = 25.0,
        frame_shift_ms: float = 10.0,
        pre_emphasis: float = 0.97,
    ) -> None:
        """Initialize the preprocessor."""
        self._sample_rate = sample_rate
        self._frame_length = int(sample_rate * frame_length_ms / 1000)
        self._frame_shift = int(sample_rate * frame_shift_ms / 1000)
        self._pre_emphasis = pre_emphasis

    def preprocess(self, signal: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess audio signal.

        Args:
            signal: Raw audio signal

        Returns:
            Preprocessed signal
        """
        signal = signal.astype(np.float64)

        if len(signal.shape) > 1:
            signal = np.mean(signal, axis=1)

        signal = signal / (np.max(np.abs(signal)) + 1e-8)

        emphasized = np.zeros_like(signal)
        emphasized[0] = signal[0]
        emphasized[1:] = signal[1:] - self._pre_emphasis * signal[:-1]

        return emphasized

    def frame(self, signal: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Divide signal into overlapping frames.

        Args:
            signal: Preprocessed signal

        Returns:
            2D array of frames (num_frames, frame_length)
        """
        n_samples = len(signal)
        n_frames = 1 + (n_samples - self._frame_length) // self._frame_shift

        n_frames = max(n_frames, 1)

        frames = np.zeros((n_frames, self._frame_length))

        for i in range(n_frames):
            start = i * self._frame_shift
            end = start + self._frame_length

            if end <= n_samples:
                frames[i] = signal[start:end]
            else:
                frames[i, : n_samples - start] = signal[start:]

        window = np.hamming(self._frame_length)
        frames = frames * window

        return frames

    @property
    def frame_length(self) -> int:
        """Get frame length in samples."""
        return self._frame_length

    @property
    def frame_shift(self) -> int:
        """Get frame shift in samples."""
        return self._frame_shift


class MFCCExtractor:
    """Extract Mel-Frequency Cepstral Coefficients.

    Implements the standard MFCC pipeline: FFT -> Mel filterbank -> DCT.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_fft: int = 512,
        n_mels: int = 40,
        fmin: float = 0.0,
        fmax: float | None = None,
    ) -> None:
        """Initialize the MFCC extractor."""
        self._sample_rate = sample_rate
        self._n_mfcc = n_mfcc
        self._n_fft = n_fft
        self._n_mels = n_mels
        self._fmin = fmin
        self._fmax = fmax or sample_rate / 2

        self._mel_filterbank = self._create_mel_filterbank()
        self._dct_matrix = self._create_dct_matrix()

    def _create_mel_filterbank(self) -> np.ndarray[Any, Any]:
        """Create Mel filterbank matrix."""
        mel_min = self._hz_to_mel(self._fmin)
        mel_max = self._hz_to_mel(self._fmax)
        mel_points = np.linspace(mel_min, mel_max, self._n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)

        bin_points = np.floor((self._n_fft + 1) * hz_points / self._sample_rate).astype(int)

        filterbank = np.zeros((self._n_mels, self._n_fft // 2 + 1))

        for i in range(self._n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            for j in range(left, center):
                if center > left:
                    filterbank[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right > center:
                    filterbank[i, j] = (right - j) / (right - center)

        return filterbank

    def _create_dct_matrix(self) -> np.ndarray[Any, Any]:
        """Create DCT matrix for MFCC computation."""
        n = self._n_mels
        k = self._n_mfcc
        dct_matrix = np.zeros((k, n))

        for i in range(k):
            for j in range(n):
                dct_matrix[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * n))

        dct_matrix[0, :] *= 1 / np.sqrt(2)
        dct_matrix *= np.sqrt(2 / n)

        return dct_matrix

    def _hz_to_mel(self, hz: float | np.ndarray[Any, Any]) -> float | np.ndarray[Any, Any]:
        """Convert frequency in Hz to Mel scale."""
        return 2595 * np.log10(1 + hz / 700)

    def _mel_to_hz(self, mel: float | np.ndarray[Any, Any]) -> float | np.ndarray[Any, Any]:
        """Convert Mel scale to frequency in Hz."""
        return 700 * (10 ** (mel / 2595) - 1)

    def extract(self, frames: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Extract MFCC features from frames.

        Args:
            frames: 2D array of frames

        Returns:
            MFCC features (n_frames, n_mfcc)
        """
        n_frames = frames.shape[0]
        mfcc = np.zeros((n_frames, self._n_mfcc))

        for i in range(n_frames):
            frame = frames[i]

            padded = np.zeros(self._n_fft)
            padded[: len(frame)] = frame
            spectrum = np.abs(np.fft.rfft(padded))

            power_spectrum = spectrum**2

            mel_spectrum = np.dot(self._mel_filterbank, power_spectrum)
            mel_spectrum = np.maximum(mel_spectrum, 1e-10)
            log_mel = np.log(mel_spectrum)

            mfcc[i] = np.dot(self._dct_matrix, log_mel)

        return mfcc

    def compute_deltas(
        self,
        features: np.ndarray[Any, Any],
        n: int = 2,
    ) -> np.ndarray[Any, Any]:
        """Compute delta (derivative) features.

        Args:
            features: Input features
            n: Number of frames to consider on each side

        Returns:
            Delta features
        """
        n_frames, n_features = features.shape
        deltas = np.zeros_like(features)

        denominator = 2 * sum(i**2 for i in range(1, n + 1))

        for t in range(n_frames):
            numerator = np.zeros(n_features)

            for i in range(1, n + 1):
                t_plus = min(t + i, n_frames - 1)
                t_minus = max(t - i, 0)
                numerator += i * (features[t_plus] - features[t_minus])

            deltas[t] = numerator / denominator

        return deltas


class PitchExtractor:
    """Extract pitch (fundamental frequency) contour.

    Uses autocorrelation method for pitch detection.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        min_pitch: float = 50.0,
        max_pitch: float = 500.0,
    ) -> None:
        """Initialize the pitch extractor."""
        self._sample_rate = sample_rate
        self._min_lag = int(sample_rate / max_pitch)
        self._max_lag = int(sample_rate / min_pitch)

    def extract(self, frames: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Extract pitch contour from frames.

        Args:
            frames: 2D array of frames

        Returns:
            Pitch values in Hz (0 for unvoiced frames)
        """
        n_frames = frames.shape[0]
        pitch = np.zeros(n_frames)

        for i in range(n_frames):
            frame = frames[i]
            pitch[i] = self._estimate_pitch(frame)

        return pitch

    def _estimate_pitch(self, frame: np.ndarray[Any, Any]) -> float:
        """Estimate pitch for a single frame using autocorrelation."""
        n = len(frame)
        autocorr = np.correlate(frame, frame, mode="full")
        autocorr = autocorr[n - 1 :]

        valid_range = autocorr[self._min_lag : self._max_lag]
        if len(valid_range) == 0:
            return 0.0

        peak_idx = np.argmax(valid_range) + self._min_lag

        if autocorr[peak_idx] < 0.3 * autocorr[0]:
            return 0.0

        pitch = self._sample_rate / peak_idx
        return float(pitch)


class EnergyExtractor:
    """Extract energy contour from audio frames."""

    def __init__(self, floor: float = 1e-10) -> None:
        """Initialize the energy extractor."""
        self._floor = floor

    def extract(self, frames: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Extract log energy from frames.

        Args:
            frames: 2D array of frames

        Returns:
            Log energy values
        """
        energy = np.sum(frames**2, axis=1)
        energy = np.maximum(energy, self._floor)
        log_energy = np.log(energy)

        return log_energy


class SpeakerEmbedding:
    """Generate speaker embeddings from acoustic features.

    Uses a simple neural network-inspired approach with statistics pooling.
    """

    def __init__(
        self,
        input_dim: int = 39,
        embedding_dim: int = 256,
        seed: int | None = 42,
    ) -> None:
        """Initialize the embedding generator.

        Args:
            input_dim: Input feature dimension.
            embedding_dim: Output embedding dimension.
            seed: Optional seed for the per-instance numpy ``Generator``
                used to initialise the demo embedding weights.  Default
                ``42`` preserves the historical deterministic behaviour
                without polluting the global ``np.random`` state.  Pass
                ``None`` to draw fresh weights from the OS entropy pool.
        """
        self._input_dim = input_dim
        self._embedding_dim = embedding_dim

        rng = np.random.default_rng(seed)
        self._weights1 = rng.standard_normal((input_dim, 128)) * 0.1
        self._weights2 = rng.standard_normal((128, 64)) * 0.1
        self._weights3 = rng.standard_normal((128, embedding_dim)) * 0.1

    def generate(self, features: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Generate speaker embedding from features.

        Args:
            features: Frame-level features (n_frames, n_features)

        Returns:
            Fixed-length speaker embedding
        """
        h1 = np.maximum(0, np.dot(features, self._weights1))

        h2 = np.maximum(0, np.dot(h1, self._weights2))

        mean_pool = np.mean(h2, axis=0)
        std_pool = np.std(h2, axis=0)
        pooled = np.concatenate([mean_pool, std_pool])

        embedding = np.dot(pooled, self._weights3)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding


class VoiceMatcher:
    """Match voice samples using embedding similarity."""

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        use_plda: bool = False,
    ) -> None:
        """Initialize the matcher."""
        self._threshold = similarity_threshold
        self._use_plda = use_plda

    def match(
        self,
        probe: VoiceFeatures,
        enrolled: VoiceFeatures,
    ) -> VoiceMatchResult:
        """Match probe against enrolled voice features.

        Args:
            probe: Probe voice features
            enrolled: Enrolled voice features

        Returns:
            Match result with similarity score
        """
        embedding_sim = self._cosine_similarity(probe.embedding, enrolled.embedding)

        probe_combined = np.concatenate(
            [
                probe.mfcc.flatten()[:1000],
                probe.delta_mfcc.flatten()[:1000],
            ]
        )
        enrolled_combined = np.concatenate(
            [
                enrolled.mfcc.flatten()[:1000],
                enrolled.delta_mfcc.flatten()[:1000],
            ]
        )

        probe_combined = probe_combined / (np.linalg.norm(probe_combined) + 1e-8)
        enrolled_combined = enrolled_combined / (np.linalg.norm(enrolled_combined) + 1e-8)

        mfcc_sim = self._cosine_similarity(probe_combined, enrolled_combined)

        similarity = 0.7 * embedding_sim + 0.3 * mfcc_sim

        is_match = similarity >= self._threshold
        confidence = min(1.0, similarity / self._threshold) if is_match else similarity * 0.5

        return VoiceMatchResult(
            similarity_score=similarity,
            is_match=is_match,
            confidence=confidence,
            embedding_distance=1 - embedding_sim,
            mfcc_distance=1 - mfcc_sim,
        )

    def _cosine_similarity(self, a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))


class VoiceLivenessDetector:
    """Detect voice presentation attacks.

    Analyzes replay artifacts, synthetic speech markers, and channel characteristics.
    """

    def __init__(
        self,
        replay_threshold: float = 0.5,
        synthetic_threshold: float = 0.5,
        channel_threshold: float = 0.5,
    ) -> None:
        """Initialize the liveness detector."""
        self._replay_threshold = replay_threshold
        self._synthetic_threshold = synthetic_threshold
        self._channel_threshold = channel_threshold

    def detect(
        self,
        audio_samples: list[np.ndarray[Any, Any]],
        sample_rate: int = 16000,
    ) -> VoiceLivenessResult:
        """Detect voice liveness from audio samples.

        Args:
            audio_samples: List of audio samples
            sample_rate: Sample rate of audio

        Returns:
            Liveness result with confidence scores
        """
        if len(audio_samples) < 1:
            return VoiceLivenessResult(
                is_live=False,
                confidence=0.0,
                replay_score=0.0,
                synthetic_score=0.0,
                channel_score=0.0,
                details={"error": "No audio samples provided"},
            )

        replay_score = self._detect_replay(audio_samples[0], sample_rate)
        synthetic_score = self._detect_synthetic(audio_samples[0], sample_rate)
        channel_score = self._analyze_channel(audio_samples[0], sample_rate)

        replay_live = replay_score > self._replay_threshold
        synthetic_live = synthetic_score > self._synthetic_threshold
        channel_live = channel_score > self._channel_threshold

        is_live = replay_live and synthetic_live and channel_live
        confidence = (replay_score + synthetic_score + channel_score) / 3.0

        return VoiceLivenessResult(
            is_live=is_live,
            confidence=confidence,
            replay_score=replay_score,
            synthetic_score=synthetic_score,
            channel_score=channel_score,
            details={
                "replay_live": replay_live,
                "synthetic_live": synthetic_live,
                "channel_live": channel_live,
            },
        )

    def _detect_replay(self, audio: np.ndarray[Any, Any], sample_rate: int) -> float:
        """Detect replay attack artifacts."""
        audio = audio.astype(np.float64)
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        n_fft = 2048
        hop_length = 512
        n_frames = (len(audio) - n_fft) // hop_length + 1

        if n_frames < 2:
            return 0.5

        spectral_flux = 0.0
        prev_spectrum = None

        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start : start + n_fft]

            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))

            spectrum = np.abs(np.fft.rfft(frame))

            if prev_spectrum is not None:
                diff = np.maximum(0, spectrum - prev_spectrum)
                spectral_flux += np.sum(diff)

            prev_spectrum = spectrum

        avg_flux = spectral_flux / max(1, n_frames - 1)

        if avg_flux < 0.001:
            return 0.2

        score = min(1.0, avg_flux / 1.0)
        return score

    def _detect_synthetic(self, audio: np.ndarray[Any, Any], sample_rate: int) -> float:
        """Detect synthetic speech artifacts."""
        audio = audio.astype(np.float64)
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        n_fft = 1024
        spectrum = np.abs(np.fft.rfft(audio[: n_fft * 10]))

        log_spectrum = np.log(np.maximum(spectrum, 1e-10))
        cepstrum = np.fft.irfft(log_spectrum)

        quefrency_range = cepstrum[sample_rate // 500 : sample_rate // 50]
        if len(quefrency_range) == 0:
            return 0.5

        pitch_peak = np.max(np.abs(quefrency_range))
        avg_level = np.mean(np.abs(cepstrum[len(cepstrum) // 4 :]))

        if pitch_peak < avg_level * 2:
            return 0.3

        freq_bins = len(spectrum)
        high_freq = spectrum[freq_bins // 2 :]
        low_freq = spectrum[: freq_bins // 2]

        high_energy = np.sum(high_freq**2)
        low_energy = np.sum(low_freq**2) + 1e-10

        ratio = high_energy / low_energy
        expected_ratio = 0.1

        if ratio < expected_ratio / 10:
            return 0.3

        return float(min(1.0, 0.5 + pitch_peak / (avg_level * 4)))

    def _analyze_channel(self, audio: np.ndarray[Any, Any], sample_rate: int) -> float:
        """Analyze recording channel characteristics."""
        audio = audio.astype(np.float64)
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        n_fft = 2048
        spectrum = np.abs(np.fft.rfft(audio[: n_fft * 20]))

        freq_bins = len(spectrum)
        bin_hz = sample_rate / 2 / freq_bins

        low_band = spectrum[int(100 / bin_hz) : int(500 / bin_hz)]
        mid_band = spectrum[int(500 / bin_hz) : int(2000 / bin_hz)]
        high_band = spectrum[int(2000 / bin_hz) : int(8000 / bin_hz)]

        low_energy = np.mean(low_band**2) if len(low_band) > 0 else 0
        mid_energy = np.mean(mid_band**2) if len(mid_band) > 0 else 0
        high_energy = np.mean(high_band**2) if len(high_band) > 0 else 0

        total_energy = low_energy + mid_energy + high_energy + 1e-10

        low_energy / total_energy
        mid_ratio = mid_energy / total_energy
        high_ratio = high_energy / total_energy

        if mid_ratio < 0.2:
            return 0.3

        if high_ratio < 0.05:
            return 0.3

        return min(1.0, 0.4 + mid_ratio + high_ratio * 0.5)


class VoiceActivityDetector:
    """Detect voice activity in audio signal.

    Uses energy-based detection with adaptive thresholding.
    """

    def __init__(
        self,
        frame_length_ms: float = 25.0,
        frame_shift_ms: float = 10.0,
        sample_rate: int = 16000,
        energy_threshold: float = 0.1,
    ) -> None:
        """Initialize the VAD."""
        self._frame_length = int(sample_rate * frame_length_ms / 1000)
        self._frame_shift = int(sample_rate * frame_shift_ms / 1000)
        self._sample_rate = sample_rate
        self._energy_threshold = energy_threshold

    def detect(self, audio: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Detect voice activity in audio.

        Args:
            audio: Audio signal

        Returns:
            Boolean array indicating voice activity per frame
        """
        audio = audio.astype(np.float64)
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        n_frames = (len(audio) - self._frame_length) // self._frame_shift + 1
        if n_frames < 1:
            return np.array([True])

        energy = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * self._frame_shift
            frame = audio[start : start + self._frame_length]
            energy[i] = np.sum(frame**2)

        normalized_energy: np.ndarray[Any, Any] = np.empty(energy.shape, dtype=np.float64)
        np.divide(energy, np.max(energy) + 1e-8, out=normalized_energy)
        energy = normalized_energy

        threshold = self._energy_threshold

        silence_frames = energy < threshold * 0.5
        if np.sum(silence_frames) > n_frames * 0.1:
            noise_level = np.mean(energy[silence_frames])
            threshold = max(self._energy_threshold, float(noise_level * 3))  # type: ignore[assignment, unused-ignore]

        vad = energy > threshold

        return vad

    def get_speech_segments(
        self,
        audio: np.ndarray[Any, Any],
        min_duration_ms: float = 100.0,
    ) -> list[tuple[int, int]]:
        """Get speech segments from audio.

        Args:
            audio: Audio signal
            min_duration_ms: Minimum segment duration

        Returns:
            List of (start_sample, end_sample) tuples
        """
        vad = self.detect(audio)
        min_frames = int(min_duration_ms / (self._frame_shift * 1000 / self._sample_rate))

        segments = []
        in_speech = False
        start_frame = 0

        for i, is_speech in enumerate(vad):
            if is_speech and not in_speech:
                start_frame = i
                in_speech = True
            elif not is_speech and in_speech:
                if i - start_frame >= min_frames:
                    start_sample = start_frame * self._frame_shift
                    end_sample = i * self._frame_shift
                    segments.append((start_sample, end_sample))
                in_speech = False

        if in_speech:
            if len(vad) - start_frame >= min_frames:
                start_sample = start_frame * self._frame_shift
                end_sample = len(audio)
                segments.append((start_sample, end_sample))

        return segments


class VoiceRecognizer:
    """Complete voice recognition system.

    Integrates feature extraction, embedding generation, matching, and liveness detection.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        match_threshold: float = 0.7,
        liveness_required: bool = True,
    ) -> None:
        """Initialize the voice recognizer."""
        self._sample_rate = sample_rate
        self._preprocessor = AudioPreprocessor(sample_rate)
        self._mfcc_extractor = MFCCExtractor(sample_rate)
        self._pitch_extractor = PitchExtractor(sample_rate)
        self._energy_extractor = EnergyExtractor()
        self._embedding_generator = SpeakerEmbedding()
        self._matcher = VoiceMatcher(match_threshold)
        self._liveness_detector = VoiceLivenessDetector()
        self._vad = VoiceActivityDetector(sample_rate=sample_rate)
        self._liveness_required = liveness_required

    def extract_features(self, audio: np.ndarray[Any, Any]) -> VoiceFeatures:
        """Extract voice features from audio.

        Args:
            audio: Audio signal

        Returns:
            VoiceFeatures containing MFCC, embeddings, and metadata
        """
        preprocessed = self._preprocessor.preprocess(audio)
        frames = self._preprocessor.frame(preprocessed)

        mfcc = self._mfcc_extractor.extract(frames)
        delta_mfcc = self._mfcc_extractor.compute_deltas(mfcc)
        delta2_mfcc = self._mfcc_extractor.compute_deltas(delta_mfcc)

        combined_features = np.concatenate([mfcc, delta_mfcc, delta2_mfcc], axis=1)
        embedding = self._embedding_generator.generate(combined_features)

        pitch = self._pitch_extractor.extract(frames)
        energy = self._energy_extractor.extract(frames)

        duration = len(audio) / self._sample_rate
        quality_score = self._compute_quality(audio, mfcc, pitch)

        return VoiceFeatures(
            mfcc=mfcc,
            delta_mfcc=delta_mfcc,
            delta2_mfcc=delta2_mfcc,
            embedding=embedding,
            pitch_contour=pitch,
            energy_contour=energy,
            duration=duration,
            sample_rate=self._sample_rate,
            quality_score=quality_score,
        )

    def verify(
        self,
        probe_audio: np.ndarray[Any, Any],
        enrolled_features: VoiceFeatures,
        liveness_samples: list[np.ndarray[Any, Any]] | None = None,
    ) -> tuple[VoiceMatchResult, VoiceLivenessResult | None]:
        """Verify a voice sample against enrolled features.

        Args:
            probe_audio: Probe audio signal
            enrolled_features: Enrolled voice features
            liveness_samples: Additional samples for liveness detection

        Returns:
            Tuple of (match_result, liveness_result)
        """
        probe_features = self.extract_features(probe_audio)
        match_result = self._matcher.match(probe_features, enrolled_features)

        liveness_result = None
        if self._liveness_required:
            samples = [probe_audio]
            if liveness_samples:
                samples.extend(liveness_samples)

            liveness_result = self._liveness_detector.detect(samples, self._sample_rate)

            if not liveness_result.is_live:
                match_result = VoiceMatchResult(
                    similarity_score=0.0,
                    is_match=False,
                    confidence=0.0,
                    embedding_distance=match_result.embedding_distance,
                    mfcc_distance=match_result.mfcc_distance,
                )

        return match_result, liveness_result

    def _compute_quality(
        self,
        audio: np.ndarray[Any, Any],
        mfcc: np.ndarray[Any, Any],
        pitch: np.ndarray[Any, Any],
    ) -> float:
        """Compute voice sample quality score."""
        vad = self._vad.detect(audio)
        speech_ratio = np.mean(vad)

        if speech_ratio < 0.1:
            return 0.2

        mfcc_var = np.mean(np.var(mfcc, axis=0))
        mfcc_quality = min(1.0, mfcc_var / 10.0)

        voiced_frames = pitch > 0
        pitch_quality = np.mean(voiced_frames)

        quality = 0.4 * speech_ratio + 0.3 * mfcc_quality + 0.3 * pitch_quality

        return float(quality)
