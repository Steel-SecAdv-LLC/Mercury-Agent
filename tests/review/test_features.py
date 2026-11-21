# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Segments x features: an artifact measured against itself.

Pins what :mod:`omni_mercury_engine.review.features` promises the scoring
path: sixteen named, finite, clipped columns per segment for every kind; a
robust standardization that a minority of unusual segments cannot move; a
refusal (422) when an artifact is too small to be its own reference; and that
the part of an artifact that is unlike the rest is the part that stands out.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.review import features
from omni_mercury_engine.review.decode import ReviewError, decode
from omni_mercury_engine.review.features import (
    AUDIO_FEATURES,
    CLIP,
    DOCUMENT_FEATURES,
    IMAGE_FEATURES,
    MIN_SEGMENTS,
    TARGET_SEGMENTS,
    WIDTH,
    extract,
    standardize,
)
from tests.review import builders


def _document() -> bytes:
    return builders.prose(1600).encode()


def _image() -> bytes:
    return builders.png(builders.texture(256, 256))


def _audio() -> bytes:
    return builders.wav(builders.tone(8.0))


class TestTheMatrix:
    def test_every_kind_has_sixteen_distinct_named_columns(self) -> None:
        for names in (DOCUMENT_FEATURES, IMAGE_FEATURES, AUDIO_FEATURES):
            assert len(names) == len(set(names)) == WIDTH == 16

    @pytest.mark.parametrize(
        ("build", "kind", "names"),
        [
            (_document, "document", DOCUMENT_FEATURES),
            (_image, "image", IMAGE_FEATURES),
            (_audio, "audio", AUDIO_FEATURES),
        ],
    )
    def test_an_artifact_becomes_segments_by_sixteen(
        self, build: object, kind: str, names: tuple[str, ...]
    ) -> None:
        segmented = extract(decode(build()))  # type: ignore[operator]
        rows = len(segmented.segments)
        assert segmented.kind == kind and segmented.names == names
        assert segmented.raw.shape == segmented.matrix.shape == (rows, WIDTH)
        assert MIN_SEGMENTS <= rows <= TARGET_SEGMENTS[kind] * 1.5
        assert np.isfinite(segmented.matrix).all()
        assert float(np.abs(segmented.matrix).max()) <= CLIP
        assert [segment["index"] for segment in segmented.segments] == list(range(rows))

    def test_each_segment_says_where_it_is(self) -> None:
        text = builders.prose(1600)
        document = extract(decode(text.encode()))
        first, last = document.segments[3]["characters"]
        assert " ".join(text[first:last].split()).startswith(document.segments[3]["excerpt"][:40])
        image = extract(decode(_image()))
        x, y, width, height = image.segments[-1]["box"]
        assert x + width <= 256 and y + height <= 256
        audio = extract(decode(_audio()))
        assert audio.segments[-1]["seconds"][1] <= 8.0
        assert audio.detail["analysed_at_hz"] == 16_000

    def test_a_large_images_segments_are_located_in_its_own_pixels(self) -> None:
        """Measured reduced, reported at full size: a box can be drawn on the upload itself."""
        pixels = builders.texture(1200, 1800)
        rng = np.random.default_rng(5)
        pixels[400:700, 900:1300] = rng.integers(0, 256, size=(300, 400), dtype=np.uint8)
        segmented = extract(decode(builders.png(pixels)))
        assert (segmented.detail["width"], segmented.detail["height"]) == (1800, 1200)
        x, y, width, height = segmented.segments[-1]["box"]
        assert 1700 < x + width <= 1800 and 1100 < y + height <= 1200
        size = np.abs(segmented.matrix).mean(axis=1)
        x, y, width, height = segmented.segments[int(np.argmax(size))]["box"]
        assert 900 - width < x < 1300 and 400 - height < y < 700

    def test_the_segment_count_does_not_grow_with_the_artifact(self) -> None:
        """The verdict is about the most anomalous segment: its odds must not scale with size."""
        short = extract(decode(builders.prose(1600).encode()))
        long = extract(decode(builders.prose(16_000).encode()))
        assert len(short.segments) == len(long.segments) == TARGET_SEGMENTS["document"]
        assert long.detail["words_per_segment"] > short.detail["words_per_segment"]


class TestStandardize:
    def test_it_is_a_robust_z_score_against_the_artifacts_own_rows(self) -> None:
        rng = np.random.default_rng(3)
        raw = rng.normal(loc=50.0, scale=4.0, size=(60, 2))
        z = standardize(raw)
        median = np.median(raw, axis=0)
        mad = 1.4826 * np.median(np.abs(raw - median), axis=0)
        assert np.allclose(z, np.clip((raw - median) / mad, -CLIP, CLIP))

    def test_a_minority_of_unusual_rows_does_not_move_the_reference(self) -> None:
        rng = np.random.default_rng(4)
        raw = rng.normal(size=(60, 1))
        spiked = raw.copy()
        spiked[:6] += 500.0
        clean, shifted = standardize(raw)[6:], standardize(spiked)[6:]
        # A tenth of the rows moved by 500 deviations; no other row moves by as much as one.
        assert float(np.abs(clean - shifted).max()) < 1.0
        assert (standardize(spiked)[:6] == CLIP).all()
        # The mean and standard deviation, by contrast, would flatten them to one value.
        ordinary = (spiked - spiked.mean()) / spiked.std()
        assert float(np.ptp(ordinary[6:])) < 0.05 < float(np.ptp(shifted))

    def test_a_column_with_no_spread_is_zero_and_nothing_is_ever_non_finite(self) -> None:
        raw = np.column_stack(
            [np.full(20, 7.0), np.r_[np.zeros(19), 1.0], np.r_[np.nan, np.inf, np.zeros(18)]]
        )
        z = standardize(raw)
        assert np.isfinite(z).all()
        assert not z[:, 0].any()
        # Most rows identical (zero MAD): the spread falls back to the standard deviation.
        assert z[-1, 1] > 3.0 and not z[:-1, 1].any()


class TestTooSmallToBeItsOwnReference:
    @pytest.mark.parametrize(
        "data",
        [
            builders.prose(120).encode(),
            builders.png(builders.texture(40, 40)),
            builders.wav(builders.tone(1.0)),
        ],
        ids=["document", "image", "audio"],
    )
    def test_it_is_refused_with_the_reason(self, data: bytes) -> None:
        with pytest.raises(ReviewError) as caught:
            extract(decode(data))
        assert caught.value.status == 422


class TestThePartUnlikeTheRestStandsOut:
    """Self-reference at work: the foreign part carries the largest deviations."""

    @staticmethod
    def _deviation(matrix: np.ndarray[Any, Any], rows: list[int]) -> tuple[float, float]:
        size = np.abs(matrix).mean(axis=1)
        rest = np.delete(size, rows)
        return float(size[rows].mean()), float(rest.mean())

    def test_a_block_of_figures_inside_prose(self) -> None:
        words = builders.prose(1600).split()
        text = " ".join(words[:800] + builders.ledger(120).split() + words[800:])
        segmented = extract(decode(text.encode()))
        foreign = [
            segment["index"]
            for segment in segmented.segments
            if "REF-" in segment["excerpt"] and segment["excerpt"].count("REF-") >= 3
        ]
        assert foreign
        inside, outside = self._deviation(segmented.matrix, foreign)
        assert inside > 3 * outside

    def test_a_patch_of_another_texture_inside_an_image(self) -> None:
        pixels = builders.texture(256, 256)
        rng = np.random.default_rng(9)
        pixels[96:160, 96:160] = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        segmented = extract(decode(builders.png(pixels)))
        foreign = [
            segment["index"]
            for segment in segmented.segments
            if 96 <= segment["box"][0] < 160 - segment["box"][2] // 2
            and 96 <= segment["box"][1] < 160 - segment["box"][3] // 2
        ]
        assert foreign
        inside, outside = self._deviation(segmented.matrix, foreign)
        assert inside > 3 * outside

    def test_a_stretch_of_noise_inside_a_recording(self) -> None:
        signal = builders.tone(10.0)
        rng = np.random.default_rng(11)
        signal[64_000:80_000] = rng.normal(scale=0.2, size=16_000).clip(-1, 1)
        segmented = extract(decode(builders.wav(signal)))
        foreign = [
            segment["index"]
            for segment in segmented.segments
            if segment["seconds"][0] >= 4.0 and segment["seconds"][1] <= 5.0
        ]
        assert foreign
        inside, outside = self._deviation(segmented.matrix, foreign)
        assert inside > 3 * outside


class TestAudioIsMeasuredInBoundedBatches:
    def test_batching_the_frames_changes_no_number(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A long recording's peak memory is one batch of frames; the features are the same."""
        decoded = decode(_audio())
        whole = extract(decoded)
        monkeypatch.setattr(features, "_FRAME_BATCH", 37)
        batched = extract(decoded)
        assert np.array_equal(whole.raw, batched.raw)
        assert np.array_equal(whole.matrix, batched.matrix)

    def test_an_unusual_sample_rate_cannot_buy_a_million_tap_filter(self) -> None:
        """Every rate a WAV may declare resamples by a ratio with terms of at most 1,000.

        The anti-aliasing filter's length follows the larger term. Left exact,
        a rate one hertz off a round number needed millions of taps (measured:
        1.7 GB for one upload); approximated, the duration is off by under 0.1%.
        """
        from fractions import Fraction

        worst = 0.0
        for rate in (4_000, 4_001, 11_025, 44_099, 44_100, 48_000, 96_001, 176_089, 191_999):
            ratio = Fraction(16_000, rate).limit_denominator(features._MAX_RESAMPLE_TERM)
            if ratio.numerator > features._MAX_RESAMPLE_TERM:
                ratio = 1 / Fraction(rate, 16_000).limit_denominator(features._MAX_RESAMPLE_TERM)
            assert max(ratio.numerator, ratio.denominator) <= features._MAX_RESAMPLE_TERM
            worst = max(worst, abs(float(ratio) * rate / 16_000 - 1))
        assert worst < 1e-3
        segmented = extract(decode(builders.wav(builders.tone(6.0, rate=44_099), rate=44_099)))
        assert abs(segmented.detail["seconds"] - 6.0) < 0.01

    def test_other_sample_rates_are_analysed_at_sixteen_kilohertz(self) -> None:
        segmented = extract(decode(builders.wav(builders.tone(8.0, rate=44_100), rate=44_100)))
        assert segmented.detail["sample_rate"] == 44_100
        assert segmented.detail["analysed_at_hz"] == 16_000
        assert abs(segmented.detail["seconds"] - 8.0) < 0.01
