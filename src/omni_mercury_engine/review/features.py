# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reduce a decoded artifact to segments x features, measured against itself.

An uploaded document, image or recording arrives with no population to compare
it to.  What it does carry is its own bulk: most of a document is written one
way, most of an image has one sensor's grain, most of a recording one room's
noise floor.  So the artifact is cut into segments, each segment is measured
by a small set of named features, and every feature is standardized against
the artifact's own median and spread.  A segment then reads as "how far from
the rest of this artifact", in robust standard deviations, whatever the
artifact is.  That is the matrix the fusion engine scores.

The scope follows from the construction and is stated wherever a result is
shown: this finds a part that is unlike the rest of the same artifact.  It
cannot say an artifact is unusual as a whole, because the whole is the
reference.

Feature sets (sixteen columns each, all computed here or by instruments this
package already had; no learned embedding, no external detector):

* **document** -- word-window statistics: word length (mean, spread), lexical
  variety, short and long words, capitals, digits, punctuation, line breaks,
  non-ASCII, character entropy, repetition, sentence length, numerals, and how
  surprising the window's letter pairs are given the rest of the same document
  (each window scored against counts that leave it out);
* **image** -- per-tile statistics from :mod:`omni_mercury_engine.vision.features`
  (intensity, gradient, orientation, local texture, frequency bands, noise
  residual, edges, lattice structure, colour);
* **audio** -- per-block statistics over the frames of
  :mod:`omni_mercury_engine.biometric.voice_recognition` (twelve cepstral
  coefficients, log energy and its spread) plus zero-crossing rate and
  spectral flatness.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np

from omni_mercury_engine.review.decode import (
    AUDIO,
    DOCUMENT,
    IMAGE,
    IMAGE_WORKING_SIDE,
    Decoded,
    ReviewError,
)

__all__ = [
    "AUDIO_FEATURES",
    "CLIP",
    "DOCUMENT_FEATURES",
    "IMAGE_FEATURES",
    "MIN_SEGMENTS",
    "TARGET_SEGMENTS",
    "WIDTH",
    "Segmented",
    "extract",
    "standardize",
]

#: Columns of every feature matrix.
WIDTH = 16
#: Fewest segments an artifact must yield: below this its own median and
#: spread are not a reference worth standardizing against.
MIN_SEGMENTS = 12
#: Segments an artifact is cut into when it is large enough.  Held (nearly)
#: constant per kind on purpose: the verdict is about the most anomalous
#: segment, and the maximum of 96 rows is not comparable with the maximum of 24.
TARGET_SEGMENTS = {"document": 48, "image": 96, "audio": 48}
#: Standardized values are clipped to this many robust deviations.
CLIP = 10.0

DOCUMENT_FEATURES = (
    "word_length_mean", "word_length_spread", "lexical_variety", "short_word_share",
    "long_word_share", "capital_letter_share", "digit_share", "punctuation_share",
    "line_break_density", "non_ascii_share", "character_entropy", "bigram_repetition",
    "sentence_length", "capitalized_word_share", "numeric_token_share", "letter_pair_surprise",
)  # fmt: skip
IMAGE_FEATURES = (
    "intensity_mean", "intensity_spread", "gradient_magnitude", "orientation_peak",
    "orientation_entropy", "texture_peak", "texture_entropy", "frequency_fine",
    "frequency_mid", "frequency_coarse", "noise_residual", "edge_density",
    "lattice_any_axis", "lattice_both_axes", "colour_saturation", "colour_opponency",
)  # fmt: skip
AUDIO_FEATURES = (
    *(f"cepstral_{index}" for index in range(1, 13)),
    "log_energy_mean", "log_energy_spread", "zero_crossing_rate", "spectral_flatness",
)  # fmt: skip

_AUDIO_RATE = 16_000
#: Frames measured at a time, so a long recording's peak memory is this many
#: frames' worth and not the whole recording's.
_FRAME_BATCH = 8192
#: Largest term of the resampling ratio.  The usual rates reduce to small
#: ratios exactly (44.1 kHz is 160/441); an unusual one is approximated (to
#: within 0.06% over every rate a WAV may declare here), so the anti-aliasing
#: filter, whose length follows the larger term, stays a few thousand taps
#: instead of millions: measured, a 768 kHz rate one hertz off a round number
#: cost 1.7 GB in the filter alone.
_MAX_RESAMPLE_TERM = 1000


@dataclass(frozen=True)
class Segmented:
    """An artifact as segments x features.

    Attributes:
        kind: ``document``, ``image`` or ``audio``.
        names: The sixteen feature names, in column order.
        raw: Features as measured, ``(segments, 16)``.
        matrix: The same, standardized against the artifact itself and clipped
            (see :func:`standardize`).  This is what is scored.
        segments: One descriptor per row saying where the segment is (character
            span and an excerpt; tile box; time span).
        detail: How the artifact was segmented.
    """

    kind: str
    names: tuple[str, ...]
    raw: np.ndarray[Any, Any]
    matrix: np.ndarray[Any, Any]
    segments: tuple[dict[str, Any], ...]
    detail: dict[str, Any] = field(default_factory=dict)


def standardize(raw: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Robust z-scores of each column against the artifact's own rows.

    ``(x - median) / (1.4826 * MAD)``, so a minority of unusual segments does
    not move the reference they are measured against.  A column whose MAD is
    zero (most segments identical) falls back to its standard deviation; one
    with no spread at all standardizes to zero.  Values are clipped to
    +/- :data:`CLIP`.

    Args:
        raw: Features ``(segments, columns)``.

    Returns:
        The standardized matrix, same shape, finite.
    """
    data = np.nan_to_num(np.asarray(raw, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    median = np.median(data, axis=0)
    scale = 1.4826 * np.median(np.abs(data - median), axis=0)
    spread = data.std(axis=0)
    scale = np.where(scale > 1e-9, scale, spread)
    usable = scale > 1e-9
    z = np.zeros_like(data)
    z[:, usable] = (data[:, usable] - median[usable]) / scale[usable]
    clipped: np.ndarray[Any, Any] = np.clip(z, -CLIP, CLIP)
    return clipped


def _too_short(kind: str, found: int, need: str) -> ReviewError:
    return ReviewError(
        f"this {kind} yields {found} segment(s); a review needs at least {MIN_SEGMENTS} "
        f"({need}), because each segment is measured against the rest of the same {kind}"
    )


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #

_WORD = re.compile(r"\S+")
_SENTENCE_END = re.compile(r"[.!?]+(?:\s|$)")
_PUNCTUATION = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def _entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def _surprise(own: Counter[str], whole: Counter[str], whole_total: int) -> float:
    """Mean bits per item of ``own`` under the rest of the artifact (add-one smoothed).

    The counts the items are scored against leave this segment's own out, so
    a segment is not made familiar by itself.
    """
    own_total = sum(own.values())
    if own_total == 0:
        return 0.0
    rest_total = whole_total - own_total + len(whole) + 1
    bits = sum(
        count * -math.log2((whole[item] - count + 1) / rest_total) for item, count in own.items()
    )
    return bits / own_total


def _letter_pairs(chunk: str) -> Counter[str]:
    letters = re.sub(r"[^a-z ]+", "", chunk.lower())
    return Counter(letters[i : i + 2] for i in range(len(letters) - 1))


def _document_row(chunk: str, pairs_seen: Counter[str]) -> list[float]:
    tokens = _WORD.findall(chunk)
    words = [token.strip("".join(_PUNCTUATION)) for token in tokens]
    words = [word for word in words if word]
    lengths = np.array([len(word) for word in words], dtype=np.float64)
    count = max(len(words), 1)
    letters = [ch for ch in chunk if ch.isalpha()]
    visible = [ch for ch in chunk if not ch.isspace()]
    n_chars, n_letters, n_visible = max(len(chunk), 1), max(len(letters), 1), max(len(visible), 1)
    bigrams = [chunk[i : i + 2] for i in range(len(chunk) - 1)]
    sentences = max(len(_SENTENCE_END.findall(chunk)), 1)
    return [
        float(lengths.mean()) if lengths.size else 0.0,
        float(lengths.std()) if lengths.size else 0.0,
        len({word.lower() for word in words}) / count,
        sum(len(word) <= 3 for word in words) / count,
        sum(len(word) >= 9 for word in words) / count,
        sum(ch.isupper() for ch in letters) / n_letters,
        sum(ch.isdigit() for ch in chunk) / n_chars,
        sum(ch in _PUNCTUATION for ch in visible) / n_visible,
        100.0 * chunk.count("\n") / n_chars,
        sum(ord(ch) > 127 for ch in chunk) / n_chars,
        _entropy(Counter(chunk)),
        1.0 - len(set(bigrams)) / max(len(bigrams), 1),
        len(words) / sentences,
        sum(word[:1].isupper() for word in words) / count,
        sum(any(ch.isdigit() for ch in word) for word in words) / count,
        _surprise(_letter_pairs(chunk), pairs_seen, sum(pairs_seen.values())),
    ]


def _document(decoded: Decoded) -> Segmented:
    text = decoded.text or ""
    spans = [(match.start(), match.end()) for match in _WORD.finditer(text)]
    window = max(30, len(spans) // TARGET_SEGMENTS[DOCUMENT])
    starts = range(0, len(spans) - window + 1, window)
    if len(starts) < MIN_SEGMENTS:
        raise _too_short(DOCUMENT, len(starts), f"about {MIN_SEGMENTS * 30} words")
    # What the whole document's letter pairs look like: each window is then
    # asked how surprising it is given the rest.
    pairs_seen = _letter_pairs(text)
    rows, segments = [], []
    for index, first in enumerate(starts):
        last = len(spans) if index == len(starts) - 1 else first + window
        begin, end = spans[first][0], spans[last - 1][1]
        chunk = text[begin:end]
        rows.append(_document_row(chunk, pairs_seen))
        segments.append(
            {
                "index": index,
                "characters": [begin, end],
                "words": [first, last],
                "excerpt": " ".join(chunk.split())[:120],
            }
        )
    raw = np.asarray(rows, dtype=np.float64)
    return Segmented(
        DOCUMENT,
        DOCUMENT_FEATURES,
        raw,
        standardize(raw),
        tuple(segments),
        {"words": len(spans), "words_per_segment": window, "format": decoded.format},
    )


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #


def _histogram_summary(
    block: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Peak share and entropy of each row's histogram (rows sum to about one)."""
    total = block.sum(axis=1, keepdims=True)
    share = np.divide(block, total, out=np.zeros_like(block), where=total > 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(share > 0, np.log2(share), 0.0)
    return share.max(axis=1), -(share * logs).sum(axis=1)


def _image(decoded: Decoded) -> Segmented:
    from PIL import Image

    from omni_mercury_engine.vision.features import extract_patch_features
    from omni_mercury_engine.vision.imaging import to_canonical

    pixels = decoded.pixels
    assert pixels is not None
    if max(pixels.shape[:2]) > IMAGE_WORKING_SIDE:  # not through decode(): reduce here
        scale = IMAGE_WORKING_SIDE / max(pixels.shape[:2])
        size = (
            max(32, round(pixels.shape[1] * scale)),
            max(32, round(pixels.shape[0] * scale)),
        )
        original = (int(pixels.shape[0]), int(pixels.shape[1]))
        pixels = np.asarray(Image.fromarray(pixels).resize(size, Image.Resampling.BILINEAR))
    else:
        detail = decoded.detail or {}
        original = (
            int(detail.get("height", pixels.shape[0])),
            int(detail.get("width", pixels.shape[1])),
        )
    height, width = original
    work_h, work_w = pixels.shape[:2]
    # One pixel of the measured image is this many of the original's.
    back = max(height / work_h, width / work_w, 1.0)
    if min(work_h, work_w) < 48:
        raise ReviewError(
            f"the image is {width} x {height} pixels; a review needs at least 48 on each side"
        )
    # Square tiles sized so the grid holds about TARGET_SEGMENTS of them.
    patch = int(math.sqrt(work_h * work_w / TARGET_SEGMENTS[IMAGE])) // 8 * 8
    patch = max(16, min(patch, min(work_h, work_w) // 3 // 8 * 8 or 16))
    features = extract_patch_features(
        to_canonical(pixels), patch_size=patch, stride=patch, scales=(1.0,)
    )
    flat = np.asarray(features.flat, dtype=np.float64)
    column = {name.split(".", 1)[1]: index for index, name in enumerate(features.names)}

    def col(name: str) -> np.ndarray[Any, Any]:
        return flat[:, column[name]] if name in column else np.zeros(flat.shape[0])

    orientation = np.stack([col(f"orientation_bin{i}") for i in range(8)], axis=1)
    texture = np.stack([col(f"texture_lbp{i}") for i in range(10)], axis=1)
    orientation_peak, orientation_entropy = _histogram_summary(np.abs(orientation))
    texture_peak, texture_entropy = _histogram_summary(np.abs(texture))
    raw = np.stack(
        [
            col("intensity_mean"), col("intensity_std"), col("gradient_magnitude"),
            orientation_peak, orientation_entropy, texture_peak, texture_entropy,
            col("frequency_fine"), col("frequency_mid"), col("frequency_coarse"),
            col("noise_residual"), col("edges_density"),
            col("lattice_any_axis"), col("lattice_both_axes"),
            col("color_saturation"),
            np.hypot(col("color_red_green"), col("color_yellow_blue")),
        ],
        axis=1,
    )  # fmt: skip
    if raw.shape[0] < MIN_SEGMENTS:
        raise _too_short(IMAGE, raw.shape[0], "a larger image")
    grid = features.grid
    segments = tuple(
        {
            "index": index,
            "box": [
                round(float(x) * back),
                round(float(y) * back),
                round(patch * back),
                round(patch * back),
            ],
            "tile": [row_index, col_index],
        }
        for index, (row_index, y, col_index, x) in enumerate(
            (r, y, c, x) for r, y in enumerate(grid.rows) for c, x in enumerate(grid.cols)
        )
    )
    return Segmented(
        IMAGE,
        IMAGE_FEATURES,
        raw,
        standardize(raw),
        segments,
        {
            "width": int(width),
            "height": int(height),
            "tile_pixels": round(patch * back),
            "grid": [len(grid.rows), len(grid.cols)],
            "colour": bool(pixels.ndim == 3),
            "format": decoded.format,
        },
    )


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #


def _audio(decoded: Decoded) -> Segmented:
    from scipy.signal import resample_poly

    from omni_mercury_engine.biometric.voice_recognition import AudioPreprocessor, MFCCExtractor

    samples, rate = decoded.samples, int(decoded.sample_rate or 0)
    assert samples is not None
    if rate != _AUDIO_RATE:
        ratio = Fraction(_AUDIO_RATE, rate).limit_denominator(_MAX_RESAMPLE_TERM)
        if ratio.numerator > _MAX_RESAMPLE_TERM:  # only for rates far below 16 kHz
            ratio = 1 / Fraction(rate, _AUDIO_RATE).limit_denominator(_MAX_RESAMPLE_TERM)
        samples = resample_poly(samples, ratio.numerator, ratio.denominator)
    seconds = samples.size / _AUDIO_RATE
    block_seconds = max(0.25, seconds / TARGET_SEGMENTS[AUDIO])
    preprocessor = AudioPreprocessor(sample_rate=_AUDIO_RATE)
    signal = preprocessor.preprocess(samples)
    length, shift = preprocessor.frame_length, preprocessor.frame_shift
    total = 1 + (signal.size - length) // shift if signal.size >= length else 0
    per_block = max(1, round(block_seconds / 0.010))  # frames step by 10 ms
    blocks = total // per_block
    if blocks < MIN_SEGMENTS:
        raise _too_short(AUDIO, blocks, f"about {MIN_SEGMENTS * 0.25:.0f} seconds")
    used = blocks * per_block
    cepstral = MFCCExtractor(sample_rate=_AUDIO_RATE)
    measured = []
    # Frame by frame the measurements are independent, so they are taken a
    # batch of frames at a time: the same numbers, a bounded peak.
    for first in range(0, used, _FRAME_BATCH):
        count = min(_FRAME_BATCH, used - first)
        piece = signal[first * shift : (first + count - 1) * shift + length]
        frames = np.asarray(preprocessor.frame(piece), dtype=np.float64)
        spectrum = np.abs(np.fft.rfft(frames, n=512, axis=1)) ** 2 + 1e-12
        measured.append(
            np.column_stack(
                [
                    np.asarray(cepstral.extract(frames), dtype=np.float64)[:, 1:13],
                    np.log(np.mean(frames**2, axis=1) + 1e-10),
                    np.mean(np.abs(np.diff(np.signbit(frames).astype(np.int8), axis=1)), axis=1),
                    np.exp(np.mean(np.log(spectrum), axis=1)) / np.mean(spectrum, axis=1),
                ]
            )
        )
    per_frame = np.vstack(measured).reshape(blocks, per_block, 15)
    energy = per_frame[:, :, 12]
    raw = np.column_stack(
        [
            per_frame[:, :, :12].mean(axis=1),
            energy.mean(axis=1),
            energy.std(axis=1),
            per_frame[:, :, 13].mean(axis=1),
            per_frame[:, :, 14].mean(axis=1),
        ]
    )
    step = per_block * 0.010
    segments = tuple(
        {"index": index, "seconds": [round(index * step, 3), round((index + 1) * step, 3)]}
        for index in range(blocks)
    )
    return Segmented(
        AUDIO,
        AUDIO_FEATURES,
        raw,
        standardize(raw),
        segments,
        {
            "seconds": round(seconds, 3),
            "sample_rate": rate,
            "analysed_at_hz": _AUDIO_RATE,
            "seconds_per_segment": round(step, 3),
            "format": decoded.format,
        },
    )


def extract(decoded: Decoded) -> Segmented:
    """Cut a decoded artifact into segments and measure each.

    Args:
        decoded: What :func:`omni_mercury_engine.review.decode.decode` recovered.

    Returns:
        Its segments x features, raw and standardized against itself.

    Raises:
        ReviewError: 422 when the artifact is too small to be its own reference.
    """
    reader = {DOCUMENT: _document, IMAGE: _image, AUDIO: _audio}[decoded.kind]
    segmented = reader(decoded)
    assert segmented.matrix.shape[1] == WIDTH
    return segmented
