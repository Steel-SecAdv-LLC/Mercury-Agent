# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reference artifacts with known splices: what the review model is fitted and measured on.

The review path asks one question of an uploaded document, image or
recording: is some part of it unlike the rest?  To fit and calibrate a model
for that, and to measure it, we need artifacts where the answer is known.
This module builds them, seeded and network-free:

* **documents** are runs of real prose taken from this repository's own
  Markdown documentation.  A spliced document has foreign material inserted:
  source code, a numeric table, an encoded blob, upper-cased prose, a repeated
  sentence, scrambled words, or prose from a different document (the hard
  case: same language, another author's subject).
* **images** are procedural textures (shaped noise with an illumination
  gradient and sensor grain).  A spliced image has regions pasted in from a
  different texture, blurred, or re-grained.
* **recordings** are procedural voiced signals (a drifting harmonic source
  through formant resonances, with syllable-rate modulation, pauses and room
  noise).  A spliced recording has intervals replaced by another voice and
  room, a noise burst, a tone, a dropout, or an overdriven copy.

Each artifact carries the exact extent of what was inserted, so every segment
the feature extractor cuts can be labelled (spliced when at least half of it
is foreign material).

What this is not: a corpus of real-world tampering.  The splices are
procedural, so the measured numbers say how well internal inconsistency of
these kinds is found, not how well forgeries in the wild are.  Everything
that reports those numbers says so.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.review.decode import AUDIO, DOCUMENT, IMAGE, Decoded

if TYPE_CHECKING:
    from collections.abc import Iterator

    from omni_mercury_engine.review.features import Segmented

REPO = Path(__file__).resolve().parents[2]
_RATE = 16_000

#: Documentation whose subject is the harm gate is left out of the prose pool:
#: a reference corpus should not be made of text about weapons screening.
_EXCLUDED_DOCS = ("WEAPONS", "RED_TEAM", "HARM", "MEANING_LEVEL", "CHANGELOG", "INVENTORY", "SKIP")


@dataclass(frozen=True)
class Reference:
    """One reference artifact and where its foreign material is.

    Attributes:
        kind: ``document``, ``image`` or ``audio``.
        decoded: The artifact as a decoder would return it.
        spliced: Whether anything was inserted.
        extents: Where: ``(start, end)`` character spans, a boolean pixel mask,
            or ``(start, end)`` second spans.
        recipe: What was inserted (``clean`` when nothing was).
        source: What the artifact's own material was drawn from (a document's
            prose file; empty for generated images and recordings).  Artifacts
            of one source resemble each other more than artifacts of
            different sources do, which is what a family test has to allow
            for.
    """

    kind: str
    decoded: Decoded
    spliced: bool
    extents: Any
    recipe: str
    source: str = ""


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #


def prose_files(split: str) -> list[Path]:
    """This repository's Markdown documents, partitioned by name.

    Args:
        split: ``train``, ``calibration`` or ``evaluation``; the three never
            share a file.

    Returns:
        The files of that split, sorted.
    """
    files = sorted(
        path
        for path in [*REPO.glob("docs/*.md"), REPO / "README.md", REPO / "ARCHITECTURE.md"]
        if not any(mark in path.name.upper() for mark in _EXCLUDED_DOCS)
        and path.stat().st_size > 6000
    )
    # A quarter to fit on, half to calibrate on, a quarter held out: how much
    # natural variation a clean document carries differs from source to
    # source, so the certificate needs as many sources as can be spared.
    wanted = {"train": (0,), "calibration": (1, 2), "evaluation": (3,)}[split]
    return [path for position, path in enumerate(files) if position % 4 in wanted]


def corpus_digest() -> dict[str, str]:
    """A SHA-256 of the running prose each split draws its documents from.

    Reference documents are cut from this repository's own documentation, so
    they change when that documentation does: a checkpoint trained before an
    edit and a measurement taken after it describe different documents.  The
    trainer writes this into the document checkpoint's provenance and the
    harness into its artifact, so whether the two describe the same corpus is
    a recorded fact, not an assumption.

    Returns:
        ``{"train": ..., "calibration": ..., "evaluation": ...}``, hex digests.
    """
    digests = {}
    for split in ("train", "calibration", "evaluation"):
        hasher = hashlib.sha256()
        for path in prose_files(split):
            hasher.update(str(path.relative_to(REPO)).encode("utf-8") + b"\x00")
            hasher.update(" ".join(_prose(path)).encode("utf-8") + b"\x00")
        digests[split] = hasher.hexdigest()
    return digests


def _code_files() -> list[Path]:
    return sorted((REPO / "src" / "omni_mercury_engine" / "utils").glob("*.py"))


_FENCE = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.S | re.M)
_NOT_PROSE = re.compile(r"^\s*(\||<|!\[|\[!\[|#{1,6}\s|[-=*_]{3,}\s*$).*$", re.M)


def _words(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").split(" ")


def _prose(path: Path) -> list[str]:
    """A Markdown file's running prose: no code fences, tables, markup lines or headings.

    A reference document has to be one kind of text, or its own tables and
    listings would be the "foreign material" a clean document is said not to
    have.  (An uploaded document that mixes prose with tables or listings
    will, correctly, show those parts as unlike the rest.)
    """
    text = _FENCE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    text = _NOT_PROSE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).split(" ")


def _foreign_text(rng: np.random.Generator, words: int, other_prose: list[str]) -> tuple[str, str]:
    recipe = str(
        rng.choice(["code", "table", "blob", "shouting", "loop", "scrambled", "other_document"])
    )
    start = int(rng.integers(0, max(1, len(other_prose) - words)))
    borrowed = other_prose[start : start + words]
    if recipe == "code":
        source = _words(_code_files()[int(rng.integers(0, len(_code_files())))])
        begin = int(rng.integers(0, max(1, len(source) - words)))
        return " ".join(source[begin : begin + words]), recipe
    if recipe == "table":
        rows = [",".join(f"{value:.3f}" for value in rng.normal(size=6)) for _ in range(words // 6)]
        return "\n".join(rows), recipe
    if recipe == "blob":
        alphabet = np.array(
            list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")
        )
        lines = ["".join(rng.choice(alphabet, size=76)) for _ in range(max(2, words // 10))]
        return "\n".join(lines), recipe
    if recipe == "shouting":
        return " ".join(borrowed).upper(), recipe
    if recipe == "loop":
        sentence = " ".join(borrowed[:9]) + "."
        return " ".join([sentence] * max(2, words // 9)), recipe
    if recipe == "scrambled":
        scrambled = ["".join(rng.permutation(list(word))) for word in borrowed]
        return " ".join(scrambled), recipe
    return " ".join(borrowed), recipe


def documents(count: int, seed: int, split: str) -> Iterator[Reference]:
    """Reference documents: alternately clean and spliced.

    Args:
        count: How many to build.
        seed: Seed for every choice made.
        split: Which third of the prose files to draw from.

    Yields:
        The documents.
    """
    rng = np.random.default_rng(seed)
    named = [(path, _prose(path)) for path in prose_files(split)]
    names = [str(path.relative_to(REPO)) for path, words in named if len(words) > 800]
    pool = [words for _, words in named if len(words) > 800]
    for number in range(count):
        chosen = int(rng.integers(0, len(pool)))
        source = pool[chosen]
        length = int(rng.integers(1500, 6000))
        start = int(rng.integers(0, max(1, len(source) - length)))
        body = source[start : start + length]
        spliced = number % 2 == 1
        extents: list[tuple[int, int]] = []
        recipes: list[str] = []
        if spliced:
            other = pool[int(rng.integers(0, len(pool)))]
            for _ in range(int(rng.integers(1, 4))):
                size = int(rng.integers(60, 260))
                at = int(rng.integers(0, max(1, len(body) - 1)))
                foreign, recipe = _foreign_text(rng, size, other)
                body = [*body[:at], "\x00" + foreign + "\x01", *body[at:]]
                recipes.append(recipe)
        text = " ".join(body)
        while "\x00" in text:  # markers become extents, then go
            begin = text.index("\x00")
            end = text.index("\x01", begin)
            text = text[:begin] + text[begin + 1 : end] + text[end + 1 :]
            extents.append((begin, end - 1))
        yield Reference(
            DOCUMENT,
            Decoded(DOCUMENT, "text", text=text, detail={"encoding": "utf-8"}),
            spliced,
            extents,
            "+".join(recipes) or "clean",
            names[chosen],
        )


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #


def _texture(rng: np.random.Generator, height: int, width: int) -> np.ndarray[Any, Any]:
    """Shaped noise: a random spectral slope, contrast, illumination and grain."""
    slope = rng.uniform(0.8, 2.6)
    fy, fx = np.fft.fftfreq(height)[:, None], np.fft.rfftfreq(width)[None, :]
    radius = np.hypot(fy, fx)
    radius[0, 0] = 1.0
    spectrum = (rng.normal(size=radius.shape) + 1j * rng.normal(size=radius.shape)) / radius**slope
    field = np.fft.irfft2(spectrum, s=(height, width))
    field = (field - field.mean()) / (field.std() + 1e-9)
    yy, xx = np.mgrid[0:height, 0:width]
    gradient = rng.uniform(-0.6, 0.6) * (yy / height) + rng.uniform(-0.6, 0.6) * (xx / width)
    grain = rng.normal(scale=rng.uniform(0.02, 0.12), size=(height, width))
    image = rng.uniform(0.35, 0.65) + rng.uniform(0.08, 0.2) * field + 0.15 * gradient + grain
    clipped: np.ndarray[Any, Any] = np.clip(image, 0.0, 1.0)
    return clipped


def _blur(image: np.ndarray[Any, Any], passes: int) -> np.ndarray[Any, Any]:
    out = image
    for _ in range(passes):
        out = (out + np.roll(out, 1, 0) + np.roll(out, -1, 0) + np.roll(out, 1, 1) + np.roll(out, -1, 1)) / 5.0  # fmt: skip
    return out


def images(count: int, seed: int) -> Iterator[Reference]:
    """Reference images: alternately clean and spliced.

    Yields:
        The images (grayscale or tinted colour, uint8).
    """
    rng = np.random.default_rng(seed)
    for number in range(count):
        height, width = int(rng.integers(320, 560)), int(rng.integers(400, 720))
        base = _texture(rng, height, width)
        mask = np.zeros((height, width), dtype=bool)
        spliced = number % 2 == 1
        recipes: list[str] = []
        if spliced:
            for _ in range(int(rng.integers(1, 4))):
                h, w = int(rng.integers(50, 150)), int(rng.integers(50, 170))
                y, x = int(rng.integers(0, height - h)), int(rng.integers(0, width - w))
                recipe = str(rng.choice(["other_texture", "blur", "regrain"]))
                region = base[y : y + h, x : x + w]
                if recipe == "other_texture":
                    region = _texture(rng, h, w)
                elif recipe == "blur":
                    region = _blur(region, int(rng.integers(3, 9)))
                else:
                    region = np.clip(region + rng.normal(scale=rng.uniform(0.12, 0.3), size=(h, w)), 0, 1)  # fmt: skip
                base[y : y + h, x : x + w] = region
                mask[y : y + h, x : x + w] = True
                recipes.append(recipe)
        if rng.random() < 0.5:
            tint = rng.uniform(0.75, 1.0, size=3)
            pixels = (base[:, :, None] * tint[None, None, :] * 255).astype(np.uint8)
        else:
            pixels = (base * 255).astype(np.uint8)
        yield Reference(
            IMAGE,
            Decoded(IMAGE, "png", pixels=pixels, detail={"width": width, "height": height}),
            spliced,
            mask,
            "+".join(recipes) or "clean",
        )


# --------------------------------------------------------------------------- #
# Recordings
# --------------------------------------------------------------------------- #


def _voice(rng: np.random.Generator, seconds: float) -> np.ndarray[Any, Any]:
    """A voiced signal: drifting pitch, formants, syllable-rate loudness, pauses, room noise."""
    n = int(seconds * _RATE)
    t = np.arange(n) / _RATE
    pitch = rng.uniform(95, 230) * (1 + 0.06 * np.sin(2 * np.pi * rng.uniform(0.2, 0.7) * t))
    phase = 2 * np.pi * np.cumsum(pitch) / _RATE
    formants = rng.uniform([300, 900, 2200], [800, 2000, 3200])
    source = np.zeros(n)
    for harmonic in range(1, 26):
        frequency = harmonic * float(np.mean(pitch))
        if frequency > 3800:
            break
        gain = sum(1.0 / (1.0 + ((frequency - f) / 140.0) ** 2) for f in formants) / harmonic**0.5
        source += gain * np.sin(harmonic * phase + rng.uniform(0, 2 * np.pi))
    syllables = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(3.0, 6.0) * t + rng.uniform(0, 6.28))
    speaking = (np.sin(2 * np.pi * rng.uniform(0.15, 0.4) * t + rng.uniform(0, 6.28)) > -0.55).astype(float)  # fmt: skip
    voiced = source / (np.abs(source).max() + 1e-9) * syllables * speaking
    noise = np.cumsum(rng.normal(size=n))
    noise = noise - np.convolve(noise, np.ones(400) / 400, mode="same")
    noise = noise / (noise.std() + 1e-9) * rng.uniform(0.01, 0.05)
    signal: np.ndarray[Any, Any] = (
        0.5 * voiced + noise + rng.normal(scale=rng.uniform(0.002, 0.01), size=n)
    )
    return signal


def recordings(count: int, seed: int) -> Iterator[Reference]:
    """Reference recordings: alternately clean and spliced (16 kHz mono).

    Yields:
        The recordings.
    """
    rng = np.random.default_rng(seed)
    for number in range(count):
        seconds = float(rng.uniform(14, 40))
        signal = _voice(rng, seconds)
        spliced = number % 2 == 1
        extents: list[tuple[float, float]] = []
        recipes: list[str] = []
        if spliced:
            for _ in range(int(rng.integers(1, 4))):
                length = float(rng.uniform(0.6, 2.6))
                start = float(rng.uniform(0, seconds - length))
                a, b = int(start * _RATE), int((start + length) * _RATE)
                recipe = str(rng.choice(["other_voice", "noise", "tone", "dropout", "overdrive"]))
                if recipe == "other_voice":
                    signal[a:b] = _voice(rng, length + 0.01)[: b - a]
                elif recipe == "noise":
                    signal[a:b] = rng.normal(scale=rng.uniform(0.05, 0.2), size=b - a)
                elif recipe == "tone":
                    tone_t = np.arange(b - a) / _RATE
                    signal[a:b] = 0.3 * np.sin(2 * np.pi * rng.uniform(300, 2500) * tone_t)
                elif recipe == "dropout":
                    signal[a:b] = 0.0
                else:
                    signal[a:b] = np.clip(signal[a:b] * rng.uniform(6, 14), -0.6, 0.6)
                extents.append((start, start + length))
                recipes.append(recipe)
        samples = np.clip(signal, -1.0, 1.0)
        yield Reference(
            AUDIO,
            Decoded(AUDIO, "wav", samples=samples, sample_rate=_RATE, detail={"seconds": seconds}),
            spliced,
            extents,
            "+".join(recipes) or "clean",
        )


def references(kind: str, count: int, seed: int, split: str) -> Iterator[Reference]:
    """Reference artifacts of one kind.

    Args:
        kind: ``document``, ``image`` or ``audio``.
        count: How many.
        seed: Seed (use different seeds for sets that must not overlap).
        split: Prose split for documents (ignored otherwise: images and
            recordings are generated, so a different seed is a different set).

    Yields:
        The artifacts.
    """
    if kind == DOCUMENT:
        yield from documents(count, seed, split)
    elif kind == IMAGE:
        yield from images(count, seed)
    else:
        yield from recordings(count, seed)


# --------------------------------------------------------------------------- #
# Labels and bytes
# --------------------------------------------------------------------------- #


def _overlap(span: tuple[float, float], extents: list[tuple[float, float]]) -> float:
    start, end = span
    covered = sum(max(0.0, min(end, b) - max(start, a)) for a, b in extents)
    return covered / max(end - start, 1e-9)


def segment_labels(segmented: Segmented, reference: Reference) -> np.ndarray[Any, Any]:
    """One label per segment: 1 when at least half of it is foreign material."""
    labels = np.zeros(len(segmented.segments), dtype=np.int64)
    if not reference.spliced:
        return labels
    for index, segment in enumerate(segmented.segments):
        if reference.kind == DOCUMENT:
            share = _overlap(tuple(segment["characters"]), reference.extents)
        elif reference.kind == AUDIO:
            share = _overlap(tuple(segment["seconds"]), reference.extents)
        else:
            x, y, w, h = segment["box"]
            share = float(reference.extents[y : y + h, x : x + w].mean()) if w and h else 0.0
        labels[index] = int(share >= 0.5)
    return labels


def to_bytes(reference: Reference) -> tuple[bytes, str]:
    """The artifact as an uploadable file: UTF-8 text, PNG, or 16-bit PCM WAV.

    Returns:
        ``(bytes, filename)``.
    """
    decoded = reference.decoded
    if reference.kind == DOCUMENT:
        return (decoded.text or "").encode("utf-8"), "reference.txt"
    if reference.kind == IMAGE:
        from PIL import Image

        assert decoded.pixels is not None
        buffer = io.BytesIO()
        Image.fromarray(decoded.pixels).save(buffer, format="PNG")
        return buffer.getvalue(), "reference.png"
    assert decoded.samples is not None
    pcm = (decoded.samples * 32767.0).astype("<i2").tobytes()
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(pcm))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, _RATE, _RATE * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(pcm))
    )
    return header + pcm, "reference.wav"


# --------------------------------------------------------------------------- #
# The committed samples
# --------------------------------------------------------------------------- #

#: Which reference artifact each committed sample is: ``(seed, index)`` into
#: :func:`references` on the evaluation split.  Each is a *spliced* artifact,
#: so a reader can check a review against what was planted (the planted
#: extents are written beside the samples).  They are examples to run, chosen
#: to be small and to show a grounded decision; they are not a measurement
#: (``research/review/harness_artifact_review.py`` is).
SAMPLES: dict[str, tuple[int, int]] = {DOCUMENT: (47, 3), IMAGE: (49, 1), AUDIO: (58, 3)}
SAMPLE_NAMES = {
    DOCUMENT: "sample_document.txt",
    IMAGE: "sample_image.png",
    AUDIO: "sample_recording.wav",
}


def sample(kind: str) -> Reference:
    """The reference artifact behind the committed sample of ``kind``."""
    seed, index = SAMPLES[kind]
    for position, reference in enumerate(references(kind, index + 1, seed, "evaluation")):
        if position == index:
            return reference
    raise LookupError(kind)


def planted(reference: Reference) -> list[Any]:
    """Where a reference's foreign material is, in the units a review reports.

    Returns:
        Character spans, pixel boxes ``[x, y, width, height]`` (the bounding
        box of each planted region is not kept by the generator, so this is
        the box of the whole mask), or second spans.
    """
    if reference.kind == IMAGE:
        rows, cols = np.nonzero(reference.extents)
        if rows.size == 0:
            return []
        return [
            [
                int(cols.min()),
                int(rows.min()),
                int(cols.max() - cols.min() + 1),
                int(rows.max() - rows.min() + 1),
            ]
        ]
    return [[round(float(a), 2), round(float(b), 2)] for a, b in reference.extents]


def write_samples(directory: Path) -> dict[str, Any]:
    """Write the three committed samples and what was planted in each.

    Returns:
        The manifest also written to ``planted.json``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {}
    for kind, name in SAMPLE_NAMES.items():
        reference = sample(kind)
        data, _ = to_bytes(reference)
        (directory / name).write_bytes(data)
        seed, index = SAMPLES[kind]
        manifest[name] = {
            "kind": kind,
            "bytes": len(data),
            "reference": {"seed": seed, "index": index, "split": "evaluation"},
            "planted": reference.recipe,
            "planted_at": planted(reference),
            "units": {DOCUMENT: "characters", IMAGE: "pixels [x, y, width, height]"}.get(
                kind, "seconds"
            ),
        }
    (directory / "planted.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Regenerate the committed samples."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument(
        "--write-samples",
        type=Path,
        metavar="DIR",
        required=True,
        help="write sample_document.txt, sample_image.png, sample_recording.wav and planted.json",
    )
    args = parser.parse_args(argv)
    for name, entry in write_samples(args.write_samples).items():
        print(
            f"{name}: {entry['bytes']} bytes, planted {entry['planted']} at {entry['planted_at']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
