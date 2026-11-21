# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Turn uploaded bytes into text, pixels or samples, within stated bounds.

An upload is whatever a caller sends, so every decoder here is bounded before
it expands anything: a compressed stream is inflated only up to a cap, an
image's declared size is checked before its pixels are read, a container is
read member by member with a byte limit.  What a decoder cannot read it
refuses with a reason; it never guesses, and it never reviews a silently
shortened artifact: one past a bound is refused (413) with the bound.  Every
reader is one linear pass over its input; none uses a pattern that can be
made to rescan it.

Formats, and what reads them:

* **documents** -- plain text (UTF-8, UTF-16 with a byte-order mark, Latin-1),
  HTML (standard-library parser), DOCX (the ``word/document.xml`` member of
  the archive, read without an XML parser) and PDF (the text operators of its
  content streams; a scanned or font-encoded PDF has no text to read and is
  refused, not approximated);
* **images** -- whatever Pillow opens (PNG, JPEG, GIF, BMP, WebP, TIFF); the
  first frame, as RGB or grayscale;
* **audio** -- uncompressed RIFF/WAVE (integer PCM of 8 to 32 bits, 32- and
  64-bit float, plain or ``WAVE_FORMAT_EXTENSIBLE``).  Compressed audio (MP3,
  AAC/M4A, Ogg, FLAC) is recognised and refused with the conversion to run,
  because decoding it needs a codec this package does not carry.

The decoders are original and small on purpose.  They are not a document
toolkit: they recover enough of an artifact to measure its internal
consistency (:mod:`omni_mercury_engine.review.features`), no more.
"""

from __future__ import annotations

import html
import io
import re
import struct
import zipfile
import zlib
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

import numpy as np

__all__ = [
    "AUDIO",
    "DOCUMENT",
    "IMAGE",
    "IMAGE_WORKING_SIDE",
    "KINDS",
    "DecodeBounds",
    "Decoded",
    "ReviewError",
    "decode",
    "sniff",
]

DOCUMENT, IMAGE, AUDIO = "document", "image", "audio"
KINDS = (DOCUMENT, IMAGE, AUDIO)

#: Share of decoded characters that must be printable for bytes to count as text.
_MIN_PRINTABLE = 0.85
#: Share of a PDF's recovered characters that must be letters, digits or spaces.
_MIN_PDF_READABLE = 0.75
#: Most content streams read from one PDF.
_MAX_PDF_STREAMS = 4096
#: Most RIFF chunks walked looking for a WAV file's format and data.
_MAX_WAV_CHUNKS = 4096
#: Longest side, in pixels, an image is measured at.  A larger one is reduced
#: to this as it is decoded (so a 25-megapixel photograph is never held as a
#: full-size array), and segment boxes are reported in the original's pixels.
#: 720 is the upper end of the sizes the image checkpoint was calibrated on,
#: and it bounds the cost of the tile statistics, which grows with the pixel
#: count (measured: 0.5 GB and about a second at 720; 1 GB and 3.5 s at 1024).
IMAGE_WORKING_SIDE = 720
#: Sample rates (hertz) and channel counts of the recordings reviewed.
_WAV_RATES = (4_000, 192_000)
_MAX_WAV_CHANNELS = 8


class ReviewError(ValueError):
    """An upload that cannot be reviewed, with the HTTP status that says why.

    Attributes:
        status: 413 (over a bound), 415 (a format that is not decoded) or 422
            (decodable in principle, but not this file).
    """

    def __init__(self, message: str, *, status: int = 422) -> None:
        """Keep the reason and the HTTP status that says why."""
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class DecodeBounds:
    """How far a decoder may expand an upload.

    The defaults are the served defaults
    (:mod:`omni_mercury_engine.serving_limits`, ``review_max_*``).  The text
    bound sits under what the harm-uplift gate reads in one decision
    (1,048,576 characters): a longer document could not be assessed by it and
    would be refused there anyway.

    Attributes:
        max_text_chars: Most characters of text a document may hold.
        max_pixels: Most pixels (width x height) an image may hold.
        max_audio_seconds: Longest recording, in seconds.
    """

    max_text_chars: int = 1_000_000
    max_pixels: int = 25_000_000
    max_audio_seconds: int = 600


@dataclass(frozen=True)
class Decoded:
    """What a decoder recovered.

    Attributes:
        kind: ``document``, ``image`` or ``audio``.
        format: The format read (``pdf``, ``png``, ``wav`` ...).
        text: A document's text (``None`` otherwise).
        pixels: An image as ``(H, W)`` or ``(H, W, 3)`` uint8, reduced to at most
            :data:`IMAGE_WORKING_SIDE` on its longer side; ``detail`` keeps the
            original width and height (``None`` otherwise).
        samples: Audio as mono float64 in ``[-1, 1]`` (``None`` otherwise).
        sample_rate: Audio sample rate in hertz.
        detail: Facts about the decode worth recording (pages, channels ...).
    """

    kind: str
    format: str
    text: str | None = None
    pixels: np.ndarray[Any, Any] | None = None
    samples: np.ndarray[Any, Any] | None = None
    sample_rate: int | None = None
    detail: dict[str, Any] | None = None


# --------------------------------------------------------------------------- #
# Recognition
# --------------------------------------------------------------------------- #

_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),
    (b"II*\x00", "tiff"),
    (b"MM\x00*", "tiff"),
)
#: Compressed audio this package recognises and does not decode.
_COMPRESSED_AUDIO: tuple[tuple[bytes, str], ...] = (
    (b"ID3", "mp3"),
    (b"\xff\xfb", "mp3"),
    (b"\xff\xf3", "mp3"),
    (b"\xff\xf2", "mp3"),
    (b"OggS", "ogg"),
    (b"fLaC", "flac"),
)
_TEXT_SUFFIXES = (
    ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".jsonl", ".log", ".yaml",
    ".yml", ".xml", ".tex", ".py", ".ini", ".cfg", ".toml",
)  # fmt: skip


def sniff(data: bytes, filename: str = "") -> tuple[str, str]:
    """Recognise an upload by its leading bytes, then by its name.

    Args:
        data: The upload.
        filename: Its name, used only when the bytes do not decide.

    Returns:
        ``(kind, format)``.

    Raises:
        ReviewError: 415 when the bytes are a format this package does not
            decode, or nothing recognisable.
    """
    head, name = data[:16], filename.lower()
    if not data:
        raise ReviewError("the upload is empty")
    for magic, fmt in _IMAGE_MAGIC:
        if head.startswith(magic):
            return IMAGE, fmt
    if head[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return IMAGE, "webp"
    if head[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return AUDIO, "wav"
    for magic, fmt in _COMPRESSED_AUDIO:
        if head.startswith(magic):
            raise ReviewError(_compressed_audio(fmt), status=415)
    if data[4:8] == b"ftyp":
        raise ReviewError(_compressed_audio("mp4/m4a"), status=415)
    if head.startswith(b"%PDF-"):
        return DOCUMENT, "pdf"
    if head.startswith(b"PK\x03\x04"):
        if name.endswith(".docx") or b"word/document.xml" in data[:4096]:
            return DOCUMENT, "docx"
        raise ReviewError(
            "a ZIP archive that is not a DOCX document is not reviewed (supported documents: "
            "text, Markdown, CSV, JSON, HTML, DOCX, PDF)",
            status=415,
        )
    if name.endswith((".html", ".htm")) or re.match(rb"\s*<(!doctype html|html)", data[:256], re.I):
        return DOCUMENT, "html"
    if name.endswith(_TEXT_SUFFIXES) or _looks_like_text(data):
        return DOCUMENT, "text"
    raise ReviewError(
        "unrecognised format (supported: text, Markdown, CSV, JSON, HTML, DOCX, PDF; "
        "PNG, JPEG, GIF, BMP, WebP, TIFF; WAV)",
        status=415,
    )


def _compressed_audio(fmt: str) -> str:
    return (
        f"{fmt} audio is compressed and is not decoded here; convert it to WAV first "
        "(for example: ffmpeg -i input -ac 1 -ar 16000 output.wav)"
    )


def _looks_like_text(data: bytes) -> bool:
    sample = data[:8192]
    if sample.startswith((b"\xff\xfe", b"\xfe\xff", b"\xef\xbb\xbf")):
        return True
    if b"\x00" in sample:
        return False
    try:
        text = sample.decode("utf-8")
    except UnicodeDecodeError:
        text = sample.decode("latin-1")
    return _printable_share(text) >= _MIN_PRINTABLE


def _printable_share(text: str) -> float:
    if not text:
        return 0.0
    return sum(ch.isprintable() or ch in "\n\r\t" for ch in text) / len(text)


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #


def _decode_text(data: bytes, bounds: DecodeBounds) -> tuple[str, dict[str, Any]]:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        text, encoding = data.decode("utf-16", errors="replace"), "utf-16"
    else:
        try:
            text, encoding = data.decode("utf-8-sig"), "utf-8"
        except UnicodeDecodeError:
            text, encoding = data.decode("latin-1"), "latin-1"
    if _printable_share(text[:20_000]) < _MIN_PRINTABLE:
        raise ReviewError("the upload does not decode as text")
    return _within(text, bounds), {"encoding": encoding}


def _within(text: str, bounds: DecodeBounds) -> str:
    """``text``, or a 413 when it is longer than a review reads."""
    if len(text) > bounds.max_text_chars:
        raise _too_much_text(bounds)
    return text


def _too_much_text(bounds: DecodeBounds) -> ReviewError:
    return ReviewError(
        f"the document holds more than {bounds.max_text_chars:,} characters of text, the most "
        "a review reads; upload it in parts",
        status=413,
    )


class _VisibleText(HTMLParser):
    """The text a reader sees: no scripts, no styles, blocks on their own lines."""

    _SKIP = frozenset({"script", "style", "noscript", "template", "head"})
    _BLOCK = frozenset(
        {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skipping = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self._SKIP:
            self._skipping += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skipping:
            self._skipping -= 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skipping:
            self.parts.append(data)


def _decode_html(data: bytes, bounds: DecodeBounds) -> tuple[str, dict[str, Any]]:
    if len(data) > bounds.max_text_chars * 8:  # markup outweighs the text it wraps
        raise _too_much_text(bounds)
    source, detail = _decode_text(data, DecodeBounds(max_text_chars=len(data)))
    parser = _VisibleText()
    parser.feed(source)
    parser.close()
    return _within("".join(parser.parts), bounds), detail


#: The WordprocessingML tags that carry text: a paragraph, a run of text, a
#: tab and a break.  ``[^<>]*`` stops at the next tag, so a tag that never
#: closes cannot make the scan run to the end of the file once per tag.
_DOCX_TAG = re.compile(r"<(/?)w:(p|t|tab|br|cr)(?=[\s/>])[^<>]*>")


def _docx_text(source: str) -> tuple[str, int]:
    """The text of a ``word/document.xml``, one linear pass over its tags.

    Returns:
        ``(text, paragraphs)``.
    """
    paragraphs: list[str] = []
    runs: list[str] = []
    text_from: int | None = None
    for tag in _DOCX_TAG.finditer(source):
        closing, name = tag.group(1) == "/", tag.group(2)
        if name == "t":
            if closing and text_from is not None:
                runs.append(html.unescape(source[text_from : tag.start()]))
                text_from = None
            elif not closing and not tag.group(0).endswith("/>"):
                text_from = tag.end()
        elif name == "p":
            if closing:
                paragraphs.append("".join(runs))
                runs = []
        elif not closing:
            runs.append("\t" if name == "tab" else "\n")
    if runs:
        paragraphs.append("".join(runs))
    return "\n\n".join(p for p in paragraphs if p.strip()), len(paragraphs)


def _decode_docx(data: bytes, bounds: DecodeBounds) -> tuple[str, dict[str, Any]]:
    cap = bounds.max_text_chars * 8  # markup outweighs the text it wraps
    try:
        with (
            zipfile.ZipFile(io.BytesIO(data)) as archive,
            archive.open("word/document.xml") as member,
        ):
            # Read through a limit: the size an archive declares is the archive's claim.
            xml = member.read(cap + 1)
    except (KeyError, zipfile.BadZipFile, zlib.error, OSError, RuntimeError) as exc:
        # RuntimeError: an encrypted member, which zipfile will not read.
        raise ReviewError(f"not a readable DOCX document: {type(exc).__name__}") from exc
    if len(xml) > cap:
        raise _too_much_text(bounds)
    text, paragraphs = _docx_text(xml.decode("utf-8", errors="replace"))
    return _within(text, bounds), {"paragraphs": paragraphs}


_PDF_TOKEN = re.compile(
    rb"\((?P<lit>(?:\\.|[^\\()])*)\)"  # (a literal string)
    rb"|<(?P<hex>[0-9A-Fa-f\s]*)>"  # <a hex string>
    rb"|(?P<op>T\*|Tj|TJ|Td|TD|Tm|ET|BT|'|\")"
)
_PDF_ESCAPE = re.compile(rb"\\([nrtbf()\\]|[0-7]{1,3}|\r?\n)")
_PDF_ESCAPES = {b"n": b"\n", b"r": b"\r", b"t": b"\t", b"b": b"\b", b"f": b"\f"}


def _pdf_literal(raw: bytes) -> str:
    def replace(match: re.Match[bytes]) -> bytes:
        token = match.group(1)
        if token in _PDF_ESCAPES:
            return _PDF_ESCAPES[token]
        if token[:1].isdigit():
            return bytes([int(token, 8) & 0xFF])
        return b"" if token.strip() == b"" else token

    data = _PDF_ESCAPE.sub(replace, raw)
    if data.startswith((b"\xfe\xff", b"\xff\xfe")):
        return data.decode("utf-16", errors="replace")
    return data.decode("latin-1")


def _pdf_hex(raw: bytes) -> str:
    digits = re.sub(rb"\s", b"", raw)
    if len(digits) % 2:
        digits += b"0"
    data = bytes.fromhex(digits.decode("ascii"))
    if data.startswith(b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    return data.decode("latin-1")


def _pdf_stream_text(content: bytes) -> str:
    """Text shown by one content stream: the strings its text operators draw."""
    parts: list[str] = []
    pending: list[str] = []
    for match in _PDF_TOKEN.finditer(content):
        if match.group("lit") is not None:
            pending.append(_pdf_literal(match.group("lit")))
        elif match.group("hex") is not None:
            pending.append(_pdf_hex(match.group("hex")))
        else:
            operator = match.group("op")
            if operator in (b"Tj", b"TJ", b"'", b'"'):
                parts.append("".join(pending))
            if operator in (b"T*", b"Td", b"TD", b"Tm", b"ET", b"'", b'"'):
                parts.append("\n")
            pending = []
    return "".join(parts)


def _pdf_streams(data: bytes) -> Any:
    """Yield each ``stream ... endstream`` body, in one pass over the file.

    Found with ``bytes.find``: a ``stream`` keyword with no ``endstream`` after
    it ends the scan, instead of being searched for again from every later
    keyword.
    """
    position = 0
    while True:
        opened = data.find(b"stream", position)
        if opened < 0:
            return
        begin = opened + 6
        if data[begin : begin + 2] == b"\r\n":
            begin += 2
        elif data[begin : begin + 1] in (b"\n", b"\r"):
            begin += 1
        else:  # the tail of "endstream", or the word in running text
            position = begin
            continue
        closed = data.find(b"endstream", begin)
        if closed < 0:
            return
        yield data[begin:closed].rstrip(b"\r\n")
        position = closed + 9


def _decode_pdf(data: bytes, bounds: DecodeBounds) -> tuple[str, dict[str, Any]]:
    budget = bounds.max_text_chars * 6  # bytes of content stream inflated in total
    pieces: list[str] = []
    streams = 0
    for raw in _pdf_streams(data):
        if streams >= _MAX_PDF_STREAMS or budget <= 0:
            # More than a review reads: refused, not reviewed in part.
            raise _too_much_text(bounds)
        streams += 1
        try:
            inflater = zlib.decompressobj()
            content = inflater.decompress(raw, budget)
        except zlib.error:
            content = raw[:budget]  # not deflated: read as written
        budget -= len(content)
        if b"BT" in content and (b"Tj" in content or b"TJ" in content):
            pieces.append(_pdf_stream_text(content))
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(pieces)).strip()
    readable = sum(ch.isalnum() or ch.isspace() for ch in text) / max(len(text), 1)
    if len(text) < 40 or readable < _MIN_PDF_READABLE:
        raise ReviewError(
            "no readable text in this PDF: it is scanned, encrypted, or its fonts map characters "
            "privately. Export it as text (or as page images) and upload that."
        )
    return _within(text, bounds), {"content_streams": streams}


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #


def _decode_image(data: bytes, bounds: DecodeBounds) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            if width * height > bounds.max_pixels:
                # Checked on the header, before any pixel is decompressed.
                raise ReviewError(
                    f"the image is {width} x {height} pixels; at most {bounds.max_pixels:,} "
                    "pixels are reviewed",
                    status=413,
                )
            image.seek(0)
            mode = "L" if image.mode in ("L", "1", "I", "I;16", "F", "LA") else "RGB"
            longest = max(width, height)
            if longest > IMAGE_WORKING_SIDE:
                scale = IMAGE_WORKING_SIDE / longest
                size = (max(32, round(width * scale)), max(32, round(height * scale)))
                # A JPEG is decoded at the smallest scale that still covers the
                # working size (a no-op for other formats), then reduced exactly.
                image.draft(mode, size)
                working = image.convert(mode).resize(size, Image.Resampling.BILINEAR)
            else:
                working = image.convert(mode)
            pixels = np.asarray(working, dtype=np.uint8)
    except ReviewError:
        raise
    except Image.DecompressionBombError as exc:
        # Pillow's own ceiling, reached while reading the header.
        raise ReviewError(
            f"the image declares more pixels than are reviewed ({bounds.max_pixels:,})", status=413
        ) from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, EOFError) as exc:
        raise ReviewError(f"not a readable image: {type(exc).__name__}") from exc
    return pixels, {"width": int(width), "height": int(height), "mode": mode}


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #

_WAVE_PCM, _WAVE_FLOAT, _WAVE_EXTENSIBLE = 1, 3, 0xFFFE


def _decode_wav(
    data: bytes, bounds: DecodeBounds
) -> tuple[np.ndarray[Any, Any], int, dict[str, Any]]:
    """Read an uncompressed RIFF/WAVE file: its ``fmt`` chunk, then its ``data`` chunk."""
    fmt: tuple[int, int, int, int] | None = None  # tag, channels, rate, bits
    payload: bytes | None = None
    offset = 12
    for _ in range(_MAX_WAV_CHUNKS):
        if offset + 8 > len(data):
            break
        name, size = data[offset : offset + 4], struct.unpack_from("<I", data, offset + 4)[0]
        body = data[offset + 8 : offset + 8 + size]
        if name == b"fmt " and len(body) >= 16:
            tag, channels, rate, _byte_rate, _align, bits = struct.unpack_from("<HHIIHH", body)
            if tag == _WAVE_EXTENSIBLE and len(body) >= 26:
                tag = struct.unpack_from("<H", body, 24)[0]  # the sub-format's first field
            fmt = (tag, channels, rate, bits)
        elif name == b"data":
            payload = body
            break
        offset += 8 + size + (size & 1)
    if fmt is None or payload is None:
        raise ReviewError("not a readable WAV file: no format or data chunk")
    tag, channels, rate, bits = fmt
    if not (1 <= channels <= _MAX_WAV_CHANNELS and _WAV_RATES[0] <= rate <= _WAV_RATES[1]):
        raise ReviewError(
            f"unsupported WAV layout: {channels} channel(s) at {rate} Hz (reviewed: up to "
            f"{_MAX_WAV_CHANNELS} channels at {_WAV_RATES[0]:,} to {_WAV_RATES[1]:,} Hz)"
        )
    width = bits // 8
    if (tag == _WAVE_PCM and bits in (8, 16, 24, 32)) or (tag == _WAVE_FLOAT and bits in (32, 64)):
        frames = len(payload) // (width * channels)
    else:
        raise ReviewError(
            f"this WAV is encoded (format tag {tag}, {bits} bits) and is not decoded here; "
            "convert it to PCM WAV (for example: ffmpeg -i input -ac 1 -ar 16000 output.wav)",
            status=415,
        )
    # Checked on the header's arithmetic, before a sample is converted.
    if frames / rate > bounds.max_audio_seconds:
        raise ReviewError(
            f"the audio is {frames / rate:.0f} s long; at most {bounds.max_audio_seconds} s "
            "are reviewed",
            status=413,
        )
    body = payload[: frames * width * channels]
    if tag == _WAVE_FLOAT:
        samples = np.frombuffer(body, dtype="<f4" if bits == 32 else "<f8").astype(np.float64)
    elif bits == 8:
        samples = np.frombuffer(body, dtype=np.uint8).astype(np.float64)
        samples -= 128.0
        samples /= 128.0
    elif bits == 24:
        raw = np.frombuffer(body, dtype=np.uint8)
        # The top byte carries the sign.
        values = raw[2::3].astype(np.int8).astype(np.int32) << 16
        values |= raw[1::3].astype(np.int32) << 8
        values |= raw[0::3]
        samples = values.astype(np.float64)
        samples /= float(1 << 23)
    else:
        samples = np.frombuffer(body, dtype="<i2" if bits == 16 else "<i4").astype(np.float64)
        samples /= float(1 << (bits - 1))
    mono = samples if channels == 1 else samples.reshape(-1, channels).mean(axis=1)
    np.nan_to_num(mono, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    np.clip(mono, -1.0, 1.0, out=mono)
    return mono, int(rate), {"channels": int(channels), "bits": int(bits), "seconds": frames / rate}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def decode(
    data: bytes, filename: str = "", *, kind: str | None = None, bounds: DecodeBounds | None = None
) -> Decoded:
    """Decode an upload into text, pixels or samples.

    Args:
        data: The uploaded bytes.
        filename: The upload's name (a hint when the bytes do not decide).
        kind: Require this kind; an upload recognised as another is refused.
        bounds: How far to expand it (defaults to :class:`DecodeBounds`).

    Returns:
        What was recovered.

    Raises:
        ReviewError: With status 413, 415 or 422 and the reason.
    """
    limits = bounds or DecodeBounds()
    found, fmt = sniff(data, filename)
    if kind is not None and kind != found:
        raise ReviewError(f"the upload is a {found} ({fmt}), not a {kind}")
    if found == IMAGE:
        pixels, detail = _decode_image(data, limits)
        return Decoded(IMAGE, fmt, pixels=pixels, detail=detail)
    if found == AUDIO:
        samples, rate, detail = _decode_wav(data, limits)
        return Decoded(AUDIO, fmt, samples=samples, sample_rate=rate, detail=detail)
    reader = {"pdf": _decode_pdf, "docx": _decode_docx, "html": _decode_html}.get(fmt, _decode_text)
    text, detail = reader(data, limits)
    return Decoded(DOCUMENT, fmt, text=text, detail=detail)
