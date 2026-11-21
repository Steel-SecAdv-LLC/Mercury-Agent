# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Small, benign uploads built in memory for the review tests.

Nothing here is read from disk or the network: each builder returns the bytes
of a file a caller might upload, so the decoders are exercised on the real
container formats (a ZIP for DOCX, deflated content streams for PDF, RIFF for
WAV) without committing binaries.
"""

from __future__ import annotations

import io
import struct
import zipfile
import zlib
from typing import Any

import numpy as np

_SENTENCES = (
    "The harbour office opens at eight and the first ferry leaves shortly after.",
    "Most mornings the fog lifts before the bakery has sold its second tray of rolls.",
    "A grey heron stands on the breakwater and watches the tide come round the point.",
    "The lighthouse keeper writes the wind and the swell into a ledger by the window.",
    "Children walk the sea wall to school and count the boats that stayed out overnight.",
    "By noon the market square smells of oranges, wet rope and frying onions.",
    "The afternoon ferry is slower because it stops at the two small islands.",
    "In the evening the harbour lights come on one by one along the quay.",
)


def prose(words: int, seed: int = 0) -> str:
    """Ordinary sentences, at least ``words`` words of them."""
    rng = np.random.default_rng(seed)
    parts: list[str] = []
    count = 0
    while count < words:
        sentence = _SENTENCES[int(rng.integers(len(_SENTENCES)))]
        parts.append(sentence)
        count += len(sentence.split())
    return " ".join(parts)


def ledger(words: int, seed: int = 1) -> str:
    """A block that is plainly not the prose around it: codes and figures."""
    rng = np.random.default_rng(seed)
    return " ".join(
        f"REF-{int(rng.integers(1000, 9999))}:{float(rng.uniform(10, 999)):.2f};"
        for _ in range(words)
    )


def docx(paragraphs: list[str]) -> bytes:
    """A minimal DOCX: one archive member, ``word/document.xml``."""
    body = "".join(
        f'<w:p><w:pPr><w:jc w:val="left"/></w:pPr><w:r><w:t xml:space="preserve">{p}</w:t></w:r></w:p>'
        for p in paragraphs
    )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", xml)
    return buffer.getvalue()


def pdf(lines: list[str], *, deflate: bool = True) -> bytes:
    """A one-page PDF whose content stream draws ``lines`` with ``Tj``."""
    drawn = b"BT /F1 11 Tf 72 720 Td 14 TL\n"
    for line in lines:
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        drawn += f"({escaped}) Tj T*\n".encode("latin-1")
    drawn += b"ET"
    stream = zlib.compress(drawn) if deflate else drawn
    flate = b"/Filter /FlateDecode " if deflate else b""
    return (
        b"%PDF-1.4\n1 0 obj << /Type /Catalog >> endobj\n"
        b"4 0 obj << "
        + flate
        + b"/Length "
        + str(len(stream)).encode()
        + b" >>\nstream\n"
        + stream
        + b"\nendstream\nendobj\ntrailer << /Root 1 0 R >>\n%%EOF\n"
    )


def png(pixels: np.ndarray[Any, Any]) -> bytes:
    """``pixels`` (uint8, HxW or HxWx3) as a PNG file."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def texture(height: int = 256, width: int = 256, seed: int = 0) -> np.ndarray[Any, Any]:
    """A smooth, noisy grayscale texture."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    base = 120 + 40 * np.sin(x / 17.0) * np.cos(y / 23.0)
    noisy = base + rng.normal(scale=6.0, size=(height, width))
    pixels: np.ndarray[Any, Any] = np.clip(noisy, 0, 255).astype(np.uint8)
    return pixels


def wav(
    samples: np.ndarray[Any, Any],
    rate: int = 16_000,
    *,
    bits: int = 16,
    channels: int = 1,
    float_format: bool = False,
    extensible: bool = False,
) -> bytes:
    """``samples`` in [-1, 1] (frames, or frames x channels) as a RIFF/WAVE file."""
    data = np.asarray(samples, dtype=np.float64).reshape(-1, channels)
    if float_format:
        payload = data.astype("<f4" if bits == 32 else "<f8").tobytes()
        tag = 3
    elif bits == 8:
        payload = np.clip(np.round(data * 127.0 + 128.0), 0, 255).astype(np.uint8).tobytes()
        tag = 1
    elif bits == 24:
        ints = np.clip(np.round(data * (2**23 - 1)), -(2**23), 2**23 - 1).astype("<i4")
        payload = ints.tobytes()
        payload = b"".join(payload[i : i + 3] for i in range(0, len(payload), 4))
        tag = 1
    else:
        scale = float(2 ** (bits - 1) - 1)
        payload = np.round(data * scale).astype("<i2" if bits == 16 else "<i4").tobytes()
        tag = 1
    width = bits // 8
    fmt = struct.pack(
        "<HHIIHH", tag, channels, rate, rate * width * channels, width * channels, bits
    )
    if extensible:
        # WAVE_FORMAT_EXTENSIBLE: the real tag is the sub-format's first field.
        fmt = (
            struct.pack(
                "<HHIIHH", 0xFFFE, channels, rate, rate * width * channels, width * channels, bits
            )
            + struct.pack("<HHI", 22, bits, 0)
            + struct.pack("<H", tag)
            + b"\x00" * 14
        )
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", len(body)) + body


def tone(seconds: float, rate: int = 16_000, seed: int = 0) -> np.ndarray[Any, Any]:
    """A wavering, harmonic, slightly noisy tone: a stand-in for a voice."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    pitch = 140 + 25 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(pitch) / rate
    voiced = sum(np.sin(k * phase) / k for k in range(1, 6))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2.3 * t) ** 2
    return np.clip(0.25 * voiced * envelope + rng.normal(scale=0.004, size=t.size), -1, 1)
