# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The upload decoders: what they read, what they refuse, and how much work they do.

An upload is whatever a caller sends.  These tests pin three things about
:mod:`omni_mercury_engine.review.decode`: each format decodes to what was put
in; anything past a bound or outside the supported formats is refused with
the status that says why (413 / 415 / 422), never reviewed in part; and no
input can make a reader rescan itself (each is one linear pass).
"""

from __future__ import annotations

import io
import struct
import time
import zipfile

import numpy as np
import pytest

from omni_mercury_engine.review.decode import (
    DecodeBounds,
    ReviewError,
    _docx_text,
    decode,
    sniff,
)
from omni_mercury_engine.serving_limits import load_serving_limits
from tests.review import builders

#: Generous: a linear reader finishes these in well under a second; the
#: quadratic ones they replace took minutes on the same inputs.
LINEAR_SECONDS = 15.0


def _refused(data: bytes, name: str = "", **kwargs: object) -> ReviewError:
    with pytest.raises(ReviewError) as caught:
        decode(data, name, **kwargs)  # type: ignore[arg-type]
    return caught.value


class TestRecognition:
    @pytest.mark.parametrize(
        ("data", "name", "expected"),
        [
            (b"\x89PNG\r\n\x1a\n" + b"\x00" * 8, "", ("image", "png")),
            (b"\xff\xd8\xff\xe0" + b"\x00" * 8, "", ("image", "jpeg")),
            (b"GIF89a" + b"\x00" * 8, "", ("image", "gif")),
            (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "", ("image", "webp")),
            (b"RIFF\x00\x00\x00\x00WAVEfmt ", "", ("audio", "wav")),
            (b"%PDF-1.7\n", "", ("document", "pdf")),
            (b"<!DOCTYPE html><html></html>", "", ("document", "html")),
            (b"plain words on a line\n", "notes.txt", ("document", "text")),
            (b"a,b\n1,2\n", "table.csv", ("document", "text")),
        ],
    )
    def test_an_upload_is_recognised_by_its_bytes_then_its_name(
        self, data: bytes, name: str, expected: tuple[str, str]
    ) -> None:
        assert sniff(data, name) == expected

    def test_the_bytes_decide_over_a_misleading_name(self) -> None:
        assert sniff(b"%PDF-1.4\n", "holiday.png") == ("document", "pdf")

    @pytest.mark.parametrize(
        "head",
        [
            b"ID3\x04\x00",
            b"\xff\xfb\x90\x00",
            b"OggS\x00",
            b"fLaC\x00",
            b"\x00\x00\x00\x20ftypM4A ",
        ],
    )
    def test_compressed_audio_is_refused_with_the_conversion_to_run(self, head: bytes) -> None:
        refusal = _refused(head + b"\x00" * 64, "voice.bin")
        assert refusal.status == 415
        assert "ffmpeg" in str(refusal)

    def test_an_archive_that_is_not_a_docx_is_refused(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("notes.txt", "hello")
        assert _refused(buffer.getvalue(), "bundle.zip").status == 415

    def test_unrecognisable_bytes_are_refused(self) -> None:
        assert _refused(bytes(range(256)) * 8, "blob.bin").status == 415

    def test_an_empty_upload_is_refused(self) -> None:
        assert _refused(b"", "empty.txt").status == 422

    def test_a_required_kind_refuses_another(self) -> None:
        refusal = _refused(b"some words\n", "a.txt", kind="image")
        assert refusal.status == 422
        assert "document" in str(refusal)


class TestDocuments:
    def test_utf8_utf16_and_latin1_text_decode(self) -> None:
        text = "café on the quay, every morning\n" * 3
        assert decode(text.encode("utf-8"), "a.txt").text == text
        assert decode(b"\xff\xfe" + text.encode("utf-16-le"), "a.txt").text == text
        assert decode(text.encode("latin-1"), "a.txt").text == text

    def test_html_keeps_what_a_reader_sees(self) -> None:
        page = (
            b"<html><head><title>t</title><style>p{color:red}</style></head><body>"
            b"<h1>Harbour</h1><p>The ferry leaves at eight &amp; returns at six.</p>"
            b"<script>var hidden = 'not shown';</script></body></html>"
        )
        text = decode(page, "page.html").text or ""
        assert "Harbour" in text
        assert "The ferry leaves at eight & returns at six." in text
        assert "hidden" not in text and "color" not in text

    def test_docx_text_is_its_paragraphs(self) -> None:
        decoded = decode(builders.docx(["First paragraph &amp; more.", "Second one."]), "a.docx")
        assert decoded.format == "docx"
        assert decoded.text == "First paragraph & more.\n\nSecond one."
        assert decoded.detail == {"paragraphs": 2}

    def test_docx_tags_that_only_start_like_a_paragraph_are_not_paragraphs(self) -> None:
        xml = (
            "<w:tbl><w:tr><w:tc><w:p><w:pPr/><w:r><w:t>cell</w:t><w:tab/><w:t>two</w:t>"
            "<w:br/><w:t/></w:r></w:p></w:tc></w:tr></w:tbl>"
        )
        assert _docx_text(xml) == ("cell\ttwo\n", 1)

    @pytest.mark.parametrize("deflate", [True, False])
    def test_pdf_text_operators_are_read(self, deflate: bool) -> None:
        lines = ["The harbour office opens at eight.", "Tides (spring) run higher \\ later."] * 3
        decoded = decode(builders.pdf(lines, deflate=deflate), "a.pdf")
        assert decoded.format == "pdf"
        assert (decoded.text or "").splitlines()[:2] == lines[:2]

    def test_a_pdf_with_no_text_layer_is_refused_not_approximated(self) -> None:
        scanned = b"%PDF-1.4\n4 0 obj << /Length 9 >>\nstream\n\x00\x01\x02\x03\x04\x05\x06\x07\x08\nendstream\nendobj\n%%EOF"
        refusal = _refused(scanned, "scan.pdf")
        assert refusal.status == 422
        assert "no readable text" in str(refusal)


class TestBoundsAreRefusalsNotTruncation:
    """Past a bound an upload is refused with 413; it is never reviewed in part."""

    def test_the_defaults_are_the_served_defaults(self) -> None:
        served = load_serving_limits()
        assert DecodeBounds() == DecodeBounds(
            max_text_chars=served.review_max_text_chars,
            max_pixels=served.review_max_pixels,
            max_audio_seconds=served.review_max_audio_seconds,
        )

    def test_the_text_bound_sits_under_what_the_harm_gate_reads(self) -> None:
        from omni_mercury_engine.cognitive.decision_gate import MAX_ASSESSED_CHARS

        assert DecodeBounds().max_text_chars + 4096 < MAX_ASSESSED_CHARS

    @pytest.mark.parametrize(
        "upload",
        [
            (builders.prose(400).encode(), "a.txt"),
            (b"<html><body><p>" + builders.prose(400).encode() + b"</p></body></html>", "a.html"),
            (builders.docx([builders.prose(400)]), "a.docx"),
            (builders.pdf([builders.prose(40, seed=i) for i in range(10)]), "a.pdf"),
        ],
        ids=["text", "html", "docx", "pdf"],
    )
    def test_a_document_over_the_text_bound_is_refused(self, upload: tuple[bytes, str]) -> None:
        data, name = upload
        assert decode(data, name).text  # within the default bound: read
        refusal = _refused(data, name, bounds=DecodeBounds(max_text_chars=200))
        assert refusal.status == 413
        assert "200" in str(refusal)

    def test_an_image_over_the_pixel_bound_is_refused_on_its_header(self) -> None:
        data = builders.png(builders.texture(64, 64))
        assert decode(data).pixels is not None
        refusal = _refused(data, bounds=DecodeBounds(max_pixels=64 * 64 - 1))
        assert refusal.status == 413

    def test_audio_over_the_duration_bound_is_refused(self) -> None:
        data = builders.wav(builders.tone(3.0))
        assert decode(data).samples is not None
        refusal = _refused(data, bounds=DecodeBounds(max_audio_seconds=2))
        assert refusal.status == 413
        assert "2 s" in str(refusal)


class TestImages:
    def test_grayscale_and_colour_decode_to_their_pixels(self) -> None:
        gray = builders.texture(48, 64)
        decoded = decode(builders.png(gray))
        assert decoded.kind == "image" and decoded.format == "png"
        assert decoded.pixels is not None and np.array_equal(decoded.pixels, gray)
        colour = np.stack([gray, gray[::-1], gray[:, ::-1]], axis=2)
        tinted = decode(builders.png(colour)).pixels
        assert tinted is not None and np.array_equal(tinted, colour)

    def test_a_large_image_is_reduced_as_it_is_decoded(self) -> None:
        """A photograph is never held as a full-size array; its own size is kept on record."""
        from omni_mercury_engine.review.decode import IMAGE_WORKING_SIDE

        decoded = decode(builders.png(builders.texture(900, 1800)))
        assert decoded.pixels is not None
        assert decoded.pixels.shape == (IMAGE_WORKING_SIDE // 2, IMAGE_WORKING_SIDE)
        assert decoded.detail == {"width": 1800, "height": 900, "mode": "L"}
        small = decode(builders.png(builders.texture(300, 400)))
        assert small.pixels is not None and small.pixels.shape == (300, 400)

    def test_a_truncated_image_is_refused(self) -> None:
        data = builders.png(builders.texture(64, 64))
        assert _refused(data[: len(data) // 2]).status == 422


class TestAudio:
    @pytest.mark.parametrize(
        ("bits", "float_format", "tolerance"),
        [
            (8, False, 1 / 100),
            (16, False, 1e-4),
            (24, False, 1e-6),
            (32, False, 1e-8),
            (32, True, 1e-6),
            (64, True, 1e-12),
        ],
    )
    def test_every_supported_sample_format_decodes_to_the_signal(
        self, bits: int, float_format: bool, tolerance: float
    ) -> None:
        signal = builders.tone(0.5)
        decoded = decode(builders.wav(signal, bits=bits, float_format=float_format))
        assert decoded.sample_rate == 16_000
        assert decoded.samples is not None and decoded.samples.shape == signal.shape
        assert float(np.max(np.abs(decoded.samples - signal))) <= tolerance

    def test_channels_are_averaged_to_one(self) -> None:
        left = builders.tone(0.25)
        stereo = np.stack([left, -left], axis=1)
        decoded = decode(builders.wav(stereo, channels=2))
        assert decoded.samples is not None
        assert float(np.max(np.abs(decoded.samples))) <= 1e-4
        assert decoded.detail is not None and decoded.detail["channels"] == 2

    def test_the_extensible_header_is_read_by_its_sub_format(self) -> None:
        signal = builders.tone(0.25)
        decoded = decode(builders.wav(signal, extensible=True))
        assert decoded.samples is not None
        assert float(np.max(np.abs(decoded.samples - signal))) <= 1e-4

    def test_an_encoded_wav_is_refused(self) -> None:
        fmt = struct.pack("<HHIIHH", 0x0055, 1, 16_000, 2000, 1, 0)  # MPEG layer 3 in a RIFF
        body = (
            b"WAVEfmt "
            + struct.pack("<I", len(fmt))
            + fmt
            + b"data"
            + struct.pack("<I", 4)
            + b"\x00" * 4
        )
        refusal = _refused(b"RIFF" + struct.pack("<I", len(body)) + body)
        assert refusal.status == 415

    @pytest.mark.parametrize(("rate", "channels"), [(3_999, 1), (192_001, 1), (16_000, 9)])
    def test_a_layout_outside_what_is_reviewed_is_refused(self, rate: int, channels: int) -> None:
        signal = np.zeros((rate // 10) * channels)
        refusal = _refused(builders.wav(signal, rate=rate, channels=channels))
        assert refusal.status == 422
        assert "unsupported WAV layout" in str(refusal)

    def test_a_wav_with_no_data_is_refused(self) -> None:
        assert _refused(b"RIFF\x04\x00\x00\x00WAVE").status == 422


class TestEveryReaderIsOneLinearPass:
    """Inputs built to make a careless pattern rescan the file once per tag.

    The first draft of the decoder used a lazy ``.*?`` between an opening
    and a closing token; when the closing token never arrives, every opening
    one scans to the end of the file.  Measured on that draft: 20,000 unclosed
    tags took 15 s (DOCX) and 34 s (PDF), and four times that per doubling.
    The inputs below are twenty to thirty times larger.
    """

    def _timed(self, data: bytes, name: str) -> float:
        started = time.perf_counter()
        try:
            decode(data, name)
        except ReviewError:
            pass
        return time.perf_counter() - started

    @pytest.mark.timeout(120)
    def test_docx_paragraphs_that_never_close(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("word/document.xml", "<w:p><w:t>" * 400_000)
        assert self._timed(buffer.getvalue(), "a.docx") < LINEAR_SECONDS

    @pytest.mark.timeout(120)
    def test_docx_tags_that_never_end(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("word/document.xml", "<w:t " * 800_000)
        assert self._timed(buffer.getvalue(), "a.docx") < LINEAR_SECONDS

    @pytest.mark.timeout(120)
    def test_pdf_streams_that_never_end(self) -> None:
        assert self._timed(b"%PDF-1.4\n" + b"stream\n" * 600_000, "a.pdf") < LINEAR_SECONDS

    @pytest.mark.timeout(120)
    def test_pdf_strings_that_never_close(self) -> None:
        content = b"BT " + b"(" * 300_000 + b" Tj ET"
        data = b"%PDF-1.4\nstream\n" + content + b"\nendstream\n"
        assert self._timed(data, "a.pdf") < LINEAR_SECONDS

    @pytest.mark.timeout(120)
    def test_wav_chunks_without_end(self) -> None:
        chunks = (b"junk" + struct.pack("<I", 0)) * 500_000
        assert self._timed(b"RIFF\x00\x00\x00\x00WAVE" + chunks, "a.wav") < LINEAR_SECONDS

    def test_a_docx_member_is_read_through_a_limit_whatever_it_declares(self) -> None:
        """A small archive that inflates far past the bound is refused, not expanded."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("word/document.xml", "<w:p><w:r><w:t>" + "a" * 4_000_000)
        data = buffer.getvalue()
        assert len(data) < 50_000  # the bomb is small on the wire
        assert _refused(data, "a.docx", bounds=DecodeBounds(max_text_chars=10_000)).status == 413

    def test_a_pdf_stream_is_inflated_only_within_the_budget(self) -> None:
        import zlib

        stream = zlib.compress(b"BT (" + b"a" * 8_000_000 + b") Tj ET")
        data = b"%PDF-1.4\nstream\n" + stream + b"\nendstream\nstream\nBT (x) Tj ET\nendstream\n"
        assert len(data) < 50_000
        assert _refused(data, "a.pdf", bounds=DecodeBounds(max_text_chars=10_000)).status == 413
