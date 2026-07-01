# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Native document generation -- Markdown, HTML, and plain text.

Turns structured content (a title, ordered sections, optional bullet points and
cited sources) into a rendered document. Standard library only (``html.escape``
for safe HTML); no templating engine, no markdown library. Deterministic output
so generated artifacts are reproducible and diffable.
"""

from __future__ import annotations

import html
import urllib.parse
from dataclasses import dataclass, field
from typing import Any


def _md_safe(text: str) -> str:
    """Neutralize raw-HTML passthrough in Markdown content.

    Markdown is not a safe-by-default sink: CommonMark and GitHub-flavored
    renderers pass raw HTML through, so untrusted content (notably
    web-extracted text) containing ``<script>`` or ``<img onerror=...>`` becomes
    scriptable markup when the generated document is displayed in a browser.
    This escapes the three HTML-significant characters (``&``, ``<``, ``>``) so
    an embedded tag renders as literal text. Markdown *structure* is emitted by
    the renderer itself (``#``, ``-``, ``##``) and is never passed through this
    function, so headings, bullets, and front-matter formatting are preserved;
    only the caller-supplied text is defanged. ``quote=False`` keeps quotes
    literal -- they are harmless in Markdown flow text and escaping them would
    corrupt ordinary prose.
    """
    return html.escape(text, quote=False)


def _render_source_link(source: str) -> str:
    """Render one source as safe HTML.

    Only ``http``/``https`` sources become clickable ``<a href>`` links; any
    other scheme (notably ``javascript:``, ``data:``, ``file:``) is rendered as
    plain escaped text. ``html.escape`` alone stops markup injection but does
    NOT neutralize a dangerous URL scheme in an ``href`` -- a ``javascript:``
    link would still execute on click -- so the scheme is allowlisted here.
    """
    e = html.escape
    scheme = urllib.parse.urlparse(source).scheme.lower()
    if scheme in ("http", "https"):
        return f'<a href="{e(source)}">{e(source)}</a>'
    return e(source)


@dataclass
class Section:
    """One document section: a heading, body text, and optional bullet points."""

    heading: str
    body: str = ""
    bullets: list[str] = field(default_factory=list)


@dataclass
class Document:
    """A rendered document plus the structured content it was built from."""

    title: str
    fmt: str
    content: str
    sections: list[Section] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """The rendered document body."""
        return self.content


class DocumentGenerator:
    """Render structured content to Markdown / HTML / plain text.

    Example::

        gen = DocumentGenerator()
        doc = gen.report(
            "Findings",
            sections=[Section("Summary", "...", bullets=["a", "b"])],
            sources=["https://example.org/a"],
            fmt="markdown",
        )
        print(doc.content)
    """

    SUPPORTED_FORMATS = ("markdown", "html", "text")

    def report(
        self,
        title: str,
        sections: list[Section] | list[tuple[str, str]] | list[dict[str, Any]],
        *,
        fmt: str = "markdown",
        metadata: dict[str, str] | None = None,
        sources: list[str] | None = None,
    ) -> Document:
        """Build and render a document.

        Args:
            title: Document title.
            sections: Sections as :class:`Section`, ``(heading, body)`` tuples,
                or ``{"heading", "body", "bullets"}`` dicts.
            fmt: ``"markdown"`` (default), ``"html"``, or ``"text"``.
            metadata: Optional key/value front-matter (e.g. author, generated_at).
            sources: Optional list of source URLs/citations appended as a
                "Sources" section.

        Raises:
            ValueError: If ``fmt`` is not supported.
        """
        if fmt not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format {fmt!r}; choose one of {self.SUPPORTED_FORMATS}")
        norm = [self._coerce_section(s) for s in sections]
        meta = dict(metadata or {})
        srcs = list(sources or [])
        renderer = {
            "markdown": self._render_markdown,
            "html": self._render_html,
            "text": self._render_text,
        }[fmt]
        content = renderer(title, norm, meta, srcs)
        return Document(
            title=title, fmt=fmt, content=content, sections=norm, metadata=meta, sources=srcs
        )

    @staticmethod
    def _coerce_section(s: Section | tuple[Any, ...] | dict[str, Any]) -> Section:
        if isinstance(s, Section):
            return s
        if isinstance(s, tuple):
            heading, body = (list(s) + ["", ""])[:2]
            return Section(heading=str(heading), body=str(body))
        if isinstance(s, dict):
            return Section(
                heading=str(s.get("heading", "")),
                body=str(s.get("body", "")),
                bullets=[str(b) for b in s.get("bullets", [])],
            )
        raise TypeError(f"section must be Section/tuple/dict, got {type(s).__name__}")

    # -- renderers ---------------------------------------------------------

    @staticmethod
    def _render_markdown(
        title: str, sections: list[Section], meta: dict[str, str], sources: list[str]
    ) -> str:
        # Content fields are HTML-neutralized (see _md_safe): the values may
        # originate from untrusted web-extracted text, and Markdown renderers
        # pass raw HTML through. Structural tokens ('#', '-', numbering) are
        # emitted here, not escaped, so document formatting is unaffected.
        s = _md_safe
        lines = [f"# {s(title)}", ""]
        if meta:
            for k, v in meta.items():
                lines.append(f"- **{s(str(k))}**: {s(str(v))}")
            lines.append("")
        for sec in sections:
            lines.append(f"## {s(sec.heading)}")
            lines.append("")
            if sec.body:
                lines.append(s(sec.body))
                lines.append("")
            for b in sec.bullets:
                lines.append(f"- {s(b)}")
            if sec.bullets:
                lines.append("")
        if sources:
            lines.append("## Sources")
            lines.append("")
            for i, src in enumerate(sources, 1):
                lines.append(f"{i}. {s(src)}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _render_html(
        title: str, sections: list[Section], meta: dict[str, str], sources: list[str]
    ) -> str:
        e = html.escape
        out = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{e(title)}</title>",
            "</head>",
            "<body>",
        ]
        out.append(f"<h1>{e(title)}</h1>")
        if meta:
            out.append("<dl>")
            for k, v in meta.items():
                out.append(f"<dt>{e(str(k))}</dt><dd>{e(str(v))}</dd>")
            out.append("</dl>")
        for sec in sections:
            out.append(f"<h2>{e(sec.heading)}</h2>")
            if sec.body:
                out.append(f"<p>{e(sec.body)}</p>")
            if sec.bullets:
                out.append("<ul>")
                out.extend(f"<li>{e(b)}</li>" for b in sec.bullets)
                out.append("</ul>")
        if sources:
            out.append("<h2>Sources</h2>")
            out.append("<ol>")
            out.extend(f"<li>{_render_source_link(s)}</li>" for s in sources)
            out.append("</ol>")
        out.extend(["</body>", "</html>"])
        return "\n".join(out) + "\n"

    @staticmethod
    def _render_text(
        title: str, sections: list[Section], meta: dict[str, str], sources: list[str]
    ) -> str:
        lines = [title, "=" * len(title), ""]
        if meta:
            for k, v in meta.items():
                lines.append(f"{k}: {v}")
            lines.append("")
        for sec in sections:
            lines.append(sec.heading)
            lines.append("-" * len(sec.heading))
            if sec.body:
                lines.append(sec.body)
            for b in sec.bullets:
                lines.append(f"  * {b}")
            lines.append("")
        if sources:
            lines.append("Sources")
            lines.append("-------")
            for i, src in enumerate(sources, 1):
                lines.append(f"[{i}] {src}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


__all__ = ["Document", "DocumentGenerator", "Section"]
