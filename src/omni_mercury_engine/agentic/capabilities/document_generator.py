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
from dataclasses import dataclass, field


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
        sections: list[Section] | list[tuple[str, str]] | list[dict],
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
        return Document(title=title, fmt=fmt, content=content, sections=norm, metadata=meta, sources=srcs)

    @staticmethod
    def _coerce_section(s: Section | tuple | dict) -> Section:
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
    def _render_markdown(title: str, sections: list[Section], meta: dict, sources: list[str]) -> str:
        lines = [f"# {title}", ""]
        if meta:
            for k, v in meta.items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")
        for sec in sections:
            lines.append(f"## {sec.heading}")
            lines.append("")
            if sec.body:
                lines.append(sec.body)
                lines.append("")
            for b in sec.bullets:
                lines.append(f"- {b}")
            if sec.bullets:
                lines.append("")
        if sources:
            lines.append("## Sources")
            lines.append("")
            for i, src in enumerate(sources, 1):
                lines.append(f"{i}. {src}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _render_html(title: str, sections: list[Section], meta: dict, sources: list[str]) -> str:
        e = html.escape
        out = ["<!DOCTYPE html>", "<html>", "<head>", f"<title>{e(title)}</title>", "</head>", "<body>"]
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
            out.extend(f'<li><a href="{e(s)}">{e(s)}</a></li>' for s in sources)
            out.append("</ol>")
        out.extend(["</body>", "</html>"])
        return "\n".join(out) + "\n"

    @staticmethod
    def _render_text(title: str, sections: list[Section], meta: dict, sources: list[str]) -> str:
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
