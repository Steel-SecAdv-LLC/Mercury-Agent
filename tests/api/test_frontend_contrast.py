# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The account stylesheet's colour pairs meet WCAG 2.2 AA in both schemes.

Measured on the stylesheet this file replaced: in the dark scheme the danger
button was white on ``#ff6d63`` (2.76:1) and links kept the browser's default
blue on the dark surface (flagged ``color-contrast`` by axe-core); in both
schemes the boundary of an input or a secondary button was the decorative
border colour, 1.26:1 to 1.63:1 against its surroundings where a control
boundary needs 3:1.

The ratios are computed from the custom properties in ``mercury.css`` with
the WCAG relative-luminance formula, for the light block and the
``prefers-color-scheme: dark`` block, so a palette change that breaks a pair
fails here rather than in a browser audit nobody runs.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_CSS = (
    Path(__file__).resolve().parents[2]
    / "src/omni_mercury_engine/api/frontend_assets/static/mercury.css"
).read_text(encoding="utf-8")

_TOKEN = re.compile(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})\s*;")


def _palettes() -> dict[str, dict[str, str]]:
    dark_at = _CSS.index("@media (prefers-color-scheme: dark)")
    light = dict(_TOKEN.findall(_CSS[_CSS.index(":root") : dark_at]))
    dark_block = _CSS[dark_at : _CSS.index("}\n}", dark_at)]
    return {"light": light, "dark": {**light, **dict(_TOKEN.findall(dark_block))}}


def _luminance(colour: str) -> float:
    channels = [int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _ratio(first: str, second: str) -> float:
    lighter, darker = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


#: (foreground token, background token, minimum ratio, what the pair is)
_TEXT = 4.5
_NON_TEXT = 3.0
_PAIRS: tuple[tuple[str, str, float, str], ...] = (
    ("text", "bg", _TEXT, "body text"),
    ("text", "surface", _TEXT, "card text"),
    ("text", "surface-2", _TEXT, "text on a raised surface"),
    ("muted", "bg", _TEXT, "hint text on the page"),
    ("muted", "surface", _TEXT, "hint text in a card"),
    ("muted", "surface-2", _TEXT, "nav link on its hover fill"),
    ("link", "bg", _TEXT, "link on the page"),
    ("link", "surface", _TEXT, "link in a card"),
    ("accent-contrast", "accent", _TEXT, "primary button label"),
    ("accent-contrast", "accent-hover", _TEXT, "primary button label, hovered"),
    ("danger-contrast", "danger-fill", _TEXT, "danger button label"),
    ("danger-contrast", "danger-fill-hover", _TEXT, "danger button label, hovered"),
    ("danger", "tint-danger", _TEXT, "error status"),
    ("ok", "tint-ok", _TEXT, "success status"),
    ("danger", "surface", _TEXT, "error text in a card"),
    ("ok", "surface", _TEXT, "success text in a card"),
    ("control-border", "surface", _NON_TEXT, "input boundary in a card"),
    ("control-border", "bg", _NON_TEXT, "control boundary on the page"),
    ("control-border", "surface-2", _NON_TEXT, "meter boundary against its track"),
    ("focus", "bg", _NON_TEXT, "focus ring on the page"),
    ("focus", "surface", _NON_TEXT, "focus ring in a card"),
    ("accent", "surface-2", _NON_TEXT, "meter fill against its track"),
    ("meter-high", "surface-2", _NON_TEXT, "full meter fill against its track"),
)


@pytest.mark.parametrize("scheme", ["light", "dark"])
@pytest.mark.parametrize(("foreground", "background", "minimum", "what"), _PAIRS)
def test_colour_pair_meets_wcag_aa(
    scheme: str, foreground: str, background: str, minimum: float, what: str
) -> None:
    palette = _palettes()[scheme]
    ratio = _ratio(palette[foreground], palette[background])
    assert ratio >= minimum, (
        f"{scheme}: {what} is --{foreground} {palette[foreground]} on --{background} "
        f"{palette[background]} = {ratio:.2f}:1, below {minimum}:1"
    )


def test_both_schemes_define_every_token_the_pairs_use() -> None:
    used = {token for pair in _PAIRS for token in pair[:2]}
    for scheme, palette in _palettes().items():
        assert used <= set(palette), f"{scheme} is missing {sorted(used - set(palette))}"


def test_links_and_focus_are_styled_not_left_to_the_browser() -> None:
    """The default link blue fails on the dark surface, and buttons had no focus ring."""
    assert re.search(r"\ba\s*\{[^}]*color:\s*var\(--link\)", _CSS)
    assert re.search(r":focus-visible\s*\{[^}]*outline:\s*2px solid var\(--focus\)", _CSS)
    assert "color-scheme: light dark" in _CSS
    assert "prefers-reduced-motion: reduce" in _CSS
