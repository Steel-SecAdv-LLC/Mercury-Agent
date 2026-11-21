# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The console's themes and markup meet WCAG 2.1 AA where they were measured not to.

An axe-core pass over every console workspace, tab and dialog in the dark,
grey and light themes (81 views) found 190 distinct failures: text under
4.5:1 on raised and tinted surfaces (2.5:1 for the primary button's label in
the light theme), radios and checkboxes that lost ``aria-checked`` when not
chosen, scrolling blocks the keyboard could not reach, and no level-one
heading once the opening screen was dismissed.

These tests pin the causes: each token against each surface the console sets
it on, the runtime's handling of a false ARIA state, and the generated
markup. The in-browser self-test (``:diag``) checks the rendered page.
"""

from __future__ import annotations

import re

import pytest

from omni_mercury_engine.api import frontend

CONSOLE = frontend._STATIC_DIR / "console"


def _blocks(css: str, selector: str) -> dict[str, str]:
    """Custom properties declared for exactly ``selector``, later blocks winning."""
    tokens: dict[str, str] = {}
    for body in re.findall(
        rf"(?:^|\}}|\*/)\s*{re.escape(selector)}\s*\{{([^}}]*)\}}", css, flags=re.M
    ):
        for name, value in re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", body):
            tokens[name] = value.strip()
    return tokens


@pytest.fixture(scope="module")
def themes() -> dict[str, dict[str, str]]:
    css = re.sub(r"/\*.*?\*/", "", (CONSOLE / "tokens.css").read_text(encoding="utf-8"), flags=re.S)
    dark = _blocks(css, ":root")
    return {
        "dark": dark,
        "grey": {**dark, **_blocks(css, "html.ds-grey")},
        "light": {**dark, **_blocks(css, "html.ds-light")},
    }


def _luminance(colour: str) -> float:
    channels = [int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(first: str, second: str) -> float:
    lighter, darker = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


#: (theme, text token, ground token): every pairing axe measured under 4.5:1.
MEASURED_PAIRS = [
    ("dark", "--fg-3", "--surface-2"),
    ("dark", "--fg-3", "--surface-3"),
    ("dark", "--brand-ink-grey", "--surface-3"),
    ("dark", "--brand-ink-grey", "--mercury-900"),
    ("grey", "--fg-3", "--surface-2"),
    ("grey", "--fg-3", "--surface-3"),
    ("grey", "--brand-cyan", "--surface-1"),
    ("grey", "--brand-cyan", "--surface-2"),
    ("grey", "--brand-cyan", "--surface-3"),
    ("grey", "--mercury-300", "--surface-3"),
    ("grey", "--cognitive", "--surface-1"),
    ("grey", "--brand-ink-grey", "--surface-1"),
    ("grey", "--brand-ink-grey", "--surface-2"),
    ("grey", "--brand-ink-grey", "--surface-3"),
    ("grey", "--normal", "--surface-3"),
    ("grey", "--warning", "--surface-3"),
    ("light", "--mercury-400", "--mercury-900"),
    ("light", "--mercury-400", "--surface-3"),
    ("light", "--mercury-400", "--void"),
    ("light", "--mercury-400", "--surface-2"),
    ("light", "--mercury-400", "--surface-1"),
    ("light", "--brand-cyan", "--void"),
    ("light", "--brand-cyan", "--mercury-900"),
    ("light", "--mercury-300", "--surface-3"),
    ("light", "--normal", "--void"),
    ("light", "--normal", "--surface-2"),
    ("light", "--normal", "--surface-1"),
    ("light", "--warning", "--void"),
    ("light", "--warning", "--surface-1"),
    ("light", "--brand-gold", "--void"),
    ("light", "--void", "--mercury-500"),  # the primary button's label
    ("light", "--void", "--normal"),  # the RUN button's label
]


@pytest.mark.parametrize(("theme", "text", "ground"), MEASURED_PAIRS)
def test_text_tokens_reach_aa_on_the_grounds_the_console_sets_them_on(
    themes: dict[str, dict[str, str]], theme: str, text: str, ground: str
) -> None:
    tokens = themes[theme]

    ratio = _contrast(tokens[text], tokens[ground])

    assert (
        ratio >= 4.5
    ), f"{theme}: {text} {tokens[text]} on {ground} {tokens[ground]} is {ratio:.2f}:1"


def test_the_grey_and_light_themes_keep_their_text_hierarchy(
    themes: dict[str, dict[str, str]],
) -> None:
    """Raising the faintest text must not make it the same as the tier above it."""
    for theme in ("dark", "grey", "light"):
        tokens = themes[theme]
        ground = tokens["--surface-1"]
        tiers = [_contrast(tokens[name], ground) for name in ("--fg", "--fg-2", "--fg-3")]
        assert tiers[0] > tiers[1] > tiers[2] >= 4.5, (theme, tiers)


def test_a_false_aria_state_is_written_not_dropped() -> None:
    """Was: ``aria-checked={false}`` removed the attribute a radio is required to carry."""
    runtime = (CONSOLE / "dc-runtime.js").read_text(encoding="utf-8")

    states = re.search(r"const ARIA_STATE = new Set\(\[([^\]]+)\]\)", runtime)
    assert states is not None
    assert {'"aria-checked"', '"aria-selected"', '"aria-pressed"', '"aria-expanded"'} <= set(
        states.group(1).split(", ")
    )
    write = runtime.index(
        'if (nv === false && ARIA_STATE.has(attr)) { el.setAttribute(attr, "false");'
    )
    drop = runtime.index("if (nv == null || nv === false) { el.removeAttribute(attr); continue; }")
    assert write < drop  # the state is written before the general rule can drop it


def test_scrolling_blocks_take_focus_and_the_page_has_a_heading() -> None:
    template = (CONSOLE / "template.js").read_text(encoding="utf-8")

    scrolling = re.findall(r'\["pre",\{([^}]*overflow:auto[^}]*)\}', template)
    assert len(scrolling) >= 6
    assert all('"tabindex":"0"' in attributes for attributes in scrolling)
    assert '["h1",{"class":"mcli-sr"},["Mercury Console"]]' in template
    assert ".mcli-sr {" in (CONSOLE / "tokens.css").read_text(encoding="utf-8")


def test_text_over_the_agent_figure_uses_the_figures_ground_not_the_themes() -> None:
    """Was: the light theme's dark teal (2.8:1) on the figure's always-dark canvas."""
    template = (CONSOLE / "template.js").read_text(encoding="utf-8")

    caption = re.search(r'\["span",\{"id":"mcli-presence-cap"[^}]*\}', template)
    assert caption is not None
    colour = re.search(r"(?<![\w-])color:(#[0-9A-Fa-f]{6})", caption.group(0))
    assert colour is not None, caption.group(0)
    assert _contrast(colour.group(1), "#050a18") >= 4.5


def test_the_watch_tab_is_compiled_into_the_console() -> None:
    logic = (CONSOLE / "logic.js").read_text(encoding="utf-8")
    template = (CONSOLE / "template.js").read_text(encoding="utf-8")
    diag = (CONSOLE / "diag.js").read_text(encoding="utf-8")

    assert '["watch", "WATCH",' in logic and 'this.A.get("/api/v1/watch?limit=25")' in logic
    assert '"value":"{{ agWatch }}"' in template and "{{ agWatchRecord }}" in template
    assert "innerHTML" not in logic.split("async watchRefresh()")[1].split("watchSync()")[0]
    assert 'for (const tb of ["act", "arch", "fleet", "trust", "watch"])' in diag
