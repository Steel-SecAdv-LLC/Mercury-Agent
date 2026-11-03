# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The CLI voice session shows what an answer is built from and what to ask next.

The loop printed the message alone, so the sources and follow-up prompts the
voice returns never reached a terminal user, and ``/clear`` printed
"cleared" without clearing anything.
"""

from __future__ import annotations

from click.testing import CliRunner

from omni_mercury_engine.cli import main


def test_voice_session_prints_sources_follow_ups_and_really_clears() -> None:
    """Three turns are cleared: the greeting, the question and the answer."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["voice", "--offline", "--domain", "general"],
        input="What is a z-score?\n/clear\n/quit\n",
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "A z-score is how many standard deviations" in result.output
    assert "[Sources: code]" in result.output
    assert "Ask next:" in result.output
    assert "[Conversation history cleared: 3 turn(s)]" in result.output
    assert "Goodbye" in result.output
