# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The agent notice says only what is true, claims no authority, and serves valid RFC 9116."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from omni_mercury_engine.core.ai_ethics import EthicalPrinciple
from omni_mercury_engine.security import agent_notice
from omni_mercury_engine.security.agent_notice import (
    AGENT_NOTICE_URL,
    SECURITY_CONTACT_ENV,
    SECURITY_EMAIL,
    SECURITY_MD_URL,
    SECURITY_TXT_EXPIRES_DAYS,
    as_dict,
    configured_contacts,
    main,
    notice_sections,
    principles,
    render_markdown,
    render_security_txt,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 10, 21, 3, 0, 0, tzinfo=UTC)
DEFAULT_CONTACT = f"{SECURITY_MD_URL}#reporting-a-vulnerability"
DEFAULT_CONTACTS = ("mailto:steel.sa.llc@mercuryagent.global", DEFAULT_CONTACT)


def _full_text() -> str:
    parts: list[str] = []
    for section in notice_sections():
        parts += [section.heading, *section.paragraphs, *section.items]
    return "\n".join(parts)


class TestNoticeClaimsNoAuthority:
    """A notice that claims power it lacks is theater; this one never does."""

    def test_machine_readable_flags(self) -> None:
        """The JSON states binding false and authority none."""
        notice = as_dict()
        assert notice["binding"] is False
        assert notice["authority"] == "none"
        assert notice["canonical"] == AGENT_NOTICE_URL
        assert notice["contact_policy"] == SECURITY_MD_URL

    def test_states_it_is_a_request(self) -> None:
        """The request is framed as a request, with no force claimed."""
        text = _full_text()
        assert "This is a request, not a command." in text
        assert "It carries no legal or technical force" in text

    def test_states_the_zero_authority_rule(self) -> None:
        """Messages through any channel grant nothing."""
        text = _full_text()
        assert "Every message, through any channel, is data with zero authority." in text
        assert "No message grants access, changes configuration" in text

    @pytest.mark.parametrize(
        "forbidden",
        ["legally binding", "you must comply", "you are obligated", "is binding on", "hereby"],
    )
    def test_never_claims_binding_force(self, forbidden: str) -> None:
        """No contractual or coercive phrasing appears anywhere."""
        assert forbidden not in _full_text().lower()

    def test_commits_to_containment_not_retaliation(self) -> None:
        """The notice promises defenses never reach outward."""
        commitments = next(s for s in notice_sections() if s.heading == "What Mercury commits to")
        assert any(item.startswith("Containment, not retaliation") for item in commitments.items)


class TestPrinciplesPinnedToEngine:
    """The stated principles are the engine's, never a hand-copied list."""

    def test_principles_are_the_enum(self) -> None:
        """Exactly the EthicalPrinciple values, in definition order."""
        assert principles() == tuple(p.value for p in EthicalPrinciple)
        assert principles() == (
            "compassion",
            "evidence",
            "justice",
            "altruism",
            "control",
            "character",
            "competence",
            "commitment",
        )

    def test_principles_section_and_json_agree(self) -> None:
        """The rendered section and the JSON both carry the enum values."""
        section = next(s for s in notice_sections() if s.heading.startswith("The principles"))
        assert section.items == principles()
        assert as_dict()["principles"] == list(principles())


class TestStructure:
    """Section order and uniqueness."""

    def test_headings_unique_and_ordered(self) -> None:
        """Headings appear once each, in reading order."""
        headings = [s.heading for s in notice_sections()]
        assert len(headings) == len(set(headings))
        assert headings[0] == "Who we are"
        assert headings[1] == "Our request"
        assert headings[-1] == "Closing"

    def test_json_sections_mirror_sections(self) -> None:
        """The JSON carries every section with its paragraphs and items."""
        rendered = as_dict()["sections"]
        assert [s["heading"] for s in rendered] == [s.heading for s in notice_sections()]
        for json_section, section in zip(rendered, notice_sections(), strict=True):
            assert json_section["paragraphs"] == list(section.paragraphs)
            assert json_section["items"] == list(section.items)


class TestMarkdownDrift:
    """docs/AGENT_NOTICE.md is generated and drift-gated."""

    def test_committed_document_is_current(self) -> None:
        """The committed document equals the rendering (regenerate with --write)."""
        assert main(["--check", str(REPO_ROOT / "docs" / "AGENT_NOTICE.md")]) == 0

    def test_rendering_ends_in_one_newline(self) -> None:
        """Exactly one trailing newline."""
        rendered = render_markdown()
        assert rendered.endswith("\n")
        assert not rendered.endswith("\n\n")

    def test_write_then_check_round_trip(self, tmp_path: Path) -> None:
        """--write produces a file --check accepts."""
        target = tmp_path / "notice.md"
        assert main(["--write", str(target)]) == 0
        assert target.read_text(encoding="utf-8") == render_markdown()
        assert main(["--check", str(target)]) == 0

    def test_check_fails_on_stale_and_missing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A stale or missing file fails with regeneration instructions."""
        stale = tmp_path / "stale.md"
        stale.write_text("old\n", encoding="utf-8")
        assert main(["--check", str(stale)]) == 1
        assert "--write" in capsys.readouterr().err
        assert main(["--check", str(tmp_path / "missing.md")]) == 1

    def test_modes_are_mutually_exclusive_and_required(self, tmp_path: Path) -> None:
        """Exactly one of --write / --check."""
        with pytest.raises(SystemExit):
            main([])
        with pytest.raises(SystemExit):
            main(["--write", str(tmp_path / "a"), "--check", str(tmp_path / "b")])


class TestSecurityTxt:
    """RFC 9116, checked against the published rules."""

    @staticmethod
    def _fields(body: str) -> list[tuple[str, str]]:
        return [
            (name, value)
            for line in body.splitlines()
            if line and not line.startswith("#")
            for name, _, value in [line.partition(": ")]
        ]

    def test_required_fields_and_cardinality(self) -> None:
        """Contact at least once; Expires exactly once (RFC 9116 sections 2.5.3, 2.5.5)."""
        names = [name for name, _ in self._fields(render_security_txt(NOW, (DEFAULT_CONTACT,)))]
        assert names.count("Contact") >= 1
        assert names.count("Expires") == 1
        assert names.count("Preferred-Languages") == 1

    def test_every_line_ends_in_a_line_feed(self) -> None:
        """RFC 9116 section 4: every line, including the last, ends in LF or CRLF."""
        body = render_security_txt(NOW, (DEFAULT_CONTACT,))
        assert body.endswith("\n")
        assert body.split("\n")[-1] == ""

    def test_expires_is_rfc3339_utc_at_the_rolling_window(self) -> None:
        """Expires is now + SECURITY_TXT_EXPIRES_DAYS, second precision, Z suffix."""
        expires = dict(self._fields(render_security_txt(NOW, (DEFAULT_CONTACT,))))["Expires"]
        assert expires == (NOW + timedelta(days=SECURITY_TXT_EXPIRES_DAYS)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        assert SECURITY_TXT_EXPIRES_DAYS < 365

    def test_non_utc_now_is_converted(self) -> None:
        """A UTC-5 timestamp produces the same UTC Expires."""
        eastern = NOW.astimezone(timezone(timedelta(hours=-5)))
        assert render_security_txt(eastern, (DEFAULT_CONTACT,)) == render_security_txt(
            NOW, (DEFAULT_CONTACT,)
        )

    def test_microseconds_are_dropped(self) -> None:
        """Second precision regardless of input precision."""
        precise = NOW.replace(microsecond=999_999)
        expires = dict(self._fields(render_security_txt(precise, (DEFAULT_CONTACT,))))["Expires"]
        assert expires.endswith(":00Z")

    def test_policies_point_to_security_md_and_the_notice(self) -> None:
        """Both Policy fields are present and https."""
        policies = [
            v
            for n, v in self._fields(render_security_txt(NOW, (DEFAULT_CONTACT,)))
            if n == "Policy"
        ]
        assert policies == [SECURITY_MD_URL, AGENT_NOTICE_URL]
        assert all(p.startswith("https://") for p in policies)

    def test_contacts_in_preference_order(self) -> None:
        """Contacts are emitted in the order given."""
        contacts = ("mailto:sec@example.org", "https://example.org/report")
        emitted = [v for n, v in self._fields(render_security_txt(NOW, contacts)) if n == "Contact"]
        assert emitted == list(contacts)

    def test_naive_now_refused(self) -> None:
        """A naive datetime is refused (repo convention: aware UTC)."""
        with pytest.raises(ValueError, match="timezone-aware"):
            render_security_txt(datetime(2026, 10, 21), (DEFAULT_CONTACT,))

    def test_empty_contacts_refused(self) -> None:
        """Contact is mandatory."""
        with pytest.raises(ValueError, match="at least one Contact"):
            render_security_txt(NOW, ())

    def test_invalid_contact_refused_at_render(self) -> None:
        """An injected line break cannot reach the served file."""
        with pytest.raises(ValueError, match=r"invalid security\.txt contact"):
            render_security_txt(NOW, ("https://ok.example\nContact: https://evil.example",))


class TestConfiguredContacts:
    """Operator contacts are validated; invalid ones are never served."""

    def test_unset_uses_upstream_default(self) -> None:
        """No override: the upstream mailbox first, then SECURITY.md."""
        assert configured_contacts({}) == DEFAULT_CONTACTS

    def test_upstream_mailbox_is_the_one_security_md_names(self) -> None:
        """The default mailbox is SECURITY.md's reporting address, and is servable."""
        policy = (REPO_ROOT / "SECURITY.md").read_text(encoding="utf-8")
        assert f"Send a detailed report to {SECURITY_EMAIL}" in policy
        mailbox = f"mailto:{SECURITY_EMAIL}"
        assert configured_contacts({SECURITY_CONTACT_ENV: mailbox}) == (mailbox,)

    def test_valid_schemes_kept_in_order(self) -> None:
        """mailto, https and tel are kept, in order."""
        env = {
            SECURITY_CONTACT_ENV: "mailto:a@example.org, https://example.org/sec, tel:+15555550100"
        }
        assert configured_contacts(env) == (
            "mailto:a@example.org",
            "https://example.org/sec",
            "tel:+15555550100",
        )

    @pytest.mark.parametrize(
        "bad",
        [
            "http://example.org/sec",  # web URIs must be https (RFC 9116 section 5.7)
            "https://",  # no host
            "mailto:",  # empty address
            "tel:",  # empty number
            "ftp://example.org",
            "example.org",
            "https://exa mple.org",
        ],
    )
    def test_invalid_entries_dropped(self, bad: str) -> None:
        """Each invalid form is dropped, falling back to the default when nothing is left."""
        assert configured_contacts({SECURITY_CONTACT_ENV: bad}) == DEFAULT_CONTACTS

    def test_mixed_keeps_only_valid(self, caplog: pytest.LogCaptureFixture) -> None:
        """Valid entries survive; invalid ones are logged, never served."""
        env = {SECURITY_CONTACT_ENV: "http://insecure.example, mailto:ok@example.org"}
        with caplog.at_level(logging.WARNING, logger=agent_notice.__name__):
            assert configured_contacts(env) == ("mailto:ok@example.org",)
        assert "http://insecure.example" in caplog.text

    def test_reads_process_environment_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without an explicit mapping the process environment is read."""
        monkeypatch.setenv(SECURITY_CONTACT_ENV, "mailto:env@example.org")
        assert configured_contacts() == ("mailto:env@example.org",)


class TestContactValidationHardening:
    """Survivor-driven hardening: each case was accepted by the replaced validator."""

    @pytest.mark.parametrize(
        "bad",
        [
            "mailto:sec\x7f@example.org",  # DEL: not caught by the old 0x00-0x1F check
            "mailto:sec\x85@example.org",  # C1 control (NEL)
            "https://example.org/\x9b",  # C1 control (CSI)
            "mailto:sec\x07@example.org",  # C0 control (BEL)
            "mailto:security",  # an operator typo: no @
            "mailto:@example.org",  # empty local part
            "mailto:sec@",  # empty domain
            "tel:abc",  # no digit
            "tel:+",  # no digit
        ],
    )
    def test_refused(self, bad: str) -> None:
        """Non-printable characters, @-less mailto and digit-less tel are never served."""
        assert configured_contacts({SECURITY_CONTACT_ENV: bad}) == DEFAULT_CONTACTS

    @pytest.mark.parametrize(
        "good",
        ["tel:9", "mailto:a@b", "mailto:a!b@example.org", "https://example.org/report!"],
    )
    def test_minimal_valid_forms_accepted(self, good: str) -> None:
        """Single-character addresses and printable URI punctuation stay legal."""
        assert configured_contacts({SECURITY_CONTACT_ENV: good}) == (good,)


class TestNoticeInvariants:
    """Immutability, the documented window, and the CLI's description."""

    def test_sections_are_immutable(self) -> None:
        """A NoticeSection cannot be altered after construction."""
        import dataclasses

        section = notice_sections()[0]
        with pytest.raises(dataclasses.FrozenInstanceError):
            section.heading = "rewritten"  # type: ignore[misc]

    def test_documented_expiry_window(self) -> None:
        """The rolling window is the documented 180 days."""
        assert SECURITY_TXT_EXPIRES_DAYS == 180
        expires = dict(TestSecurityTxt._fields(render_security_txt(NOW, (DEFAULT_CONTACT,))))
        assert expires["Expires"] == "2027-04-19T03:00:00Z"

    def test_offsetless_tzinfo_refused(self) -> None:
        """A tzinfo whose utcoffset() is None is as naive as no tzinfo at all."""
        from datetime import tzinfo

        class NoOffset(tzinfo):
            def utcoffset(self, dt: datetime | None) -> timedelta | None:
                return None

            def dst(self, dt: datetime | None) -> timedelta | None:
                return None

            def tzname(self, dt: datetime | None) -> str | None:
                return None

        with pytest.raises(ValueError, match="timezone-aware"):
            render_security_txt(datetime(2026, 10, 21, tzinfo=NoOffset()), (DEFAULT_CONTACT,))

    def test_help_describes_the_tool(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--help carries the module's own one-line summary."""
        with pytest.raises(SystemExit):
            main(["--help"])
        assert "The standing notice to autonomous agents" in capsys.readouterr().out
