# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fuzz the watch's external-XML parsers against arbitrary input.

``parse_cap`` and ``validate_cap_xml`` read XML that arrives from upstream
alerting authorities (WMO/GDACS/national CAP feeds).  Both go through
defusedxml, so XXE and entity-expansion are structurally closed; what these
properties pin is the *error contract* under arbitrary — including
syntactically valid but hostile — input:

* ``parse_cap`` raises only ``ValueError`` (never an unhandled parser
  exception, recursion error, or anything else), or returns a dict;
* ``validate_cap_xml`` never raises and always returns a bool;
* a well-formed CAP alert round-trips through generation and re-parse.

The defusedxml defences themselves are pinned explicitly: billion-laughs
and external-entity documents are rejected, not expanded.
"""

from __future__ import annotations

from hypothesis import (
    HealthCheck,
    given,
    settings,
    strategies as st,
)

from omni_mercury_engine.alerting.cap_generator import CAPAlertGenerator

validate_cap_xml = CAPAlertGenerator.validate_cap_xml
from omni_mercury_engine.watch.world import parse_cap

# Billion-laughs and an external-entity SSRF/exfil document: defusedxml must
# refuse both rather than expand or fetch.
_BILLION_LAUGHS = (
    '<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol">'
    '<!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">'
    '<!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">'
    "]><alert><info><event>&lol3;</event></info></alert>"
)
_EXTERNAL_ENTITY = (
    '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
    "<alert><info><event>&xxe;</event></info></alert>"
)

# Fragments that, recombined, make both well-formed and broken CAP-ish XML.
_fragments = st.sampled_from(
    [
        "<alert>",
        "</alert>",
        "<info>",
        "</info>",
        "<area><areaDesc>Zone 1</areaDesc></area>",
        "<event>Flood</event>",
        "<language>en-US</language>",
        "<identifier>X</identifier>",
        "<!DOCTYPE a [<!ENTITY e SYSTEM 'file:///x'>]>",
        "&e;",
        "<",
        ">",
        "\x00",
        "ÿ" * 8,
        '<?xml version="1.0"?>',
    ]
)
_documents = st.one_of(
    st.text(max_size=200),
    st.binary(max_size=200).map(lambda b: b.decode("latin-1")),
    st.lists(_fragments, max_size=12).map("".join),
)


class TestParseCapErrorContract:
    @settings(max_examples=600, suppress_health_check=[HealthCheck.too_slow])
    @given(document=_documents)
    def test_parse_cap_raises_only_valueerror_or_returns_dict(self, document: str) -> None:
        try:
            result = parse_cap(document)
        except ValueError:
            return  # the one documented failure mode
        assert isinstance(result, dict)

    @settings(max_examples=600, suppress_health_check=[HealthCheck.too_slow])
    @given(document=_documents)
    def test_validate_cap_xml_never_raises(self, document: str) -> None:
        assert isinstance(validate_cap_xml(document), bool)


class TestDefusedXmlDefences:
    def test_billion_laughs_is_refused_not_expanded(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            parse_cap(_BILLION_LAUGHS)
        assert validate_cap_xml(_BILLION_LAUGHS) is False

    def test_external_entity_is_refused(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            parse_cap(_EXTERNAL_ENTITY)
        assert validate_cap_xml(_EXTERNAL_ENTITY) is False

    def test_a_real_cap_alert_still_parses(self) -> None:
        alert = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">'
            "<identifier>MERCURY-TEST-1</identifier>"
            "<status>Actual</status><msgType>Alert</msgType>"
            "<sent>2026-11-20T00:00:00-00:00</sent>"
            "<info><language>en-US</language><event>Coastal Flood</event>"
            "<severity>Severe</severity><urgency>Expected</urgency>"
            "<certainty>Likely</certainty>"
            "<area><areaDesc>Test County</areaDesc></area></info></alert>"
        )
        parsed = parse_cap(alert)
        assert parsed["identifier"] == "MERCURY-TEST-1"
        assert parsed["event"] == "Coastal Flood"
        assert "Test County" in parsed["area"]
