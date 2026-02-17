"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.

Tests for the CAP (Common Alerting Protocol) 1.2 alert generator:
  - CAPAlertGenerator instantiation
  - generate_alert produces valid XML
  - from_detection produces valid XML from numpy score arrays
  - validate_cap_xml accepts/rejects XML correctly
  - CAP enum values match the CAP 1.2 specification
  - DOMAIN_CATEGORY_MAP covers all 15 Mercury domains
"""

from __future__ import annotations

import numpy as np
import pytest
from defusedxml.ElementTree import fromstring

from omni_mercury_engine.alerting.cap_generator import (
    DOMAIN_CATEGORY_MAP,
    CAPAlertGenerator,
    CAPCategory,
    CAPCertainty,
    CAPMsgType,
    CAPScope,
    CAPSeverity,
    CAPStatus,
    CAPUrgency,
)

# CAP namespace for element lookups
_NS = "{urn:oasis:names:tc:emergency:cap:1.2}"


def _find(root, path):
    """Find element with CAP namespace prefix."""
    ns_path = "/".join(f"{_NS}{p}" for p in path.split("/"))
    return root.find(ns_path)


# ======================================================================
# CAPAlertGenerator instantiation
# ======================================================================


class TestCAPAlertGeneratorInit:
    """Tests for CAPAlertGenerator construction."""

    def test_default_instantiation(self):
        """Generator should be constructible with no arguments."""
        gen = CAPAlertGenerator()
        assert isinstance(gen, CAPAlertGenerator)

    def test_default_sender(self):
        """Default sender should be the Mercury project email."""
        gen = CAPAlertGenerator()
        assert "mercury" in gen.sender.lower()

    def test_custom_sender(self):
        """Custom sender string should be stored."""
        gen = CAPAlertGenerator(sender="custom@example.org")
        assert gen.sender == "custom@example.org"

    def test_custom_status_and_scope(self):
        """Custom status and scope should be stored as defaults."""
        gen = CAPAlertGenerator(
            status=CAPStatus.TEST,
            scope=CAPScope.PRIVATE,
        )
        assert gen.default_status is CAPStatus.TEST
        assert gen.default_scope is CAPScope.PRIVATE


# ======================================================================
# generate_alert produces valid XML
# ======================================================================


class TestGenerateAlert:
    """Tests for generate_alert XML output."""

    @pytest.fixture()
    def generator(self) -> CAPAlertGenerator:
        return CAPAlertGenerator(sender="test@example.com")

    def test_returns_string(self, generator):
        """generate_alert must return a string."""
        xml = generator.generate_alert(
            domain="earthquake",
            headline="Test headline",
            description="Test description",
            anomaly_score=0.85,
            area_description="Test area",
        )
        assert isinstance(xml, str)

    def test_xml_declaration(self, generator):
        """Output must start with an XML declaration."""
        xml = generator.generate_alert(
            domain="earthquake",
            headline="Test",
            description="Desc",
            anomaly_score=0.5,
            area_description="Area",
        )
        assert xml.startswith("<?xml")

    def test_parses_as_valid_xml(self, generator):
        """Output must be parseable by ElementTree."""
        xml = generator.generate_alert(
            domain="tsunami",
            headline="Tsunami warning",
            description="Large wave detected",
            anomaly_score=0.92,
            area_description="Pacific Coast",
        )
        root = fromstring(xml)
        assert root.tag.endswith("alert")

    def test_required_cap_elements_present(self, generator):
        """CAP 1.2 required top-level elements must be present."""
        xml = generator.generate_alert(
            domain="hurricane",
            headline="Hurricane alert",
            description="Category 4",
            anomaly_score=0.95,
            area_description="Gulf Coast",
        )
        root = fromstring(xml)
        for tag in ("identifier", "sender", "sent", "status", "msgType", "scope"):
            assert _find(root, tag) is not None, f"Missing required element: {tag}"

    def test_info_block_elements_present(self, generator):
        """CAP info block must contain category, event, severity, etc."""
        xml = generator.generate_alert(
            domain="wildfire",
            headline="Wildfire",
            description="Active burn",
            anomaly_score=0.75,
            area_description="Northern California",
        )
        root = fromstring(xml)
        info = _find(root, "info")
        assert info is not None
        for tag in ("category", "event", "urgency", "severity", "certainty"):
            assert _find(info, tag) is not None, f"Missing info element: {tag}"

    def test_sender_in_xml(self, generator):
        """Sender element must contain the configured sender."""
        xml = generator.generate_alert(
            domain="flood",
            headline="Flood",
            description="Rising water",
            anomaly_score=0.6,
            area_description="River basin",
        )
        root = fromstring(xml)
        assert _find(root, "sender").text == "test@example.com"

    def test_area_description_in_xml(self, generator):
        """Area description must appear in the area block."""
        xml = generator.generate_alert(
            domain="landslide",
            headline="Landslide",
            description="Slope failure",
            anomaly_score=0.7,
            area_description="Mountain Pass Region",
        )
        root = fromstring(xml)
        area = _find(root, "info/area")
        assert area is not None
        assert area.find(f"{_NS}areaDesc").text == "Mountain Pass Region"

    def test_coordinates_produce_circle_element(self, generator):
        """When coordinates are given, a circle element should appear."""
        xml = generator.generate_alert(
            domain="earthquake",
            headline="EQ",
            description="Shaking",
            anomaly_score=0.8,
            area_description="Epicenter",
            coordinates=(37.7749, -122.4194),
        )
        root = fromstring(xml)
        circle = _find(root, "info/area/circle")
        assert circle is not None
        assert "37.7749" in circle.text
        assert "-122.4194" in circle.text

    def test_geocode_in_xml(self, generator):
        """When geocode is given, geocode elements should appear."""
        xml = generator.generate_alert(
            domain="tornado",
            headline="Tornado",
            description="Funnel spotted",
            anomaly_score=0.9,
            area_description="Oklahoma",
            geocode={"FIPS6": "400000"},
        )
        root = fromstring(xml)
        geocode = _find(root, "info/area/geocode")
        assert geocode is not None
        assert geocode.find(f"{_NS}valueName").text == "FIPS6"
        assert geocode.find(f"{_NS}value").text == "400000"

    def test_mercury_parameters_present(self, generator):
        """Mercury-specific parameters (score, domain) must appear."""
        xml = generator.generate_alert(
            domain="sepsis",
            headline="Sepsis alert",
            description="Elevated markers",
            anomaly_score=0.65,
            area_description="ICU Ward 3",
        )
        root = fromstring(xml)
        params = root.findall(f"{_NS}info/{_NS}parameter")
        value_names = [p.find(f"{_NS}valueName").text for p in params]
        assert "MercuryAnomalyScore" in value_names
        assert "MercuryDomain" in value_names

    def test_custom_parameters_included(self, generator):
        """Extra user-supplied parameters should appear in the XML."""
        xml = generator.generate_alert(
            domain="financial",
            headline="Market anomaly",
            description="Unusual trading",
            anomaly_score=0.55,
            area_description="NYSE",
            parameters={"CustomKey": "CustomValue"},
        )
        root = fromstring(xml)
        params = root.findall(f"{_NS}info/{_NS}parameter")
        value_names = {p.find(f"{_NS}valueName").text: p.find(f"{_NS}value").text for p in params}
        assert value_names.get("CustomKey") == "CustomValue"

    def test_severity_extreme_for_high_score(self, generator):
        """Score >= 0.9 should map to Extreme severity."""
        xml = generator.generate_alert(
            domain="earthquake",
            headline="Major EQ",
            description="M8.0",
            anomaly_score=0.95,
            area_description="Fault zone",
        )
        root = fromstring(xml)
        severity = _find(root, "info/severity").text
        assert severity == "Extreme"

    def test_severity_minor_for_low_score(self, generator):
        """Score in [0.3, 0.5) should map to Minor severity."""
        xml = generator.generate_alert(
            domain="earthquake",
            headline="Minor EQ",
            description="M2.5",
            anomaly_score=0.35,
            area_description="Fault zone",
        )
        root = fromstring(xml)
        severity = _find(root, "info/severity").text
        assert severity == "Minor"


# ======================================================================
# from_detection (numpy scores -> XML)
# ======================================================================


class TestFromDetection:
    """Tests for from_detection with numpy score arrays."""

    @pytest.fixture()
    def generator(self) -> CAPAlertGenerator:
        return CAPAlertGenerator()

    def test_returns_string(self, generator):
        """from_detection must return a string."""
        scores = np.array([0.1, 0.3, 0.5, 0.9])
        xml = generator.from_detection(domain="earthquake", scores=scores)
        assert isinstance(xml, str)

    def test_valid_xml(self, generator):
        """Output must be parseable XML."""
        scores = np.array([0.2, 0.4, 0.6, 0.8, 0.95])
        xml = generator.from_detection(domain="tsunami", scores=scores)
        root = fromstring(xml)
        assert root.tag.endswith("alert")

    def test_passes_validation(self, generator):
        """from_detection output should pass validate_cap_xml."""
        scores = np.array([0.7, 0.85, 0.3])
        xml = generator.from_detection(domain="hurricane", scores=scores)
        assert CAPAlertGenerator.validate_cap_xml(xml)

    def test_max_score_in_headline(self, generator):
        """Headline should contain the maximum anomaly score."""
        scores = np.array([0.1, 0.2, 0.99])
        xml = generator.from_detection(domain="wildfire", scores=scores)
        root = fromstring(xml)
        headline = _find(root, "info/headline").text
        assert "0.99" in headline

    def test_anomaly_count_in_headline(self, generator):
        """Headline should mention the number of anomalies (score > 0.5)."""
        scores = np.array([0.1, 0.2, 0.6, 0.8, 0.9])
        xml = generator.from_detection(domain="flood", scores=scores)
        root = fromstring(xml)
        headline = _find(root, "info/headline").text
        # 3 scores > 0.5
        assert "3 anomalies" in headline

    def test_metadata_in_description(self, generator):
        """Supplied metadata keys should appear in the description."""
        scores = np.array([0.5, 0.7])
        xml = generator.from_detection(
            domain="earthquake",
            scores=scores,
            metadata={"magnitude": "7.8", "depth_km": "10"},
        )
        root = fromstring(xml)
        desc = _find(root, "info/description").text
        assert "magnitude" in desc
        assert "7.8" in desc

    def test_area_description_default(self, generator):
        """Default area description should be used when none is given."""
        scores = np.array([0.5])
        xml = generator.from_detection(domain="pandemic", scores=scores)
        root = fromstring(xml)
        area_desc = _find(root, "info/area/areaDesc").text
        assert area_desc == "Unspecified Area"

    def test_area_description_custom(self, generator):
        """Custom area description should appear in the output."""
        scores = np.array([0.5])
        xml = generator.from_detection(
            domain="pandemic",
            scores=scores,
            area_description="Southeast Asia",
        )
        root = fromstring(xml)
        area_desc = _find(root, "info/area/areaDesc").text
        assert area_desc == "Southeast Asia"

    def test_extra_parameters_present(self, generator):
        """from_detection should inject MercuryMeanScore, AnomalyCount, TotalPoints."""
        scores = np.array([0.1, 0.6, 0.8])
        xml = generator.from_detection(domain="energy", scores=scores)
        root = fromstring(xml)
        params = root.findall(f"{_NS}info/{_NS}parameter")
        value_names = [p.find(f"{_NS}valueName").text for p in params]
        assert "MercuryMeanScore" in value_names
        assert "MercuryAnomalyCount" in value_names
        assert "MercuryTotalPoints" in value_names

    def test_single_score(self, generator):
        """Should handle a single-element score array."""
        scores = np.array([0.85])
        xml = generator.from_detection(domain="sepsis", scores=scores)
        assert CAPAlertGenerator.validate_cap_xml(xml)


# ======================================================================
# validate_cap_xml
# ======================================================================


class TestValidateCapXml:
    """Tests for static validate_cap_xml method."""

    def test_valid_xml_accepted(self):
        """Well-formed CAP XML should pass validation."""
        gen = CAPAlertGenerator()
        xml = gen.generate_alert(
            domain="earthquake",
            headline="Test",
            description="Test",
            anomaly_score=0.5,
            area_description="Test",
        )
        assert CAPAlertGenerator.validate_cap_xml(xml) is True

    def test_empty_string_rejected(self):
        """Empty string is not valid XML."""
        assert CAPAlertGenerator.validate_cap_xml("") is False

    def test_nonsense_rejected(self):
        """Random text is not valid XML."""
        assert CAPAlertGenerator.validate_cap_xml("not xml at all") is False

    def test_valid_xml_but_missing_cap_elements(self):
        """Well-formed XML that is missing required CAP elements should fail."""
        xml = '<?xml version="1.0" ?><root><child>text</child></root>'
        assert CAPAlertGenerator.validate_cap_xml(xml) is False

    def test_missing_info_block(self):
        """XML with top-level CAP tags but no info block should fail."""
        xml = (
            '<?xml version="1.0" ?>'
            "<alert>"
            "<identifier>id1</identifier>"
            "<sender>s</sender>"
            "<sent>2025-01-01T00:00:00+0000</sent>"
            "<status>Actual</status>"
            "<msgType>Alert</msgType>"
            "<scope>Public</scope>"
            "</alert>"
        )
        assert CAPAlertGenerator.validate_cap_xml(xml) is False

    def test_partial_info_block_fails(self):
        """Info block missing required sub-elements should fail."""
        xml = (
            '<?xml version="1.0" ?>'
            "<alert>"
            "<identifier>id1</identifier>"
            "<sender>s</sender>"
            "<sent>2025-01-01T00:00:00+0000</sent>"
            "<status>Actual</status>"
            "<msgType>Alert</msgType>"
            "<scope>Public</scope>"
            "<info><category>Geo</category></info>"
            "</alert>"
        )
        assert CAPAlertGenerator.validate_cap_xml(xml) is False

    def test_malformed_xml_rejected(self):
        """Syntactically broken XML should be rejected."""
        xml = '<?xml version="1.0" ?><alert><unclosed>'
        assert CAPAlertGenerator.validate_cap_xml(xml) is False


# ======================================================================
# CAP enum values
# ======================================================================


class TestCAPEnumValues:
    """Verify that all CAP enums carry the values from the CAP 1.2 spec."""

    def test_cap_status_values(self):
        """CAPStatus must have Actual, Exercise, System, Test, Draft."""
        expected = {"Actual", "Exercise", "System", "Test", "Draft"}
        actual = {s.value for s in CAPStatus}
        assert actual == expected

    def test_cap_msg_type_values(self):
        """CAPMsgType must have Alert, Update, Cancel, Ack, Error."""
        expected = {"Alert", "Update", "Cancel", "Ack", "Error"}
        actual = {m.value for m in CAPMsgType}
        assert actual == expected

    def test_cap_scope_values(self):
        """CAPScope must have Public, Restricted, Private."""
        expected = {"Public", "Restricted", "Private"}
        actual = {s.value for s in CAPScope}
        assert actual == expected

    def test_cap_category_values(self):
        """CAPCategory must cover all CAP 1.2 categories."""
        expected = {
            "Geo",
            "Met",
            "Safety",
            "Security",
            "Rescue",
            "Fire",
            "Health",
            "Env",
            "Transport",
            "Infra",
            "CBRNE",
            "Other",
        }
        actual = {c.value for c in CAPCategory}
        assert actual == expected

    def test_cap_severity_values(self):
        """CAPSeverity must have Extreme, Severe, Moderate, Minor, Unknown."""
        expected = {"Extreme", "Severe", "Moderate", "Minor", "Unknown"}
        actual = {s.value for s in CAPSeverity}
        assert actual == expected

    def test_cap_certainty_values(self):
        """CAPCertainty must have Observed, Likely, Possible, Unlikely, Unknown."""
        expected = {"Observed", "Likely", "Possible", "Unlikely", "Unknown"}
        actual = {c.value for c in CAPCertainty}
        assert actual == expected

    def test_cap_urgency_values(self):
        """CAPUrgency must have Immediate, Expected, Future, Past, Unknown."""
        expected = {"Immediate", "Expected", "Future", "Past", "Unknown"}
        actual = {u.value for u in CAPUrgency}
        assert actual == expected


# ======================================================================
# DOMAIN_CATEGORY_MAP coverage
# ======================================================================


class TestDomainCategoryMap:
    """Verify DOMAIN_CATEGORY_MAP covers all Mercury domains."""

    EXPECTED_DOMAINS = {
        "earthquake",
        "tsunami",
        "hurricane",
        "tornado",
        "flood",
        "wildfire",
        "volcanic",
        "landslide",
        "sepsis",
        "pandemic",
        "financial",
        "energy",
        "marine",
        "network_security",
        "fema",
    }

    def test_all_domains_present(self):
        """Every known Mercury domain must be in DOMAIN_CATEGORY_MAP."""
        for domain in self.EXPECTED_DOMAINS:
            assert domain in DOMAIN_CATEGORY_MAP, f"Missing domain: {domain}"

    def test_no_unexpected_domains(self):
        """Map should not contain domains outside the known set."""
        assert set(DOMAIN_CATEGORY_MAP.keys()) == self.EXPECTED_DOMAINS

    def test_all_values_are_cap_category(self):
        """Every value in the map must be a CAPCategory enum member."""
        for domain, category in DOMAIN_CATEGORY_MAP.items():
            assert isinstance(category, CAPCategory), (
                f"DOMAIN_CATEGORY_MAP['{domain}'] is {type(category)}, " f"expected CAPCategory"
            )

    def test_earthquake_is_geo(self):
        """Earthquake domain should map to Geo category."""
        assert DOMAIN_CATEGORY_MAP["earthquake"] is CAPCategory.GEO

    def test_hurricane_is_met(self):
        """Hurricane domain should map to Met category."""
        assert DOMAIN_CATEGORY_MAP["hurricane"] is CAPCategory.MET

    def test_wildfire_is_fire(self):
        """Wildfire domain should map to Fire category."""
        assert DOMAIN_CATEGORY_MAP["wildfire"] is CAPCategory.FIRE

    def test_sepsis_is_health(self):
        """Sepsis domain should map to Health category."""
        assert DOMAIN_CATEGORY_MAP["sepsis"] is CAPCategory.HEALTH

    def test_network_security_is_security(self):
        """Network security domain should map to Security category."""
        assert DOMAIN_CATEGORY_MAP["network_security"] is CAPCategory.SECURITY

    def test_energy_is_infra(self):
        """Energy domain should map to Infra category."""
        assert DOMAIN_CATEGORY_MAP["energy"] is CAPCategory.INFRA
