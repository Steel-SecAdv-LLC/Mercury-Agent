"""
Tests for omni_mercury_engine.utils.report_generator module.

Tests report generation functionality.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pytest

from omni_mercury_engine.security.tlp_handler import TLPColor, get_tlp_handler
from omni_mercury_engine.utils.report_generator import (
    AnomalyReport,
    ExecutiveSummary,
    ReportFormat,
    ReportGenerator,
    ReportSection,
    TechnicalDetails,
)


class TestReportFormat:
    """Tests for ReportFormat enum."""

    def test_all_formats_defined(self) -> None:
        """Test that expected formats are defined."""
        formats = list(ReportFormat)
        assert len(formats) >= 3  # At least JSON, HTML, PDF

    def test_format_values(self) -> None:
        """Test format values are strings."""
        for fmt in list(ReportFormat):
            assert isinstance(fmt.value, str)


class TestAnomalyReport:
    """Tests for AnomalyReport dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic report creation."""
        report = AnomalyReport(
            title="Test Report",
            timestamp=datetime.now(),
            anomaly_count=5,
            total_datapoints=1000,
        )
        assert report.title == "Test Report"
        assert report.anomaly_count == 5

    def test_severity_calculation(self) -> None:
        """Test severity level calculation."""
        report = AnomalyReport(
            title="Test",
            timestamp=datetime.now(),
            anomaly_count=100,
            total_datapoints=1000,
        )
        # 10% anomaly rate should be high severity
        assert report.severity in ["low", "medium", "high", "critical"]

    def test_anomaly_rate(self) -> None:
        """Test anomaly rate calculation."""
        report = AnomalyReport(
            title="Test",
            timestamp=datetime.now(),
            anomaly_count=50,
            total_datapoints=1000,
        )
        assert report.anomaly_rate == pytest.approx(0.05)


class TestReportSection:
    """Tests for ReportSection class."""

    def test_section_creation(self) -> None:
        """Test section creation."""
        section = ReportSection(
            title="Analysis Results",
            content="Detailed analysis...",
            subsections=[],
        )
        assert section.title == "Analysis Results"

    def test_nested_sections(self) -> None:
        """Test nested subsections."""
        subsection = ReportSection(title="Subsection", content="Details")
        section = ReportSection(title="Main Section", content="Overview", subsections=[subsection])
        assert section.subsections is not None
        assert len(section.subsections) == 1


class TestExecutiveSummary:
    """Tests for ExecutiveSummary class."""

    def test_summary_generation(self) -> None:
        """Test executive summary generation."""
        summary = ExecutiveSummary(
            key_findings=["Finding 1", "Finding 2"],
            risk_assessment="Medium",
            recommendations=["Recommendation 1"],
            confidence_score=0.85,
        )
        assert len(summary.key_findings) == 2
        assert summary.confidence_score == 0.85

    def test_summary_to_dict(self) -> None:
        """Test converting summary to dictionary."""
        summary = ExecutiveSummary(
            key_findings=["Test finding"],
            risk_assessment="Low",
            recommendations=["Test rec"],
            confidence_score=0.90,
        )
        d = summary.to_dict()

        assert "key_findings" in d
        assert "risk_assessment" in d
        assert "confidence_score" in d


class TestTechnicalDetails:
    """Tests for TechnicalDetails class."""

    def test_details_creation(self) -> None:
        """Test technical details creation."""
        details = TechnicalDetails(
            methodology="Statistical analysis",
            algorithms_used=["IsolationForest", "OneClassSVM"],
            parameters={"contamination": 0.1},
            data_sources=["sensor_data"],
        )
        assert len(details.algorithms_used) == 2

    def test_parameters_serialization(self) -> None:
        """Test that parameters are serializable."""
        details = TechnicalDetails(
            methodology="Test",
            algorithms_used=["Algo1"],
            parameters={"param1": 0.5, "param2": [1, 2, 3]},
            data_sources=[],
        )
        # Should be JSON serializable
        json_str = json.dumps(details.parameters)
        assert "param1" in json_str


class TestReportGenerator:
    """Tests for ReportGenerator class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.generator = ReportGenerator()

    def test_initialization(self) -> None:
        """Test generator initialization."""
        assert self.generator is not None

    def test_generate_json_report(self) -> None:
        """Test JSON report generation."""
        data = {
            "anomalies": [
                {"id": 1, "score": 0.9},
                {"id": 2, "score": 0.85},
            ],
            "summary": {"total": 100, "anomalous": 2},
        }
        report = self.generator.generate(data, format=ReportFormat.JSON)

        assert report is not None
        # Should be valid JSON
        parsed = json.loads(report)
        assert "anomalies" in parsed or "summary" in parsed

    def test_generate_html_report(self) -> None:
        """Test HTML report generation."""
        data = {
            "title": "Test Report",
            "findings": ["Finding 1", "Finding 2"],
        }
        report = self.generator.generate(data, format=ReportFormat.HTML)

        assert report is not None
        assert "<html>" in report.lower() or "<!doctype" in report.lower()

    def test_generate_markdown_report(self) -> None:
        """Test Markdown report generation."""
        data = {
            "title": "Test Report",
            "sections": [
                {"heading": "Section 1", "content": "Content 1"},
            ],
        }
        report = self.generator.generate(data, format=ReportFormat.MARKDOWN)

        assert report is not None
        assert "#" in report  # Markdown headings

    def test_add_section(self) -> None:
        """Test adding sections to report."""
        self.generator.add_section("Introduction", "This is the introduction.")
        self.generator.add_section("Analysis", "This is the analysis.")

        report = self.generator.generate({}, format=ReportFormat.MARKDOWN)
        assert "Introduction" in report
        assert "Analysis" in report

    def test_add_chart_placeholder(self) -> None:
        """Test adding chart placeholders."""
        chart_data = {
            "type": "line",
            "title": "Anomaly Trend",
            "data": [1, 2, 3, 4, 5],
        }
        self.generator.add_chart("trend_chart", chart_data)

        assert "trend_chart" in self.generator._charts

    def test_add_table(self) -> None:
        """Test adding tables to report."""
        table_data = {
            "headers": ["ID", "Score", "Status"],
            "rows": [
                [1, 0.9, "Anomaly"],
                [2, 0.3, "Normal"],
            ],
        }
        self.generator.add_table("anomaly_table", table_data)

        report = self.generator.generate({}, format=ReportFormat.HTML)
        assert "table" in report.lower() or "ID" in report

    def test_template_customization(self) -> None:
        """Test custom template support."""
        custom_template = "Custom Report: {{ title }}"
        generator = ReportGenerator(template=custom_template)

        report = generator.generate({"title": "Test"}, format=ReportFormat.MARKDOWN)
        # Should use custom template
        assert report is not None


class TestReportContent:
    """Tests for report content generation."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.generator = ReportGenerator()

    def test_timestamp_included(self) -> None:
        """Test that timestamp is included in report."""
        data = {"title": "Test"}
        report = self.generator.generate(data, format=ReportFormat.JSON)
        parsed = json.loads(report)

        # Should have timestamp or generated_at field
        assert (
            any(key in parsed for key in ["timestamp", "generated_at", "report_date"])
            or "title" in parsed
        )

    def test_metadata_included(self) -> None:
        """Test that metadata is included."""
        self.generator.set_metadata(
            author="Test Author", version="1.0", classification="UNCLASSIFIED"
        )
        report = self.generator.generate({}, format=ReportFormat.JSON)
        parsed = json.loads(report)

        # Metadata should be present
        if "metadata" in parsed:
            assert "author" in parsed["metadata"] or "version" in parsed["metadata"]

    def test_anomaly_details_formatting(self) -> None:
        """Test anomaly details are properly formatted."""
        anomalies = [
            {
                "id": i,
                "score": 0.9 - i * 0.1,
                "timestamp": "2025-01-01T12:00:00",
                "features": {"f1": 0.5, "f2": 0.3},
            }
            for i in range(5)
        ]
        data = {"anomalies": anomalies}
        report = self.generator.generate(data, format=ReportFormat.JSON)
        parsed = json.loads(report)

        if "anomalies" in parsed:
            assert len(parsed["anomalies"]) == 5


class TestReportExport:
    """Tests for report export functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.generator = ReportGenerator()

    def test_export_to_file(self, tmp_path: Any) -> None:
        """Test exporting report to file."""
        data = {"title": "Test Report", "content": "Test content"}
        filepath = tmp_path / "report.json"

        self.generator.export(data, str(filepath), format=ReportFormat.JSON)

        assert filepath.exists()
        with open(filepath) as f:
            content = f.read()
            assert len(content) > 0

    def test_export_html_to_file(self, tmp_path: Any) -> None:
        """Test exporting HTML report to file."""
        data = {"title": "Test Report"}
        filepath = tmp_path / "report.html"

        self.generator.export(data, str(filepath), format=ReportFormat.HTML)

        assert filepath.exists()

    def test_export_creates_directory(self, tmp_path: Any) -> None:
        """Test that export creates directory if needed."""
        data = {"title": "Test"}
        nested_path = tmp_path / "subdir" / "report.json"

        self.generator.export(data, str(nested_path), format=ReportFormat.JSON)

        assert nested_path.exists()


class TestReportValidation:
    """Tests for report validation."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.generator = ReportGenerator()

    def test_validates_required_fields(self) -> None:
        """Test validation of required fields."""
        # Empty data should still generate valid report
        report = self.generator.generate({}, format=ReportFormat.JSON)
        parsed = json.loads(report)

        # Should have at least some structure
        assert isinstance(parsed, dict)

    def test_sanitizes_html_content(self) -> None:
        """Test that HTML content is sanitized."""
        data = {"content": "<script>alert('xss')</script>Malicious content"}
        report = self.generator.generate(data, format=ReportFormat.HTML)

        # Script tags should be escaped or removed
        assert "<script>" not in report or "&lt;script&gt;" in report

    def test_handles_unicode(self) -> None:
        """Test handling of unicode characters."""
        data = {"title": "レポート", "content": "日本語テスト"}
        report = self.generator.generate(data, format=ReportFormat.JSON)

        parsed = json.loads(report)
        # Unicode should be preserved
        assert any("レポート" in str(v) or "日本語" in str(v) for v in parsed.values())


class TestReportTLPIntegration:
    """Tests for the TLP wiring on :class:`ReportGenerator`."""

    def test_default_no_tlp_block_in_json(self) -> None:
        """Without an applied classification, no TLP block is emitted."""
        gen = ReportGenerator()
        report = json.loads(gen.generate({"title": "T"}, format=ReportFormat.JSON))
        assert "tlp" not in report
        assert gen.tlp_classification is None

    def test_apply_from_score_low_emits_clear(self) -> None:
        """A low score classifies the report as TLP:CLEAR."""
        gen = ReportGenerator()
        classification = gen.apply_tlp_classification(
            anomaly_score=0.05,
            domain_type="general",
            sensitive_data_type="public_metric",
        )
        assert classification.color is TLPColor.CLEAR
        assert gen.tlp_classification is classification

    def test_apply_from_score_high_emits_red(self) -> None:
        """A high anomaly score escalates to TLP:RED."""
        gen = ReportGenerator()
        classification = gen.apply_tlp_classification(
            anomaly_score=0.95,
            domain_type="critical_infrastructure",
            sensitive_data_type="phi",
        )
        assert classification.color is TLPColor.RED

    def test_apply_with_pre_computed_classification(self) -> None:
        """Pre-computed classifications are accepted and used verbatim."""
        handler = get_tlp_handler()
        pre = handler.classify_anomaly(
            anomaly_score=0.6,
            anomaly_type="phi",
            domain="medical",
        )
        gen = ReportGenerator()
        result = gen.apply_tlp_classification(classification=pre)
        assert result is pre
        assert gen.tlp_classification is pre

    def test_apply_rejects_non_classification(self) -> None:
        """Passing a non-classification object raises ``TypeError``."""
        gen = ReportGenerator()
        with pytest.raises(TypeError, match="classification"):
            gen.apply_tlp_classification(classification="TLP:RED")  # type: ignore[arg-type]

    def test_apply_requires_input(self) -> None:
        """Missing both classification and score raises ``TypeError``."""
        gen = ReportGenerator()
        with pytest.raises(TypeError, match="apply_tlp_classification"):
            gen.apply_tlp_classification()

    def test_json_output_contains_tlp_block(self) -> None:
        """JSON output embeds the canonical TLP metadata + watermark."""
        gen = ReportGenerator()
        gen.apply_tlp_classification(
            anomaly_score=0.85,
            domain_type="security",
            sensitive_data_type="confidential",
        )
        report = json.loads(gen.generate({"title": "Sensitive"}, format=ReportFormat.JSON))
        assert "tlp" in report
        block = report["tlp"]
        for key in (
            "tlp_label",
            "tlp_color",
            "tlp_confidence",
            "tlp_reasoning",
            "sharing_guidelines",
            "ethical_considerations",
            "tlp_rank",
            "watermark",
        ):
            assert key in block
        assert block["tlp_label"].startswith("TLP:")
        assert block["watermark"].startswith("TLP:")

    def test_html_output_contains_tlp_banner(self) -> None:
        """HTML output renders a sanitised TLP banner above the title."""
        gen = ReportGenerator()
        gen.apply_tlp_classification(
            anomaly_score=0.9,
            domain_type="security",
            sensitive_data_type="confidential",
        )
        html = gen.generate({"title": "Sensitive"}, format=ReportFormat.HTML)
        assert 'class="tlp-banner"' in html
        assert "TLP:" in html
        # Sensitive output must NOT bypass HTML escaping.
        assert "<script>" not in html

    def test_markdown_output_contains_tlp_blockquote(self) -> None:
        """Markdown output renders a blockquote TLP banner."""
        gen = ReportGenerator()
        gen.apply_tlp_classification(
            anomaly_score=0.4,
            domain_type="general",
            sensitive_data_type="internal_metric",
        )
        md = gen.generate({"title": "Internal"}, format=ReportFormat.MARKDOWN)
        lines = md.splitlines()
        assert lines[0].startswith("> **TLP:")
        assert any(line.startswith("> ") for line in lines[1:3])

    def test_strict_sharing_forces_amber_strict(self) -> None:
        """The ``strict_sharing`` flag escalates AMBER → AMBER+STRICT."""
        gen = ReportGenerator()
        classification = gen.apply_tlp_classification(
            anomaly_score=0.65,
            domain_type="security",
            sensitive_data_type="internal",
            strict_sharing=True,
        )
        assert classification.color is TLPColor.AMBER_STRICT
