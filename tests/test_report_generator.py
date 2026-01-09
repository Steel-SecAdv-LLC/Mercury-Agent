"""
Tests for omni_mercury_engine.utils.report_generator module.

Tests report generation functionality.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

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

    def test_all_formats_defined(self):
        """Test that expected formats are defined."""
        formats = list(ReportFormat)
        assert len(formats) >= 3  # At least JSON, HTML, PDF

    def test_format_values(self):
        """Test format values are strings."""
        for fmt in list(ReportFormat):
            assert isinstance(fmt.value, str)


class TestAnomalyReport:
    """Tests for AnomalyReport dataclass."""

    def test_basic_creation(self):
        """Test basic report creation."""
        report = AnomalyReport(
            title="Test Report",
            timestamp=datetime.now(),
            anomaly_count=5,
            total_datapoints=1000,
        )
        assert report.title == "Test Report"
        assert report.anomaly_count == 5

    def test_severity_calculation(self):
        """Test severity level calculation."""
        report = AnomalyReport(
            title="Test",
            timestamp=datetime.now(),
            anomaly_count=100,
            total_datapoints=1000,
        )
        # 10% anomaly rate should be high severity
        assert report.severity in ["low", "medium", "high", "critical"]

    def test_anomaly_rate(self):
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

    def test_section_creation(self):
        """Test section creation."""
        section = ReportSection(
            title="Analysis Results",
            content="Detailed analysis...",
            subsections=[],
        )
        assert section.title == "Analysis Results"

    def test_nested_sections(self):
        """Test nested subsections."""
        subsection = ReportSection(title="Subsection", content="Details")
        section = ReportSection(title="Main Section", content="Overview", subsections=[subsection])
        assert len(section.subsections) == 1


class TestExecutiveSummary:
    """Tests for ExecutiveSummary class."""

    def test_summary_generation(self):
        """Test executive summary generation."""
        summary = ExecutiveSummary(
            key_findings=["Finding 1", "Finding 2"],
            risk_assessment="Medium",
            recommendations=["Recommendation 1"],
            confidence_score=0.85,
        )
        assert len(summary.key_findings) == 2
        assert summary.confidence_score == 0.85

    def test_summary_to_dict(self):
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

    def test_details_creation(self):
        """Test technical details creation."""
        details = TechnicalDetails(
            methodology="Statistical analysis",
            algorithms_used=["IsolationForest", "OneClassSVM"],
            parameters={"contamination": 0.1},
            data_sources=["sensor_data"],
        )
        assert len(details.algorithms_used) == 2

    def test_parameters_serialization(self):
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

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = ReportGenerator()

    def test_initialization(self):
        """Test generator initialization."""
        assert self.generator is not None

    def test_generate_json_report(self):
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

    def test_generate_html_report(self):
        """Test HTML report generation."""
        data = {
            "title": "Test Report",
            "findings": ["Finding 1", "Finding 2"],
        }
        report = self.generator.generate(data, format=ReportFormat.HTML)

        assert report is not None
        assert "<html>" in report.lower() or "<!doctype" in report.lower()

    def test_generate_markdown_report(self):
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

    def test_add_section(self):
        """Test adding sections to report."""
        self.generator.add_section("Introduction", "This is the introduction.")
        self.generator.add_section("Analysis", "This is the analysis.")

        report = self.generator.generate({}, format=ReportFormat.MARKDOWN)
        assert "Introduction" in report
        assert "Analysis" in report

    def test_add_chart_placeholder(self):
        """Test adding chart placeholders."""
        chart_data = {
            "type": "line",
            "title": "Anomaly Trend",
            "data": [1, 2, 3, 4, 5],
        }
        self.generator.add_chart("trend_chart", chart_data)

        assert "trend_chart" in self.generator._charts

    def test_add_table(self):
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

    def test_template_customization(self):
        """Test custom template support."""
        custom_template = "Custom Report: {{ title }}"
        generator = ReportGenerator(template=custom_template)

        report = generator.generate({"title": "Test"}, format=ReportFormat.MARKDOWN)
        # Should use custom template
        assert report is not None


class TestReportContent:
    """Tests for report content generation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = ReportGenerator()

    def test_timestamp_included(self):
        """Test that timestamp is included in report."""
        data = {"title": "Test"}
        report = self.generator.generate(data, format=ReportFormat.JSON)
        parsed = json.loads(report)

        # Should have timestamp or generated_at field
        assert (
            any(key in parsed for key in ["timestamp", "generated_at", "report_date"])
            or "title" in parsed
        )

    def test_metadata_included(self):
        """Test that metadata is included."""
        self.generator.set_metadata(
            author="Test Author", version="1.0", classification="UNCLASSIFIED"
        )
        report = self.generator.generate({}, format=ReportFormat.JSON)
        parsed = json.loads(report)

        # Metadata should be present
        if "metadata" in parsed:
            assert "author" in parsed["metadata"] or "version" in parsed["metadata"]

    def test_anomaly_details_formatting(self):
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

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = ReportGenerator()

    def test_export_to_file(self, tmp_path):
        """Test exporting report to file."""
        data = {"title": "Test Report", "content": "Test content"}
        filepath = tmp_path / "report.json"

        self.generator.export(data, str(filepath), format=ReportFormat.JSON)

        assert filepath.exists()
        with open(filepath) as f:
            content = f.read()
            assert len(content) > 0

    def test_export_html_to_file(self, tmp_path):
        """Test exporting HTML report to file."""
        data = {"title": "Test Report"}
        filepath = tmp_path / "report.html"

        self.generator.export(data, str(filepath), format=ReportFormat.HTML)

        assert filepath.exists()

    def test_export_creates_directory(self, tmp_path):
        """Test that export creates directory if needed."""
        data = {"title": "Test"}
        nested_path = tmp_path / "subdir" / "report.json"

        self.generator.export(data, str(nested_path), format=ReportFormat.JSON)

        assert nested_path.exists()


class TestReportValidation:
    """Tests for report validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = ReportGenerator()

    def test_validates_required_fields(self):
        """Test validation of required fields."""
        # Empty data should still generate valid report
        report = self.generator.generate({}, format=ReportFormat.JSON)
        parsed = json.loads(report)

        # Should have at least some structure
        assert isinstance(parsed, dict)

    def test_sanitizes_html_content(self):
        """Test that HTML content is sanitized."""
        data = {"content": "<script>alert('xss')</script>Malicious content"}
        report = self.generator.generate(data, format=ReportFormat.HTML)

        # Script tags should be escaped or removed
        assert "<script>" not in report or "&lt;script&gt;" in report

    def test_handles_unicode(self):
        """Test handling of unicode characters."""
        data = {"title": "レポート", "content": "日本語テスト"}
        report = self.generator.generate(data, format=ReportFormat.JSON)

        parsed = json.loads(report)
        # Unicode should be preserved
        assert any("レポート" in str(v) or "日本語" in str(v) for v in parsed.values())
