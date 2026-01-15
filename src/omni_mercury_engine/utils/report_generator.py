"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

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
"""

from __future__ import annotations


"""
Report Generator - Plain English Auto-Reporting

Automated report generation for non-technical users:
- Plain English summaries
- PDF report generation
- Email notifications
- Executive dashboards
- CSV/Excel exports

"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from html import escape
from typing import Any


class ReportFormat(str, Enum):
    """Report output format enumeration."""

    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    PDF = "pdf"


@dataclass
class AnomalyReport:
    """Anomaly detection report dataclass."""

    title: str
    timestamp: datetime
    anomaly_count: int
    total_datapoints: int

    @property
    def anomaly_rate(self) -> float:
        """Calculate anomaly rate as ratio of anomalies to total datapoints."""
        if self.total_datapoints == 0:
            return 0.0
        return self.anomaly_count / self.total_datapoints

    @property
    def severity(self) -> str:
        """Calculate severity level based on anomaly rate."""
        rate = self.anomaly_rate
        if rate >= 0.15:
            return "critical"
        elif rate >= 0.08:
            return "high"
        elif rate >= 0.03:
            return "medium"
        return "low"


@dataclass
class ReportSection:
    """Report section with optional subsections."""

    title: str
    content: str
    subsections: list[ReportSection] | None = None


@dataclass
class ExecutiveSummary:
    """Executive summary for reports."""

    key_findings: list[str]
    risk_assessment: str
    recommendations: list[str]
    confidence_score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert executive summary to dictionary."""
        return {
            "key_findings": self.key_findings,
            "risk_assessment": self.risk_assessment,
            "recommendations": self.recommendations,
            "confidence_score": self.confidence_score,
        }


@dataclass
class TechnicalDetails:
    """Technical details for reports."""

    methodology: str
    algorithms_used: list[str]
    parameters: dict[str, Any]
    data_sources: list[str]


class ReportGenerator:
    """General-purpose report generator with multiple format support."""

    def __init__(self, template: str | None = None) -> None:
        self._template = template
        self._sections: list[ReportSection] = []
        self._charts: dict[str, dict[str, Any]] = {}
        self._tables: dict[str, dict[str, Any]] = {}
        self._metadata: dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)

    def add_section(self, title: str, content: str) -> None:
        """Add a section to the report."""
        self._sections.append(ReportSection(title=title, content=content))

    def add_chart(self, name: str, chart_data: dict[str, Any]) -> None:
        """Add a chart placeholder to the report."""
        self._charts[name] = chart_data

    def add_table(self, name: str, table_data: dict[str, Any]) -> None:
        """Add a table to the report."""
        self._tables[name] = table_data

    def set_metadata(self, **kwargs: Any) -> None:
        """Set report metadata."""
        self._metadata.update(kwargs)

    def generate(self, data: dict[str, Any], format: ReportFormat) -> str:
        """Generate report in specified format."""
        if format == ReportFormat.JSON:
            return self._generate_json(data)
        elif format == ReportFormat.HTML:
            return self._generate_html(data)
        elif format in (ReportFormat.MARKDOWN, ReportFormat.PDF):
            return self._generate_markdown(data)
        return self._generate_json(data)

    def _generate_json(self, data: dict[str, Any]) -> str:
        """Generate JSON report."""
        output = dict(data)
        output["generated_at"] = datetime.now().isoformat()
        if self._metadata:
            output["metadata"] = self._metadata
        if self._sections:
            output["sections"] = [{"title": s.title, "content": s.content} for s in self._sections]
        if self._charts:
            output["charts"] = self._charts
        if self._tables:
            output["tables"] = self._tables
        return json.dumps(output, indent=2, default=str)

    def _generate_html(self, data: dict[str, Any]) -> str:
        """Generate HTML report with sanitized content."""
        title = escape(str(data.get("title", "Report")))
        content = escape(str(data.get("content", "")))

        sections_html = ""
        for section in self._sections:
            sections_html += f"<h2>{escape(section.title)}</h2>\n"
            sections_html += f"<p>{escape(section.content)}</p>\n"

        tables_html = ""
        for name, table_data in self._tables.items():
            headers = table_data.get("headers", [])
            rows = table_data.get("rows", [])
            tables_html += f"<h3>{escape(name)}</h3>\n<table border='1'>\n"
            tables_html += (
                "<tr>" + "".join(f"<th>{escape(str(h))}</th>" for h in headers) + "</tr>\n"
            )
            for row in rows:
                tables_html += (
                    "<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>\n"
                )
            tables_html += "</table>\n"

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 8px; text-align: left; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>{content}</p>
    {sections_html}
    {tables_html}
</body>
</html>"""

    def _generate_markdown(self, data: dict[str, Any]) -> str:
        """Generate Markdown report."""
        if self._template:
            try:
                return self._template.replace("{{ title }}", str(data.get("title", "")))
            except (TypeError, AttributeError):
                # Template substitution failed; fall back to default generation
                pass

        lines = []
        title = data.get("title", "Report")
        lines.append(f"# {title}")
        lines.append("")

        if data.get("content"):
            lines.append(str(data["content"]))
            lines.append("")

        for section in self._sections:
            lines.append(f"## {section.title}")
            lines.append(section.content)
            lines.append("")

        if data.get("sections"):
            for section in data["sections"]:
                lines.append(f"## {section.get('heading', section.get('title', ''))}")
                lines.append(section.get("content", ""))
                lines.append("")

        return "\n".join(lines)

    def export(self, data: dict[str, Any], path: str, format: ReportFormat) -> None:
        """Export report to file, creating directories if needed."""
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        content = self.generate(data, format)
        with open(path, "w") as f:
            f.write(content)


@dataclass
class ReportConfig:
    """Report configuration"""

    title: str
    report_type: str
    include_recommendations: bool = True
    include_visualizations: bool = False
    format: str = "text"  # text, pdf, html, json
    output_path: str | None = None


class PlainEnglishReportGenerator:
    """
    Generate plain English reports from analysis results.

    Converts technical output to human-readable summaries.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def generate_medical_report(
        self, results: dict[str, Any], config: ReportConfig | None = None
    ) -> str:
        """
        Generate plain English medical report.

        Args:
            results: Medical analysis results
            config: Report configuration

        Returns:
            Plain English report text
        """
        if config is None:
            config = ReportConfig(title="Medical Analysis Report", report_type="medical")

        report_lines = []

        report_lines.append("=" * 80)
        report_lines.append(f"{config.title.upper()}")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80)
        report_lines.append("")

        subspecialty = results.get("subspecialty", "general")
        report_lines.append(f"SUBSPECIALTY: {subspecialty.upper()}")
        report_lines.append("")

        if "cardiac_risk_detected" in results:
            risk_status = "DETECTED" if results["cardiac_risk_detected"] else "NORMAL"
            report_lines.append(f"CARDIAC RISK STATUS: {risk_status}")

            if results.get("arrhythmia_type"):
                rhythm = results["arrhythmia_type"].replace("_", " ").title()
                report_lines.append(f"Heart Rhythm: {rhythm}")

            if results.get("mi_risk") is not None:
                mi_risk_pct = results["mi_risk"] * 100
                report_lines.append(f"Myocardial Infarction Risk: {mi_risk_pct:.1f}%")

            if results.get("heart_failure_risk") is not None:
                hf_risk_pct = results["heart_failure_risk"] * 100
                report_lines.append(f"Heart Failure Risk: {hf_risk_pct:.1f}%")

        elif "sepsis_detected" in results:
            status = "DETECTED" if results["sepsis_detected"] else "NOT DETECTED"
            report_lines.append(f"SEPSIS STATUS: {status}")

            if results.get("sepsis_stage"):
                stage = results["sepsis_stage"].replace("_", " ").upper()
                report_lines.append(f"Sepsis Stage: {stage}")

            if results.get("sofa_score") is not None:
                report_lines.append(f"SOFA Score: {results['sofa_score']}/24")

            if results.get("organ_dysfunctions"):
                report_lines.append("Organ Dysfunctions:")
                for organ in results["organ_dysfunctions"]:
                    report_lines.append(f"  - {organ.upper()}")

        elif "emergency_detected" in results:
            status = "DETECTED" if results["emergency_detected"] else "NOT DETECTED"
            report_lines.append(f"NEUROLOGICAL EMERGENCY: {status}")

            if results.get("emergency_type"):
                etype = results["emergency_type"].replace("_", " ").title()
                report_lines.append(f"Emergency Type: {etype}")

            if results.get("stroke_detected"):
                report_lines.append("STROKE DETECTED: YES")

        if config.include_recommendations and "recommendations" in results:
            report_lines.append("")
            report_lines.append("CLINICAL RECOMMENDATIONS:")
            report_lines.append("-" * 80)
            for i, rec in enumerate(results["recommendations"], 1):
                report_lines.append(f"{i}. {rec}")

        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("⚠️  DISCLAIMER: This is a simulation-based analysis tool.")
        report_lines.append(
            "Always consult qualified medical professionals for diagnosis and treatment."
        )
        report_lines.append("=" * 80)

        return "\n".join(report_lines)

    def generate_security_report(
        self, results: dict[str, Any], config: ReportConfig | None = None
    ) -> str:
        """Generate plain English security report"""

        if config is None:
            config = ReportConfig(title="Security Intelligence Report", report_type="security")

        report_lines = []

        report_lines.append("=" * 80)
        report_lines.append(f"{config.title.upper()}")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80)
        report_lines.append("")

        intel_type = results.get("intel_type", "general").upper()
        report_lines.append(f"INTELLIGENCE TYPE: {intel_type}")
        report_lines.append("")

        if "threat_detected" in results:
            status = "DETECTED" if results["threat_detected"] else "CLEAR"
            threat_level = results.get("threat_severity", "unknown").upper()

            report_lines.append(f"THREAT STATUS: {status}")
            report_lines.append(f"Threat Level: {threat_level}")

            if results.get("apt_group"):
                report_lines.append(f"APT Attribution: {results['apt_group'].upper()}")

            if results.get("malware_family"):
                report_lines.append(f"Malware Family: {results['malware_family'].upper()}")

            if results.get("c2_indicators"):
                report_lines.append("\nCommand & Control Indicators:")
                for indicator in results["c2_indicators"]:
                    report_lines.append(f"  - {indicator}")

        if "anomaly_detected" in results:
            status = "DETECTED" if results["anomaly_detected"] else "CLEAR"
            report_lines.append(f"NETWORK ANOMALY: {status}")

            if results.get("anomaly_type"):
                atype = results["anomaly_type"].replace("_", " ").title()
                report_lines.append(f"Anomaly Type: {atype}")

        if "emanation_detected" in results:
            status = "DETECTED" if results["emanation_detected"] else "CLEAR"
            report_lines.append(f"EM EMANATIONS: {status}")

            if results.get("threat_level"):
                report_lines.append(f"TEMPEST Threat Level: {results['threat_level'].upper()}")

        if config.include_recommendations:
            rec_key = next((k for k in results if "recommendation" in k.lower()), None)
            if rec_key and results.get(rec_key):
                report_lines.append("")
                report_lines.append("RECOMMENDED ACTIONS:")
                report_lines.append("-" * 80)
                for i, rec in enumerate(results[rec_key], 1):
                    report_lines.append(f"{i}. {rec}")

        report_lines.append("")
        report_lines.append("=" * 80)

        return "\n".join(report_lines)

    def generate_humanitarian_report(
        self, results: dict[str, Any], config: ReportConfig | None = None
    ) -> str:
        """Generate plain English humanitarian report"""

        if config is None:
            config = ReportConfig(title="Humanitarian Crisis Report", report_type="humanitarian")

        report_lines = []

        report_lines.append("=" * 80)
        report_lines.append(f"{config.title.upper()}")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80)
        report_lines.append("")

        crisis_type = results.get("crisis_type", "general").upper()
        report_lines.append(f"CRISIS TYPE: {crisis_type}")
        report_lines.append("")

        if "anomaly_detected" in results:
            status = "DETECTED" if results["anomaly_detected"] else "NORMAL"
            report_lines.append(f"CRISIS STATUS: {status}")

            if results.get("severity"):
                severity_pct = (
                    float(results["severity"]) * 100
                    if isinstance(results["severity"], (int, float))
                    else 0
                )
                report_lines.append(f"Severity Level: {severity_pct:.1f}%")

            if results.get("humanitarian_impact"):
                impact = results["humanitarian_impact"].upper()
                report_lines.append(f"Humanitarian Impact: {impact}")

        report_lines.append("")
        report_lines.append("=" * 80)

        return "\n".join(report_lines)


class PDFReportGenerator:
    """
    PDF report generation (requires reportlab).

    Creates professional PDF reports from analysis results.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def generate_pdf(self, text_report: str, output_path: str) -> bool:
        """
        Generate PDF from text report.

        Args:
            text_report: Plain text report
            output_path: Output PDF file path

        Returns:
            Success status
        """
        try:
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.lib.styles import getSampleStyleSheet
                from reportlab.lib.units import inch
                from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
            except ImportError:
                self.logger.warning("reportlab not installed - saving as text instead")
                with open(output_path.replace(".pdf", ".txt"), "w") as f:
                    f.write(text_report)
                return False

            doc = SimpleDocTemplate(output_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            for line in text_report.split("\n"):
                if line.strip():
                    if line.startswith("="):
                        story.append(Spacer(1, 0.2 * inch))
                    else:
                        para = Paragraph(line, styles["Normal"])
                        story.append(para)
                        story.append(Spacer(1, 0.1 * inch))

            doc.build(story)
            self.logger.info(f"PDF report generated: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to generate PDF: {e}")
            return False


class EmailReportSender:
    """
    Email report sender (requires smtplib).

    Sends analysis reports via email.
    """

    def __init__(self, smtp_config: dict[str, str] | None = None) -> None:
        self.logger = logging.getLogger(__name__)
        self.smtp_config = smtp_config or {}

    def send_email_report(
        self, report: str, recipient: str, subject: str = "Mercury Agent ♱ Analysis Report"
    ) -> bool:
        """
        Send report via email.

        Args:
            report: Report text
            recipient: Recipient email address
            subject: Email subject

        Returns:
            Success status
        """
        try:
            import smtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText

            smtp_server = self.smtp_config.get("server", "smtp.gmail.com")
            smtp_port = int(self.smtp_config.get("port", 587))
            sender_email = self.smtp_config.get("sender_email")
            sender_password = self.smtp_config.get("password")

            if not sender_email or not sender_password:
                self.logger.warning("SMTP credentials not configured")
                return False

            msg = MIMEMultipart()
            msg["From"] = sender_email
            msg["To"] = recipient
            msg["Subject"] = subject

            msg.attach(MIMEText(report, "plain"))

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)

            self.logger.info(f"Email report sent to {recipient}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return False


class ReportManager:
    """
    Unified report management system.

    Coordinates text, PDF, and email report generation.
    """

    def __init__(self, smtp_config: dict[str, str] | None = None) -> None:
        self.text_generator = PlainEnglishReportGenerator()
        self.pdf_generator = PDFReportGenerator()
        self.email_sender = EmailReportSender(smtp_config)
        self.logger = logging.getLogger(__name__)

    def generate_report(self, results: dict[str, Any], config: ReportConfig) -> str:
        """
        Generate report in specified format.

        Args:
            results: Analysis results
            config: Report configuration

        Returns:
            Generated report (text format)
        """
        if config.report_type == "medical":
            text_report = self.text_generator.generate_medical_report(results, config)
        elif config.report_type == "security":
            text_report = self.text_generator.generate_security_report(results, config)
        elif config.report_type == "humanitarian":
            text_report = self.text_generator.generate_humanitarian_report(results, config)
        else:
            text_report = json.dumps(results, indent=2)

        if config.output_path:
            if config.format == "pdf":
                self.pdf_generator.generate_pdf(text_report, config.output_path)
            elif config.format == "html":
                self._save_html_report(text_report, config.output_path)
            elif config.format == "json":
                with open(config.output_path, "w") as f:
                    json.dump(results, f, indent=2, default=str)
            else:
                with open(config.output_path, "w") as f:
                    f.write(text_report)

        return text_report

    def send_email_report(
        self, results: dict[str, Any], recipient: str, config: ReportConfig
    ) -> bool:
        """
        Generate and email report.

        Args:
            results: Analysis results
            recipient: Recipient email
            config: Report configuration

        Returns:
            Success status
        """
        report = self.generate_report(results, config)
        return self.email_sender.send_email_report(report, recipient, config.title)

    def _save_html_report(self, text_report: str, output_path: str) -> None:
        """Save report as HTML"""

        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Mercury Agent ♱ Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                pre {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <pre>{text_report}</pre>
        </body>
        </html>
        """

        with open(output_path, "w") as f:
            f.write(html_template)
