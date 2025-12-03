"""
OMNI ♱ AVA (O♱A)
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
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class ReportConfig:
    """Report configuration"""

    title: str
    report_type: str
    include_recommendations: bool = True
    include_visualizations: bool = False
    format: str = "text"  # text, pdf, html, json
    output_path: Optional[str] = None


class PlainEnglishReportGenerator:
    """
    Generate plain English reports from analysis results.

    Converts technical output to human-readable summaries.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_medical_report(
        self, results: Dict[str, Any], config: Optional[ReportConfig] = None
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
        self, results: Dict[str, Any], config: Optional[ReportConfig] = None
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
        self, results: Dict[str, Any], config: Optional[ReportConfig] = None
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

    def __init__(self):
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

    def __init__(self, smtp_config: Optional[Dict[str, str]] = None):
        self.logger = logging.getLogger(__name__)
        self.smtp_config = smtp_config or {}

    def send_email_report(
        self, report: str, recipient: str, subject: str = "OMNI ♱ AVA Analysis Report"
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

    def __init__(self, smtp_config: Optional[Dict[str, str]] = None):
        self.text_generator = PlainEnglishReportGenerator()
        self.pdf_generator = PDFReportGenerator()
        self.email_sender = EmailReportSender(smtp_config)
        self.logger = logging.getLogger(__name__)

    def generate_report(self, results: Dict[str, Any], config: ReportConfig) -> str:
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
        self, results: Dict[str, Any], recipient: str, config: ReportConfig
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
            <title>OMNI ♱ AVA Report</title>
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
