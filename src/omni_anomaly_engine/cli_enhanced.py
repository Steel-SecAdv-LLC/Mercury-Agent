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
Enhanced Command-Line Interface for OMNI ♱ AVA

Comprehensive CLI with medical, security, humanitarian, and accessibility features.
"""

import click
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

from omni_anomaly_engine import OmniAnomalyEngine


@click.group()
@click.version_option(version="1.0.0")
def main() -> None:
    """OMNI ♱ AVA: ML-Centric Anomaly Detection Framework for Humanitarian Impact"""
    pass


@main.command()
@click.option("--input", "-i", required=True, help="Input data file (CSV/JSON)")
@click.option("--detector", "-d", default="fusion", help="Detector type")
@click.option("--output", "-o", help="Output file for results")
@click.option("--threshold", "-t", default=0.5, type=float, help="Anomaly threshold")
@click.option("--report", "-r", is_flag=True, help="Generate plain English report")
def detect(input: str, detector: str, output: str, threshold: float, report: bool) -> None:
    """General anomaly detection"""
    engine = OmniAnomalyEngine(mode=detector)
    data = _load_data(input)

    if detector == "fusion":
        results = engine.detect_with_fusion(data)
    else:
        results = engine.detect(data, detector_types=[detector])

    if report:
        _print_plain_english_report(results, "General Anomaly Detection")

    _save_or_print_results(results, output)


@main.command("run-medical")
@click.option(
    "--subspecialty",
    type=click.Choice(["cardiology", "neurocritical", "sepsis", "general"]),
    default="general",
    help="Medical subspecialty",
)
@click.option("--ecg-file", help="ECG signal file (for cardiology)")
@click.option("--vitals-file", help="Vital signs time series file")
@click.option("--biomarkers-file", help="Cardiac biomarkers JSON file")
@click.option("--patient-file", help="Patient demographics JSON file")
@click.option("--output", "-o", help="Output file for results")
@click.option("--report", "-r", is_flag=True, help="Generate clinical report")
def run_medical(
    subspecialty: str,
    ecg_file: Optional[str],
    vitals_file: Optional[str],
    biomarkers_file: Optional[str],
    patient_file: Optional[str],
    output: Optional[str],
    report: bool,
) -> None:
    """Run medical subspecialty analysis"""

    click.echo(f"🏥 Running Medical Analysis - Subspecialty: {subspecialty.upper()}")

    patient_data = {}

    if ecg_file:
        patient_data["ecg_signal"] = _load_data(ecg_file)
        click.echo(f"  ✓ Loaded ECG signal from {ecg_file}")

    if vitals_file:
        patient_data["vital_signs_sequence"] = _load_data(vitals_file)
        click.echo(f"  ✓ Loaded vital signs from {vitals_file}")

    if biomarkers_file:
        with open(biomarkers_file) as f:
            patient_data["biomarkers"] = json.load(f)
        click.echo(f"  ✓ Loaded biomarkers from {biomarkers_file}")

    if patient_file:
        with open(patient_file) as f:
            patient_data["demographics"] = json.load(f)
        click.echo(f"  ✓ Loaded patient data from {patient_file}")

    results = _run_medical_subspecialty(subspecialty, patient_data)

    if report:
        _print_medical_report(results, subspecialty)

    _save_or_print_results(results, output)


@main.command("run-security")
@click.option(
    "--intel-type",
    type=click.Choice(["cybint", "traffic", "tempest", "fusion", "general"]),
    default="general",
    help="Security intelligence type",
)
@click.option("--threat-file", help="Threat indicators file")
@click.option("--network-file", help="Network traffic capture file")
@click.option("--spectrum-file", help="RF spectrum data file")
@click.option("--output", "-o", help="Output file for results")
@click.option("--report", "-r", is_flag=True, help="Generate security report")
def run_security(
    intel_type: str,
    threat_file: Optional[str],
    network_file: Optional[str],
    spectrum_file: Optional[str],
    output: Optional[str],
    report: bool,
) -> None:
    """Run security intelligence analysis"""

    click.echo(f"🔒 Running Security Analysis - Type: {intel_type.upper()}")

    security_data = {}

    if threat_file:
        security_data["threat_features"] = _load_data(threat_file)
        click.echo(f"  ✓ Loaded threat indicators from {threat_file}")

    if network_file:
        with open(network_file) as f:
            security_data["network_data"] = json.load(f)
        click.echo(f"  ✓ Loaded network data from {network_file}")

    if spectrum_file:
        with open(spectrum_file) as f:
            security_data["spectrum_data"] = json.load(f)
        click.echo(f"  ✓ Loaded spectrum data from {spectrum_file}")

    results = _run_security_analysis(intel_type, security_data)

    if report:
        _print_security_report(results, intel_type)

    _save_or_print_results(results, output)


@main.command("run-humanitarian")
@click.option(
    "--crisis-type",
    type=click.Choice(["disaster", "pandemic", "refugee", "missing_persons", "general"]),
    default="general",
    help="Humanitarian crisis type",
)
@click.option("--data-file", required=True, help="Crisis data file")
@click.option("--output", "-o", help="Output file for results")
@click.option("--report", "-r", is_flag=True, help="Generate humanitarian report")
def run_humanitarian(crisis_type: str, data_file: str, output: Optional[str], report: bool) -> None:
    """Run humanitarian crisis detection"""

    click.echo(f"🌍 Running Humanitarian Analysis - Crisis: {crisis_type.upper()}")

    crisis_data = _load_data(data_file)
    click.echo(f"  ✓ Loaded crisis data from {data_file}")

    results = _run_humanitarian_analysis(crisis_type, crisis_data)

    if report:
        _print_humanitarian_report(results, crisis_type)

    _save_or_print_results(results, output)


@main.command("run-schumann")
@click.option("--resonance-file", required=True, help="Schumann resonance data file")
@click.option("--seismic-file", help="Optional seismic data for correlation")
@click.option("--output", "-o", help="Output file for results")
@click.option("--report", "-r", is_flag=True, help="Generate analysis report")
def run_schumann(
    resonance_file: str, seismic_file: Optional[str], output: Optional[str], report: bool
) -> None:
    """Run Schumann resonance analysis for disaster precursors"""

    click.echo("🌐 Running Schumann Resonance Analysis")

    from omni_anomaly_engine.space.schumann_resonance import SchumannResonanceDetector

    detector = SchumannResonanceDetector()
    resonance_data = _load_data(resonance_file)

    results = detector.detect_anomaly(resonance_data)

    if seismic_file:
        seismic_data = _load_data(seismic_file)
        results["seismic_correlation"] = _correlate_schumann_seismic(resonance_data, seismic_data)

    if report:
        click.echo("\n" + "=" * 60)
        click.echo("SCHUMANN RESONANCE ANALYSIS REPORT")
        click.echo("=" * 60)
        click.echo(f"Anomaly Detected: {results.get('is_anomaly', False)}")
        click.echo(f"Confidence: {results.get('anomaly_score', 0):.2%}")
        if results.get("precursor_detected"):
            click.echo("\n⚠️  EARTHQUAKE/DISASTER PRECURSOR DETECTED")
        click.echo("=" * 60 + "\n")

    _save_or_print_results(results, output)


@main.command("run-chemistry")
@click.option(
    "--analysis-type",
    type=click.Choice(["isotope", "rare_earth", "general"]),
    default="general",
    help="Chemistry analysis type",
)
@click.option("--sample-file", required=True, help="Chemical sample data file")
@click.option("--output", "-o", help="Output file for results")
@click.option("--report", "-r", is_flag=True, help="Generate analysis report")
def run_chemistry(
    analysis_type: str, sample_file: str, output: Optional[str], report: bool
) -> None:
    """Run chemistry/isotope analysis"""

    click.echo(f"🧪 Running Chemistry Analysis - Type: {analysis_type.upper()}")

    sample_data = _load_data(sample_file)
    click.echo(f"  ✓ Loaded sample data from {sample_file}")

    results = _run_chemistry_analysis(analysis_type, sample_data)

    if report:
        _print_chemistry_report(results, analysis_type)

    _save_or_print_results(results, output)


@main.command()
@click.option("--reference", "-r", required=True, help="Reference face image")
@click.option("--test", "-t", help="Test face image to match")
def biometric(reference: str, test: str) -> None:
    """Biometric face matching"""
    engine = OmniAnomalyEngine()
    result = engine.detect_biometric(reference, test)
    click.echo(json.dumps(result, indent=2, default=str))


@main.command()
@click.option("--data", "-d", required=True, help="Training data directory")
@click.option("--output", "-o", required=True, help="Model output path")
@click.option("--epochs", "-e", default=50, type=int, help="Training epochs")
def train(data: str, output: str, epochs: int) -> None:
    """Train fusion model"""
    engine = OmniAnomalyEngine(mode="fusion")
    click.echo(f"Training fusion model on {data}...")
    engine.train_fusion_model(data, epochs=epochs)
    engine.save_model(output)
    click.echo(f"Model saved to {output}")


@main.command("demo")
@click.option(
    "--type",
    "demo_type",
    type=click.Choice(["medical", "security", "humanitarian", "all"]),
    default="all",
    help="Demo type to run",
)
def run_demo(demo_type: str) -> None:
    """Run interactive demos (no-code)"""

    click.echo("🚀 OMNI ♱ AVA Interactive Demo")
    click.echo("=" * 60)

    if demo_type in ["medical", "all"]:
        click.echo("\n[MEDICAL DEMO] Sepsis Detection Simulation")
        _run_sepsis_demo()

    if demo_type in ["security", "all"]:
        click.echo("\n[SECURITY DEMO] Cyber Threat Detection Simulation")
        _run_cybersecurity_demo()

    if demo_type in ["humanitarian", "all"]:
        click.echo("\n[HUMANITARIAN DEMO] Disaster Response Simulation")
        _run_humanitarian_demo()


def _run_medical_subspecialty(subspecialty: str, patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run medical subspecialty analysis"""

    if subspecialty == "cardiology":
        from omni_anomaly_engine.medical.cardiology_predictor import CardiologyPredictor

        predictor = CardiologyPredictor()
        result = predictor.predict_cardiac_risk(patient_data)
        return {
            "subspecialty": "cardiology",
            "cardiac_risk_detected": result.cardiac_risk_detected,
            "arrhythmia_type": result.arrhythmia_type,
            "mi_risk": result.mi_risk,
            "recommendations": result.clinical_recommendations,
        }

    elif subspecialty == "neurocritical":
        from omni_anomaly_engine.medical.neurocritical_care import NeurocriticalCarePredictor

        predictor = NeurocriticalCarePredictor()
        result = predictor.predict_neurocritical_emergency(patient_data)
        return {
            "subspecialty": "neurocritical_care",
            "emergency_detected": result.neurological_emergency_detected,
            "emergency_type": result.emergency_type,
            "stroke_detected": result.stroke_detected,
            "recommendations": result.clinical_recommendations,
        }

    elif subspecialty == "sepsis":
        from omni_anomaly_engine.medical.sepsis_detector import SepsisDetector

        detector = SepsisDetector()
        result = detector.detect_sepsis(patient_data)
        return {
            "subspecialty": "sepsis",
            "sepsis_detected": result.sepsis_detected,
            "sepsis_stage": result.sepsis_stage,
            "sofa_score": result.sofa_score,
            "recommendations": result.clinical_recommendations,
        }

    else:
        from omni_anomaly_engine.medical.medical_cure_predictor import MedicalCurePredictor

        predictor = MedicalCurePredictor()
        result = predictor.predict_and_cure(patient_data)
        return {
            "subspecialty": "general_medical",
            "disease_risk_detected": result.disease_risk_detected,
            "disease_type": result.disease_type,
            "recommendations": result.recommendations,
        }


def _run_security_analysis(intel_type: str, security_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run security intelligence analysis"""

    if intel_type == "cybint":
        from omni_anomaly_engine.security.cybint_subprocessor import CYBINTSubProcessor

        processor = CYBINTSubProcessor()
        result = processor.process_cybint(security_data)
        return {
            "intel_type": "cybint",
            "threat_detected": result.threat_detected,
            "apt_group": result.apt_group,
            "malware_family": result.malware_family,
            "recommendations": result.recommended_actions,
        }

    elif intel_type == "traffic":
        from omni_anomaly_engine.security.traffic_analysis import TrafficAnalysisEngine

        engine = TrafficAnalysisEngine()
        result = engine.analyze_traffic(security_data)
        return {
            "intel_type": "traffic_analysis",
            "anomaly_detected": result.anomaly_detected,
            "anomaly_type": result.anomaly_type,
            "recommendations": result.recommended_actions,
        }

    elif intel_type == "tempest":
        from omni_anomaly_engine.security.tempest_detection import TEMPESTDetector

        detector = TEMPESTDetector()
        result = detector.detect_tempest_threats(security_data)
        return {
            "intel_type": "tempest",
            "emanation_detected": result.emanation_detected,
            "threat_level": result.threat_level,
            "countermeasures": result.countermeasures,
        }

    else:
        from omni_anomaly_engine.security.intelligence_fusion import IntelligenceFusionEngine

        engine = IntelligenceFusionEngine()
        result = engine.fuse_intelligence(security_data)
        return {
            "intel_type": "fusion",
            "threat_detected": result.threat_detected,
            "threat_level": result.threat_level,
            "recommendations": result.recommended_actions,
        }


def _run_humanitarian_analysis(crisis_type: str, crisis_data: np.ndarray) -> Dict[str, Any]:
    """Run humanitarian crisis analysis"""

    engine = OmniAnomalyEngine()
    results = engine.detect(crisis_data)

    return {
        "crisis_type": crisis_type,
        "anomaly_detected": results.get("is_anomaly", False),
        "severity": results.get("anomaly_score", 0),
        "humanitarian_impact": "high" if results.get("anomaly_score", 0) > 0.7 else "moderate",
    }


def _run_chemistry_analysis(analysis_type: str, sample_data: np.ndarray) -> Dict[str, Any]:
    """Run chemistry analysis"""

    from omni_anomaly_engine.models.chemistry import ChemistryAnomalyModel

    model = ChemistryAnomalyModel()
    result = model.predict(sample_data)

    return {
        "analysis_type": analysis_type,
        "anomaly_detected": result.get("is_anomaly", False),
        "confidence": result.get("anomaly_score", 0),
    }


def _correlate_schumann_seismic(
    resonance_data: np.ndarray, seismic_data: np.ndarray
) -> Dict[str, Any]:
    """Correlate Schumann resonance with seismic activity"""

    if len(resonance_data) != len(seismic_data):
        min_len = min(len(resonance_data), len(seismic_data))
        resonance_data = resonance_data[:min_len]
        seismic_data = seismic_data[:min_len]

    correlation = np.corrcoef(resonance_data.flatten(), seismic_data.flatten())[0, 1]

    return {
        "correlation_coefficient": float(correlation),
        "significant_correlation": abs(correlation) > 0.6,
        "precursor_likelihood": float(abs(correlation)),
    }


def _print_medical_report(results: Dict[str, Any], subspecialty: str) -> None:
    """Print plain English medical report"""

    click.echo("\n" + "=" * 60)
    click.echo(f"MEDICAL ANALYSIS REPORT - {subspecialty.upper()}")
    click.echo("=" * 60)

    for key, value in results.items():
        if key != "recommendations":
            click.echo(f"{key.replace('_', ' ').title()}: {value}")

    if "recommendations" in results:
        click.echo("\nCLINICAL RECOMMENDATIONS:")
        for rec in results["recommendations"]:
            click.echo(f"  • {rec}")

    click.echo("=" * 60 + "\n")


def _print_security_report(results: Dict[str, Any], intel_type: str) -> None:
    """Print plain English security report"""

    click.echo("\n" + "=" * 60)
    click.echo(f"SECURITY INTELLIGENCE REPORT - {intel_type.upper()}")
    click.echo("=" * 60)

    for key, value in results.items():
        if key != "recommendations":
            click.echo(f"{key.replace('_', ' ').title()}: {value}")

    if "recommendations" in results:
        click.echo("\nRECOMMENDED ACTIONS:")
        for rec in results["recommendations"]:
            click.echo(f"  • {rec}")

    click.echo("=" * 60 + "\n")


def _print_humanitarian_report(results: Dict[str, Any], crisis_type: str) -> None:
    """Print plain English humanitarian report"""

    click.echo("\n" + "=" * 60)
    click.echo(f"HUMANITARIAN CRISIS REPORT - {crisis_type.upper()}")
    click.echo("=" * 60)

    for key, value in results.items():
        click.echo(f"{key.replace('_', ' ').title()}: {value}")

    click.echo("=" * 60 + "\n")


def _print_chemistry_report(results: Dict[str, Any], analysis_type: str) -> None:
    """Print plain English chemistry report"""

    click.echo("\n" + "=" * 60)
    click.echo(f"CHEMISTRY ANALYSIS REPORT - {analysis_type.upper()}")
    click.echo("=" * 60)

    for key, value in results.items():
        click.echo(f"{key.replace('_', ' ').title()}: {value}")

    click.echo("=" * 60 + "\n")


def _print_plain_english_report(results: Dict[str, Any], title: str) -> None:
    """Print plain English analysis report"""

    click.echo("\n" + "=" * 60)
    click.echo(title.upper())
    click.echo("=" * 60)

    is_anomaly = results.get("is_anomaly", False)
    confidence = results.get("anomaly_prob", 0)

    if is_anomaly:
        click.echo(f"⚠️  ANOMALY DETECTED (Confidence: {confidence:.1%})")
    else:
        click.echo(f"✓ No anomaly detected (Confidence: {1-confidence:.1%})")

    click.echo("=" * 60 + "\n")


def _run_sepsis_demo() -> None:
    """Run interactive sepsis detection demo"""

    vital_signs = {
        "respiratory_rate_bpm": 24,
        "gcs_score": 13,
        "systolic_bp_mmhg": 95,
    }

    lab_values = {
        "platelets_k_ul": 80,
        "bilirubin_mg_dl": 2.5,
        "creatinine_mg_dl": 2.2,
        "pao2_fio2_ratio": 180,
        "mean_arterial_pressure": 60,
    }

    from omni_anomaly_engine.medical.sepsis_detector import SepsisDetector

    detector = SepsisDetector()
    result = detector.detect_sepsis(
        {
            "vital_signs": vital_signs,
            "laboratory_values": lab_values,
        }
    )

    click.echo(f"  Sepsis Detected: {result.sepsis_detected}")
    click.echo(f"  Sepsis Stage: {result.sepsis_stage}")
    click.echo(f"  SOFA Score: {result.sofa_score}")
    click.echo(f"  Recommendations: {', '.join(result.clinical_recommendations[:2])}")


def _run_cybersecurity_demo() -> None:
    """Run interactive cybersecurity demo"""

    threat_features = np.random.randn(256) * 0.5

    from omni_anomaly_engine.security.cybint_subprocessor import CYBINTSubProcessor

    processor = CYBINTSubProcessor()
    result = processor.process_cybint({"threat_features": threat_features})

    click.echo(f"  Threat Detected: {result.threat_detected}")
    click.echo(f"  Threat Severity: {result.threat_severity}")
    click.echo(f"  APT Group: {result.apt_group or 'Unknown'}")


def _run_humanitarian_demo() -> None:
    """Run interactive humanitarian demo"""

    click.echo("  Crisis Type: Natural Disaster")
    click.echo("  Affected Population: 10,000+")
    click.echo("  Response Recommended: Immediate")


def _load_data(filepath: str) -> np.ndarray:
    """Load data from file"""
    path = Path(filepath)

    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
            if isinstance(data, list):
                return np.array(data)
            return np.array([data])

    elif path.suffix == ".csv":
        data = np.loadtxt(path, delimiter=",", dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def _save_or_print_results(results: Dict[str, Any], output: Optional[str]) -> None:
    """Save results to file or print to console"""

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        click.echo(f"\n✓ Results saved to {output}")
    else:
        click.echo("\n" + json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
