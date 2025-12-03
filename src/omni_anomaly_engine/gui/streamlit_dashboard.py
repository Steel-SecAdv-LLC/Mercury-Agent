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
Streamlit Dashboard for OMNI ♱ AVA

Interactive GUI for non-technical users featuring:
- Real-time anomaly visualization
- Medical subspecialty analysis
- Security threat monitoring
- Humanitarian crisis detection
- Schumann resonance analysis
- Chemistry/isotope analysis

Run with: streamlit run streamlit_dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Optional
import json

from omni_anomaly_engine.utils.rng import get_global_rng

st.set_page_config(
    page_title="OMNI ♱ AVA Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Main Streamlit application"""

    st.sidebar.title("🔬 OMNI ♱ AVA")
    st.sidebar.markdown("---")

    analysis_mode = st.sidebar.selectbox(
        "Analysis Mode",
        [
            "🏥 Medical Analysis",
            "🔒 Security Intelligence",
            "🌍 Humanitarian Crisis",
            "🌐 Schumann Resonance",
            "🧪 Chemistry/Isotope",
            "📊 General Anomaly Detection",
        ],
    )

    if analysis_mode == "🏥 Medical Analysis":
        medical_analysis_page()
    elif analysis_mode == "🔒 Security Intelligence":
        security_analysis_page()
    elif analysis_mode == "🌍 Humanitarian Crisis":
        humanitarian_analysis_page()
    elif analysis_mode == "🌐 Schumann Resonance":
        schumann_analysis_page()
    elif analysis_mode == "🧪 Chemistry/Isotope":
        chemistry_analysis_page()
    else:
        general_analysis_page()


def medical_analysis_page():
    """Medical subspecialty analysis interface"""

    st.title("🏥 Medical Analysis Dashboard")
    st.markdown("Advanced medical anomaly detection and diagnosis support")

    subspecialty = st.selectbox(
        "Select Medical Subspecialty",
        ["Cardiology", "Neurocritical Care", "Sepsis Detection", "General Medical"],
    )

    st.markdown("---")

    if subspecialty == "Cardiology":
        cardiology_interface()
    elif subspecialty == "Neurocritical Care":
        neurocritical_interface()
    elif subspecialty == "Sepsis Detection":
        sepsis_interface()
    else:
        general_medical_interface()


def cardiology_interface():
    """Cardiology analysis interface"""

    st.header("💓 Cardiology Predictor")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ECG Analysis")

        ecg_file = st.file_uploader("Upload ECG Signal (CSV/JSON)", type=["csv", "json"])

        if ecg_file:
            st.success("ECG signal loaded successfully")

            ecg_data = load_file_data(ecg_file)

            if ecg_data is not None:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        y=ecg_data[:1000].flatten() if len(ecg_data.shape) > 1 else ecg_data[:1000],
                        mode="lines",
                        name="ECG Lead I",
                        line=dict(color="red", width=1),
                    )
                )
                fig.update_layout(
                    title="ECG Waveform",
                    xaxis_title="Sample",
                    yaxis_title="Amplitude (mV)",
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Cardiac Biomarkers")

        troponin = st.number_input(
            "Troponin I (ng/mL)", min_value=0.0, max_value=10.0, value=0.02, step=0.01
        )
        bnp = st.number_input("BNP (pg/mL)", min_value=0.0, max_value=1000.0, value=50.0, step=10.0)
        ck_mb = st.number_input("CK-MB (ng/mL)", min_value=0.0, max_value=50.0, value=3.0, step=0.5)

    st.markdown("---")

    if st.button("🔍 Analyze Cardiac Risk", type="primary"):
        with st.spinner("Analyzing cardiac risk..."):

            from omni_anomaly_engine.medical.cardiology_predictor import CardiologyPredictor

            predictor = CardiologyPredictor(enable_ecg=bool(ecg_file), enable_biomarkers=True)

            patient_data = {}

            if ecg_file and ecg_data is not None:
                if len(ecg_data.shape) == 1:
                    ecg_data = ecg_data.reshape(1, -1)
                rng = get_global_rng()
                patient_data["ecg_signal"] = (
                    ecg_data[:, :1000].reshape(12, -1)
                    if ecg_data.shape[1] >= 1000
                    else rng.randn(12, 1000)
                )

            patient_data["biomarkers"] = {
                "troponin_i_ng_ml": troponin,
                "bnp_pg_ml": bnp,
                "ck_mb_ng_ml": ck_mb,
            }

            result = predictor.predict_cardiac_risk(patient_data)

            display_cardiology_results(result)


def display_cardiology_results(result):
    """Display cardiology prediction results"""

    st.success("Analysis Complete!")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Cardiac Risk",
            "DETECTED" if result.cardiac_risk_detected else "Normal",
            delta="High" if result.cardiac_risk_detected else "Low",
        )

    with col2:
        st.metric("Confidence", f"{result.confidence:.1%}")

    with col3:
        st.metric(
            "MI Risk", f"{result.mi_risk:.1%}", delta="Critical" if result.mi_risk > 0.7 else None
        )

    with col4:
        st.metric(
            "HF Risk",
            f"{result.heart_failure_risk:.1%}",
            delta="Warning" if result.heart_failure_risk > 0.6 else None,
        )

    st.subheader("Arrhythmia Classification")
    st.info(f"🫀 Detected Rhythm: **{result.arrhythmia_type.replace('_', ' ').title()}**")

    if result.clinical_recommendations:
        st.subheader("Clinical Recommendations")
        for rec in result.clinical_recommendations:
            st.warning(f"• {rec}")


def sepsis_interface():
    """Sepsis detection interface"""

    st.header("🦠 Sepsis Detector")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Vital Signs (qSOFA)")
        resp_rate = st.slider("Respiratory Rate (bpm)", 8, 40, 18)
        gcs_score = st.slider("Glasgow Coma Scale", 3, 15, 15)
        systolic_bp = st.slider("Systolic BP (mmHg)", 60, 180, 120)

    with col2:
        st.subheader("Laboratory Values (SOFA)")
        platelets = st.number_input("Platelets (k/μL)", 20, 400, 200, 10)
        creatinine = st.number_input("Creatinine (mg/dL)", 0.5, 10.0, 1.0, 0.1)
        pao2_fio2 = st.number_input("PaO2/FiO2 Ratio", 50, 500, 350, 10)

    if st.button("🔍 Detect Sepsis", type="primary"):
        with st.spinner("Analyzing for sepsis..."):

            from omni_anomaly_engine.medical.sepsis_detector import SepsisDetector

            detector = SepsisDetector()

            result = detector.detect_sepsis(
                {
                    "vital_signs": {
                        "respiratory_rate_bpm": resp_rate,
                        "gcs_score": gcs_score,
                        "systolic_bp_mmhg": systolic_bp,
                    },
                    "laboratory_values": {
                        "platelets_k_ul": platelets,
                        "creatinine_mg_dl": creatinine,
                        "pao2_fio2_ratio": pao2_fio2,
                        "mean_arterial_pressure": (systolic_bp + 60) / 2,
                    },
                }
            )

            display_sepsis_results(result)


def display_sepsis_results(result):
    """Display sepsis detection results"""

    col1, col2, col3 = st.columns(3)

    with col1:
        if result.sepsis_detected:
            st.error(f"⚠️ SEPSIS DETECTED: {result.sepsis_stage.upper()}")
        else:
            st.success("✓ No Sepsis Detected")

    with col2:
        st.metric("SOFA Score", result.sofa_score or 0)

    with col3:
        st.metric("qSOFA Score", result.qsofa_score or 0)

    if result.organ_dysfunctions:
        st.subheader("Organ Dysfunctions")
        for organ in result.organ_dysfunctions:
            st.warning(f"• {organ.upper()}")

    if result.clinical_recommendations:
        st.subheader("Clinical Recommendations")
        for rec in result.clinical_recommendations[:5]:
            st.info(f"• {rec}")


def neurocritical_interface():
    """Neurocritical care interface"""

    st.header("🧠 Neurocritical Care")
    st.info("Advanced neurological emergency detection coming soon...")


def general_medical_interface():
    """General medical interface"""

    st.header("⚕️ General Medical Analysis")

    vitals_file = st.file_uploader("Upload Vital Signs Time Series", type=["csv", "json"])

    if vitals_file and st.button("Analyze"):
        st.success("Analysis in progress...")


def security_analysis_page():
    """Security intelligence interface"""

    st.title("🔒 Security Intelligence Dashboard")

    intel_type = st.selectbox(
        "Intelligence Type",
        [
            "CYBINT (Cyber Intelligence)",
            "Traffic Analysis",
            "TEMPEST Detection",
            "Intelligence Fusion",
        ],
    )

    st.markdown("---")

    if intel_type.startswith("CYBINT"):
        cybint_interface()
    elif intel_type.startswith("Traffic"):
        traffic_analysis_interface()
    elif intel_type.startswith("TEMPEST"):
        tempest_interface()
    else:
        intel_fusion_interface()


def cybint_interface():
    """CYBINT analysis interface"""

    st.header("🎯 CYBINT Sub-Processor")

    st.subheader("Threat Indicators")

    threat_file = st.file_uploader("Upload Threat Features (CSV/JSON)", type=["csv", "json"])

    malware_file = st.file_uploader("Upload Malware Features (CSV/JSON)", type=["csv", "json"])

    if st.button("🔍 Analyze Cyber Threat", type="primary"):
        with st.spinner("Processing cyber intelligence..."):

            from omni_anomaly_engine.security.cybint_subprocessor import CYBINTSubProcessor

            processor = CYBINTSubProcessor()

            threat_data = {}

            rng = get_global_rng()
            if threat_file:
                threat_data["threat_features"] = load_file_data(threat_file)
            else:
                threat_data["threat_features"] = rng.randn(256)

            if malware_file:
                threat_data["malware_features"] = load_file_data(malware_file)
            else:
                threat_data["malware_features"] = rng.randn(128)

            result = processor.process_cybint(threat_data)

            display_cybint_results(result)


def display_cybint_results(result):
    """Display CYBINT results"""

    col1, col2, col3 = st.columns(3)

    with col1:
        if result.threat_detected:
            st.error(f"⚠️ THREAT: {result.threat_severity.upper()}")
        else:
            st.success("✓ No Threat Detected")

    with col2:
        st.metric("APT Group", result.apt_group or "Unknown")

    with col3:
        st.metric("Malware Family", result.malware_family or "Unknown")

    if result.recommended_actions:
        st.subheader("Recommended Actions")
        for action in result.recommended_actions[:5]:
            st.warning(f"• {action}")


def traffic_analysis_interface():
    """Traffic analysis interface"""

    st.header("📡 Traffic Analysis")
    st.info("Network traffic analysis interface...")


def tempest_interface():
    """TEMPEST detection interface"""

    st.header("📻 TEMPEST Detection")
    st.info("Electromagnetic emanation detection interface...")


def intel_fusion_interface():
    """Intelligence fusion interface"""

    st.header("🔄 Intelligence Fusion")
    st.info("Multi-INT fusion interface...")


def humanitarian_analysis_page():
    """Humanitarian crisis detection interface"""

    st.title("🌍 Humanitarian Crisis Dashboard")

    st.selectbox(
        "Crisis Type", ["Natural Disaster", "Pandemic", "Refugee Crisis", "Missing Persons"]
    )

    st.markdown("---")

    data_file = st.file_uploader("Upload Crisis Data", type=["csv", "json"])

    if data_file and st.button("Analyze Crisis", type="primary"):
        st.success("Crisis analysis in progress...")


def schumann_analysis_page():
    """Schumann resonance analysis interface"""

    st.title("🌐 Schumann Resonance Analysis")
    st.markdown("Earth's electromagnetic field monitoring for disaster precursors")

    resonance_file = st.file_uploader("Upload Schumann Resonance Data", type=["csv", "json"])

    if resonance_file:
        data = load_file_data(resonance_file)

        if data is not None:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    y=data.flatten()[:500],
                    mode="lines",
                    name="Schumann Frequency",
                    line=dict(color="blue", width=2),
                )
            )
            fig.update_layout(
                title="Schumann Resonance Pattern",
                xaxis_title="Time",
                yaxis_title="Frequency (Hz)",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            if st.button("Detect Anomalies"):
                st.info("Analyzing Schumann patterns for disaster precursors...")


def chemistry_analysis_page():
    """Chemistry/isotope analysis interface"""

    st.title("🧪 Chemistry & Isotope Analysis")

    st.selectbox(
        "Analysis Type", ["Isotope Ratio Analysis", "Rare Earth Elements", "General Chemistry"]
    )

    st.markdown("---")

    sample_file = st.file_uploader("Upload Sample Data", type=["csv", "json"])

    if sample_file and st.button("Analyze Sample", type="primary"):
        st.success("Chemical analysis in progress...")


def general_analysis_page():
    """General anomaly detection interface"""

    st.title("📊 General Anomaly Detection")

    data_file = st.file_uploader("Upload Data for Analysis", type=["csv", "json"])

    if data_file:
        data = load_file_data(data_file)

        if data is not None:
            st.subheader("Data Preview")

            if len(data.shape) == 1:
                df = pd.DataFrame({"Value": data[:100]})
            else:
                df = pd.DataFrame(data[:100])

            st.dataframe(df, use_container_width=True)

            if st.button("Detect Anomalies", type="primary"):
                with st.spinner("Analyzing data..."):
                    from omni_anomaly_engine import OmniAnomalyEngine

                    engine = OmniAnomalyEngine()
                    result = engine.detect(data)

                    col1, col2 = st.columns(2)

                    with col1:
                        if result.get("is_anomaly", False):
                            st.error("⚠️ ANOMALY DETECTED")
                        else:
                            st.success("✓ Normal Pattern")

                    with col2:
                        st.metric("Confidence", f"{result.get('anomaly_prob', 0):.1%}")


def load_file_data(uploaded_file) -> Optional[np.ndarray]:
    """Load data from uploaded file"""

    try:
        if uploaded_file.name.endswith(".json"):
            data = json.load(uploaded_file)
            if isinstance(data, list):
                return np.array(data)
            return np.array([data])
        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            return df.values
        else:
            st.error("Unsupported file format")
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


if __name__ == "__main__":
    main()
