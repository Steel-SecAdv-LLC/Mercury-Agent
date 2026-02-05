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

from typing import Any


"""
Streamlit Dashboard for Mercury Agent ♱

Interactive GUI for non-technical users featuring:
- Real-time anomaly visualization
- Medical subspecialty analysis
- Security threat monitoring
- Humanitarian crisis detection
- Schumann resonance analysis
- Chemistry/isotope analysis

Run with: streamlit run streamlit_dashboard.py
"""

import json

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from omni_mercury_engine.utils.rng import get_global_rng


st.set_page_config(
    page_title="Mercury Agent ♱ Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Main Streamlit application"""

    st.sidebar.title("🔬 Mercury Agent ♱")
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


def medical_analysis_page() -> None:
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


def cardiology_interface() -> None:
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
                        line={"color": "red", "width": 1},
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

            from omni_mercury_engine.medical.cardiology.cardiology_predictor import (
                CardiologyPredictor,
            )

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


def display_cardiology_results(result: Any) -> None:
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


def sepsis_interface() -> None:
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

            from omni_mercury_engine.medical.critical_care.sepsis_detector import SepsisDetector

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


def display_sepsis_results(result: Any) -> None:
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


def neurocritical_interface() -> None:
    """Neurocritical care interface for neurological emergency detection.

    Provides assessment tools for:
    - Stroke risk assessment (NIHSS scoring)
    - Intracranial pressure monitoring
    - Seizure detection patterns
    - Glasgow Coma Scale tracking
    - Brain perfusion analysis
    """
    st.header("🧠 Neurocritical Care")
    st.markdown("Advanced neurological emergency detection and monitoring")

    # Tab-based interface for different neurocritical assessments
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Stroke Assessment", "ICP Monitoring", "Seizure Detection", "Consciousness Assessment"]
    )

    with tab1:
        stroke_assessment_panel()

    with tab2:
        icp_monitoring_panel()

    with tab3:
        seizure_detection_panel()

    with tab4:
        consciousness_assessment_panel()


def stroke_assessment_panel() -> None:
    """Stroke risk assessment using NIHSS components."""
    st.subheader("Stroke Risk Assessment (NIHSS-based)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Level of Consciousness**")
        loc_responsiveness = st.selectbox(
            "LOC Responsiveness",
            ["Alert", "Drowsy", "Obtunded", "Coma"],
            index=0,
        )
        loc_questions = st.selectbox(
            "LOC Questions (age, month)",
            ["Both correct", "One correct", "Neither correct"],
            index=0,
        )
        loc_commands = st.selectbox(
            "LOC Commands (grip, close eyes)",
            ["Both correct", "One correct", "Neither correct"],
            index=0,
        )

        st.markdown("**Motor Function**")
        best_gaze = st.selectbox(
            "Best Gaze",
            ["Normal", "Partial gaze palsy", "Complete gaze palsy"],
            index=0,
        )
        facial_palsy = st.selectbox(
            "Facial Palsy",
            ["Normal", "Minor", "Partial", "Complete"],
            index=0,
        )

    with col2:
        st.markdown("**Limb Motor**")
        arm_motor_left = st.selectbox(
            "Left Arm Motor",
            ["No drift", "Drift", "Some effort", "No effort", "No movement"],
            index=0,
        )
        arm_motor_right = st.selectbox(
            "Right Arm Motor",
            ["No drift", "Drift", "Some effort", "No effort", "No movement"],
            index=0,
        )
        leg_motor_left = st.selectbox(
            "Left Leg Motor",
            ["No drift", "Drift", "Some effort", "No effort", "No movement"],
            index=0,
        )
        leg_motor_right = st.selectbox(
            "Right Leg Motor",
            ["No drift", "Drift", "Some effort", "No effort", "No movement"],
            index=0,
        )

        st.markdown("**Sensory & Language**")
        sensory = st.selectbox(
            "Sensory",
            ["Normal", "Mild loss", "Severe loss"],
            index=0,
        )
        language = st.selectbox(
            "Language/Aphasia",
            ["Normal", "Mild aphasia", "Severe aphasia", "Global aphasia"],
            index=0,
        )

    # Timing information
    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        onset_known = st.checkbox("Symptom onset time known")
        if onset_known:
            onset_hours = st.number_input("Hours since onset", 0.0, 48.0, 2.0, 0.5)
        else:
            onset_hours = None
    with col4:
        thrombolysis_candidate = st.checkbox("Potential thrombolysis candidate")
        prior_stroke = st.checkbox("Prior stroke history")

    if st.button("Calculate Stroke Risk", type="primary"):
        # Calculate NIHSS-like score
        score = 0
        score += ["Alert", "Drowsy", "Obtunded", "Coma"].index(loc_responsiveness)
        score += ["Both correct", "One correct", "Neither correct"].index(loc_questions)
        score += ["Both correct", "One correct", "Neither correct"].index(loc_commands)
        score += ["Normal", "Partial gaze palsy", "Complete gaze palsy"].index(best_gaze)
        score += ["Normal", "Minor", "Partial", "Complete"].index(facial_palsy)
        for motor in [arm_motor_left, arm_motor_right, leg_motor_left, leg_motor_right]:
            score += ["No drift", "Drift", "Some effort", "No effort", "No movement"].index(motor)
        score += ["Normal", "Mild loss", "Severe loss"].index(sensory)
        score += ["Normal", "Mild aphasia", "Severe aphasia", "Global aphasia"].index(language)

        # Display results
        st.markdown("### Assessment Results")
        col_r1, col_r2, col_r3 = st.columns(3)

        with col_r1:
            severity = "Minor" if score <= 4 else "Moderate" if score <= 15 else "Severe"
            st.metric("NIHSS Score", score, delta=severity)

        with col_r2:
            if onset_known and onset_hours is not None:
                if onset_hours <= 4.5 and thrombolysis_candidate:
                    st.success("Within thrombolysis window")
                elif onset_hours <= 24:
                    st.warning("Extended window - consider thrombectomy")
                else:
                    st.info("Outside acute window")

        with col_r3:
            risk_score = min(1.0, score / 25 + (0.2 if prior_stroke else 0))
            st.metric("Risk Level", f"{risk_score:.0%}")

        # Recommendations
        st.subheader("Clinical Recommendations")
        recommendations = []
        if score > 4:
            recommendations.append("Urgent neuroimaging (CT/MRI) recommended")
        if onset_known and onset_hours is not None and onset_hours <= 4.5:
            recommendations.append("Evaluate for IV thrombolysis eligibility")
        if score > 10:
            recommendations.append("Consider neurology/stroke team consult")
            recommendations.append("Monitor for hemorrhagic transformation")
        if score > 15:
            recommendations.append("ICU admission recommended")
            recommendations.append("Airway management may be required")

        for rec in recommendations:
            st.warning(f"• {rec}")


def icp_monitoring_panel() -> None:
    """Intracranial pressure monitoring interface."""
    st.subheader("Intracranial Pressure Monitoring")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Pressure Readings**")
        icp_current = st.number_input("Current ICP (mmHg)", 0, 60, 12)
        map_current = st.number_input("Mean Arterial Pressure (mmHg)", 40, 150, 85)
        cpp_calculated = map_current - icp_current
        st.metric("Cerebral Perfusion Pressure", f"{cpp_calculated} mmHg")

        st.markdown("**Waveform Analysis**")
        p2_p1_ratio = st.slider("P2/P1 Ratio", 0.5, 2.0, 0.8, 0.1)
        waveform_quality = "Normal" if p2_p1_ratio < 1.0 else "Elevated (reduced compliance)"

    with col2:
        st.markdown("**Patient Parameters**")
        head_elevation = st.slider("Head of Bed Elevation (degrees)", 0, 45, 30)
        st.selectbox(
            "Sedation Level (RASS)",
            [
                "-5 (Unarousable)",
                "-4 (Deep sedation)",
                "-3 (Moderate sedation)",
                "-2 (Light sedation)",
                "-1 (Drowsy)",
                "0 (Alert)",
                "+1 (Restless)",
            ],
            index=5,
        )
        pupil_reactivity = st.selectbox(
            "Pupil Reactivity",
            ["Both reactive", "Left fixed", "Right fixed", "Both fixed"],
            index=0,
        )

    # Upload ICP trend data
    icp_file = st.file_uploader("Upload ICP Trend Data (CSV)", type=["csv"])

    # Process uploaded ICP file if available
    icp_trend_data = None
    if icp_file is not None:
        try:
            import pandas as pd

            icp_trend_data = pd.read_csv(icp_file)
            st.success(f"Loaded {len(icp_trend_data)} ICP readings from file")

            # Display trend if data available
            if "icp" in icp_trend_data.columns.str.lower():
                icp_col = next(c for c in icp_trend_data.columns if "icp" in c.lower())
                st.line_chart(icp_trend_data[icp_col])

                # Calculate trend statistics
                icp_values = icp_trend_data[icp_col].dropna()
                if len(icp_values) > 0:
                    st.markdown("**Trend Statistics:**")
                    st.write(f"- Mean ICP: {icp_values.mean():.1f} mmHg")
                    st.write(f"- Max ICP: {icp_values.max():.1f} mmHg")
                    st.write(f"- % Time > 20 mmHg: {(icp_values > 20).mean():.1%}")
        except Exception as e:
            st.error(f"Error reading ICP file: {e}")

    if st.button("Analyze ICP Status", type="primary"):
        st.markdown("### ICP Analysis Results")

        # Determine status
        status_col1, status_col2, status_col3 = st.columns(3)

        with status_col1:
            if icp_current < 20:
                st.success(f"ICP: {icp_current} mmHg (Normal)")
            elif icp_current < 25:
                st.warning(f"ICP: {icp_current} mmHg (Elevated)")
            else:
                st.error(f"ICP: {icp_current} mmHg (Critical)")

        with status_col2:
            if cpp_calculated >= 60:
                st.success(f"CPP: {cpp_calculated} mmHg (Adequate)")
            elif cpp_calculated >= 50:
                st.warning(f"CPP: {cpp_calculated} mmHg (Marginal)")
            else:
                st.error(f"CPP: {cpp_calculated} mmHg (Inadequate)")

        with status_col3:
            st.info(f"Waveform: {waveform_quality}")

        # Recommendations
        st.subheader("Management Recommendations")
        recommendations = []
        if icp_current >= 20:
            recommendations.append("Consider osmotic therapy (mannitol/hypertonic saline)")
        if cpp_calculated < 60:
            recommendations.append("Optimize MAP to maintain CPP > 60 mmHg")
        if head_elevation < 30:
            recommendations.append("Elevate head of bed to 30 degrees")
        if p2_p1_ratio > 1.0:
            recommendations.append("Reduced intracranial compliance - monitor closely")
        if "fixed" in pupil_reactivity.lower():
            recommendations.append("URGENT: Evaluate for herniation syndrome")
            recommendations.append("Consider emergent decompressive surgery")

        for rec in recommendations:
            if "URGENT" in rec:
                st.error(f"• {rec}")
            else:
                st.warning(f"• {rec}")


def seizure_detection_panel() -> None:
    """Seizure detection and EEG pattern analysis."""
    st.subheader("Seizure Detection & EEG Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Clinical Observations**")
        seizure_type = st.selectbox(
            "Observed Seizure Type",
            [
                "None observed",
                "Focal aware",
                "Focal impaired awareness",
                "Generalized tonic-clonic",
                "Absence",
                "Myoclonic",
                "Status epilepticus",
            ],
            index=0,
        )
        duration_seconds = st.number_input("Duration (seconds)", 0, 600, 0)
        st.checkbox("Post-ictal state present")

        st.markdown("**Risk Factors**")
        prior_seizures = st.checkbox("History of seizures/epilepsy")
        brain_injury = st.checkbox("Recent brain injury/surgery")
        metabolic_derangement = st.checkbox("Metabolic abnormality present")

    with col2:
        st.markdown("**EEG Findings (if available)**")
        eeg_pattern = st.selectbox(
            "Predominant EEG Pattern",
            [
                "Normal background",
                "Generalized slowing",
                "Focal slowing",
                "Periodic discharges",
                "Spike-wave complexes",
                "Suppression-burst",
                "Electrographic seizure",
            ],
            index=0,
        )
        st.selectbox(
            "Lateralization",
            ["Bilateral", "Left hemisphere", "Right hemisphere", "Multifocal"],
            index=0,
        )

        st.markdown("**Antiepileptic Drugs**")
        current_aed = st.multiselect(
            "Current AEDs",
            [
                "Levetiracetam",
                "Phenytoin",
                "Valproate",
                "Lacosamide",
                "Carbamazepine",
                "Phenobarbital",
                "Midazolam",
            ],
        )

    # Upload EEG data
    eeg_file = st.file_uploader("Upload EEG Data (EDF/CSV)", type=["edf", "csv"])

    # Process uploaded EEG file if available
    if eeg_file is not None:
        try:
            file_type = eeg_file.name.split(".")[-1].lower()

            if file_type == "csv":
                import pandas as pd

                eeg_data = pd.read_csv(eeg_file)
                st.success(
                    f"Loaded EEG data: {eeg_data.shape[0]} samples, {eeg_data.shape[1]} channels"
                )

                # Display EEG waveform preview (first channel)
                if len(eeg_data.columns) > 0:
                    st.markdown("**EEG Signal Preview (first channel):**")
                    preview_data = eeg_data.iloc[: min(1000, len(eeg_data)), 0]
                    st.line_chart(preview_data)

                    # Basic spectral analysis
                    import numpy as np

                    signal = eeg_data.iloc[:, 0].values
                    if len(signal) > 256:
                        # Simple power spectral density estimate
                        fft_vals = np.abs(np.fft.rfft(signal[:1024]))
                        delta_power = np.mean(fft_vals[1:4])  # 0.5-4 Hz
                        theta_power = np.mean(fft_vals[4:8])  # 4-8 Hz
                        alpha_power = np.mean(fft_vals[8:13])  # 8-13 Hz
                        beta_power = np.mean(fft_vals[13:30])  # 13-30 Hz

                        st.markdown("**Spectral Power (relative):**")
                        col_a, col_b, col_c, col_d = st.columns(4)
                        with col_a:
                            st.metric("Delta", f"{delta_power:.1f}")
                        with col_b:
                            st.metric("Theta", f"{theta_power:.1f}")
                        with col_c:
                            st.metric("Alpha", f"{alpha_power:.1f}")
                        with col_d:
                            st.metric("Beta", f"{beta_power:.1f}")

            elif file_type == "edf":
                st.info("EDF files require pyedflib library. Using header information only.")
                # Read basic info from EDF header
                st.write(f"File: {eeg_file.name}")

        except Exception as e:
            st.error(f"Error reading EEG file: {e}")

    if st.button("Analyze Seizure Risk", type="primary"):
        st.markdown("### Seizure Analysis Results")

        # Calculate risk score
        risk_score = 0.0
        if seizure_type != "None observed":
            risk_score += 0.4
        if seizure_type == "Status epilepticus":
            risk_score += 0.4
        if duration_seconds > 60:
            risk_score += 0.2
        if prior_seizures:
            risk_score += 0.1
        if brain_injury:
            risk_score += 0.1
        if metabolic_derangement:
            risk_score += 0.1
        if eeg_pattern in ["Electrographic seizure", "Periodic discharges"]:
            risk_score += 0.3
        risk_score = min(1.0, risk_score)

        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            if risk_score < 0.3:
                st.success(f"Seizure Risk: {risk_score:.0%} (Low)")
            elif risk_score < 0.6:
                st.warning(f"Seizure Risk: {risk_score:.0%} (Moderate)")
            else:
                st.error(f"Seizure Risk: {risk_score:.0%} (High)")

        with col_r2:
            st.metric("Duration", f"{duration_seconds}s" if duration_seconds > 0 else "N/A")

        with col_r3:
            st.info(f"EEG: {eeg_pattern}")

        # Recommendations
        st.subheader("Clinical Recommendations")
        recommendations = []
        if seizure_type == "Status epilepticus":
            recommendations.append("URGENT: Initiate status epilepticus protocol")
            recommendations.append("First-line: Benzodiazepines (lorazepam/midazolam)")
        if seizure_type != "None observed" and not current_aed:
            recommendations.append("Consider antiepileptic drug initiation")
        if duration_seconds > 300:
            recommendations.append("Prolonged seizure - consider refractory status protocol")
        if eeg_pattern in ["Periodic discharges", "Electrographic seizure"]:
            recommendations.append("Continue EEG monitoring")
        if metabolic_derangement:
            recommendations.append("Correct underlying metabolic abnormality")

        for rec in recommendations:
            if "URGENT" in rec:
                st.error(f"• {rec}")
            else:
                st.warning(f"• {rec}")


def consciousness_assessment_panel() -> None:
    """Glasgow Coma Scale and consciousness assessment."""
    st.subheader("Consciousness Assessment (GCS)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Eye Opening (E)**")
        eye_response = st.radio(
            "Eye Opening",
            [
                "4 - Spontaneous",
                "3 - To voice",
                "2 - To pain",
                "1 - None",
            ],
            index=0,
        )
        e_score = int(eye_response[0])

        st.markdown("**Verbal Response (V)**")
        verbal_response = st.radio(
            "Verbal Response",
            [
                "5 - Oriented",
                "4 - Confused",
                "3 - Inappropriate words",
                "2 - Incomprehensible sounds",
                "1 - None",
            ],
            index=0,
        )
        v_score = int(verbal_response[0])

    with col2:
        st.markdown("**Motor Response (M)**")
        motor_response = st.radio(
            "Motor Response",
            [
                "6 - Obeys commands",
                "5 - Localizes pain",
                "4 - Withdraws from pain",
                "3 - Abnormal flexion",
                "2 - Extension",
                "1 - None",
            ],
            index=0,
        )
        m_score = int(motor_response[0])

        st.markdown("**Pupil Assessment**")
        pupil_left = st.selectbox("Left Pupil", ["Reactive", "Sluggish", "Fixed"], index=0)
        pupil_right = st.selectbox("Right Pupil", ["Reactive", "Sluggish", "Fixed"], index=0)

    gcs_total = e_score + v_score + m_score
    pupil_reactivity_score = (1 if pupil_left == "Reactive" else 0) + (
        1 if pupil_right == "Reactive" else 0
    )
    gcs_p = gcs_total - (2 - pupil_reactivity_score)  # GCS-Pupils score

    st.markdown("---")
    st.markdown("### Assessment Results")

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)

    with col_r1:
        if gcs_total <= 8:
            st.error(f"GCS: {gcs_total}/15 (Severe)")
        elif gcs_total <= 12:
            st.warning(f"GCS: {gcs_total}/15 (Moderate)")
        else:
            st.success(f"GCS: {gcs_total}/15 (Mild)")

    with col_r2:
        st.metric("GCS Components", f"E{e_score}V{v_score}M{m_score}")

    with col_r3:
        st.metric("GCS-Pupils", f"{gcs_p}/15")

    with col_r4:
        pupil_status = "Normal" if pupil_reactivity_score == 2 else "Abnormal"
        st.metric("Pupils", pupil_status)

    # Recommendations
    st.subheader("Clinical Recommendations")
    recommendations = []
    if gcs_total <= 8:
        recommendations.append("Airway protection likely required - consider intubation")
        recommendations.append("Urgent neuroimaging recommended")
        recommendations.append("ICU admission indicated")
    if gcs_total <= 12:
        recommendations.append("Close neurological monitoring required")
        recommendations.append("Frequent GCS reassessment (q1-2h)")
    if pupil_reactivity_score < 2:
        recommendations.append("Pupil abnormality - evaluate for increased ICP or herniation")
    if m_score <= 3:
        recommendations.append("Abnormal motor response - concerning for brainstem dysfunction")

    for rec in recommendations:
        if "Urgent" in rec or "Airway" in rec:
            st.error(f"• {rec}")
        else:
            st.warning(f"• {rec}")


def general_medical_interface() -> None:
    """General medical interface"""

    st.header("⚕️ General Medical Analysis")

    vitals_file = st.file_uploader("Upload Vital Signs Time Series", type=["csv", "json"])

    if vitals_file and st.button("Analyze"):
        st.success("Analysis in progress...")


def security_analysis_page() -> None:
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


def cybint_interface() -> None:
    """CYBINT analysis interface"""

    st.header("🎯 CYBINT Sub-Processor")

    st.subheader("Threat Indicators")

    threat_file = st.file_uploader("Upload Threat Features (CSV/JSON)", type=["csv", "json"])

    malware_file = st.file_uploader("Upload Malware Features (CSV/JSON)", type=["csv", "json"])

    if st.button("🔍 Analyze Cyber Threat", type="primary"):
        with st.spinner("Processing cyber intelligence..."):

            from omni_mercury_engine.security.cybint_subprocessor import CYBINTSubProcessor

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


def display_cybint_results(result: Any) -> None:
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


def traffic_analysis_interface() -> None:
    """Traffic analysis interface"""

    st.header("📡 Traffic Analysis")
    st.info("Network traffic analysis interface...")


def tempest_interface() -> None:
    """TEMPEST detection interface"""

    st.header("📻 TEMPEST Detection")
    st.info("Electromagnetic emanation detection interface...")


def intel_fusion_interface() -> None:
    """Intelligence fusion interface"""

    st.header("🔄 Intelligence Fusion")
    st.info("Multi-INT fusion interface...")


def humanitarian_analysis_page() -> None:
    """Humanitarian crisis detection interface"""

    st.title("🌍 Humanitarian Crisis Dashboard")

    st.selectbox(
        "Crisis Type", ["Natural Disaster", "Pandemic", "Refugee Crisis", "Missing Persons"]
    )

    st.markdown("---")

    data_file = st.file_uploader("Upload Crisis Data", type=["csv", "json"])

    if data_file and st.button("Analyze Crisis", type="primary"):
        st.success("Crisis analysis in progress...")


def schumann_analysis_page() -> None:
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
                    line={"color": "blue", "width": 2},
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


def chemistry_analysis_page() -> None:
    """Chemistry/isotope analysis interface"""

    st.title("🧪 Chemistry & Isotope Analysis")

    st.selectbox(
        "Analysis Type", ["Isotope Ratio Analysis", "Rare Earth Elements", "General Chemistry"]
    )

    st.markdown("---")

    sample_file = st.file_uploader("Upload Sample Data", type=["csv", "json"])

    if sample_file and st.button("Analyze Sample", type="primary"):
        st.success("Chemical analysis in progress...")


def general_analysis_page() -> None:
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
                    from omni_mercury_engine import OmniMercuryEngine

                    engine = OmniMercuryEngine()
                    result = engine.detect(data)

                    col1, col2 = st.columns(2)

                    with col1:
                        if result.get("is_anomaly", False):
                            st.error("⚠️ ANOMALY DETECTED")
                        else:
                            st.success("✓ Normal Pattern")

                    with col2:
                        st.metric("Confidence", f"{result.get('anomaly_prob', 0):.1%}")


def load_file_data(uploaded_file: Any) -> np.ndarray[Any, Any] | None:
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
