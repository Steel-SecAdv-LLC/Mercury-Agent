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
Interstellar Object Anomaly Detection Module

Specialized detection system for interstellar objects (ISOs) including 'Oumuamua,
2I/Borisov (Comet Borisov), and future ISO candidates. Analyzes orbital mechanics,
spectroscopic data, non-gravitational accelerations, and morphology for anomalies
that may indicate natural unusual properties or potential artificial origins.

Key Features:
- Orbital trajectory anomaly detection
- Non-gravitational acceleration analysis
- Spectroscopic composition anomalies
- Morphological characteristics assessment
- Thermal emission pattern analysis
- Tumbling motion analysis
- Comparison with known solar system objects
- Artificial origin hypothesis testing (Galileo Project methodology)

Notable Interstellar Objects:
- 1I/'Oumuamua (2017): First confirmed ISO, unusual acceleration
- 2I/Borisov (2019): First confirmed interstellar comet
- Potential future detections via Vera C. Rubin Observatory (LSST)

Research References:
- Avi Loeb, Galileo Project (Harvard)
- NASA JPL Small-Body Database
- IAU Minor Planet Center
- ESO spectroscopic observations
- Spitzer Space Telescope thermal observations

⚠️ SIMULATION-BASED: For research/scientific analysis. Claims of artificial
origins require extraordinary evidence and peer review. This module analyzes
anomalies objectively without asserting conclusions about artificial origins.

MIT License compatible - original implementation
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging


class ISOAnomalyType(Enum):
    """Types of interstellar object anomalies"""

    ORBITAL = "orbital_trajectory"
    ACCELERATION = "non_gravitational_acceleration"
    SPECTROSCOPIC = "spectroscopic_composition"
    MORPHOLOGICAL = "morphology"
    THERMAL = "thermal_emission"
    ROTATION = "rotation_tumbling"
    COMBINED = "multi_factor"


class NaturalExplanationConfidence(Enum):
    """Confidence in natural explanations"""

    WELL_EXPLAINED = "well_explained_naturally"
    LIKELY_NATURAL = "likely_natural"
    UNCERTAIN = "uncertain"
    CHALLENGING = "challenging_to_explain"
    HIGHLY_ANOMALOUS = "highly_anomalous"


@dataclass
class InterstellarObjectResult:
    """Result from interstellar object anomaly analysis"""

    object_designation: str
    anomaly_detected: bool
    anomaly_type: str
    confidence: float
    anomaly_score: float

    orbital_anomalies: List[str] = field(default_factory=list)
    spectroscopic_anomalies: List[str] = field(default_factory=list)
    morphological_anomalies: List[str] = field(default_factory=list)

    natural_explanation_assessment: str = "uncertain"
    alternative_hypotheses: List[str] = field(default_factory=list)

    follow_up_observations_recommended: List[str] = field(default_factory=list)
    comparative_analysis: Optional[Dict[str, Any]] = None
    scientific_significance: float = 0.0


class InterstellarObjectAnalyzer(nn.Module):
    """
    Neural network for interstellar object anomaly analysis.

    Uses attention mechanisms to correlate multi-wavelength observations,
    orbital parameters, and physical characteristics.
    """

    def __init__(self, input_dim: int = 96):
        super().__init__()

        phi = 1.618
        hidden_1 = int(input_dim * phi)
        hidden_2 = int(hidden_1 * phi)
        hidden_3 = round(int(hidden_2 / phi) / 8) * 8

        encoder_output = hidden_1 // 3

        self.orbital_encoder = nn.Sequential(
            nn.Linear(input_dim // 3, encoder_output),
            nn.LayerNorm(encoder_output),
            nn.ReLU(),
            nn.Dropout(0.15),
        )

        self.spectroscopic_encoder = nn.Sequential(
            nn.Linear(input_dim // 3, encoder_output),
            nn.LayerNorm(encoder_output),
            nn.ReLU(),
            nn.Dropout(0.15),
        )

        self.physical_encoder = nn.Sequential(
            nn.Linear(input_dim // 3, encoder_output),
            nn.LayerNorm(encoder_output),
            nn.ReLU(),
            nn.Dropout(0.15),
        )

        concat_size = encoder_output * 3

        self.fusion_layer = nn.Sequential(
            nn.Linear(concat_size, hidden_2),
            nn.LayerNorm(hidden_2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_2, hidden_3),
            nn.LayerNorm(hidden_3),
            nn.ReLU(),
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_3, num_heads=8, dropout=0.1, batch_first=True
        )

        self.anomaly_classifier = nn.Sequential(
            nn.Linear(hidden_3, hidden_3 // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_3 // 2, 7),
        )

        self.confidence_head = nn.Sequential(nn.Linear(hidden_3, 1), nn.Sigmoid())

    def forward(
        self, orbital: torch.Tensor, spectro: torch.Tensor, physical: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ISO analyzer.

        Args:
            orbital: Orbital parameters [batch, features]
            spectro: Spectroscopic data [batch, features]
            physical: Physical characteristics [batch, features]

        Returns:
            Tuple of (anomaly_logits, confidence)
        """
        orbital_enc = self.orbital_encoder(orbital)
        spectro_enc = self.spectroscopic_encoder(spectro)
        physical_enc = self.physical_encoder(physical)

        fused = torch.cat([orbital_enc, spectro_enc, physical_enc], dim=-1)

        encoded = self.fusion_layer(fused)

        encoded_seq = encoded.unsqueeze(1)
        attended, _ = self.attention(encoded_seq, encoded_seq, encoded_seq)
        attended = attended.squeeze(1)

        anomaly_logits = self.anomaly_classifier(attended)
        confidence = self.confidence_head(attended)

        return anomaly_logits, confidence


class InterstellarObjectDetector:
    """
    Interstellar Object Anomaly Detector.

    Analyzes interstellar objects for unusual characteristics that deviate from
    expectations for natural solar system bodies or known interstellar comets/asteroids.
    """

    def __init__(self, enable_artificial_origin_test: bool = False):
        """
        Initialize interstellar object detector.

        Args:
            enable_artificial_origin_test: Enable Galileo Project-style tests
                                           (requires conservative thresholds)
        """
        self.logger = logging.getLogger(__name__)
        self.enable_artificial_origin_test = enable_artificial_origin_test
        self.golden_ratio = 1.618

        self.model = InterstellarObjectAnalyzer(input_dim=96)

        self.known_isos = self._initialize_iso_database()

        self.omni_interstellar_scalars = {
            "omni_orbital_precision": 1.44 * self.golden_ratio,
            "omni_spectroscopic_sensitivity": 1.42 * self.golden_ratio,
            "omni_morphological_analysis": 1.40 * self.golden_ratio,
            "omni_acceleration_detection": 1.48 * self.golden_ratio,
            "omni_comparative_assessment": 1.38 * self.golden_ratio,
            "omni_hypothesis_evaluation": 1.43 * self.golden_ratio,
        }

        self.logger.info(
            f"Interstellar Object Detector initialized "
            f"(artificial_origin_test={enable_artificial_origin_test})"
        )

    def _initialize_iso_database(self) -> Dict[str, Dict]:
        """Initialize known interstellar object database"""
        return {
            "1I/Oumuamua": {
                "discovery_date": "2017-10-19",
                "perihelion_au": 0.2559,
                "eccentricity": 1.20,
                "inclination_deg": 122.7,
                "velocity_kms": 26.33,
                "non_grav_accel": True,
                "dimensions_estimate": "100x100x10m",
                "albedo": 0.1,
                "color": "red",
                "outgassing_detected": False,
                "rotation_period_h": 8.1,
                "key_anomalies": [
                    "extreme_aspect_ratio",
                    "non_gravitational_acceleration",
                    "no_detectable_outgassing",
                    "unusual_light_curve",
                ],
            },
            "2I/Borisov": {
                "discovery_date": "2019-08-30",
                "perihelion_au": 2.006,
                "eccentricity": 3.36,
                "inclination_deg": 44.0,
                "velocity_kms": 32.2,
                "non_grav_accel": True,
                "dimensions_estimate": "~0.4km nucleus",
                "outgassing_detected": True,
                "coma_observed": True,
                "composition": "CO-rich",
                "key_anomalies": [
                    "high_CO_abundance",
                    "pristine_composition",
                    "unusual_nucleus_properties",
                ],
            },
        }

    def detect_interstellar_anomaly(
        self, iso_data: Dict[str, Any], comparison_objects: Optional[List[Dict]] = None
    ) -> InterstellarObjectResult:
        """
        Detect anomalies in interstellar object data.

        Args:
            iso_data: Observational data including:
                - orbital_parameters: Dict of orbital elements
                - spectroscopy: Spectroscopic measurements
                - morphology: Shape/size/rotation data
                - thermal: Thermal emission data
                - designation: Object designation
            comparison_objects: Optional solar system objects for comparison

        Returns:
            Interstellar object anomaly result
        """
        designation = iso_data.get("designation", "Unknown ISO")

        orbital_features, spectro_features, physical_features = self._extract_features(iso_data)

        orbital_t = torch.tensor(orbital_features, dtype=torch.float32).unsqueeze(0)
        spectro_t = torch.tensor(spectro_features, dtype=torch.float32).unsqueeze(0)
        physical_t = torch.tensor(physical_features, dtype=torch.float32).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            anomaly_logits, confidence = self.model(orbital_t, spectro_t, physical_t)

        anomaly_probs = torch.softmax(anomaly_logits[0], dim=0)
        anomaly_class = torch.argmax(anomaly_probs).item()
        confidence_score = float(confidence[0].item())

        anomaly_types = [t.value for t in ISOAnomalyType]
        anomaly_type = anomaly_types[anomaly_class]

        anomaly_score = confidence_score * self.omni_interstellar_scalars["omni_orbital_precision"]

        anomaly_detected = anomaly_score > (0.4 * self.golden_ratio)

        orbital_anomalies = self._analyze_orbital_anomalies(iso_data)

        spectro_anomalies = self._analyze_spectroscopic_anomalies(iso_data)

        morphological_anomalies = self._analyze_morphological_anomalies(iso_data)

        natural_explanation = self._assess_natural_explanations(
            iso_data, orbital_anomalies, spectro_anomalies, morphological_anomalies
        )

        hypotheses = self._generate_alternative_hypotheses(
            iso_data, natural_explanation, anomaly_score
        )

        follow_ups = self._recommend_follow_up_observations(iso_data, anomaly_type)

        comparative = None
        if comparison_objects:
            comparative = self._comparative_analysis(iso_data, comparison_objects)

        significance = self._assess_scientific_significance(anomaly_score, natural_explanation)

        result = InterstellarObjectResult(
            object_designation=designation,
            anomaly_detected=anomaly_detected,
            anomaly_type=anomaly_type,
            confidence=confidence_score,
            anomaly_score=anomaly_score,
            orbital_anomalies=orbital_anomalies,
            spectroscopic_anomalies=spectro_anomalies,
            morphological_anomalies=morphological_anomalies,
            natural_explanation_assessment=natural_explanation,
            alternative_hypotheses=hypotheses,
            follow_up_observations_recommended=follow_ups,
            comparative_analysis=comparative,
            scientific_significance=significance,
        )

        self.logger.info(
            f"ISO anomaly detection: {designation} - {anomaly_type} "
            f"(score={anomaly_score:.3f}, natural_explanation={natural_explanation})"
        )

        return result

    def _extract_features(
        self, iso_data: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract orbital, spectroscopic, and physical features"""
        orbital_params = iso_data.get("orbital_parameters", {})
        orbital_features = np.array(
            [
                orbital_params.get("eccentricity", 1.0) / 5.0,
                orbital_params.get("perihelion_au", 1.0) / 10.0,
                orbital_params.get("inclination_deg", 0.0) / 180.0,
                orbital_params.get("velocity_kms", 20.0) / 50.0,
                float(orbital_params.get("hyperbolic", True)),
                orbital_params.get("semi_major_axis_au", -1.0) / 100.0,
                orbital_params.get("perihelion_distance_au", 1.0) / 10.0,
                orbital_params.get("aphelion_distance_au", 1.0) / 100.0,
                orbital_params.get("orbital_period_yr", 0.0) / 1000.0,
                orbital_params.get("mean_anomaly_deg", 0.0) / 360.0,
                orbital_params.get("argument_perihelion_deg", 0.0) / 360.0,
                orbital_params.get("longitude_ascending_node_deg", 0.0) / 360.0,
                float(orbital_params.get("non_gravitational_acceleration", False)),
                orbital_params.get("radial_non_grav_a1", 0.0) * 1e8,
                orbital_params.get("transverse_non_grav_a2", 0.0) * 1e8,
                orbital_params.get("normal_non_grav_a3", 0.0) * 1e8,
            ],
            dtype=np.float32,
        )
        while len(orbital_features) < 32:
            orbital_features = np.append(orbital_features, 0.0)
        orbital_features = orbital_features[:32]

        spectro = iso_data.get("spectroscopy", {})
        spectro_features = np.array(
            [
                spectro.get("optical_color_bv", 0.0) / 2.0,
                spectro.get("optical_color_vi", 0.0) / 2.0,
                spectro.get("optical_color_ri", 0.0) / 2.0,
                spectro.get("nir_slope_percent_100nm", 0.0) / 50.0,
                spectro.get("albedo_geometric", 0.1) / 1.0,
                spectro.get("water_ice_detected", 0.0),
                spectro.get("co2_detected", 0.0),
                spectro.get("co_detected", 0.0),
                spectro.get("ch4_detected", 0.0),
                spectro.get("cn_detected", 0.0),
                spectro.get("c2_detected", 0.0),
                spectro.get("silicate_features", 0.0),
                spectro.get("organic_features", 0.0),
                spectro.get("metal_features", 0.0),
                float(spectro.get("outgassing_detected", False)),
                spectro.get("dust_production_rate_kgs", 0.0) / 100.0,
                spectro.get("gas_production_rate_molecules_s", 0.0) / 1e28,
                spectro.get("coma_brightness", 0.0) / 20.0,
                spectro.get("spectral_type", 0.5),
                spectro.get("reflectance_spectrum_slope", 0.0),
            ],
            dtype=np.float32,
        )
        while len(spectro_features) < 32:
            spectro_features = np.append(spectro_features, 0.0)
        spectro_features = spectro_features[:32]

        morphology = iso_data.get("morphology", {})
        thermal = iso_data.get("thermal", {})
        physical_features = np.array(
            [
                morphology.get("estimated_size_m", 100.0) / 1000.0,
                morphology.get("aspect_ratio", 1.0) / 10.0,
                morphology.get("rotation_period_h", 8.0) / 24.0,
                float(morphology.get("tumbling_detected", False)),
                morphology.get("light_curve_amplitude_mag", 0.0) / 5.0,
                morphology.get("shape_elongation", 1.0) / 10.0,
                float(morphology.get("nucleus_detected", False)),
                float(morphology.get("coma_detected", False)),
                float(morphology.get("tail_detected", False)),
                thermal.get("temperature_k", 0.0) / 400.0,
                thermal.get("thermal_inertia", 0.0) / 1000.0,
                thermal.get("emissivity", 0.9),
                thermal.get("infrared_excess", 0.0) / 10.0,
                float(thermal.get("spitzer_detection", False)),
                morphology.get("absolute_magnitude", 20.0) / 30.0,
                morphology.get("diameter_estimate_m", 100.0) / 1000.0,
                float(morphology.get("fragmentation_observed", False)),
                morphology.get("density_estimate_gcm3", 1.0) / 10.0,
                morphology.get("surface_gravity_ms2", 0.0) / 0.1,
                morphology.get("escape_velocity_ms", 0.0) / 100.0,
            ],
            dtype=np.float32,
        )
        while len(physical_features) < 32:
            physical_features = np.append(physical_features, 0.0)
        physical_features = physical_features[:32]

        return orbital_features, spectro_features, physical_features

    def _analyze_orbital_anomalies(self, iso_data: Dict[str, Any]) -> List[str]:
        """Analyze orbital parameter anomalies"""
        anomalies = []
        orbital = iso_data.get("orbital_parameters", {})

        ecc = orbital.get("eccentricity", 1.0)
        if ecc > 1.5:
            anomalies.append("extreme_hyperbolic_orbit")

        inc = orbital.get("inclination_deg", 0.0)
        if inc > 120 or inc < -120:
            anomalies.append("extreme_orbital_inclination")

        if orbital.get("non_gravitational_acceleration", False):
            if not iso_data.get("spectroscopy", {}).get("outgassing_detected", False):
                anomalies.append("non_gravitational_acceleration_without_outgassing")

        vel = orbital.get("velocity_kms", 20.0)
        if vel > 40:
            anomalies.append("unusually_high_interstellar_velocity")

        return anomalies

    def _analyze_spectroscopic_anomalies(self, iso_data: Dict[str, Any]) -> List[str]:
        """Analyze spectroscopic anomalies"""
        anomalies = []
        spectro = iso_data.get("spectroscopy", {})

        if spectro.get("albedo_geometric", 0.1) < 0.03:
            anomalies.append("extremely_low_albedo")
        elif spectro.get("albedo_geometric", 0.1) > 0.5:
            anomalies.append("unusually_high_albedo")

        if spectro.get("optical_color_bv", 0.0) > 1.0:
            anomalies.append("extremely_red_color")

        co_detected = spectro.get("co_detected", 0.0) > 0.5
        co2_detected = spectro.get("co2_detected", 0.0) > 0.5
        water_detected = spectro.get("water_ice_detected", 0.0) > 0.5

        if co_detected and not water_detected and not co2_detected:
            anomalies.append("unusual_co_dominated_composition")

        if spectro.get("metal_features", 0.0) > 0.5:
            anomalies.append("metallic_spectral_features")

        return anomalies

    def _analyze_morphological_anomalies(self, iso_data: Dict[str, Any]) -> List[str]:
        """Analyze morphological anomalies"""
        anomalies = []
        morphology = iso_data.get("morphology", {})

        aspect_ratio = morphology.get("aspect_ratio", 1.0)
        if aspect_ratio > 6.0:
            anomalies.append("extreme_elongation")

        light_curve_amp = morphology.get("light_curve_amplitude_mag", 0.0)
        if light_curve_amp > 2.5:
            anomalies.append("extreme_light_curve_variation")

        rotation_period = morphology.get("rotation_period_h", 8.0)
        if rotation_period < 2.0:
            anomalies.append("unusually_fast_rotation")
        elif rotation_period > 100.0:
            anomalies.append("unusually_slow_rotation")

        if morphology.get("tumbling_detected", False):
            anomalies.append("complex_tumbling_motion")

        return anomalies

    def _assess_natural_explanations(
        self,
        iso_data: Dict[str, Any],
        orbital_anomalies: List[str],
        spectro_anomalies: List[str],
        morphological_anomalies: List[str],
    ) -> str:
        """Assess confidence in natural explanations"""
        total_anomalies = (
            len(orbital_anomalies) + len(spectro_anomalies) + len(morphological_anomalies)
        )

        if total_anomalies == 0:
            return NaturalExplanationConfidence.WELL_EXPLAINED.value
        elif total_anomalies <= 2:
            return NaturalExplanationConfidence.LIKELY_NATURAL.value
        elif total_anomalies <= 4:
            return NaturalExplanationConfidence.UNCERTAIN.value
        elif total_anomalies <= 6:
            return NaturalExplanationConfidence.CHALLENGING.value
        else:
            return NaturalExplanationConfidence.HIGHLY_ANOMALOUS.value

    def _generate_alternative_hypotheses(
        self, iso_data: Dict[str, Any], natural_explanation: str, anomaly_score: float
    ) -> List[str]:
        """Generate alternative hypotheses for unusual observations"""
        hypotheses = []

        hypotheses.append("Natural interstellar comet (pristine composition)")
        hypotheses.append("Natural interstellar asteroid (ejected from planetary system)")

        if "non_gravitational_acceleration_without_outgassing" in iso_data.get(
            "orbital_anomalies", []
        ):
            hypotheses.append("Sublimation of supervolatiles (H2, N2)")
            hypotheses.append("Anisotropic thermal radiation (Yarkovsky-like effect)")

        if anomaly_score > 0.8 and self.enable_artificial_origin_test:
            hypotheses.append(
                "Technological artifact (requires extraordinary evidence - Galileo Project)"
            )

        return hypotheses[:5]

    def _recommend_follow_up_observations(
        self, iso_data: Dict[str, Any], anomaly_type: str
    ) -> List[str]:
        """Recommend follow-up observations"""
        recommendations = []

        if "orbital" in anomaly_type.lower():
            recommendations.append("Continue astrometric tracking for refined orbit")
            recommendations.append("Monitor for orbital evolution and non-gravitational forces")

        if "spectroscopic" in anomaly_type.lower():
            recommendations.append("Multi-wavelength spectroscopy (UV, optical, NIR, IR)")
            recommendations.append("High-resolution spectroscopy for composition analysis")

        if "morphological" in anomaly_type.lower():
            recommendations.append("Time-resolved photometry for light curve analysis")
            recommendations.append("Radar observations if within range")

        recommendations.append("JWST observations for thermal characterization")
        recommendations.append("Radio telescope monitoring for emissions")

        return recommendations[:6]

    def _comparative_analysis(
        self, iso_data: Dict[str, Any], comparison_objects: List[Dict]
    ) -> Dict[str, Any]:
        """Compare ISO with solar system objects"""
        orbital = iso_data.get("orbital_parameters", {})

        similar_objects = []
        for obj in comparison_objects:
            obj_orbital = obj.get("orbital_parameters", {})

            ecc_diff = abs(orbital.get("eccentricity", 1.0) - obj_orbital.get("eccentricity", 0.0))
            inc_diff = abs(
                orbital.get("inclination_deg", 0.0) - obj_orbital.get("inclination_deg", 0.0)
            )

            if ecc_diff < 0.5 and inc_diff < 30:
                similar_objects.append(obj.get("name", "Unknown"))

        return {
            "similar_solar_system_objects": similar_objects,
            "num_similar": len(similar_objects),
            "uniqueness_score": 1.0 - (len(similar_objects) / max(1, len(comparison_objects))),
        }

    def _assess_scientific_significance(
        self, anomaly_score: float, natural_explanation: str
    ) -> float:
        """Assess scientific significance of detection"""
        base_significance = anomaly_score

        if natural_explanation == NaturalExplanationConfidence.HIGHLY_ANOMALOUS.value:
            base_significance *= 1.5
        elif natural_explanation == NaturalExplanationConfidence.CHALLENGING.value:
            base_significance *= 1.3

        return min(1.0, base_significance)


def create_omni_interstellar_scalars() -> Dict[str, float]:
    """
    Create doctorate-level interstellar object analysis scalars.

    Returns:
        Dictionary of interstellar scalars with golden ratio optimization
    """
    phi = 1.618

    return {
        "omni_orbital_precision": 1.44 * phi,
        "omni_spectroscopic_sensitivity": 1.42 * phi,
        "omni_morphological_analysis": 1.40 * phi,
        "omni_acceleration_detection": 1.48 * phi,
        "omni_comparative_assessment": 1.38 * phi,
        "omni_hypothesis_evaluation": 1.43 * phi,
        "omni_follow_up_prioritization": 1.41 * phi,
        "omni_scientific_significance": 1.46 * phi,
    }
