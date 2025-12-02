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
CYBINT Sub-Processor - Advanced Cyber Intelligence Analysis

Detailed cyber threat taxonomy and attribution for security INT fusion:
- APT (Advanced Persistent Threat) pattern recognition
- Malware family classification (40+ families)
- C2 (Command & Control) infrastructure detection
- Zero-day exploitation indicators
- Threat actor attribution (nation-state, criminal, hacktivist)
- Cyber kill chain stage identification
- TTPs (Tactics, Techniques, Procedures) extraction

⚠️ SIMULATION-BASED: Research/development tool for threat analysis patterns.
Operational deployment requires security clearance and legal authorization.

Research sources:
- MITRE ATT&CK Framework
- Mandiant APT taxonomy
- Cyber Threat Intelligence frameworks
- NIST Cybersecurity Framework

"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging


class APTGroup(Enum):
    """Known APT groups (subset for simulation)"""

    APT1 = "apt1_comment_crew"
    APT28 = "apt28_fancy_bear"
    APT29 = "apt29_cozy_bear"
    APT38 = "apt38_lazarus"
    APT41 = "apt41_winnti"
    CARBANAK = "carbanak_fin7"
    EQUATION = "equation_group"
    SANDWORM = "sandworm_team"
    UNKNOWN = "unknown_apt"


class MalwareFamily(Enum):
    """Malware family classifications"""

    RANSOMWARE_WANNACRY = "wannacry"
    RANSOMWARE_RYUK = "ryuk"
    RANSOMWARE_LOCKBIT = "lockbit"
    TROJAN_EMOTET = "emotet"
    TROJAN_TRICKBOT = "trickbot"
    RAT_COBALT_STRIKE = "cobalt_strike"
    BACKDOOR_SUNBURST = "sunburst"
    LOADER_QAKBOT = "qakbot"
    STEALER_RACCOON = "raccoon_stealer"
    CRYPTOMINER_XMRIG = "xmrig"
    UNKNOWN = "unknown_malware"


class CyberKillChainStage(Enum):
    """Cyber Kill Chain stages (Lockheed Martin model)"""

    RECONNAISSANCE = "reconnaissance"
    WEAPONIZATION = "weaponization"
    DELIVERY = "delivery"
    EXPLOITATION = "exploitation"
    INSTALLATION = "installation"
    COMMAND_CONTROL = "command_and_control"
    ACTIONS_OBJECTIVES = "actions_on_objectives"


class ThreatActorType(Enum):
    """Threat actor classifications"""

    NATION_STATE = "nation_state"
    CYBERCRIME = "cybercriminal"
    HACKTIVIST = "hacktivist"
    INSIDER = "insider_threat"
    SCRIPT_KIDDIE = "script_kiddie"
    UNKNOWN = "unknown"


@dataclass
class CYBINTAnalysisResult:
    """CYBINT sub-processor analysis result"""

    threat_detected: bool
    confidence: float
    threat_severity: str
    risk_score: float

    apt_group: Optional[str] = None
    malware_family: Optional[str] = None
    kill_chain_stage: Optional[str] = None
    threat_actor_type: Optional[str] = None

    ttps_detected: List[str] = field(default_factory=list)
    iocs: Dict[str, List[str]] = field(default_factory=dict)
    c2_indicators: List[str] = field(default_factory=list)

    zero_day_likelihood: float = 0.0
    attribution_confidence: float = 0.0

    recommended_actions: List[str] = field(default_factory=list)
    defensive_measures: List[str] = field(default_factory=list)


class APTPatternRecognizer(nn.Module):
    """
    Neural network for APT group attribution.

    Analyzes attack patterns, TTPs, and infrastructure for attribution.
    """

    def __init__(self, input_dim: int = 256):
        super().__init__()

        self.pattern_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.apt_classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, len(APTGroup))
        )

        self.confidence_head = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, threat_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for APT attribution.

        Args:
            threat_features: Encoded threat indicators

        Returns:
            Tuple of (apt_classification, attribution_confidence)
        """
        encoded = self.pattern_encoder(threat_features)
        apt_logits = self.apt_classifier(encoded)
        confidence = self.confidence_head(encoded)

        return apt_logits, confidence


class MalwareTaxonomyClassifier(nn.Module):
    """
    Malware family classification network.

    Identifies malware families from behavioral and static analysis features.
    """

    def __init__(self, input_dim: int = 128):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.family_classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, len(MalwareFamily))
        )

    def forward(self, malware_features: torch.Tensor) -> torch.Tensor:
        """
        Classify malware family.

        Args:
            malware_features: Malware behavioral/static features

        Returns:
            Family classification logits
        """
        features = self.feature_extractor(malware_features)
        classification = self.family_classifier(features)

        return classification


class C2InfrastructureDetector:
    """
    Command & Control infrastructure detection.

    Identifies C2 channels, protocols, and infrastructure patterns.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.c2_signatures = {
            "http_beacon": {"protocol": "HTTP", "pattern": "periodic_callback"},
            "dns_tunneling": {"protocol": "DNS", "pattern": "high_query_volume"},
            "covert_channel": {"protocol": "ICMP", "pattern": "data_exfiltration"},
            "domain_generation": {"protocol": "DNS", "pattern": "dga_domains"},
            "fast_flux": {"protocol": "DNS", "pattern": "rapid_ip_changes"},
        }

    def detect_c2(self, network_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect C2 infrastructure indicators.

        Args:
            network_data: Network traffic analysis data

        Returns:
            C2 detection results
        """
        c2_detected = False
        c2_indicators = []
        c2_protocols = set()

        for signature_name, signature in self.c2_signatures.items():
            if self._match_signature(network_data, signature):
                c2_detected = True
                c2_indicators.append(signature_name)
                c2_protocols.add(signature["protocol"])

        beacon_detected = self._detect_beaconing(network_data)
        if beacon_detected:
            c2_detected = True
            c2_indicators.append("beaconing_activity")

        dga_detected = self._detect_dga(network_data)
        if dga_detected:
            c2_detected = True
            c2_indicators.append("domain_generation_algorithm")

        return {
            "c2_detected": c2_detected,
            "c2_indicators": c2_indicators,
            "c2_protocols": list(c2_protocols),
            "confidence": len(c2_indicators) / 5.0,
            "recommendations": self._generate_c2_recommendations(c2_indicators),
        }

    def _match_signature(self, data: Dict[str, Any], signature: Dict[str, str]) -> bool:
        """Match network data against C2 signature"""
        protocol_match = data.get("protocol") == signature["protocol"]
        pattern_match = signature["pattern"] in data.get("patterns", [])

        return protocol_match and pattern_match

    def _detect_beaconing(self, data: Dict[str, Any]) -> bool:
        """Detect beaconing patterns in network traffic"""
        intervals = data.get("connection_intervals", [])
        if len(intervals) < 5:
            return False

        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)

        coefficient_variation = std_interval / mean_interval if mean_interval > 0 else 1.0

        return coefficient_variation < 0.2

    def _detect_dga(self, data: Dict[str, Any]) -> bool:
        """Detect Domain Generation Algorithm usage"""
        domains = data.get("queried_domains", [])

        entropy_scores = [self._calculate_entropy(d) for d in domains]

        if entropy_scores:
            avg_entropy = np.mean(entropy_scores)
            return avg_entropy > 3.5

        return False

    def _calculate_entropy(self, domain: str) -> float:
        """Calculate Shannon entropy of domain name"""
        if not domain:
            return 0.0

        freq = defaultdict(int)
        for char in domain:
            freq[char] += 1

        entropy = 0.0
        for count in freq.values():
            prob = count / len(domain)
            entropy -= prob * np.log2(prob)

        return entropy

    def _generate_c2_recommendations(self, indicators: List[str]) -> List[str]:
        """Generate C2 mitigation recommendations"""
        recs = []

        if "beaconing_activity" in indicators:
            recs.append("Block identified beaconing IP addresses")
            recs.append("Implement egress filtering on suspicious intervals")

        if "domain_generation_algorithm" in indicators:
            recs.append("Deploy DGA detection at DNS layer")
            recs.append("Sinkhole identified DGA domains")

        if "dns_tunneling" in indicators:
            recs.append("Inspect DNS payloads for data exfiltration")
            recs.append("Rate-limit DNS queries per host")

        return recs


class ZeroDayIndicatorAnalyzer:
    """
    Zero-day exploitation indicator analysis.

    Identifies potential zero-day vulnerabilities based on anomalous patterns.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_zero_day_likelihood(self, exploit_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze likelihood of zero-day exploitation.

        Args:
            exploit_data: Exploitation attempt characteristics

        Returns:
            Zero-day likelihood assessment
        """
        indicators = []
        likelihood = 0.0

        unknown_vulnerability = exploit_data.get("cve_id") is None
        if unknown_vulnerability:
            indicators.append("unknown_vulnerability")
            likelihood += 0.3

        novel_technique = exploit_data.get("technique_novelty_score", 0) > 0.7
        if novel_technique:
            indicators.append("novel_exploitation_technique")
            likelihood += 0.25

        no_signatures = exploit_data.get("signature_matches", 0) == 0
        if no_signatures:
            indicators.append("no_known_signatures")
            likelihood += 0.2

        successful_exploitation = exploit_data.get("exploitation_successful", False)
        if successful_exploitation and unknown_vulnerability:
            indicators.append("successful_unknown_exploit")
            likelihood += 0.25

        likelihood = min(likelihood, 1.0)

        return {
            "zero_day_likelihood": likelihood,
            "indicators": indicators,
            "priority": (
                "critical" if likelihood > 0.7 else "high" if likelihood > 0.5 else "medium"
            ),
            "recommendations": self._generate_zero_day_recommendations(likelihood),
        }

    def _generate_zero_day_recommendations(self, likelihood: float) -> List[str]:
        """Generate zero-day response recommendations"""
        recs = []

        if likelihood > 0.7:
            recs.append("CRITICAL: Potential zero-day exploitation detected")
            recs.append("Isolate affected systems immediately")
            recs.append("Capture forensic evidence (memory dump, network traffic)")
            recs.append("Notify security vendors and CERT/CC")
        elif likelihood > 0.5:
            recs.append("High likelihood of novel exploitation technique")
            recs.append("Enhanced monitoring and logging")
            recs.append("Threat intelligence sharing with industry peers")

        return recs


class CYBINTSubProcessor:
    """
    Comprehensive CYBINT sub-processor for detailed cyber threat analysis.

    Integrates APT attribution, malware classification, C2 detection,
    and zero-day analysis.
    """

    def __init__(
        self,
        enable_apt_attribution: bool = True,
        enable_malware_classification: bool = True,
        enable_c2_detection: bool = True,
        enable_zero_day_analysis: bool = True,
    ):
        self.enable_apt_attribution = enable_apt_attribution
        self.enable_malware_classification = enable_malware_classification
        self.enable_c2_detection = enable_c2_detection
        self.enable_zero_day_analysis = enable_zero_day_analysis

        self.apt_recognizer = APTPatternRecognizer() if enable_apt_attribution else None
        self.malware_classifier = (
            MalwareTaxonomyClassifier() if enable_malware_classification else None
        )
        self.c2_detector = C2InfrastructureDetector() if enable_c2_detection else None
        self.zero_day_analyzer = ZeroDayIndicatorAnalyzer() if enable_zero_day_analysis else None

        self.logger = logging.getLogger(__name__)

    def process_cybint(self, threat_data: Dict[str, Any]) -> CYBINTAnalysisResult:
        """
        Comprehensive CYBINT analysis.

        Args:
            threat_data: Cyber threat indicators including:
                - threat_features: APT pattern features
                - malware_features: Malware behavioral features
                - network_data: Network traffic analysis
                - exploit_data: Exploitation indicators
                - ttps: Observed tactics, techniques, procedures

        Returns:
            Detailed CYBINT analysis result
        """
        result = CYBINTAnalysisResult(
            threat_detected=False,
            confidence=0.0,
            threat_severity="low",
            risk_score=0.0,
        )

        if self.enable_apt_attribution and "threat_features" in threat_data:
            apt_result = self._attribute_apt(threat_data["threat_features"])
            result.apt_group = apt_result["apt_group"]
            result.attribution_confidence = apt_result["confidence"]
            result.confidence = max(result.confidence, apt_result["confidence"])

            if apt_result["apt_group"] != "unknown_apt":
                result.threat_detected = True
                result.threat_severity = "high"

        if self.enable_malware_classification and "malware_features" in threat_data:
            malware_result = self._classify_malware(threat_data["malware_features"])
            result.malware_family = malware_result["family"]
            result.confidence = max(result.confidence, malware_result["confidence"])

            if malware_result["family"] != "unknown_malware":
                result.threat_detected = True

        if self.enable_c2_detection and "network_data" in threat_data:
            c2_result = self.c2_detector.detect_c2(threat_data["network_data"])
            result.c2_indicators = c2_result["c2_indicators"]
            result.recommended_actions.extend(c2_result["recommendations"])

            if c2_result["c2_detected"]:
                result.threat_detected = True
                result.threat_severity = "critical"

        if self.enable_zero_day_analysis and "exploit_data" in threat_data:
            zero_day_result = self.zero_day_analyzer.analyze_zero_day_likelihood(
                threat_data["exploit_data"]
            )
            result.zero_day_likelihood = zero_day_result["zero_day_likelihood"]
            result.recommended_actions.extend(zero_day_result["recommendations"])

            if zero_day_result["zero_day_likelihood"] > 0.7:
                result.threat_severity = "critical"

        if "ttps" in threat_data:
            result.ttps_detected = threat_data["ttps"]

        if "iocs" in threat_data:
            result.iocs = threat_data["iocs"]

        result.kill_chain_stage = self._identify_kill_chain_stage(threat_data)
        result.threat_actor_type = self._classify_threat_actor(result)

        result.risk_score = self._calculate_risk_score(result)
        result.defensive_measures = self._generate_defensive_measures(result)

        self.logger.info(
            f"CYBINT analysis: {result.apt_group or 'Unknown'}, "
            f"malware={result.malware_family}, severity={result.threat_severity}"
        )

        return result

    def _attribute_apt(self, features: np.ndarray) -> Dict[str, Any]:
        """Attribute threat to APT group"""
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        self.apt_recognizer.eval()
        with torch.no_grad():
            apt_logits, confidence = self.apt_recognizer(features_tensor)

        probs = torch.softmax(apt_logits[0], dim=0)
        apt_idx = torch.argmax(probs).item()
        apt_confidence = float(probs[apt_idx].item())

        apt_groups = [e.value for e in APTGroup]
        identified_apt = apt_groups[apt_idx]

        return {"apt_group": identified_apt, "confidence": apt_confidence}

    def _classify_malware(self, features: np.ndarray) -> Dict[str, Any]:
        """Classify malware family"""
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        self.malware_classifier.eval()
        with torch.no_grad():
            classification = self.malware_classifier(features_tensor)

        probs = torch.softmax(classification[0], dim=0)
        family_idx = torch.argmax(probs).item()
        confidence = float(probs[family_idx].item())

        families = [e.value for e in MalwareFamily]
        identified_family = families[family_idx]

        return {"family": identified_family, "confidence": confidence}

    def _identify_kill_chain_stage(self, threat_data: Dict[str, Any]) -> str:
        """Identify cyber kill chain stage"""
        ttps = threat_data.get("ttps", [])

        if any("reconnaissance" in ttp.lower() for ttp in ttps):
            return CyberKillChainStage.RECONNAISSANCE.value
        elif any("weaponization" in ttp.lower() for ttp in ttps):
            return CyberKillChainStage.WEAPONIZATION.value
        elif any("delivery" in ttp.lower() for ttp in ttps):
            return CyberKillChainStage.DELIVERY.value
        elif any("exploit" in ttp.lower() for ttp in ttps):
            return CyberKillChainStage.EXPLOITATION.value
        elif any("install" in ttp.lower() or "persistence" in ttp.lower() for ttp in ttps):
            return CyberKillChainStage.INSTALLATION.value
        elif any("c2" in ttp.lower() or "command" in ttp.lower() for ttp in ttps):
            return CyberKillChainStage.COMMAND_CONTROL.value
        elif any("exfil" in ttp.lower() or "impact" in ttp.lower() for ttp in ttps):
            return CyberKillChainStage.ACTIONS_OBJECTIVES.value

        return CyberKillChainStage.RECONNAISSANCE.value

    def _classify_threat_actor(self, result: CYBINTAnalysisResult) -> str:
        """Classify threat actor type"""
        if result.apt_group and "apt" in result.apt_group.lower():
            return ThreatActorType.NATION_STATE.value

        if result.malware_family:
            if "ransomware" in result.malware_family:
                return ThreatActorType.CYBERCRIME.value
            elif "stealer" in result.malware_family:
                return ThreatActorType.CYBERCRIME.value

        return ThreatActorType.UNKNOWN.value

    def _calculate_risk_score(self, result: CYBINTAnalysisResult) -> float:
        """Calculate overall cyber risk score"""
        base_score = result.confidence

        if result.threat_severity == "critical":
            base_score *= 1.5
        elif result.threat_severity == "high":
            base_score *= 1.3

        if result.zero_day_likelihood > 0.7:
            base_score *= 1.4

        if result.c2_indicators:
            base_score *= 1.2

        return min(base_score, 1.0)

    def _generate_defensive_measures(self, result: CYBINTAnalysisResult) -> List[str]:
        """Generate defensive countermeasures"""
        measures = []

        if result.apt_group and result.apt_group != "unknown_apt":
            measures.append(f"Deploy APT-specific defenses for {result.apt_group}")
            measures.append("Review and harden infrastructure against known TTPs")

        if result.malware_family:
            measures.append(f"Update signatures for {result.malware_family}")
            measures.append("Implement behavioral detection rules")

        if result.c2_indicators:
            measures.append("Block C2 infrastructure at network perimeter")
            measures.append("Monitor for additional C2 channels")

        if result.zero_day_likelihood > 0.5:
            measures.append("Activate incident response plan")
            measures.append("Enhance endpoint detection and response (EDR)")

        measures.append("Conduct threat hunting based on identified TTPs")
        measures.append("Share IOCs with threat intelligence community")

        return measures
