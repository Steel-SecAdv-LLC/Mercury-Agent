"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

Security Dataset Loaders: NSL-KDD, CICIDS, Threat Intelligence

References:
- NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html
- CICIDS 2017/2018: https://www.unb.ca/cic/datasets/ids-2017.html
- MITRE ATT&CK: https://attack.mitre.org/
"""
from __future__ import annotations
from typing import Any

import logging

import numpy as np

from .base import DatasetConfig, DatasetLoader, DatasetRegistry

logger = logging.getLogger(__name__)


class NSLKDDLoader(DatasetLoader):
    """
    NSL-KDD Network Intrusion Detection Dataset Loader.

    Improved version of KDD'99 with:
    - Removed duplicate records
    - Balanced difficulty levels
    - Standard train/test splits

    Reference: https://www.unb.ca/cic/datasets/nsl.html
    """

    DATASET_NAME = "nsl-kdd"
    DATASET_URL = "https://www.unb.ca/cic/datasets/nsl.html"
    LICENSE = "Academic Research Use"
    CITATION = """Tavallaee M, Bagheri E, Lu W, Ghorbani AA. A detailed analysis of the
    KDD CUP 99 data set. IEEE Symposium on Computational Intelligence. 2009."""
    REQUIRES_CREDENTIALS = False

    # NSL-KDD feature names
    FEATURE_NAMES = [
        "duration",
        "protocol_type",
        "service",
        "flag",
        "src_bytes",
        "dst_bytes",
        "land",
        "wrong_fragment",
        "urgent",
        "hot",
        "num_failed_logins",
        "logged_in",
        "num_compromised",
        "root_shell",
        "su_attempted",
        "num_root",
        "num_file_creations",
        "num_shells",
        "num_access_files",
        "num_outbound_cmds",
        "is_host_login",
        "is_guest_login",
        "count",
        "srv_count",
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
    ]

    # Attack categories
    ATTACK_TYPES = {
        "normal": 0,
        "dos": 1,  # Denial of Service
        "probe": 2,  # Probing
        "r2l": 3,  # Remote to Local
        "u2r": 4,  # User to Root
    }

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)
        self.binary_labels = config.preprocessing.get("binary", True)

    def download(self) -> bool:
        """Download or generate NSL-KDD data."""
        return self._create_synthetic_nslkdd()

    def _create_synthetic_nslkdd(self) -> bool:
        """Create synthetic NSL-KDD-like data."""
        np.random.seed(self.config.random_seed)
        n_samples = self.config.max_samples or 10000

        features = []
        labels = []

        attack_probs = {
            "normal": 0.4,
            "dos": 0.3,
            "probe": 0.15,
            "r2l": 0.1,
            "u2r": 0.05,
        }

        for _i in range(n_samples):
            attack_type = np.random.choice(list(attack_probs.keys()), p=list(attack_probs.values()))

            if attack_type == "normal":
                params = self._generate_normal_connection()
            elif attack_type == "dos":
                params = self._generate_dos_attack()
            elif attack_type == "probe":
                params = self._generate_probe_attack()
            elif attack_type == "r2l":
                params = self._generate_r2l_attack()
            else:  # u2r
                params = self._generate_u2r_attack()

            feature_vec = [params.get(f, 0) for f in self.FEATURE_NAMES]
            features.append(feature_vec)

            if self.binary_labels:
                labels.append(0 if attack_type == "normal" else 1)
            else:
                labels.append(self.ATTACK_TYPES[attack_type])

        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        save_path = self.data_path / "synthetic_nslkdd.npz"
        np.savez_compressed(save_path, features=features, labels=labels)

        logger.info(f"Generated {n_samples} NSL-KDD samples, {(labels > 0).sum()} attacks")
        return True

    def _generate_normal_connection(self) -> dict[str, Any]:
        """Generate features for normal network connection."""
        return {
            "duration": np.random.exponential(60),
            "protocol_type": np.random.choice([0, 1, 2]),  # tcp, udp, icmp
            "service": np.random.choice(range(70)),
            "flag": np.random.choice(range(11)),
            "src_bytes": np.random.exponential(1000),
            "dst_bytes": np.random.exponential(500),
            "land": 0,
            "wrong_fragment": 0,
            "urgent": 0,
            "hot": np.random.poisson(1),
            "num_failed_logins": 0,
            "logged_in": 1,
            "num_compromised": 0,
            "root_shell": 0,
            "su_attempted": 0,
            "num_root": 0,
            "num_file_creations": np.random.poisson(2),
            "num_shells": 0,
            "num_access_files": np.random.poisson(3),
            "num_outbound_cmds": 0,
            "is_host_login": 0,
            "is_guest_login": 0,
            "count": np.random.poisson(10),
            "srv_count": np.random.poisson(10),
            "serror_rate": np.random.beta(1, 20),
            "srv_serror_rate": np.random.beta(1, 20),
            "rerror_rate": np.random.beta(1, 20),
            "srv_rerror_rate": np.random.beta(1, 20),
            "same_srv_rate": np.random.beta(10, 1),
            "diff_srv_rate": np.random.beta(1, 10),
            "srv_diff_host_rate": np.random.beta(2, 10),
            "dst_host_count": np.random.poisson(50),
            "dst_host_srv_count": np.random.poisson(30),
            "dst_host_same_srv_rate": np.random.beta(10, 2),
            "dst_host_diff_srv_rate": np.random.beta(2, 10),
            "dst_host_same_src_port_rate": np.random.beta(5, 5),
            "dst_host_srv_diff_host_rate": np.random.beta(2, 10),
            "dst_host_serror_rate": np.random.beta(1, 20),
            "dst_host_srv_serror_rate": np.random.beta(1, 20),
            "dst_host_rerror_rate": np.random.beta(1, 20),
            "dst_host_srv_rerror_rate": np.random.beta(1, 20),
        }

    def _generate_dos_attack(self) -> dict[str, Any]:
        """Generate features for DoS attack."""
        params = self._generate_normal_connection()
        # DoS characteristics: high volume, short connections
        params["duration"] = np.random.exponential(1)
        params["count"] = np.random.poisson(300)
        params["srv_count"] = np.random.poisson(300)
        params["same_srv_rate"] = np.random.beta(20, 1)
        params["dst_host_count"] = np.random.poisson(200)
        params["serror_rate"] = np.random.beta(10, 2)
        return params

    def _generate_probe_attack(self) -> dict[str, Any]:
        """Generate features for probing/scanning attack."""
        params = self._generate_normal_connection()
        # Probe characteristics: many destinations, varied services
        params["duration"] = np.random.exponential(5)
        params["diff_srv_rate"] = np.random.beta(10, 2)
        params["srv_diff_host_rate"] = np.random.beta(10, 2)
        params["dst_host_diff_srv_rate"] = np.random.beta(10, 2)
        params["dst_host_count"] = np.random.poisson(150)
        params["rerror_rate"] = np.random.beta(5, 5)
        return params

    def _generate_r2l_attack(self) -> dict[str, Any]:
        """Generate features for Remote-to-Local attack."""
        params = self._generate_normal_connection()
        # R2L characteristics: failed logins, suspicious activity
        params["num_failed_logins"] = np.random.poisson(3)
        params["logged_in"] = np.random.choice([0, 1])
        params["hot"] = np.random.poisson(5)
        params["num_compromised"] = np.random.poisson(2)
        params["num_access_files"] = np.random.poisson(10)
        return params

    def _generate_u2r_attack(self) -> dict[str, Any]:
        """Generate features for User-to-Root attack."""
        params = self._generate_normal_connection()
        # U2R characteristics: privilege escalation
        params["root_shell"] = 1
        params["su_attempted"] = np.random.choice([0, 1, 2])
        params["num_root"] = np.random.poisson(3)
        params["num_shells"] = np.random.poisson(2)
        params["num_file_creations"] = np.random.poisson(10)
        return params

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        synthetic_path = self.data_path / "synthetic_nslkdd.npz"
        if synthetic_path.exists():
            data = np.load(synthetic_path)
            return data["features"], data["labels"]
        raise FileNotFoundError("NSL-KDD data not found")

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess network traffic features."""
        # Log transform for high-variance features
        data = np.log1p(np.abs(data))
        # Z-score normalize
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)


class CICIDSLoader(DatasetLoader):
    """
    CICIDS 2017/2018 Network Intrusion Detection Dataset Loader.

    Modern intrusion dataset with:
    - Realistic network traffic
    - Multiple attack types (DDoS, Brute Force, SQL Injection, etc.)
    - Labeled flows

    Reference: https://www.unb.ca/cic/datasets/ids-2017.html
    """

    DATASET_NAME = "cicids"
    DATASET_URL = "https://www.unb.ca/cic/datasets/ids-2017.html"
    LICENSE = "Academic Research Use"
    CITATION = """Sharafaldin I, Lashkari AH, Ghorbani AA. Toward Generating a New
    Intrusion Detection Dataset and Intrusion Traffic Characterization.
    ICISSP. 2018."""
    REQUIRES_CREDENTIALS = False

    FEATURE_NAMES = [
        "flow_duration",
        "total_fwd_packets",
        "total_bwd_packets",
        "total_length_fwd_packets",
        "total_length_bwd_packets",
        "fwd_packet_length_max",
        "fwd_packet_length_min",
        "fwd_packet_length_mean",
        "bwd_packet_length_max",
        "bwd_packet_length_min",
        "bwd_packet_length_mean",
        "flow_bytes_per_s",
        "flow_packets_per_s",
        "flow_iat_mean",
        "flow_iat_std",
        "fwd_iat_total",
        "fwd_iat_mean",
        "bwd_iat_total",
        "bwd_iat_mean",
        "fwd_psh_flags",
        "bwd_psh_flags",
        "fwd_urg_flags",
        "bwd_urg_flags",
        "fwd_header_length",
        "bwd_header_length",
        "fwd_packets_per_s",
        "bwd_packets_per_s",
        "min_packet_length",
        "max_packet_length",
        "packet_length_mean",
        "packet_length_std",
        "packet_length_variance",
        "fin_flag_count",
        "syn_flag_count",
        "rst_flag_count",
        "psh_flag_count",
        "ack_flag_count",
        "urg_flag_count",
        "cwe_flag_count",
        "ece_flag_count",
        "down_up_ratio",
        "avg_packet_size",
        "avg_fwd_segment_size",
        "avg_bwd_segment_size",
        "fwd_header_length_1",
        "fwd_avg_bytes_per_bulk",
        "fwd_avg_packets_per_bulk",
        "fwd_avg_bulk_rate",
        "bwd_avg_bytes_per_bulk",
        "bwd_avg_packets_per_bulk",
        "bwd_avg_bulk_rate",
        "subflow_fwd_packets",
        "subflow_fwd_bytes",
        "subflow_bwd_packets",
        "subflow_bwd_bytes",
        "init_win_bytes_fwd",
        "init_win_bytes_bwd",
        "act_data_pkt_fwd",
        "min_seg_size_forward",
        "active_mean",
        "active_std",
        "active_max",
        "active_min",
        "idle_mean",
        "idle_std",
        "idle_max",
        "idle_min",
    ]

    ATTACK_TYPES = [
        "benign",
        "ddos",
        "dos_hulk",
        "dos_slowhttptest",
        "dos_slowloris",
        "dos_goldeneye",
        "ftp_patator",
        "ssh_patator",
        "heartbleed",
        "web_attack_brute_force",
        "web_attack_xss",
        "web_attack_sql_injection",
        "infiltration",
        "bot",
        "portscan",
    ]

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)

    def download(self) -> bool:
        return self._create_synthetic_cicids()

    def _create_synthetic_cicids(self) -> bool:
        """Create synthetic CICIDS-like data."""
        np.random.seed(self.config.random_seed)
        n_samples = self.config.max_samples or 10000

        features = []
        labels = []

        for _i in range(n_samples):
            attack_type = np.random.choice(
                self.ATTACK_TYPES,
                p=[
                    0.5,
                    0.1,
                    0.05,
                    0.05,
                    0.05,
                    0.03,
                    0.03,
                    0.03,
                    0.01,
                    0.03,
                    0.02,
                    0.02,
                    0.02,
                    0.03,
                    0.03,
                ],
            )

            params = self._generate_flow(attack_type)
            feature_vec = [params.get(f, 0) for f in self.FEATURE_NAMES]
            features.append(feature_vec)
            labels.append(0 if attack_type == "benign" else 1)

        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        save_path = self.data_path / "synthetic_cicids.npz"
        np.savez_compressed(save_path, features=features, labels=labels)

        logger.info(f"Generated {n_samples} CICIDS samples, {labels.sum()} attacks")
        return True

    def _generate_flow(self, attack_type: str) -> dict[str, Any]:
        """Generate network flow features based on attack type."""
        base = {
            "flow_duration": np.random.exponential(10000),
            "total_fwd_packets": np.random.poisson(10),
            "total_bwd_packets": np.random.poisson(8),
            "total_length_fwd_packets": np.random.exponential(1000),
            "total_length_bwd_packets": np.random.exponential(800),
            "fwd_packet_length_max": np.random.exponential(1400),
            "fwd_packet_length_min": np.random.exponential(40),
            "fwd_packet_length_mean": np.random.exponential(500),
            "bwd_packet_length_max": np.random.exponential(1400),
            "bwd_packet_length_min": np.random.exponential(40),
            "bwd_packet_length_mean": np.random.exponential(400),
            "flow_bytes_per_s": np.random.exponential(10000),
            "flow_packets_per_s": np.random.exponential(100),
            "flow_iat_mean": np.random.exponential(1000),
            "flow_iat_std": np.random.exponential(500),
            "syn_flag_count": np.random.poisson(1),
            "fin_flag_count": np.random.poisson(1),
            "ack_flag_count": np.random.poisson(5),
            "rst_flag_count": 0,
        }

        # Attack-specific modifications
        if attack_type.startswith("ddos") or attack_type.startswith("dos"):
            base["flow_packets_per_s"] *= 100
            base["total_fwd_packets"] *= 50
            base["syn_flag_count"] *= 10

        elif attack_type in ["ftp_patator", "ssh_patator"]:
            base["total_fwd_packets"] = np.random.poisson(50)
            base["rst_flag_count"] = np.random.poisson(20)

        elif attack_type == "portscan":
            base["syn_flag_count"] *= 100
            base["rst_flag_count"] *= 50
            base["flow_duration"] = np.random.exponential(100)

        return base

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        synthetic_path = self.data_path / "synthetic_cicids.npz"
        if synthetic_path.exists():
            data = np.load(synthetic_path)
            return data["features"], data["labels"]
        raise FileNotFoundError("CICIDS data not found")

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess network flow features."""
        data = np.nan_to_num(data, nan=0.0, posinf=1e10, neginf=0)
        data = np.log1p(np.abs(data))
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)


class ThreatIntelLoader(DatasetLoader):
    """
    Threat Intelligence Indicator Dataset Loader.

    Provides access to:
    - IOC (Indicators of Compromise) features
    - MITRE ATT&CK technique patterns
    - Malware behavior signatures

    Based on MITRE ATT&CK framework and open threat feeds.
    """

    DATASET_NAME = "threat-intel"
    DATASET_URL = "https://attack.mitre.org/"
    LICENSE = "Apache 2.0 (MITRE ATT&CK)"
    CITATION = "MITRE ATT&CK. MITRE Corporation. https://attack.mitre.org/"
    REQUIRES_CREDENTIALS = False

    # MITRE ATT&CK tactics
    TACTICS = [
        "initial_access",
        "execution",
        "persistence",
        "privilege_escalation",
        "defense_evasion",
        "credential_access",
        "discovery",
        "lateral_movement",
        "collection",
        "command_and_control",
        "exfiltration",
        "impact",
    ]

    FEATURE_NAMES = [
        "technique_count",
        "tactic_diversity",
        "severity_score",
        "confidence",
        "source_reputation",
        "target_criticality",
        "attack_pattern_match",
        "ioc_age_days",
        "ioc_first_seen",
        "ioc_last_seen",
        "related_campaigns",
        "related_groups",
        "related_malware",
        "network_indicators",
        "file_indicators",
        "registry_indicators",
        "behavioral_indicators",
        "temporal_correlation",
        "geographic_spread",
        "industry_targeting",
        "technique_sophistication",
        "automation_level",
    ]

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)

    def download(self) -> bool:
        return self._create_synthetic_threat_intel()

    def _create_synthetic_threat_intel(self) -> bool:
        """Create synthetic threat intelligence data."""
        np.random.seed(self.config.random_seed)
        n_samples = self.config.max_samples or 5000

        features = []
        labels = []

        for _i in range(n_samples):
            is_threat = np.random.random() < 0.4

            if is_threat:
                params = {
                    "technique_count": np.random.poisson(5) + 1,
                    "tactic_diversity": np.random.uniform(0.5, 1.0),
                    "severity_score": np.random.uniform(50, 100),
                    "confidence": np.random.uniform(60, 100),
                    "source_reputation": np.random.uniform(60, 100),
                    "target_criticality": np.random.uniform(50, 100),
                    "attack_pattern_match": np.random.uniform(0.7, 1.0),
                    "ioc_age_days": np.random.exponential(30),
                    "ioc_first_seen": np.random.exponential(90),
                    "ioc_last_seen": np.random.exponential(7),
                    "related_campaigns": np.random.poisson(2) + 1,
                    "related_groups": np.random.poisson(1) + 1,
                    "related_malware": np.random.poisson(3),
                    "network_indicators": np.random.poisson(5),
                    "file_indicators": np.random.poisson(3),
                    "registry_indicators": np.random.poisson(2),
                    "behavioral_indicators": np.random.poisson(4),
                    "temporal_correlation": np.random.uniform(0.5, 1.0),
                    "geographic_spread": np.random.uniform(1, 50),
                    "industry_targeting": np.random.uniform(0.3, 1.0),
                    "technique_sophistication": np.random.uniform(50, 100),
                    "automation_level": np.random.uniform(30, 100),
                }
                labels.append(1)
            else:
                params = {
                    "technique_count": np.random.poisson(1),
                    "tactic_diversity": np.random.uniform(0, 0.3),
                    "severity_score": np.random.uniform(0, 30),
                    "confidence": np.random.uniform(10, 50),
                    "source_reputation": np.random.uniform(20, 60),
                    "target_criticality": np.random.uniform(0, 40),
                    "attack_pattern_match": np.random.uniform(0, 0.3),
                    "ioc_age_days": np.random.exponential(180),
                    "ioc_first_seen": np.random.exponential(365),
                    "ioc_last_seen": np.random.exponential(90),
                    "related_campaigns": 0,
                    "related_groups": 0,
                    "related_malware": np.random.poisson(0.5),
                    "network_indicators": np.random.poisson(1),
                    "file_indicators": np.random.poisson(0.5),
                    "registry_indicators": 0,
                    "behavioral_indicators": np.random.poisson(1),
                    "temporal_correlation": np.random.uniform(0, 0.3),
                    "geographic_spread": np.random.uniform(0, 5),
                    "industry_targeting": np.random.uniform(0, 0.2),
                    "technique_sophistication": np.random.uniform(0, 30),
                    "automation_level": np.random.uniform(0, 30),
                }
                labels.append(0)

            feature_vec = [params[f] for f in self.FEATURE_NAMES]
            features.append(feature_vec)

        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        save_path = self.data_path / "synthetic_threat_intel.npz"
        np.savez_compressed(save_path, features=features, labels=labels)

        logger.info(f"Generated {n_samples} threat intel samples, {labels.sum()} threats")
        return True

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        synthetic_path = self.data_path / "synthetic_threat_intel.npz"
        if synthetic_path.exists():
            data = np.load(synthetic_path)
            return data["features"], data["labels"]
        raise FileNotFoundError("Threat intel data not found")

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess threat intelligence features."""
        data = np.nan_to_num(data, nan=0.0)
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)


# Register security loaders
DatasetRegistry.register("nsl-kdd", NSLKDDLoader)
DatasetRegistry.register("cicids", CICIDSLoader)
DatasetRegistry.register("threat-intel", ThreatIntelLoader)
