# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Economic & Financial Crisis Detector - Market Anomaly Detection.

Comprehensive financial crisis detection for economic resilience:
- Market crash prediction (stock market volatility)
- Banking crisis detection (systemic risk)
- Currency crisis monitoring (forex anomalies)
- Fraud detection (algorithmic trading manipulation)
- Systemic risk assessment (interconnected failures)
- Credit default surge prediction
- Liquidity crisis detection
- Contagion modeling (cross-market correlations)

Integrations:
- FININT (Financial Intelligence) via intelligence_fusion.py
- Chaos theory (chaos_evolutionary.py) for tipping points
- Economic resilience framework
- Network analysis for systemic risk
- Time series anomaly detection

Research sources:
- IMF Financial Stability Reports
- BIS (Bank for International Settlements)
- Federal Reserve Economic Data (FRED)
- Academic research on financial crises
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
from torch import nn

from omni_mercury_engine.core.exceptions import InputModalityError
from omni_mercury_engine.detectors._hazard_input import (
    InputGrader,
    finite_scalar,
)
from omni_mercury_engine.detectors.provenance import DecisionSource
from omni_mercury_engine.security.safe_torch import safe_torch_load


def _supplied(
    data: dict[str, Any],
    key: str,
    *,
    unit: str,
    lo: float | None = None,
    hi: float | None = None,
    missing: list[str] | None = None,
) -> float | None:
    """The supplied value validated as a finite number in range, or ``None``.

    An absent key is recorded in ``missing`` and returns ``None`` — never a
    fabricated default reported as a measurement. A supplied-but-bad value
    (NaN, inf, bool, a string) is refused: a NaN VIX used to grade "stable".
    """
    if key not in data or data[key] is None:
        if missing is not None:
            missing.append(key)
        return None
    return finite_scalar(data[key], key, unit=unit, lo=lo, hi=hi)


#: The keys each section of ``predict_financial_crisis`` reads.  An absent
#: key is named as missing, never defaulted -- but a misspelt key used to go
#: unread, so a stressed market graded "stable".
FINANCIAL_SECTIONS: dict[str, frozenset[str]] = {
    "market_data": frozenset({"price_change_pct", "vix", "volume_vs_average"}),
    "banking_data": frozenset({"cds_spread_bps", "default_rate_pct", "liquidity_ratio"}),
    "network_data": frozenset(
        {"interconnectedness_score", "concentration_ratio", "cross_border_exposure"}
    ),
    "currency_data": frozenset({"volatility"}),
    "trading_data": frozenset({"trading_features", "volume_anomaly_score"}),
}

_UNBOUNDED = float("inf")

#: Unit and physical range of every scalar ``predict_financial_crisis`` grades
#: -- the same bounds the section detectors apply when they read a value, so
#: a value set aside here would have been refused there.  ``trading_data``
#: carries a feature vector and is graded separately.
FINANCIAL_INPUTS: dict[str, dict[str, tuple[str, float, float]]] = {
    "market_data": {
        "price_change_pct": ("%", -100.0, 100.0),
        "vix": ("index", 0.0, 200.0),
        "volume_vs_average": ("ratio", 0.0, _UNBOUNDED),
    },
    "banking_data": {
        "cds_spread_bps": ("bps", 0.0, _UNBOUNDED),
        "default_rate_pct": ("%", 0.0, 100.0),
        "liquidity_ratio": ("ratio", 0.0, _UNBOUNDED),
    },
    "network_data": {
        "interconnectedness_score": ("[0, 1]", 0.0, 1.0),
        "concentration_ratio": ("[0, 1]", 0.0, 1.0),
        "cross_border_exposure": ("[0, 1]", 0.0, 1.0),
    },
    "currency_data": {"volatility": ("fraction", 0.0, _UNBOUNDED)},
}


class CrisisType(Enum):
    """Financial crisis classifications."""

    MARKET_CRASH = "market_crash"
    BANKING_CRISIS = "banking_crisis"
    CURRENCY_CRISIS = "currency_crisis"
    SOVEREIGN_DEBT = "sovereign_debt_crisis"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    SYSTEMIC_CRISIS = "systemic_crisis"


class CrisisSeverity(Enum):
    """Crisis severity levels."""

    STABLE = "stable"
    STRESS = "stress"
    CRISIS = "crisis"
    SEVERE_CRISIS = "severe_crisis"
    SYSTEMIC_COLLAPSE = "systemic_collapse"


@dataclass
class FinancialCrisisPredictionResult:
    """Financial crisis prediction results."""

    crisis_imminent: bool
    confidence: float
    crisis_type: str
    severity_level: str

    market_volatility_index: float | None = None
    systemic_risk_score: float = 0.0
    contagion_probability: float = 0.0

    market_crash_detected: bool = False
    panic_selling: bool = False
    banking_stress: bool = False
    currency_instability: bool = False
    liquidity_shortage: bool = False

    fraud_indicators: list[str] = field(default_factory=list)
    interconnected_failures: list[str] = field(default_factory=list)

    vix_level: float | None = None
    credit_default_swap_spread: float | None = None

    policy_recommendations: list[str] = field(default_factory=list)
    intervention_actions: list[str] = field(default_factory=list)
    affected_sectors: list[str] = field(default_factory=list)

    # Partial-input contract: True when part of the record was set aside;
    # ``set_aside`` names each piece and why.
    partial: bool = False
    set_aside: list[dict[str, Any]] = field(default_factory=list)


#: SEC-approved market-wide circuit breaker Level 1: a 7% decline of the
#: S&P 500 from the prior close halts all US equity trading (SEC Release
#: 34-67090, June 2012; NYSE Rule 7.12).  Level 2 is 13%, Level 3 is 20%.
#: On the 54-year NASDAQ Composite daily record (FRED NASDAQCOM,
#: 1971-02-05..2026-10-21, n=14,045 trading days) a close-to-close decline
#: of 7% or more occurred on 16 days (rate 0.11%), all inside recognised
#: crises: 1987-10 (Black Monday), 1997-10 (Asian crisis), 1998-08 (LTCM),
#: 2000-04/2000-12/2001-01 (dot-com), 2008-09..12 (GFC), 2020-03 (COVID).
MARKET_CRASH_DECLINE_PCT: float = -7.0

#: VIX severity ladder calibrated on the full CBOE VIX daily history (FRED
#: VIXCLS, 1990-01-02..2026-10-21, n=9,300): moderate >= 28.5 (90th
#: percentile), high >= 32.9 (95th), extreme >= 46.7 (99th).  By
#: construction the base exceedance rates are 10%, 5% and 1% of trading
#: days; the extreme band covers only 1997-98, 2001-02, 2008-09, 2011,
#: 2015, 2018, 2020 stress episodes.
VIX_SEVERITY_LADDER: tuple[tuple[str, float], ...] = (
    ("extreme", 46.7),
    ("high", 32.9),
    ("moderate", 28.5),
)


class MarketCrashDetector:
    """Stock market crash detection via price decline and volatility."""

    decision_source = (
        DecisionSource(
            kind="published",
            description=(
                "A market crash is a one-day decline of 7% or more, the "
                "SEC-approved market-wide circuit breaker Level 1 halt line."
            ),
            citation=(
                "SEC Release No. 34-67090 (June 2012) approving the "
                "market-wide circuit breaker recalibration; NYSE Rule 7.12 "
                "(Level 1 = 7%, Level 2 = 13%, Level 3 = 20% S&P 500 decline)"
            ),
        ),
        DecisionSource(
            kind="calibrated",
            description=(
                "VIX severity ladder at the 90th/95th/99th percentiles "
                "(28.5/32.9/46.7) of the full VIX daily history."
            ),
            false_alarm_rate=0.10,
            calibration_data=(
                "FRED VIXCLS daily closes 1990-01-02..2026-10-21 (n=9,300); "
                "band base rates 10%/5%/1% by construction"
            ),
        ),
        DecisionSource(
            kind="experimental",
            description=(
                "panic_selling pairs a day of -3% or worse (1.9% of NASDAQ "
                "Composite trading days, FRED NASDAQCOM 1971-2026) with "
                "volume above twice average; no public long-horizon volume "
                "baseline was available to calibrate the volume leg."
            ),
        ),
    )

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)

    def detect_market_crash(self, market_data: dict[str, Any]) -> dict[str, Any]:
        """Detect market crash from price and volatility data.

        Args:
            market_data: Stock indices, volatility, volume

        Returns:
            Market crash detection results
        """
        missing: list[str] = []
        price_change_pct = _supplied(
            market_data, "price_change_pct", unit="%", lo=-100.0, hi=100.0, missing=missing
        )
        volatility_index = _supplied(
            market_data, "vix", unit="index", lo=0.0, hi=200.0, missing=missing
        )
        volume_surge = _supplied(
            market_data, "volume_vs_average", unit="ratio", lo=0.0, missing=missing
        )

        # Each judgment uses only supplied measurements; an absent one is
        # named in missing_inputs, never replaced by a quiet default (a
        # fabricated VIX of 20 used to be reported as the measured level).
        # The crash line is the SEC market-wide circuit breaker Level 1
        # (-7%); the retired rule (-5% AND VIX > 30, neither sourced) also
        # required a VIX print, so 1987-style records with no VIX could
        # never grade as a crash.
        crash_detected = (
            price_change_pct is not None and price_change_pct <= MARKET_CRASH_DECLINE_PCT
        )

        severity = "unknown" if volatility_index is None else "low"
        if volatility_index is not None:
            for band, floor in VIX_SEVERITY_LADDER:
                if volatility_index >= floor:
                    severity = band
                    break

        panic_selling = (
            volume_surge is not None
            and price_change_pct is not None
            and volume_surge > 2.0
            and price_change_pct < -3.0
        )

        return {
            "crash_detected": crash_detected,
            "vix": None if volatility_index is None else float(volatility_index),
            "price_change_pct": None if price_change_pct is None else float(price_change_pct),
            "severity": severity,
            "panic_selling": panic_selling,
            "missing_inputs": missing,
        }


#: Basel III Liquidity Coverage Ratio: banks must hold high-quality liquid
#: assets covering at least 100% of 30-day stressed net outflows (BCBS 238,
#: "Basel III: The Liquidity Coverage Ratio and liquidity risk monitoring
#: tools", January 2013).  The phase-in floor at introduction was 60%
#: (1 Jan 2015), rising 10 points a year to 100% in 2019; below that floor
#: is graded a liquidity crisis.
BASEL_LCR_MINIMUM: float = 1.0
BASEL_LCR_PHASE_IN_FLOOR: float = 0.6


class BankingStressDetector:
    """Banking sector stress detection via credit metrics."""

    decision_source = (
        DecisionSource(
            kind="calibrated",
            description=(
                "Bank 5y senior CDS above 300 bps marks severe distress: calm "
                "pre-2007 major-bank spreads ran 10-50 bps, while failure "
                "points printed far above the line (Bear Stearns ~740 bps in "
                "the weeks before its March 2008 collapse; Lehman 376 bps on "
                "2008-08-20; Morgan Stanley 796 bps on 2008-09-17).  No "
                "calm-period observation in the dated table exceeds it."
            ),
            false_alarm_rate=0.0,
            calibration_data=(
                "dated public CDS record, 2004-2011 (press/Markit figures as "
                "tabulated in tests/detectors/test_financial_crisis_detector.py)"
            ),
        ),
        DecisionSource(
            kind="calibrated",
            description=(
                "Loan delinquency at or above 3% marks banking stress: on the "
                "full quarterly history every exceedance falls inside the S&L "
                "crisis era (1985Q1-1994Q1) or the GFC and its workout "
                "(2008Q2-2013Q2); zero exceedances 1994Q2-2008Q1 and after "
                "2013Q2."
            ),
            false_alarm_rate=0.0,
            calibration_data=(
                "FRED DRALACBS (delinquency rate on all loans, all US "
                "commercial banks), 1985Q1..2026Q2, n=166 quarters"
            ),
        ),
        DecisionSource(
            kind="published",
            description=(
                "Liquidity ratio below 1.0 is stress (Basel III LCR minimum "
                "100%); below 0.6 (the Basel phase-in floor) is a liquidity "
                "crisis."
            ),
            citation=(
                "BCBS 238, 'Basel III: The Liquidity Coverage Ratio and "
                "liquidity risk monitoring tools', January 2013 (minimum "
                "100%; 60% phase-in floor at 1 Jan 2015)"
            ),
        ),
    )

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)

    def detect_banking_stress(self, banking_data: dict[str, Any]) -> dict[str, Any]:
        """Detect banking sector stress.

        Args:
            banking_data: Credit spreads, default rates, liquidity ratios

        Returns:
            Banking stress detection results
        """
        missing: list[str] = []
        cds_spread_bps = _supplied(
            banking_data, "cds_spread_bps", unit="bps", lo=0.0, missing=missing
        )
        default_rate_pct = _supplied(
            banking_data, "default_rate_pct", unit="%", lo=0.0, hi=100.0, missing=missing
        )
        liquidity_ratio = _supplied(
            banking_data, "liquidity_ratio", unit="ratio", lo=0.0, missing=missing
        )
        cds_threshold = 300.0
        default_threshold = 3.0
        liquidity_threshold = BASEL_LCR_MINIMUM

        # Stress is judged from supplied measurements only; absent metrics
        # are named, never filled with healthy-looking defaults (CDS 100 bps
        # and a 1.2 liquidity ratio used to be reported for empty input).
        stress_score = 0.0
        if cds_spread_bps is not None and cds_spread_bps > cds_threshold:
            stress_score += 0.4
        if default_rate_pct is not None and default_rate_pct > default_threshold:
            stress_score += 0.3
        if liquidity_ratio is not None and liquidity_ratio < liquidity_threshold:
            stress_score += 0.3
        banking_stress = stress_score > 0.0

        liquidity_crisis = (
            liquidity_ratio is not None and liquidity_ratio < BASEL_LCR_PHASE_IN_FLOOR
        )

        return {
            "banking_stress": banking_stress,
            "stress_score": float(stress_score),
            "liquidity_crisis": liquidity_crisis,
            "cds_spread_bps": None if cds_spread_bps is None else float(cds_spread_bps),
            "missing_inputs": missing,
        }


class FraudDetector(nn.Module):
    """Algorithmic trading fraud and market manipulation detection."""

    def __init__(self, input_dim: int = 64) -> None:
        """Initialize the instance."""
        super().__init__()

        phi = 1.618

        self.pattern_encoder = nn.Sequential(
            nn.Linear(input_dim, int(128 * phi)),
            nn.BatchNorm1d(int(128 * phi)),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(int(128 * phi), int(64 * phi)),
            nn.BatchNorm1d(int(64 * phi)),
            nn.ReLU(),
            nn.Linear(int(64 * phi), 64),
        )

        self.fraud_classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, trading_patterns: torch.Tensor) -> torch.Tensor:
        """Detect fraudulent trading patterns."""
        features = self.pattern_encoder(trading_patterns)
        fraud_prob = self.fraud_classifier(features)

        return fraud_prob


class SystemicRiskAnalyzer:
    """Systemic risk assessment via network contagion modeling."""

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)

    def assess_systemic_risk(self, network_data: dict[str, Any]) -> dict[str, Any]:
        """Assess systemic risk in financial network.

        Args:
            network_data: Interconnection matrix, exposures

        Returns:
            Systemic risk assessment
        """
        missing: list[str] = []
        components = {
            "interconnectedness_score": (
                0.4,
                _supplied(
                    network_data,
                    "interconnectedness_score",
                    unit="[0, 1]",
                    lo=0.0,
                    hi=1.0,
                    missing=missing,
                ),
            ),
            "concentration_ratio": (
                0.3,
                _supplied(
                    network_data,
                    "concentration_ratio",
                    unit="[0, 1]",
                    lo=0.0,
                    hi=1.0,
                    missing=missing,
                ),
            ),
            "cross_border_exposure": (
                0.3,
                _supplied(
                    network_data,
                    "cross_border_exposure",
                    unit="[0, 1]",
                    lo=0.0,
                    hi=1.0,
                    missing=missing,
                ),
            ),
        }
        supplied = {k: (w, v) for k, (w, v) in components.items() if v is not None}

        # The score is the weighted mean over SUPPLIED components, weights
        # renormalised; nothing is scored at all when no component was
        # measured (0.5/0.3/0.2 defaults used to stand in as measurements).
        if supplied:
            total_weight = sum(w for w, _ in supplied.values())
            systemic_risk_score = sum(w * v for w, v in supplied.values()) / total_weight
            contagion_probability = min(systemic_risk_score * 1.5, 1.0)
        else:
            systemic_risk_score = None
            contagion_probability = None

        concentration = components["concentration_ratio"][1]
        too_big_to_fail = concentration is not None and concentration > 0.5

        return {
            "systemic_risk_score": (
                None if systemic_risk_score is None else float(systemic_risk_score)
            ),
            "contagion_probability": (
                None if contagion_probability is None else float(contagion_probability)
            ),
            "too_big_to_fail": too_big_to_fail,
            "missing_inputs": missing,
        }


class FinancialCrisisDetector:
    """Comprehensive financial crisis detection system.

    Integrates market, banking, fraud, and systemic risk analysis.
    """

    decision_source = (
        *MarketCrashDetector.decision_source,
        *BankingStressDetector.decision_source,
        DecisionSource(
            kind="experimental",
            description=(
                "Aggregation and remaining screens are engineering rules with "
                "no published source: crisis_imminent at two or more fired "
                "indicators, currency instability at exchange-rate volatility "
                "above 0.15, systemic risk at a network score above 0.7, and "
                "fraud as a half indicator once trained weights load."
            ),
        ),
    )

    def __init__(
        self,
        enable_market_detection: bool = True,
        enable_banking_detection: bool = True,
        enable_fraud_detection: bool = True,
        enable_systemic_analysis: bool = True,
    ):
        """Initialize the instance."""
        self.enable_market = enable_market_detection
        self.enable_banking = enable_banking_detection
        self.enable_fraud = enable_fraud_detection
        self.enable_systemic = enable_systemic_analysis

        self.market_detector = MarketCrashDetector() if enable_market_detection else None
        self.banking_detector = BankingStressDetector() if enable_banking_detection else None
        self.fraud_detector = FraudDetector() if enable_fraud_detection else None
        self.systemic_analyzer = SystemicRiskAnalyzer() if enable_systemic_analysis else None

        # Anti-theater guard (mirrors EMP's IntentionalEMIDetector): the
        # FraudDetector ships with random weights and no labelled fraud
        # corpus. Until real weights load via load_fraud_weights(), its
        # output is noise, so _detect_fraud must NOT consult it — random
        # init could append "Algorithmic manipulation detected" to a result.
        self._fraud_trained = False
        self._warned_untrained_fraud = False

        self.logger = logging.getLogger(__name__)

    def load_fraud_weights(self, checkpoint_path: str) -> None:
        """Load trained FraudDetector weights; only then is the network consulted.

        Args:
            checkpoint_path: Torch checkpoint holding the module state dict
                (or a dict with a ``fraud_detector`` state dict).

        Raises:
            RuntimeError: If fraud detection is disabled on this detector.
        """
        if self.fraud_detector is None:
            raise RuntimeError("fraud detection is disabled on this detector")
        checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
        state = (
            checkpoint.get("fraud_detector", checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        self.fraud_detector.load_state_dict(state)
        self.fraud_detector.eval()
        self._fraud_trained = True

    def predict_financial_crisis(
        self, financial_data: dict[str, Any]
    ) -> FinancialCrisisPredictionResult:
        """Comprehensive financial crisis prediction.

        Args:
            financial_data: Multi-source financial data including:
                - market_data: Stock indices, volatility, volume
                - banking_data: Credit spreads, default rates, liquidity
                - trading_data: Algorithmic trading patterns (graded only
                  once fraud weights are loaded)
                - network_data: Financial interconnections
                - currency_data: Exchange-rate volatility

        Returns:
            Financial crisis prediction with policy recommendations

        Unusable pieces -- an unknown key or section, an empty or
        non-mapping section, a non-finite or out-of-range value (see
        :data:`FINANCIAL_INPUTS`) -- are set aside and reported on the
        result (``partial``, ``set_aside``); the usable rest is graded.

        Raises:
            InputModalityError: If the record is not a mapping.
            InvalidObservationError: If no gradable section remains.
        """
        financial_data, grader = self._grade_sections(financial_data)
        result = FinancialCrisisPredictionResult(
            crisis_imminent=False,
            confidence=0.0,
            crisis_type="market_crash",
            severity_level="stable",
            partial=grader.partial,
            set_aside=list(grader.set_aside),
        )

        crisis_indicators: float = 0

        if (
            self.enable_market
            and "market_data" in financial_data
            and self.market_detector is not None
        ):
            market_result = self.market_detector.detect_market_crash(financial_data["market_data"])
            result.market_crash_detected = market_result["crash_detected"]
            result.panic_selling = market_result["panic_selling"]
            result.vix_level = market_result["vix"]
            result.market_volatility_index = market_result["vix"]

            if market_result["crash_detected"]:
                crisis_indicators += 1
                result.confidence = max(result.confidence, 0.7)
                result.crisis_type = CrisisType.MARKET_CRASH.value
            # Panic selling short of the circuit-breaker line is a half
            # indicator (mirroring fraud): the volume surge used to be
            # computed and then discarded, so volume_vs_average changed
            # nothing about the verdict.
            if market_result["panic_selling"] and not market_result["crash_detected"]:
                crisis_indicators += 0.5
                result.confidence = max(result.confidence, 0.5)

        if (
            self.enable_banking
            and "banking_data" in financial_data
            and self.banking_detector is not None
        ):
            banking_result = self.banking_detector.detect_banking_stress(
                financial_data["banking_data"]
            )
            result.banking_stress = banking_result["banking_stress"]
            result.liquidity_shortage = banking_result["liquidity_crisis"]
            result.credit_default_swap_spread = banking_result["cds_spread_bps"]

            if banking_result["banking_stress"]:
                crisis_indicators += 1
                result.confidence = max(result.confidence, 0.8)
                # Banking stress ON TOP OF a detected market crash is
                # systemic; banking stress alone is a banking crisis. The
                # former check compared crisis_type against its default
                # string, so banking stress alone read "systemic_crisis".
                if result.market_crash_detected:
                    result.crisis_type = CrisisType.SYSTEMIC_CRISIS.value
                else:
                    result.crisis_type = CrisisType.BANKING_CRISIS.value

        if self.enable_fraud and "trading_data" in financial_data:
            fraud_result = self._detect_fraud(financial_data["trading_data"])
            if fraud_result["fraud_detected"]:
                result.fraud_indicators.append("Algorithmic manipulation detected")
                crisis_indicators += 0.5

        if (
            self.enable_systemic
            and "network_data" in financial_data
            and self.systemic_analyzer is not None
        ):
            systemic_result = self.systemic_analyzer.assess_systemic_risk(
                financial_data["network_data"]
            )
            if systemic_result["systemic_risk_score"] is not None:
                result.systemic_risk_score = systemic_result["systemic_risk_score"]
                result.contagion_probability = systemic_result["contagion_probability"]
                if systemic_result["systemic_risk_score"] > 0.7:
                    crisis_indicators += 1
                    result.crisis_type = CrisisType.SYSTEMIC_CRISIS.value

        if "currency_data" in financial_data:
            currency_volatility = _supplied(
                financial_data["currency_data"], "volatility", unit="fraction", lo=0.0
            )
            result.currency_instability = (
                currency_volatility is not None and currency_volatility > 0.15
            )
            if result.currency_instability:
                crisis_indicators += 0.5

        result.crisis_imminent = crisis_indicators >= 2

        result.severity_level = self._determine_severity(crisis_indicators, result)

        result.policy_recommendations = self._generate_policy_recommendations(result)
        result.intervention_actions = self._generate_interventions(result)
        result.affected_sectors = self._identify_affected_sectors(result, financial_data)

        return result

    def _grade_sections(self, financial_data: Any) -> tuple[dict[str, Any], InputGrader]:
        """Keep every usable measurement; set the rest aside with its reason.

        A misspelt key used to go unread, so a stressed market graded
        "stable"; it is now set aside and named, and the rest is graded.
        """
        if not isinstance(financial_data, dict):
            raise InputModalityError(
                "predict_financial_crisis expects a mapping of sections "
                f"(market_data, banking_data, ...), got {type(financial_data).__name__}"
            )
        grader = InputGrader()
        graded: dict[str, Any] = dict(
            grader.grade_sections(
                {k: v for k, v in financial_data.items() if k != "trading_data"},
                FINANCIAL_INPUTS,
            )
        )
        if "trading_data" in financial_data:
            trading = self._grade_trading(financial_data["trading_data"], grader)
            if trading:
                graded["trading_data"] = trading
        gradable = {"market_data", "banking_data", "network_data", "currency_data"}
        if self._fraud_trained:
            gradable.add("trading_data")
        # Trading data alone feeds only the fraud network, which stays
        # unconsulted until its weights load.
        grader.require_usable(
            gradable & set(graded),
            f"no gradable section: supply at least one of {sorted(gradable)}; "
            "an unmeasured economy is not a stable one",
        )
        return graded, grader

    @staticmethod
    def _grade_trading(trading: Any, grader: InputGrader) -> dict[str, Any]:
        """Usable ``trading_data``: a finite feature vector and/or a finite score."""
        if not isinstance(trading, dict):
            grader.discard(
                "trading_data", f"must be a mapping, got {type(trading).__name__}", trading
            )
            return {}
        accepted = FINANCIAL_SECTIONS["trading_data"]
        usable: dict[str, Any] = {}
        for key, value in trading.items():
            where = f"trading_data.{key}"
            if key not in accepted:
                grader.discard(where, f"unknown key; accepted: {sorted(accepted)}")
            elif key == "trading_features":
                try:
                    array = np.asarray(value, dtype=np.float64)
                except (TypeError, ValueError):
                    grader.discard(where, "must be a numeric vector", value)
                    continue
                if array.size and np.all(np.isfinite(array)):
                    usable[key] = value
                else:
                    grader.discard(where, "must be a non-empty vector of finite values")
            else:
                number = grader.usable_scalar(value, where, unit="score")
                if number is not None:
                    usable[key] = number
        if not usable:
            grader.discard(
                "trading_data",
                "no usable measurement in this section" if trading else "empty section",
            )
        return usable

    def _detect_fraud(self, trading_data: dict[str, Any]) -> dict[str, Any]:
        """Detect fraudulent trading patterns (trained network only)."""
        if self.fraud_detector is None:
            return {"fraud_detected": False, "fraud_probability": None, "consulted": False}

        if not self._fraud_trained:
            if not self._warned_untrained_fraud:
                self.logger.warning(
                    "FraudDetector is untrained (no checkpoint loaded); fraud is not "
                    "assessed. A randomly-initialised network's verdicts are noise, "
                    "not detections — call load_fraud_weights() once weights exist."
                )
                self._warned_untrained_fraud = True
            return {"fraud_detected": False, "fraud_probability": None, "consulted": False}

        if "trading_features" in trading_data:
            features = trading_data["trading_features"]
        else:
            volume_anomaly = trading_data.get("volume_anomaly_score", 0.0)
            features = np.array([volume_anomaly])
            features = np.pad(features, (0, 63), mode="constant")

        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        self.fraud_detector.eval()
        with torch.no_grad():
            fraud_prob = self.fraud_detector(features_tensor)

        fraud_detected = float(fraud_prob[0].item()) > 0.7

        return {
            "fraud_detected": fraud_detected,
            "fraud_probability": float(fraud_prob[0].item()),
            "consulted": True,
        }

    def _determine_severity(
        self, indicators: float, result: FinancialCrisisPredictionResult
    ) -> str:
        """Determine crisis severity level."""
        if indicators >= 3 and result.systemic_risk_score > 0.8:
            return CrisisSeverity.SYSTEMIC_COLLAPSE.value
        elif indicators >= 2 and result.contagion_probability > 0.7:
            return CrisisSeverity.SEVERE_CRISIS.value
        elif indicators >= 2:
            return CrisisSeverity.CRISIS.value
        elif indicators >= 1:
            return CrisisSeverity.STRESS.value
        else:
            return CrisisSeverity.STABLE.value

    def _generate_policy_recommendations(
        self, result: FinancialCrisisPredictionResult
    ) -> list[str]:
        """Generate policy recommendations."""
        recommendations = []

        if result.severity_level in ["systemic_collapse", "severe_crisis"]:
            recommendations.append("Emergency liquidity provision by central bank")
            recommendations.append("Coordinate international monetary policy response")
            recommendations.append("Consider capital controls if currency crisis")

        if result.banking_stress:
            recommendations.append("Strengthen bank capital requirements")
            recommendations.append("Implement stress testing for major institutions")

        if result.market_crash_detected:
            recommendations.append("Consider circuit breakers for market volatility")

        if result.fraud_indicators:
            recommendations.append("Enhance market surveillance and enforcement")

        return recommendations

    def _generate_interventions(self, result: FinancialCrisisPredictionResult) -> list[str]:
        """Generate intervention actions."""
        interventions = []

        if result.severity_level == "systemic_collapse":
            interventions.append("CRITICAL: Activate financial crisis task force")
            interventions.append("Emergency lending facilities for banks")
            interventions.append("Coordinate with IMF for international support")

        elif result.severity_level == "severe_crisis":
            interventions.append("Increase market monitoring frequency")
            interventions.append("Prepare bailout packages for systemically important institutions")

        if result.liquidity_shortage:
            interventions.append("Inject liquidity into banking system")

        return interventions

    def _identify_affected_sectors(
        self, result: FinancialCrisisPredictionResult, data: dict[str, Any]
    ) -> list[str]:
        """Identify affected economic sectors."""
        sectors = []

        if result.banking_stress:
            sectors.append("Financial Services")

        if result.market_crash_detected:
            sectors.extend(["Equity Markets", "Investment Management"])

        if "sector_data" in data:
            vulnerable_sectors = data["sector_data"].get("high_leverage_sectors", [])
            sectors.extend(vulnerable_sectors)

        return list(set(sectors))
