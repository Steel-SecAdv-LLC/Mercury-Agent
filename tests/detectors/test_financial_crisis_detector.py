# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for Financial Crisis Detector."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("torch")

import pytest
import torch

from omni_mercury_engine.core.exceptions import InputModalityError, InvalidObservationError
from omni_mercury_engine.detectors.economic.financial_crisis_detector import (
    BankingStressDetector,
    CrisisSeverity,
    CrisisType,
    FinancialCrisisDetector,
    FinancialCrisisPredictionResult,
    FraudDetector,
    MarketCrashDetector,
    SystemicRiskAnalyzer,
)


class TestCrisisEnums:
    """Tests for crisis enumerations."""

    def test_crisis_type_values(self) -> None:
        """Test CrisisType enum values."""
        assert CrisisType.MARKET_CRASH.value == "market_crash"
        assert CrisisType.BANKING_CRISIS.value == "banking_crisis"
        assert CrisisType.CURRENCY_CRISIS.value == "currency_crisis"
        assert CrisisType.SOVEREIGN_DEBT.value == "sovereign_debt_crisis"
        assert CrisisType.LIQUIDITY_CRISIS.value == "liquidity_crisis"
        assert CrisisType.SYSTEMIC_CRISIS.value == "systemic_crisis"

    def test_crisis_severity_values(self) -> None:
        """Test CrisisSeverity enum values."""
        assert CrisisSeverity.STABLE.value == "stable"
        assert CrisisSeverity.STRESS.value == "stress"
        assert CrisisSeverity.CRISIS.value == "crisis"
        assert CrisisSeverity.SEVERE_CRISIS.value == "severe_crisis"
        assert CrisisSeverity.SYSTEMIC_COLLAPSE.value == "systemic_collapse"


class TestFinancialCrisisPredictionResult:
    """Tests for prediction result dataclass."""

    def test_default_values(self) -> None:
        """Test default values of result dataclass."""
        result = FinancialCrisisPredictionResult(
            crisis_imminent=False,
            confidence=0.0,
            crisis_type="stable",
            severity_level="stable",
        )
        assert result.market_crash_detected is False
        assert result.banking_stress is False
        assert result.currency_instability is False
        assert result.liquidity_shortage is False
        assert result.systemic_risk_score == 0.0
        assert result.contagion_probability == 0.0
        assert result.fraud_indicators == []
        assert result.policy_recommendations == []


class TestMarketCrashDetector:
    """Tests for MarketCrashDetector."""

    @pytest.fixture
    def detector(self):
        """Create MarketCrashDetector instance."""
        return MarketCrashDetector()

    def test_detect_no_crash_stable_market(self, detector: Any) -> None:
        """Test detection in stable market conditions."""
        market_data = {
            "price_change_pct": 0.5,
            "vix": 15.0,
            "volume_vs_average": 1.0,
        }
        result = detector.detect_market_crash(market_data)

        assert result["crash_detected"] is False
        assert result["severity"] == "low"
        assert result["panic_selling"] is False

    def test_detect_crash_high_volatility(self, detector: Any) -> None:
        """Test detection during high volatility crash."""
        market_data = {
            "price_change_pct": -7.0,
            "vix": 45.0,
            "volume_vs_average": 3.0,
        }
        result = detector.detect_market_crash(market_data)

        assert result["crash_detected"] is True
        assert result["severity"] == "high"
        assert result["panic_selling"] is True

    def test_detect_extreme_volatility(self, detector: Any) -> None:
        """Test detection during extreme volatility."""
        market_data = {
            "price_change_pct": -10.0,
            "vix": 55.0,
            "volume_vs_average": 4.0,
        }
        result = detector.detect_market_crash(market_data)

        assert result["crash_detected"] is True
        assert result["severity"] == "extreme"

    def test_detect_moderate_volatility(self, detector: Any) -> None:
        """Test detection during moderate volatility."""
        market_data = {
            "price_change_pct": -2.0,
            "vix": 29.0,
            "volume_vs_average": 1.2,
        }
        result = detector.detect_market_crash(market_data)

        assert result["crash_detected"] is False
        # 29.0 sits in the calibrated moderate band (p90 = 28.5 <= VIX < 32.9).
        assert result["severity"] == "moderate"

    def test_vix_below_p90_is_low(self, detector: Any) -> None:
        """Below the calibrated 90th percentile (28.5) severity is low."""
        result = detector.detect_market_crash({"price_change_pct": -2.0, "vix": 28.0})
        assert result["severity"] == "low"

    def test_crash_line_is_the_circuit_breaker_level_1(self, detector: Any) -> None:
        """The crash bar is the SEC market-wide circuit breaker (7%)."""
        under = detector.detect_market_crash({"price_change_pct": -6.9, "vix": 40.0})
        assert under["crash_detected"] is False
        at = detector.detect_market_crash({"price_change_pct": -7.0, "vix": 40.0})
        assert at["crash_detected"] is True

    def test_crash_graded_without_a_vix_print(self, detector: Any) -> None:
        """A 1987-style record with no VIX still grades as a crash.

        The retired rule required a VIX print alongside the -5% line, so
        Black Monday (-20.5%, VIX not yet published) could never fire.
        """
        result = detector.detect_market_crash({"price_change_pct": -20.5})
        assert result["crash_detected"] is True
        assert result["severity"] == "unknown"
        assert "vix" in result["missing_inputs"]

    @pytest.mark.parametrize(
        ("date", "price_change_pct", "vix", "is_crash", "severity"),
        [
            # Dated crisis days: NASDAQ Composite close-to-close change
            # (FRED NASDAQCOM) with the VIX close (FRED VIXCLS).
            ("1987-10-19", -11.35, None, True, "unknown"),
            ("2008-09-29", -9.14, 46.72, True, "extreme"),
            ("2008-10-15", -8.47, 69.25, True, "extreme"),
            ("2020-03-12", -9.43, 75.47, True, "extreme"),
            ("2020-03-16", -12.32, 82.69, True, "extreme"),
            # Dated stress-but-not-crash days (S&P 500 changes: -6.66% on
            # 2011-08-08 per the public record; -4.10% per FRED SP500).
            ("2011-08-08", -6.66, 48.00, False, "extreme"),
            ("2018-02-05", -4.10, 37.32, False, "high"),
            # Dated calm days (FRED SP500 changes).
            ("2017-06-15", -0.22, 10.90, False, "low"),
            ("2024-05-15", 1.17, 12.45, False, "low"),
        ],
    )
    def test_dated_historical_record(
        self,
        detector: Any,
        date: str,
        price_change_pct: float,
        vix: float | None,
        is_crash: bool,
        severity: str,
    ) -> None:
        """Calibration regression on the dated historical crisis table."""
        record: dict[str, Any] = {"price_change_pct": price_change_pct}
        if vix is not None:
            record["vix"] = vix
        result = detector.detect_market_crash(record)
        assert result["crash_detected"] is is_crash, date
        assert result["severity"] == severity, date

    def test_missing_inputs_are_named_not_defaulted(self, detector: Any) -> None:
        """Absent metrics are reported as None and named, never fabricated.

        Regression: empty input used to come back with vix 20.0 and
        price_change_pct 0.0 reported as if they had been measured.
        """
        market_data: dict[str, Any] = {}
        result = detector.detect_market_crash(market_data)

        assert result["crash_detected"] is False
        assert result["vix"] is None
        assert result["price_change_pct"] is None
        assert result["severity"] == "unknown"
        assert set(result["missing_inputs"]) == {
            "price_change_pct",
            "vix",
            "volume_vs_average",
        }

    def test_bad_supplied_values_are_refused(self, detector: Any) -> None:
        """A supplied NaN VIX raises instead of grading the market stable."""
        from omni_mercury_engine.detectors._hazard_input import InvalidObservationError

        with pytest.raises(InvalidObservationError, match="vix"):
            detector.detect_market_crash({"vix": float("nan"), "price_change_pct": -8.0})


class TestBankingStressDetector:
    """Tests for BankingStressDetector."""

    @pytest.fixture
    def detector(self):
        """Create BankingStressDetector instance."""
        return BankingStressDetector()

    def test_detect_no_stress(self, detector: Any) -> None:
        """Test detection with no banking stress."""
        banking_data = {
            "cds_spread_bps": 100.0,
            "default_rate_pct": 1.0,
            "liquidity_ratio": 1.5,
        }
        result = detector.detect_banking_stress(banking_data)

        assert result["banking_stress"] is False
        assert result["stress_score"] == 0.0
        assert result["liquidity_crisis"] is False

    def test_detect_high_cds_spreads(self, detector: Any) -> None:
        """Test detection with high CDS spreads."""
        banking_data = {
            "cds_spread_bps": 400.0,
            "default_rate_pct": 1.0,
            "liquidity_ratio": 1.2,
        }
        result = detector.detect_banking_stress(banking_data)

        assert result["banking_stress"] is True
        assert result["stress_score"] >= 0.4

    def test_detect_high_default_rate(self, detector: Any) -> None:
        """Test detection with high default rate."""
        banking_data = {
            "cds_spread_bps": 100.0,
            "default_rate_pct": 5.0,
            "liquidity_ratio": 1.2,
        }
        result = detector.detect_banking_stress(banking_data)

        assert result["banking_stress"] is True
        assert result["stress_score"] >= 0.3

    def test_detect_liquidity_crisis(self, detector: Any) -> None:
        """Test detection of liquidity crisis."""
        banking_data = {
            "cds_spread_bps": 100.0,
            "default_rate_pct": 1.0,
            "liquidity_ratio": 0.4,
        }
        result = detector.detect_banking_stress(banking_data)

        assert result["banking_stress"] is True
        assert result["liquidity_crisis"] is True

    def test_detect_multiple_stress_factors(self, detector: Any) -> None:
        """Test detection with multiple stress factors."""
        banking_data = {
            "cds_spread_bps": 400.0,
            "default_rate_pct": 5.0,
            "liquidity_ratio": 0.6,
        }
        result = detector.detect_banking_stress(banking_data)

        assert result["banking_stress"] is True
        assert result["stress_score"] == 1.0


class TestFraudDetector:
    """Tests for FraudDetector neural network."""

    @pytest.fixture
    def detector(self):
        """Create FraudDetector instance."""
        return FraudDetector(input_dim=64)

    def test_initialization(self, detector: Any) -> None:
        """Test model initialization."""
        assert detector.pattern_encoder is not None
        assert detector.fraud_classifier is not None

    def test_forward_pass(self, detector: Any) -> None:
        """Test forward pass through model."""
        input_tensor = torch.randn(10, 64)
        output = detector(input_tensor)

        assert output.shape == (10, 1)
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)

    def test_batch_processing(self, detector: Any) -> None:
        """Test processing of batched inputs."""
        # batch_size > 1 required for BatchNorm during training
        batch_sizes = [2, 5, 20]
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 64)
            output = detector(input_tensor)
            assert output.shape == (batch_size, 1)

        # batch_size=1 works in eval mode
        detector.eval()
        single_input = torch.randn(1, 64)
        single_output = detector(single_input)
        assert single_output.shape == (1, 1)


class TestSystemicRiskAnalyzer:
    """Tests for SystemicRiskAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create SystemicRiskAnalyzer instance."""
        return SystemicRiskAnalyzer()

    def test_low_systemic_risk(self, analyzer: Any) -> None:
        """Test assessment with low systemic risk."""
        network_data = {
            "interconnectedness_score": 0.2,
            "concentration_ratio": 0.1,
            "cross_border_exposure": 0.1,
        }
        result = analyzer.assess_systemic_risk(network_data)

        assert result["systemic_risk_score"] < 0.5
        assert result["too_big_to_fail"] is False

    def test_high_systemic_risk(self, analyzer: Any) -> None:
        """Test assessment with high systemic risk."""
        network_data = {
            "interconnectedness_score": 0.9,
            "concentration_ratio": 0.7,
            "cross_border_exposure": 0.6,
        }
        result = analyzer.assess_systemic_risk(network_data)

        assert result["systemic_risk_score"] > 0.7
        assert result["contagion_probability"] > 0.5
        assert result["too_big_to_fail"] is True

    def test_contagion_capped_at_one(self, analyzer: Any) -> None:
        """Test that contagion probability is capped at 1.0."""
        network_data = {
            "interconnectedness_score": 1.0,
            "concentration_ratio": 1.0,
            "cross_border_exposure": 1.0,
        }
        result = analyzer.assess_systemic_risk(network_data)

        assert result["contagion_probability"] <= 1.0


class TestFinancialCrisisDetector:
    """Tests for comprehensive FinancialCrisisDetector."""

    @pytest.fixture
    def detector(self):
        """Create FinancialCrisisDetector instance."""
        return FinancialCrisisDetector()

    @pytest.fixture
    def stable_financial_data(self):
        """Create stable financial conditions data."""
        return {
            "market_data": {
                "price_change_pct": 0.5,
                "vix": 15.0,
                "volume_vs_average": 1.0,
            },
            "banking_data": {
                "cds_spread_bps": 100.0,
                "default_rate_pct": 1.0,
                "liquidity_ratio": 1.5,
            },
            "trading_data": {"volume_anomaly_score": 0.1},
            "network_data": {
                "interconnectedness_score": 0.3,
                "concentration_ratio": 0.2,
                "cross_border_exposure": 0.2,
            },
        }

    @pytest.fixture
    def crisis_financial_data(self):
        """Create crisis financial conditions data."""
        return {
            "market_data": {
                "price_change_pct": -8.0,
                "vix": 50.0,
                "volume_vs_average": 3.0,
            },
            "banking_data": {
                "cds_spread_bps": 500.0,
                "default_rate_pct": 5.0,
                "liquidity_ratio": 0.5,
            },
            "trading_data": {"volume_anomaly_score": 0.9},
            "network_data": {
                "interconnectedness_score": 0.9,
                "concentration_ratio": 0.8,
                "cross_border_exposure": 0.7,
            },
        }

    def test_initialization_all_enabled(self, detector: Any) -> None:
        """Test initialization with all detectors enabled."""
        assert detector.market_detector is not None
        assert detector.banking_detector is not None
        assert detector.fraud_detector is not None
        assert detector.systemic_analyzer is not None

    def test_initialization_selective_enablement(self) -> None:
        """Test initialization with selective enablement."""
        detector = FinancialCrisisDetector(
            enable_market_detection=True,
            enable_banking_detection=False,
            enable_fraud_detection=False,
            enable_systemic_analysis=True,
        )

        assert detector.market_detector is not None
        assert detector.banking_detector is None
        assert detector.fraud_detector is None
        assert detector.systemic_analyzer is not None

    def test_predict_stable_conditions(self, detector: Any, stable_financial_data: Any) -> None:
        """Test prediction in stable conditions."""
        result = detector.predict_financial_crisis(stable_financial_data)

        assert isinstance(result, FinancialCrisisPredictionResult)
        assert result.crisis_imminent is False
        assert result.severity_level == "stable"

    def test_predict_crisis_conditions(self, detector: Any, crisis_financial_data: Any) -> None:
        """Test prediction in crisis conditions."""
        result = detector.predict_financial_crisis(crisis_financial_data)

        assert result.crisis_imminent is True
        assert result.severity_level in ["crisis", "severe_crisis", "systemic_collapse"]
        assert result.market_crash_detected is True
        assert result.banking_stress is True

    def test_policy_recommendations_generated(
        self, detector: Any, crisis_financial_data: Any
    ) -> None:
        """Test that policy recommendations are generated."""
        result = detector.predict_financial_crisis(crisis_financial_data)

        assert len(result.policy_recommendations) > 0

    def test_intervention_actions_generated(
        self, detector: Any, crisis_financial_data: Any
    ) -> None:
        """Test that intervention actions are generated."""
        result = detector.predict_financial_crisis(crisis_financial_data)

        assert len(result.intervention_actions) > 0

    def test_affected_sectors_identified(self, detector: Any, crisis_financial_data: Any) -> None:
        """Test that affected sectors are identified."""
        result = detector.predict_financial_crisis(crisis_financial_data)

        assert "Financial Services" in result.affected_sectors

    def test_currency_crisis_detection(self, detector: Any) -> None:
        """Test currency crisis detection."""
        data = {
            "currency_data": {"volatility": 0.2},
            "market_data": {"price_change_pct": 0.0, "vix": 20.0},
        }
        result = detector.predict_financial_crisis(data)

        assert result.currency_instability is True

    def test_vix_level_captured(self, detector: Any, stable_financial_data: Any) -> None:
        """Test that VIX level is captured in result."""
        result = detector.predict_financial_crisis(stable_financial_data)

        assert result.vix_level == 15.0
        assert result.market_volatility_index == 15.0

    def test_systemic_risk_score_captured(self, detector: Any, crisis_financial_data: Any) -> None:
        """Test that systemic risk score is captured."""
        result = detector.predict_financial_crisis(crisis_financial_data)

        assert result.systemic_risk_score > 0.0
        assert result.contagion_probability > 0.0

    def test_empty_data_handling(self, detector: Any) -> None:
        """Empty financial data is refused: it used to grade as stable."""
        with pytest.raises(InvalidObservationError, match="no gradable section"):
            detector.predict_financial_crisis({})

    def test_partial_data_handling(self, detector: Any) -> None:
        """Test handling of partial financial data."""
        # 2020-03-09 (dated): S&P 500 -7.60%, past the circuit-breaker line.
        partial_data = {"market_data": {"price_change_pct": -7.6, "vix": 54.46}}
        result = detector.predict_financial_crisis(partial_data)

        assert result.market_crash_detected is True
        # Should not crash with missing data
        assert result.banking_stress is False

    def test_panic_selling_reaches_the_verdict(self, detector: Any) -> None:
        """volume_vs_average must change the outcome, not be discarded.

        A heavy-volume -3.5% day short of the circuit breaker is half a
        crisis indicator; the same day on normal volume contributes nothing.
        """
        heavy = detector.predict_financial_crisis(
            {"market_data": {"price_change_pct": -3.5, "vix": 30.0, "volume_vs_average": 3.0}}
        )
        calm = detector.predict_financial_crisis(
            {"market_data": {"price_change_pct": -3.5, "vix": 30.0, "volume_vs_average": 1.0}}
        )
        assert heavy.panic_selling is True
        assert calm.panic_selling is False
        assert heavy.confidence > calm.confidence


class TestHonestyContract:
    """Fabrication and mislabel regressions on the full predict path."""

    def test_banking_stress_alone_is_a_banking_crisis(self) -> None:
        """Regression: banking stress with no market crash read 'systemic_crisis'
        because the former check compared crisis_type against its default string."""
        detector = FinancialCrisisDetector()
        result = detector.predict_financial_crisis(
            {
                "banking_data": {
                    "cds_spread_bps": 500.0,
                    "default_rate_pct": 1.0,
                    "liquidity_ratio": 1.2,
                }
            }
        )
        assert result.banking_stress is True
        assert result.crisis_type == "banking_crisis"

    def test_empty_input_reports_no_fabricated_measurements(self) -> None:
        """Regression: empty input used to report CDS 100 bps as a measurement.

        Empty sections are now refused outright; a partly measured record
        still reports the unmeasured values as None.
        """
        detector = FinancialCrisisDetector()
        with pytest.raises(InvalidObservationError, match="no measurement"):
            detector.predict_financial_crisis(
                {"market_data": {}, "banking_data": {}, "network_data": {}}
            )
        result = detector.predict_financial_crisis({"market_data": {"price_change_pct": -1.0}})
        assert result.vix_level is None
        assert result.credit_default_swap_spread is None
        assert result.systemic_risk_score == 0.0  # dataclass default, nothing scored
        assert result.severity_level == "stable"

    @pytest.mark.parametrize(
        "record",
        [
            {"market_data": {"VIX": 60.0}},  # misspelt: read as unmeasured
            {"banking_data": {"cds_bps": 900.0}},
            {"network_data": {"interconnectedness": 0.95}},
            {"currency_data": {"vol": 0.4}},
        ],
    )
    def test_misspelt_measurement_is_refused(self, record: dict[str, Any]) -> None:
        """A misspelt key used to go unread, so a stressed market graded stable."""
        with pytest.raises(InvalidObservationError, match="unknown keys"):
            FinancialCrisisDetector().predict_financial_crisis(record)

    @pytest.mark.parametrize("section", ["market_data", "banking_data", "currency_data"])
    def test_non_mapping_section_is_refused(self, section: str) -> None:
        with pytest.raises(InputModalityError, match="mapping"):
            FinancialCrisisDetector().predict_financial_crisis({section: [1.0]})

    def test_untrained_fraud_network_is_not_consulted(self) -> None:
        """Regression: a randomly-initialised FraudDetector was run and its noise
        could append 'Algorithmic manipulation detected' to the result."""
        detector = FinancialCrisisDetector()
        fraud = detector._detect_fraud({"volume_anomaly_score": 0.9})
        assert fraud == {"fraud_detected": False, "fraud_probability": None, "consulted": False}
        # Trading data alone feeds only the unconsulted network: refused.
        with pytest.raises(InvalidObservationError, match="no gradable section"):
            detector.predict_financial_crisis({"trading_data": {"volume_anomaly_score": 0.9}})
        result = detector.predict_financial_crisis(
            {
                "market_data": {"price_change_pct": 0.5, "vix": 15.0},
                "trading_data": {"volume_anomaly_score": 0.9},
            }
        )
        assert result.fraud_indicators == []

    def test_fraud_is_consulted_once_weights_load(self, tmp_path: Any) -> None:
        import torch

        from omni_mercury_engine.detectors.economic.financial_crisis_detector import (
            FraudDetector,
        )

        detector = FinancialCrisisDetector()
        trained = FraudDetector()
        path = tmp_path / "fraud.pt"
        torch.save(trained.state_dict(), path)
        detector.load_fraud_weights(str(path))
        fraud = detector._detect_fraud({"volume_anomaly_score": 0.9})
        assert fraud["consulted"] is True
        assert isinstance(fraud["fraud_probability"], float)
