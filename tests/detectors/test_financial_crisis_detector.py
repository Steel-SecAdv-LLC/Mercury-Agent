# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for Financial Crisis Detector."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("torch")

import pytest
import torch

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
            "vix": 28.0,
            "volume_vs_average": 1.2,
        }
        result = detector.detect_market_crash(market_data)

        assert result["crash_detected"] is False
        assert result["severity"] == "moderate"

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
        """Test handling of empty financial data."""
        result = detector.predict_financial_crisis({})

        assert result.crisis_imminent is False
        assert result.severity_level == "stable"

    def test_partial_data_handling(self, detector: Any) -> None:
        """Test handling of partial financial data."""
        partial_data = {"market_data": {"price_change_pct": -6.0, "vix": 40.0}}
        result = detector.predict_financial_crisis(partial_data)

        assert result.market_crash_detected is True
        # Should not crash with missing data
        assert result.banking_stress is False


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
        """Regression: empty input used to report CDS 100 bps as a measurement."""
        detector = FinancialCrisisDetector()
        result = detector.predict_financial_crisis(
            {"market_data": {}, "banking_data": {}, "network_data": {}}
        )
        assert result.vix_level is None
        assert result.credit_default_swap_spread is None
        assert result.systemic_risk_score == 0.0  # dataclass default, nothing scored
        assert result.severity_level == "stable"

    def test_untrained_fraud_network_is_not_consulted(self) -> None:
        """Regression: a randomly-initialised FraudDetector was run and its noise
        could append 'Algorithmic manipulation detected' to the result."""
        detector = FinancialCrisisDetector()
        fraud = detector._detect_fraud({"volume_anomaly_score": 0.9})
        assert fraud == {"fraud_detected": False, "fraud_probability": None, "consulted": False}
        result = detector.predict_financial_crisis({"trading_data": {"volume_anomaly_score": 0.9}})
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
