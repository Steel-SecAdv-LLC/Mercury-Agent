"""
Tests for omni_mercury_engine.security.int_sources module.

Tests intelligence source processors (OSINT, COMINT, HUMINT, etc.).
"""

from __future__ import annotations

from omni_mercury_engine.security.int_sources import (
    COMINTAnalysisResult,
    COMINTProcessor,
    CYBINTProcessor,
    ELINTAnalysisResult,
    ELINTProcessor,
    FININTAnalysisResult,
    FININTProcessor,
    GEOINTAnalysisResult,
    GEOINTProcessor,
    HUMINTAnalysisResult,
    HUMINTProcessor,
    MASINTAnalysisResult,
    MASINTProcessor,
    OSINTAnalysisResult,
    OSINTProcessor,
    SIGINTAnalysisResult,
    SIGINTProcessor,
)


class TestOSINTProcessor:
    """Tests for OSINTProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = OSINTProcessor()

    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor is not None
        assert len(self.processor.source_reliability) > 0

    def test_source_reliability_values(self):
        """Test that source reliability values are valid."""
        for source, reliability in self.processor.source_reliability.items():
            assert 0.0 <= reliability <= 1.0
            assert isinstance(source, str)

    def test_analyze_basic(self):
        """Test basic analysis with minimal data."""
        data = {
            "source_type": "mainstream_media",
            "content": "Test content for analysis.",
        }
        result = self.processor.analyze(data)

        assert isinstance(result, OSINTAnalysisResult)
        assert result.source_credibility == 0.85
        assert 0.0 <= result.information_quality <= 1.0
        assert 0.0 <= result.corroboration_level <= 1.0

    def test_analyze_unknown_source(self):
        """Test analysis with unknown source type."""
        data = {"source_type": "unknown_source", "content": "Test content."}
        result = self.processor.analyze(data)

        assert result.source_credibility == 0.50  # Default

    def test_analyze_with_threat_keywords(self):
        """Test that threat keywords trigger anomaly detection."""
        data = {
            "source_type": "social_media",
            "content": "There was a threat and attack on the target location.",
        }
        result = self.processor.analyze(data)

        assert len(result.anomaly_indicators) > 0
        assert any("threat_keyword" in ind for ind in result.anomaly_indicators)

    def test_analyze_with_citations(self):
        """Test that citations improve quality score."""
        data = {
            "source_type": "academic",
            "content": "A" * 150,  # Long content
            "has_citations": True,
            "author_verified": True,
        }
        result = self.processor.analyze(data)

        assert result.information_quality > 0.7

    def test_analyze_with_corroboration(self):
        """Test corroboration level calculation."""
        data = {
            "source_type": "government_public",
            "content": "Official statement.",
            "corroborating_sources": 5,
        }
        result = self.processor.analyze(data)

        assert result.corroboration_level == 1.0

    def test_analyze_temporal_anomaly(self):
        """Test temporal anomaly detection."""
        data = {
            "source_type": "social_media",
            "content": "Normal content",
            "unusual_posting_time": True,
        }
        result = self.processor.analyze(data)

        assert "temporal_anomaly" in result.anomaly_indicators

    def test_analyze_coordinated_campaign(self):
        """Test coordinated campaign detection."""
        data = {
            "source_type": "social_media",
            "content": "Campaign content",
            "coordinated_campaign": True,
        }
        result = self.processor.analyze(data)

        assert "coordinated_information_operation" in result.anomaly_indicators

    def test_sentiment_analysis(self):
        """Test sentiment analysis is included."""
        data = {"source_type": "blogs", "content": "Very positive happy content!"}
        result = self.processor.analyze(data)

        assert result.sentiment_analysis is not None
        assert isinstance(result.sentiment_analysis, dict)

    def test_entity_extraction(self):
        """Test entity extraction is included."""
        data = {"source_type": "mainstream_media", "content": "Test content with entities."}
        result = self.processor.analyze(data)

        assert isinstance(result.entity_mentions, list)


class TestCOMINTProcessor:
    """Tests for COMINTProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = COMINTProcessor()

    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor is not None

    def test_analyze_basic(self):
        """Test basic COMINT analysis."""
        data = {
            "communication_type": "radio",
            "frequency": 145.5,
            "duration_seconds": 60,
            "content": "Test transmission",
        }
        result = self.processor.analyze(data)

        assert isinstance(result, COMINTAnalysisResult)
        assert hasattr(result, "intercept_quality")
        assert hasattr(result, "anomaly_score")

    def test_analyze_encrypted(self):
        """Test analysis of encrypted communication."""
        data = {
            "communication_type": "digital",
            "encrypted": True,
            "content": "encrypted_blob",
        }
        result = self.processor.analyze(data)

        assert result is not None


class TestHUMINTProcessor:
    """Tests for HUMINTProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = HUMINTProcessor()

    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor is not None

    def test_analyze_basic(self):
        """Test basic HUMINT analysis."""
        data = {
            "source_reliability": "B",
            "information_credibility": "2",
            "report_content": "Asset reports activity in sector.",
        }
        result = self.processor.analyze(data)

        assert isinstance(result, HUMINTAnalysisResult)
        assert hasattr(result, "source_reliability_score")
        assert hasattr(result, "information_credibility_score")

    def test_reliability_coding(self):
        """Test NATO reliability coding."""
        for code in ["A", "B", "C", "D", "E", "F"]:
            data = {
                "source_reliability": code,
                "information_credibility": "1",
                "report_content": "Test",
            }
            result = self.processor.analyze(data)
            assert 0.0 <= result.source_reliability_score <= 1.0


class TestGEOINTProcessor:
    """Tests for GEOINTProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = GEOINTProcessor()

    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor is not None

    def test_analyze_basic(self):
        """Test basic GEOINT analysis."""
        data = {
            "imagery_type": "satellite",
            "resolution_meters": 0.5,
            "coordinates": {"lat": 40.7128, "lon": -74.0060},
            "timestamp": "2025-01-01T12:00:00Z",
        }
        result = self.processor.analyze(data)

        assert isinstance(result, GEOINTAnalysisResult)
        assert hasattr(result, "spatial_anomalies")

    def test_analyze_change_detection(self):
        """Test change detection analysis."""
        data = {
            "imagery_type": "satellite",
            "previous_imagery": True,
            "change_detected": True,
            "change_magnitude": 0.75,
        }
        result = self.processor.analyze(data)

        assert result is not None


class TestSIGINTProcessor:
    """Tests for SIGINTProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = SIGINTProcessor()

    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor is not None

    def test_analyze_basic(self):
        """Test basic SIGINT analysis."""
        data = {
            "signal_type": "radio",
            "frequency_mhz": 145.5,
            "modulation": "FM",
            "signal_strength_dbm": -50,
        }
        result = self.processor.analyze(data)

        assert isinstance(result, SIGINTAnalysisResult)
        assert hasattr(result, "signal_classification")


class TestELINTProcessor:
    """Tests for ELINTProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ELINTProcessor()

    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor is not None

    def test_analyze_basic(self):
        """Test basic ELINT analysis."""
        data = {
            "emitter_type": "radar",
            "frequency_ghz": 9.5,
            "pulse_width_us": 1.0,
            "pri_us": 1000,
        }
        result = self.processor.analyze(data)

        assert isinstance(result, ELINTAnalysisResult)
        assert hasattr(result, "emitter_classification")


class TestMASINTProcessor:
    """Tests for MASINTProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = MASINTProcessor()

    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor is not None

    def test_analyze_basic(self):
        """Test basic MASINT analysis."""
        data = {
            "measurement_type": "seismic",
            "sensor_readings": [0.1, 0.2, 0.15, 0.18],
            "timestamp": "2025-01-01T12:00:00Z",
        }
        result = self.processor.analyze(data)

        assert isinstance(result, MASINTAnalysisResult)
        assert hasattr(result, "signature_classification")


class TestCYBINTProcessor:
    """Tests for CYBINTProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = CYBINTProcessor()

    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor is not None

    def test_analyze_basic(self):
        """Test basic CYBINT analysis."""
        data = {
            "threat_type": "malware",
            "indicators": ["hash123", "domain.com"],
            "severity": "high",
        }
        result = self.processor.analyze(data)

        assert result is not None


class TestFININTProcessor:
    """Tests for FININTProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = FININTProcessor()

    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor is not None

    def test_analyze_basic(self):
        """Test basic FININT analysis."""
        data = {
            "transaction_type": "wire_transfer",
            "amount": 50000,
            "currency": "USD",
            "origin_country": "US",
            "destination_country": "CH",
        }
        result = self.processor.analyze(data)

        assert isinstance(result, FININTAnalysisResult)
        assert hasattr(result, "risk_score")

    def test_suspicious_pattern_detection(self):
        """Test suspicious pattern detection."""
        data = {
            "transaction_type": "wire_transfer",
            "amount": 9999,  # Just below reporting threshold
            "currency": "USD",
            "structuring_pattern": True,
        }
        result = self.processor.analyze(data)

        assert result.risk_score > 0.5


class TestAnalysisResultDataclasses:
    """Tests for analysis result dataclasses."""

    def test_osint_result_defaults(self):
        """Test OSINTAnalysisResult default values."""
        result = OSINTAnalysisResult(
            source_credibility=0.8, information_quality=0.7, corroboration_level=0.5
        )
        assert result.anomaly_indicators == []
        assert result.entity_mentions == []
        assert result.sentiment_analysis is None

    def test_comint_result_creation(self):
        """Test COMINTAnalysisResult creation."""
        result = COMINTAnalysisResult(
            intercept_quality=0.9, anomaly_score=0.3, communication_type="radio"
        )
        assert result.intercept_quality == 0.9
        assert result.anomaly_score == 0.3

    def test_humint_result_creation(self):
        """Test HUMINTAnalysisResult creation."""
        result = HUMINTAnalysisResult(
            source_reliability_score=0.8,
            information_credibility_score=0.7,
            report_assessment="confirmed",
        )
        assert result.source_reliability_score == 0.8
