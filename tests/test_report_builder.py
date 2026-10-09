"""Unit tests for MarketIncidentReportBuilder."""

from kaz_fin_nlp.entity_extractor import EntityType, FinancialEntity
from kaz_fin_nlp.report_builder import (
    MarketIncidentReport,
    MarketIncidentReportBuilder,
)
from kaz_fin_nlp.sentiment_analyzer import (
    MarketUrgency,
    SentimentLabel,
    SentimentResult,
)


def test_build_report_structure() -> None:
    builder = MarketIncidentReportBuilder(processing_cell="KASE_Alpha")
    sentiment = SentimentResult(
        label=SentimentLabel.NEGATIVE,
        polarity=-0.75,
        confidence=0.92,
        urgency=MarketUrgency.CRITICAL,
    )
    entities = [
        FinancialEntity(
            text="HSBK",
            entity_type=EntityType.TICKER,
            normalized_value="HSBK (Halyk Bank Kazakhstan)",
            start_char=0,
            end_char=4,
            confidence=0.98,
        ),
        FinancialEntity(
            text="Halyk Bank",
            entity_type=EntityType.BANK_FINTECH,
            normalized_value="Halyk Bank",
            start_char=5,
            end_char=15,
            confidence=0.95,
        ),
        FinancialEntity(
            text="25 млрд тенге",
            entity_type=EntityType.MONETARY_VALUE,
            normalized_value="25 млрд тенге",
            start_char=20,
            end_char=33,
            confidence=0.90,
        ),
    ]

    report: MarketIncidentReport = builder.build_report(
        headline="HSBK Halyk Bank дефолт қаупі туралы мәлімдеді: 25 млрд тенге",
        source="KASE_FEED",
        language="kk",
        sentiment=sentiment,
        entities=entities,
        report_id="TEST-REP-001",
    )

    assert report.report_id == "TEST-REP-001"
    assert report.urgency == MarketUrgency.CRITICAL
    assert "HSBK (Halyk Bank Kazakhstan)" in report.detected_tickers
    assert "Halyk Bank" in report.detected_entities
    assert any("25 млрд тенге" in m for m in report.monetary_impacts)
    assert report.regulatory.kz_law_94_v_compliant is True
    assert report.regulatory.data_residency_local is True
    assert len(report.recommended_actions) >= 3
    assert (
        "[SYSTEM: Central Asian Financial Risk Analyst]" in report.llm_prompt_synthesis
    )

    # Check JSON export
    dump = report.model_dump(mode="json")
    assert dump["report_id"] == "TEST-REP-001"
    assert dump["sentiment_label"] == "NEGATIVE"


def test_build_report_high_and_positive_urgency() -> None:
    builder = MarketIncidentReportBuilder()

    # Test HIGH urgency
    res_high = SentimentResult(
        label=SentimentLabel.NEGATIVE,
        polarity=-0.45,
        confidence=0.8,
        urgency=MarketUrgency.HIGH,
    )
    rep_high = builder.build_report("High risk news", "KASE", "ru", res_high, [])
    assert any("NOTIFY:" in a for a in rep_high.recommended_actions)

    # Test POSITIVE sentiment
    res_pos = SentimentResult(
        label=SentimentLabel.POSITIVE,
        polarity=0.3,
        confidence=0.7,
        urgency=MarketUrgency.LOW,
    )
    rep_pos = builder.build_report("Positive news", "KASE", "kk", res_pos, [])
    assert any("LOG:" in a for a in rep_pos.recommended_actions)

    # Test NEUTRAL / ROUTINE
    res_neut = SentimentResult(
        label=SentimentLabel.NEUTRAL,
        polarity=0.0,
        confidence=0.5,
        urgency=MarketUrgency.LOW,
    )
    rep_neut = builder.build_report("Routine news", "KASE", "ru", res_neut, [])
    assert any("ROUTINE:" in a for a in rep_neut.recommended_actions)
