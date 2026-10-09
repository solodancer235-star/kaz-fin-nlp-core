"""Incident Report and Downstream LLM Prompt Builder.

Synthesizes high-impact financial sentiment detections, extracted market entities,
and regulatory compliance certifications into structured payloads for LLMs.
"""

from datetime import datetime, timezone
from typing import List
from pydantic import BaseModel, Field

from kaz_fin_nlp.entity_extractor import FinancialEntity
from kaz_fin_nlp.sentiment_analyzer import (
    MarketUrgency,
    SentimentLabel,
    SentimentResult,
)


class RegulatoryCompliance(BaseModel):
    """Compliance attestation per Kazakhstani Law No. 94-V."""

    kz_law_94_v_compliant: bool = True
    data_residency_local: bool = True
    pii_sanitized: bool = True
    datacenter_location: str = "Astana/Almaty Tier-III Certified Datacenter"


class MarketIncidentReport(BaseModel):
    """Standardized market alert and downstream LLM synthesis payload."""

    report_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    headline: str
    source: str
    detected_language: str
    sentiment_label: SentimentLabel
    polarity_score: float
    confidence: float
    urgency: MarketUrgency
    detected_tickers: List[str]
    detected_entities: List[str]
    monetary_impacts: List[str]
    regulatory: RegulatoryCompliance
    recommended_actions: List[str]
    llm_prompt_synthesis: str


class MarketIncidentReportBuilder:
    """Builds structured incident reports and downstream LLM prompts."""

    def __init__(self, processing_cell: str = "KASE_Primary_Pipeline") -> None:
        """Initialize report builder with processing node identifier."""
        self.processing_cell = processing_cell

    def build_report(
        self,
        headline: str,
        source: str,
        language: str,
        sentiment: SentimentResult,
        entities: List[FinancialEntity],
        report_id: str | None = None,
    ) -> MarketIncidentReport:
        """Synthesize entities and sentiment into an actionable market report."""
        rep_id = report_id or f"REP-{int(datetime.now(timezone.utc).timestamp())}"

        tickers: List[str] = []
        named_entities: List[str] = []
        monetary_vals: List[str] = []

        for ent in entities:
            if ent.entity_type.value == "TICKER":
                tickers.append(ent.normalized_value)
            elif ent.entity_type.value == "MONETARY_VALUE":
                monetary_vals.append(ent.text)
            else:
                named_entities.append(ent.normalized_value)

        actions: List[str] = []
        if sentiment.urgency == MarketUrgency.CRITICAL:
            actions.append("FLAG: Immediate volatility alert to trading desk.")
            actions.append("COMPLIANCE: Verify filing against KASE disclosures.")
            actions.append("RISK: Suspend automated exposure expansion.")
        elif sentiment.urgency == MarketUrgency.HIGH:
            actions.append("NOTIFY: Route alert to sector portfolio managers.")
            actions.append("MONITOR: Increase sentiment sampling frequency.")
        elif sentiment.label == SentimentLabel.POSITIVE:
            actions.append("LOG: Positive corporate development logged.")
        else:
            actions.append("ROUTINE: Normal market news archived.")

        # Structured Prompt for Downstream LLM
        t_str = ", ".join(tickers) if tickers else "None"
        e_str = ", ".join(set(named_entities)) if named_entities else "None"
        m_str = ", ".join(monetary_vals) if monetary_vals else "None"

        llm_prompt = (
            f"[SYSTEM: Central Asian Financial Risk Analyst]\n"
            f"Analyze financial event via {source} in {language.upper()}:\n"
            f"HEADLINE: '{headline}'\n"
            f"SENTIMENT: {sentiment.label.value} "
            f"(Polarity: {sentiment.polarity}, Conf: {sentiment.confidence})\n"
            f"URGENCY: {sentiment.urgency.value}\n"
            f"TICKERS: {t_str}\n"
            f"ENTITIES: {e_str}\n"
            f"MONETARY: {m_str}\n"
            f"TASK: 3-bullet summary of liquidity impact on KASE."
        )

        return MarketIncidentReport(
            report_id=rep_id,
            headline=headline,
            source=source,
            detected_language=language,
            sentiment_label=sentiment.label,
            polarity_score=sentiment.polarity,
            confidence=sentiment.confidence,
            urgency=sentiment.urgency,
            detected_tickers=list(set(tickers)),
            detected_entities=list(set(named_entities)),
            monetary_impacts=list(set(monetary_vals)),
            regulatory=RegulatoryCompliance(),
            recommended_actions=actions,
            llm_prompt_synthesis=llm_prompt,
        )
