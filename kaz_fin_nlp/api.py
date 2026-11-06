"""REST inference service for the Kazakhstani NLP Financial News System.

Wraps the computational core (preprocessor, sentiment analyzer, entity
extractor, report builder, telemetry) in a FastAPI application. This is the
unit that is containerised with Docker and deployed to Kubernetes.
"""

import time
from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel, Field

from kaz_fin_nlp import (
    BilingualTextPreprocessor,
    FinancialEntityExtractor,
    FinancialSentimentAnalyzer,
    MarketIncidentReport,
    MarketIncidentReportBuilder,
    StreamTelemetryTracker,
    __version__,
)

app = FastAPI(
    title="kaz-fin-nlp-core",
    version=__version__,
    description="Bilingual (Kazakh/Russian) financial news NLP API",
)

_preprocessor = BilingualTextPreprocessor()
_analyzer = FinancialSentimentAnalyzer()
_extractor = FinancialEntityExtractor()
_builder = MarketIncidentReportBuilder(processing_cell="K8S_API_Pipeline")
_telemetry = StreamTelemetryTracker()


class ArticleRequest(BaseModel):
    """Incoming news article."""

    text: str = Field(..., min_length=1, max_length=20000)
    source: str = Field(default="API", max_length=64)


class AnalysisResponse(BaseModel):
    """Analysis result: incident report plus processing metadata."""

    report: MarketIncidentReport
    pii_redacted_count: int
    latency_ms: float


@app.get("/health")
def health() -> Dict[str, str]:
    """Liveness/readiness probe used by Docker HEALTHCHECK and Kubernetes."""
    return {"status": "ok", "version": __version__}


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(article: ArticleRequest) -> AnalysisResponse:
    """Run the full NLP pipeline on one article."""
    start = time.perf_counter()
    # PII is removed before any downstream component sees the text (Law 94-V).
    safe_text, _ = _preprocessor.sanitize_pii(article.text)
    doc = _preprocessor.process(article.text)
    sentiment = _analyzer.analyze(doc.stems, raw_text=doc.cleaned_text)
    entities = _extractor.extract_entities(safe_text)
    report = _builder.build_report(
        headline=safe_text,
        source=article.source,
        language=doc.detected_language.value,
        sentiment=sentiment,
        entities=entities,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    _telemetry.record_document(sentiment, latency_ms, doc.pii_redacted_count)
    return AnalysisResponse(
        report=report,
        pii_redacted_count=doc.pii_redacted_count,
        latency_ms=round(latency_ms, 3),
    )


@app.get("/metrics")
def metrics() -> Dict[str, float | int | bool]:
    """Rolling telemetry snapshot (throughput, latency percentiles, NFR1)."""
    m = _telemetry.get_metrics()
    return {
        "total_articles": m.total_articles,
        "total_pii_redactions": m.total_pii_redactions,
        "avg_latency_ms": m.avg_latency_ms,
        "p95_latency_ms": m.p95_latency_ms,
        "p99_latency_ms": m.p99_latency_ms,
        "nfr1_target_met": m.nfr1_target_met,
    }
