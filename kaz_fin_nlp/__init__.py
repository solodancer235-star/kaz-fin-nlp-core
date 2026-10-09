"""Kazakhstani NLP Financial News System - Core Computational Module.

A high-throughput, bilingual (Kazakh & Russian) text processing,
financial sentiment analysis, entity extraction, and compliance framework.
"""

from kaz_fin_nlp.preprocessor import (
    BilingualTextPreprocessor,
    Language,
    PreprocessedDocument,
    PreprocessingConfig,
)
from kaz_fin_nlp.sentiment_analyzer import (
    FinancialSentimentAnalyzer,
    MarketUrgency,
    SentimentConfig,
    SentimentLabel,
    SentimentResult,
)
from kaz_fin_nlp.entity_extractor import (
    EntityType,
    FinancialEntity,
    FinancialEntityExtractor,
)
from kaz_fin_nlp.telemetry import (
    StreamTelemetryTracker,
    TelemetryMetrics,
)
from kaz_fin_nlp.report_builder import (
    MarketIncidentReport,
    MarketIncidentReportBuilder,
    RegulatoryCompliance,
)

__version__ = "0.1.0"

__all__ = [
    "BilingualTextPreprocessor",
    "Language",
    "PreprocessedDocument",
    "PreprocessingConfig",
    "FinancialSentimentAnalyzer",
    "MarketUrgency",
    "SentimentConfig",
    "SentimentLabel",
    "SentimentResult",
    "EntityType",
    "FinancialEntity",
    "FinancialEntityExtractor",
    "StreamTelemetryTracker",
    "TelemetryMetrics",
    "MarketIncidentReport",
    "MarketIncidentReportBuilder",
    "RegulatoryCompliance",
    "__version__",
]
