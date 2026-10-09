"""Unit tests for StreamTelemetryTracker."""

import concurrent.futures
import pytest
from kaz_fin_nlp.sentiment_analyzer import (
    MarketUrgency,
    SentimentLabel,
    SentimentResult,
)
from kaz_fin_nlp.telemetry import StreamTelemetryTracker


def test_telemetry_recording_and_metrics() -> None:
    tracker = StreamTelemetryTracker(window_size=100)
    assert tracker.get_metrics().total_articles == 0

    dummy_pos = SentimentResult(
        label=SentimentLabel.POSITIVE,
        polarity=0.8,
        confidence=0.9,
        urgency=MarketUrgency.LOW,
    )
    dummy_neg = SentimentResult(
        label=SentimentLabel.NEGATIVE,
        polarity=-0.7,
        confidence=0.85,
        urgency=MarketUrgency.CRITICAL,
    )

    tracker.record_document(dummy_pos, latency_ms=1.5, pii_redacted_count=2)
    tracker.record_document(dummy_neg, latency_ms=2.5, pii_redacted_count=1)

    m = tracker.get_metrics()
    assert m.total_articles == 2
    assert m.positive_articles == 1
    assert m.negative_articles == 1
    assert m.total_pii_redactions == 3
    assert m.avg_latency_ms == pytest.approx(2.0, rel=0.1)
    assert m.throughput_articles_sec > 0


def test_telemetry_reset() -> None:
    tracker = StreamTelemetryTracker()
    dummy = SentimentResult(
        label=SentimentLabel.NEUTRAL,
        polarity=0.0,
        confidence=0.5,
        urgency=MarketUrgency.LOW,
    )
    tracker.record_document(dummy, latency_ms=1.0)
    assert tracker.get_metrics().total_articles == 1

    tracker.reset()
    assert tracker.get_metrics().total_articles == 0
    assert tracker.get_metrics().avg_latency_ms == 0.0


def test_telemetry_thread_safety() -> None:
    tracker = StreamTelemetryTracker(window_size=2000)
    dummy = SentimentResult(
        label=SentimentLabel.POSITIVE,
        polarity=0.5,
        confidence=0.8,
        urgency=MarketUrgency.MEDIUM,
    )

    def worker() -> None:
        for _ in range(50):
            tracker.record_document(dummy, latency_ms=1.2, pii_redacted_count=1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker) for _ in range(4)]
        concurrent.futures.wait(futures)

    metrics = tracker.get_metrics()
    assert metrics.total_articles == 200
    assert metrics.total_pii_redactions == 200
