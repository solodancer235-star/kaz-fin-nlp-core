"""Telemetry and Real-Time Performance Monitor.

Tracks real-time processing throughput, inference latency percentiles (P95, P99),
sentiment distributions, and validates operational throughput against Assignment 2 NFR1
(>1,000 articles/min).
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict
import numpy as np

from kaz_fin_nlp.sentiment_analyzer import SentimentLabel, SentimentResult


@dataclass
class TelemetryMetrics:
    """Consolidated telemetry snapshot."""

    total_articles: int
    positive_articles: int
    neutral_articles: int
    negative_articles: int
    total_pii_redactions: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_articles_sec: float
    throughput_articles_min: float
    nfr1_target_met: bool


class StreamTelemetryTracker:
    """Thread-safe real-time performance telemetry tracker."""

    def __init__(self, window_size: int = 1000) -> None:
        """Initialize telemetry with sliding window capacity."""
        self._lock = threading.Lock()
        self._window_size = window_size
        self._latencies: deque[float] = deque(maxlen=window_size)
        self._sentiment_counts: Dict[SentimentLabel, int] = {
            SentimentLabel.POSITIVE: 0,
            SentimentLabel.NEUTRAL: 0,
            SentimentLabel.NEGATIVE: 0,
        }
        self._total_articles: int = 0
        self._total_pii_redacted: int = 0
        self._start_time: float = time.perf_counter()

    def record_document(
        self,
        sentiment: SentimentResult,
        latency_ms: float,
        pii_redacted_count: int = 0,
    ) -> None:
        """Record processed document telemetry."""
        with self._lock:
            self._total_articles += 1
            self._latencies.append(latency_ms)
            self._sentiment_counts[sentiment.label] = (
                self._sentiment_counts.get(sentiment.label, 0) + 1
            )
            self._total_pii_redacted += pii_redacted_count

    def get_metrics(self) -> TelemetryMetrics:
        """Calculate and return instantaneous telemetry metrics."""
        with self._lock:
            total = self._total_articles
            elapsed = max(0.001, time.perf_counter() - self._start_time)
            t_sec = total / elapsed
            t_min = t_sec * 60.0

            if len(self._latencies) > 0:
                lats = np.array(list(self._latencies))
                avg_lat = float(np.mean(lats))
                p95_lat = float(np.percentile(lats, 95))
                p99_lat = float(np.percentile(lats, 99))
            else:
                avg_lat = 0.0
                p95_lat = 0.0
                p99_lat = 0.0

            return TelemetryMetrics(
                total_articles=total,
                positive_articles=self._sentiment_counts[SentimentLabel.POSITIVE],
                neutral_articles=self._sentiment_counts[SentimentLabel.NEUTRAL],
                negative_articles=self._sentiment_counts[SentimentLabel.NEGATIVE],
                total_pii_redactions=self._total_pii_redacted,
                avg_latency_ms=round(avg_lat, 2),
                p95_latency_ms=round(p95_lat, 2),
                p99_latency_ms=round(p99_lat, 2),
                throughput_articles_sec=round(t_sec, 2),
                throughput_articles_min=round(t_min, 1),
                nfr1_target_met=t_min >= 1000.0,
            )

    def reset(self) -> None:
        """Reset internal metrics accumulator."""
        with self._lock:
            self._latencies.clear()
            self._sentiment_counts = {
                SentimentLabel.POSITIVE: 0,
                SentimentLabel.NEUTRAL: 0,
                SentimentLabel.NEGATIVE: 0,
            }
            self._total_articles = 0
            self._total_pii_redacted = 0
            self._start_time = time.perf_counter()
