"""Command-Line Interface for Kazakhstani NLP Financial News System.

Provides commands for benchmarking throughput (>1,000 articles/min),
real-time text sentiment analysis, interactive news stream simulation,
and LLM incident report export.
"""

import argparse
import json
import sys
import time
from typing import List, Tuple

from kaz_fin_nlp import (
    BilingualTextPreprocessor,
    FinancialEntityExtractor,
    FinancialSentimentAnalyzer,
    MarketIncidentReportBuilder,
    MarketUrgency,
    StreamTelemetryTracker,
    __version__,
)

SAMPLE_ARTICLES: List[Tuple[str, str, str]] = [
    (
        "KASE: Kaspi.kz акциялары 12% өсім көрсетіп, тарихи максимумға жетті. "
        "Табыс 450 млрд теңге.",
        "KASE_WIRE",
        "kk",
    ),
    (
        "Народный банк Казахстана (HSBK) объявил о рекордной выплате "
        "дивидендов в размере 120 млрд тенге.",
        "KAPITAL_KZ",
        "ru",
    ),
    (
        "Ұлттық Банк базалық мөлшерлемені көтерді, инфляция мен "
        "теңгенің әлсіреу тәуекелі сақталуда.",
        "TENGRINEWS",
        "kk",
    ),
    (
        "КазТрансОйл (KZTO) увеличил чистую прибыль на 35% благодаря росту транзита.",
        "KASE_WIRE",
        "ru",
    ),
    (
        "АРРФР оштрафовало финансовую организацию за нарушение "
        "нормативов ликвидности на 15 млн тенге.",
        "KAPITAL_KZ",
        "ru",
    ),
    (
        "Қазақстан қор биржасында сауда-саттық көлемі ұлғайып, "
        "оң динамика байқалды.",
        "KASE_WIRE",
        "kk",
    ),
    (
        "Дефолт по облигациям частной строительной компании вызвал "
        "падение индекса на 4.2%.",
        "KAPITAL_KZ",
        "ru",
    ),
    (
        "Bank CenterCredit (CCBN) кеңейту бағдарламасын сәтті "
        "аяқтап, пайдасын еселеді.",
        "TENGRINEWS",
        "kk",
    ),
]


def run_benchmark(iterations: int = 200) -> int:
    """Benchmark computational module throughput and latency."""
    print("=" * 70)
    print(f"  KAZAKHSTANI FINANCIAL NLP CORE - BENCHMARK ({iterations} articles)")
    print("=" * 70)

    preprocessor = BilingualTextPreprocessor()
    analyzer = FinancialSentimentAnalyzer()
    extractor = FinancialEntityExtractor()
    telemetry = StreamTelemetryTracker()

    t_start = time.perf_counter()

    for i in range(iterations):
        sample_text, _, _ = SAMPLE_ARTICLES[i % len(SAMPLE_ARTICLES)]
        if i % 7 == 0:
            sample_text += (
                f" Байланыс: +7 (701) 555-12-{i % 90 + 10}, " f"ЖСН: 980512300456."
            )

        t0 = time.perf_counter()
        doc = preprocessor.process(sample_text)
        sentiment = analyzer.analyze(doc.stems, raw_text=doc.cleaned_text)
        _ = extractor.extract_entities(sample_text)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        telemetry.record_document(
            sentiment=sentiment,
            latency_ms=latency_ms,
            pii_redacted_count=doc.pii_redacted_count,
        )

    total_elapsed = time.perf_counter() - t_start
    metrics = telemetry.get_metrics()

    print(f"Execution completed in {total_elapsed:.3f} seconds.")
    print("-" * 70)
    print(f"  Articles Analyzed          : {metrics.total_articles}")
    print(f"  Throughput (Articles/Sec)  : {metrics.throughput_articles_sec}")
    print(f"  Throughput (Articles/Min)  : {metrics.throughput_articles_min:,}")
    flag = "[PASS]" if metrics.nfr1_target_met else "[FAIL]"
    print(f"  NFR1 Compliance (>1000/min): {flag}")
    print(f"  Average Latency            : {metrics.avg_latency_ms} ms")
    print(f"  P95 Latency                : {metrics.p95_latency_ms} ms")
    print(f"  P99 Latency                : {metrics.p99_latency_ms} ms")
    print(f"  PII Masked (Law No. 94-V)  : {metrics.total_pii_redactions} items")
    print("-" * 70)
    p_pct = (metrics.positive_articles / metrics.total_articles) * 100
    n_pct = (metrics.neutral_articles / metrics.total_articles) * 100
    neg_pct = (metrics.negative_articles / metrics.total_articles) * 100
    print(f"  Positive : {metrics.positive_articles} ({p_pct:.1f}%)")
    print(f"  Neutral  : {metrics.neutral_articles} ({n_pct:.1f}%)")
    print(f"  Negative : {metrics.negative_articles} ({neg_pct:.1f}%)")
    print("=" * 70)
    return 0


def run_analyze(text: str) -> int:
    """Analyze single news headline or document."""
    preprocessor = BilingualTextPreprocessor()
    analyzer = FinancialSentimentAnalyzer()
    extractor = FinancialEntityExtractor()
    builder = MarketIncidentReportBuilder()

    doc = preprocessor.process(text)
    sentiment = analyzer.analyze(doc.stems, raw_text=doc.cleaned_text)
    entities = extractor.extract_entities(text)
    report = builder.build_report(
        headline=text,
        source="CLI_INPUT",
        language=doc.detected_language.value,
        sentiment=sentiment,
        entities=entities,
    )

    print("\n--- Document Analysis Results ---")
    print(f"Detected Language   : {doc.detected_language.value.upper()}")
    print(f"PII Redactions      : {doc.pii_redacted_count}")
    print(f"Cleaned Text        : {doc.cleaned_text}")
    print(f"Sentiment Label     : {sentiment.label.value}")
    print(f"Polarity Score      : {sentiment.polarity}")
    print(f"Confidence          : {sentiment.confidence}")
    print(f"Market Urgency      : {sentiment.urgency.value}")
    print(f"Inference Latency   : {sentiment.inference_time_ms} ms")
    print("\nExtracted Entities:")
    for ent in entities:
        print(f"  - [{ent.entity_type.value}] {ent.text} -> {ent.normalized_value}")
    print("\nRecommended Actions:")
    for act in report.recommended_actions:
        print(f"  * {act}")
    return 0


def run_stream_demo(articles_count: int = 10, delay: float = 0.2) -> int:
    """Simulate real-time streaming ingestion with formatted HUD."""
    print("=" * 75)
    print("  KAZAKHSTANI NLP CORE - REAL-TIME STREAMING HUD")
    print("  Monitoring: KASE, Tengrinews, Kapital.kz feeds")
    print("=" * 75)

    preprocessor = BilingualTextPreprocessor()
    analyzer = FinancialSentimentAnalyzer()
    extractor = FinancialEntityExtractor()
    telemetry = StreamTelemetryTracker()

    for idx in range(articles_count):
        text, source, _ = SAMPLE_ARTICLES[idx % len(SAMPLE_ARTICLES)]
        t0 = time.perf_counter()
        doc = preprocessor.process(text)
        sentiment = analyzer.analyze(doc.stems, raw_text=doc.cleaned_text)
        _ = extractor.extract_entities(text)
        latency = (time.perf_counter() - t0) * 1000.0

        telemetry.record_document(sentiment, latency, doc.pii_redacted_count)

        urgency_badge = {
            MarketUrgency.CRITICAL: "[CRITICAL]",
            MarketUrgency.HIGH: "[HIGH    ]",
            MarketUrgency.MEDIUM: "[MEDIUM  ]",
            MarketUrgency.LOW: "[LOW     ]",
        }[sentiment.urgency]

        print(
            f"[{time.strftime('%H:%M:%S')}] {urgency_badge} | "
            f"{doc.detected_language.value.upper()} | "
            f"{sentiment.label.value:<8} (pol:{sentiment.polarity:+.2f}) | "
            f"lat:{latency:.1f}ms | {source:<10} | {text[:38]}..."
        )
        time.sleep(delay)

    print("-" * 75)
    final_m = telemetry.get_metrics()
    print(
        f"Stream simulation concluded. Latency: {final_m.avg_latency_ms} ms "
        f"| FPS: {final_m.throughput_articles_sec} articles/s"
    )
    return 0


def run_export_report(output_file: str) -> int:
    """Generate and write sample LLM incident prompt JSON."""
    preprocessor = BilingualTextPreprocessor()
    analyzer = FinancialSentimentAnalyzer()
    extractor = FinancialEntityExtractor()
    builder = MarketIncidentReportBuilder()

    headline = (
        "Дефолт эмитента и резкое падение акций на KASE: Kaspi.kz и Halyk Bank "
        "выделили 50 млрд тенге ликвидности для стабилизации рынка."
    )
    doc = preprocessor.process(headline)
    sentiment = analyzer.analyze(doc.stems, raw_text=doc.cleaned_text)
    entities = extractor.extract_entities(headline)
    report = builder.build_report(
        headline=headline,
        source="KASE_OFFICIAL_WIRE",
        language=doc.detected_language.value,
        sentiment=sentiment,
        entities=entities,
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
    print(f"Sample LLM incident report exported to: {output_file}")
    return 0


def main(argv: List[str] | None = None) -> int:
    """CLI Entrypoint."""
    parser = argparse.ArgumentParser(
        prog="kaz-fin-nlp",
        description="Kazakhstani NLP Financial News System - Core Engine CLI",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    bench_p = subparsers.add_parser(
        "benchmark", help="Run high-throughput processing benchmark"
    )
    bench_p.add_argument(
        "--iterations", type=int, default=200, help="Number of simulated news articles"
    )

    ana_p = subparsers.add_parser("analyze", help="Analyze individual text or article")
    ana_p.add_argument(
        "--text", type=str, required=True, help="Input news headline or text snippet"
    )

    stream_p = subparsers.add_parser(
        "stream-demo", help="Simulate live news wire streaming with HUD"
    )
    stream_p.add_argument(
        "--articles", type=int, default=12, help="Number of streaming articles"
    )
    stream_p.add_argument(
        "--delay", type=float, default=0.15, help="Delay between articles in seconds"
    )

    rep_p = subparsers.add_parser(
        "export-report", help="Export sample JSON incident report for LLM"
    )
    rep_p.add_argument(
        "--output",
        type=str,
        default="sample_market_report.json",
        help="Destination JSON path",
    )

    args = parser.parse_args(argv)

    if args.command == "benchmark":
        return run_benchmark(iterations=args.iterations)
    elif args.command == "analyze":
        return run_analyze(args.text)
    elif args.command == "stream-demo":
        return run_stream_demo(articles_count=args.articles, delay=args.delay)
    elif args.command == "export-report":
        return run_export_report(args.output)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
