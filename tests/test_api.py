"""Integration tests for the FastAPI inference service."""

from fastapi.testclient import TestClient

from kaz_fin_nlp import __version__
from kaz_fin_nlp.api import app

client = TestClient(app)


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "version": __version__}


def test_analyze_russian_positive_with_ticker() -> None:
    r = client.post(
        "/analyze",
        json={
            "text": "КазТрансОйл (KZTO) увеличил чистую прибыль на 35%.",
            "source": "KASE_WIRE",
        },
    )
    assert r.status_code == 200
    body = r.json()
    report = body["report"]
    assert report["source"] == "KASE_WIRE"
    assert report["sentiment_label"] == "POSITIVE"
    assert any(t.startswith("KZTO") for t in report["detected_tickers"])
    assert report["regulatory"]["kz_law_94_v_compliant"] is True
    assert body["latency_ms"] >= 0


def test_analyze_redacts_pii_before_report() -> None:
    iin = "990101300123"
    r = client.post("/analyze", json={"text": f"Директор ИИН {iin} уволен"})
    assert r.status_code == 200
    body = r.json()
    assert body["pii_redacted_count"] >= 1
    assert iin not in body["report"]["headline"]
    assert iin not in body["report"]["llm_prompt_synthesis"]


def test_analyze_rejects_empty_text() -> None:
    assert client.post("/analyze", json={"text": ""}).status_code == 422


def test_metrics_after_requests() -> None:
    client.post("/analyze", json={"text": "Kaspi.kz табыс өсім көрсетті"})
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["total_articles"] >= 1
    assert "p95_latency_ms" in body
