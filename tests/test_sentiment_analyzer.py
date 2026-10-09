"""Unit tests for FinancialSentimentAnalyzer."""

import pytest
from kaz_fin_nlp.sentiment_analyzer import (
    FinancialSentimentAnalyzer,
    MarketUrgency,
    SentimentLabel,
)


@pytest.fixture
def analyzer() -> FinancialSentimentAnalyzer:
    return FinancialSentimentAnalyzer()


def test_positive_sentiment_kazakh(analyzer: FinancialSentimentAnalyzer) -> None:
    tokens = ["kaspi", "таза", "пайда", "өсім", "көрсетті", "дивидент"]
    res = analyzer.analyze(tokens, raw_text="Kaspi пайда өсімін көрсетті")
    assert res.label == SentimentLabel.POSITIVE
    assert res.polarity > 0.1
    assert res.confidence >= 0.5
    assert "пайда" in res.positive_hits


def test_positive_sentiment_russian(analyzer: FinancialSentimentAnalyzer) -> None:
    tokens = ["халык", "банк", "рост", "чистая прибыль", "дивиденды", "укрепление"]
    res = analyzer.analyze(tokens, raw_text="Халык банк зафиксировал рост прибыли")
    assert res.label == SentimentLabel.POSITIVE
    assert res.polarity > 0.2
    assert "рост" in res.positive_hits


def test_negative_sentiment_kazakh(analyzer: FinancialSentimentAnalyzer) -> None:
    tokens = ["құлдырау", "залал", "шығын", "әлсіреу", "инфляция"]
    res = analyzer.analyze(tokens, raw_text="Нарықта құлдырау мен шығын байқалды")
    assert res.label == SentimentLabel.NEGATIVE
    assert res.polarity < -0.1
    assert "құлдырау" in res.negative_hits


def test_negative_sentiment_russian(analyzer: FinancialSentimentAnalyzer) -> None:
    tokens = ["падение", "убыток", "спад", "штраф"]
    res = analyzer.analyze(tokens, raw_text="Убыток и падение показателей банка")
    assert res.label == SentimentLabel.NEGATIVE
    assert res.polarity < -0.1
    assert "убыток" in res.negative_hits


def test_neutral_sentiment(analyzer: FinancialSentimentAnalyzer) -> None:
    tokens = ["астана", "қаржы", "орталығында", "кездесу", "өтті"]
    res = analyzer.analyze(tokens, raw_text="Астана қаржы орталығында кездесу өтті")
    assert res.label == SentimentLabel.NEUTRAL
    assert abs(res.polarity) <= 0.05


def test_critical_trigger_urgency(analyzer: FinancialSentimentAnalyzer) -> None:
    tokens = ["дефолт", "облигациялар", "бойынша", "тіркелді"]
    res = analyzer.analyze(tokens, raw_text="Облигациялар бойынша дефолт тіркелді")
    assert res.label == SentimentLabel.NEGATIVE
    assert res.urgency == MarketUrgency.CRITICAL
    assert len(res.critical_triggers) > 0


def test_negation_inversion(analyzer: FinancialSentimentAnalyzer) -> None:
    tokens_pos = ["өсім"]
    res_pos = analyzer.analyze(tokens_pos)

    tokens_neg = ["өсім", "болмады"]
    res_neg = analyzer.analyze(tokens_neg)

    assert res_pos.polarity > 0
    assert res_neg.polarity < res_pos.polarity


def test_booster_multiplier(analyzer: FinancialSentimentAnalyzer) -> None:
    tokens_normal = ["пайда"]
    res_normal = analyzer.analyze(tokens_normal)

    tokens_boosted = ["өте", "пайда"]
    res_boosted = analyzer.analyze(tokens_boosted)

    assert res_boosted.polarity >= res_normal.polarity


def test_empty_tokens(analyzer: FinancialSentimentAnalyzer) -> None:
    res = analyzer.analyze([])
    assert res.label == SentimentLabel.NEUTRAL
    assert res.polarity == 0.0
    assert res.confidence == 0.5
    assert res.urgency == MarketUrgency.LOW
