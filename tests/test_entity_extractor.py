"""Unit tests for FinancialEntityExtractor."""

import pytest
from kaz_fin_nlp.entity_extractor import (
    EntityType,
    FinancialEntityExtractor,
)


@pytest.fixture
def extractor() -> FinancialEntityExtractor:
    return FinancialEntityExtractor()


def test_extract_kase_tickers(extractor: FinancialEntityExtractor) -> None:
    text = "Сегодня на KASE активно торговались акции HSBK, KSPI и KZTO."
    entities = extractor.extract_entities(text)
    tickers = [e.text for e in entities if e.entity_type == EntityType.TICKER]
    assert "HSBK" in tickers
    assert "KSPI" in tickers
    assert "KZTO" in tickers


def test_extract_banks_and_fintech(extractor: FinancialEntityExtractor) -> None:
    text = (
        "Kaspi.kz және Halyk Bank қызметтерді ұсынды, "
        "ал Банк ЦентрКредит несие көлемін арттырды."
    )
    entities = extractor.extract_entities(text)
    bank_entities = [
        e.normalized_value for e in entities if e.entity_type == EntityType.BANK_FINTECH
    ]
    assert "Kaspi.kz" in bank_entities
    assert "Halyk Bank" in bank_entities
    assert "Bank CenterCredit" in bank_entities


def test_extract_regulators(extractor: FinancialEntityExtractor) -> None:
    text = "Национальный Банк және АРРФР жаңа реттеу талаптарын енгізді."
    entities = extractor.extract_entities(text)
    regulators = [
        e.normalized_value for e in entities if e.entity_type == EntityType.REGULATOR
    ]
    assert "National Bank of Kazakhstan" in regulators
    assert "ARDFM" in regulators


def test_extract_monetary_values(extractor: FinancialEntityExtractor) -> None:
    text = (
        "Таза пайда 450 млрд теңге құрады, "
        "ал инвестиция көлемі 120 млн доллар болды."
    )
    entities = extractor.extract_entities(text)
    amounts = [e.text for e in entities if e.entity_type == EntityType.MONETARY_VALUE]
    assert any("450 млрд" in a for a in amounts)
    assert any("120 млн" in a for a in amounts)


def test_no_entities_text(extractor: FinancialEntityExtractor) -> None:
    text = "Бүгін ауа райы ашық болады деп күтілуде."
    entities = extractor.extract_entities(text)
    assert len(entities) == 0
