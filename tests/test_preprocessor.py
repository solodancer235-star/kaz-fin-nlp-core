"""Unit tests for BilingualTextPreprocessor."""

import pytest
from kaz_fin_nlp.preprocessor import (
    BilingualTextPreprocessor,
    Language,
    PreprocessingConfig,
)


@pytest.fixture
def preprocessor() -> BilingualTextPreprocessor:
    return BilingualTextPreprocessor()


def test_language_detection_kazakh(preprocessor: BilingualTextPreprocessor) -> None:
    text = "Қазақстан қор биржасында сауда-саттық көлемі ұлғайып, өсім байқалды."
    assert preprocessor.detect_language(text) == Language.KAZAKH


def test_language_detection_russian(preprocessor: BilingualTextPreprocessor) -> None:
    text = "Народный банк Казахстана объявил о рекордных результатах за квартал."
    assert preprocessor.detect_language(text) == Language.RUSSIAN


def test_language_detection_empty_or_unknown(
    preprocessor: BilingualTextPreprocessor,
) -> None:
    assert preprocessor.detect_language("") == Language.UNKNOWN
    assert preprocessor.detect_language("12345 67890 !@#$%") == Language.UNKNOWN


def test_pii_sanitization_iin(preprocessor: BilingualTextPreprocessor) -> None:
    text = "Клиенттің ЖСН: 980512300456 расталды."
    cleaned, count = preprocessor.sanitize_pii(text)
    assert "[KZ-IIN-REDACTED]" in cleaned
    assert "980512300456" not in cleaned
    assert count == 1


def test_pii_sanitization_phone(preprocessor: BilingualTextPreprocessor) -> None:
    text = "Call center: +7 (701) 555-12-34 or 87051234567 for trading support."
    cleaned, count = preprocessor.sanitize_pii(text)
    assert "[KZ-PHONE-REDACTED]" in cleaned
    assert "+7 (701) 555-12-34" not in cleaned
    assert count >= 1


def test_pii_sanitization_iban(preprocessor: BilingualTextPreprocessor) -> None:
    text = "Recipient account: KZ86125KZT1234567890 at Halyk Bank."
    cleaned, count = preprocessor.sanitize_pii(text)
    assert "[KZ-IBAN-REDACTED]" in cleaned
    assert "KZ86125KZT1234567890" not in cleaned
    assert count == 1


def test_normalization_and_html_stripping(
    preprocessor: BilingualTextPreprocessor,
) -> None:
    html_text = "<p>Акциялардың <b>бағасы</b> өсті.</p>"
    norm = preprocessor.normalize_text(html_text)
    assert "<p>" not in norm
    assert "<b>" not in norm
    assert "акциялардың бағасы өсті" in norm


def test_kazakh_suffix_stripping(preprocessor: BilingualTextPreprocessor) -> None:
    word1 = "банктердің"
    stem1 = preprocessor.strip_kazakh_suffix(word1)
    assert stem1 == "банк"

    word2 = "акцияларды"
    stem2 = preprocessor.strip_kazakh_suffix(word2)
    assert stem2 == "акция"

    short_word = "күн"
    assert preprocessor.strip_kazakh_suffix(short_word) == "күн"


def test_tokenization_and_stopwords(preprocessor: BilingualTextPreprocessor) -> None:
    text = "Бұл өсім Kaspi үшін өте маңызды және тиімді болды."
    tokens = preprocessor.tokenize(text)
    assert "бұл" not in tokens
    assert "үшін" not in tokens
    assert "және" not in tokens
    assert "өсім" in tokens
    assert "kaspi" in tokens


def test_full_pipeline_process(preprocessor: BilingualTextPreprocessor) -> None:
    raw = (
        "<div>Kaspi.kz компаниясы қаржылық есепте таза пайданың өсімін көрсетті. "
        "ЖСН басшысы: 850415300123.</div>"
    )
    doc = preprocessor.process(raw)
    assert doc.detected_language == Language.KAZAKH
    assert doc.pii_redacted_count >= 1
    assert "redacted" in doc.cleaned_text.lower()
    assert len(doc.tokens) > 0
    assert len(doc.stems) > 0
    assert doc.metadata["token_count"] == len(doc.tokens)


def test_config_toggling() -> None:
    custom_cfg = PreprocessingConfig(
        lowercase=False,
        strip_html_tags=False,
        mask_pii=False,
        strip_kazakh_suffixes=False,
        remove_stopwords=False,
    )
    prep = BilingualTextPreprocessor(custom_cfg)
    raw = "<b>ЖСН: 123456789012 және Kaspi</b>"
    doc = prep.process(raw)
    assert "<b>" in doc.cleaned_text
    assert "123456789012" in doc.cleaned_text
    assert doc.pii_redacted_count == 0
