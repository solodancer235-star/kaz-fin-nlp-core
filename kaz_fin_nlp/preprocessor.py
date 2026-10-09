"""Bilingual Preprocessor for Kazakh and Russian Financial News.

Handles character normalization, Kazakh Cyrillic glyph standardization,
Kazakh agglutinative morphological affix stripping, noise cleaning,
and PII sanitization in compliance with Law of RK No. 94-V.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Set, Tuple


class Language(str, Enum):
    """Supported language classifications."""

    KAZAKH = "kk"
    RUSSIAN = "ru"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class PreprocessingConfig:
    """Configuration parameters for bilingual text preprocessing."""

    lowercase: bool = True
    normalize_kazakh_letters: bool = True
    strip_html_tags: bool = True
    mask_pii: bool = True
    strip_kazakh_suffixes: bool = True
    remove_stopwords: bool = True
    min_stem_length: int = 3


@dataclass
class PreprocessedDocument:
    """Output container for preprocessed bilingual document."""

    raw_text: str
    cleaned_text: str
    tokens: List[str]
    stems: List[str]
    detected_language: Language
    pii_redacted_count: int
    metadata: dict = field(default_factory=dict)


class BilingualTextPreprocessor:
    """Robust bilingual text preprocessor for Central Asian financial news."""

    # Specific Kazakh Cyrillic characters
    KAZAKH_SPECIFIC_CHARS: Set[str] = {
        "ә",
        "ғ",
        "қ",
        "ң",
        "ө",
        "ұ",
        "ү",
        "һ",
        "і",
        "Ә",
        "Ғ",
        "Қ",
        "Ң",
        "Ө",
        "Ұ",
        "Ү",
        "Һ",
        "І",
    }

    # Kazakh agglutinative inflectional suffixes (longest to shortest)
    KAZAKH_SUFFIXES: List[str] = [
        "лардан",
        "лерден",
        "дардан",
        "дерден",
        "тардан",
        "терден",
        "лардың",
        "лердің",
        "дардың",
        "дердің",
        "тардың",
        "тердің",
        "лармен",
        "лермен",
        "дармен",
        "дермен",
        "тармен",
        "термен",
        "ларға",
        "лерге",
        "дарға",
        "дерге",
        "тарға",
        "терге",
        "ларды",
        "лерді",
        "дарды",
        "дерді",
        "тарды",
        "терді",
        "ларда",
        "лерде",
        "дарда",
        "дерде",
        "тарда",
        "терде",
        "лар",
        "лер",
        "дар",
        "дер",
        "тар",
        "тер",
        "ымыз",
        "іміз",
        "ңыз",
        "ңіз",
        "тары",
        "тері",
        "сы",
        "сі",
        "ым",
        "ім",
        "ың",
        "ің",
        "нан",
        "нен",
        "дан",
        "ден",
        "тан",
        "тен",
        "ның",
        "нің",
        "дың",
        "дің",
        "тың",
        "тің",
        "мен",
        "бен",
        "пен",
        "ған",
        "ген",
        "қан",
        "кен",
        "ға",
        "ге",
        "қа",
        "ке",
        "да",
        "де",
        "та",
        "те",
        "ны",
        "ні",
        "ды",
        "ді",
        "ты",
        "ті",
    ]

    # Common financial stopwords in Kazakh & Russian
    STOPWORDS: Set[str] = {
        "и",
        "в",
        "во",
        "не",
        "что",
        "он",
        "на",
        "я",
        "с",
        "со",
        "как",
        "а",
        "то",
        "все",
        "она",
        "так",
        "его",
        "но",
        "да",
        "ты",
        "к",
        "у",
        "же",
        "вы",
        "за",
        "бы",
        "по",
        "только",
        "ее",
        "мне",
        "было",
        "вот",
        "от",
        "меня",
        "еще",
        "о",
        "из",
        "ему",
        "теперь",
        "когда",
        "даже",
        "вдруг",
        "ли",
        "если",
        "уже",
        "или",
        "ни",
        "быть",
        "был",
        "него",
        "до",
        "вас",
        "нибудь",
        "опять",
        "және",
        "мен",
        "бен",
        "пен",
        "де",
        "да",
        "те",
        "та",
        "үшін",
        "туралы",
        "бойынша",
        "арқылы",
        "дейін",
        "кейін",
        "бірақ",
        "немесе",
        "яғни",
        "осы",
        "бұл",
        "сол",
        "бір",
        "ең",
        "тек",
        "ғана",
        "қана",
        "болды",
        "болады",
        "деп",
        "үйде",
        "болып",
        "бар",
        "жоқ",
        "етеді",
        "етті",
    }

    # Kazakh letter normalization map for damaged chars
    KAZAKH_CHAR_MAP = {
        "i": "і",
        "I": "І",
        "h": "һ",
        "H": "Һ",
    }

    # Regex patterns for Law No. 94-V PII redaction
    IIN_REGEX = re.compile(r"\b\d{12}\b")
    PHONE_REGEX = re.compile(
        r"(?:\+?7|8)[\s\-\.]?\(?[7][0-7][0-9]\)?[\s\-\.]?[0-9]{3}[\s\-\.]?"
        r"[0-9]{2}[\s\-\.]?[0-9]{2}"
    )
    IBAN_REGEX = re.compile(r"\bKZ[0-9]{2}[A-Z0-9]{16}\b", re.IGNORECASE)

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        """Initialize preprocessor with given configuration."""
        self.config = config or PreprocessingConfig()

    def detect_language(self, text: str) -> Language:
        """Classify document language as Kazakh, Russian, or Mixed."""
        if not text or not text.strip():
            return Language.UNKNOWN

        kazakh_glyph_hits = sum(1 for ch in text if ch in self.KAZAKH_SPECIFIC_CHARS)
        cyrillic_chars = len(re.findall(r"[\u0400-\u04FF]", text))

        if cyrillic_chars == 0:
            return Language.UNKNOWN

        lower_tokens = set(re.findall(r"\b[а-яёәғқңөұүһі]{2,}\b", text.lower()))
        kazakh_word_markers = {
            "және",
            "туралы",
            "бойынша",
            "теңге",
            "қаржы",
            "өсім",
            "үкімет",
            "нарық",
        }
        marker_hits = len(lower_tokens.intersection(kazakh_word_markers))

        if kazakh_glyph_hits > 0 or marker_hits >= 2:
            ratio = kazakh_glyph_hits / cyrillic_chars
            if cyrillic_chars > 30 and ratio < 0.01 and marker_hits == 0:
                return Language.MIXED
            return Language.KAZAKH
        return Language.RUSSIAN

    def sanitize_pii(self, text: str) -> Tuple[str, int]:
        """Redact sensitive Personal Identifiable Information (KZ Law No. 94-V)."""
        count = 0
        redacted = text

        redacted, iban_subs = self.IBAN_REGEX.subn("[KZ-IBAN-REDACTED]", redacted)
        count += iban_subs

        redacted, phone_subs = self.PHONE_REGEX.subn("[KZ-PHONE-REDACTED]", redacted)
        count += phone_subs

        redacted, iin_subs = self.IIN_REGEX.subn("[KZ-IIN-REDACTED]", redacted)
        count += iin_subs

        return redacted, count

    def normalize_text(self, text: str) -> str:
        """Standardize letterforms, strip HTML tags, and clean whitespace."""
        res = text

        if self.config.strip_html_tags:
            res = re.sub(r"<[^>]+>", " ", res)

        if self.config.normalize_kazakh_letters:
            for bad, good in self.KAZAKH_CHAR_MAP.items():
                res = res.replace(bad, good)

        if self.config.lowercase:
            res = res.lower()

        res = re.sub(r"\s+", " ", res).strip()
        return res

    def strip_kazakh_suffix(self, token: str) -> str:
        """Strip agglutinative grammatical suffixes while preserving word root."""
        if len(token) <= self.config.min_stem_length:
            return token

        stem = token
        for suffix in self.KAZAKH_SUFFIXES:
            if stem.endswith(suffix):
                candidate = stem[: -len(suffix)]
                if len(candidate) >= self.config.min_stem_length:
                    stem = candidate
                    break
        return stem

    def tokenize(self, text: str) -> List[str]:
        """Extract alphanumeric word tokens supporting Kazakh and Russian Cyrillic."""
        tokens = re.findall(r"[а-яёәғқңөұүһіa-z0-9\-]+", text.lower())
        valid_tokens = [t.strip("-") for t in tokens if len(t.strip("-")) > 0]
        if self.config.remove_stopwords:
            valid_tokens = [t for t in valid_tokens if t not in self.STOPWORDS]
        return valid_tokens

    def process(self, text: str) -> PreprocessedDocument:
        """Execute full preprocessing pipeline on input financial text."""
        pii_count = 0
        working_text = text

        if self.config.mask_pii:
            working_text, pii_count = self.sanitize_pii(working_text)

        cleaned = self.normalize_text(working_text)
        lang = self.detect_language(text)
        tokens = self.tokenize(cleaned)

        stems: List[str] = []
        if self.config.strip_kazakh_suffixes:
            for tok in tokens:
                stems.append(self.strip_kazakh_suffix(tok))
        else:
            stems = list(tokens)

        return PreprocessedDocument(
            raw_text=text,
            cleaned_text=cleaned,
            tokens=tokens,
            stems=stems,
            detected_language=lang,
            pii_redacted_count=pii_count,
            metadata={"token_count": len(tokens), "stem_count": len(stems)},
        )
