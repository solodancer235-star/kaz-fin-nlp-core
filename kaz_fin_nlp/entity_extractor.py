"""Financial Named Entity and Stock Ticker Extractor.

Extracts official KASE market tickers, leading Kazakhstani banking institutions,
financial regulatory authorities, currencies, and numeric monetary valuations.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Pattern


class EntityType(str, Enum):
    """Supported entity classification categories."""

    TICKER = "TICKER"
    BANK_FINTECH = "BANK_FINTECH"
    REGULATOR = "REGULATOR"
    CURRENCY = "CURRENCY"
    MONETARY_VALUE = "MONETARY_VALUE"


@dataclass
class FinancialEntity:
    """Container for parsed domain entities."""

    text: str
    entity_type: EntityType
    normalized_value: str
    start_char: int
    end_char: int
    confidence: float


class FinancialEntityExtractor:
    """Domain-specific entity recognition engine for Kazakh financial media."""

    KASE_TICKERS: Dict[str, str] = {
        "KZTO": "KazTransOil JSC",
        "HSBK": "Halyk Bank Kazakhstan",
        "KSPI": "Kaspi.kz JSC",
        "CCBN": "Bank CenterCredit JSC",
        "KMGZ": "NC KazMunayGas JSC",
        "KEGC": "KEGOC JSC",
        "KZAP": "NAC Kazatomprom JSC",
        "KZTK": "Kazakhtelecom JSC",
        "KCST": "Kcell JSC",
        "BAST": "BAST JSC",
    }

    BANK_ALIASES: Dict[str, str] = {
        "kaspi.kz": "Kaspi.kz",
        "kaspi bank": "Kaspi Bank",
        "каспи": "Kaspi.kz",
        "kaspi": "Kaspi.kz",
        "halyk bank": "Halyk Bank",
        "халық банк": "Halyk Bank",
        "народный банк": "Halyk Bank",
        "halyk": "Halyk Bank",
        "банк центркредит": "Bank CenterCredit",
        "центркредит": "Bank CenterCredit",
        "bcc": "Bank CenterCredit",
        "freedom finance": "Freedom Holding",
        "фридом финанс": "Freedom Holding",
        "freedom holding": "Freedom Holding",
        "jusan bank": "Jusan Bank",
        "жусан банк": "Jusan Bank",
        "jusan": "Jusan Bank",
        "fortebank": "ForteBank",
        "фортебанк": "ForteBank",
        "forte": "ForteBank",
        "евразийский банк": "Eurasian Bank",
        "bereke bank": "Bereke Bank",
        "береке банк": "Bereke Bank",
    }

    REGULATOR_ALIASES: Dict[str, str] = {
        "национальный банк": "National Bank of Kazakhstan",
        "ұлттық банк": "National Bank of Kazakhstan",
        "нбрк": "National Bank of Kazakhstan",
        "national bank": "National Bank of Kazakhstan",
        "аррфр": "ARDFM",
        "агентство по регулированию финансового рынка": "ARDFM",
        "kase": "Kazakhstan Stock Exchange",
        "қазақстан қор биржасы": "Kazakhstan Stock Exchange",
        "казахстанская фондовая биржа": "Kazakhstan Stock Exchange",
    }

    CURRENCY_PATTERN: Pattern = re.compile(
        r"\b(kzt|usd|eur|rub|теңге|тенге|доллар|долларов|евро|рубль|рублей|₸|\$|€|₽)\b",
        re.IGNORECASE,
    )

    MONETARY_PATTERN: Pattern = re.compile(
        r"(\d+(?:[\s,.]\d+)?)\s*(мың|млн|млрд|трлн|тыс\.?|млн\.?|млрд\.?|трлн\.)"
        r"?\s*(теңге|тенге|доллар(?:ов)?|евро|рубл(?:ей|я)|kzt|usd|eur|rub|₸|\$|€|₽)?",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        """Compile matching patterns."""
        ticker_regex = r"\b(" + "|".join(self.KASE_TICKERS.keys()) + r")\b"
        self.ticker_pattern = re.compile(ticker_regex)

    def extract_entities(self, text: str) -> List[FinancialEntity]:
        """Parse all financial entities from target text."""
        entities: List[FinancialEntity] = []
        lower_text = text.lower()

        # 1. KASE Tickers
        for match in self.ticker_pattern.finditer(text):
            symbol = match.group(1)
            entities.append(
                FinancialEntity(
                    text=symbol,
                    entity_type=EntityType.TICKER,
                    normalized_value=f"{symbol} ({self.KASE_TICKERS[symbol]})",
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=0.98,
                )
            )

        # 2. Bank & Fintech entities
        for alias, canon in self.BANK_ALIASES.items():
            pattern = re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)
            for match in pattern.finditer(lower_text):
                entities.append(
                    FinancialEntity(
                        text=text[match.start() : match.end()],
                        entity_type=EntityType.BANK_FINTECH,
                        normalized_value=canon,
                        start_char=match.start(),
                        end_char=match.end(),
                        confidence=0.95,
                    )
                )

        # 3. Regulators
        for alias, canon in self.REGULATOR_ALIASES.items():
            pattern = re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)
            for match in pattern.finditer(lower_text):
                entities.append(
                    FinancialEntity(
                        text=text[match.start() : match.end()],
                        entity_type=EntityType.REGULATOR,
                        normalized_value=canon,
                        start_char=match.start(),
                        end_char=match.end(),
                        confidence=0.96,
                    )
                )

        # 4. Monetary Quantities
        for match in self.MONETARY_PATTERN.finditer(text):
            raw_val = match.group(0).strip()
            num_part = match.group(1)
            scale_part = match.group(2) or ""
            curr_part = match.group(3) or ""

            if scale_part or curr_part:
                entities.append(
                    FinancialEntity(
                        text=raw_val,
                        entity_type=EntityType.MONETARY_VALUE,
                        normalized_value=f"{num_part} {scale_part} {curr_part}".strip(),
                        start_char=match.start(),
                        end_char=match.end(),
                        confidence=0.92,
                    )
                )

        return entities
