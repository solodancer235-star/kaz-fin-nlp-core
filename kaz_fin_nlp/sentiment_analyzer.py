"""Bilingual Financial Sentiment Scoring Engine for Central Asian Markets.

Performs domain-specific polarity analysis, urgency rating, and market event
classification for Kazakhstani financial communications.
"""

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set


class SentimentLabel(str, Enum):
    """Categorical sentiment classification."""

    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"


class MarketUrgency(str, Enum):
    """Urgency level for automated trading and risk monitoring desks."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class SentimentConfig:
    """Configuration thresholds for financial sentiment analysis."""

    neutral_threshold: float = 0.05
    critical_threshold: float = 0.65
    high_threshold: float = 0.35
    medium_threshold: float = 0.15
    booster_multiplier: float = 1.6
    negation_window: int = 3


@dataclass
class SentimentResult:
    """Structured evaluation output of the sentiment analyzer."""

    label: SentimentLabel
    polarity: float
    confidence: float
    urgency: MarketUrgency
    positive_hits: List[str] = field(default_factory=list)
    negative_hits: List[str] = field(default_factory=list)
    critical_triggers: List[str] = field(default_factory=list)
    inference_time_ms: float = 0.0


class FinancialSentimentAnalyzer:
    """Specialized bilingual financial sentiment analyzer."""

    # Domain financial lexicon (unique terms and their directional weights)
    LEXICON: Dict[str, float] = {
        # --- Kazakh Positive ---
        "өсім": 1.2,
        "өсті": 1.2,
        "өсу": 1.1,
        "пайда": 1.5,
        "табыс": 1.4,
        "жоғарылау": 1.0,
        "нығаю": 1.3,
        "нығайды": 1.3,
        "дивидент": 1.2,
        "инвестиция": 1.1,
        "өтімділік": 1.0,
        "жетістік": 1.2,
        "кеңейту": 1.0,
        "қолдау": 0.9,
        "үстеме": 0.8,
        "сенім": 1.0,
        "рентабельді": 1.3,
        "көбею": 1.0,
        "жақсару": 1.2,
        "жақсарды": 1.2,
        # --- Kazakh Negative ---
        "құлдырау": -1.5,
        "төмендеу": -1.1,
        "төмендеді": -1.1,
        "залал": -1.6,
        "шығын": -1.4,
        "әлсіреу": -1.2,
        "әлсіреді": -1.2,
        "айыппұл": -1.3,
        "қарыз": -0.9,
        "тәуекел": -0.8,
        "банкроттық": -2.0,
        "тоқырау": -1.3,
        "қайтарылмаған": -1.2,
        "бұзушылық": -1.4,
        "дағдарыс": -1.6,
        # --- Russian Positive ---
        "рост": 1.2,
        "вырос": 1.2,
        "выросла": 1.2,
        "прибыль": 1.5,
        "доход": 1.4,
        "чистая прибыль": 1.8,
        "укрепление": 1.3,
        "укрепился": 1.3,
        "дивиденды": 1.2,
        "инвестиции": 1.1,
        "профицит": 1.4,
        "успех": 1.1,
        "расширение": 1.0,
        "поддержка": 0.9,
        "апсайд": 1.2,
        "повышение": 1.0,
        "повысил": 1.1,
        "рекордная": 1.3,
        "позитив": 1.0,
        "улучшение": 1.2,
        # --- Russian Negative & Shared ---
        "падение": -1.4,
        "упал": -1.3,
        "упала": -1.3,
        "снижение": -1.1,
        "снизился": -1.1,
        "убыток": -1.6,
        "чистый убыток": -1.8,
        "дефолт": -2.0,
        "девальвация": -1.7,
        "инфляция": -1.0,
        "штраф": -1.3,
        "санкция": -1.5,
        "санкции": -1.5,
        "кризис": -1.6,
        "банкротство": -2.0,
        "волатильность": -0.8,
        "долг": -0.9,
        "отзыв": -1.5,
        "обвал": -1.8,
        "претензия": -1.1,
        "стагнация": -1.2,
    }

    # High-severity trigger phrases that demand immediate market alerts
    CRITICAL_TRIGGERS: Set[str] = {
        "дефолт",
        "банкроттық",
        "банкротство",
        "девальвация",
        "отзыв лицензии",
        "лицензиясынан айыру",
        "обвал",
        "санкция",
        "санкции",
    }

    # Negation particles
    NEGATIONS: Set[str] = {"не", "нет", "без", "ни", "емес", "жоқ", "болмады"}

    # Boosters / Intensifiers
    BOOSTERS: Set[str] = {
        "өте",
        "күрт",
        "айтарлықтай",
        "елеулі",
        "аса",
        "очень",
        "резко",
        "значительно",
        "существенно",
        "крайне",
    }

    def __init__(self, config: SentimentConfig | None = None) -> None:
        """Initialize analyzer with specified sensitivity thresholds."""
        self.config = config or SentimentConfig()

    def analyze(self, tokens: List[str], raw_text: str = "") -> SentimentResult:
        """Calculate sentiment polarity, confidence, and market urgency."""
        t_start = time.perf_counter()

        score = 0.0
        positive_hits: List[str] = []
        negative_hits: List[str] = []
        critical_hits: List[str] = []

        lower_raw = raw_text.lower()
        for trig in self.CRITICAL_TRIGGERS:
            if trig in lower_raw:
                critical_hits.append(trig)

        n = len(tokens)
        for i, token in enumerate(tokens):
            if token in self.LEXICON:
                weight = self.LEXICON[token]

                multiplier = 1.0
                window_start = max(0, i - self.config.negation_window)
                preceding = tokens[window_start:i]
                following_window = tokens[i + 1 : min(n, i + 3)]

                if any(b in preceding for b in self.BOOSTERS):
                    multiplier *= self.config.booster_multiplier

                is_negated = any(neg in preceding for neg in self.NEGATIONS) or any(
                    neg in following_window for neg in {"емес", "жоқ", "болмады"}
                )

                if is_negated:
                    weight = -weight * 0.85

                effective_weight = weight * multiplier
                score += effective_weight

                if effective_weight > 0:
                    positive_hits.append(token)
                elif effective_weight < 0:
                    negative_hits.append(token)

        total_hits = len(positive_hits) + len(negative_hits)
        if total_hits > 0:
            polarity = math.tanh(score / max(1.5, math.sqrt(total_hits)))
        else:
            polarity = 0.0

        if total_hits == 0:
            confidence = 0.50
        else:
            scale = 1.0 - math.exp(-0.4 * total_hits)
            confidence = min(0.99, 0.55 + 0.44 * scale)

        if polarity > self.config.neutral_threshold:
            label = SentimentLabel.POSITIVE
        elif polarity < -self.config.neutral_threshold:
            label = SentimentLabel.NEGATIVE
        else:
            label = SentimentLabel.NEUTRAL

        abs_pol = abs(polarity)
        if len(critical_hits) > 0 or abs_pol >= self.config.critical_threshold:
            urgency = MarketUrgency.CRITICAL
        elif abs_pol >= self.config.high_threshold:
            urgency = MarketUrgency.HIGH
        elif abs_pol >= self.config.medium_threshold:
            urgency = MarketUrgency.MEDIUM
        else:
            urgency = MarketUrgency.LOW

        inference_time_ms = (time.perf_counter() - t_start) * 1000.0

        return SentimentResult(
            label=label,
            polarity=round(polarity, 4),
            confidence=round(confidence, 4),
            urgency=urgency,
            positive_hits=positive_hits,
            negative_hits=negative_hits,
            critical_triggers=critical_hits,
            inference_time_ms=round(inference_time_ms, 3),
        )
