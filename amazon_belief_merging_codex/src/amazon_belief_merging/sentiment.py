from __future__ import annotations

from dataclasses import dataclass
import re

POSITIVE = {
    "amazing", "awesome", "best", "comfortable", "durable", "easy", "excellent",
    "fast", "favorite", "fine", "good", "great", "happy", "impressed", "love",
    "loved", "nice", "perfect", "pleased", "quality", "recommend", "reliable",
    "quick", "quickly", "solid", "strong", "sturdy", "useful", "well", "wonderful",
    "works", "worth",
}
NEGATIVE = {
    "awful", "bad", "broke", "broken", "cheap", "complaint", "defect", "defective",
    "difficult", "disappointed", "expensive", "fail", "failed", "flimsy", "hate",
    "hated", "issue", "junk", "missing", "poor", "problem", "return", "returned",
    "slow", "stopped", "terrible", "uncomfortable", "useless", "waste", "weak",
    "worse", "worst",
}
NEGATORS = {"not", "never", "no", "hardly", "barely", "isn't", "wasn't", "doesn't", "didn't"}
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_'-]+")


@dataclass(frozen=True)
class SentimentResult:
    score: float
    label: str
    evidence_score: float


def score_sentence(text: str, rating: float | None = None) -> SentimentResult:
    tokens = [m.group(0).lower() for m in TOKEN_RE.finditer(text)]
    raw = 0.0
    hits = 0
    for idx, token in enumerate(tokens):
        polarity = 0
        if token in POSITIVE:
            polarity = 1
        elif token in NEGATIVE:
            polarity = -1
        if polarity:
            window = tokens[max(0, idx - 3):idx]
            if any(t in NEGATORS for t in window):
                polarity *= -1
            raw += polarity
            hits += 1

    if hits:
        lexical = 0.5 + 0.5 * max(-1.0, min(1.0, raw / max(2.0, hits)))
    else:
        lexical = 0.5

    if rating is not None:
        rating_score = max(0.0, min(1.0, (float(rating) - 1.0) / 4.0))
        score = 0.65 * lexical + 0.35 * rating_score if hits else rating_score
    else:
        score = lexical

    if score >= 0.62:
        label = "positive"
    elif score <= 0.38:
        label = "negative"
    else:
        label = "mixed"
    return SentimentResult(score=score, label=label, evidence_score=abs(score - 0.5) * 2)
