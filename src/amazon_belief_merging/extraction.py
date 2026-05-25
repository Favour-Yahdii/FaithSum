from __future__ import annotations

from dataclasses import dataclass, field
import re

from .embeddings import Embedder, HashingEmbedder, build_embedder, cosine_matrix
from .schema import Aspect
from .sentiment import score_sentence

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
CLAUSE_RE = re.compile(r"\s+(?:but|although|though|however|while|whereas)\s+|,\s*(?:but|although|though|however|while|whereas)\s+")


@dataclass
class AspectOpinion:
    aspect: str
    score: float
    polarity: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ReviewBeliefBase:
    review_id: str
    product_id: str
    opinions: dict[str, AspectOpinion]
    rating: float | None = None


class AspectOpinionExtractor:
    def __init__(
        self,
        aspects: list[Aspect],
        embedder: Embedder | None = None,
        similarity_threshold: float = 0.15,
        top_k_aspects: int = 2,
    ):
        self.aspects = aspects
        self.embedder = embedder or build_embedder()
        self.similarity_threshold = similarity_threshold
        self.top_k_aspects = top_k_aspects
        self._aspect_vectors = self.embedder.encode([a.embedding_text for a in aspects])

    def extract(
        self,
        review_text: str,
        *,
        product_id: str,
        review_id: str,
        rating: float | None = None,
    ) -> ReviewBeliefBase:
        sentences = split_sentences(review_text)
        if not sentences:
            return ReviewBeliefBase(review_id=review_id, product_id=product_id, opinions={}, rating=rating)

        sentence_vectors = self.embedder.encode(sentences)
        sims = cosine_matrix(sentence_vectors, self._aspect_vectors)
        by_aspect: dict[str, list[tuple[str, float, float]]] = {}

        for sent_idx, sentence in enumerate(sentences):
            ranked = sorted(
                enumerate(sims[sent_idx].tolist()),
                key=lambda item: item[1],
                reverse=True,
            )[: self.top_k_aspects]
            sent_sentiment = score_sentence(sentence, rating=rating)
            for aspect_idx, sim in ranked:
                aspect = self.aspects[aspect_idx]
                keyword_bonus = keyword_overlap(sentence, aspect)
                confidence = max(float(sim), keyword_bonus)
                if isinstance(self.embedder, HashingEmbedder) and keyword_bonus <= 0 and confidence < 0.3:
                    continue
                if confidence < self.similarity_threshold and keyword_bonus <= 0:
                    continue
                by_aspect.setdefault(aspect.name, []).append((sentence, sent_sentiment.score, confidence))

        opinions: dict[str, AspectOpinion] = {}
        for aspect_name, rows in by_aspect.items():
            weighted_sum = sum(score * max(conf, 0.05) for _, score, conf in rows)
            weight = sum(max(conf, 0.05) for _, _, conf in rows)
            score = weighted_sum / weight if weight else 0.5
            polarity = "positive" if score >= 0.62 else "negative" if score <= 0.38 else "mixed"
            evidence = [row[0] for row in sorted(rows, key=lambda r: r[2], reverse=True)[:3]]
            opinions[aspect_name] = AspectOpinion(
                aspect=aspect_name,
                score=round(float(score), 4),
                polarity=polarity,
                evidence=evidence,
                confidence=round(float(min(1.0, weight / max(1, len(rows)))), 4),
            )

        return ReviewBeliefBase(review_id=review_id, product_id=product_id, opinions=opinions, rating=rating)


def split_sentences(text: str) -> list[str]:
    chunks: list[str] = []
    for sentence in SENTENCE_RE.split(str(text)):
        chunks.extend(CLAUSE_RE.split(sentence))
    return [s.strip(" ,;:") for s in chunks if len(s.strip(" ,;:")) > 2]


def keyword_overlap(sentence: str, aspect: Aspect) -> float:
    lowered = sentence.lower()
    hits = sum(1 for keyword in aspect.keywords if keyword.lower() in lowered)
    return min(1.0, hits / 3.0)
