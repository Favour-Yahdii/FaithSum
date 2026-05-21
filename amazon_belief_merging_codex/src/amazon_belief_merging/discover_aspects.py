from __future__ import annotations

from collections import Counter
import json
import re
from pathlib import Path
import nltk
import gensim
from gensim.parsing.preprocessing import STOPWORDS as sw
from .pipeline import read_reviews, infer_column, TEXT_COLUMNS
from nltk.corpus import stopwords

nltk.download("stopwords")

common_words = {
    "about", "after", "again", "also", "amazon", "and", "are", "because", "been",
    "being", "bought", "but", "could", "does", "doesn", "for", "from", "had", "has",
    "have", "just", "like", "more", "much", "only", "order", "product", "really",
    "review", "than", "that", "the", "their", "there", "these", "thing", "this",
    "very", "was", "were", "when", "with", "would", "your",
}

updated_stopwords = sw.union(common_words)

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_'-]+")


def discover_candidate_aspects(input_path: str | Path, output_path: str | Path, *, text_col: str | None = None, top_n: int = 60) -> None:
    frame = read_reviews(input_path)
    text_col = text_col or infer_column(frame, TEXT_COLUMNS, "reviews")
    unigram_counts: Counter[str] = Counter()
    bigram_counts: Counter[str] = Counter()
    for text in frame[text_col].dropna().astype(str):
        tokens = [
            t.lower().strip("'")
            for t in TOKEN_RE.findall(text)
            if len(t) > 2 and t.lower() not in updated_stopwords
        ]
        unigram_counts.update(tokens)
        bigram_counts.update(" ".join(pair) for pair in zip(tokens, tokens[1:]))

    candidates = []
    for phrase, count in bigram_counts.most_common(top_n):
        candidates.append({"candidate": phrase, "count": count, "type": "bigram"})
    for phrase, count in unigram_counts.most_common(top_n):
        candidates.append({"candidate": phrase, "count": count, "type": "unigram"})

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(candidates[:top_n], indent=2), encoding="utf-8")
