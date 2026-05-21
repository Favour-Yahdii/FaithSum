from __future__ import annotations

from collections import Counter
import hashlib
import math
import re
from typing import Iterable, Protocol

import numpy as np

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_'-]+")


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Small dependency-free fallback for lexical semantic matching.

    It is not a replacement for sentence transformers, but it gives a stable
    vector space for aspect routing when no embedding model is installed.
    """

    def __init__(self, dim: int = 2048):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            counts = Counter(tokenize(text))
            for token, count in counts.items():
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest, "little") % self.dim
                rows[i, idx] += 1.0 + math.log(count)
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return rows / norms


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)


def build_embedder(model_name: str | None = None) -> Embedder:
    if model_name:
        return SentenceTransformerEmbedder(model_name)
    return HashingEmbedder()


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T


def tokenize(text: str) -> Iterable[str]:
    for match in TOKEN_RE.finditer(text.lower()):
        token = match.group(0).strip("'")
        if len(token) > 2:
            yield token
