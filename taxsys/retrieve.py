"""Retrieval over the cited corpus.

Hybrid lexical scoring with authority weighting and hard date scoping. The date
filter is not a ranking signal, it is a filter: a provision not in force for the
tax year asked about is never returned, however well it matches.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .corpus import Chunk, Corpus

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-]*")
_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "to", "in", "for",
    "on", "and", "or", "it", "this", "that", "with", "as", "at", "by", "i", "my", "me",
    "can", "do", "does", "if", "any", "from", "not", "you", "your", "claim", "deduct",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    matched: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.id,
            "citation": self.chunk.citation,
            "heading": self.chunk.heading,
            "kind": self.chunk.kind,
            "binding": self.chunk.is_binding_authority,
            "score": round(self.score, 3),
            "matched": self.matched,
            "text": self.chunk.text,
            "source_url": self.chunk.source_url,
            "effective_from": self.chunk.effective_from,
            "effective_to": self.chunk.effective_to,
            "span_hash": self.chunk.span_hash,
        }


class Retriever:
    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus
        self._docs = {c.id: Counter(tokenize(f"{c.heading} {c.citation} {c.text}")) for c in corpus.chunks}
        self._df: Counter[str] = Counter()
        for counts in self._docs.values():
            self._df.update(counts.keys())
        self._n = max(1, len(self._docs))
        self._avg_len = (sum(sum(c.values()) for c in self._docs.values()) / self._n) if self._docs else 1.0

    def search(
        self,
        query: str,
        *,
        jurisdiction: str,
        year_label: str,
        limit: int = 8,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> list[Hit]:
        terms = tokenize(query)
        if not terms:
            return []
        hits: list[Hit] = []
        for chunk in self.corpus.chunks:
            if chunk.jurisdiction != jurisdiction.upper():
                continue
            # Hard filter, never a ranking nudge. Out-of-force law is not an answer.
            if not chunk.in_force_for_year(jurisdiction, year_label):
                continue
            counts = self._docs[chunk.id]
            length = max(1, sum(counts.values()))
            score = 0.0
            matched: list[str] = []
            for term in terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                matched.append(term)
                idf = math.log(1 + (self._n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * length / self._avg_len))
            if score <= 0:
                continue
            # Authority weighting: a statute outranks a guide on equal match.
            score *= 1 + (chunk.authority_rank / 200)
            hits.append(Hit(chunk=chunk, score=score, matched=matched))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]
