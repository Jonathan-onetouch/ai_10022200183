# Student Name: jonatahn ahuche
# Student Index Number: 10022200183

from dataclasses import dataclass
from typing import Dict, List, Tuple
import re

from src.chunking import Chunk
from src.embeddings import EmbeddingModel
from src.vector_store import FaissVectorStore


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def _keyword_score(query: str, chunk_text: str) -> float:
    q = _tokenize(query)
    c = _tokenize(chunk_text)
    if not q or not c:
        return 0.0
    return len(q.intersection(c)) / len(q)


def expand_query(query: str) -> str:
    expansions = {
        "budget": "appropriation expenditure revenue fiscal policy",
        "election": "constituency parliamentary votes turnout results",
        "won": "winner highest votes total votes aggregate",
        "winner": "won highest votes total votes aggregate",
        "inflation": "price index cpi macroeconomic stability",
    }
    extended = [query]
    for key, addendum in expansions.items():
        if key in query.lower():
            extended.append(addendum)
    return " ".join(extended)


@dataclass
class RetrievalResult:
    chunk: Chunk
    vector_score: float
    keyword_score: float
    final_score: float


class HybridRetriever:
    def __init__(self, embedder: EmbeddingModel, store: FaissVectorStore, v_weight: float, k_weight: float):
        self.embedder = embedder
        self.store = store
        self.v_weight = v_weight
        self.k_weight = k_weight

    def retrieve(self, query: str, k: int = 5, use_expansion: bool = True) -> List[RetrievalResult]:
        q = expand_query(query) if use_expansion else query
        q_vector = self.embedder.encode([q])
        vector_hits = self.store.search(q_vector, k=max(k * 3, 10))
        ranked: List[RetrievalResult] = []
        for chunk, vector_score in vector_hits:
            k_score = _keyword_score(query, chunk.text)
            final = (self.v_weight * vector_score) + (self.k_weight * k_score)
            ranked.append(
                RetrievalResult(
                    chunk=chunk,
                    vector_score=vector_score,
                    keyword_score=k_score,
                    final_score=final,
                )
            )
        ranked.sort(key=lambda x: x.final_score, reverse=True)
        return ranked[:k]
