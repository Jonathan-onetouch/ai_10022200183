# Student Name: jonatahn ahuche
# Student Index Number: 10022200183

from typing import List, Tuple
import json
from pathlib import Path

import faiss
import numpy as np

from src.chunking import Chunk, chunk_to_dict


class FaissVectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[Chunk] = []

    def add(self, vectors: np.ndarray, chunks: List[Chunk]) -> None:
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[Chunk, float]]:
        scores, ids = self.index.search(query_vector, k)
        results: List[Tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, index_path: str, chunks_path: str) -> None:
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(chunks_path, "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk_to_dict(chunk), ensure_ascii=True) + "\n")

    @classmethod
    def load(cls, index_path: str, chunks_path: str) -> "FaissVectorStore":
        index = faiss.read_index(index_path)
        store = cls(index.d)
        store.index = index
        chunks: List[Chunk] = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                chunks.append(Chunk(**data))
        store.chunks = chunks
        return store
