# Student Name: jonatahn ahuche
# Student Index Number: 10022200183

from dataclasses import dataclass, asdict
from typing import List
import re

from src.data_loader import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    title: str
    text: str
    metadata: dict


def fixed_char_chunking(doc: Document, size: int = 800, overlap: int = 120) -> List[Chunk]:
    chunks: List[Chunk] = []
    start = 0
    idx = 0
    while start < len(doc.text):
        end = min(start + size, len(doc.text))
        chunk_text = doc.text[start:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}_fixed_{idx}",
                    doc_id=doc.doc_id,
                    source=doc.source,
                    title=doc.title,
                    text=chunk_text,
                    metadata={**doc.metadata, "strategy": "fixed", "start": start, "end": end},
                )
            )
            idx += 1
        start += max(1, size - overlap)
    return chunks


def sentence_chunking(doc: Document, target_words: int = 140, overlap_sentences: int = 1) -> List[Chunk]:
    sentences = re.split(r"(?<=[.!?])\s+", doc.text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks: List[Chunk] = []
    current: List[str] = []
    idx = 0

    for sentence in sentences:
        current.append(sentence)
        words = sum(len(s.split()) for s in current)
        if words >= target_words:
            chunk_text = " ".join(current).strip()
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}_sent_{idx}",
                    doc_id=doc.doc_id,
                    source=doc.source,
                    title=doc.title,
                    text=chunk_text,
                    metadata={**doc.metadata, "strategy": "sentence"},
                )
            )
            idx += 1
            current = current[-overlap_sentences:] if overlap_sentences > 0 else []

    if current:
        chunk_text = " ".join(current).strip()
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}_sent_{idx}",
                doc_id=doc.doc_id,
                source=doc.source,
                title=doc.title,
                text=chunk_text,
                metadata={**doc.metadata, "strategy": "sentence"},
            )
        )
    return chunks


def build_chunks(documents: List[Document], strategy: str = "sentence") -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for doc in documents:
        if strategy == "fixed":
            all_chunks.extend(fixed_char_chunking(doc))
        else:
            all_chunks.extend(sentence_chunking(doc))
    return all_chunks


def chunk_to_dict(chunk: Chunk) -> dict:
    return asdict(chunk)
