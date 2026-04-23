# Student Name: jonatahn ahuche
# Student Index Number: 10022200183

from typing import List
from src.retriever import RetrievalResult


SYSTEM_PROMPT = (
    "You are the Academic City RAG assistant. Answer ONLY from the supplied context. "
    "If the context is insufficient, say: 'I do not have enough evidence from the provided sources.' "
    "Do not fabricate figures, dates, or names. "
    "If context indicates a dataset schema limitation (for example missing constituency-level fields), "
    "state that limitation explicitly."
)


def build_context(results: List[RetrievalResult], max_chars: int = 5000) -> str:
    blocks = []
    used = 0
    for r in sorted(results, key=lambda x: x.final_score, reverse=True):
        block = (
            f"[ChunkID: {r.chunk.chunk_id} | Source: {r.chunk.source} | Score: {r.final_score:.4f}]\n"
            f"{r.chunk.text}\n"
        )
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n".join(blocks)


def build_prompt(user_query: str, context: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"User question: {user_query}\n"
        "Return a concise, evidence-grounded answer and cite the chunk IDs you used."
    )
