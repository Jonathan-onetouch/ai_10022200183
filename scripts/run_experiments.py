# Student Name: jonatahn ahuche
# Student Index Number: 10022200183

import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import AppConfig
from src.data_loader import load_all_documents
from src.chunking import build_chunks
from src.embeddings import EmbeddingModel
from src.vector_store import FaissVectorStore
from src.retriever import HybridRetriever
from src.evaluation import run_adversarial_evaluation


def chunking_comparison_report() -> dict:
    cfg = AppConfig()
    docs = load_all_documents(cfg.csv_path, cfg.pdf_path)
    embedder = EmbeddingModel(cfg.embedding_model)
    queries = [
        "Which fiscal policy measures are highlighted in the 2025 budget?",
        "Which constituency had strong vote counts in the election dataset?",
    ]
    report = {}
    for strategy in ["fixed", "sentence"]:
        chunks = build_chunks(docs, strategy=strategy)
        vectors = embedder.encode([c.text for c in chunks])
        store = FaissVectorStore(vectors.shape[1])
        store.add(vectors, chunks)
        retriever = HybridRetriever(embedder, store, cfg.vector_weight, cfg.keyword_weight)

        samples = []
        for q in queries:
            top = retriever.retrieve(q, k=3)
            samples.append(
                {
                    "query": q,
                    "top_results": [
                        {"chunk_id": t.chunk.chunk_id, "score": t.final_score, "text_preview": t.chunk.text[:180]}
                        for t in top
                    ],
                }
            )
        report[strategy] = {"num_chunks": len(chunks), "sample_retrievals": samples}
    return report


def main() -> None:
    Path("logs").mkdir(exist_ok=True)
    chunk_report = chunking_comparison_report()
    if os.getenv("GROQ_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip():
        adv_report = run_adversarial_evaluation()
    else:
        adv_report = [
            {
                "note": "GROQ_API_KEY (or OPENAI_API_KEY) not set; adversarial RAG vs pure LLM evaluation was skipped."
            }
        ]

    with open("logs/chunking_comparison.json", "w", encoding="utf-8") as f:
        json.dump(chunk_report, f, indent=2, ensure_ascii=True)
    with open("logs/adversarial_report.json", "w", encoding="utf-8") as f:
        json.dump(adv_report, f, indent=2, ensure_ascii=True)

    print("Experiment logs generated in logs/.")


if __name__ == "__main__":
    main()
