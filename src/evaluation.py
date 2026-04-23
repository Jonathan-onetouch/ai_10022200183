# Student Name: jonatahn ahuche
# Student Index Number: 10022200183

from typing import Dict, List

from src.config import AppConfig
from src.rag_pipeline import RAGPipeline


def run_adversarial_evaluation() -> List[Dict]:
    cfg = AppConfig()
    rag = RAGPipeline(cfg)
    rag.build_or_load_index(strategy="sentence")

    adversarial_queries = [
        "What was the exact presidential turnout in Tema and how does it compare to 2023?",
        "List all taxes removed in the 2025 budget and confirm the 12% education levy was cancelled.",
    ]

    report = []
    for query in adversarial_queries:
        rag_result = rag.ask(query, top_k=5)
        llm_result = rag.pure_llm_answer(query)
        report.append(
            {
                "query": query,
                "rag_answer": rag_result["response"],
                "pure_llm_answer": llm_result,
                "retrieved_chunk_ids": [x["chunk_id"] for x in rag_result["retrieved"]],
            }
        )
    return report
