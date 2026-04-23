# Student Name: jonatahn ahuche
# Student Index Number: 10022200183

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from openai import OpenAI
from dotenv import load_dotenv

from src.config import AppConfig, get_llm_api_key
from src.data_loader import load_all_documents
from src.chunking import build_chunks
from src.embeddings import EmbeddingModel
from src.vector_store import FaissVectorStore
from src.retriever import HybridRetriever
from src.prompting import build_context, build_prompt


class RAGPipeline:
    def __init__(self, config: AppConfig):
        load_dotenv()
        self.config = config
        self.embedder = EmbeddingModel(config.embedding_model)
        self.client = OpenAI(api_key=get_llm_api_key(), base_url=config.llm_base_url)
        self.store: FaissVectorStore | None = None
        self.retriever: HybridRetriever | None = None
        Path(config.logs_dir).mkdir(parents=True, exist_ok=True)

    def build_or_load_index(self, strategy: str = "sentence") -> None:
        if (
            Path(self.config.index_path).exists()
            and Path(self.config.chunks_path).exists()
            and strategy == "sentence"
        ):
            self.store = FaissVectorStore.load(self.config.index_path, self.config.chunks_path)
        else:
            docs = load_all_documents(self.config.csv_path, self.config.pdf_path)
            chunks = build_chunks(docs, strategy=strategy)
            vectors = self.embedder.encode([c.text for c in chunks])
            store = FaissVectorStore(vectors.shape[1])
            store.add(vectors, chunks)
            store.save(self.config.index_path, self.config.chunks_path)
            self.store = store
        self.retriever = HybridRetriever(
            self.embedder,
            self.store,
            v_weight=self.config.vector_weight,
            k_weight=self.config.keyword_weight,
        )

    def ask(self, user_query: str, top_k: int | None = None) -> Dict:
        if self.retriever is None:
            self.build_or_load_index()
        k = top_k or self.config.top_k
        retrieved = self.retriever.retrieve(user_query, k=k, use_expansion=True)
        context = build_context(retrieved, self.config.max_context_chars)
        final_prompt = build_prompt(user_query, context)

        response = self.client.chat.completions.create(
            model=self.config.llm_model,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": "Follow the instructions in the user message exactly."},
                {"role": "user", "content": final_prompt},
            ],
        )
        answer = response.choices[0].message.content or ""

        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": user_query,
            "retrieved": [
                {
                    "chunk_id": r.chunk.chunk_id,
                    "source": r.chunk.source,
                    "final_score": r.final_score,
                    "vector_score": r.vector_score,
                    "keyword_score": r.keyword_score,
                    "text": r.chunk.text,
                }
                for r in retrieved
            ],
            "prompt": final_prompt,
            "response": answer,
        }
        self._log(payload)
        return payload

    def pure_llm_answer(self, user_query: str) -> str:
        response = self.client.chat.completions.create(
            model=self.config.llm_model,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": "Answer clearly and concisely."},
                {"role": "user", "content": user_query},
            ],
        )
        return response.choices[0].message.content or ""

    def _log(self, payload: Dict) -> None:
        log_path = Path(self.config.logs_dir) / "rag_pipeline_logs.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
