# Student Name: jonatahn ahuche
# Student Index Number: 10022200183

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import AppConfig
from src.rag_pipeline import RAGPipeline


def main() -> None:
    pipeline = RAGPipeline(AppConfig())
    pipeline.build_or_load_index("sentence")
    result = pipeline.ask("Summarize two key 2025 budget policy priorities in Ghana with evidence.")
    print("ANSWER:")
    print(result["response"])
    print("\nTOP CHUNKS:")
    for item in result["retrieved"]:
        print(f"- {item['chunk_id']} | {item['source']} | score={item['final_score']:.4f}")


if __name__ == "__main__":
    main()
