# ai_10022200183

Student Name: jonatahn ahuche  
Student Index Number: 10022200183

This project implements a **manual Retrieval-Augmented Generation (RAG)** chatbot for Academic City using:
- Ghana election results CSV dataset
- 2025 Ghana budget statement PDF

No end-to-end RAG framework (LangChain/LlamaIndex) is used. Core RAG components are manually implemented.

## 1) Features
- Custom data cleaning and preprocessing
- Two chunking strategies (`fixed`, `sentence`) with overlap logic
- Embedding pipeline with `sentence-transformers`
- Custom vector retrieval with `FAISS`
- Hybrid retrieval extension (vector + keyword score) + query expansion
- Prompt template with hallucination control
- Context window management
- Full RAG pipeline logging:
  - retrieved chunks
  - similarity scores
  - final prompt
- Adversarial testing + pure LLM comparison
- Streamlit UI

## 2) Project Structure
- `app.py` - Streamlit app
- `src/` - RAG components
- `scripts/download_data.py` - fetches the required datasets
- `scripts/run_experiments.py` - chunking comparison + adversarial reports
- `logs/` - experiment outputs and manual logs
- `docs/architecture.md` - architecture and design rationale
- `docs/submission_checklist.md` - GitHub/cloud/email checklist

## 3) Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py
```

Create a `.env` file or export:
```bash
set GROQ_API_KEY=your_key_here
```

## 4) Run the App
```bash
streamlit run app.py
```

## 5) Run Experiments
```bash
python scripts/run_experiments.py
```

Outputs:
- `logs/chunking_comparison.json`
- `logs/adversarial_report.json`
- `logs/rag_pipeline_logs.jsonl` (generated while chatting)

## 6) Part-by-Part Mapping to Exam Requirements
- **Part A**: `src/data_loader.py`, `src/chunking.py`, `logs/chunking_comparison.json`
- **Part B**: `src/embeddings.py`, `src/vector_store.py`, `src/retriever.py`
- **Part C**: `src/prompting.py`, prompt iterations in `logs/manual_experiment_log.md`
- **Part D**: `src/rag_pipeline.py`, `app.py`
- **Part E**: `src/evaluation.py`, `logs/adversarial_report.json`
- **Part F**: `docs/architecture.md`
- **Part G**: Hybrid retrieval with query expansion and weighted domain scoring

## 7) Required Submission Steps
1. Rename repository to `ai_10022200183`.
2. Replace placeholders with your real details in **all files**.
3. Push repository to GitHub.
4. Deploy app to cloud (Streamlit Community Cloud or Render).
5. Add collaborator: `godwin.danso@acity.edu.gh` or `GodwinDansoAcity`.
6. Send email with:
   - GitHub link
   - deployed app URL
   - any extra docs/video link
   - subject: `CS4241-Introduction to Artificial Intelligence-2026:[10022200183 jonatahn ahuche]`
