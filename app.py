# Student Name: jonatahn ahuche
# Student Index Number: 10022200183

import streamlit as st
import json
from datetime import date
from pathlib import Path
from uuid import uuid4

from src.config import AppConfig
from src.rag_pipeline import RAGPipeline


st.set_page_config(page_title="Academic City RAG Assistant", layout="wide")
MANUAL_LOGS_PATH = Path("logs/manual_experiment_entries.json")


def load_manual_logs() -> list[dict]:
    if not MANUAL_LOGS_PATH.exists():
        return []
    with open(MANUAL_LOGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manual_logs(entries: list[dict]) -> None:
    MANUAL_LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANUAL_LOGS_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=True)


st.markdown(
    """
<style>
    :root {
        --primary: #0D2B52;
        --primary-2: #0A2342;
        --bg: #F3F5F9;
        --card: #FFFFFF;
        --text: #1F2937;
        --accent: #2563EB;
        --muted: #6B7280;
        --line: #E5E7EB;
    }
    .stApp { background: var(--bg); color: var(--text); }
    /* Sidebar: force dark chrome so Streamlit defaults do not wash out text/buttons */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #050d1a 0%, #0D2B52 55%, #0a1f3d 100%) !important;
    }
    [data-testid="stSidebar"] .block-container {
        background: transparent !important;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        background: transparent !important;
        gap: 0.35rem !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] li {
        color: #E8EEF7 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #F8FAFC !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(148, 163, 184, 0.35) !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        background-color: rgba(15, 23, 42, 0.92) !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(148, 163, 184, 0.5) !important;
        font-weight: 600;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: rgba(37, 99, 235, 0.45) !important;
        border-color: #93C5FD !important;
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] .stButton > button:focus {
        box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.55) !important;
    }
    .logo-title { font-size: 1.55rem; font-weight: 700; margin-bottom: 0.1rem; }
    .logo-sub { font-size: 0.8rem; color: #B6C9DB; margin-bottom: 1rem; }
    .section-chip {
        background: rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 0.58rem 0.75rem;
        margin-bottom: 0.4rem; font-size: 0.92rem;
    }
    .section-chip.active { background: #EEF2FF; color: #0D2B52 !important; }
    .topbar {
        background: var(--card); border-radius: 12px; padding: 0.95rem 1.05rem; box-shadow: 0 3px 10px rgba(15, 23, 42, 0.06);
        margin-bottom: 0.65rem; border: 1px solid #ECEFF5;
    }
    .top-title { font-size: 1.35rem; font-weight: 700; margin-bottom: 0.25rem; }
    .dataset-pill {
        display: inline-block; background: #EEF2FF; color: #1E3A8A; border: 1px solid #DCE6FF; border-radius: 999px;
        padding: 0.2rem 0.5rem; margin-right: 0.35rem; font-size: 0.75rem;
    }
    .online-dot {
        display: inline-block; width: 8px; height: 8px; border-radius: 999px; background: #22C55E; margin-right: 6px;
    }
    .chat-wrap { height: 63vh; overflow-y: auto; padding-right: 0.2rem; }
    .msg-user, .msg-bot {
        border-radius: 12px; padding: 0.85rem 0.95rem; margin: 0.52rem 0; width: fit-content; max-width: 92%;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
    }
    .msg-user { margin-left: auto; background: #E6F0FF; border: 1px solid #CFE0FF; }
    .msg-bot { margin-right: auto; background: var(--card); border: 1px solid var(--line); }
    .panel {
        background: var(--card); border-radius: 12px; padding: 0.85rem; box-shadow: 0 3px 10px rgba(15, 23, 42, 0.06);
        margin-bottom: 0.7rem; border: 1px solid #ECEFF5;
    }
    .panel h4 { margin: 0 0 0.52rem 0; font-size: 0.88rem; letter-spacing: 0.3px; }
    .ctx-item { border: 1px solid var(--line); border-radius: 10px; padding: 0.5rem; margin-bottom: 0.45rem; background: #FBFCFF; }
    .small-muted { color: var(--muted); font-size: 0.8rem; }
    .source-list { border-top: 1px solid var(--line); margin-top: 0.7rem; padding-top: 0.55rem; }
    .profile {
        background: rgba(255,255,255,0.12); border-radius: 12px; padding: 0.72rem; font-size: 0.85rem;
    }
    .recent-title {
        margin-top: 0.6rem; margin-bottom: 0.4rem; font-size: 0.78rem; color: #A9BDD7;
        letter-spacing: 0.7px; text-transform: uppercase;
    }
    div[data-testid="stChatInput"] {
        position: sticky;
        bottom: 0.4rem;
        background: transparent;
        padding-top: 0.35rem;
    }
    div[data-testid="stChatInput"] textarea {
        border-radius: 14px !important;
        border: 1px solid #D6DEEA !important;
        min-height: 52px !important;
        background: #FFFFFF !important;
    }
    .footer-note { text-align: center; color: #9AA4B2; font-size: 0.74rem; margin-top: 0.35rem; }
    .avatar-bot {
        width: 36px; height: 36px; border-radius: 999px; background: #0D2B52; color: #fff;
        display: inline-flex; align-items: center; justify-content: center; font-size: 0.95rem; font-weight: 700;
    }
</style>
""",
    unsafe_allow_html=True,
)

if "pipeline" not in st.session_state:
    cfg = AppConfig()
    try:
        st.session_state.pipeline = RAGPipeline(cfg)
        st.session_state.pipeline.build_or_load_index(strategy="sentence")
        st.session_state.pipeline_error = None
    except ValueError as e:
        # Do not crash app when cloud secret is missing; allow manual logs page to remain usable.
        st.session_state.pipeline = None
        st.session_state.pipeline_error = str(e)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_result" not in st.session_state:
    st.session_state.active_result = None
if "active_page" not in st.session_state:
    st.session_state.active_page = "chat"
if "manual_logs" not in st.session_state:
    st.session_state.manual_logs = load_manual_logs()
if "show_add_log_form" not in st.session_state:
    st.session_state.show_add_log_form = False
if "selected_log_id" not in st.session_state:
    st.session_state.selected_log_id = None

# Sidebar
with st.sidebar:
    st.markdown('<div class="logo-title">ACity RAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="logo-sub">Academic City AI Assistant</div>', unsafe_allow_html=True)
    if st.button("Chat", use_container_width=True, key="nav_chat"):
        st.session_state.active_page = "chat"
    if st.button("+  New Chat", use_container_width=True, key="nav_new_chat"):
        st.session_state.active_page = "chat"
        st.session_state.messages = []
        st.session_state.active_result = None
    st.markdown("### Chat History")
    if st.session_state.messages:
        recent_queries = [m["content"] for m in st.session_state.messages if m["role"] == "user"][-6:]
        for q in recent_queries[::-1]:
            st.markdown(f'<div class="section-chip active">{q[:40]}{"..." if len(q) > 40 else ""}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-chip active">No previous chats yet</div>', unsafe_allow_html=True)
    st.markdown("### Navigate")
    if st.button("Datasets", use_container_width=True, key="nav_datasets"):
        st.session_state.active_page = "datasets"
    if st.button("Logs & Experiments", use_container_width=True, key="nav_logs"):
        st.session_state.active_page = "logs"
    if st.button("Evaluation Results", use_container_width=True, key="nav_evaluation"):
        st.session_state.active_page = "evaluation"
    if st.button("Settings", use_container_width=True, key="nav_settings"):
        st.session_state.active_page = "settings"
    st.markdown('<div class="recent-title">Recent Chats</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-chip">2025 Budget Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-chip">Ghana Election 2024 Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-chip">What is electoral commission?</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        '<div class="profile"><b>Student</b><br/>Name: jonatahn ahuche<br/>Index: 10022200183</div>',
        unsafe_allow_html=True,
    )

if st.session_state.active_page == "logs":
    st.markdown(
        """
<div class="topbar">
    <div class="top-title">Manual Logs & Experiments</div>
    <span class="small-muted">Digital laboratory notebook (manual-entry only)</span>
</div>
""",
        unsafe_allow_html=True,
    )

    left, right = st.columns([2, 1], gap="large")
    with left:
        if st.button("+ Add New Experiment Log"):
            st.session_state.show_add_log_form = not st.session_state.show_add_log_form

        if st.session_state.show_add_log_form:
            with st.form("manual_experiment_form", clear_on_submit=True):
                title = st.text_input("Experiment Title")
                log_date = st.date_input("Date", value=date.today())
                dataset = st.text_input("Dataset Used")
                objective = st.text_area("Objective (manual input)", height=90)
                steps = st.text_area("Steps Performed (manual bullet points or textarea)", height=120)
                observations = st.text_area("Observations", height=120)
                conclusion = st.text_area("Conclusion", height=90)
                status = st.selectbox("Status", ["Draft", "Completed"])
                saved = st.form_submit_button("Save Manual Log")
                if saved:
                    if not title.strip():
                        st.error("Experiment Title is required.")
                    else:
                        new_entry = {
                            "id": str(uuid4()),
                            "title": title.strip(),
                            "date": str(log_date),
                            "dataset_used": dataset.strip(),
                            "objective": objective,
                            "steps_performed": steps,
                            "observations": observations,
                            "conclusion": conclusion,
                            "status": status,
                        }
                        st.session_state.manual_logs = [new_entry] + st.session_state.manual_logs
                        save_manual_logs(st.session_state.manual_logs)
                        st.success("Manual experiment log saved.")
                        st.session_state.show_add_log_form = False
                        st.rerun()

        st.markdown("### Saved Experiment Logs")
        if not st.session_state.manual_logs:
            st.info("No manual experiment logs saved yet.")
        else:
            for entry in st.session_state.manual_logs:
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown(
                        f"""
<div class="panel">
  <div><b>{entry['title']}</b></div>
  <div class="small-muted">Date: {entry['date']}  |  Status: {entry['status']}</div>
</div>
""",
                        unsafe_allow_html=True,
                    )
                with c2:
                    if st.button("View", key=f"view_{entry['id']}"):
                        st.session_state.selected_log_id = entry["id"]
                        st.rerun()

    with right:
        selected = None
        if st.session_state.selected_log_id:
            selected = next(
                (x for x in st.session_state.manual_logs if x["id"] == st.session_state.selected_log_id),
                None,
            )
        st.markdown('<div class="panel"><h4>VIEW LOG</h4>', unsafe_allow_html=True)
        if selected:
            st.markdown(f"**Experiment Title**\n\n{selected['title']}")
            st.markdown(f"**Date**\n\n{selected['date']}")
            st.markdown(f"**Status**\n\n{selected['status']}")
            st.markdown(f"**Dataset Used**\n\n{selected['dataset_used']}")
            st.markdown(f"**Objective**\n\n{selected['objective']}")
            st.markdown(f"**Steps Performed**\n\n{selected['steps_performed']}")
            st.markdown(f"**Observations**\n\n{selected['observations']}")
            st.markdown(f"**Conclusion**\n\n{selected['conclusion']}")
            st.caption("Displayed exactly as manually entered. No AI rewriting.")
        else:
            st.markdown('<div class="small-muted">Select a saved log to view its exact content.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.active_page == "datasets":
    cfg_ds = AppConfig()
    st.markdown(
        """
<div class="topbar">
    <div class="top-title">Datasets</div>
    <span class="small-muted">Sources used to build the retrieval index</span>
</div>
""",
        unsafe_allow_html=True,
    )
    csv_p = Path(cfg_ds.csv_path)
    pdf_p = Path(cfg_ds.pdf_path)
    st.markdown("### Files on disk")
    st.markdown(
        "| File | Path | Present |\n|------|------|--------|\n"
        f"| Election results (CSV) | `{cfg_ds.csv_path}` | **{'Yes' if csv_p.is_file() else 'No'}** |\n"
        f"| 2025 budget (PDF) | `{cfg_ds.pdf_path}` | **{'Yes' if pdf_p.is_file() else 'No'}** |\n"
    )
    st.markdown("### What each dataset is for")
    st.write(
        "- **Ghana_Election_Result.csv** — constituency-level election results used to answer questions about votes, parties, and regions.\n"
        "- **2025-Budget-Statement-and-Economic-Policy_v4.pdf** — national budget text used for fiscal policy, revenue, and macro targets."
    )
    st.info(
        "If files are missing locally or on Streamlit Cloud, run `python scripts/download_data.py` "
        "or upload the data folder per your deployment docs."
    )

elif st.session_state.active_page == "evaluation":
    st.markdown(
        """
<div class="topbar">
    <div class="top-title">Evaluation Results</div>
    <span class="small-muted">Outputs from `scripts/run_experiments.py`</span>
</div>
""",
        unsafe_allow_html=True,
    )
    adv_path = Path("logs/adversarial_report.json")
    chunk_path = Path("logs/chunking_comparison.json")
    if adv_path.is_file():
        st.markdown("### Adversarial report")
        with open(adv_path, encoding="utf-8") as f:
            adv_data = json.load(f)
        st.json(adv_data)
    else:
        st.warning(f"No file at `{adv_path}`. Run experiments to generate it.")
    if chunk_path.is_file():
        st.markdown("### Chunking comparison")
        with open(chunk_path, encoding="utf-8") as f:
            chunk_data = json.load(f)
        st.json(chunk_data)
    else:
        st.warning(f"No file at `{chunk_path}`. Run experiments to generate it.")

elif st.session_state.active_page == "settings":
    cfg_st = AppConfig()
    st.markdown(
        """
<div class="topbar">
    <div class="top-title">Settings</div>
    <span class="small-muted">Runtime configuration (read-only)</span>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("### Models & retrieval")
    st.write(
        f"- **Embedding model:** `{cfg_st.embedding_model}`\n"
        f"- **LLM model:** `{cfg_st.llm_model}`\n"
        f"- **LLM base URL:** `{cfg_st.llm_base_url}`\n"
        f"- **Default top_k:** {cfg_st.top_k}\n"
        f"- **Hybrid weights:** vector {cfg_st.vector_weight}, keyword {cfg_st.keyword_weight}\n"
        f"- **Max context chars:** {cfg_st.max_context_chars}\n"
        f"- **Temperature:** {cfg_st.temperature}\n"
    )
    st.markdown("### Paths")
    st.code(
        f"data_dir: {cfg_st.data_dir}\ncsv: {cfg_st.csv_path}\npdf: {cfg_st.pdf_path}\n"
        f"index: {cfg_st.index_path}\nlogs: {cfg_st.logs_dir}",
        language="text",
    )
    if st.session_state.get("pipeline_error"):
        st.error(
            "LLM API key is not set in the environment. Set `GROQ_API_KEY` or `OPENAI_API_KEY` "
            "in Streamlit **Secrets** (or your `.env`) so chat works."
        )
    else:
        st.success("Pipeline initialized; LLM key is available in this session.")

else:
    # Top bar (main chat)
    top_left, top_right = st.columns([4.5, 1.5])
    with top_left:
        st.markdown(
            """
<div class="topbar">
    <div class="top-title">Academic City RAG Assistant</div>
    <span class="small-muted">Using datasets:</span>
    <span class="dataset-pill">Ghana_Election_Result.csv</span>
    <span class="dataset-pill">2025-Budget-Statement-and-Economic-Policy_v4.pdf</span>
    <span class="small-muted" style="margin-left: 0.7rem;"><span class="online-dot"></span>Online</span>
</div>
""",
            unsafe_allow_html=True,
        )
    with top_right:
        st.button("About Project", use_container_width=True)
        st.button("Theme", use_container_width=True)

    main_col, side_col = st.columns([2.15, 1], gap="large")

    with main_col:
        st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                response = msg["content"]
                st.markdown('<div style="display:flex; gap:8px; align-items:flex-start;"><div class="avatar-bot">AI</div><div class="msg-bot" style="flex:1;">', unsafe_allow_html=True)
                st.markdown(response)
                if msg.get("sources"):
                    st.markdown('<div class="source-list"><b>Sources (Top K Retrieved)</b></div>', unsafe_allow_html=True)
                    for s in msg["sources"]:
                        st.markdown(
                            f"- `{s['source']}` | `{s['chunk_id']}` | score `{s['final_score']:.2f}`"
                        )
                if msg.get("meta"):
                    st.markdown(
                        f"<div class='small-muted'>Model: {msg['meta']['model']} | Response time: {msg['meta']['latency']}</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="footer-note">ACity RAG Assistant  •  Retrieval-Augmented Generation System  •  Built for CS4241 Project</div>', unsafe_allow_html=True)

    with side_col:
        result = st.session_state.active_result
        st.markdown('<div class="panel"><h4>RETRIEVED CONTEXT</h4>', unsafe_allow_html=True)
        st.markdown("<div class='small-muted' style='float:right; margin-top:-1.55rem;'>3 chunks</div>", unsafe_allow_html=True)
        if result and result.get("retrieved"):
            for item in result["retrieved"][:3]:
                snippet = item["text"][:140] + ("..." if len(item["text"]) > 140 else "")
                st.markdown(
                    f"""
<div class="ctx-item">
  <div><b>{item["source"]}</b></div>
  <div class="small-muted">{snippet}</div>
  <div class="small-muted">Score: {item["final_score"]:.2f}</div>
</div>
""",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div class="small-muted">No retrieval yet.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel"><h4>SIMILARITY SCORES</h4>', unsafe_allow_html=True)
        if result and result.get("retrieved"):
            for i, item in enumerate(result["retrieved"][:3], start=1):
                score = max(0.0, min(1.0, float(item["final_score"])))
                st.markdown(f"Chunk {i}")
                st.progress(score)
                st.markdown(f"<div class='small-muted'>{score:.2f}</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="small-muted">No scores yet.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel"><h4>PIPELINE LOG</h4>', unsafe_allow_html=True)
        if result:
            for step in [
                "Query Received",
                "Retrieval Completed",
                "Context Selected",
                "Prompt Constructed",
                "Response Generated",
            ]:
                st.markdown(f"- {step}")
        else:
            st.markdown('<div class="small-muted">No pipeline event yet.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get("pipeline_error"):
        st.warning(
            "LLM service is not configured. Add `GROQ_API_KEY` (or `OPENAI_API_KEY`) in app secrets/environment "
            "to enable chat responses."
        )

    user_input = st.chat_input("Ask anything about Ghana elections or the 2025 budget...")
    if user_input and user_input.strip():
        if st.session_state.pipeline is None:
            st.error("Chat is unavailable until API key is configured in deployment secrets.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input.strip()})
            with st.spinner("Thinking..."):
                result = st.session_state.pipeline.ask(user_input.strip(), top_k=3)
            st.session_state.active_result = result
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["response"],
                    "sources": result["retrieved"][:3],
                    "meta": {"model": AppConfig().llm_model, "latency": "~3-5s"},
                }
            )
            st.rerun()
