"""Modern, interactive Streamlit UI for the Autonomous Multi-Agent AI Research System."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src.agents.graph import build_research_graph
from src.agents.nodes import ResearchAgents
from src.config import get_settings
from src.llm import GeminiClient
from src.logger import UILogHandler, setup_logger
from src.permissions import PermissionManager, PermissionRequest
from src.tools.web_browser import WebBrowserTool

st.set_page_config(
    page_title="Autonomous Multi-Agent AI Research System",
    page_icon="🔬",
    layout="wide",
)

# Custom CSS for a cleaner, modern look
st.markdown(
    """
    <style>
    .hero-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: #f8fafc;
        margin-bottom: 1.5rem;
        border: 1px solid #334155;
    }
    .hero-box h1 { margin: 0; font-size: 1.9rem; }
    .hero-box p { margin: 0.4rem 0 0; color: #cbd5e1; font-size: 1rem; }
    .agent-pill {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        background: #e2e8f0;
        color: #0f172a;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-box">
        <h1>🔬 Autonomous Multi-Agent AI Research System</h1>
        <p>6 specialized AI agents that plan, search the web, read pages, compare facts, and write cited reports.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

settings = get_settings()

MODEL_INFO = {
    "gemini-3.8-flash": "⚡ Latest, fast & smart (Recommended)",
    "gemini-3.8-pro": "🧠 Deepest thinking (Best for complex analysis)",
    "gemini-3.5-flash": "🚀 Lightweight & quick",
    "gemini-3.5-pro": "🎯 High-accuracy reasoning",
}

with st.sidebar:
    st.header("🎛️ Controls & Settings")

    env_key_present = settings.has_valid_api_key()
    if env_key_present:
        st.success(f"🔑 API Key active ({settings.masked_api_key()})")
        api_key = settings.gemini_api_key
    else:
        api_key = st.text_input(
            "🔑 Gemini API Key",
            type="password",
            help="Your secret key from Google AI Studio. Never shown or saved to disk.",
        )

    st.divider()
    st.subheader("🤖 AI Brain")
    model_name = st.selectbox(
        "Choose Gemini Model",
        options=list(MODEL_INFO.keys()),
        index=0,
        format_func=lambda m: f"{m} — {MODEL_INFO[m]}",
        help="Pick which Gemini model powers your 6 research agents.",
    )

    st.divider()
    st.subheader("🔍 Research Depth")
    max_iterations = st.slider(
        "Research Rounds",
        min_value=1,
        max_value=4,
        value=2,
        help="How many times the AI double-checks for missing info and searches again.",
    )
    max_search_results = st.slider(
        "Links per Search",
        min_value=2,
        max_value=10,
        value=settings.max_search_results,
        help="How many search results to collect for each search query.",
    )
    max_pages_per_step = st.slider(
        "Pages to Read per Round",
        min_value=1,
        max_value=6,
        value=settings.max_pages_per_step,
        help="How many websites the Investigation Agent opens and reads in detail.",
    )

    st.divider()
    st.subheader("🛡️ Safety & Permissions")
    require_permission = st.toggle(
        "Ask permission before sensitive actions",
        value=True,
        help="Controls whether agents need your approval before visiting websites or saving files.",
    )
    allow_external_browsing = st.checkbox(
        "Allow visiting external websites",
        value=True,
        help="Lets the Investigation Agent open and read webpages found in search.",
    )
    allow_disk_save = st.checkbox(
        "Allow saving report to disk",
        value=True,
        help="Saves a copy of the final Markdown report in the reports/ folder.",
    )

# Agent team overview
st.markdown(
    """
    <div>
        <span class="agent-pill">1. Manager</span>
        <span class="agent-pill">2. Web Researcher</span>
        <span class="agent-pill">3. Investigator</span>
        <span class="agent-pill">4. Analyst</span>
        <span class="agent-pill">5. Fact-Checker</span>
        <span class="agent-pill">6. Reporter</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Quick-start example prompts
st.caption("💡 **Quick-start ideas** (click one to fill):")
col1, col2, col3 = st.columns(3)
if col1.button("🎮 Review & rate a website", use_container_width=True):
    st.session_state["query_input"] = "Review and rate this web app/game: https://business-dynasty.vercel.app"
if col2.button("🔋 Compare 2026 battery tech", use_container_width=True):
    st.session_state["query_input"] = "Compare solid-state vs lithium-ion battery breakthroughs in 2025-2026 with sources."
if col3.button("🤖 Top AI agent frameworks", use_container_width=True):
    st.session_state["query_input"] = "Compare LangGraph, CrewAI, and AutoGen for multi-agent systems."

query = st.text_area(
    "What would you like the AI team to research?",
    key="query_input",
    placeholder="Type any topic, question, or URL to investigate...",
    height=100,
)

run_clicked = st.button("🚀 Start Autonomous Research", type="primary", use_container_width=True)

if run_clicked:
    if not api_key or api_key == "your_gemini_api_key_here":
        st.error("Please provide a valid Gemini API key in `.env` or the sidebar.")
        st.stop()
    if not query.strip():
        st.warning("Please enter a research instruction.")
        st.stop()

    os.environ["GEMINI_API_KEY"] = api_key

    log_handler = UILogHandler(max_records=150)
    logger = setup_logger(log_dir=settings.logs_dir, level=settings.log_level, ui_handler=log_handler)

    def permission_callback(req: PermissionRequest) -> bool:
        if req.action == "visit_webpage":
            return allow_external_browsing
        if req.action == "save_report":
            return allow_disk_save
        return False

    perm_manager = PermissionManager(
        require_approval=require_permission,
        callback=permission_callback,
        logger=logger,
    )

    status_box = st.status("Initializing multi-agent research team...", expanded=True)
    log_placeholder = st.empty()

    def on_agent_status(agent_name: str, msg: str) -> None:
        status_box.write(f"**{agent_name}**: {msg}")
        log_placeholder.code("\n".join(log_handler.records_buffer), language="text")

    try:
        llm = GeminiClient(api_key=api_key, model=model_name, logger=logger)
        with WebBrowserTool(
            headless=settings.headless_browser,
            permission_manager=perm_manager,
            logger=logger,
        ) as browser:
            agents = ResearchAgents(
                llm=llm,
                browser=browser,
                permission_manager=perm_manager,
                reports_dir=settings.reports_dir,
                max_search_results=max_search_results,
                max_pages_per_step=max_pages_per_step,
                status_callback=on_agent_status,
                logger=logger,
            )
            graph = build_research_graph(agents)
            initial_state = {
                "user_query": query.strip(),
                "objective": "",
                "sub_questions": [],
                "search_queries": [],
                "discovered_urls": [],
                "sources": [],
                "analysis_notes": "",
                "knowledge_gaps": [],
                "fact_check_results": [],
                "final_report": "",
                "report_path": None,
                "iteration": 0,
                "max_iterations": max_iterations,
                "messages": [],
                "errors": [],
            }
            result = graph.invoke(initial_state)

        status_box.update(label="✅ Research complete!", state="complete", expanded=False)

        tab_report, tab_sources, tab_facts, tab_logs = st.tabs(
            ["📄 Final Report", "🔗 Sources", "✅ Fact-Check", "📜 Agent Logs"]
        )

        with tab_report:
            st.markdown(result.get("final_report", "No report generated."))
            if result.get("report_path"):
                st.info(f"Report saved to: `{result['report_path']}`")
            st.download_button(
                "⬇️ Download Markdown Report",
                data=result.get("final_report", ""),
                file_name="research_report.md",
                mime="text/markdown",
            )

        with tab_sources:
            sources = result.get("sources", [])
            if not sources:
                st.write("No sources were collected.")
            for idx, src in enumerate(sources, 1):
                st.markdown(f"**{idx}. [{src['title']}]({src['url']})**")
                st.caption(src["snippet"])

        with tab_facts:
            facts = result.get("fact_check_results", [])
            if facts:
                st.dataframe(facts, use_container_width=True)
            else:
                st.write("No fact-check results available.")

        with tab_logs:
            st.code("\n".join(log_handler.records_buffer), language="text")

    except Exception as exc:
        status_box.update(label="❌ Research failed", state="error", expanded=True)
        st.error(f"An error occurred: {exc}")
