"""Modern Streamlit UI for the Autonomous Multi-Agent AI Research System."""

from __future__ import annotations

import os
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

st.title("🔬 Autonomous Multi-Agent AI Research System")
st.caption("Powered by Gemini API, LangGraph, and Playwright")

settings = get_settings()

with st.sidebar:
    st.header("⚙️ Configuration")
    env_key_present = settings.has_valid_api_key()
    if env_key_present:
        st.success(f"API Key loaded from .env ({settings.masked_api_key()})")
        api_key = settings.gemini_api_key
    else:
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Stored only in memory for this session. Never logged or exposed.",
        )

    model_name = st.selectbox(
        "Gemini Model",
        options=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        index=0,
    )
    max_iterations = st.slider("Max Autonomous Iterations", min_value=1, max_value=4, value=2)
    require_permission = st.checkbox(
        "Ask permission before sensitive actions",
        value=True,
        help="Prompts before visiting external domains or saving reports to disk.",
    )
    allow_external_browsing = st.checkbox("Pre-approve external web browsing", value=True)
    allow_disk_save = st.checkbox("Pre-approve saving report to disk", value=True)

query = st.text_area(
    "Research Instruction",
    placeholder="e.g., Compare the latest breakthroughs in solid-state batteries in 2025-2026 with credible sources.",
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
                max_search_results=settings.max_search_results,
                max_pages_per_step=settings.max_pages_per_step,
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

        status_box.update(label="Research complete!", state="complete", expanded=False)

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
        status_box.update(label="Research failed", state="error", expanded=True)
        st.error(f"An error occurred: {exc}")
