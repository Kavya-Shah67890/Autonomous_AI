"""Autonomous Multi-Agent AI Research System — Self-Contained Streamlit App."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from operator import add
from pathlib import Path
from typing import Annotated, Any, Callable, Deque, Dict, List, Optional, Type, TypeVar, TypedDict
from urllib.parse import quote_plus, urlparse

import streamlit as st
import requests
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langgraph.graph import END, START, StateGraph
from playwright.sync_api import Browser, Playwright, sync_playwright
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------------------------
# Logging & Secret Redaction
# ---------------------------------------------------------------------------
_SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),
    re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)([^\s,;]+)"),
]


def redact_secrets(text: str) -> str:
    if not text:
        return text
    redacted = _SECRET_PATTERNS[0].sub("[REDACTED_GEMINI_KEY]", text)
    redacted = _SECRET_PATTERNS[1].sub(r"\1[REDACTED]", redacted)
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    if env_key and len(env_key) > 6 and env_key in redacted:
        redacted = redacted.replace(env_key, "[REDACTED_GEMINI_KEY]")
    return redacted


class SecretRedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_secrets(record.msg)
        if record.args:
            record.args = tuple(redact_secrets(a) if isinstance(a, str) else a for a in record.args)
        return True


class UILogHandler(logging.Handler):
    def __init__(self, max_records: int = 200):
        super().__init__()
        self.records_buffer: Deque[str] = deque(maxlen=max_records)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records_buffer.append(self.format(record))
        except Exception:
            self.handleError(record)


def setup_logger(log_dir: Optional[Path] = None, level: str = "INFO", ui_handler: Optional[UILogHandler] = None) -> logging.Logger:
    logger = logging.getLogger("research_system_app")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    redacting_filter = SecretRedactingFilter()

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "research_system.log", encoding="utf-8")
        fh.setFormatter(formatter)
        fh.addFilter(redacting_filter)
        logger.addHandler(fh)

    if ui_handler is not None:
        ui_handler.setFormatter(formatter)
        ui_handler.addFilter(redacting_filter)
        logger.addHandler(ui_handler)

    return logger


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------
@dataclass
class PermissionRequest:
    action: str
    target: str
    reason: str


class PermissionManager:
    def __init__(
        self,
        require_approval: bool = True,
        callback: Optional[Callable[[PermissionRequest], bool]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.require_approval = require_approval
        self.callback = callback
        self.logger = logger or logging.getLogger("research_system_app")
        self._approved: set[tuple[str, str]] = set()

    def request_permission(self, action: str, target: str, reason: str) -> bool:
        if not self.require_approval:
            return True
        key = (action, target)
        if key in self._approved:
            return True
        if self.callback is None:
            return False
        approved = bool(self.callback(PermissionRequest(action=action, target=target, reason=reason)))
        if approved:
            self._approved.add(key)
        return approved


# ---------------------------------------------------------------------------
# Gemini Client
# ---------------------------------------------------------------------------
T = TypeVar("T", bound=BaseModel)


def friendly_error_message(exc: Exception) -> str:
    """Convert raw API exceptions into short, user-friendly messages."""
    msg = str(exc)
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
        return "⏳ Rate limit reached (free tier allows ~5 requests/minute). Please wait 30 seconds and try again, or pick another model."
    if "401" in msg or "403" in msg or "API_KEY_INVALID" in msg:
        return "🔑 Invalid Gemini API key. Please check your key in the sidebar."
    if "404" in msg or "NOT_FOUND" in msg:
        return "🤖 The selected Gemini model is unavailable. Please choose another model in the sidebar."
    return "⚠️ Something went wrong while contacting the AI. Please try again."


class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-4-hydrogen", logger: Optional[logging.Logger] = None):
        if not api_key or api_key == "your_gemini_api_key_here":
            raise ValueError("A valid GEMINI_API_KEY is required.")
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.logger = logger or logging.getLogger("research_system_app")

    def generate_text(self, prompt: str, system_instruction: Optional[str] = None, temperature: float = 0.2, max_retries: int = 3) -> str:
        config = types.GenerateContentConfig(temperature=temperature, system_instruction=system_instruction)
        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._client.models.generate_content(model=self.model, contents=prompt, config=config)
                return (resp.text or "").strip()
            except Exception as exc:
                last_err = exc
                self.logger.warning("Gemini text call failed (attempt %d/%d): %s", attempt, max_retries, exc)
                if attempt < max_retries:
                    time.sleep(1.5 * attempt)
        raise RuntimeError(f"Gemini API failed after {max_retries} attempts: {last_err}") from last_err

    def generate_structured(self, prompt: str, response_schema: Type[T], system_instruction: Optional[str] = None, temperature: float = 0.1, max_retries: int = 3) -> T:
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=response_schema,
        )
        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._client.models.generate_content(model=self.model, contents=prompt, config=config)
                data: Any = json.loads((resp.text or "").strip())
                return response_schema.model_validate(data)
            except Exception as exc:
                last_err = exc
                self.logger.warning("Gemini structured call failed (attempt %d/%d): %s", attempt, max_retries, exc)
                if attempt < max_retries:
                    time.sleep(1.5 * attempt)
        raise RuntimeError(f"Gemini structured API failed after {max_retries} attempts: {last_err}") from last_err


# ---------------------------------------------------------------------------
# Playwright Web Browser Tool (with auto-install)
# ---------------------------------------------------------------------------
@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass
class PageContent:
    url: str
    title: str
    text: str
    links: List[str]
    error: Optional[str] = None


class WebBrowserTool:
    def __init__(self, headless: bool = True, permission_manager: Optional[PermissionManager] = None, logger: Optional[logging.Logger] = None):
        self.headless = headless
        self.permission_manager = permission_manager
        self.logger = logger or logging.getLogger("research_system_app")
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None

    def __enter__(self) -> "WebBrowserTool":
        self._ensure_browser()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()

    def _ensure_browser(self) -> Browser:
        if self._browser is None:
            self._playwright = sync_playwright().start()
            try:
                self._browser = self._playwright.chromium.launch(headless=self.headless)
            except Exception as exc:
                if "Executable doesn't exist" in str(exc):
                    self.logger.info("Playwright Chromium not found. Installing automatically...")
                    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
                    self._browser = self._playwright.chromium.launch(headless=self.headless)
                else:
                    raise
        return self._browser

    def search_google(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Fast web search using DuckDuckGo HTML via requests, with Playwright fallback."""
        self.logger.info("Searching web for: %s", query)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        }
        try:
            resp = requests.get(f"https://html.duckduckgo.com/html/?q={quote_plus(query)}", headers=headers, timeout=8)
            if resp.ok:
                results = self._parse_ddg_html(resp.text, max_results)
                if results:
                    self.logger.info("Found %d search results for '%s'", len(results), query)
                    return results
        except Exception as exc:
            self.logger.warning("Fast search failed for '%s': %s", query, exc)

        # Fallback to Playwright if needed
        results = []
        browser = self._ensure_browser()
        context = browser.new_context(user_agent=headers["User-Agent"])
        page = context.new_page()
        try:
            page.goto(f"https://html.duckduckgo.com/html/?q={quote_plus(query)}", timeout=10000, wait_until="domcontentloaded")
            results = self._parse_ddg_html(page.content(), max_results)
        except Exception as exc:
            self.logger.error("Playwright search failed for '%s': %s", query, exc)
        finally:
            context.close()
        self.logger.info("Found %d search results for '%s'", len(results), query)
        return results

    @staticmethod
    def _parse_google_html(html: str, max_results: int) -> List[SearchResult]:
        soup = BeautifulSoup(html, "html.parser")
        results: List[SearchResult] = []
        for g in soup.select("div.g"):
            link_el, title_el = g.select_one("a[href]"), g.select_one("h3")
            if not link_el or not title_el:
                continue
            href = link_el.get("href", "")
            if not href.startswith("http"):
                continue
            snippet_el = g.select_one("div.VwiC3b, span.aCOpRe")
            results.append(SearchResult(title=title_el.get_text(strip=True), url=href, snippet=snippet_el.get_text(" ", strip=True) if snippet_el else ""))
            if len(results) >= max_results:
                break
        return results

    @staticmethod
    def _parse_ddg_html(html: str, max_results: int) -> List[SearchResult]:
        soup = BeautifulSoup(html, "html.parser")
        results: List[SearchResult] = []
        for item in soup.select("div.result"):
            link_el = item.select_one("a.result__a")
            if not link_el:
                continue
            href = link_el.get("href", "")
            if not href.startswith("http"):
                continue
            snippet_el = item.select_one("a.result__snippet")
            results.append(SearchResult(title=link_el.get_text(strip=True), url=href, snippet=snippet_el.get_text(" ", strip=True) if snippet_el else ""))
            if len(results) >= max_results:
                break
        return results

    def fetch_page(self, url: str, max_chars: int = 6000) -> PageContent:
        domain = urlparse(url).netloc
        if self.permission_manager is not None:
            allowed = self.permission_manager.request_permission(
                action="visit_webpage",
                target=url,
                reason=f"Investigation Agent wants to read content from external domain '{domain}'.",
            )
            if not allowed:
                return PageContent(url=url, title="", text="", links=[], error="Permission denied by user.")

        self.logger.info("Exploring webpage: %s", url)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        }
        try:
            resp = requests.get(url, headers=headers, timeout=8)
            if resp.ok and len(resp.text) > 300:
                soup = BeautifulSoup(resp.text, "html.parser")
                title = (soup.title.string.strip() if soup.title and soup.title.string else url)
                for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
                    tag.decompose()
                text = " ".join(soup.get_text(" ", strip=True).split())[:max_chars]
                if len(text) > 150:
                    links = [a.get("href", "") for a in soup.select("a[href]") if a.get("href", "").startswith("http")][:15]
                    return PageContent(url=url, title=title, text=text, links=links)
        except Exception:
            pass

        # Fallback to Playwright for JS-rendered pages
        browser = self._ensure_browser()
        context = browser.new_context(user_agent=headers["User-Agent"])
        page = context.new_page()
        try:
            page.goto(url, timeout=12000, wait_until="domcontentloaded")
            page.wait_for_timeout(500)
            title = page.title() or url
            soup = BeautifulSoup(page.content(), "html.parser")
            for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
                tag.decompose()
            text = " ".join(soup.get_text(" ", strip=True).split())[:max_chars]
            links = [a.get("href", "") for a in soup.select("a[href]") if a.get("href", "").startswith("http")][:15]
            return PageContent(url=url, title=title, text=text, links=links)
        except Exception as exc:
            self.logger.error("Failed to fetch %s: %s", url, exc)
            return PageContent(url=url, title="", text="", links=[], error=str(exc))
        finally:
            context.close()


# ---------------------------------------------------------------------------
# LangGraph 6-Agent State & Nodes
# ---------------------------------------------------------------------------
class ResearchPlan(BaseModel):
    objective: str = Field(description="Clear restatement of the user's research goal.")
    sub_questions: List[str] = Field(description="3 to 5 focused sub-questions to answer.")
    search_queries: List[str] = Field(description="2 to 4 concrete search queries.")


class ClaimVerification(BaseModel):
    claim: str
    status: str = Field(description="Verified, Partially Verified, Unverified, or Contradicted")
    evidence: str
    source_urls: List[str]


class FactCheckSummary(BaseModel):
    verifications: List[ClaimVerification]
    overall_reliability: str


class AgentMessage(TypedDict):
    agent: str
    content: str


class SourceDocument(TypedDict):
    url: str
    title: str
    snippet: str
    content: str
    links: List[str]


class ResearchState(TypedDict):
    user_query: str
    objective: str
    sub_questions: List[str]
    search_queries: List[str]
    discovered_urls: List[str]
    sources: Annotated[List[SourceDocument], add]
    analysis_notes: str
    knowledge_gaps: List[str]
    fact_check_results: List[Dict[str, object]]
    final_report: str
    report_path: Optional[str]
    iteration: int
    max_iterations: int
    messages: Annotated[List[AgentMessage], add]
    errors: Annotated[List[str], add]


class ResearchAgents:
    def __init__(
        self,
        llm: GeminiClient,
        browser: WebBrowserTool,
        permission_manager: Optional[PermissionManager] = None,
        reports_dir: Optional[Path] = None,
        max_search_results: int = 5,
        max_pages_per_step: int = 3,
        status_callback: Optional[Callable[[str, str], None]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.llm = llm
        self.browser = browser
        self.permission_manager = permission_manager
        self.reports_dir = reports_dir or (BASE_DIR / "reports")
        self.max_search_results = max_search_results
        self.max_pages_per_step = max_pages_per_step
        self.status_callback = status_callback
        self.logger = logger or logging.getLogger("research_system_app")

    def _notify(self, agent_name: str, message: str) -> None:
        self.logger.info("[%s] %s", agent_name, message)
        if self.status_callback:
            self.status_callback(agent_name, message)

    def manager_node(self, state: ResearchState) -> dict:
        iteration = state.get("iteration", 0) + 1
        self._notify("Manager Agent", f"Planning research iteration {iteration}...")
        gaps = state.get("knowledge_gaps", [])
        if iteration == 1:
            prompt = f"User Research Request:\n{state['user_query']}\n\nDecompose this into a clear objective, 3-5 sub-questions, and 2-4 web search queries."
        else:
            prompt = f"User Research Request:\n{state['user_query']}\n\nPrevious knowledge gaps:\n{gaps}\n\nGenerate 2 targeted follow-up search queries."

        try:
            plan = self.llm.generate_structured(prompt=prompt, response_schema=ResearchPlan, system_instruction="You are the central Manager Agent coordinating an autonomous research team.")
            return {
                "objective": plan.objective,
                "sub_questions": plan.sub_questions,
                "search_queries": plan.search_queries,
                "iteration": iteration,
                "messages": [{"agent": "Manager Agent", "content": f"Iteration {iteration} plan ready."}],
            }
        except Exception as exc:
            self.logger.error("Manager Agent failed: %s", exc)
            return {
                "objective": state["user_query"],
                "sub_questions": [state["user_query"]],
                "search_queries": [state["user_query"]],
                "iteration": iteration,
                "errors": [f"Manager Agent error: {exc}"],
                "messages": [{"agent": "Manager Agent", "content": f"Fallback plan used: {exc}"}],
            }

    def web_research_node(self, state: ResearchState) -> dict:
        queries = state.get("search_queries", [])
        self._notify("Web Research Agent", f"Searching web for {len(queries)} queries...")
        existing_urls = set(state.get("discovered_urls", []))
        new_urls: list[str] = []

        # Also include any direct URLs mentioned in the user's prompt
        for token in state["user_query"].split():
            if token.startswith("http://") or token.startswith("https://"):
                clean_url = token.rstrip(".,);]")
                if clean_url not in existing_urls and clean_url not in new_urls:
                    new_urls.append(clean_url)

        for query in queries:
            for res in self.browser.search_google(query, max_results=self.max_search_results):
                if res.url not in existing_urls and res.url not in new_urls:
                    new_urls.append(res.url)

        combined_urls = list(existing_urls) + new_urls
        return {
            "discovered_urls": combined_urls,
            "messages": [{"agent": "Web Research Agent", "content": f"Discovered {len(new_urls)} new URLs."}],
        }

    def investigation_node(self, state: ResearchState) -> dict:
        already_fetched = {s["url"] for s in state.get("sources", [])}
        candidates = [u for u in state.get("discovered_urls", []) if u not in already_fetched]
        to_visit = candidates[: self.max_pages_per_step]
        self._notify("Investigation Agent", f"Exploring {len(to_visit)} webpages...")
        fetched_sources: list[SourceDocument] = []
        for url in to_visit:
            page = self.browser.fetch_page(url)
            if page.error or not page.text:
                continue
            fetched_sources.append({"url": page.url, "title": page.title, "snippet": page.text[:300], "content": page.text, "links": page.links})
        return {
            "sources": fetched_sources,
            "messages": [{"agent": "Investigation Agent", "content": f"Extracted content from {len(fetched_sources)} webpages."}],
        }

    def analysis_node(self, state: ResearchState) -> dict:
        sources = state.get("sources", [])
        self._notify("Analysis Agent", f"Comparing {len(sources)} collected sources...")
        if not sources:
            return {"analysis_notes": "No source content was collected.", "knowledge_gaps": [state["user_query"]], "messages": [{"agent": "Analysis Agent", "content": "No sources to analyze."}]}
        context_blocks = [f"[{i}] {s['title']} ({s['url']})\n{s['content'][:2000]}" for i, s in enumerate(sources, 1)]
        prompt = (
            f"Research Objective: {state.get('objective', state['user_query'])}\n"
            f"Sub-questions: {state.get('sub_questions', [])}\n\n"
            "Collected Sources:\n" + "\n\n---\n\n".join(context_blocks) + "\n\n"
            "Compare the sources, synthesize key findings, note contradictions, and list any remaining knowledge gaps on lines starting with 'GAP:'."
        )
        try:
            analysis = self.llm.generate_text(prompt=prompt, system_instruction="You are the Analysis Agent. Compare sources objectively and cite [number].")
            gaps = [line.replace("GAP:", "").strip() for line in analysis.splitlines() if line.strip().startswith("GAP:")]
            return {"analysis_notes": analysis, "knowledge_gaps": gaps, "messages": [{"agent": "Analysis Agent", "content": f"Analysis complete ({len(gaps)} gaps)."}]}
        except Exception as exc:
            self.logger.error("Analysis Agent failed: %s", exc)
            return {"analysis_notes": f"Analysis failed: {exc}", "knowledge_gaps": [], "errors": [str(exc)], "messages": [{"agent": "Analysis Agent", "content": f"Error: {exc}"}]}

    def fact_checking_node(self, state: ResearchState) -> dict:
        self._notify("Fact-Checking Agent", "Verifying claims against source evidence...")
        sources, analysis = state.get("sources", []), state.get("analysis_notes", "")
        if not sources or not analysis:
            return {"fact_check_results": [], "messages": [{"agent": "Fact-Checking Agent", "content": "Skipped: insufficient evidence."}]}
        source_list = "\n".join(f"- {s['url']}: {s['content'][:1000]}" for s in sources[:6])
        prompt = f"Analysis:\n{analysis[:4000]}\n\nEvidence:\n{source_list}\n\nExtract 3-5 key claims and verify each."
        try:
            summary = self.llm.generate_structured(prompt=prompt, response_schema=FactCheckSummary, system_instruction="You are the Fact-Checking Agent. Be strict and cite exact URLs.")
            return {"fact_check_results": [v.model_dump() for v in summary.verifications], "messages": [{"agent": "Fact-Checking Agent", "content": f"Verified {len(summary.verifications)} claims."}]}
        except Exception as exc:
            self.logger.error("Fact-Checking Agent failed: %s", exc)
            return {"fact_check_results": [], "errors": [str(exc)], "messages": [{"agent": "Fact-Checking Agent", "content": f"Error: {exc}"}]}

    def report_node(self, state: ResearchState) -> dict:
        self._notify("Report Agent", "Generating final research report...")
        sources = state.get("sources", [])
        source_refs = "\n".join(f"[{i}] [{s['title']}]({s['url']})" for i, s in enumerate(sources, 1))
        prompt = (
            f"User Query: {state['user_query']}\nObjective: {state.get('objective', '')}\n\n"
            f"Analysis:\n{state.get('analysis_notes', '')}\n\nFact-Check Results:\n{state.get('fact_check_results', [])}\n\n"
            f"Sources:\n{source_refs}\n\nWrite a detailed Markdown report with Executive Summary, Key Findings (inline links), Fact-Check Summary, and References."
        )
        try:
            report_md = self.llm.generate_text(prompt=prompt, system_instruction="You are the Report Agent. Produce clear, accurate, well-cited Markdown reports.", temperature=0.3)
        except Exception as exc:
            self.logger.error("Report Agent failed: %s", exc)
            report_md = f"# Research Report\n\n{friendly_error_message(exc)}\n\n## Sources\n{source_refs}"

        saved_path: Optional[str] = None
        if self.permission_manager is None or self.permission_manager.request_permission(action="save_report", target=str(self.reports_dir), reason="Save final report to disk."):
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            out_file = self.reports_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            out_file.write_text(report_md, encoding="utf-8")
            saved_path = str(out_file)
        return {"final_report": report_md, "report_path": saved_path, "messages": [{"agent": "Report Agent", "content": "Final report generated."}]}


def build_research_graph(agents: ResearchAgents):
    def should_continue(state: ResearchState) -> str:
        if state.get("knowledge_gaps") and state.get("iteration", 1) < state.get("max_iterations", 2):
            return "manager"
        return "fact_checker"

    graph = StateGraph(ResearchState)
    graph.add_node("manager", agents.manager_node)
    graph.add_node("web_researcher", agents.web_research_node)
    graph.add_node("investigator", agents.investigation_node)
    graph.add_node("analyzer", agents.analysis_node)
    graph.add_node("fact_checker", agents.fact_checking_node)
    graph.add_node("reporter", agents.report_node)

    graph.add_edge(START, "manager")
    graph.add_edge("manager", "web_researcher")
    graph.add_edge("web_researcher", "investigator")
    graph.add_edge("investigator", "analyzer")
    graph.add_conditional_edges("analyzer", should_continue, {"manager": "manager", "fact_checker": "fact_checker"})
    graph.add_edge("fact_checker", "reporter")
    graph.add_edge("reporter", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Autonomous Multi-Agent AI Research System", page_icon="🔬", layout="wide")

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
    <div class="hero-box">
        <h1>🔬 Autonomous Multi-Agent AI Research System</h1>
        <p>6 specialized AI agents that plan, search the web, read pages, compare facts, and write cited reports.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

MODEL_INFO = {
    "gemini-4-hydrogen": "⚡ Gemini 4 Hydrogen — Fast everyday research (Recommended)",
    "gemini-4-neon": "🤖 Gemini 4 Neon — Complex coding & analysis",
    "gemini-4-argon": "🧪 Gemini 4 Argon — Deep multi-step reasoning",
    "gemini-4-carbon": "🧠 Gemini 4 Carbon — Strongest, heaviest tasks",
    "gemini-3.5-flash-lite": "💨 Flash Lite — Ultra-fast & budget-friendly",
}

env_api_key = os.getenv("GEMINI_API_KEY", "").strip()
with st.sidebar:
    st.header("🎛️ Controls & Settings")
    if env_api_key and env_api_key != "your_gemini_api_key_here":
        masked = f"{env_api_key[:4]}...{env_api_key[-4:]}" if len(env_api_key) > 8 else "****"
        st.success(f"🔑 API Key active ({masked})")
        api_key = env_api_key
    else:
        api_key = st.text_input("🔑 Gemini API Key", type="password", help="Get a free key at aistudio.google.com. Never saved.")

    st.divider()
    st.subheader("🤖 AI Brain")
    model_name = st.selectbox(
        "Choose Gemini Model",
        options=list(MODEL_INFO.keys()),
        index=0,
    )
    st.caption(MODEL_INFO[model_name])

    st.divider()
    st.subheader("🔍 Research Depth")
    max_iterations = st.slider("Research Rounds", 1, 4, 1, help="How many times the AI double-checks for missing info.")
    max_search_results = st.slider("Links per Search", 2, 10, 3, help="How many search links to collect per query.")
    max_pages_per_step = st.slider("Pages to Read per Round", 1, 6, 2, help="How many websites to open and read in detail.")

    st.divider()
    st.subheader("🛡️ Safety & Permissions")
    require_permission = st.toggle("Ask permission before sensitive actions", value=True)
    allow_external_browsing = st.checkbox("Allow visiting external websites", value=True)
    allow_disk_save = st.checkbox("Allow saving report to disk", value=True)

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

with st.expander("👥 Meet the 6 AI Agents", expanded=False):
    st.markdown(
        "- **1. Manager**: Breaks your question into sub-questions and search queries.\n"
        "- **2. Web Researcher**: Searches the web for relevant links.\n"
        "- **3. Investigator**: Opens and reads webpages to extract key information.\n"
        "- **4. Analyst**: Compares sources and spots contradictions or gaps.\n"
        "- **5. Fact-Checker**: Verifies claims against the collected evidence.\n"
        "- **6. Reporter**: Writes the final cited Markdown report."
    )

st.caption("💡 **Quick-start ideas** (click one to fill):")
c1, c2, c3 = st.columns(3)
if c1.button("🎮 Review Website", use_container_width=True):
    st.session_state["query_input"] = "Review and rate this web app/game: https://business-dynasty.vercel.app"
if c2.button("🔋 2026 Battery Tech", use_container_width=True):
    st.session_state["query_input"] = "Compare solid-state vs lithium-ion battery breakthroughs in 2025-2026 with sources."
if c3.button("🤖 AI Agent Frameworks", use_container_width=True):
    st.session_state["query_input"] = "Compare LangGraph, CrewAI, and AutoGen for multi-agent systems."

query = st.text_area("What would you like the AI team to research?", key="query_input", placeholder="Type any topic, question, or URL to investigate...", height=100)

if st.button("🚀 Start Autonomous Research", type="primary", use_container_width=True):
    if not api_key or api_key == "your_gemini_api_key_here":
        st.error("Please enter your Gemini API key in the sidebar.")
        st.stop()
    if not query.strip():
        st.warning("Please enter a research instruction.")
        st.stop()

    os.environ["GEMINI_API_KEY"] = api_key
    log_handler = UILogHandler(max_records=150)
    logger = setup_logger(log_dir=BASE_DIR / "logs", level="INFO", ui_handler=log_handler)

    def permission_callback(req: PermissionRequest) -> bool:
        if req.action == "visit_webpage":
            return allow_external_browsing
        if req.action == "save_report":
            return allow_disk_save
        return False

    perm_manager = PermissionManager(require_approval=require_permission, callback=permission_callback, logger=logger)
    status_box = st.status("Initializing multi-agent research team...", expanded=True)
    log_placeholder = st.empty()

    def on_agent_status(agent_name: str, msg: str) -> None:
        status_box.write(f"**{agent_name}**: {msg}")
        log_placeholder.code("\n".join(log_handler.records_buffer), language="text")

    try:
        llm = GeminiClient(api_key=api_key, model=model_name, logger=logger)
        with WebBrowserTool(headless=True, permission_manager=perm_manager, logger=logger) as browser:
            agents = ResearchAgents(
                llm=llm,
                browser=browser,
                permission_manager=perm_manager,
                reports_dir=BASE_DIR / "reports",
                max_search_results=max_search_results,
                max_pages_per_step=max_pages_per_step,
                status_callback=on_agent_status,
                logger=logger,
            )
            result = build_research_graph(agents).invoke(
                {
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
            )

        status_box.update(label="✅ Research complete!", state="complete", expanded=False)
        tab_report, tab_sources, tab_facts, tab_logs = st.tabs(["📄 Final Report", "🔗 Sources", "✅ Fact-Check", "📜 Agent Logs"])
        with tab_report:
            st.markdown(result.get("final_report", "No report generated."))
            if result.get("report_path"):
                st.info(f"Report saved to: `{result['report_path']}`")
            st.download_button("⬇️ Download Markdown Report", data=result.get("final_report", ""), file_name="research_report.md", mime="text/markdown")
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
        st.error(friendly_error_message(exc))
