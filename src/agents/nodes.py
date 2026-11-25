"""Implementation of the 6 specialized research agents."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from src.agents.state import FactCheckSummary, ResearchPlan, ResearchState, SourceDocument
from src.llm import GeminiClient
from src.permissions import PermissionManager
from src.tools.web_browser import WebBrowserTool


class ResearchAgents:
    """Container for all 6 collaborative research agents."""

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
        self.reports_dir = reports_dir or Path("reports")
        self.max_search_results = max_search_results
        self.max_pages_per_step = max_pages_per_step
        self.status_callback = status_callback
        self.logger = logger or logging.getLogger("research_system.agents")

    def _notify(self, agent_name: str, message: str) -> None:
        self.logger.info("[%s] %s", agent_name, message)
        if self.status_callback:
            self.status_callback(agent_name, message)

    def manager_node(self, state: ResearchState) -> dict:
        """1. Manager Agent: Understands instructions, plans & coordinates."""
        iteration = state.get("iteration", 0) + 1
        self._notify("Manager Agent", f"Planning research iteration {iteration}...")

        gaps = state.get("knowledge_gaps", [])
        if iteration == 1:
            prompt = (
                f"User Research Request:\n{state['user_query']}\n\n"
                "Decompose this into a clear objective, 3-5 sub-questions, and 2-4 web search queries."
            )
        else:
            prompt = (
                f"User Research Request:\n{state['user_query']}\n\n"
                f"Previous knowledge gaps identified by Analysis Agent:\n{gaps}\n\n"
                "Generate 2 targeted follow-up search queries to fill these gaps."
            )

        try:
            plan = self.llm.generate_structured(
                prompt=prompt,
                response_schema=ResearchPlan,
                system_instruction="You are the central Manager Agent coordinating an autonomous research team.",
            )
            return {
                "objective": plan.objective,
                "sub_questions": plan.sub_questions,
                "search_queries": plan.search_queries,
                "iteration": iteration,
                "messages": [
                    {
                        "agent": "Manager Agent",
                        "content": f"Iteration {iteration} plan ready with {len(plan.search_queries)} queries.",
                    }
                ],
            }
        except Exception as exc:
            self.logger.error("Manager Agent failed: %s", exc)
            return {
                "objective": state["user_query"],
                "sub_questions": [state["user_query"]],
                "search_queries": [state["user_query"]],
                "iteration": iteration,
                "errors": [f"Manager Agent error: {exc}"],
                "messages": [{"agent": "Manager Agent", "content": f"Fallback plan used due to error: {exc}"}],
            }

    def web_research_node(self, state: ResearchState) -> dict:
        """2. Web Research Agent: Searches Google/web for relevant sources."""
        queries = state.get("search_queries", [])
        self._notify("Web Research Agent", f"Searching web for {len(queries)} queries...")

        existing_urls = set(state.get("discovered_urls", []))
        new_urls: list[str] = []

        for query in queries:
            results = self.browser.search_google(query, max_results=self.max_search_results)
            for res in results:
                if res.url not in existing_urls and res.url not in new_urls:
                    new_urls.append(res.url)

        combined_urls = list(existing_urls) + new_urls
        return {
            "discovered_urls": combined_urls,
            "messages": [
                {
                    "agent": "Web Research Agent",
                    "content": f"Discovered {len(new_urls)} new URLs (total: {len(combined_urls)}).",
                }
            ],
        }

    def investigation_node(self, state: ResearchState) -> dict:
        """3. Investigation Agent: Explores webpages and extracts content."""
        already_fetched = {s["url"] for s in state.get("sources", [])}
        candidates = [u for u in state.get("discovered_urls", []) if u not in already_fetched]
        to_visit = candidates[: self.max_pages_per_step]

        self._notify("Investigation Agent", f"Exploring {len(to_visit)} webpages...")
        fetched_sources: list[SourceDocument] = []

        for url in to_visit:
            page = self.browser.fetch_page(url)
            if page.error:
                self.logger.warning("Skipping %s: %s", url, page.error)
                continue
            if page.text:
                fetched_sources.append(
                    {
                        "url": page.url,
                        "title": page.title,
                        "snippet": page.text[:300],
                        "content": page.text,
                        "links": page.links,
                    }
                )

        return {
            "sources": fetched_sources,
            "messages": [
                {
                    "agent": "Investigation Agent",
                    "content": f"Extracted content from {len(fetched_sources)} webpages.",
                }
            ],
        }

    def analysis_node(self, state: ResearchState) -> dict:
        """4. Analysis Agent: Compares collected information and identifies gaps."""
        sources = state.get("sources", [])
        self._notify("Analysis Agent", f"Comparing {len(sources)} collected sources...")

        if not sources:
            return {
                "analysis_notes": "No source content was collected.",
                "knowledge_gaps": [state["user_query"]],
                "messages": [{"agent": "Analysis Agent", "content": "No sources available to analyze."}],
            }

        context_blocks = []
        for idx, src in enumerate(sources, 1):
            context_blocks.append(f"[{idx}] {src['title']} ({src['url']})\n{src['content'][:2000]}")

        prompt = (
            f"Research Objective: {state.get('objective', state['user_query'])}\n"
            f"Sub-questions: {state.get('sub_questions', [])}\n\n"
            "Collected Sources:\n" + "\n\n---\n\n".join(context_blocks) + "\n\n"
            "Compare the sources, synthesize key findings, note any contradictions, "
            "and list any remaining knowledge gaps (if any) on a new line starting with 'GAP:'."
        )

        try:
            analysis = self.llm.generate_text(
                prompt=prompt,
                system_instruction="You are the Analysis Agent. Compare sources objectively and cite [number].",
            )
            gaps = [
                line.replace("GAP:", "").strip()
                for line in analysis.splitlines()
                if line.strip().startswith("GAP:")
            ]
            return {
                "analysis_notes": analysis,
                "knowledge_gaps": gaps,
                "messages": [
                    {
                        "agent": "Analysis Agent",
                        "content": f"Completed comparative analysis. Identified {len(gaps)} knowledge gaps.",
                    }
                ],
            }
        except Exception as exc:
            self.logger.error("Analysis Agent failed: %s", exc)
            return {
                "analysis_notes": "Analysis failed due to an LLM error.",
                "knowledge_gaps": [],
                "errors": [f"Analysis Agent error: {exc}"],
                "messages": [{"agent": "Analysis Agent", "content": f"Error: {exc}"}],
            }

    def fact_checking_node(self, state: ResearchState) -> dict:
        """5. Fact-Checking Agent: Verifies claims against collected evidence."""
        self._notify("Fact-Checking Agent", "Verifying claims against source evidence...")
        sources = state.get("sources", [])
        analysis = state.get("analysis_notes", "")

        if not sources or not analysis:
            return {
                "fact_check_results": [],
                "messages": [{"agent": "Fact-Checking Agent", "content": "Skipped: insufficient evidence."}],
            }

        source_list = "\n".join(f"- {s['url']}: {s['content'][:1000]}" for s in sources[:6])
        prompt = (
            f"Analysis to verify:\n{analysis[:4000]}\n\n"
            f"Available Evidence:\n{source_list}\n\n"
            "Extract 3-5 key factual claims from the analysis and verify each against the evidence."
        )

        try:
            summary = self.llm.generate_structured(
                prompt=prompt,
                response_schema=FactCheckSummary,
                system_instruction="You are the Fact-Checking Agent. Be strict and cite exact source URLs.",
            )
            results = [v.model_dump() for v in summary.verifications]
            return {
                "fact_check_results": results,
                "messages": [
                    {
                        "agent": "Fact-Checking Agent",
                        "content": f"Verified {len(results)} claims (Overall: {summary.overall_reliability}).",
                    }
                ],
            }
        except Exception as exc:
            self.logger.error("Fact-Checking Agent failed: %s", exc)
            return {
                "fact_check_results": [],
                "errors": [f"Fact-Checking Agent error: {exc}"],
                "messages": [{"agent": "Fact-Checking Agent", "content": f"Error: {exc}"}],
            }

    def report_node(self, state: ResearchState) -> dict:
        """6. Report Agent: Produces a detailed Markdown report with source links."""
        self._notify("Report Agent", "Generating final research report...")

        sources = state.get("sources", [])
        source_refs = "\n".join(f"[{i}] [{s['title']}]({s['url']})" for i, s in enumerate(sources, 1))
        fact_checks = state.get("fact_check_results", [])

        prompt = (
            f"User Query: {state['user_query']}\n"
            f"Objective: {state.get('objective', '')}\n\n"
            f"Analysis:\n{state.get('analysis_notes', '')}\n\n"
            f"Fact-Check Results:\n{fact_checks}\n\n"
            f"Sources:\n{source_refs}\n\n"
            "Write a comprehensive, well-structured Markdown research report with:\n"
            "1. Executive Summary\n"
            "2. Key Findings (with inline Markdown links to sources)\n"
            "3. Fact-Check & Reliability Summary\n"
            "4. References\n"
        )

        try:
            report_md = self.llm.generate_text(
                prompt=prompt,
                system_instruction="You are the Report Agent. Produce clear, accurate, well-cited Markdown reports.",
                temperature=0.3,
            )
        except Exception as exc:
            self.logger.error("Report Agent failed: %s", exc)
            report_md = f"# Research Report\n\nError generating report: {exc}\n\n## Sources\n{source_refs}"

        saved_path: Optional[str] = None
        if self.permission_manager is None or self.permission_manager.request_permission(
            action="save_report",
            target=str(self.reports_dir),
            reason="Report Agent wants to save the final Markdown report to disk.",
        ):
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            out_file = self.reports_dir / filename
            out_file.write_text(report_md, encoding="utf-8")
            saved_path = str(out_file)
            self.logger.info("Saved report to %s", saved_path)

        return {
            "final_report": report_md,
            "report_path": saved_path,
            "messages": [{"agent": "Report Agent", "content": "Final report generated."}],
        }
