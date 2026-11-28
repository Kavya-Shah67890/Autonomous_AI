"""Unit and integration tests for the Autonomous Multi-Agent AI Research System."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from src.agents.graph import build_research_graph
from src.agents.nodes import ResearchAgents
from src.agents.state import ClaimVerification, FactCheckSummary, ResearchPlan
from src.logger import redact_secrets
from src.permissions import PermissionManager
from src.tools.web_browser import PageContent, SearchResult


def test_secret_redaction():
    text = "Using api_key=AIzaSyDummySecretKey1234567890abcdef in request"
    redacted = redact_secrets(text)
    assert "AIzaSyDummySecretKey" not in redacted
    assert "[REDACTED" in redacted


def test_permission_manager_denies_without_callback():
    pm = PermissionManager(require_approval=True, callback=None)
    assert pm.request_permission("visit_webpage", "https://example.com", "test") is False


def test_permission_manager_approves_with_callback():
    pm = PermissionManager(require_approval=True, callback=lambda req: req.action == "visit_webpage")
    assert pm.request_permission("visit_webpage", "https://example.com", "test") is True
    assert pm.request_permission("save_report", "reports/", "test") is False


def test_multi_agent_graph_end_to_end(tmp_path: Path):
    mock_llm = MagicMock()
    mock_llm.generate_structured.side_effect = [
        ResearchPlan(
            objective="Test objective",
            sub_questions=["What is X?"],
            search_queries=["X overview"],
        ),
        FactCheckSummary(
            verifications=[
                ClaimVerification(
                    claim="X is a test system.",
                    status="Verified",
                    evidence="Confirmed by example.com",
                    source_urls=["https://example.com/x"],
                )
            ],
            overall_reliability="High",
        ),
    ]
    mock_llm.generate_text.side_effect = [
        "Comparative analysis: [1] confirms X is a test system.",
        "# Research Report\n\nX is a test system. [Source](https://example.com/x)",
    ]

    mock_browser = MagicMock()
    mock_browser.search_google.return_value = [
        SearchResult(title="Example X", url="https://example.com/x", snippet="X is a test system.")
    ]
    mock_browser.fetch_page.return_value = PageContent(
        url="https://example.com/x",
        title="Example X",
        text="X is a test system used for verification.",
        links=["https://example.com/y"],
    )

    pm = PermissionManager(require_approval=False)
    agents = ResearchAgents(
        llm=mock_llm,
        browser=mock_browser,
        permission_manager=pm,
        reports_dir=tmp_path / "reports",
    )
    graph = build_research_graph(agents)

    result = graph.invoke(
        {
            "user_query": "Explain X",
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
            "max_iterations": 2,
            "messages": [],
            "errors": [],
        }
    )

    assert "Research Report" in result["final_report"]
    assert len(result["sources"]) == 1
    assert len(result["fact_check_results"]) == 1
    assert result["report_path"] is not None
    assert Path(result["report_path"]).exists()
