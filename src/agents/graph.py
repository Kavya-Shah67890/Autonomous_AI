"""LangGraph workflow connecting all 6 research agents."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agents.nodes import ResearchAgents
from src.agents.state import ResearchState


def should_continue_research(state: ResearchState) -> str:
    """Decide whether to loop back to Manager Agent for another autonomous research step."""
    gaps = state.get("knowledge_gaps", [])
    iteration = state.get("iteration", 1)
    max_iterations = state.get("max_iterations", 2)

    if gaps and iteration < max_iterations:
        return "manager"
    return "fact_checker"


def build_research_graph(agents: ResearchAgents):
    """Build and compile the multi-agent LangGraph StateGraph."""
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
    graph.add_conditional_edges(
        "analyzer",
        should_continue_research,
        {
            "manager": "manager",
            "fact_checker": "fact_checker",
        },
    )
    graph.add_edge("fact_checker", "reporter")
    graph.add_edge("reporter", END)

    return graph.compile()
