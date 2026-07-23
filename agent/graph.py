"""
LangGraph StateGraph assembly for the Trip Planner Agent.

Flow:
  START → fetch_weather → search_accommodation → search_activities
        → merge_itinerary → validate
            → [budget_ok] → generate_report → END
            → [retry_count < max] → refine_accommodation → merge_itinerary (loop)
            → [retry_count >= max] → generate_report (with warnings) → END
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import TripState
from agent.nodes import (
    fetch_weather_node,
    search_accommodation_node,
    search_activities_node,
    merge_itinerary_node,
    validate_node,
    refine_accommodation_node,
    generate_report_node,
    MAX_RETRIES,
)


def route_after_validation(state: dict) -> str:
    """Conditional router after the validate node."""
    if state.get("budget_ok", True):
        return "generate_report"

    retry_count = state.get("retry_count", 0)
    if retry_count < state.get("max_retries", MAX_RETRIES):
        return "refine_accommodation"

    return "generate_report"


def build_graph() -> StateGraph:
    """Assemble and compile the trip planner graph."""
    graph = StateGraph(TripState)

    # Add nodes
    graph.add_node("fetch_weather", fetch_weather_node)
    graph.add_node("search_accommodation", search_accommodation_node)
    graph.add_node("search_activities", search_activities_node)
    graph.add_node("merge_itinerary", merge_itinerary_node)
    graph.add_node("validate", validate_node)
    graph.add_node("refine_accommodation", refine_accommodation_node)
    graph.add_node("generate_report", generate_report_node)

    # Linear edges
    graph.set_entry_point("fetch_weather")
    graph.add_edge("fetch_weather", "search_accommodation")
    graph.add_edge("search_accommodation", "search_activities")
    graph.add_edge("search_activities", "merge_itinerary")
    graph.add_edge("merge_itinerary", "validate")

    # Conditional edge after validation
    graph.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "generate_report": "generate_report",
            "refine_accommodation": "refine_accommodation",
        },
    )

    # Retry loop: refine → re-merge
    graph.add_edge("refine_accommodation", "merge_itinerary")

    # Terminal
    graph.add_edge("generate_report", END)

    return graph


def compile_graph(checkpointer=None):
    """Build and compile the graph with optional checkpointing."""
    graph = build_graph()
    if checkpointer is None:
        checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
