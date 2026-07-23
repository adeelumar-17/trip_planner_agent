from __future__ import annotations
from typing import TypedDict, Annotated
from operator import add


class TripState(TypedDict, total=False):
    """Full state object flowing through the LangGraph trip planner."""

    # --- User inputs ---
    destination: str
    start_date: str          # ISO format: "2025-07-20"
    end_date: str            # ISO format: "2025-07-25"
    budget: float            # Total accommodation budget in USD
    interests: list[str]     # e.g. ["hiking", "museums", "food"]

    # --- Geocoding (resolved from destination) ---
    latitude: float
    longitude: float

    # --- Tool outputs ---
    weather_data: dict                  # {date_str: {temp, description, code, ...}}
    accommodation_options: list[dict]   # [{name, price_per_night, address, ...}]
    activity_options: list[dict]        # [{name, category, address, indoor, ...}]

    # --- LLM-generated plan ---
    itinerary: dict                     # {date_str: {weather, activity, accommodation}}

    # --- Validation / control flow ---
    budget_ok: bool
    retry_count: int                    # capped at MAX_RETRIES (3)
    max_retries: int                    # default 3
    validation_errors: list[str]

    # --- Agent trace (for report generation) ---
    tool_calls_log: Annotated[list[dict], add]  # append-only log of every tool call
    decision_log: Annotated[list[str], add]     # plain-language decisions made by the agent

    # --- Output ---
    report_markdown: str                # final generated Markdown report
    report_path: str                    # filesystem path to saved .md file
