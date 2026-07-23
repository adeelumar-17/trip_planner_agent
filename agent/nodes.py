"""
Node functions for the Trip Planner LangGraph.

Each node takes the full TripState and returns a partial dict
that gets merged back into the state by LangGraph.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from agent.tools import (
    geocode_destination,
    get_weather_forecast,
    search_accommodation,
    search_activities,
)

MAX_RETRIES = 3

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY", ""),
    temperature=0.3,
    max_tokens=4096,
)


# ---------------------------------------------------------------------------
# NODE 1: Geocode + Fetch Weather
# ---------------------------------------------------------------------------

def fetch_weather_node(state: dict) -> dict:
    destination = state["destination"]
    start_date = state["start_date"]
    end_date = state["end_date"]

    geo = geocode_destination(destination)
    if "error" in geo:
        return {
            "validation_errors": [f"Geocoding failed: {geo['error']}"],
            "tool_calls_log": [{
                "tool": "geocode_destination",
                "input": {"destination": destination},
                "result_summary": geo["error"],
            }],
        }

    lat, lon = geo["lat"], geo["lon"]

    weather = get_weather_forecast(lat, lon, start_date, end_date)
    if "error" in weather:
        return {
            "latitude": lat,
            "longitude": lon,
            "validation_errors": [f"Weather fetch failed: {weather['error']}"],
            "tool_calls_log": [
                {
                    "tool": "geocode_destination",
                    "input": {"destination": destination},
                    "result_summary": f"Resolved to {geo['display_name']} ({lat}, {lon})",
                },
                {
                    "tool": "get_weather_forecast",
                    "input": {"lat": lat, "lon": lon, "start_date": start_date, "end_date": end_date},
                    "result_summary": weather["error"],
                },
            ],
        }

    bad_days = [d for d, w in weather.items() if w["is_bad_weather"]]
    summary = f"Fetched {len(weather)} days of forecast. Bad weather on: {bad_days if bad_days else 'none'}"

    return {
        "latitude": lat,
        "longitude": lon,
        "weather_data": weather,
        "tool_calls_log": [
            {
                "tool": "geocode_destination",
                "input": {"destination": destination},
                "result_summary": f"Resolved to {geo['display_name']} ({lat}, {lon})",
            },
            {
                "tool": "get_weather_forecast",
                "input": {"lat": lat, "lon": lon, "start_date": start_date, "end_date": end_date},
                "result_summary": summary,
            },
        ],
        "decision_log": [f"📍 Geocoded '{destination}' → {geo['display_name']} ({lat:.4f}, {lon:.4f})"],
    }


# ---------------------------------------------------------------------------
# NODE 2: Search Accommodation
# ---------------------------------------------------------------------------

def search_accommodation_node(state: dict) -> dict:
    lat = state.get("latitude")
    lon = state.get("longitude")
    if lat is None or lon is None:
        return {"validation_errors": ["Cannot search accommodation: no coordinates available"]}

    destination = state["destination"]
    budget = state["budget"]
    start = datetime.strptime(state["start_date"], "%Y-%m-%d")
    end = datetime.strptime(state["end_date"], "%Y-%m-%d")
    num_days = max((end - start).days, 1)

    results = search_accommodation(lat, lon, destination, budget, num_days)
    within_budget = [r for r in results if r.get("total_estimated", 0) <= budget]
    source = results[0]["source"] if results else "none"

    return {
        "accommodation_options": within_budget if within_budget else results,
        "tool_calls_log": [{
            "tool": "search_accommodation",
            "input": {"destination": destination, "budget": budget, "num_days": num_days},
            "result_summary": f"Found {len(results)} options ({len(within_budget)} within ${budget} budget) via {source}",
        }],
        "decision_log": [
            f"🏨 Found {len(results)} accommodation options, {len(within_budget)} within budget"
            + (f" — showing all since none fit perfectly" if not within_budget else "")
        ],
    }


# ---------------------------------------------------------------------------
# NODE 3: Search Activities
# ---------------------------------------------------------------------------

def search_activities_node(state: dict) -> dict:
    lat = state.get("latitude")
    lon = state.get("longitude")
    if lat is None or lon is None:
        return {"validation_errors": ["Cannot search activities: no coordinates available"]}

    destination = state["destination"]
    interests = state.get("interests", ["sightseeing"])

    results = search_activities(lat, lon, destination, interests)
    indoor_count = sum(1 for r in results if r.get("indoor"))
    source = results[0]["source"] if results else "none"

    return {
        "activity_options": results,
        "tool_calls_log": [{
            "tool": "search_activities",
            "input": {"destination": destination, "interests": interests},
            "result_summary": f"Found {len(results)} activities ({indoor_count} indoor) via {source}",
        }],
        "decision_log": [
            f"🎯 Found {len(results)} activities matching {interests} — {indoor_count} are indoor (useful for rain days)"
        ],
    }


# ---------------------------------------------------------------------------
# NODE 4: Merge Itinerary (LLM Call)
# ---------------------------------------------------------------------------

MERGE_SYSTEM_PROMPT = """You are a travel planning assistant. Given weather forecasts, accommodation options, and activity options, create a day-by-day itinerary.

RULES:
1. Every day MUST have a weather summary, an activity, and an accommodation assignment.
2. If a day has bad weather (rain, thunderstorm, heavy snow), you MUST assign an INDOOR activity for that day. Swap any outdoor activity for an indoor one.
3. Pick ONE accommodation that fits the budget for the entire stay (same hotel/hostel each night).
4. Distribute activities across days — don't repeat the same activity.
5. If weather is nice, prefer outdoor activities.

Return ONLY valid JSON with this exact structure (no markdown, no explanation):
{
  "selected_accommodation": {"name": "...", "estimated_price_per_night": 0, "total_estimated": 0},
  "days": {
    "YYYY-MM-DD": {
      "weather": "description (temp_min°C - temp_max°C)",
      "activity": {"name": "...", "reason": "..."},
      "is_weather_swap": false
    }
  },
  "weather_swaps": ["Day X: swapped Y for Z because of rain"],
  "reasoning": "brief explanation of choices"
}"""


def merge_itinerary_node(state: dict) -> dict:
    weather_data = state.get("weather_data", {})
    accommodation = state.get("accommodation_options", [])
    activities = state.get("activity_options", [])
    budget = state.get("budget", 0)

    user_msg = f"""Destination: {state.get('destination')}
Budget: ${budget} total for accommodation
Dates: {state.get('start_date')} to {state.get('end_date')}
Interests: {state.get('interests', [])}

WEATHER DATA:
{json.dumps(weather_data, indent=2)}

ACCOMMODATION OPTIONS:
{json.dumps(accommodation, indent=2)}

ACTIVITY OPTIONS:
{json.dumps(activities, indent=2)}

Create the day-by-day itinerary now. Remember: swap outdoor activities for indoor ones on bad weather days."""

    try:
        response = llm.invoke([
            SystemMessage(content=MERGE_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])

        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        itinerary = json.loads(content)

        swaps = itinerary.get("weather_swaps", [])
        decisions = [f"🗓️ LLM merged itinerary: {itinerary.get('reasoning', 'no reasoning provided')}"]
        if swaps:
            for swap in swaps:
                decisions.append(f"🔄 Weather swap: {swap}")

        return {
            "itinerary": itinerary,
            "tool_calls_log": [{
                "tool": "llm_merge_itinerary",
                "input": {"num_days": len(weather_data), "num_activities": len(activities), "num_accommodations": len(accommodation)},
                "result_summary": f"Created itinerary with {len(itinerary.get('days', {}))} days, {len(swaps)} weather swaps",
            }],
            "decision_log": decisions,
        }
    except json.JSONDecodeError as e:
        return {
            "itinerary": {},
            "validation_errors": [f"LLM returned invalid JSON: {e}"],
            "tool_calls_log": [{
                "tool": "llm_merge_itinerary",
                "input": {},
                "result_summary": f"Failed to parse LLM response as JSON: {e}",
            }],
        }
    except Exception as e:
        return {
            "itinerary": {},
            "validation_errors": [f"LLM call failed: {e}"],
            "tool_calls_log": [{
                "tool": "llm_merge_itinerary",
                "input": {},
                "result_summary": f"LLM error: {e}",
            }],
        }


# ---------------------------------------------------------------------------
# NODE 5: Validate
# ---------------------------------------------------------------------------

def validate_node(state: dict) -> dict:
    errors = []
    itinerary = state.get("itinerary", {})
    weather_data = state.get("weather_data", {})
    budget = state.get("budget", 0)

    days = itinerary.get("days", {})
    selected = itinerary.get("selected_accommodation", {})

    if not days:
        errors.append("Itinerary has no days")

    for date_str in weather_data:
        if date_str not in days:
            errors.append(f"Missing itinerary entry for {date_str}")
        else:
            day = days[date_str]
            if not day.get("activity"):
                errors.append(f"No activity assigned for {date_str}")
            if not day.get("weather"):
                errors.append(f"No weather entry for {date_str}")

    total_cost = selected.get("total_estimated", 0)
    budget_ok = total_cost <= budget if total_cost > 0 else True

    if not budget_ok:
        errors.append(f"Accommodation total ${total_cost:.2f} exceeds budget ${budget:.2f}")

    return {
        "budget_ok": budget_ok,
        "validation_errors": errors,
        "retry_count": state.get("retry_count", 0),
        "max_retries": MAX_RETRIES,
        "decision_log": [
            f"✅ Validation {'passed' if not errors else 'failed'}: {len(errors)} error(s)"
            + (f" — budget {'OK' if budget_ok else 'EXCEEDED'} (${total_cost:.2f} / ${budget:.2f})" if total_cost > 0 else "")
        ],
    }


# ---------------------------------------------------------------------------
# NODE 6: Refine Accommodation (retry loop)
# ---------------------------------------------------------------------------

def refine_accommodation_node(state: dict) -> dict:
    retry_count = state.get("retry_count", 0) + 1
    budget = state.get("budget", 0)
    lat = state.get("latitude")
    lon = state.get("longitude")
    destination = state.get("destination", "")
    start = datetime.strptime(state["start_date"], "%Y-%m-%d")
    end = datetime.strptime(state["end_date"], "%Y-%m-%d")
    num_days = max((end - start).days, 1)

    tighter_budget = budget * (0.8 if retry_count == 1 else 0.6)

    results = search_accommodation(lat, lon, destination, tighter_budget, num_days, max_results=10)
    within_budget = [r for r in results if r.get("total_estimated", 0) <= budget]

    return {
        "accommodation_options": within_budget if within_budget else results,
        "retry_count": retry_count,
        "tool_calls_log": [{
            "tool": "search_accommodation (retry)",
            "input": {"destination": destination, "tighter_budget": tighter_budget, "retry": retry_count},
            "result_summary": f"Retry {retry_count}: searched with ${tighter_budget:.0f} budget, found {len(within_budget)}/{len(results)} within original ${budget} budget",
        }],
        "decision_log": [
            f"🔄 Budget exceeded → retry {retry_count}/{MAX_RETRIES}: re-searching accommodation with tighter filter (${tighter_budget:.0f})"
        ],
    }


# ---------------------------------------------------------------------------
# NODE 7: Generate Report
# ---------------------------------------------------------------------------

def generate_report_node(state: dict) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}.md"

    report_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, filename)

    itinerary = state.get("itinerary", {})
    weather_data = state.get("weather_data", {})
    tool_log = state.get("tool_calls_log", [])
    decision_log = state.get("decision_log", [])
    validation_errors = state.get("validation_errors", [])
    retry_count = state.get("retry_count", 0)

    days = itinerary.get("days", {})
    selected_accom = itinerary.get("selected_accommodation", {})

    # Build itinerary table
    table_rows = []
    for date_str in sorted(days.keys()):
        day = days[date_str]
        activity = day.get("activity", {})
        weather = day.get("weather", "N/A")
        swap_marker = " 🔄" if day.get("is_weather_swap") else ""
        table_rows.append(
            f"| {date_str} | {weather} | {activity.get('name', 'N/A')}{swap_marker} | {activity.get('reason', '')} |"
        )
    itinerary_table = "\n".join(table_rows) if table_rows else "| No itinerary data available |"

    # Build tool calls section
    tool_entries = []
    for i, entry in enumerate(tool_log, 1):
        tool_entries.append(
            f"{i}. **{entry.get('tool', 'unknown')}**\n"
            f"   - Input: `{json.dumps(entry.get('input', {}))}`\n"
            f"   - Result: {entry.get('result_summary', 'N/A')}"
        )
    tool_section = "\n".join(tool_entries) if tool_entries else "No tool calls recorded."

    # Build decisions section
    decision_section = "\n".join(f"- {d}" for d in decision_log) if decision_log else "No decisions logged."

    # Validation outcome
    if not validation_errors:
        validation_section = "✅ **All checks passed.** The itinerary is complete and within budget."
    else:
        items = "\n".join(f"- ⚠️ {e}" for e in validation_errors)
        validation_section = f"⚠️ **Validation completed with warnings:**\n{items}"

    report = f"""# 🌤️ Trip Planning Report

> Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 1. Run Summary

| Field | Value |
|-------|-------|
| **Destination** | {state.get('destination', 'N/A')} |
| **Dates** | {state.get('start_date', 'N/A')} → {state.get('end_date', 'N/A')} |
| **Budget** | ${state.get('budget', 0):.2f} |
| **Interests** | {', '.join(state.get('interests', []))} |

---

## 2. Tool Calls Made

{tool_section}

---

## 3. Decision Points

{decision_section}

---

## 4. Validation Outcome

{validation_section}

---

## 5. Final Itinerary

**Accommodation:** {selected_accom.get('name', 'N/A')} — ${selected_accom.get('estimated_price_per_night', 0):.2f}/night (${selected_accom.get('total_estimated', 0):.2f} total)

| Date | Weather | Activity | Reason |
|------|---------|----------|--------|
{itinerary_table}

> 🔄 = Activity was swapped due to bad weather

---

## 6. Retry Count

**{retry_count}** correction loop(s) were needed during this run.

---

## 7. Architecture Note

This trip planner is an **agentic application** built with LangGraph, a framework for building stateful, multi-step AI workflows. Unlike a simple chatbot that responds in one shot, this agent follows a structured graph of steps: it fetches real weather data, searches for accommodation and activities using external APIs, then uses an LLM to intelligently merge everything into a coherent plan. If the plan fails validation (e.g., the accommodation exceeds the budget), the agent automatically loops back to search again with tighter filters — this self-correction behavior is what makes it "agentic." Every decision the agent makes is logged, creating a transparent audit trail of its reasoning.

---

*Report generated by Weather-Aware Trip Planner Agent*
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    return {
        "report_markdown": report,
        "report_path": report_path,
        "decision_log": [f"📄 Report saved to {report_path}"],
    }
