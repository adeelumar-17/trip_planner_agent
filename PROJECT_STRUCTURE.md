# 🌤️ Weather-Aware Trip Planner Agent: Architecture & Structure

This document provides a deep dive into the modular architecture and inner workings of the Trip Planner Agent built with LangGraph, Groq, and Streamlit.

## 🏗️ 1. Project Directory Structure

The project is highly modular, separating concerns into distinct files to ensure maintainability, testability, and clarity.

```text
trip_planner_agent/
├── agent/
│   ├── __init__.py      # Package initialization
│   ├── state.py         # Defines the typed schema representing the agent's memory
│   ├── tools.py         # Independent API wrappers (Weather, Geoapify, Tavily)
│   ├── nodes.py         # Discrete operations (fetching data, LLM planning, validation)
│   └── graph.py         # LangGraph StateGraph assembly and conditional routing
├── app.py               # Streamlit frontend UI for user interaction and status display
├── test_e2e.py          # End-to-end headless testing script
├── test_pricing.py      # Quick test for Tavily pricing enrichment
├── .env                 # Environment variables (API keys)
└── requirements.txt     # Project dependencies
```

---

## ⚙️ 2. Core Concepts & LangGraph Working

This application is built using **LangGraph**, treating the workflow as a state machine. Rather than a single monolithic LLM prompt, the process is divided into functional nodes that read and write to a shared memory object (the "State").

### The State (`agent/state.py`)
The `TripState` is a `TypedDict` that stores everything the agent knows at any given time. As execution flows through the nodes, they incrementally update this state.
- **Inputs**: Destination, dates, budget, interests.
- **Data Collections**: Weather forecasts, accommodation options, activity options.
- **Agent Output**: The final day-by-day itinerary.
- **Audit Logs**: Append-only arrays (`tool_calls_log`, `decision_log`) that keep track of what the agent did.

### The Tools (`agent/tools.py`)
Tools are pure Python functions that handle external API communication. They are decoupled from the agent logic, making them easy to unit-test.
1. **Geocoding**: Converts a city name into `latitude` & `longitude` via Geoapify.
2. **Weather (Open-Meteo)**: Fetches historical/forecasted weather and flags "bad weather" (rain, snow, thunderstorms).
3. **Accommodation (Geoapify + Tavily)**: Finds local hotels/hostels via Geoapify, then uses Tavily web search to scrape and extract real nightly prices in USD.
4. **Activities (Geoapify + Tavily)**: Finds local activities matching user interests, flagging them as "indoor" or "outdoor".

### The Nodes (`agent/nodes.py`)
Nodes are the actual steps in the graph. Each node takes the `TripState`, performs an action, and returns a dictionary of updates to merge back into the state.
- `fetch_weather_node`: Calls the geocoding and weather tools.
- `search_accommodation_node`: Uses coordinates to find housing options within the budget.
- `search_activities_node`: Uses coordinates to find activities.
- `merge_itinerary_node`: The core LLM call. It provides the collected data to the LLaMA model via Groq, instructing it to build a valid day-by-day JSON itinerary. It enforces the **weather-swap logic** (scheduling indoor activities if bad weather is detected).
- `validate_node`: Programmatic checks. Does every day have an activity? Is the total accommodation cost strictly under the user's budget?
- `refine_accommodation_node`: If validation fails (budget exceeded), this node acts as a self-correction step. It tightens the search parameters to force cheaper options.
- `generate_report_node`: Compiles the final state, logs, and itinerary into a detailed Markdown file for the user.

### The Graph (`agent/graph.py`)
This file wires the nodes together into a directed graph and defines the routing logic.

```mermaid
graph TD
    Start((START)) --> FetchWeather[fetch_weather]
    FetchWeather --> SearchAcc[search_accommodation]
    SearchAcc --> SearchAct[search_activities]
    SearchAct --> MergeItin[merge_itinerary (LLM)]
    MergeItin --> Validate[validate]
    
    Validate -->|budget_ok == True| GenReport[generate_report]
    Validate -->|budget_ok == False<br>AND retries < max| RefineAcc[refine_accommodation]
    
    RefineAcc --> MergeItin
    
    GenReport --> End((END))
```

- **Linear Flow**: It starts by sequentially gathering data (Weather → Accommodation → Activities) and then merging them.
- **Conditional Routing**: After `validate`, the graph makes a decision. If the budget is broken, it loops back to `refine_accommodation` and re-runs the LLM merge. If it passes (or max retries are hit), it proceeds to the end.

---

## 💻 3. The Frontend (`app.py`)

The Streamlit UI provides a visual window into the agent's "mind". 
- It captures user inputs and triggers the LangGraph execution.
- Using `app.stream()`, it listens to updates from the graph in real-time, allowing it to update the UI progress bar and show the user exactly which node is currently executing.
- Finally, it parses the output state to render beautiful CSS cards for the day-by-day itinerary and provides the generated Markdown report as a downloadable file.

## 🚀 4. Why This is "Agentic"

This project demonstrates true agentic design rather than simple conversational AI:
1. **Tool Use**: The LLM isn't hallucinating data; deterministic Python functions fetch real-world, real-time data first, and the LLM acts as a synthesizer.
2. **Self-Correction**: The `validate` -> `refine` loop allows the system to realize it made a mistake (e.g., picking a hotel that is too expensive) and autonomously correct itself without human intervention.
3. **Structured Output**: The LLM is forced to output strict JSON schemas, allowing the programmatic parts of the pipeline to reliably process its decisions.
