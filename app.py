"""
Streamlit UI for the Weather-Aware Trip Planner Agent.

Features:
  - Input form for destination, dates, budget, interests
  - Live-updating status showing which node is executing
  - Final itinerary display with cards
  - Markdown report with download button
"""

import os
import uuid
import urllib.parse
import streamlit as st
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Secrets: bridge st.secrets → os.environ for Streamlit Cloud,
# fall back to .env for local development.
# ---------------------------------------------------------------------------
_SECRET_KEYS = ["GROQ_API_KEY", "GEOAPIFY_API_KEY", "TAVILY_API_KEY"]

for key in _SECRET_KEYS:
    if key not in os.environ:
        try:
            os.environ[key] = st.secrets[key]
        except (KeyError, FileNotFoundError):
            pass  # key not in secrets either — will be empty

# Local .env fallback (no-op on Streamlit Cloud where .env doesn't exist)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agent.graph import compile_graph

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🌤️ Trip Planner Agent",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }

    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        color: white;
    }

    .main-header p {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }

    .status-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }

    .status-card.active {
        background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
        border-left-color: #764ba2;
        animation: pulse 2s infinite;
    }

    .status-card.done {
        border-left-color: #27ae60;
        background: linear-gradient(135deg, #27ae6010 0%, #2ecc7110 100%);
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    .day-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        transition: transform 0.2s ease;
    }

    .day-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    }

    .day-card.rain {
        border-left-color: #e74c3c;
    }

    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }

    .metric-box {
        flex: 1;
        background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }

    .metric-box .value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #667eea;
    }

    .metric-box .label {
        font-size: 0.8rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🌤️ Weather-Aware Trip Planner</h1>
    <p>AI agent that plans your trip based on real weather forecasts, local accommodation, and your interests</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — Input form
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🗺️ Plan Your Trip")
    st.markdown("---")

    destination = st.text_input(
        "📍 Destination",
        value="London",
        placeholder="e.g. Paris, Tokyo, New York",
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "📅 Start Date",
            value=date.today() + timedelta(days=1),
            min_value=date.today(),
        )
    with col2:
        end_date = st.date_input(
            "📅 End Date",
            value=date.today() + timedelta(days=4),
            min_value=date.today() + timedelta(days=1),
        )

    budget = st.number_input(
        "💰 Total Accommodation Budget (USD)",
        min_value=50,
        max_value=10000,
        value=500,
        step=50,
    )

    interest_options = [
        "Museums", "Food", "Hiking", "Shopping", "Nightlife",
        "History", "Nature", "Art", "Sports", "Beach",
        "Architecture", "Wellness", "Family",
    ]
    interests = st.multiselect(
        "🎯 Interests",
        options=interest_options,
        default=["Museums", "Food"],
    )

    st.markdown("---")
    plan_button = st.button("🚀 Plan My Trip", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Node display names + emojis for live status
# ---------------------------------------------------------------------------
NODE_DISPLAY = {
    "fetch_weather": ("🌦️", "Fetching Weather Forecast"),
    "search_accommodation": ("🏨", "Searching Accommodation"),
    "search_activities": ("🎯", "Finding Activities"),
    "merge_itinerary": ("🧠", "AI Building Itinerary"),
    "validate": ("✅", "Validating Plan"),
    "refine_accommodation": ("🔄", "Refining Budget Search"),
    "generate_report": ("📄", "Generating Report"),
}


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
if plan_button:
    if not destination.strip():
        st.error("Please enter a destination.")
    elif end_date <= start_date:
        st.error("End date must be after start date.")
    elif not interests:
        st.error("Please select at least one interest.")
    else:
        # Prepare initial state
        initial_state = {
            "destination": destination.strip(),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "budget": float(budget),
            "interests": [i.lower() for i in interests],
            "retry_count": 0,
            "max_retries": 3,
            "tool_calls_log": [],
            "decision_log": [],
            "validation_errors": [],
        }

        # Compile graph
        app = compile_graph()
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        # Status display
        st.markdown("### ⚡ Agent Execution")
        status_container = st.container()
        completed_nodes = []

        with status_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            node_statuses = st.container()

        # Stream the graph execution step by step
        total_steps = 7
        step_count = 0
        final_state = None

        try:
            for event in app.stream(initial_state, config=config, stream_mode="updates"):
                for node_name, node_output in event.items():
                    if node_name == "__end__":
                        continue

                    step_count += 1
                    completed_nodes.append(node_name)

                    emoji, display_name = NODE_DISPLAY.get(node_name, ("⚙️", node_name))
                    progress = min(step_count / total_steps, 1.0)
                    progress_bar.progress(progress)
                    status_text.markdown(f"**{emoji} {display_name}...**")

                    with node_statuses:
                        st.markdown(
                            f'<div class="status-card done">{emoji} <strong>{display_name}</strong> — Complete ✓</div>',
                            unsafe_allow_html=True,
                        )

            progress_bar.progress(1.0)
            status_text.markdown("**✅ Planning complete!**")

            # Get final state
            final_state = app.get_state(config).values

        except Exception as e:
            st.error(f"Agent execution failed: {e}")
            st.exception(e)

        # -------------------------------------------------------------------
        # Display results
        # -------------------------------------------------------------------
        if final_state:
            st.markdown("---")

            itinerary = final_state.get("itinerary", {})
            days = itinerary.get("days", {})
            selected_accom = itinerary.get("selected_accommodation", {})
            weather_data = final_state.get("weather_data", {})

            # Metrics row
            num_days = len(days)
            total_cost = selected_accom.get("total_estimated", 0)
            swaps = len(itinerary.get("weather_swaps", []))
            retries = final_state.get("retry_count", 0)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📅 Days Planned", num_days)
            with col2:
                st.metric("💰 Est. Cost", f"${total_cost:.0f}")
            with col3:
                st.metric("🔄 Weather Swaps", swaps)
            with col4:
                st.metric("🔁 Budget Retries", retries)

            # Accommodation
            accom_name = selected_accom.get("name", "N/A")
            accom_address = selected_accom.get("address", destination)
            accom_maps_url = selected_accom.get("maps_url") or f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote_plus(f'{accom_name}, {accom_address}')}"

            st.markdown("### 🏨 Selected Accommodation")
            st.info(
                f"**[{accom_name}]({accom_maps_url})** — "
                f"${selected_accom.get('estimated_price_per_night', 0):.2f}/night "
                f"(${total_cost:.2f} total)\n\n"
                f"📍 **Address:** {accom_address} ([View on Google Maps 🗺️]({accom_maps_url}))"
            )

            # Day-by-day itinerary
            st.markdown("### 🗓️ Day-by-Day Itinerary")

            for date_str in sorted(days.keys()):
                day = days[date_str]
                activity = day.get("activity", {})
                weather_info = weather_data.get(date_str, {})
                is_bad = weather_info.get("is_bad_weather", False)
                is_swap = day.get("is_weather_swap", False)

                card_class = "day-card rain" if is_bad else "day-card"
                swap_badge = ' <span style="color: #e74c3c; font-weight: 600;">🔄 Weather Swap</span>' if is_swap else ""

                location = activity.get("location", activity.get("address", destination))
                act_cost = activity.get("estimated_cost", 0)
                cost_text = f"${act_cost:.2f}" if act_cost > 0 else "Free entry"

                act_name = activity.get("name", "N/A")
                maps_url = activity.get("maps_url") or f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote_plus(f'{act_name}, {location}')}"

                html_content = (
                    f'<div class="{card_class}">'
                    f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                    f'<h4 style="margin: 0; color: #333;">📅 {date_str}</h4>'
                    f'{swap_badge}'
                    f'</div>'
                    f'<p style="color: #555; font-size: 0.9rem; margin: 0.4rem 0;">🌡️ {day.get("weather", "N/A")}</p>'
                    f'<div style="background: #f8f9fa; padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0; border: 1px solid #e9ecef;">'
                    f'<p style="margin: 0; font-size: 1.05rem; color: #2c3e50;"><strong>🎯 <a href="{maps_url}" target="_blank" style="color: #2c3e50; text-decoration: underline;">{act_name}</a></strong></p>'
                    f'<p style="color: #666; font-size: 0.88rem; margin: 0.25rem 0;">📍 <strong>Location:</strong> {location} (<a href="{maps_url}" target="_blank" style="color: #667eea; text-decoration: underline;">View on Google Maps 🗺️</a>)</p>'
                    f'<p style="color: #27ae60; font-size: 0.88rem; margin: 0.25rem 0; font-weight: 600;">💵 <strong>Est. Activity Cost:</strong> {cost_text}</p>'
                    f'<p style="color: #7f8c8d; font-size: 0.85rem; margin-top: 0.4rem; font-style: italic;">{activity.get("reason", "")}</p>'
                    f'</div>'
                    f'</div>'
                )
                st.markdown(html_content, unsafe_allow_html=True)

            # Report
            st.markdown("---")
            st.markdown("### 📄 Agent Report")

            report_md = final_state.get("report_markdown", "")
            report_path = final_state.get("report_path", "")

            if report_md:
                with st.expander("View Full Report", expanded=False):
                    st.markdown(report_md)

                st.download_button(
                    label="⬇️ Download Report (.md)",
                    data=report_md,
                    file_name=os.path.basename(report_path) if report_path else "trip_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

            # Decision log
            decision_log = final_state.get("decision_log", [])
            if decision_log:
                with st.expander("🧠 Agent Decision Log", expanded=False):
                    for decision in decision_log:
                        st.markdown(f"- {decision}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #999; font-size: 0.8rem;">'
    'Built with LangGraph + Groq + Open-Meteo + Geoapify | Weather-Aware Trip Planner Agent'
    '</p>',
    unsafe_allow_html=True,
)
