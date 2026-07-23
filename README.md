# Weather-Aware Trip & Accommodation Planner Agent

An agentic trip planner powered by **LangGraph** that fetches real weather forecasts, searches for accommodation and activities, and builds a day-by-day itinerary — with self-correction loops for budget overruns and bad weather.

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API keys
cp .env.example .env
# Edit .env and add your keys (at minimum GROQ_API_KEY + GEOAPIFY_API_KEY)

# 4. Run the app
streamlit run app.py
```

## Required API Keys

| Key | Required? | Purpose | Get it at |
|-----|-----------|---------|-----------|
| `GROQ_API_KEY` | **Yes** | LLM inference (LLaMA 3.3-70B) | [console.groq.com](https://console.groq.com/keys) |
| `GEOAPIFY_API_KEY` | **Yes** | Accommodation & activity search (Places API) | [myprojects.geoapify.com](https://myprojects.geoapify.com/) |
| `TAVILY_API_KEY` | Fallback | Web search (used if Geoapify data is thin) | [app.tavily.com](https://app.tavily.com/home) |

> **Weather** uses Open-Meteo (free, no key required).

## Architecture

```
START → fetch_weather → search_accommodation → search_activities
      → merge_itinerary (LLM) → validate
          → [pass] → generate_report → END
          → [fail + retries left] → refine_accommodation → merge (loop)
          → [fail + no retries] → generate_report (with warnings) → END
```

## Project Structure

```
trip_planner_agent/
├── .streamlit/
│   ├── config.toml             # Theme & server config
│   └── secrets.toml.example    # Template for Streamlit Cloud secrets
├── agent/
│   ├── state.py      # Typed state schema
│   ├── nodes.py      # Node functions (fetch, search, merge, validate, refine, report)
│   ├── graph.py      # LangGraph StateGraph assembly + conditional edges
│   └── tools.py      # API wrappers (Open-Meteo, Geoapify Places, Tavily)
├── app.py            # Streamlit frontend
├── requirements.txt
├── .env.example      # Local development keys template
├── .gitignore
└── README.md
```

## ☁️ Deploy to Streamlit Cloud

1. **Push this project to a GitHub repository** (public or private).

2. **Go to [share.streamlit.io](https://share.streamlit.io/)** and sign in with GitHub.

3. **Click "New app"** and select:
   - **Repository**: your repo
   - **Branch**: `main`
   - **Main file path**: `app.py`

4. **Add your API keys** in the app's **Settings → Secrets** panel. Paste:
   ```toml
   GROQ_API_KEY = "gsk_..."
   GEOAPIFY_API_KEY = "..."
   TAVILY_API_KEY = "tvly-..."
   ```

5. **Click Deploy.** The app will install dependencies from `requirements.txt` and start automatically.

> See [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) for the template.
