"""
Tool wrappers for the Trip Planner Agent.

Three data sources:
  1. Open-Meteo (weather) — free, no key
  2. Geoapify Places API (accommodation + activities) — free tier, key required
  3. Tavily web search (fallback) — optional key

Each function is self-contained and unit-testable outside the graph.
"""

from __future__ import annotations

import os
import requests
from datetime import datetime, timedelta

def _get_geoapify_key() -> str:
    return os.getenv("GEOAPIFY_API_KEY", "")

def _get_tavily_key() -> str:
    return os.getenv("TAVILY_API_KEY", "")

TIMEOUT = 15  # seconds for all HTTP calls

# ---------------------------------------------------------------------------
# 1. GEOCODING — resolve a destination name to lat/lon via Geoapify
# ---------------------------------------------------------------------------

def geocode_destination(destination: str) -> dict:
    """Return {lat, lon, display_name} for a destination string.

    Uses Geoapify Geocoding API (free tier).
    """
    url = "https://api.geoapify.com/v1/geocode/search"
    params = {
        "text": destination,
        "limit": 1,
        "apiKey": _get_geoapify_key(),
    }
    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        features = resp.json().get("features", [])
        if not features:
            return {"error": f"Could not geocode '{destination}'"}
        props = features[0]["properties"]
        return {
            "lat": props["lat"],
            "lon": props["lon"],
            "display_name": props.get("formatted", destination),
        }
    except requests.RequestException as exc:
        return {"error": f"Geocoding failed: {exc}"}


# ---------------------------------------------------------------------------
# 2. WEATHER — Open-Meteo (free, no key)
# ---------------------------------------------------------------------------

WMO_DESCRIPTIONS = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snowfall", 73: "Moderate snowfall", 75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}

BAD_WEATHER_CODES = {
    45, 48, 55, 56, 57, 63, 65, 66, 67, 73, 75, 77,
    81, 82, 85, 86, 95, 96, 99,
}


def get_weather_forecast(
    lat: float, lon: float, start_date: str, end_date: str
) -> dict:
    """Fetch daily weather from Open-Meteo.

    Returns: {
        "2025-07-20": {
            "temp_max": 28.3,
            "temp_min": 18.1,
            "weather_code": 3,
            "description": "Overcast",
            "is_bad_weather": False,
            "precipitation_sum": 0.0,
        }, ...
    }
    On failure returns {"error": "..."}.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,weathercode,precipitation_sum",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "auto",
    }
    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json().get("daily", {})
        dates = data.get("time", [])
        if not dates:
            return {"error": "Open-Meteo returned no forecast data"}

        result = {}
        for i, date_str in enumerate(dates):
            code = data["weathercode"][i]
            result[date_str] = {
                "temp_max": data["temperature_2m_max"][i],
                "temp_min": data["temperature_2m_min"][i],
                "weather_code": code,
                "description": WMO_DESCRIPTIONS.get(code, f"Code {code}"),
                "is_bad_weather": code in BAD_WEATHER_CODES,
                "precipitation_sum": data["precipitation_sum"][i],
            }
        return result
    except requests.RequestException as exc:
        return {"error": f"Weather fetch failed: {exc}"}


# ---------------------------------------------------------------------------
# 3. ACCOMMODATION — Geoapify Places + Tavily pricing enrichment
# ---------------------------------------------------------------------------

ACCOMMODATION_CATEGORIES = "accommodation.hotel,accommodation.hostel,accommodation.guest_house,accommodation.motel,accommodation.apartment"

def search_accommodation(
    lat: float, lon: float, destination: str,
    budget: float, num_days: int, max_results: int = 8,
) -> list[dict]:
    """Search for accommodation near the destination.

    Strategy:
      1. Discover places via Geoapify (names + addresses)
      2. Enrich with real pricing via Tavily web search
      3. Fall back to pure Tavily search if Geoapify returns nothing
    """
    results = _geoapify_places(lat, lon, ACCOMMODATION_CATEGORIES, max_results)
    per_night_budget = budget / max(num_days, 1)

    if results:
        # Enrich Geoapify results with Tavily pricing
        enriched = _enrich_with_tavily_prices(results, destination, num_days, per_night_budget)
        return enriched

    return _tavily_accommodation_fallback(destination, budget, num_days, max_results)


def _enrich_with_tavily_prices(
    places: list[dict], destination: str, num_days: int, per_night_budget: float
) -> list[dict]:
    """Use Tavily to look up real pricing for Geoapify-discovered places."""
    place_names = [p.get("name", "hotel") for p in places[:6]]  # cap to avoid too many API calls
    query = (
        f"hotels accommodation pricing per night in {destination}: "
        + ", ".join(place_names)
        + f". Average nightly rate in USD."
    )
    pricing_results = _tavily_search(query, max_results=5)

    # Build a pricing lookup from Tavily content
    pricing_snippets = " ".join(
        r.get("content", "") + " " + r.get("title", "")
        for r in pricing_results
    ).lower()

    enriched = []
    for place in places:
        name = place.get("name", "Unnamed")
        price = _extract_price_for_place(name, pricing_snippets, per_night_budget, place)

        enriched.append({
            "name": name,
            "address": place.get("address", ""),
            "category": place.get("category", "accommodation"),
            "estimated_price_per_night": price,
            "total_estimated": round(price * num_days, 2),
            "source": "geoapify+tavily" if pricing_results else "geoapify",
        })
    return enriched


def _extract_price_for_place(
    name: str, pricing_text: str, per_night_budget: float, place: dict
) -> float:
    """Try to extract a real price from Tavily snippets for a specific place.

    Searches for the place name in the pricing text and looks for nearby
    dollar amounts. Falls back to category-based estimation.
    """
    import re

    name_lower = name.lower()
    # Try to find price near the place name in the text
    name_words = name_lower.split()
    for word in name_words:
        if len(word) < 3:
            continue
        # Look for patterns like "$120", "120 usd", "$120/night", "from $85"
        pattern = rf'{re.escape(word)}[^$]*?\$(\d+(?:\.\d{{2}})?)'
        match = re.search(pattern, pricing_text)
        if match:
            return round(float(match.group(1)), 2)

        # Also try reverse: price then name
        pattern_rev = rf'\$(\d+(?:\.\d{{2}})?)[^.]*?{re.escape(word)}'
        match_rev = re.search(pattern_rev, pricing_text)
        if match_rev:
            return round(float(match_rev.group(1)), 2)

    # Try to extract any general price range from the text for this destination
    all_prices = re.findall(r'\$(\d+(?:\.\d{2})?)', pricing_text)
    if all_prices:
        prices = [float(p) for p in all_prices if 20 < float(p) < 1000]
        if prices:
            # Use the median as a reasonable estimate
            prices.sort()
            median_price = prices[len(prices) // 2]
            # Apply category multiplier for differentiation
            category = place.get("category", "")
            multipliers = {
                "accommodation.hostel": 0.5,
                "accommodation.guest_house": 0.7,
                "accommodation.motel": 0.75,
                "accommodation.apartment": 0.9,
                "accommodation.hotel": 1.0,
            }
            mult = multipliers.get(category, 0.85)
            return round(median_price * mult, 2)

    # Final fallback: category-based heuristic
    return _estimate_accommodation_price(place, per_night_budget)


def _estimate_accommodation_price(place: dict, per_night_budget: float) -> float:
    """Last-resort heuristic price estimate based on accommodation category."""
    category = place.get("category", "")
    price_ratios = {
        "accommodation.hostel": 0.35,
        "accommodation.guest_house": 0.55,
        "accommodation.motel": 0.60,
        "accommodation.apartment": 0.75,
        "accommodation.hotel": 0.90,
    }
    ratio = price_ratios.get(category, 0.70)
    base = per_night_budget * ratio
    variance = (hash(place.get("name", "")) % 30 - 15)
    return round(max(20, base + variance), 2)


# ---------------------------------------------------------------------------
# 4. ACTIVITIES — Per-interest search for variety (Geoapify + Tavily)
# ---------------------------------------------------------------------------

INTEREST_TO_CATEGORIES = {
    "hiking": "natural,natural.forest,leisure.park",
    "museums": "entertainment.museum,entertainment.culture,entertainment.gallery,building.historic,tourism.sights",
    "food": "catering.restaurant,catering.cafe,catering.fast_food",
    "shopping": "commercial.shopping_mall,commercial.marketplace,commercial",
    "nightlife": "entertainment.nightclub,catering.bar,catering.pub",
    "history": "tourism.sights,heritage,building.historic",
    "nature": "natural,natural.water,leisure.park,natural.forest",
    "art": "entertainment.museum,entertainment.culture,entertainment.gallery",
    "sports": "sport,leisure.fitness_centre,activity.sport_club",
    "beach": "beach,natural.water,leisure.swimming_pool",
    "architecture": "tourism.sights,building.historic,building.place_of_worship",
    "wellness": "service.beauty,leisure.spa",
    "family": "entertainment.theme_park,entertainment.zoo,entertainment.aquarium",
}

INDOOR_CATEGORIES = {
    "entertainment.museum", "entertainment.culture", "entertainment.gallery",
    "commercial.shopping_mall", "catering.restaurant", "catering.cafe",
    "catering.fast_food", "catering.bar", "catering.pub", "catering",
    "entertainment.cinema", "leisure.spa", "leisure.fitness_centre",
    "service.beauty", "entertainment.aquarium", "building.historic", "heritage",
}


def search_activities(
    lat: float, lon: float, destination: str,
    interests: list[str], max_results: int = 10,
) -> list[dict]:
    """Search for activities matching user interests.

    Searches each interest SEPARATELY via Geoapify to guarantee variety,
    then supplements with Tavily for any interest with thin results.
    Each result includes an `indoor` boolean for weather-swap logic
    and an `interest` label showing which interest it belongs to.
    """
    per_interest = max(max_results // max(len(interests), 1), 3)
    all_results: list[dict] = []
    seen_names: set[str] = set()

    for interest in interests:
        key = interest.lower().strip()
        categories = INTEREST_TO_CATEGORIES.get(
            key, "tourism.sights,entertainment,leisure.park"
        )

        # Search Geoapify for this specific interest
        geo_results = _geoapify_places(lat, lon, categories, per_interest)

        interest_results = []
        for place in geo_results:
            name = place.get("name", "Unnamed")
            if name.lower() in seen_names or name == "Unnamed":
                continue
            seen_names.add(name.lower())
            cat = place.get("category", "")
            all_cats = place.get("all_categories", [cat])
            interest_results.append({
                "name": name,
                "address": place.get("address", ""),
                "category": cat,
                "interest": key,
                "indoor": _is_indoor(all_cats),
                "source": "geoapify",
            })

        # If Geoapify returned < 2 for this interest, supplement with Tavily
        if len(interest_results) < 2:
            tavily_extras = _tavily_interest_search(destination, key, per_interest)
            for item in tavily_extras:
                if item["name"].lower() not in seen_names:
                    seen_names.add(item["name"].lower())
                    item["interest"] = key
                    interest_results.append(item)

        all_results.extend(interest_results)

    # If we still have very few results overall, do a broad Tavily fallback
    if len(all_results) < 3:
        fallback = _tavily_activities_fallback(destination, interests, max_results)
        for item in fallback:
            if item["name"].lower() not in seen_names:
                seen_names.add(item["name"].lower())
                all_results.append(item)

    return all_results[:max_results]


def _is_indoor(categories: list[str] | str) -> bool:
    """Check if category string or list indicates an indoor venue."""
    if isinstance(categories, str):
        categories = [categories]
    for cat in categories:
        if any(ic in cat for ic in INDOOR_CATEGORIES):
            return True
    return False


def _tavily_interest_search(
    destination: str, interest: str, max_results: int = 5
) -> list[dict]:
    """Use Tavily to find activities for a specific interest."""
    query = f"top places to visit for {interest} in {destination}"
    results = _tavily_search(query, max_results)

    junk_words = {"definition", "meaning", "forum", "shop now", "buy", "official store", "dictionary", "wikipedia"}
    indoor_keywords = {
        "museum", "gallery", "mall", "restaurant", "cafe", "cinema",
        "spa", "aquarium", "indoor", "theater", "theatre", "bar",
    }
    outdoor_keywords = {
        "park", "trail", "hike", "beach", "garden", "mountain",
        "lake", "river", "outdoor", "walk", "nature",
    }

    enriched = []
    for r in results:
        title = r.get("title", "Activity").strip()
        title_lower = title.lower()
        if any(jw in title_lower for jw in junk_words):
            continue

        is_indoor = any(kw in title_lower for kw in indoor_keywords)
        is_outdoor = any(kw in title_lower for kw in outdoor_keywords)
        if not is_indoor and not is_outdoor:
            is_indoor = interest in {"museums", "food", "shopping", "nightlife", "art", "wellness"}

        enriched.append({
            "name": title,
            "address": destination,
            "category": f"tavily.{interest}",
            "indoor": is_indoor,
            "source": "tavily",
            "url": r.get("url", ""),
        })
    return enriched


# ---------------------------------------------------------------------------
# GEOAPIFY SHARED HELPER
# ---------------------------------------------------------------------------

def _geoapify_places(
    lat: float, lon: float, categories: str, limit: int = 10
) -> list[dict]:
    """Low-level call to Geoapify Places API v2 with circle filter."""
    if not _get_geoapify_key():
        return []

    url = "https://api.geoapify.com/v2/places"
    params = {
        "categories": categories,
        "filter": f"circle:{lon},{lat},10000",  # 10 km radius
        "limit": limit,
        "apiKey": _get_geoapify_key(),
    }
    headers = {"Accept": "application/json"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
        resp.raise_for_status()
        features = resp.json().get("features", [])
        places = []
        for feat in features:
            props = feat.get("properties", {})
            cats = props.get("categories", [])
            places.append({
                "name": props.get("name", props.get("address_line1", "Unnamed")),
                "address": props.get("formatted", props.get("address_line1", "")),
                "category": cats[0] if cats else "",
                "all_categories": cats,
                "lat": props.get("lat"),
                "lon": props.get("lon"),
            })
        return places
    except requests.RequestException:
        return []


# ---------------------------------------------------------------------------
# TAVILY FALLBACK HELPERS
# ---------------------------------------------------------------------------

def _tavily_accommodation_fallback(
    destination: str, budget: float, num_days: int, max_results: int
) -> list[dict]:
    """Use Tavily web search as a fallback for accommodation with real pricing."""
    import re
    per_night = budget / max(num_days, 1)
    results = _tavily_search(
        f"best hotels hostels accommodation in {destination} price per night USD under ${per_night:.0f}",
        max_results,
    )
    enriched = []
    for i, r in enumerate(results):
        title = r.get("title", f"Option {i+1}")
        content = (r.get("content", "") + " " + title).lower()

        # Try to extract a real price from the snippet
        price_matches = re.findall(r'\$(\d+(?:\.\d{2})?)', content)
        valid_prices = [float(p) for p in price_matches if 20 < float(p) < 1000]

        if valid_prices:
            price = min(valid_prices)  # take the lowest (likely the starting rate)
        else:
            price = round(per_night * (0.5 + (i * 0.1)), 2)

        enriched.append({
            "name": title,
            "address": destination,
            "category": "accommodation",
            "estimated_price_per_night": price,
            "total_estimated": round(price * num_days, 2),
            "source": "tavily",
            "url": r.get("url", ""),
        })
    return enriched


def _tavily_activities_fallback(
    destination: str, interests: list[str], max_results: int
) -> list[dict]:
    """Use Tavily web search as a fallback for activities."""
    interests_str = ", ".join(interests) if interests else "sightseeing"
    results = _tavily_search(
        f"top things to do in {destination}: {interests_str}",
        max_results,
    )
    enriched = []
    indoor_keywords = {"museum", "gallery", "mall", "restaurant", "cafe", "cinema", "spa", "aquarium", "indoor"}
    for r in results:
        title = r.get("title", "Activity")
        is_indoor = any(kw in title.lower() for kw in indoor_keywords)
        enriched.append({
            "name": title,
            "address": destination,
            "category": "activity",
            "indoor": is_indoor,
            "source": "tavily",
            "url": r.get("url", ""),
        })
    return enriched


def _tavily_search(query: str, max_results: int = 5) -> list[dict]:
    """Execute a Tavily search. Returns [] on failure or missing key."""
    if not _get_tavily_key():
        return []
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=_get_tavily_key())
        response = client.search(query=query, max_results=max_results)
        return response.get("results", [])
    except Exception:
        return []
