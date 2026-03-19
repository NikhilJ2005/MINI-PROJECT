"""
=============================================================================
 Web Search Module -- Serper API Integration
=============================================================================
 Provides real, verified URLs for learning resources using the Serper
 Google Search API. Gracefully degrades if SERPER_API_KEY is not set.

 Used to enrich AI-generated learning paths with actual working links
 instead of hallucinated URLs.

 Improvements:
   - Multiple result fallback (tries top 3 results)
   - Snippet enrichment (includes search result description)
   - Normalized cache keys to avoid duplicates
   - Batch-friendly with skill-level deduplication
=============================================================================
"""

import os
import re
import time
from typing import Dict, List, Optional

import requests
from loguru import logger

_api_key: Optional[str] = None
_available: Optional[bool] = None

# Simple in-memory cache: {query_key: {url, title, snippet, ts}}
_cache: Dict[str, Dict] = {}
_CACHE_TTL = 3600  # 1 hour

SERPER_URL = "https://google.serper.dev/search"

# Track searched skills to avoid duplicate API calls in one batch
_batch_skill_cache: Dict[str, Dict] = {}


def _get_api_key() -> Optional[str]:
    """Get the Serper API key from environment."""
    global _api_key, _available
    if _available is not None:
        return _api_key
    _api_key = os.environ.get("SERPER_API_KEY", "")
    if not _api_key or _api_key == "your_key_here":
        _api_key = None
        _available = False
        logger.info("[WebSearch] Serper API key not set -- live links disabled.")
    else:
        _available = True
        logger.info("[WebSearch] Serper API initialized for live resource links.")
    return _api_key


def is_available() -> bool:
    """Check if Serper web search is available."""
    _get_api_key()
    return _available or False


def _normalize_query(query: str) -> str:
    """Normalize query for consistent cache keys."""
    return re.sub(r'\s+', ' ', query.lower().strip())


def search_learning_resource(skill: str, resource_name: str = "") -> Optional[Dict]:
    """
    Search for a learning resource URL using Serper Google Search.
    Tries multiple results and returns the best one with snippet.

    Args:
        skill: The skill to find resources for (e.g., "Docker")
        resource_name: Optional specific resource name

    Returns:
        Dict with {title, url, snippet} or None
    """
    api_key = _get_api_key()
    if not api_key:
        return None

    # Build search query
    if resource_name:
        query = f"{resource_name} {skill} tutorial"
    else:
        query = f"learn {skill} tutorial free"

    # Check cache with normalized key
    cache_key = _normalize_query(query)
    if cache_key in _cache:
        entry = _cache[cache_key]
        if time.time() - entry["ts"] < _CACHE_TTL:
            return {"title": entry["title"], "url": entry["url"], "snippet": entry.get("snippet", "")}

    # Check batch-level skill cache
    skill_key = _normalize_query(skill)
    if skill_key in _batch_skill_cache:
        entry = _batch_skill_cache[skill_key]
        if time.time() - entry.get("ts", 0) < _CACHE_TTL:
            return {"title": entry["title"], "url": entry["url"], "snippet": entry.get("snippet", "")}

    try:
        response = requests.post(
            SERPER_URL,
            json={"q": query, "num": 5},
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
            },
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()

        # Get organic results -- try multiple for best match
        organic = data.get("organic", [])
        if not organic:
            return None

        # Filter out low-quality results (very short titles, suspicious domains)
        skip_domains = {"pinterest.com", "quora.com", "facebook.com"}
        best = None
        for result in organic[:5]:
            url = result.get("link", "")
            title = result.get("title", "")
            if not url or not title:
                continue
            # Skip social media and low-quality sources
            if any(d in url for d in skip_domains):
                continue
            best = result
            break

        if not best:
            best = organic[0]

        entry = {
            "title": best.get("title", ""),
            "url": best.get("link", ""),
            "snippet": best.get("snippet", ""),
            "ts": time.time(),
        }

        # Cache at both query and skill level
        _cache[cache_key] = entry
        _batch_skill_cache[skill_key] = entry

        return {"title": entry["title"], "url": entry["url"], "snippet": entry["snippet"]}

    except requests.RequestException as e:
        logger.warning(f"[WebSearch] Serper API error: {e}")
        return None


def enrich_learning_path(learning_path: List[Dict]) -> List[Dict]:
    """
    Enrich an AI-generated learning path with real, verified URLs from Serper.

    For each resource in the learning path, searches for the actual URL and
    replaces hallucinated URLs with real ones. Adds a 'verified' flag and
    includes search snippet for context.

    Args:
        learning_path: List of learning path items from Groq LLM

    Returns:
        The same list with enriched/verified resource URLs
    """
    if not is_available() or not learning_path:
        return learning_path

    # Clear batch skill cache for this run
    _batch_skill_cache.clear()

    enriched = []
    for item in learning_path:
        skill = item.get("skill", "")
        resources = item.get("resources", [])
        enriched_resources = []

        for resource in resources:
            # Handle both string and dict resources
            if isinstance(resource, str):
                resource_name = resource
                resource_dict = {"name": resource}
            elif isinstance(resource, dict):
                resource_name = resource.get("name", "")
                resource_dict = dict(resource)
            else:
                enriched_resources.append(resource)
                continue

            # Search for the real URL
            search_result = search_learning_resource(skill, resource_name)
            if search_result and search_result.get("url"):
                resource_dict["url"] = search_result["url"]
                resource_dict["verified"] = True
                # Add snippet for context
                if search_result.get("snippet"):
                    resource_dict["description"] = search_result["snippet"]
                if not resource_dict.get("name"):
                    resource_dict["name"] = search_result.get("title", resource_name)
            else:
                resource_dict["verified"] = False

            enriched_resources.append(resource_dict)

            # Small delay between searches to be respectful
            time.sleep(0.15)

        item_copy = dict(item)
        item_copy["resources"] = enriched_resources
        enriched.append(item_copy)

    verified_count = sum(
        1 for item in enriched
        for r in item.get("resources", [])
        if isinstance(r, dict) and r.get("verified")
    )
    total_resources = sum(
        len(item.get("resources", []))
        for item in enriched
    )
    logger.info(f"[WebSearch] Enriched learning path: {verified_count}/{total_resources} verified links")

    return enriched
