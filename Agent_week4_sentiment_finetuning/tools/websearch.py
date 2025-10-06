# tools/websearch.py
import os
from typing import Dict, Any, List

def web_search(query: str, k: int = 5) -> Dict[str, Any]:
    """Return top-k web results {title, url, snippet}. Requires TAVILY_API_KEY."""
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return {"error": "No web backend. Set TAVILY_API_KEY or replace web_search implementation."}
    from tavily import TavilyClient
    client = TavilyClient(api_key=key)
    res = client.search(query=query, max_results=k, include_answer=False, include_raw_content=False)
    items: List[Dict[str, str]] = []
    for r in res.get("results", []):
        items.append({"title": r.get("title",""), "url": r.get("url",""), "snippet": r.get("content","")})
    return {"results": items}