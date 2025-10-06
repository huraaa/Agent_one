import requests
from bs4 import BeautifulSoup
from pathlib import Path

def search_web(query: str, max_results: int = 5):
    # Simple starter: Bing/Web fallback via DuckDuckGo HTML (no API key).
    # Replace with Tavily/SerpAPI for reliability.
    r = requests.get("https://duckduckgo.com/html/", params={"q": query}, timeout=20)
    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    for a in soup.select(".result__title a")[:max_results]:
        href = a.get("href")
        title = a.get_text(" ", strip=True)
        results.append({"title": title, "url": href})
    return results

def fetch_url(url: str, max_chars: int = 8000):
    r = requests.get(url, timeout=30, headers={"User-Agent": "agent/0.1"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    # crude readability
    for tag in soup(["script","style","nav","footer","header","noscript"]):
        tag.decompose()
    text = " ".join(soup.get_text(" ", strip=True).split())
    return text[:max_chars]

def write_file(path: str, content: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} chars to {p}"

def read_file(path: str, max_chars: int = 8000):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    return text[:max_chars]
