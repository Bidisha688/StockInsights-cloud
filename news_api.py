import os
import time
import requests
from typing import Iterable, Optional
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv(override=True)
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")


def _request_with_retries(url: str, headers=None, max_retries: int = 3, timeout: int = 15):
    """GET with simple exponential backoff for transient errors."""
    exc = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, headers=headers or {}, timeout=timeout)
            # Retry on common transient codes
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * attempt)
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            exc = e
            time.sleep(1.0 * attempt)
    if exc:
        raise exc


def _csv(items: Optional[Iterable[str]]) -> Optional[str]:
    if not items:
        return None
    cleaned = [str(x).strip() for x in items if str(x).strip()]
    return ",".join(cleaned) if cleaned else None


def fetch_news(
    query: str,
    days: int = 7,
    page_size: int = 50,
    *,
    strict_in_title: bool = False,               # True -> use qInTitle=<query> (tighter results)
    language: str = "en",
    sort_by: str = "publishedAt",                # relevancy | popularity | publishedAt
    include_sources: Optional[Iterable[str]] = None,  # e.g., ["bloomberg", "the-verge"]
    exclude_domains: Optional[Iterable[str]] = None,  # e.g., ["example.com", "foo.org"]
) -> list[dict]:
    """
    Fetch recent news via NewsAPI.

    Args:
        query: search term (company/ticker)
        days: lookback window
        page_size: 1..100 (NewsAPI max)
        strict_in_title: if True, use qInTitle= to require the term in the headline
        language: article language (default 'en')
        sort_by: relevancy | popularity | publishedAt
        include_sources: optional whitelist of sources (NewsAPI 'sources' param, comma-separated IDs)
        exclude_domains: optional comma-separated domain filter to exclude

    Returns:
        List of normalized article dicts with fields:
        title, url, publishedAt, source, content, description
    """
    if not NEWSAPI_KEY:
        raise RuntimeError("Missing NEWSAPI_KEY in environment (.env or Streamlit secrets).")

    q = (query or "").strip()
    if not q:
        return []

    page_size = max(1, min(int(page_size), 100))  # clamp to API limits
    from_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

    # Build query params
    q_param = f"qInTitle={quote_plus(q)}" if strict_in_title else f"q={quote_plus(q)}"
    src_param = f"&sources={_csv(include_sources)}" if _csv(include_sources) else ""
    excl_param = f"&excludeDomains={_csv(exclude_domains)}" if _csv(exclude_domains) else ""

    url = (
        "https://newsapi.org/v2/everything?"
        f"{q_param}&"
        f"from={from_date}&"
        f"sortBy={quote_plus(sort_by)}&"
        f"language={quote_plus(language)}&"
        f"pageSize={page_size}"
        f"{src_param}"
        f"{excl_param}"
    )

    headers = {"X-Api-Key": NEWSAPI_KEY}

    r = _request_with_retries(url, headers=headers, max_retries=3, timeout=20)
    data = r.json()
    articles = data.get("articles", []) or []

    # Normalize fields we use elsewhere
    return [
        {
            "title": a.get("title"),
            "url": a.get("url"),
            "publishedAt": a.get("publishedAt"),
            "source": (a.get("source") or {}).get("name"),
            "content": a.get("content") or "",
            "description": a.get("description") or "",
        }
        for a in articles
    ]
