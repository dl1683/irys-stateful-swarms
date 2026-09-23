"""DuckDuckGo web search for the swarm — free, no API key required.

Ported from Swarm Studio. Synchronous (workers run in ThreadPoolExecutor).
Enable via SWARM_WEB_SEARCH=1 environment variable.
"""
from __future__ import annotations

import ipaddress
import logging
import os
import socket
import time
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

MAX_WEB_SEARCHES = 5
MAX_PAGES_PER_SEARCH = 3
MAX_PAGE_CHARS = 15_000


def web_search_enabled() -> bool:
    return os.getenv("SWARM_WEB_SEARCH", "").strip() in ("1", "true", "yes")


def search_web(query: str, max_results: int = 8) -> list[dict]:
    """Search DuckDuckGo. Returns list of {title, url, snippet}."""
    try:
        raw = _ddg_search(query, max_results)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in raw
        ]
    except Exception as e:
        logger.error("DuckDuckGo search failed for %r: %s", query, e)
        return []


def fetch_page_text(url: str, max_chars: int = MAX_PAGE_CHARS) -> str:
    """Fetch a URL and extract main text content."""
    if not _is_safe_url(url):
        logger.debug("Blocked SSRF attempt: %s", url)
        return ""
    try:
        import httpx
        import trafilatura

        with httpx.Client(
            timeout=15,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; Irys/1.0)"},
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            html = resp.text

        text = trafilatura.extract(
            html, include_links=False, include_comments=False,
        ) or ""
        return text[:max_chars]
    except Exception as e:
        logger.debug("Failed to fetch %s: %s", url, e)
        return ""


def search_and_browse(query: str, *, max_pages: int = MAX_PAGES_PER_SEARCH) -> str:
    """Search DDG + fetch top page contents. Returns formatted text block."""
    results = search_web(query)
    if not results:
        return ""

    parts = [f"Search: {query}\n"]
    parts.append("SNIPPETS:")
    for r in results:
        parts.append(f"- {r['title']}: {r['snippet']}")
    parts.append("")

    urls = [r["url"] for r in results if r["url"].startswith("http")][:max_pages]
    for url in urls:
        text = fetch_page_text(url)
        if text and len(text) > 100:
            title = next((r["title"] for r in results if r["url"] == url), url)
            parts.append(f"PAGE: {title} ({url})")
            parts.append(text)
            parts.append("")

    return "\n".join(parts)


def run_web_searches(queries: list[str]) -> str:
    """Run multiple search queries and return combined results text."""
    if not queries:
        return ""
    blocks = []
    for query in queries[:MAX_WEB_SEARCHES]:
        result = search_and_browse(query)
        if result:
            blocks.append(result)
    return "\n---\n".join(blocks)


def _ddg_search(query: str, max_results: int) -> list[dict]:
    from ddgs import DDGS

    for attempt in range(3):
        try:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))
        except Exception as e:
            if attempt < 2:
                time.sleep(1 * (attempt + 1))
                logger.debug("DDG retry %d for %r: %s", attempt + 1, query, e)
            else:
                raise


def _is_safe_url(url: str) -> bool:
    """Block SSRF: reject non-HTTP schemes, unresolvable hosts, private IPs, and metadata endpoints."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        hostname = parsed.hostname or ""
    except Exception:
        return False

    if not hostname:
        return False

    blocked_hosts = {
        "localhost", "127.0.0.1", "0.0.0.0", "[::1]",
        "metadata.google.internal",
    }
    if hostname.lower() in blocked_hosts:
        return False
    if hostname.startswith("169.254."):
        return False

    # Check direct literal IP
    try:
        addr = ipaddress.ip_address(hostname)
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
            return False
        return True
    except ValueError:
        pass

    # Resolve hostname via DNS and check all resolved IP addresses
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addr_info = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
        if not addr_info:
            return False

        for item in addr_info:
            sockaddr = item[4]
            ip_str = sockaddr[0]
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
    except (socket.gaierror, socket.herror, ValueError):
        return False

    return True
