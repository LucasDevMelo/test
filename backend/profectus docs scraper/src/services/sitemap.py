from __future__ import annotations

import re
from typing import Iterable, List
from urllib.parse import urljoin, urlparse

import httpx


SITEMAP_PATHS = ["/sitemap.xml", "/sitemap_index.xml"]


def _extract_urls_from_sitemap_xml(xml_text: str) -> List[str]:
    # Simple regex-based extract; robust parsing can be added later
    # Matches <loc>https://example.com/path</loc>
    return re.findall(r"<loc>\s*([^<]+?)\s*</loc>", xml_text, flags=re.IGNORECASE)


def fetch_sitemap_seeds(base_url: str, *, timeout: float = 10.0) -> Iterable[str]:
    parsed = urlparse(base_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        for path in SITEMAP_PATHS:
            url = urljoin(origin, path)
            try:
                resp = client.get(url)
                if resp.status_code == 200 and "xml" in (resp.headers.get("content-type") or ""):
                    for u in _extract_urls_from_sitemap_xml(resp.text):
                        yield u.strip()
            except Exception:
                continue
