from __future__ import annotations

import re
from collections import deque
from typing import Deque, Iterable, Optional, Set
from urllib.parse import urljoin, urlparse

from src.models.db import Database
from src.lib.url import normalize_url


class UrlFrontier:
    def __init__(
        self,
        db: Database,
        *,
        allowed_patterns: list[str],
        deny_patterns: list[str],
        filter_substring: Optional[str] = None,
    ) -> None:
        self.db = db
        self.allowed_res = [re.compile(p) for p in allowed_patterns]
        self.deny_res = [re.compile(p) for p in deny_patterns]
        self.filter_substring = filter_substring
        self._queue: Deque[str] = deque()
        self._seen: Set[str] = set()

    def _is_in_scope(self, url: str) -> bool:
        if self.allowed_res and not any(r.search(url) for r in self.allowed_res):
            return False
        if self.deny_res and any(r.search(url) for r in self.deny_res):
            return False
        if self.filter_substring and self.filter_substring not in url:
            return False
        return True

    def enqueue(self, url: str, *, via_url: Optional[str] = None) -> None:
        url = normalize_url(url)
        if url in self._seen:
            return
        if not self._is_in_scope(url):
            return
        self._seen.add(url)
        self._queue.append(url)
        self.db.upsert_url(url, status="queued", via_url=via_url)

    def seed(self, start_urls: Iterable[str]) -> None:
        for u in start_urls:
            self.enqueue(u, via_url=None)

    def has_next(self) -> bool:
        return bool(self._queue)

    def next_url(self) -> Optional[str]:
        if not self._queue:
            return None
        return self._queue.popleft()

    def discover_links(self, source_url: str, links: Iterable[str]) -> None:
        for href in links:
            if href.startswith("#"):
                continue
            if href.startswith("mailto:") or href.startswith("javascript:"):
                continue
            self.enqueue(urljoin(source_url, href), via_url=source_url)
