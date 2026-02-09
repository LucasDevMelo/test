from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urlparse

import httpx


@dataclass
class HttpResponse:
    url: str
    status_code: int
    content_type: Optional[str]
    text: str
    content_bytes: bytes


class HttpClient:
    def __init__(self, user_agent: str, rate_limit_per_host_per_sec: float) -> None:
        self.user_agent = user_agent
        self.rate_limit = max(0.0, float(rate_limit_per_host_per_sec))
        self._last_request_ts_by_host: Dict[str, float] = {}
        self._client = httpx.Client(follow_redirects=True, timeout=15.0, headers={"User-Agent": self.user_agent})

    def _respect_rate_limit(self, host: str) -> None:
        if self.rate_limit <= 0:
            return
        min_interval = 1.0 / self.rate_limit
        last = self._last_request_ts_by_host.get(host)
        now = time.time()
        if last is not None:
            elapsed = now - last
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        self._last_request_ts_by_host[host] = time.time()

    def get(self, url: str, *, max_retries: int = 3) -> HttpResponse:
        host = urlparse(url).netloc
        backoff = 0.5
        for attempt in range(max_retries):
            try:
                self._respect_rate_limit(host)
                r = self._client.get(url)
                ct = r.headers.get("content-type")
                return HttpResponse(
                    url=str(r.url),
                    status_code=r.status_code,
                    content_type=ct,
                    text=r.text,
                    content_bytes=r.content,
                )
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(backoff)
                backoff *= 2

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
