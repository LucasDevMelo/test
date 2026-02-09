from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse
from urllib import robotparser

import httpx


@dataclass
class RobotsResult:
    allowed: bool
    crawl_delay: Optional[float]


class RobotsPolicy:
    def __init__(self, user_agent: str) -> None:
        self.user_agent = user_agent
        self._cache: dict[str, robotparser.RobotFileParser] = {}

    def _get_parser_for(self, url: str) -> robotparser.RobotFileParser:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin in self._cache:
            return self._cache[origin]
        robots_url = f"{origin}/robots.txt"
        rp = robotparser.RobotFileParser()
        # Use httpx to fetch with short timeout; fall back to allow if fails
        try:
            with httpx.Client(timeout=5.0, follow_redirects=True) as client:
                resp = client.get(robots_url)
                if resp.status_code >= 400:
                    # Treat missing robots as allow
                    rp.parse("")
                else:
                    rp.parse(resp.text.splitlines())
        except Exception:
            rp.parse("")
        self._cache[origin] = rp
        return rp

    def check(self, url: str) -> RobotsResult:
        rp = self._get_parser_for(url)
        allowed = rp.can_fetch(self.user_agent, url)
        try:
            delay = rp.crawl_delay(self.user_agent)
        except Exception:
            delay = None
        return RobotsResult(allowed=allowed, crawl_delay=delay)
