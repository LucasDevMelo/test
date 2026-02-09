from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List, Tuple

from bs4 import BeautifulSoup


@dataclass
class ParsedPage:
    title: str
    headings: list[tuple[str, str]]
    full_text: str
    links: list[str]
    content_type: str | None
    status_code: int
    checksum: str


_EMOJI_REGEX = re.compile(
    (
        r"["
        r"\U0001F600-\U0001F64F"  # Emoticons
        r"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        r"\U0001F680-\U0001F6FF"  # Transport & Map
        r"\U0001F1E6-\U0001F1FF"  # Regional country flags
        r"\u2600-\u26FF"          # Misc symbols
        r"\u2700-\u27BF"          # Dingbats
        r"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        r"\U0001FA70-\U0001FAFF"  # Symbols & Pictographs Extended-A
        r"\u200D"                  # Zero Width Joiner
        r"\uFE0E-\uFE0F"          # Variation Selectors
        r"]+"
    ),
    re.UNICODE,
)


def _remove_emojis(text: str) -> str:
    if not text:
        return text
    return _EMOJI_REGEX.sub("", text)


def _normalize_whitespace(text: str) -> str:
    # Collapse multiple spaces/newlines into a single space
    return re.sub(r"\s+", " ", text).strip()


def _strip_boilerplate_tags(soup: BeautifulSoup) -> None:
    # Remove non-content areas before extracting text/links
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    for tag in soup.select("header, footer"):
        tag.decompose()


def parse_html(url: str, html_text: str, *, content_type: str | None, status_code: int) -> ParsedPage:
    soup = BeautifulSoup(html_text, "html.parser")

    # Title from full document
    raw_title = (soup.title.string or "").strip() if soup.title else ""
    title = _normalize_whitespace(_remove_emojis(raw_title))

    # Try to focus on main content region to avoid header/footer/nav noise
    content_root = (
        soup.select_one("main")
        or soup.select_one('[role="main"]')
        or soup.select_one("#main-content")
        or soup.select_one('[id*="main-content"]')
        or soup.select_one('[id*="content"]')
        or soup.body
        or soup
    )

    # Clean boilerplate within the chosen content root
    _strip_boilerplate_tags(content_root)  # type: ignore[arg-type]

    # Headings from content only
    headings: list[tuple[str, str]] = []
    for level in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        for tag in content_root.find_all(level):  # type: ignore[attr-defined]
            text = tag.get_text(strip=True)
            if not text:
                continue
            clean_text = _normalize_whitespace(_remove_emojis(text))
            if clean_text:
                headings.append((level, clean_text))

    # Full text from content only
    full_text_raw = content_root.get_text(" ", strip=True)  # type: ignore[attr-defined]
    full_text = _normalize_whitespace(_remove_emojis(full_text_raw))

    # Links discovered only from content
    links: list[str] = []
    for a in content_root.find_all("a"):  # type: ignore[attr-defined]
        href = a.get("href")
        if not href:
            continue
        links.append(href)

    checksum = hashlib.sha256(html_text.encode("utf-8", errors="ignore")).hexdigest()

    return ParsedPage(
        title=title,
        headings=headings,
        full_text=full_text,
        links=links,
        content_type=content_type,
        status_code=status_code,
        checksum=checksum,
    )
