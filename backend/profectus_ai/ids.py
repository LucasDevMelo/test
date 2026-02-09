from __future__ import annotations

import hashlib
from typing import Optional
from urllib.parse import urlparse, urlunparse


def _stable_hash(value: str, *, length: int = 12) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return digest[:length]


def canonicalize_url(url: str) -> str:
    parsed = urlparse(url)
    # Drop fragments and normalize trailing slash.
    normalized = parsed._replace(fragment="", query="")
    cleaned = urlunparse(normalized)
    if cleaned.endswith("/") and len(cleaned) > len(parsed.scheme) + 3:
        cleaned = cleaned.rstrip("/")
    return cleaned


def make_doc_id(*, source: str, url: Optional[str] = None, video_id: Optional[str] = None) -> str:
    if source.lower() == "youtube":
        if not video_id:
            raise ValueError("video_id is required for YouTube doc_id")
        key = f"youtube::{video_id}"
        prefix = "yt"
    else:
        if not url:
            raise ValueError("url is required for Help Center doc_id")
        canonical = canonicalize_url(url)
        key = f"help::{canonical}"
        prefix = "hc"
    return f"{prefix}_{_stable_hash(key)}"


def make_chunk_id(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}::chunk_{chunk_index}"


def make_span_id(
    chunk_id: str,
    *,
    start: Optional[int] = None,
    end: Optional[int] = None,
    time_start: Optional[float] = None,
    time_end: Optional[float] = None,
) -> str:
    if start is not None and end is not None:
        return f"{chunk_id}::span_{start}_{end}"
    if time_start is not None and time_end is not None:
        start_tag = str(time_start).replace(".", "_")
        end_tag = str(time_end).replace(".", "_")
        return f"{chunk_id}::t_{start_tag}_{end_tag}"
    raise ValueError("Either text span (start/end) or time span (time_start/time_end) is required.")
