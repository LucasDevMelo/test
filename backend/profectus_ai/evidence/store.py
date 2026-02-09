"""Centralized singleton stores for corpus and index data.

This module provides singleton accessors to avoid duplicate store instances
across the codebase. All modules that need corpus or index data should
import from here rather than creating their own instances.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional
from urllib.parse import urlparse

from profectus_ai.config import DOCS_SCRAPER_DB, INDEX_OVERVIEW_PATH, LOCAL_FINAL_DATA_JSON
from profectus_ai.ids import canonicalize_url, make_chunk_id, make_doc_id
from profectus_ai.text_cleaning import clean_transcript_text

if TYPE_CHECKING:
    from profectus_ai.models import IndexEntry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorpusRecord:
    """A single chunk from the unified corpus."""

    chunk_id: str
    doc_id: str
    source: str
    url: str
    text: str
    chunk_index: int
    tags: List[str]
    hierarchy: List[str]
    time_start: Optional[float]
    time_end: Optional[float]


class CorpusStore:
    """In-memory store for corpus chunks, keyed by chunk_id."""

    def __init__(self, data_path: Path = LOCAL_FINAL_DATA_JSON) -> None:
        self.data_path = data_path
        self._records: Dict[str, CorpusRecord] = {}
        self._by_doc_id: Dict[str, List[CorpusRecord]] = {}
        self._load()

    def _load(self) -> None:
        if not self.data_path.exists():
            logger.warning("Corpus data not found: %s", self.data_path)
            return
        with self.data_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for item in payload:
            source = item.get("source") or "Help Center"
            url = item.get("url") or ""
            metadata = item.get("metadata", {}) or {}
            time_start = metadata.get("time_start")
            time_end = metadata.get("time_end")
            video_id = metadata.get("video_id") or item.get("id")
            try:
                doc_id = make_doc_id(
                    source=source,
                    url=url if source != "YouTube" else None,
                    video_id=video_id if source == "YouTube" else None,
                )
            except ValueError:
                continue
            chunk_index = int(item.get("chunk_index") or 0)
            chunk_id = make_chunk_id(doc_id, chunk_index)
            text = str(item.get("text") or "")
            if source == "YouTube":
                text = clean_transcript_text(text)
            tags = metadata.get("tags", []) or []
            hierarchy = _hierarchy_from_url(url) if source != "YouTube" else ["youtube"]
            record = CorpusRecord(
                chunk_id=chunk_id,
                doc_id=doc_id,
                source=source,
                url=url,
                text=text,
                chunk_index=chunk_index,
                tags=list(tags),
                hierarchy=hierarchy,
                time_start=float(time_start) if time_start is not None else None,
                time_end=float(time_end) if time_end is not None else None,
            )
            self._records[chunk_id] = record
            self._by_doc_id.setdefault(doc_id, []).append(record)
        self._augment_help_center_records()
        logger.info("Loaded %d chunks from corpus", len(self._records))

    def _augment_help_center_records(self) -> None:
        if not DOCS_SCRAPER_DB.exists():
            return

        existing_urls = {
            canonicalize_url(record.url)
            for record in self._records.values()
            if record.source == "Help Center" and record.url
        }
        added_chunks = 0
        added_docs = 0

        conn = sqlite3.connect(str(DOCS_SCRAPER_DB))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        rows = cur.execute(
            """
            SELECT url, title, headings, full_text, meta_keywords
            FROM pages
            WHERE url IS NOT NULL AND full_text IS NOT NULL
            """
        ).fetchall()
        for row in rows:
            url = canonicalize_url(row["url"])
            if not url or url in existing_urls:
                continue
            text = str(row["full_text"] or "")
            if not text.strip():
                continue
            headings = _parse_headings(row["headings"])
            meta_keywords = row["meta_keywords"] or ""
            hierarchy = _hierarchy_from_url(url) or ["root"]
            tags = _normalize_tags(meta_keywords.split(",") + headings)
            if not tags:
                tags = _normalize_tags([row["title"] or url] + hierarchy)

            doc_id = make_doc_id(source="Help Center", url=url)
            chunks = _chunk_text(text)
            if not chunks:
                continue

            added_docs += 1
            for idx, chunk in enumerate(chunks, start=1):
                chunk_id = make_chunk_id(doc_id, idx)
                record = CorpusRecord(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    source="Help Center",
                    url=url,
                    text=chunk,
                    chunk_index=idx,
                    tags=tags,
                    hierarchy=hierarchy,
                    time_start=None,
                    time_end=None,
                )
                self._records[chunk_id] = record
                self._by_doc_id.setdefault(doc_id, []).append(record)
                added_chunks += 1

        conn.close()
        if added_docs:
            logger.info(
                "Augmented corpus with %d docs (%d chunks) from Help Center DB",
                added_docs,
                added_chunks,
            )

    def get(self, chunk_id: str) -> Optional[CorpusRecord]:
        """Get a single chunk by chunk_id."""
        return self._records.get(chunk_id)

    def get_by_doc_id(self, doc_id: str) -> List[CorpusRecord]:
        """Get all chunks for a document."""
        return self._by_doc_id.get(doc_id, [])

    def records(self) -> List[CorpusRecord]:
        """Return all records."""
        return list(self._records.values())


class IndexOverviewStore:
    """In-memory store for index overview entries, keyed by doc_id."""

    def __init__(self, path: Path = INDEX_OVERVIEW_PATH) -> None:
        self.path = path
        self._entries: Dict[str, "IndexEntry"] = {}
        self._load()

    def _load(self) -> None:
        from profectus_ai.models import IndexEntry

        if not self.path.exists():
            logger.warning("Index overview not found: %s", self.path)
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                entry = IndexEntry.model_validate(payload)
                self._entries[entry.doc_id] = entry
        logger.info("Loaded %d entries from index overview", len(self._entries))

    def get(self, doc_id: str) -> Optional["IndexEntry"]:
        """Get an index entry by doc_id."""
        return self._entries.get(doc_id)

    def all_entries(self) -> List["IndexEntry"]:
        """Return all index entries."""
        return list(self._entries.values())


def _hierarchy_from_url(url: str) -> List[str]:
    """Extract hierarchy path components from URL."""
    path = urlparse(url).path
    return [part for part in path.split("/") if part]


def _parse_headings(raw: str | None) -> List[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            headings: List[str] = []
            for item in parsed:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    text = str(item[1]).strip()
                else:
                    text = str(item).strip()
                if text:
                    headings.append(text)
            return headings
    except json.JSONDecodeError:
        pass
    return []


def _normalize_tags(values: Iterable[str]) -> List[str]:
    seen = set()
    tags: List[str] = []
    for raw in values:
        tag = raw.strip()
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        tags.append(tag)
    return tags


def _chunk_text(text: str, *, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return [cleaned] if cleaned else []

    chunks: List[str] = []
    start = 0
    text_len = len(cleaned)
    while start < text_len:
        end = min(text_len, start + max_chars)
        if end < text_len:
            split_at = cleaned.rfind(" ", start, end)
            if split_at > start + 100:
                end = split_at
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = max(0, end - overlap)
    return chunks


# Singleton instances
_CORPUS_STORE: Optional[CorpusStore] = None
_INDEX_STORE: Optional[IndexOverviewStore] = None


def get_corpus_store(data_path: Path = LOCAL_FINAL_DATA_JSON) -> CorpusStore:
    """Get or create the singleton CorpusStore."""
    global _CORPUS_STORE
    if _CORPUS_STORE is None:
        _CORPUS_STORE = CorpusStore(data_path)
    return _CORPUS_STORE


def get_index_store(path: Path = INDEX_OVERVIEW_PATH) -> IndexOverviewStore:
    """Get or create the singleton IndexOverviewStore."""
    global _INDEX_STORE
    if _INDEX_STORE is None:
        _INDEX_STORE = IndexOverviewStore(path)
    return _INDEX_STORE


def reset_stores() -> None:
    """Reset singleton stores (useful for testing)."""
    global _CORPUS_STORE, _INDEX_STORE
    _CORPUS_STORE = None
    _INDEX_STORE = None
