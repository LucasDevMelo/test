"""Full document retrieval from Help Center and YouTube sources.

This tool loads complete document text (truncated to a max length)
for a given doc_id, supporting both Help Center (SQLite) and
YouTube (JSON corpus) sources.
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from profectus_ai.config import DOCS_SCRAPER_DB
from profectus_ai.evidence.store import get_corpus_store, get_index_store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceDocument:
    """Full document content with metadata."""

    doc_id: str
    source: str
    title: str
    url: str
    text: str
    truncated: bool


@contextmanager
def _db_connection(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for safe SQLite connections."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _load_help_center_text(url: str) -> str:
    """Load full text from Help Center SQLite database."""
    if not DOCS_SCRAPER_DB.exists():
        logger.warning("Help Center database not found: %s", DOCS_SCRAPER_DB)
        return ""
    try:
        with _db_connection(DOCS_SCRAPER_DB) as conn:
            cur = conn.cursor()
            row = cur.execute("SELECT full_text FROM pages WHERE url = ?", (url,)).fetchone()
            return row["full_text"] if row and row["full_text"] else ""
    except sqlite3.Error as exc:
        logger.warning("SQLite error loading %s: %s", url, exc)
        return ""


def _load_youtube_text(doc_id: str) -> str:
    """Load full text from YouTube corpus by concatenating chunks."""
    store = get_corpus_store()
    records = store.get_by_doc_id(doc_id)
    if not records:
        return ""
    # Sort by chunk index and concatenate
    sorted_records = sorted(records, key=lambda r: r.chunk_index)
    texts = [r.text for r in sorted_records]
    return "\n\n".join(texts)


def open_source(doc_id: str, *, max_chars: int = 4000) -> dict:
    """Load full document text for a doc_id.

    Args:
        doc_id: The document identifier.
        max_chars: Maximum characters to return (default 4000).

    Returns:
        Dictionary with doc_id, source, title, url, text, truncated, and optional error.
    """
    index = get_index_store()
    entry = index.get(doc_id)

    # Handle missing doc_id gracefully (no exception)
    if entry is None:
        logger.warning("Unknown doc_id: %s", doc_id)
        return {
            "doc_id": doc_id,
            "source": "unknown",
            "title": "",
            "url": "",
            "text": "",
            "truncated": False,
            "error": f"Unknown doc_id: {doc_id}",
        }

    # Load text based on source
    if entry.source == "Help Center":
        text = _load_help_center_text(entry.doc_url)
    else:
        text = _load_youtube_text(entry.doc_id)

    # Truncate if needed
    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True

    doc = SourceDocument(
        doc_id=entry.doc_id,
        source=entry.source,
        title=entry.title,
        url=entry.doc_url,
        text=text,
        truncated=truncated,
    )
    return {
        "doc_id": doc.doc_id,
        "source": doc.source,
        "title": doc.title,
        "url": doc.url,
        "text": doc.text,
        "truncated": doc.truncated,
    }
