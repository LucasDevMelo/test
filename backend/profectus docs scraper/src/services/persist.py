from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Optional

from src.models.db import Database
from src.services.parse import ParsedPage


def _get_existing_checksum(db: Database, url: str) -> Optional[str]:
    cur = db.conn.cursor()
    row = cur.execute("SELECT checksum FROM pages WHERE url = ?", (url,)).fetchone()
    return row[0] if row else None


def upsert_page(db: Database, url: str, page: ParsedPage) -> bool:
    existing = _get_existing_checksum(db, url)
    cur = db.conn.cursor()
    changed = existing != page.checksum
    if existing is None:
        cur.execute(
            """
            INSERT INTO pages (id, url, scraped_at, title, headings, full_text, links, outgoing_domains, status_code, content_type, checksum, language, meta_keywords, meta_description)
            VALUES (?, ?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL)
            """,
            (
                str(uuid.uuid4()),
                url,
                page.title,
                json.dumps(page.headings, ensure_ascii=False),
                page.full_text,
                json.dumps(page.links, ensure_ascii=False),
                json.dumps([], ensure_ascii=False),
                page.status_code,
                page.content_type,
                page.checksum,
            ),
        )
    else:
        # Always refresh derived content fields on re-parse to reflect parser improvements,
        # even if the raw HTML checksum hasn't changed.
        cur.execute(
            """
            UPDATE pages
            SET scraped_at = datetime('now'),
                title = ?,
                headings = ?,
                full_text = ?,
                links = ?,
                outgoing_domains = ?,
                status_code = ?,
                content_type = ?,
                checksum = ?
            WHERE url = ?
            """,
            (
                page.title,
                json.dumps(page.headings, ensure_ascii=False),
                page.full_text,
                json.dumps(page.links, ensure_ascii=False),
                json.dumps([], ensure_ascii=False),
                page.status_code,
                page.content_type,
                page.checksum,
                url,
            ),
        )
    db.conn.commit()
    return changed
