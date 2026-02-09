from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable
import json

from src.models.db import Database, EXPORTS_DIR


def export_pages_csv(db: Database, out_path: Path | None = None) -> Path:
    out = (out_path or (EXPORTS_DIR / "pages.csv"))
    cur = db.conn.cursor()
    rows = cur.execute(
        "SELECT url, scraped_at, title, status_code, content_type, checksum FROM pages ORDER BY url"
    ).fetchall()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["url", "scraped_at", "title", "status_code", "content_type", "checksum"])
        for r in rows:
            w.writerow([r["url"], r["scraped_at"], r["title"], r["status_code"], r["content_type"], r["checksum"]])
    return out


def export_pages_full_csv(db: Database, out_path: Path | None = None) -> Path:
    out = (out_path or (EXPORTS_DIR / "pages_full.csv"))
    cur = db.conn.cursor()
    rows = cur.execute(
        """
        SELECT url, scraped_at, title, headings, full_text, links, outgoing_domains,
               status_code, content_type, checksum
        FROM pages
        ORDER BY url
        """
    ).fetchall()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "url",
            "scraped_at",
            "title",
            "headings",
            "full_text",
            "links",
            "outgoing_domains",
            "status_code",
            "content_type",
            "checksum",
        ])
        for r in rows:
            w.writerow([
                r["url"],
                r["scraped_at"],
                r["title"],
                r["headings"],  # already JSON string
                r["full_text"],
                r["links"],     # already JSON string
                r["outgoing_domains"],  # JSON string
                r["status_code"],
                r["content_type"],
                r["checksum"],
            ])
    return out
