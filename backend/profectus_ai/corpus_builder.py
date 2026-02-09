from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from typing import Iterable, List

from sentence_transformers import SentenceTransformer

from profectus_ai.config import (
    DEFAULT_EMBEDDING_MODEL,
    DOCS_SCRAPER_DB,
    LOCAL_FINAL_DATA_JSON,
    YOUTUBE_JSON_PATHS,
)
from profectus_ai.ids import canonicalize_url, make_doc_id
from profectus_ai.text_cleaning import clean_transcript_text


@dataclass(frozen=True)
class HelpCenterPage:
    url: str
    title: str
    headings: List[str]
    meta_keywords: List[str]
    full_text: str


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


def _hierarchy_from_url(url: str) -> List[str]:
    path = urlparse(url).path
    parts = [p for p in path.split("/") if p]
    return parts or ["root"]


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


def load_help_center_pages(db_path: Path = DOCS_SCRAPER_DB) -> List[HelpCenterPage]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT url, title, headings, full_text, meta_keywords
        FROM pages
        WHERE url IS NOT NULL AND full_text IS NOT NULL
        """
    ).fetchall()

    pages: List[HelpCenterPage] = []
    for row in rows:
        url = canonicalize_url(row["url"])
        title = row["title"] or url
        headings = _parse_headings(row["headings"])
        meta_keywords = [kw.strip() for kw in (row["meta_keywords"] or "").split(",") if kw.strip()]
        full_text = row["full_text"] or ""
        if not full_text.strip():
            continue
        pages.append(
            HelpCenterPage(
                url=url,
                title=title,
                headings=headings,
                meta_keywords=meta_keywords,
                full_text=full_text,
            )
        )
    conn.close()
    return pages


def _load_youtube_items(paths: Iterable[Path]) -> List[dict]:
    items: List[dict] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
            if isinstance(payload, list):
                items.extend(payload)
    return items


def _embed_batches(model: SentenceTransformer, texts: List[str], *, batch_size: int) -> List[List[float]]:
    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeds = model.encode(batch)
        vectors.extend([v.tolist() for v in embeds])
    return vectors


def build_corpus(
    *,
    output_path: Path = LOCAL_FINAL_DATA_JSON,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
) -> Path:
    pages = load_help_center_pages()
    youtube_items = _load_youtube_items(YOUTUBE_JSON_PATHS)

    records: List[dict] = []

    # Help Center records
    for page in pages:
        hierarchy = _hierarchy_from_url(page.url)
        tags = _normalize_tags(page.meta_keywords + page.headings)
        if not tags:
            tags = _normalize_tags([page.title] + hierarchy)
        doc_id = make_doc_id(source="Help Center", url=page.url)
        chunks = _chunk_text(page.full_text)
        for idx, chunk in enumerate(chunks, start=1):
            chunk_id = f"{doc_id}::chunk_{idx}"
            records.append(
                {
                    "id": chunk_id,
                    "source": "Help Center",
                    "title": page.title,
                    "url": page.url,
                    "chunk_index": idx,
                    "text": chunk,
                    "embedding_vector": [],
                    "metadata": {
                        "tags": tags,
                        "category": "help_center",
                    },
                }
            )

    # YouTube records
    for item in youtube_items:
        metadata = item.get("metadata", {}) or {}
        cleaned = clean_transcript_text(str(item.get("text") or ""))
        record = {
            "id": item.get("id"),
            "source": "YouTube",
            "title": item.get("title") or metadata.get("title") or "",
            "url": item.get("url") or metadata.get("url") or "",
            "chunk_index": int(item.get("chunk_index") or 0),
            "text": cleaned,
            "embedding_vector": [],
            "metadata": {
                **metadata,
                "tags": metadata.get("tags", []) or [],
                "category": metadata.get("category", "youtube"),
                "video_id": metadata.get("video_id") or item.get("id"),
            },
        }
        records.append(record)

    model = SentenceTransformer(embedding_model)
    texts = [record["text"] for record in records]
    vectors = _embed_batches(model, texts, batch_size=batch_size)
    for record, vector in zip(records, vectors):
        record["embedding_vector"] = vector

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    return output_path


if __name__ == "__main__":
    out = build_corpus()
    print(f"[corpus_builder] wrote {out}")
