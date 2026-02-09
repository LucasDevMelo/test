from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple
from urllib.parse import urlparse

from profectus_ai.config import (
    DOCS_SCRAPER_DB,
    INDEX_OVERVIEW_COVERAGE_PATH,
    INDEX_OVERVIEW_META_PATH,
    INDEX_OVERVIEW_PATH,
    LOCAL_FINAL_DATA_JSON,
    SCHEMA_VERSION,
    YOUTUBE_JSON_PATHS,
)
from profectus_ai.ids import canonicalize_url, make_doc_id
from profectus_ai.models import IndexEntry, IndexOverviewMetadata, Provenance
from profectus_ai.text_cleaning import clean_transcript_text


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def _summarize(text: str, *, max_chars: int = 280) -> str:
    cleaned = " ".join(text.split())
    return cleaned[:max_chars]


def _hierarchy_from_url(url: str) -> List[str]:
    path = urlparse(url).path
    parts = [p for p in path.split("/") if p]
    return parts


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


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _merge_entries(existing: IndexEntry, candidate: IndexEntry) -> IndexEntry:
    existing_dt = _parse_dt(existing.updated_at)
    candidate_dt = _parse_dt(candidate.updated_at)
    base = candidate if candidate_dt and (not existing_dt or candidate_dt > existing_dt) else existing
    other = existing if base is candidate else candidate

    merged_tags = _normalize_tags(base.tags + other.tags)
    merged_hierarchy = base.hierarchy if len(base.hierarchy) >= len(other.hierarchy) else other.hierarchy
    merged_title = base.title if len(base.title) >= len(other.title) else other.title
    merged_summary = base.summary if len(base.summary) >= len(other.summary) else other.summary
    merged_updated = base.updated_at or other.updated_at

    return IndexEntry(
        doc_id=base.doc_id,
        source=base.source,
        title=merged_title,
        summary=merged_summary,
        hierarchy=merged_hierarchy,
        tags=merged_tags,
        updated_at=merged_updated,
        source_id=base.source_id or other.source_id,
        doc_url=base.doc_url,
        provenance=base.provenance,
    )


def load_help_center_entries(db_path: Path = DOCS_SCRAPER_DB) -> List[IndexEntry]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT url, title, headings, full_text, scraped_at, meta_keywords, meta_description
        FROM pages
        WHERE url IS NOT NULL AND full_text IS NOT NULL
        """
    ).fetchall()
    entries_by_url: dict[str, IndexEntry] = {}
    for row in rows:
        url = canonicalize_url(row["url"])
        title = row["title"] or url
        headings = _parse_headings(row["headings"])
        meta_keywords = row["meta_keywords"] or ""
        meta_description = row["meta_description"] or ""
        full_text = row["full_text"] or ""
        summary_source = meta_description or full_text
        summary = _summarize(summary_source) if summary_source else title
        tags = _normalize_tags(meta_keywords.split(",") + headings)
        hierarchy = _hierarchy_from_url(url) or ["root"]
        if not tags:
            tags = _normalize_tags([title] + hierarchy)
        doc_id = make_doc_id(source="Help Center", url=url)
        provenance = Provenance(source="Help Center", url=canonicalize_url(url))
        entry = IndexEntry(
            doc_id=doc_id,
            source="Help Center",
            title=title,
            summary=summary,
            hierarchy=hierarchy,
            tags=tags,
            updated_at=row["scraped_at"],
            source_id=None,
            doc_url=canonicalize_url(url),
            provenance=provenance,
        )
        if url in entries_by_url:
            entries_by_url[url] = _merge_entries(entries_by_url[url], entry)
        else:
            entries_by_url[url] = entry
    conn.close()
    return list(entries_by_url.values())


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


def load_youtube_entries(paths: Iterable[Path] = YOUTUBE_JSON_PATHS) -> List[IndexEntry]:
    items = _load_youtube_items(paths)
    grouped: dict[str, list[dict]] = {}
    for item in items:
        metadata = item.get("metadata", {}) or {}
        video_id = metadata.get("video_id") or item.get("id")
        if not video_id:
            continue
        # Use base video_id (strip _chunk suffix)
        base_video_id = video_id.split("_chunk")[0]
        grouped.setdefault(base_video_id, []).append(item)

    entries: List[IndexEntry] = []
    for video_id, group in grouped.items():
        first = group[0]
        metadata = first.get("metadata", {}) or {}
        title = metadata.get("title") or first.get("title") or video_id
        url = metadata.get("url") or first.get("url") or f"https://youtu.be/{video_id}"

        tags = _normalize_tags(
            [tag for item in group for tag in (item.get("metadata", {}) or {}).get("tags", [])]
        )
        cleaned_text = clean_transcript_text(first.get("text", ""))
        summary = _summarize(cleaned_text) or title
        doc_id = make_doc_id(source="YouTube", video_id=video_id)

        # Extract timestamp information from metadata if available
        time_start = metadata.get("time_start")
        time_end = metadata.get("time_end")

        provenance = Provenance(
            source="YouTube",
            url=url,
            timestamp_start=time_start,
            timestamp_end=time_end,
        )
        entries.append(
            IndexEntry(
                doc_id=doc_id,
                source="YouTube",
                title=title,
                summary=summary,
                hierarchy=["youtube"],
                tags=tags,
                updated_at=None,
                source_id=video_id,
                doc_url=url,
                provenance=provenance,
            )
        )
    return entries


def write_coverage_report(
    entries: List[IndexEntry],
    *,
    corpus_path: Path = LOCAL_FINAL_DATA_JSON,
    output_path: Path = INDEX_OVERVIEW_COVERAGE_PATH,
) -> None:
    if not corpus_path.exists():
        print(f"[index_overview] coverage skipped (missing corpus): {corpus_path}")
        return

    with corpus_path.open("r", encoding="utf-8") as handle:
        corpus = json.load(handle)

    index_urls = {
        canonicalize_url(entry.doc_url)
        for entry in entries
        if entry.source == "Help Center" and entry.doc_url
    }
    corpus_urls = {
        canonicalize_url(item.get("url"))
        for item in corpus
        if item.get("source") == "Help Center" and item.get("url")
    }

    missing_in_corpus = sorted(index_urls - corpus_urls)
    missing_in_index = sorted(corpus_urls - index_urls)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "build_timestamp": _utc_now_iso(),
        "corpus_path": str(corpus_path),
        "index_help_center_docs": len(index_urls),
        "corpus_help_center_docs": len(corpus_urls),
        "missing_in_corpus": missing_in_corpus,
        "missing_in_index": missing_in_index,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2)
    print(
        f"[index_overview] coverage report: {len(missing_in_corpus)} missing in corpus, "
        f"{len(missing_in_index)} missing in index"
    )


def validate_entries(entries: Iterable[IndexEntry]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    for entry in entries:
        if not entry.doc_id:
            errors.append("Missing doc_id")
        if not entry.title:
            errors.append(f"Missing title for {entry.doc_id}")
        if not entry.doc_url:
            errors.append(f"Missing doc_url for {entry.doc_id}")
        if not entry.provenance.url:
            errors.append(f"Missing provenance url for {entry.doc_id}")
        if entry.source == "YouTube" and entry.provenance.timestamp_start is None:
            warnings.append(f"Missing YouTube timestamp for {entry.doc_id}")
    return errors, warnings


def write_index_overview(
    entries: List[IndexEntry],
    *,
    output_path: Path = INDEX_OVERVIEW_PATH,
    meta_path: Path = INDEX_OVERVIEW_META_PATH,
) -> IndexOverviewMetadata:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            line = json.dumps(entry.model_dump(), ensure_ascii=True)
            handle.write(line + "\n")

    checksum = _sha256_file(output_path)
    counts = Counter(entry.source for entry in entries)
    metadata = IndexOverviewMetadata(
        schema_version=SCHEMA_VERSION,
        build_timestamp=_utc_now_iso(),
        source_counts=dict(counts),
        checksum=checksum,
        total_docs=len(entries),
    )
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata.model_dump(), handle, ensure_ascii=True, indent=2)
    return metadata


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_index_overview() -> IndexOverviewMetadata:
    help_center = load_help_center_entries()
    youtube = load_youtube_entries()
    entries = help_center + youtube

    errors, warnings = validate_entries(entries)
    if errors:
        raise ValueError("Index overview validation failed: " + "; ".join(errors))
    for warning in warnings:
        print(f"[index_overview] warning: {warning}")

    metadata = write_index_overview(entries)
    write_coverage_report(entries)
    print(
        f"[index_overview] wrote {len(entries)} entries to {INDEX_OVERVIEW_PATH} "
        f"with schema {metadata.schema_version}"
    )
    return metadata


if __name__ == "__main__":
    build_index_overview()
