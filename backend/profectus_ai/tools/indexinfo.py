"""Index overview search with BM25 ranking.

This tool provides keyword search over the document index
with BM25-based relevance scoring for better ranking.
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from profectus_ai.config import INDEX_OVERVIEW_PATH
from profectus_ai.models import IndexEntry, IndexInfoRequest, IndexInfoResponse

logger = logging.getLogger(__name__)


class BM25Scorer:
    """BM25 relevance scorer for IndexEntry documents.

    Uses the BM25 formula to compute relevance scores for documents
    against a query. Better than simple substring matching because
    it considers:
    - Term frequency (more occurrences = more relevant)
    - Inverse document frequency (rare terms = more valuable)
    - Document length normalization (fair comparison)
    """

    def __init__(
        self,
        entries: List[IndexEntry],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.entries = entries
        self.n = len(entries)
        self.avg_dl = 0.0
        self.doc_freqs: Counter[str] = Counter()
        self.doc_tokens: List[List[str]] = []
        self._build_stats()

    def _tokenize(self, text: str) -> List[str]:
        """Extract lowercase word tokens from text."""
        return [t.lower() for t in re.findall(r"\w+", text) if len(t) > 1]

    def _entry_text(self, entry: IndexEntry) -> str:
        """Combine searchable fields into a single text."""
        # Weight title more heavily by repeating it
        return " ".join([
            entry.title,
            entry.title,  # Title appears twice for more weight
            entry.summary,
            " ".join(entry.tags),
            " ".join(entry.hierarchy),
        ])

    def _build_stats(self) -> None:
        """Pre-compute term statistics for BM25."""
        total_len = 0
        for entry in self.entries:
            text = self._entry_text(entry)
            tokens = self._tokenize(text)
            self.doc_tokens.append(tokens)
            total_len += len(tokens)
            # Count document frequency (each term counted once per doc)
            for token in set(tokens):
                self.doc_freqs[token] += 1
        self.avg_dl = total_len / self.n if self.n else 1.0
        logger.debug(
            "BM25 stats: %d docs, avg_dl=%.1f, unique_terms=%d",
            self.n, self.avg_dl, len(self.doc_freqs)
        )

    def score(self, query: str, entry_idx: int) -> float:
        """Compute BM25 score for a query against a document."""
        if entry_idx >= len(self.doc_tokens):
            return 0.0

        doc_tokens = self.doc_tokens[entry_idx]
        doc_len = len(doc_tokens)
        query_tokens = self._tokenize(query)

        if not query_tokens or not doc_tokens:
            return 0.0

        # Count term frequencies in this document
        doc_tf = Counter(doc_tokens)

        total_score = 0.0
        for qt in query_tokens:
            tf = doc_tf.get(qt, 0)
            if tf == 0:
                continue
            df = self.doc_freqs.get(qt, 0)
            # IDF with smoothing
            idf = math.log((self.n - df + 0.5) / (df + 0.5) + 1)
            # TF component with length normalization
            tf_component = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            )
            total_score += idf * tf_component

        return total_score

    def score_all(self, query: str) -> List[Tuple[int, float]]:
        """Score all documents and return (index, score) pairs with score > 0."""
        results = []
        for idx in range(len(self.entries)):
            score = self.score(query, idx)
            if score > 0:
                results.append((idx, score))
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results


def _load_entries(path: Path) -> List[IndexEntry]:
    """Load IndexEntry objects from JSONL file."""
    entries: List[IndexEntry] = []
    if not path.exists():
        logger.warning("Index overview not found: %s", path)
        return entries
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            entries.append(IndexEntry.model_validate(payload))
    logger.debug("Loaded %d index entries from %s", len(entries), path)
    return entries


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO datetime string, returning None on failure."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _truncate(text: str, max_chars: int) -> str:
    """Normalize whitespace and truncate to max_chars."""
    cleaned = " ".join(text.split())
    if max_chars <= 0:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip()


def _apply_detail(entry: IndexEntry, detail: str, summary_max_chars: int) -> IndexEntry:
    """Adjust output granularity to reduce token usage."""
    mode = (detail or "summary").lower()
    summary = entry.summary
    if mode in {"summary", "title"}:
        summary = _truncate(summary, summary_max_chars)
    tags = entry.tags
    hierarchy = entry.hierarchy
    if mode == "title":
        summary = ""
        tags = []
        hierarchy = []
    return entry.model_copy(update={"summary": summary, "tags": tags, "hierarchy": hierarchy})


def _match_tags(entry_tags: Iterable[str], query_tags: Iterable[str]) -> bool:
    """Check if any query tag matches any entry tag (case-insensitive)."""
    entry_lower = {tag.lower() for tag in entry_tags}
    for tag in query_tags:
        if tag.lower() in entry_lower:
            return True
    return False


def _match_hierarchy(entry_hierarchy: List[str], prefix: List[str]) -> bool:
    """Check if entry hierarchy starts with the given prefix."""
    if len(prefix) > len(entry_hierarchy):
        return False
    return [p.lower() for p in entry_hierarchy[: len(prefix)]] == [p.lower() for p in prefix]


def indexinfo(
    request: IndexInfoRequest,
    *,
    index_path: Path = INDEX_OVERVIEW_PATH,
) -> IndexInfoResponse:
    """Search the index overview with optional filters and BM25 ranking.

    Filters are applied before scoring. If a query is provided,
    results are ranked by BM25 relevance score (highest first).
    Without a query, results are sorted by doc_id.
    """
    entries = _load_entries(index_path)

    # Apply filters first (reduces scoring work)
    if request.sources:
        allowed = set(request.sources)
        entries = [entry for entry in entries if entry.source in allowed]

    if request.tags:
        entries = [entry for entry in entries if _match_tags(entry.tags, request.tags)]

    if request.hierarchy_prefix:
        entries = [
            entry
            for entry in entries
            if _match_hierarchy(entry.hierarchy, request.hierarchy_prefix)
        ]

    if request.updated_after:
        after_dt = _parse_iso(request.updated_after)
        if after_dt:
            entries = [
                entry
                for entry in entries
                if _parse_iso(entry.updated_at) and _parse_iso(entry.updated_at) > after_dt
            ]

    # Score and rank if query provided
    if request.query:
        scorer = BM25Scorer(entries)
        scored = scorer.score_all(request.query)
        # Reorder entries by score
        entries = [entries[idx] for idx, _ in scored]
        logger.debug("BM25 query '%s' matched %d entries", request.query, len(entries))
    else:
        # No query - sort by doc_id for deterministic ordering
        entries.sort(key=lambda item: item.doc_id)

    # Pagination
    total = len(entries)
    offset = max(request.offset, 0)
    limit = max(request.limit, 0)
    page = entries[offset : offset + limit] if limit else []

    # Apply detail granularity to reduce payload size
    detail = request.detail or "summary"
    summary_max_chars = max(request.summary_max_chars, 0)
    page = [_apply_detail(entry, detail, summary_max_chars) for entry in page]

    return IndexInfoResponse(
        total=total,
        offset=offset,
        limit=limit,
        items=page,
    )
