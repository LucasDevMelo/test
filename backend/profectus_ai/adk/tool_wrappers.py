from __future__ import annotations

import os
import contextvars
from contextlib import contextmanager
from typing import List, Optional

from profectus_ai.evidence.fusion import (
    compute_source_weights,
    infer_preferred_source,
    merge_rag_candidates,
)
from profectus_ai.models import IndexInfoRequest, RagInfoRequest, ReadSpansRequest
from profectus_ai.query_normalization import normalize_query
from profectus_ai.tools.indexinfo import indexinfo
from profectus_ai.tools.open_source import open_source
from profectus_ai.tools.raginfo import raginfo
from profectus_ai.tools.read_spans import read_spans

RAG_MODE_OVERRIDE: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "profectus_rag_mode", default=None
)


@contextmanager
def rag_mode_override(mode: Optional[str]):
    token = RAG_MODE_OVERRIDE.set(mode)
    try:
        yield
    finally:
        RAG_MODE_OVERRIDE.reset(token)


def adk_indexinfo(
    query: Optional[str] = None,
    sources: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    hierarchy_prefix: Optional[List[str]] = None,
    updated_after: Optional[str] = None,
    detail: str = "summary",
    summary_max_chars: int = 280,
    limit: int = 10,
    offset: int = 0,
) -> dict:
    """Search the index overview for candidate documents."""
    normalized_query = normalize_query(query) if query else query
    request = IndexInfoRequest(
        query=normalized_query,
        sources=sources,
        tags=tags,
        hierarchy_prefix=hierarchy_prefix,
        updated_after=updated_after,
        detail=detail,
        summary_max_chars=summary_max_chars,
        limit=limit,
        offset=offset,
    )
    return indexinfo(request).model_dump()


def adk_raginfo(
    query: str,
    top_k: int = 8,
    sources: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    hierarchy_prefix: Optional[List[str]] = None,
) -> dict:
    """Run semantic similarity search over embedded chunks."""
    normalized_query = normalize_query(query)
    request = RagInfoRequest(
        query=normalized_query,
        top_k=top_k,
        sources=sources,
        tags=tags,
        hierarchy_prefix=hierarchy_prefix,
    )
    return raginfo(request).model_dump()


def adk_raginfo_dual(
    query: str,
    top_k: int = 8,
    tags: Optional[List[str]] = None,
    hierarchy_prefix: Optional[List[str]] = None,
) -> dict:
    """Run semantic search and merge by source with weights.

    Mode is controlled by PROFECTUS_RAG_MODE:
      - "single" (default): one search across all sources, then split by source
      - "dual": two searches (Help Center + YouTube) and merge
    """
    normalized_query = normalize_query(query)
    mode_override = RAG_MODE_OVERRIDE.get()
    mode = (mode_override or os.environ.get("PROFECTUS_RAG_MODE", "single")).lower().strip()
    if mode in {"off", "none", "fast", "index"}:
        preferred_source = infer_preferred_source(query)
        return {
            "candidates": [],
            "by_source": {"Help Center": [], "YouTube": []},
            "preferred_source": preferred_source,
            "source_weights": {},
            "errors": {"rag": "skipped (fast mode)"},
        }
    preferred_source = infer_preferred_source(query)
    allow_youtube = preferred_source == "YouTube"
    if mode != "dual":
        sources = None if allow_youtube else ["Help Center"]
        response = raginfo(
            RagInfoRequest(
                query=normalized_query,
                top_k=top_k,
                sources=sources,
                tags=tags,
                hierarchy_prefix=hierarchy_prefix,
            )
        )
        candidates = response.candidates
        candidates_by_source = {}
        for candidate in candidates:
            source = candidate.provenance.source if candidate.provenance else "Help Center"
            candidates_by_source.setdefault(source, []).append(candidate)
        source_weights = compute_source_weights(
            candidates_by_source,
            preferred_source=preferred_source,
        )
        merged = merge_rag_candidates(candidates_by_source, source_weights)
        return {
            "candidates": [c.model_dump() for c in merged],
            "by_source": {
                source: [c.model_dump() for c in candidates]
                for source, candidates in candidates_by_source.items()
            },
            "preferred_source": preferred_source,
            "source_weights": source_weights,
            "errors": {},
        }

    candidates_by_source = {}
    errors: dict[str, str] = {}
    sources = ("Help Center", "YouTube") if allow_youtube else ("Help Center",)
    for source in sources:
        try:
            response = raginfo(
                RagInfoRequest(
                    query=normalized_query,
                    top_k=top_k,
                    sources=[source],
                    tags=tags,
                    hierarchy_prefix=hierarchy_prefix,
                )
            )
            candidates_by_source[source] = response.candidates
        except Exception as exc:
            errors[source] = str(exc)
            candidates_by_source[source] = []

    preferred_source = infer_preferred_source(query)
    source_weights = compute_source_weights(
        candidates_by_source,
        preferred_source=preferred_source,
    )
    merged = merge_rag_candidates(candidates_by_source, source_weights)

    return {
        "candidates": [c.model_dump() for c in merged],
        "by_source": {
            source: [c.model_dump() for c in candidates]
            for source, candidates in candidates_by_source.items()
        },
        "preferred_source": preferred_source,
        "source_weights": source_weights,
        "errors": errors,
    }


def adk_read_spans(
    chunk_ids: List[str],
    start: Optional[int] = None,
    end: Optional[int] = None,
    time_start: Optional[float] = None,
    time_end: Optional[float] = None,
) -> dict:
    """Fetch evidence spans for specific chunk IDs."""
    request = ReadSpansRequest(
        chunk_ids=chunk_ids,
        start=start,
        end=end,
        time_start=time_start,
        time_end=time_end,
    )
    return read_spans(request).model_dump()


def adk_open_source(doc_id: str, max_chars: int = 4000) -> dict:
    """Load full document text for a doc_id (truncated)."""
    return open_source(doc_id, max_chars=max_chars)
