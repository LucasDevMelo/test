"""RAG-based semantic and lexical search.

This tool provides semantic search using FAISS embeddings with
automatic fallback to lexical search when FAISS is unavailable.
"""
from __future__ import annotations

import logging
import threading
import re
from typing import List, Tuple

from profectus_ai.evidence.store import get_corpus_store
from profectus_ai.models import Provenance, RagCandidate, RagInfoRequest, RagInfoResponse

logger = logging.getLogger(__name__)

_FAISS_STORE = None
_FAISS_LOCK = threading.Lock()


def _get_faiss_store():
    """Get the FAISS store singleton, or raise if unavailable."""
    global _FAISS_STORE
    if _FAISS_STORE is None:
        with _FAISS_LOCK:
            if _FAISS_STORE is None:
                try:
                    from profectus_ai.retrieval.faiss_store import FaissStore
                except ModuleNotFoundError as exc:
                    raise RuntimeError(
                        "faiss or sentence-transformers is not installed. "
                        "Install dependencies to enable raginfo."
                    ) from exc
                _FAISS_STORE = FaissStore()
    return _FAISS_STORE


def _match_tags(entry_tags: List[str], query_tags: List[str]) -> bool:
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


def _lexical_score(query_tokens: List[str], text: str) -> float:
    """Score text by counting matching query tokens."""
    haystack = text.lower()
    return float(sum(1 for token in query_tokens if token in haystack))


def _lexical_search(request: RagInfoRequest) -> Tuple[List[RagCandidate], str]:
    """Fallback lexical search when FAISS is unavailable."""
    store = get_corpus_store()
    tokens = [t for t in re.findall(r"\w+", request.query.lower()) if len(t) > 2]
    scored = []
    for record in store.records():
        score = _lexical_score(tokens, record.text)
        if score <= 0:
            continue
        if request.sources and record.source not in request.sources:
            continue
        if request.tags and not _match_tags(record.tags, request.tags):
            continue
        if request.hierarchy_prefix and not _match_hierarchy(record.hierarchy, request.hierarchy_prefix):
            continue
        scored.append((record, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    candidates: List[RagCandidate] = []
    for rank, (record, score) in enumerate(scored[: request.top_k], start=1):
        snippet = " ".join(record.text.split())[:240]
        # Create proper Provenance object (fixes type error)
        provenance = Provenance(
            source=record.source,
            url=record.url,
            timestamp_start=record.time_start,
            timestamp_end=record.time_end,
        )
        candidates.append(
            RagCandidate(
                chunk_id=record.chunk_id,
                doc_id=record.doc_id,
                snippet=snippet,
                rank=rank,
                provenance=provenance,
                score=score,
                tags=record.tags,
                hierarchy=record.hierarchy,
            )
        )
    return candidates, "lexical"


def _get_source(candidate: RagCandidate) -> str:
    """Extract source from candidate provenance (handles both object and dict)."""
    prov = candidate.provenance
    if hasattr(prov, "source"):
        return prov.source
    if isinstance(prov, dict):
        return prov.get("source", "")
    return ""


def raginfo(request: RagInfoRequest) -> RagInfoResponse:
    """Execute semantic or lexical search over the corpus."""
    mode = "faiss"
    try:
        store = _get_faiss_store()
        candidates = store.search(request.query, top_k=request.top_k)
    except Exception as exc:
        logger.warning("FAISS search failed, falling back to lexical: %s", exc)
        candidates, mode = _lexical_search(request)

    # Apply filters (already applied in lexical search, but needed for FAISS)
    if mode == "faiss":
        if request.sources:
            allowed = set(request.sources)
            candidates = [c for c in candidates if _get_source(c) in allowed]

        if request.tags:
            candidates = [
                c for c in candidates
                if hasattr(c, "tags") and c.tags and _match_tags(c.tags, request.tags)
            ]

        if request.hierarchy_prefix:
            candidates = [
                c for c in candidates
                if hasattr(c, "hierarchy") and c.hierarchy and _match_hierarchy(c.hierarchy, request.hierarchy_prefix)
            ]

    # Re-rank with sequential numbers
    ranked: List[RagCandidate] = []
    for rank, candidate in enumerate(candidates, start=1):
        ranked.append(
            RagCandidate(
                chunk_id=candidate.chunk_id,
                doc_id=candidate.doc_id,
                snippet=candidate.snippet,
                rank=rank,
                provenance=candidate.provenance,
                score=candidate.score,
                tags=getattr(candidate, "tags", []),
                hierarchy=getattr(candidate, "hierarchy", []),
            )
        )

    logger.debug("raginfo returned %d candidates (mode=%s)", len(ranked), mode)
    return RagInfoResponse(candidates=ranked)
