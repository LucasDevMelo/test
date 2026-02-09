"""Evidence span building from corpus data.

This module provides the build_span function for constructing EvidenceSpan
objects from corpus records. For the CorpusStore and CorpusRecord classes,
use the centralized store module.
"""
from __future__ import annotations

from typing import Optional

from profectus_ai.evidence.store import CorpusRecord, CorpusStore, get_corpus_store
from profectus_ai.models import EvidenceSpan, Provenance

# Re-export for backwards compatibility
__all__ = ["CorpusRecord", "CorpusStore", "build_span", "get_corpus_store"]


def build_span(
    store: CorpusStore,
    *,
    chunk_id: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
    time_start: Optional[float] = None,
    time_end: Optional[float] = None,
) -> EvidenceSpan:
    record = store.get(chunk_id)
    if record is None:
        raise KeyError(f"Unknown chunk_id: {chunk_id}")

    text = record.text
    if start is not None and end is not None:
        text = text[start:end]

    if time_start is None:
        time_start = record.time_start
    if time_end is None:
        time_end = record.time_end

    provenance = Provenance(
        source=record.source,
        url=record.url,
        timestamp_start=time_start,
        timestamp_end=time_end,
    )
    span_id = f"{chunk_id}::span"
    return EvidenceSpan(
        span_id=span_id,
        doc_id=record.doc_id,
        chunk_id=record.chunk_id,
        text=text,
        start=start,
        end=end,
        time_start=time_start,
        time_end=time_end,
        provenance=provenance,
    )

