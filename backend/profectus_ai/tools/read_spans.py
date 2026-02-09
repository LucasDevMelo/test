"""Read evidence spans from the corpus.

This tool fetches exact text segments from the unified corpus
given a list of chunk IDs and optional text/time boundaries.
"""
from __future__ import annotations

import logging

from profectus_ai.evidence.spans import build_span
from profectus_ai.evidence.store import get_corpus_store
from profectus_ai.models import ReadSpansRequest, ReadSpansResponse

logger = logging.getLogger(__name__)


def read_spans(request: ReadSpansRequest) -> ReadSpansResponse:
    """Fetch evidence spans for the given chunk IDs."""
    store = get_corpus_store()
    spans = []
    for chunk_id in request.chunk_ids:
        try:
            span = build_span(
                store,
                chunk_id=chunk_id,
                start=request.start,
                end=request.end,
                time_start=request.time_start,
                time_end=request.time_end,
            )
            spans.append(span)
        except KeyError:
            logger.warning("read_spans skipped missing chunk_id: %s", chunk_id)
    return ReadSpansResponse(spans=spans)
