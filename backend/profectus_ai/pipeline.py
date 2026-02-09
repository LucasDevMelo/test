from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from profectus_ai.evidence.context import assemble_evidence_blocks
from profectus_ai.evidence.fusion import (
    compute_source_weights,
    fuse_candidates,
    infer_preferred_source,
    merge_rag_candidates,
    select_top_rag_candidates,
)
from profectus_ai.models import IndexInfoRequest, RagInfoRequest, ReadSpansRequest
from profectus_ai.query_normalization import normalize_query
from profectus_ai.tools.indexinfo import indexinfo
from profectus_ai.tools.raginfo import raginfo
from profectus_ai.tools.read_spans import read_spans

logger = logging.getLogger(__name__)


def run_pipeline(query: str, *, top_k: int = 5) -> Dict[str, Any]:
    normalized_query = normalize_query(query)
    index_response = indexinfo(IndexInfoRequest(query=normalized_query, limit=top_k))
    rag_errors: Dict[str, str] = {}
    candidates_by_source = {}
    for source in ("Help Center", "YouTube"):
        try:
            response = raginfo(RagInfoRequest(query=normalized_query, top_k=top_k, sources=[source]))
            candidates_by_source[source] = response.candidates
        except RuntimeError as exc:
            rag_errors[source] = str(exc)
            candidates_by_source[source] = []

    preferred_source = infer_preferred_source(query)
    source_weights = compute_source_weights(
        candidates_by_source,
        preferred_source=preferred_source,
    )
    logger.info(
        "raginfo counts by source: %s (preferred=%s)",
        {k: len(v) for k, v in candidates_by_source.items()},
        preferred_source,
    )
    logger.info("raginfo source weights: %s", source_weights)
    rag_candidates = merge_rag_candidates(candidates_by_source, source_weights)
    fused = fuse_candidates(index_response.items, rag_candidates, top_k=top_k)
    top_rag = select_top_rag_candidates(rag_candidates, top_k=top_k)

    spans_response = read_spans(ReadSpansRequest(chunk_ids=[c.chunk_id for c in top_rag]))
    evidence_blocks = assemble_evidence_blocks(spans_response.spans)

    return {
        "query": query,
        "indexinfo": [item.model_dump() for item in index_response.items],
        "raginfo": [item.model_dump() for item in rag_candidates],
        "raginfo_by_source": {
            source: [item.model_dump() for item in candidates]
            for source, candidates in candidates_by_source.items()
        },
        "raginfo_preferred_source": preferred_source,
        "raginfo_source_weights": source_weights,
        "raginfo_errors": rag_errors,
        "fused_docs": [doc.__dict__ for doc in fused],
        "evidence": evidence_blocks,
    }


def _write_output(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a minimal agentic retrieval pipeline.")
    parser.add_argument("query", help="Query text")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default="data/pipeline_output.json")
    args = parser.parse_args()

    result = run_pipeline(args.query, top_k=args.top_k)
    _write_output(result, Path(args.out))
    print(f"[pipeline] wrote {args.out}")
