from __future__ import annotations

from typing import List

from profectus_ai.evidence.deeplinks import build_deep_link
from profectus_ai.models import EvidenceSpan


def assemble_evidence_blocks(spans: List[EvidenceSpan]) -> List[dict]:
    blocks = []
    for span in spans:
        blocks.append(
            {
                "span_id": span.span_id,
                "doc_id": span.doc_id,
                "chunk_id": span.chunk_id,
                "source": span.provenance.source,
                "url": build_deep_link(span.provenance),
                "text": span.text,
            }
        )
    return blocks
