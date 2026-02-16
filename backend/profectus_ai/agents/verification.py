"""Evidence verification with confidence scoring.

Verifies that evidence is sufficient to answer a query and
computes a confidence score based on coverage metrics.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from profectus_ai.models import EvidenceSpan


@dataclass(frozen=True)
class VerificationResult:
    """Result of evidence verification.

    Attributes:
        ok: Whether evidence is sufficient.
        confidence: Confidence score from 0.0 to 1.0.
        reason: Explanation if verification failed.
        coverage_details: Detailed metrics about evidence coverage.
    """

    ok: bool
    confidence: float = 0.0
    reason: Optional[str] = None
    coverage_details: Dict[str, object] = field(default_factory=dict)


def verify_evidence(
    spans: List[EvidenceSpan],
    query: str = "",
    answer: str = "", 
    confidence_threshold: float = 0.4,
) -> VerificationResult:
    """Verify evidence coverage and compute confidence score."""
    if not spans:
        return VerificationResult(
            ok=False,
            confidence=0.0,
            reason="no_evidence",
            coverage_details={"span_count": 0},
        )

    density_score = _compute_span_score(spans)
    
    diversity_score = _compute_diversity_score(spans)
    
    relevance_score = _compute_relevance_score(spans, query) if query else 0.5
    
    faithfulness_score = _compute_faithfulness_score(answer, spans) if answer else 1.0

    confidence = (
        relevance_score * 0.45 +
        faithfulness_score * 0.30 +
        diversity_score * 0.15 +
        density_score * 0.10
    )

    confidence = max(0.0, min(1.0, confidence))

    ok = confidence >= confidence_threshold
    reason = None if ok else "low_confidence"

    coverage_details = {
        "span_count": len(spans),
        "sources": list(_get_sources(spans)),
        "relevance_score": round(relevance_score, 3),     # 45%
        "faithfulness_score": round(faithfulness_score, 3), # 30%
        "diversity_score": round(diversity_score, 3),     # 15%
        "density_score": round(density_score, 3),         # 10%
        "confidence_threshold": confidence_threshold,
    }

    return VerificationResult(
        ok=ok,
        confidence=round(confidence, 3),
        reason=reason,
        coverage_details=coverage_details,
    )


def _compute_span_score(spans: List[EvidenceSpan]) -> float:
    """Score based on number of spans (diminishing returns)."""
    # Cap at 5 spans for max score
    return min(len(spans) / 5.0, 1.0)


def _compute_diversity_score(spans: List[EvidenceSpan]) -> float:
    """Score based on source diversity."""
    sources = _get_sources(spans)
    # Two sources (Help Center + YouTube) = max score
    return len(sources) / 2.0


def _get_sources(spans: List[EvidenceSpan]) -> Set[str]:
    """Extract unique sources from spans (Handles objects and dicts)."""
    sources = set()
    for s in spans:
        if not s.provenance:
            continue
            
        # Tenta acessar como objeto (.source)
        if hasattr(s.provenance, "source"):
            sources.add(s.provenance.source)
        # Tenta acessar como dicionário (['source'])
        elif isinstance(s.provenance, dict) and "source" in s.provenance:
            sources.add(s.provenance["source"])
        # Fallback
        else:
            sources.add("Unknown")
            
    return sources


def _compute_relevance_score(spans: List[EvidenceSpan], query: str) -> float:
    """Score based on keyword overlap between query and evidence."""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.5  # Neutral score if no query

    # Combine all span text
    evidence_text = " ".join(s.text for s in spans)
    evidence_tokens = set(_tokenize(evidence_text))

    # Compute overlap
    overlap_count = sum(1 for t in query_tokens if t in evidence_tokens)
    overlap_ratio = overlap_count / len(query_tokens)

    # Scale up slightly (overlap > 0.5 is good)
    return min(overlap_ratio * 1.5, 1.0)


def _tokenize(text: str) -> List[str]:
    """Extract lowercase word tokens."""
    return [t.lower() for t in re.findall(r"\w+", text) if len(t) > 2]


def map_claims_to_spans(
    claims: List[str],
    spans: List[EvidenceSpan],
) -> Dict[str, List[str]]:
    """Map claims to supporting span IDs.

    For now, this is a simple mapping where each claim maps to
    all available spans. Future versions could use semantic matching.

    Args:
        claims: List of claim strings to verify.
        spans: List of evidence spans.

    Returns:
        Dict mapping claim -> list of supporting span_ids.
    """
    span_ids = [span.span_id for span in spans]
    return {claim: span_ids for claim in claims}


def extract_claims(answer: str) -> List[str]:
    """Extract factual claims from an answer.

    Simple heuristic: split into sentences and filter.
    Future versions could use NLP for better extraction.

    Args:
        answer: The generated answer text.

    Returns:
        List of claim strings.
    """
    # Split on sentence boundaries
    sentences = re.split(r"[.!?]+", answer)

    # Filter to likely factual claims (not questions, not very short)
    claims = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20:
            continue
        if sentence.endswith("?"):
            continue
        # Skip meta-sentences
        if any(
            phrase in sentence.lower()
            for phrase in ["i couldn't find", "would you like", "let me know"]
        ):
            continue
        claims.append(sentence)

    return claims


def _compute_faithfulness_score(answer: str, spans: List[EvidenceSpan]) -> float:
    """It calculates whether the words in the answer exist in the evidence (Fidelity)."""
    if not answer or not spans:
        return 0.5
    
    answer_tokens = set(_tokenize(answer))
    if not answer_tokens:
        return 1.0
    evidence_text = " ".join(str(s.text) for s in spans if s.text)    
    evidence_tokens = set(_tokenize(evidence_text))
    
    supported_tokens = sum(1 for t in answer_tokens if t in evidence_tokens)
    
    return supported_tokens / len(answer_tokens)
