from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from profectus_ai.models import IndexEntry, RagCandidate


@dataclass
class DocCandidate:
    doc_id: str
    source: str
    title: str
    score: float = 0.0
    ranks: list[int] = field(default_factory=list)
    snippets: list[str] = field(default_factory=list)


def _rrf_score(rank: int, *, k: int = 60) -> float:
    return 1.0 / (k + rank)


def infer_preferred_source(query: str) -> Optional[str]:
    """Infer which source should be favored for this query."""
    normalized = query.lower()
    youtube_terms = [
        "youtube",
        "video",
        "tutorial",
        "walkthrough",
        "webinar",
        "course",
        "demo video",
        "watch",
    ]
    help_terms = [
        "password",
        "login",
        "account",
        "billing",
        "subscription",
        "refund",
        "invoice",
        "cancel",
        "pricing",
        "support",
        "help center",
    ]
    if any(term in normalized for term in youtube_terms):
        return "YouTube"
    if any(term in normalized for term in help_terms):
        return "Help Center"
    return None


def compute_source_weights(
    candidates_by_source: Dict[str, List[RagCandidate]],
    *,
    preferred_source: Optional[str] = None,
    min_weight: float = 0.4,
    preferred_boost: float = 0.2,
    non_preferred_penalty: float = 0.9,
) -> Dict[str, float]:
    """Compute a weight per source based on result counts and preference."""
    if preferred_source and not candidates_by_source.get(preferred_source):
        preferred_source = None
    counts = {source: len(candidates) for source, candidates in candidates_by_source.items()}
    max_count = max(counts.values()) if counts else 0
    weights: Dict[str, float] = {}
    for source, count in counts.items():
        if count == 0 or max_count == 0:
            weights[source] = 0.0
            continue
        base = count / max_count
        weight = max(min_weight, base)
        if preferred_source:
            if source == preferred_source:
                weight = min(1.0, weight + preferred_boost)
            else:
                weight = max(0.0, weight * non_preferred_penalty)
        weights[source] = weight
    return weights


def merge_rag_candidates(
    candidates_by_source: Dict[str, List[RagCandidate]],
    source_weights: Dict[str, float],
) -> List[RagCandidate]:
    """Merge rag candidates with source-weighted ranks."""
    merged: List[tuple[float, RagCandidate]] = []
    for source, candidates in candidates_by_source.items():
        weight = max(source_weights.get(source, 1.0), 0.0)
        if weight <= 0:
            continue
        for candidate in candidates:
            effective_rank = candidate.rank / weight
            merged.append((effective_rank, candidate))

    merged.sort(key=lambda item: item[0])
    ranked: List[RagCandidate] = []
    for rank, (_, candidate) in enumerate(merged, start=1):
        ranked.append(candidate.model_copy(update={"rank": rank}))
    return ranked


def fuse_candidates(
    index_entries: Iterable[IndexEntry],
    rag_candidates: Iterable[RagCandidate],
    *,
    rrf_k: int = 60,
    top_k: int = 10,
) -> List[DocCandidate]:
    doc_map: dict[str, DocCandidate] = {}

    for rank, entry in enumerate(index_entries, start=1):
        score = _rrf_score(rank, k=rrf_k)
        doc = doc_map.get(entry.doc_id)
        if not doc:
            doc = DocCandidate(
                doc_id=entry.doc_id,
                source=entry.source,
                title=entry.title,
            )
            doc_map[entry.doc_id] = doc
        doc.score += score
        doc.ranks.append(rank)

    for candidate in rag_candidates:
        score = _rrf_score(candidate.rank, k=rrf_k)
        doc = doc_map.get(candidate.doc_id)
        if not doc:
            doc = DocCandidate(
                doc_id=candidate.doc_id,
                source=candidate.provenance.source,
                title=candidate.provenance.url,
            )
            doc_map[candidate.doc_id] = doc
        doc.score += score
        doc.ranks.append(candidate.rank)
        doc.snippets.append(candidate.snippet)

    ranked = sorted(doc_map.values(), key=lambda item: item.score, reverse=True)
    return ranked[:top_k]


def select_top_rag_candidates(
    rag_candidates: Iterable[RagCandidate], *, top_k: int = 8
) -> List[RagCandidate]:
    """Select top candidates, ensuring basic source coverage when available."""
    seen = set()
    selected: List[RagCandidate] = []
    by_source: Dict[str, List[RagCandidate]] = {}

    for candidate in rag_candidates:
        source = candidate.provenance.source if candidate.provenance else ""
        by_source.setdefault(source, []).append(candidate)

    # Ensure at least one per source if present
    for source, candidates in by_source.items():
        for candidate in candidates:
            if candidate.doc_id in seen:
                continue
            selected.append(candidate)
            seen.add(candidate.doc_id)
            break

    # Fill remaining slots by rank order
    for candidate in rag_candidates:
        if len(selected) >= top_k:
            break
        if candidate.doc_id in seen:
            continue
        selected.append(candidate)
        seen.add(candidate.doc_id)

    return selected[:top_k]
