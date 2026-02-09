from __future__ import annotations

from typing import Dict, List


_SYNONYM_EXPANSIONS: Dict[str, List[str]] = {
    "condition block": ["trade rule block", "trade rule"],
    "my dashboard": ["my library", "trading bots library"],
    "trading bots dashboard": ["trading bots library", "library"],
    "pro logic blocks": ["variables", "modify variables", "formula block"],
    "one-click export": ["export", "metatrader 5", "mt5"],
    "metatrader 5": ["mt5"],
    "mt5": ["metatrader 5", "meta trader"],
}


def normalize_query(query: str) -> str:
    """Expand common product-specific terms to improve retrieval."""
    if not query:
        return query
    lowered = query.lower()
    expansions: List[str] = []
    for phrase, extra_terms in _SYNONYM_EXPANSIONS.items():
        if phrase in lowered:
            for term in extra_terms:
                if term not in lowered:
                    expansions.append(term)
    if not expansions:
        return query
    return f"{query} {' '.join(expansions)}"
