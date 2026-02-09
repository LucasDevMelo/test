from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


HIGH_RISK_KEYWORDS = {
    "investment_advice": [
        "investment advice",
        "stock recommendation",
        "buy stock",
        "financial advice",
    ],
    "legal_advice": [
        "legal advice",
        "contract",
        "law",
        "lawyer",
    ],
    "account_change": [
        "change my account",
        "update account",
        "account modification",
    ],
    "billing": [
        "billing",
        "charged",
        "invoice",
        "payment",
    ],
}

HUMAN_REQUEST_KEYWORDS = [
    "human",
    "agent",
    "real person",
    "customer service",
    "support agent",
    "talk to someone",
]


@dataclass(frozen=True)
class EscalationDecision:
    should_escalate: bool
    reasons: List[str]
    detail: Optional[str] = None


def detect_high_risk(query: str) -> Optional[str]:
    normalized = query.lower()
    for category, keywords in HIGH_RISK_KEYWORDS.items():
        for keyword in keywords:
            if keyword in normalized:
                return category
    return None


def detect_human_request(query: str) -> bool:
    normalized = query.lower()
    return any(keyword in normalized for keyword in HUMAN_REQUEST_KEYWORDS)
