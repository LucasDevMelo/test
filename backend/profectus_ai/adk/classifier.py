"""Query classification for intelligent routing.

Classifies user queries to determine how the agent should respond:
- TRIVIAL: Greetings, acknowledgments - respond directly
- FACTUAL: Questions about Profectus - must retrieve evidence
- HIGH_RISK: Sensitive topics - retrieve + flag for escalation
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Tuple


class QueryType(Enum):
    """Classification of user query intent."""

    TRIVIAL = "trivial"
    FACTUAL = "factual"
    HIGH_RISK = "high_risk"


# Patterns that indicate trivial queries (greetings, acknowledgments)
TRIVIAL_PATTERNS = [
    # Greetings with optional filler words
    r"^(hi|hello|hey|hiya|howdy)(\s+(there|everyone|all|folks))?[\s\.\!\?]*$",
    r"^(good\s+(morning|afternoon|evening|day))[\s\.\!\?]*$",
    # Thanks with optional additions
    r"^(thanks|thank you|ty|thx)(\s+(so much|very much|a lot))?[\s\.\!\?]*$",
    # Acknowledgments
    r"^(ok|okay|sure|got it|great|perfect|awesome|cool|nice|sounds good)[\s\.\!\?]*$",
    # Farewells
    r"^(bye|goodbye|see you|cya|later|cheers)[\s\.\!\?]*$",
    # Simple yes/no
    r"^(yes|no|yep|nope|yeah|nah)[\s\.\!\?]*$",
    # Short filler or reactions
    r"^(lol|lmao|rofl|haha|hahaha|hehe|hmm|huh|wat|wut|what|k|okey)[\s\.\!\?]*$",
    # Punctuation-only reactions
    r"^[\?\!\.]{1,5}$",
]

INTERROGATIVE_PREFIXES = {
    "what",
    "whats",
    "what's",
    "how",
    "where",
    "when",
    "why",
    "who",
    "which",
    "can",
    "could",
    "do",
    "does",
    "did",
    "is",
    "are",
    "should",
    "would",
    "will",
    "may",
    "might",
}

DOMAIN_KEYWORDS = {
    "profectus",
    "mt5",
    "metatrader",
    "bot",
    "strategy",
    "trading",
    "trade",
    "expert advisor",
    "ea",
    "block",
    "builder",
    "library",
    "portfolio",
    "template",
    "indicator",
    "formula",
    "variable",
    "order",
    "stop loss",
    "take profit",
    "password",
    "login",
    "account",
    "subscription",
    "billing",
    "refund",
    "dashboard",
}

# Keywords that indicate high-risk topics requiring escalation
HIGH_RISK_KEYWORDS = {
    # Billing and payments
    "billing": "billing_inquiry",
    "payment": "payment_issue",
    "refund": "refund_request",
    "charge": "billing_inquiry",
    "invoice": "billing_inquiry",
    "subscription": "subscription_change",
    "cancel subscription": "subscription_change",
    "cancel my account": "account_deletion",
    # Account management
    "delete account": "account_deletion",
    "delete my account": "account_deletion",
    "close account": "account_deletion",
    "change email": "account_change",
    "change password": "account_change",
    "reset password": "account_change",
    # Legal and complaints
    "legal": "legal_inquiry",
    "lawsuit": "legal_inquiry",
    "lawyer": "legal_inquiry",
    "attorney": "legal_inquiry",
    "complaint": "formal_complaint",
    "sue": "legal_inquiry",
    # Financial advice (outside scope)
    "investment advice": "investment_advice",
    "financial advice": "investment_advice",
    "should i buy": "investment_advice",
    "should i sell": "investment_advice",
    "stock recommendation": "investment_advice",
}

# Keywords indicating user wants human assistance
HUMAN_REQUEST_KEYWORDS = [
    "human",
    "real person",
    "agent",
    "customer service",
    "support agent",
    "talk to someone",
    "speak to someone",
    "representative",
    "manager",
    "supervisor",
]


def classify_query(query: str) -> Tuple[QueryType, str]:
    """Classify a query and return its type with reason.

    Args:
        query: The user's query text.

    Returns:
        Tuple of (QueryType, reason_string).
        Reason explains why this classification was chosen.
    """
    normalized = query.strip().lower()

    # Empty queries
    if not normalized:
        return QueryType.TRIVIAL, "empty_query"

    # Check trivial patterns (greetings, acknowledgments)
    for pattern in TRIVIAL_PATTERNS:
        if re.match(pattern, normalized, re.IGNORECASE):
            return QueryType.TRIVIAL, "greeting_or_acknowledgment"

    # Off-topic or non-question statements with no domain signal
    if "?" not in normalized and len(normalized) <= 140:
        starts_with_interrogative = any(
            normalized.startswith(prefix + " ") or normalized == prefix
            for prefix in INTERROGATIVE_PREFIXES
        )
        if not starts_with_interrogative:
            if not any(keyword in normalized for keyword in DOMAIN_KEYWORDS):
                return QueryType.TRIVIAL, "off_topic_statement"

    # Check high-risk keywords
    for keyword, reason in HIGH_RISK_KEYWORDS.items():
        if keyword in normalized:
            return QueryType.HIGH_RISK, reason

    # Check human request keywords
    for keyword in HUMAN_REQUEST_KEYWORDS:
        if keyword in normalized:
            return QueryType.HIGH_RISK, "human_request"

    # Default to factual - requires evidence retrieval
    return QueryType.FACTUAL, "requires_evidence"


def get_classification_guidance(query_type: QueryType, reason: str) -> str:
    """Get guidance text for the agent based on classification.

    Returns instruction text telling the agent how to handle
    this type of query.
    """
    if query_type == QueryType.TRIVIAL:
        return (
            "This is a trivial query (greeting/acknowledgment). "
            "Respond politely without retrieving evidence."
        )

    if query_type == QueryType.HIGH_RISK:
        guidance_map = {
            "billing_inquiry": "This is a billing inquiry. Retrieve relevant billing documentation, but note this may require human escalation.",
            "payment_issue": "This is a payment issue. Retrieve FAQ info, but prepare to escalate to support.",
            "refund_request": "This is a refund request. This MUST be escalated to human support.",
            "subscription_change": "This is a subscription change request. May require human verification.",
            "account_deletion": "This is an account deletion request. MUST be escalated to human support.",
            "account_change": "This is an account change request. Retrieve how-to docs if available.",
            "legal_inquiry": "This is a legal inquiry. DO NOT provide legal advice. Escalate immediately.",
            "formal_complaint": "This is a formal complaint. Document and escalate to support team.",
            "investment_advice": "This involves financial advice which is outside our scope. Politely decline and clarify our service.",
            "human_request": "User explicitly requested human assistance. Acknowledge and prepare escalation.",
        }
        return guidance_map.get(reason, "High-risk query. Proceed carefully and prepare for escalation.")

    # FACTUAL
    return (
        "This is a factual query about Profectus. "
        "You MUST retrieve evidence before answering. "
        "Use indexinfo and raginfo results, then call read_spans for exact text."
    )
