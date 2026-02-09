# faiss_index.py
FAISS_INDEX_PATH = "faiss_index.index"
METADATA_PATH = "faiss_metadata.json"
DATA_JSON_PATH = "data.json"


# query_service.py
# ---------- Final ranked retrieval output size ----------
TOP_K = 5


# llm_service.py
VERTEX_PROJECT = "profectus-student-challenge"
VERTEX_LOCATION = "us-central1"
DEFAULT_LLM_MODEL = "gemini-2.5-flash"


# context_layer.py
# ---------- Context & citation thresholds ----------
DOC_CONTEXT_THRESHOLD = 0.55      # Minimum rerank_score to include in LLM context
DOC_CITATION_THRESHOLD = 0.65     # Minimum rerank_score to allow citation

# ---------- Models ----------
DEFAULT_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# main.py
# ---------- Confidence ----------
CONFIDENCE_THRESHOLD = 0.4
NO_CITATION_CONFIDENCE_CAP = 0.6

# ---------- High-risk intents ----------
HIGH_RISK_KEYWORDS = {
    "investment_advice": [
        "investment advice",
        "stock recommendation",
        "buy stock",
        "financial advice"
    ],
    "legal_advice": [
        "legal advice",
        "contract",
        "law",
        "lawyer"
    ],
    "account_change": [
        "change my account",
        "update account",
        "account modification"
    ],
    "billing": [
        "billing",
        "charged",
        "invoice",
        "payment"
    ]
}

# ---------- Human escalation ----------
HUMAN_REQUEST_KEYWORDS = [
    "human",
    "agent",
    "real person",
    "customer service",
    "support agent",
    "talk to someone",
]
