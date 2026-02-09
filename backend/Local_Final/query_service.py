import numpy as np
from faiss_index import load_faiss_index
from config import TOP_K
from sentence_transformers import CrossEncoder

# Load FAISS index and metadata once
index, ids, raw_data, metadata = load_faiss_index()

# Load CrossEncoder reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def compute_keyword_tag_boost(query_text, item):
    query_tokens = query_text.lower().split()
    text = item.get("text", "").lower()
    tags = [t.lower() for t in item.get("tags", [])]

    boost = 0.0

    # Keyword match in text → small boost
    for token in query_tokens:
        if token in text:
            boost += 0.1

    # Tag match → higher boost
    for token in query_tokens:
        if token in tags:
            boost += 0.3

    return boost

def search(query_vector, query_text, k=TOP_K, initial_k=None):
    """
    1. FAISS semantic retrieval
    2. Keyword/tag boosting
    3. CrossEncoder rerank
    4. Final score fusion & sort
    """

    if initial_k is None:
        initial_k = max(50, k * 5)  # Retrieve more candidates for hybrid rerank

    # Step 1: FAISS semantic retrieval
    query_vector = np.array([query_vector], dtype='float32')
    D, I = index.search(query_vector, initial_k)

    candidates = []
    for dist, idx in zip(D[0], I[0]):
        faiss_sim = 1 / (1 + dist)
        item = {
            **raw_data[idx],
            "faiss_score": float(faiss_sim)
        }
        candidates.append(item)

    # Step 2: Keyword / tag boosting
    for c in candidates:
        boost = compute_keyword_tag_boost(query_text, c)
        c["boost_score"] = boost

    # Step 3: CrossEncoder rerank
    pairs = [(query_text, c.get("text", "")) for c in candidates]
    rerank_scores = reranker.predict(pairs)
    for c, score in zip(candidates, rerank_scores):
        c["rerank_score"] = float(score)

    # Step 4: Final score fusion
    for c in candidates:
        c["final_score"] = (
            0.6 * c["rerank_score"] +
            0.2 * c["faiss_score"] +
            0.2 * c["boost_score"]
        )

    # Step 5: Sort by final_score
    ranked = sorted(candidates, key=lambda x: x["final_score"], reverse=True)

    return ranked[:k]
