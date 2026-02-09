import re
import ftfy
from sentence_transformers import SentenceTransformer, util
from config import (
    DOC_CONTEXT_THRESHOLD,
    DOC_CITATION_THRESHOLD,
    DEFAULT_EMBEDDING_MODEL_NAME,
)

default_model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL_NAME)

def clean_text(text: str) -> str:
    """Fix encoding, remove excess spaces, and trim pipes or bars."""
    text = ftfy.fix_text(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" |")


def deduplicate_by_semantics(chunks, model, threshold=0.95):
    """
    Removes redundant text chunks that are semantically very similar.
    Uses cosine similarity between embeddings to detect duplicates.
    """
    if len(chunks) <= 1:
        return chunks

    embeddings = model.encode([c["text"] for c in chunks])
    keep, keep_embeddings = [], []

    for i, emb in enumerate(embeddings):
        if not keep_embeddings:
            keep.append(chunks[i])
            keep_embeddings.append(emb)
            continue

        sims = util.cos_sim(emb, keep_embeddings).flatten()
        if all(s < threshold for s in sims):
            keep.append(chunks[i])
            keep_embeddings.append(emb)

    return keep


def build_context_layer(results, query=None, max_chars_per_doc=1500, embedder=None):
    """
    Builds the context layer for the LLM.

    Responsibilities:
    - Prepare high-quality context for the LLM
    - Decide whether the system has enough context to answer
    - Decide which documents are eligible for citation
    (Escalation decisions are handled elsewhere)
    """
    model = embedder or default_model

    if not results:
        return {
            "context_text": "No relevant context found.",
            "sources": [],
            "has_context": False
        }

    # Step 1: Clean and normalize results
    cleaned_results = []
    for r in results:
        text = clean_text(r.get("text", ""))
        if len(text) < 50:
            continue

        cleaned_results.append({
            "text": text,
            "url": r.get("url", ""),
            "meta": r.get("metadata", {}),
            "id": r.get("id", "unknown"),
            "score": r.get("score", 0),
            "faiss_score": r.get("faiss_score", 0),
            "rerank_score": r.get("rerank_score", 0)
        })

    if not cleaned_results:
        return {
            "context_text": "No usable text after cleaning.",
            "sources": [],
            "has_context": False
        }

    # Step 2: Group chunks by document ID
    doc_groups = {}
    for chunk in cleaned_results:
        doc_id = chunk["id"].split("_chunk")[0]
        doc_groups.setdefault(doc_id, []).append(chunk)

    # Step 3: Deduplicate and merge chunks per document
    merged_docs = []
    for doc_id, chunks in doc_groups.items():
        deduped_chunks = deduplicate_by_semantics(
            chunks, model=model, threshold=0.95
        )

        merged_text = " ".join(c["text"] for c in deduped_chunks)
        if len(merged_text) > max_chars_per_doc:
            merged_text = merged_text[:max_chars_per_doc] + "\n...[truncated]..."

        first_chunk = deduped_chunks[0]

        merged_docs.append({
            "id": first_chunk["id"],
            "url": first_chunk["url"],
            "meta": first_chunk["meta"],
            "text": merged_text,
            "score": max(c["score"] for c in deduped_chunks),
            "rerank_score": max(c["rerank_score"] for c in deduped_chunks),
            "faiss_score": max(c["faiss_score"] for c in deduped_chunks)
        })

    # Step 4: Context inclusion gate (NOT escalation)
    context_docs = [
        doc for doc in merged_docs
        if doc.get("rerank_score", 0.0) >= DOC_CONTEXT_THRESHOLD
    ]

    if not context_docs:
        return {
            "context_text": "No documents passed the context relevance threshold.",
            "sources": [],
            "has_context": False
        }

    # Step 5: Sort documents by rerank score (highest first)
    context_docs.sort(
        key=lambda x: x.get("rerank_score", 0),
        reverse=True
    )

    # Step 6: Mark citation eligibility
    for doc in context_docs:
        doc["can_cite"] = doc.get("rerank_score", 0.0) >= DOC_CITATION_THRESHOLD

    # Step 7: Build formatted context blocks for LLM
    blocks = []
    for r in context_docs:
        block = (
            f"[Document]\n"
            f"url: {r['url']}\n"
            f"category: {r['meta'].get('category', 'N/A')}\n"
            f"tags: {', '.join(r['meta'].get('tags', []))}\n"
            f"content: {r['text']}\n"
            f"[/Document]\n"
        )
        blocks.append(block)

    return {
        "context_text": "\n".join(blocks),
        "sources": [
            {"url": r["url"]}
            for r in context_docs
            if r.get("can_cite")
        ],
        "has_context": True
    }
