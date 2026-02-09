"""
Build FAISS index from the unified corpus with timestamp metadata.

This script creates the FAISS vector index for RAG search.
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from profectus_ai.config import (
    LOCAL_FINAL_DATA_JSON,
    LOCAL_FINAL_FAISS_INDEX,
    LOCAL_FINAL_FAISS_METADATA,
)


def build_faiss_index(
    corpus_path: Path = LOCAL_FINAL_DATA_JSON,
    index_path: Path = LOCAL_FINAL_FAISS_INDEX,
    metadata_path: Path = LOCAL_FINAL_FAISS_METADATA,
):
    """Build FAISS index from the unified corpus."""

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    print(f"Loading corpus from: {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"Found {len(raw_data)} entries in corpus")

    embeddings = []
    ids = []
    metadata = []
    source_counts = {}

    for item in raw_data:
        # Skip entries without embeddings
        if "embedding_vector" not in item:
            print(f"  Warning: No embedding for {item.get('id', 'unknown')}")
            continue
        if not item.get("embedding_vector"):
            print(f"  Warning: Empty embedding for {item.get('id', 'unknown')}")
            continue

        embeddings.append(item["embedding_vector"])
        ids.append(item["id"])
        source = item.get("source", "Help Center")
        source_counts[source] = source_counts.get(source, 0) + 1

        # Extract metadata with timestamp info
        item_metadata = item.get("metadata", {}) or {}

        # Build rich metadata for search results
        metadata.append({
            "id": item.get("id"),
            "source": source,
            "title": item_metadata.get("title") or item.get("title", ""),
            "url": item_metadata.get("url") or item.get("url", ""),
            "video_id": item_metadata.get("video_id", ""),
            "platform": item_metadata.get("platform", "YouTube"),
            "time_start": item_metadata.get("time_start"),
            "time_end": item_metadata.get("time_end"),
            "timestamp": item_metadata.get("timestamp"),
            "duration": item_metadata.get("duration"),
            "tags": item_metadata.get("tags", []),
        })

    # Ensure consistent embedding dimensions
    if not embeddings:
        raise ValueError("No embeddings found in corpus")

    dim_set = set(len(e) for e in embeddings)
    if len(dim_set) != 1:
        raise ValueError(f"Inconsistent embedding dimensions: {dim_set}")
    dimension = dim_set.pop()

    print(f"Embedding dimension: {dimension}")
    print(f"Building index with {len(embeddings)} vectors...")
    print(f"Source counts: {source_counts}")

    embedding_matrix = np.array(embeddings, dtype='float32')

    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)

    print(f"FAISS index created with {index.ntotal} vectors")

    # Save index and metadata
    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))
    print(f"FAISS index saved to: {index_path}")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({"data": raw_data, "metadata": metadata}, f, ensure_ascii=False, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    return index, metadata


def main():
    """Build FAISS index from the unified corpus."""
    import argparse

    parser = argparse.ArgumentParser(description="Build FAISS index from unified corpus")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=LOCAL_FINAL_DATA_JSON,
        help="Path to corpus JSON file",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=LOCAL_FINAL_FAISS_INDEX,
        help="Path to save FAISS index",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=LOCAL_FINAL_FAISS_METADATA,
        help="Path to save metadata JSON",
    )

    args = parser.parse_args()

    build_faiss_index(
        corpus_path=args.corpus,
        index_path=args.index,
        metadata_path=args.metadata,
    )


if __name__ == "__main__":
    main()
