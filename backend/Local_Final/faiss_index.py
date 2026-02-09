import faiss
import numpy as np
import json
from config import FAISS_INDEX_PATH, METADATA_PATH, DATA_JSON_PATH

def build_faiss_index():
    # Load your JSON data
    with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    embeddings, ids, metadata = [], [], []

    for item in raw_data:
        embeddings.append(item["embedding_vector"])
        ids.append(item["id"])
        # Keep only auxiliary info in metadata
        metadata.append({
            "tags": item.get("metadata", {}).get("tags", []),
            "category": item.get("metadata", {}).get("category", "")
        })

    # Ensure consistent embedding dimensions
    dim_set = set(len(e) for e in embeddings)
    if len(dim_set) != 1:
        raise ValueError(f"Inconsistent embedding dimensions: {dim_set}")
    dimension = dim_set.pop()

    embedding_matrix = np.array(embeddings, dtype='float32')

    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)
    print(f"FAISS index created with {index.ntotal} vectors")

    # Save index and metadata
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump({"ids": ids, "data": raw_data, "metadata": metadata}, f, ensure_ascii=False)
    print("FAISS index and metadata saved.")

    return index, ids, metadata

def load_faiss_index():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = data["ids"]
    raw_data = data["data"]      # Full raw data (includes text, url, etc.)
    metadata = data["metadata"]  # Only auxiliary info
    return index, ids, raw_data, metadata