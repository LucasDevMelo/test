from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List
from urllib.parse import urlparse

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from profectus_ai.config import (
    DEFAULT_EMBEDDING_MODEL,
    LOCAL_FINAL_FAISS_INDEX,
    LOCAL_FINAL_FAISS_METADATA,
)
from profectus_ai.ids import make_chunk_id, make_doc_id
from profectus_ai.models import Provenance


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    doc_id: str
    snippet: str
    score: float
    provenance: Provenance
    tags: list[str]
    hierarchy: list[str]


class FaissStore:
    def __init__(
        self,
        *,
        index_path: Path = LOCAL_FINAL_FAISS_INDEX,
        metadata_path: Path = LOCAL_FINAL_FAISS_METADATA,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.index_path = index_path
        self.metadata_path = metadata_path
        self._index = faiss.read_index(str(index_path))
        with metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self._raw_data = payload["data"]
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        model_path = _resolve_embedding_model(embedding_model)
        self._model_path = model_path
        self._device = os.environ.get("PROFECTUS_EMBEDDING_DEVICE", "cpu")
        self._model = SentenceTransformer(model_path, local_files_only=True, device=self._device)

    def embed_query(self, query: str) -> np.ndarray:
        try:
            vector = self._model.encode(query)
        except RuntimeError as exc:
            message = str(exc)
            if "meta tensor" in message.lower() and self._device != "cpu":
                # Retry on CPU if the model failed to materialize on the original device.
                self._device = "cpu"
                self._model = SentenceTransformer(
                    self._model_path,
                    local_files_only=True,
                    device="cpu",
                )
                vector = self._model.encode(query)
            else:
                raise
        return np.array([vector], dtype="float32")

    def search(self, query: str, *, top_k: int = 10) -> List[Candidate]:
        query_vector = self.embed_query(query)
        distances, indices = self._index.search(query_vector, top_k)

        candidates: List[Candidate] = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            item = self._raw_data[idx]
            source = item.get("source", "Help Center")
            url = item.get("url") or ""
            metadata = item.get("metadata", {}) or {}
            video_id = metadata.get("video_id") or item.get("id")
            tags = metadata.get("tags", []) or []
            hierarchy = _hierarchy_from_url(url) if source != "YouTube" else ["youtube"]
            doc_id = make_doc_id(
                source=source,
                url=url if source != "YouTube" else None,
                video_id=video_id if source == "YouTube" else None,
            )
            chunk_index = int(item.get("chunk_index") or 0)
            chunk_id = make_chunk_id(doc_id, chunk_index)
            snippet = " ".join(str(item.get("text", "")).split())[:240]
            score = float(1 / (1 + dist))

            # Extract timestamp info from metadata for YouTube videos
            timestamp_start = metadata.get("time_start")
            timestamp_end = metadata.get("time_end")

            provenance = Provenance(
                source=source,
                url=url,
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
            )
            candidates.append(
                Candidate(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    snippet=snippet,
                    score=score,
                    provenance=provenance,
                    tags=list(tags),
                    hierarchy=hierarchy,
                )
            )
        return candidates


def _hierarchy_from_url(url: str) -> list[str]:
    path = urlparse(url).path
    return [part for part in path.split("/") if part]


def _resolve_embedding_model(model_name: str) -> str:
    env_path = os.environ.get("PROFECTUS_EMBEDDING_MODEL_PATH")
    if env_path:
        path = Path(env_path)
        if not path.exists():
            raise RuntimeError(f"Embedding model path not found: {env_path}")
        return str(path)

    cache_root = Path.home() / ".cache" / "torch" / "sentence_transformers"
    normalized = model_name.replace("/", "_")
    candidates = [
        cache_root / model_name,
        cache_root / normalized,
        cache_root / f"sentence-transformers_{normalized}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    model_variants = [model_name]
    if "/" not in model_name:
        model_variants.append(f"sentence-transformers/{model_name}")

    for variant in model_variants:
        hub_root = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / f"models--{variant.replace('/', '--')}"
            / "snapshots"
        )
        if hub_root.exists():
            snapshots = sorted([p for p in hub_root.iterdir() if p.is_dir()])
            if snapshots:
                return str(snapshots[0])

    raise RuntimeError(
        f"Embedding model '{model_name}' not available offline. "
        "Set PROFECTUS_EMBEDDING_MODEL_PATH or cache the model locally."
    )
