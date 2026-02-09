from __future__ import annotations

import json
from pathlib import Path

from profectus_ai.config import LOCAL_FINAL_DATA_JSON, LOCAL_FINAL_FAISS_METADATA


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_ids(items: list[dict]) -> set[str]:
    return {str(item.get("id")) for item in items if item.get("id")}


def _count_by_source(items: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        source = item.get("source", "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


def diagnose(
    *,
    corpus_path: Path = LOCAL_FINAL_DATA_JSON,
    metadata_path: Path = LOCAL_FINAL_FAISS_METADATA,
    limit: int = 10,
) -> int:
    if not corpus_path.exists():
        print(f"[drift] corpus not found: {corpus_path}")
        return 2
    if not metadata_path.exists():
        print(f"[drift] metadata not found: {metadata_path}")
        return 2

    corpus = _load_json(corpus_path)
    metadata_payload = _load_json(metadata_path)
    faiss_data = metadata_payload.get("data", [])

    corpus_ids = _extract_ids(corpus)
    faiss_ids = _extract_ids(faiss_data)

    missing_in_faiss = sorted(corpus_ids - faiss_ids)
    missing_in_corpus = sorted(faiss_ids - corpus_ids)

    print("[drift] corpus entries:", len(corpus))
    print("[drift] faiss entries:", len(faiss_data))
    print("[drift] corpus by source:", _count_by_source(corpus))
    print("[drift] faiss by source:", _count_by_source(faiss_data))
    print(f"[drift] missing in faiss: {len(missing_in_faiss)}")
    print(f"[drift] missing in corpus: {len(missing_in_corpus)}")

    if missing_in_faiss:
        print("[drift] sample missing in faiss:")
        for chunk_id in missing_in_faiss[:limit]:
            print(f"  - {chunk_id}")
    if missing_in_corpus:
        print("[drift] sample missing in corpus:")
        for chunk_id in missing_in_corpus[:limit]:
            print(f"  - {chunk_id}")

    return 0


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose FAISS metadata drift vs corpus.")
    parser.add_argument("--corpus", type=Path, default=LOCAL_FINAL_DATA_JSON)
    parser.add_argument("--metadata", type=Path, default=LOCAL_FINAL_FAISS_METADATA)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    raise SystemExit(
        diagnose(
            corpus_path=args.corpus,
            metadata_path=args.metadata,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()
