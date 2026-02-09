from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

DOCS_SCRAPER_DB = REPO_ROOT / "profectus docs scraper" / "data" / "profectus_docs.sqlite"
LOCAL_FINAL_DIR = REPO_ROOT / "Local_Final"
LOCAL_FINAL_DATA_JSON = LOCAL_FINAL_DIR / "data.json"
LOCAL_FINAL_FAISS_INDEX = LOCAL_FINAL_DIR / "faiss_index.index"
LOCAL_FINAL_FAISS_METADATA = LOCAL_FINAL_DIR / "faiss_metadata.json"

YOUTUBE_DIR = REPO_ROOT / "youtube"
YOUTUBE_JSON_PATHS = [
    YOUTUBE_DIR / "final_data_with_timestamps.json",
    YOUTUBE_DIR / "final_data_improved_fixed_clean.json",  # Fallback
]

INDEX_OVERVIEW_DIR = DATA_DIR
INDEX_OVERVIEW_PATH = INDEX_OVERVIEW_DIR / "index_overview.jsonl"
INDEX_OVERVIEW_META_PATH = INDEX_OVERVIEW_DIR / "index_overview.meta.json"
INDEX_OVERVIEW_COVERAGE_PATH = INDEX_OVERVIEW_DIR / "index_overview.coverage.json"

YOUTUBE_RAW_DIR = DATA_DIR / "youtube_raw"

SCHEMA_VERSION = "1.0.0"

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
