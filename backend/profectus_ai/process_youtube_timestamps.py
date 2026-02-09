"""
Process YouTube transcripts with timestamps into RAG corpus format.

Creates two types of entries:
1. Full transcript entries (for complete video context)
2. Chunked entries (for granular search with timestamp metadata)

Both types include time_start/time_end for UX (jump to video position).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Set

from sentence_transformers import SentenceTransformer

from profectus_ai.config import (
    DEFAULT_EMBEDDING_MODEL,
    YOUTUBE_DIR,
    YOUTUBE_RAW_DIR,
)
from profectus_ai.text_cleaning import clean_transcript_text

# Input/Output paths
TIMESTAMPS_DATA_DIR = YOUTUBE_RAW_DIR
OUTPUT_YOUTUBE_JSON = YOUTUBE_DIR / "final_data_with_timestamps.json"
EXISTING_CORPUS = YOUTUBE_DIR / "final_data_improved_fixed_clean.json"


class TimestampedTranscriptProcessor:
    """Process timestamped YouTube transcripts for RAG system."""

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        chunk_size: int = 1000,
    ):
        self.chunk_size = chunk_size  # Characters per chunk (simpler than token-based)

        # Load embedding model in offline mode
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.embedding_model = SentenceTransformer(embedding_model, local_files_only=True)

    def load_existing_video_ids(self, corpus_path: Path = EXISTING_CORPUS) -> Set[str]:
        """Load video IDs that are already in the corpus."""
        if not corpus_path.exists():
            return set()

        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus = json.load(f)

        return {item['id'] for item in corpus if 'id' in item}

    def load_existing_corpus(self, corpus_path: Path = EXISTING_CORPUS) -> List[Dict[str, Any]]:
        """Load existing corpus data."""
        if not corpus_path.exists():
            return []

        with open(corpus_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_transcripts(
        self,
        data_dir: Path = TIMESTAMPS_DATA_DIR,
        existing_ids: Set[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """Load only new transcript JSON files (not already in corpus)."""
        existing_ids = existing_ids or set()
        transcripts = []
        skipped = []

        for json_file in sorted(data_dir.glob("*.json")):
            video_id = json_file.stem
            if video_id in existing_ids:
                skipped.append(video_id)
                continue

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if self._validate_transcript_data(data):
                        transcripts.append(data)
            except Exception as e:
                print(f"  Error loading {json_file}: {e}")

        print(f"Skipped {len(skipped)} existing videos")
        return transcripts

    def _validate_transcript_data(self, data: Dict[str, Any]) -> bool:
        """Validate transcript has required fields."""
        return all(k in data for k in ['video_id', 'title', 'url', 'transcript'])

    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to YouTube timestamp format (MM:SS or HH:MM:SS)."""
        if seconds is None:
            return None
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"

    def _create_youtube_url_with_timestamp(self, base_url: str, timestamp: float) -> str:
        """Create YouTube URL with timestamp parameter."""
        if timestamp is None:
            return base_url
        return f"{base_url}&t={int(timestamp)}" if "?" in base_url else f"{base_url}?t={int(timestamp)}"

    def _get_video_duration(self, transcript: List[Dict]) -> float:
        """Get total video duration from transcript."""
        if not transcript:
            return 0
        last_entry = transcript[-1]
        return last_entry['start'] + last_entry['duration']

    def build_full_transcript_entry(
        self,
        transcript_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create a single entry with full concatenated transcript text."""
        video_id = transcript_data['video_id']
        title = transcript_data['title']
        url = transcript_data['url']
        transcript = transcript_data['transcript']

        # Concatenate all transcript text
        full_text = " ".join([clean_transcript_text(s['text']) for s in transcript])
        duration = self._get_video_duration(transcript)

        # Create metadata with timestamp range
        metadata = {
            "video_id": video_id,
            "title": title,
            "url": url,
            "platform": "YouTube",
            "duration": round(duration, 2),
            "transcript_count": len(transcript),
            "time_start": 0.0,
            "time_end": round(duration, 2),
        }

        # Generate embedding
        embedding = self.embedding_model.encode(full_text, show_progress_bar=False).tolist()

        return {
            "id": video_id,
            "source": "YouTube",
            "title": title,
            "url": url,
            "text": full_text,
            "chunk_index": 0,
            "embedding_vector": embedding,
            "metadata": metadata,
        }

    def build_chunked_entries(
        self,
        transcript_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Create time-based chunks with timestamp ranges."""
        video_id = transcript_data['video_id']
        title = transcript_data['title']
        url = transcript_data['url']
        transcript = transcript_data['transcript']

        if not transcript:
            return []

        duration = self._get_video_duration(transcript)
        num_chunks = max(3, int(duration / 120))  # ~2 minutes per chunk
        chunk_duration = duration / num_chunks

        entries = []

        for chunk_idx in range(num_chunks):
            time_start = chunk_idx * chunk_duration
            time_end = (chunk_idx + 1) * chunk_duration

            # Find transcript segments within this time range
            segments_in_chunk = [
                s for s in transcript
                if s['start'] >= time_start and s['start'] < time_end
            ]

            if not segments_in_chunk:
                continue

            # Concatenate text from segments
            chunk_text = " ".join([clean_transcript_text(s['text']) for s in segments_in_chunk])

            # Skip very short chunks
            if len(chunk_text.strip()) < 50:
                continue

            # Create metadata with timestamp range
            metadata = {
                "video_id": video_id,
                "title": title,
                "url": self._create_youtube_url_with_timestamp(url, time_start),
                "platform": "YouTube",
                "time_start": round(time_start, 2),
                "time_end": round(time_end, 2),
                "timestamp": self._format_timestamp(time_start),
            }

            # Generate embedding
            embedding = self.embedding_model.encode(chunk_text, show_progress_bar=False).tolist()

            entries.append({
                "id": f"{video_id}_chunk{chunk_idx}",
                "source": "YouTube",
                "title": title,
                "url": metadata["url"],
                "text": chunk_text,
                "chunk_index": chunk_idx + 1,
                "embedding_vector": embedding,
                "metadata": metadata,
            })

        return entries

    def process(
        self,
        include_full_transcripts: bool = True,
        include_chunks: bool = True,
        existing_corpus_path: Path = EXISTING_CORPUS,
    ) -> List[Dict[str, Any]]:
        """Process only new transcripts and merge with existing corpus."""
        # Load existing corpus
        existing_entries = self.load_existing_corpus(existing_corpus_path)
        existing_ids = self._get_existing_ids_from_corpus(existing_entries)

        print(f"Existing corpus has {len(existing_ids)} videos")

        # Load only new transcripts
        new_transcripts = self.load_transcripts(
            data_dir=TIMESTAMPS_DATA_DIR,
            existing_ids=existing_ids,
        )

        print(f"\nProcessing {len(new_transcripts)} new transcripts...")

        all_entries = list(existing_entries)  # Start with existing

        for transcript_data in new_transcripts:
            video_id = transcript_data['video_id']
            print(f"  Processing {video_id}...")

            # Add full transcript entry
            if include_full_transcripts:
                entry = self.build_full_transcript_entry(transcript_data)
                all_entries.append(entry)

            # Add chunked entries
            if include_chunks:
                chunks = self.build_chunked_entries(transcript_data)
                all_entries.extend(chunks)
                print(f"    Created {len(chunks)} chunks")

        return all_entries

    def _get_existing_ids_from_corpus(self, corpus: List[Dict[str, Any]]) -> Set[str]:
        """Extract unique video IDs from corpus entries."""
        ids = set()
        for item in corpus:
            if 'id' in item:
                # For chunked entries, extract base video ID
                video_id = item['id'].split('_chunk')[0]
                ids.add(video_id)
        return ids

    def save_corpus(self, entries: List[Dict[str, Any]], output_path: Path) -> None:
        """Save processed corpus to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        print(f"\nSaved corpus to: {output_path}")


def main():
    """Main entry point for processing YouTube transcripts."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process YouTube transcripts with timestamps into RAG corpus format"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=TIMESTAMPS_DATA_DIR,
        help="Directory containing timestamped transcript JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_YOUTUBE_JSON,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--existing",
        type=Path,
        default=EXISTING_CORPUS,
        help="Existing corpus to merge with",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess all videos (ignore existing)",
    )

    args = parser.parse_args()

    processor = TimestampedTranscriptProcessor()

    if args.force:
        # Process all transcripts, ignore existing
        print("Force mode: processing all transcripts...")
        # Load all transcripts
        transcripts = processor.load_transcripts(args.input_dir, existing_ids=set())
        entries = []
        for t in transcripts:
            entries.append(processor.build_full_transcript_entry(t))
            entries.extend(processor.build_chunked_entries(t))
    else:
        # Only process new videos
        entries = processor.process(existing_corpus_path=args.existing)

    processor.save_corpus(entries, args.output)

    # Print summary
    unique_videos = processor._get_existing_ids_from_corpus(entries)
    print(f"\nSummary:")
    print(f"  Total entries: {len(entries)}")
    print(f"  Unique videos: {len(unique_videos)}")


if __name__ == "__main__":
    main()
