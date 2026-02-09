import json
import sys
from pathlib import Path
from yt_dlp import YoutubeDL
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

# Fix Windows console encoding issue
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from profectus_ai.config import YOUTUBE_RAW_DIR

# We will save files as: {video_id}.json

def ensure_data_folder():
    YOUTUBE_RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using folder: {YOUTUBE_RAW_DIR}")

def get_channel_videos(channel_url):
    """
    Fetches all video IDs and Titles from a channel using yt-dlp.
    Uses extract_flat=True to avoid downloading video data, making it very fast.
    """
    print(f"Fetching video list from: {channel_url}")
    ydl_opts = {
        'quiet': True,
        'extract_flat': 'in_playlist', 
        'ignoreerrors': True, # Skip geo-restricted or private videos in the list
    }

    videos = []
    with YoutubeDL(ydl_opts) as ydl:
        try:
            result = ydl.extract_info(channel_url, download=False)
            
            # Channels return an 'entries' list. Single videos return the dict directly.
            if 'entries' in result:
                for entry in result['entries']:
                    if entry:
                        videos.append({
                            'id': entry['id'],
                            'title': entry.get('title', 'Unknown Title'),
                            'url': f"https://www.youtube.com/watch?v={entry['id']}"
                        })
            else:
                # It was a single video URL passed instead of a channel
                videos.append({
                    'id': result['id'],
                    'title': result.get('title', 'Unknown Title'),
                    'url': result['webpage_url']
                })
        except Exception as e:
            print(f"Error fetching channel info: {e}")
            return []

    return videos

def get_timestamped_transcript(video_id):
    """
    Fetches the transcript using youtube-transcript-api.
    Returns a list of dicts: {'text': '...', 'start': 0.0, 'duration': 1.5}
    """
    try:
        # Fetch transcript. usually tries to find English first, then auto-generated
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)
        # Convert snippets to list of dicts
        return [{'text': s.text, 'start': s.start, 'duration': s.duration} for s in transcript.snippets]
    except (NoTranscriptFound, TranscriptsDisabled):
        print(f"  No transcript found for {video_id}")
        return None
    except Exception as e:
        print(f"  Error fetching transcript for {video_id}: {e}")
        return None

def process_videos(target_url):
    ensure_data_folder()
    
    # 1. Get list of all videos
    videos = get_channel_videos(target_url)
    print(f"Found {len(videos)} videos on channel/page.\n")

    count = 0
    for video in videos:
        video_id = video['id']
        file_path = YOUTUBE_RAW_DIR / f"{video_id}.json"

        # 2. Check if file already exists
        if file_path.exists():
            print(f"[SKIP] Video '{video['title']}' already exists.")
            continue

        print(f"[NEW] Processing: {video['title']} ({video_id})")

        # 3. Get Transcript
        transcript_data = get_timestamped_transcript(video_id)

        if transcript_data:
            # 4. Structure the data for RAG usage
            output_data = {
                "video_id": video_id,
                "title": video['title'],
                "url": video['url'],
                "transcript": transcript_data
            }

            # 5. Save to JSON
            with file_path.open('w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"  -> Saved to {file_path}")
            count += 1
        else:
            print(f"  -> Skipped (No transcript data).")

    print(f"\nFinished. {count} new transcripts downloaded.")

if __name__ == "__main__":
    # You can change this URL to any specific video or channel
    TARGET_URL = "https://www.youtube.com/@profectusai/videos"
    
    # Allow command line argument usage
    if len(sys.argv) > 1:
        TARGET_URL = sys.argv[1]

    process_videos(TARGET_URL)
