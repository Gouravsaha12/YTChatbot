from youtube_transcript_api import YouTubeTranscriptApi
import os

def get_video_id(url: str) -> str:
    """Extracts the video ID from a YouTube URL."""
    try:
        return url.split("youtu.be/")[1].split("?")[0]
    except IndexError:
        raise ValueError("Invalid YouTube URL. Please provide a youtu.be link.")

def fetch_and_save_transcript(video_id: str) -> str:
    """Fetches the transcript and saves it to a text file."""
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id)
    
    os.makedirs("data", exist_ok=True)
    file_path = f"data/{video_id}.txt"
    with open(file_path, "w", encoding="utf-8") as file:
        for snippet in fetched_transcript:      
            file.write(f"{snippet.text} ")
    
    return file_path
