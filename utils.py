import subprocess
import assemblyai as aai
from fastapi import UploadFile

async def save_file(file: UploadFile, destination: str):
    """Save uploaded file to disk."""
    with open(destination, "wb") as out_file:
        while chunk := await file.read(1024 * 1024):
            out_file.write(chunk)

def extract_audio(video_path: str, audio_path: str):
    """Extract audio track from video using ffmpeg."""
    command = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "mp3", audio_path]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def transcribe_audio(audio_path: str, transcript_path: str, ai_config) -> str:
    """Transcribe audio file using AssemblyAI."""
    transcript = aai.Transcriber(config=ai_config).transcribe(audio_path)
    if transcript.status == "error":
        raise RuntimeError(f"Transcription failed: {transcript.error}")

    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript.text)

    return transcript.text