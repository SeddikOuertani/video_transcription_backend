import os
import subprocess
import assemblyai as aai
from fastapi import UploadFile
import assemblyai as aai


ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
if not ASSEMBLYAI_API_KEY:
    raise RuntimeError("ASSEMBLYAI_API_KEY not set in environment variables")
aai.settings.api_key = ASSEMBLYAI_API_KEY
config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.best)


async def save_file(file: UploadFile, destination: str):
    """Save uploaded file to disk."""
    with open(destination, "wb") as out_file:
        while chunk := await file.read(1024 * 1024):
            out_file.write(chunk)

def extract_audio(video_path: str, audio_path: str):
    """Extract audio track from video using ffmpeg."""
    command = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "mp3", audio_path]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def transcribe_audio(audio_path: str, transcript_path: str) -> str:
    """Transcribe audio file using AssemblyAI."""
    transcript = aai.Transcriber(config=config).transcribe(audio_path)
    if transcript.status == "error":
        raise RuntimeError(f"Transcription failed: {transcript.error}")

    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript.text)

    return transcript.text