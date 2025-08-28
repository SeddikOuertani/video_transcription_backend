import os
import uuid
import asyncio
import subprocess
from typing import Dict, AsyncGenerator

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

import assemblyai as aai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Create directories if they don't exist
UPLOAD_DIR = "uploads"
AUDIO_DIR = "audios"
TRANSCRIPT_DIR = "transcripts"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
if not ASSEMBLYAI_API_KEY:
    raise RuntimeError("ASSEMBLYAI_API_KEY not set in environment variables")
aai.settings.api_key = ASSEMBLYAI_API_KEY
config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.best)

@app.post("/transcribe-video/")
async def transcribe_video(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("video/"):
        return JSONResponse(
            content={"error": "Only video files are allowed"},
            status_code=400
        )

    # Prepare file paths
    video_path = os.path.join(UPLOAD_DIR, file.filename)
    audio_path = os.path.join(AUDIO_DIR, f"{os.path.splitext(file.filename)[0]}.mp3")
    transcript_path = os.path.join(TRANSCRIPT_DIR, f"{os.path.splitext(file.filename)[0]}.txt")

    # Save uploaded video
    await save_file(file, video_path)

    # Extract audio (blocking operation offloaded to thread)
    await asyncio.to_thread(extract_audio, video_path, audio_path)

    # Transcribe audio (blocking operation offloaded to thread)
    await asyncio.to_thread(transcribe_audio, audio_path, transcript_path)

    return {"success": True, "message": "video transcribed successfully"}


@app.post("/transcribe-video-stream/")
async def transcribe_video_stream(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        return JSONResponse(content={"error": "Only video files are allowed"}, status_code=400)

    video_path = os.path.join(UPLOAD_DIR, file.filename)
    audio_path = os.path.join(AUDIO_DIR, f"{os.path.splitext(file.filename)[0]}.mp3")
    transcript_path = os.path.join(TRANSCRIPT_DIR, f"{os.path.splitext(file.filename)[0]}.txt")

    await save_file(file, video_path)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            yield "data: File saved successfully\n\n"

            # --- Step 2: Extract audio ---
            yield "data: Extracting audio from video...\n\n"
            await asyncio.to_thread(extract_audio, video_path, audio_path)
            yield "data: Audio extraction done\n\n"

            # --- Step 3: Transcription ---
            yield "data: Transcribing audio...\n\n"
            transcript = await asyncio.to_thread(transcribe_audio, audio_path, transcript_path)

            yield f"data: Transcription completed! Transcript saved at {transcript_path}\n\n"
            yield f"data: {transcript}\n\n"

        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"

        yield "data: DONE\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def save_file(file: UploadFile, destination: str):
    """Save uploaded file asynchronously."""
    with open(destination, "wb") as out_file:
        while chunk := await file.read(1024 * 1024):
            out_file.write(chunk)

def extract_audio(video_path: str, audio_path: str):
    """Extract audio from video using ffmpeg."""
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",               # no video
        "-acodec", "mp3",    # audio codec
        audio_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def transcribe_audio(audio_path: str, transcript_path: str):
    """Transcribe audio and save transcript."""
    transcript = aai.Transcriber(config=config).transcribe(audio_path)
    if transcript.status == "error":
        raise RuntimeError(f"Transcription failed: {transcript.error}")

    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript.text)