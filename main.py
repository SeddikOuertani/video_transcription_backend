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

# In-memory job store (replace with DB/Redis in production)
jobs: Dict[str, Dict] = {}


@app.post("/jobs/")
async def create_job(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        return JSONResponse(content={"error": "Only video files are allowed"}, status_code=400)

    job_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    audio_path = os.path.join(AUDIO_DIR, f"{job_id}.mp3")
    transcript_path = os.path.join(TRANSCRIPT_DIR, f"{job_id}.txt")

    # Initialize job state
    jobs[job_id] = {
        "status": "created",
        "video": video_path,
        "audio": audio_path,
        "transcript": transcript_path,
        "error": None,
        "result": None,
        "steps": []
    }

    # Save uploaded file
    await save_file(file, video_path)
    jobs[job_id]["steps"].append("File saved")
    jobs[job_id]["status"] = "ready"

    return {"job_id": job_id, "message": "Job created. Use /jobs/{job_id}/stream to track progress."}


@app.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    if job_id not in jobs:
        return JSONResponse(content={"error": "Job not found"}, status_code=404)

    job = jobs[job_id]

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Step 1: Extract audio
            job["status"] = "extracting"
            job["steps"].append("Extracting audio")
            yield "data: Extracting audio...\n\n"
            await asyncio.to_thread(extract_audio, job["video"], job["audio"])
            yield "data: Audio extracted\n\n"

            # Step 2: Transcribe
            job["status"] = "transcribing"
            job["steps"].append("Transcribing")
            yield "data: Transcribing audio...\n\n"
            transcript = await asyncio.to_thread(transcribe_audio, job["audio"], job["transcript"])
            job["status"] = "completed"
            job["result"] = transcript
            job["steps"].append("Completed")

            yield f"data: Transcription completed\n\n"
            yield f"data: {transcript}\n\n"

        except Exception as e:
            job["status"] = "error"
            job["error"] = str(e)
            yield f"data: ERROR: {str(e)}\n\n"

        yield "data: DONE\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        return JSONResponse(content={"error": "Job not found"}, status_code=404)
    return jobs[job_id]


# --- Helpers ---

async def save_file(file: UploadFile, destination: str):
    """Save uploaded file to disk."""
    with open(destination, "wb") as out_file:
        while chunk := await file.read(1024 * 1024):
            out_file.write(chunk)


def extract_audio(video_path: str, audio_path: str):
    """Extract audio track from video."""
    command = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "mp3", audio_path]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def transcribe_audio(audio_path: str, transcript_path: str) -> str:
    """Transcribe audio file."""
    transcript = aai.Transcriber(config=config).transcribe(audio_path)
    if transcript.status == "error":
        raise RuntimeError(f"Transcription failed: {transcript.error}")

    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript.text)

    return transcript.text