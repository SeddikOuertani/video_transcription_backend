import os
import uuid
import asyncio
from typing import Dict

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from dotenv import load_dotenv

from utils import save_file, extract_audio, transcribe_audio   # 👈 import helpers

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

# In-memory job store (replace with DB/Redis in production)
jobs: Dict[str, Dict] = {}


# ----------------------------
# API: Create job
# ----------------------------
@app.post("/jobs/")
async def create_job(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        return JSONResponse(content={"error": "Only video files are allowed"}, status_code=400)

    job_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    audio_path = os.path.join(AUDIO_DIR, f"{job_id}.mp3")
    transcript_path = os.path.join(TRANSCRIPT_DIR, f"{job_id}.txt")

    # Initialize job state (standardized fields)
    jobs[job_id] = {
        "id": job_id,
        "status": "pending",
        "video_path": video_path,
        "audio_path": audio_path,
        "transcript_path": transcript_path,
        "error_message": None,
        "transcript_text": None,
        "steps": []
    }

    # Save uploaded file
    await save_file(file, video_path)
    jobs[job_id]["steps"].append("Video uploaded")

    # Run processing in background
    asyncio.create_task(process_job(job_id))

    return {"job_id": job_id, "status": "pending"}


# ----------------------------
# API: Get all jobs
# ----------------------------
@app.get("/jobs/")
async def list_jobs():
    """Return all jobs and their statuses."""
    return {"jobs": list(jobs.values())}

async def process_job(job_id: str):
    """Background task: extract audio and transcribe."""
    job = jobs[job_id]
    try:
        # Step 1: Extract audio
        job["status"] = "extracting_audio"
        job["steps"].append("Extracting audio")
        await asyncio.to_thread(extract_audio, job["video_path"], job["audio_path"])

        # Step 2: Transcribe
        job["status"] = "transcribing"
        job["steps"].append("Transcribing audio")
        transcript = await asyncio.to_thread(transcribe_audio, job["audio_path"], job["transcript_path"])

        # Completed
        job["status"] = "completed"
        job["transcript_text"] = transcript
        job["steps"].append("Completed")

    except Exception as e:
        job["status"] = "failed"
        job["error_message"] = str(e)
        job["steps"].append("Failed")
