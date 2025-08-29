import os
import uuid
import asyncio
from typing import Dict, AsyncGenerator

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

import assemblyai as aai
from utils import save_file, extract_audio, transcribe_audio   # 👈 import helpers
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
if not ASSEMBLYAI_API_KEY:
    raise RuntimeError("ASSEMBLYAI_API_KEY not set in environment variables")
aai.settings.api_key = ASSEMBLYAI_API_KEY
config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.best)



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

    # Create a queue for streaming updates
    queue: asyncio.Queue[str] = asyncio.Queue()

    # Initialize job state (standardized fields)
    jobs[job_id] = {
        "id": job_id,
        "status": "pending",
        "video_path": video_path,
        "audio_path": audio_path,
        "transcript_path": transcript_path,
        "error_message": None,
        "transcript_text": None,
        "steps": [],
        "queue": queue,   # 👈 store queue
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
    return {"jobs": [
        {k: v for k, v in job.items() if k != "queue"}  # don’t expose queue
        for job in jobs.values()
    ]}

# ----------------------------
# API: Stream job updates  
# ----------------------------
@app.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    if job_id not in jobs:
        return JSONResponse(content={"error": "Job not found"}, status_code=404)

    queue: asyncio.Queue[str] = jobs[job_id]["queue"]

    async def event_generator() -> AsyncGenerator[str, None]:
        while True:
            message = await queue.get()
            yield f"data: {message}\n\n"
            if message == "DONE":
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")



# ----------------------------
# API: process job 
# ----------------------------
async def process_job(job_id: str):
    job = jobs[job_id]
    q: asyncio.Queue[str] = job["queue"]

    try:
        job["status"] = "extracting_audio"
        job["steps"].append("Extracting audio")
        await q.put("Extracting audio...")
        await asyncio.to_thread(extract_audio, job["video_path"], job["audio_path"])
        await q.put("Audio extracted")

        job["status"] = "transcribing"
        job["steps"].append("Transcribing")
        await q.put("Transcribing audio...")
        transcript = await asyncio.to_thread(
            transcribe_audio, job["audio_path"], job["transcript_path"], config
        )

        job["status"] = "completed"
        job["transcript_text"] = transcript
        job["steps"].append("Completed")

        await q.put("Transcription completed")

    except Exception as e:
        job["status"] = "failed"
        job["error_message"] = str(e)
        job["steps"].append("Failed")
        await q.put(f"ERROR: {str(e)}")

    finally:
        await q.put("DONE")