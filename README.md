# 🎬 Video Transcriber with Streaming Progress

This project provides a **FastAPI-based service** to upload videos, extract audio, and transcribe them using [AssemblyAI](https://www.assemblyai.com/).  
The cool part 👉 You can **stream the job’s progress** in real time while it’s being processed!

---

## ✨ Features
- 📤 Upload a video via API (`POST /jobs/`)  
- ⚙️ Background job starts **immediately**  
- 📡 Subscribe to **live transcription progress** (`GET /jobs/{job_id}/stream`)  
- 📑 Check all jobs and statuses (`GET /jobs/`)  
- ✅ Written with **FastAPI** and **AssemblyAI SDK**

---

## 📦 Dependencies
Install Python dependencies with:

```bash
pip install -r requirements.txt
```

Additionally, you need FFmpeg installed on your system (not a Python dependency):

### Ubuntu/Debian:
```
sudo apt update && sudo apt install ffmpeg
```

### macOS (Homebrew):
```
brew install ffmpeg
```

### Windows:
Download FFmpeg from https://ffmpeg.org/

## 🚀 Running the Project

Clone this repo:
```
git clone https://github.com/yourusername/video-transcriber.git
cd video-transcriber
```

Create a .env file and add your AssemblyAI API key:
```
ASSEMBLYAI_API_KEY=your_api_key_here
```

Start the FastAPI server:
```
uvicorn main:app --reload
```

Open your browser at 👉 http://localhost:8000/docs

## 🔌 API Endpoints
- POST /jobs/ → Upload a video & start a transcription job
- GET /jobs/ → List all jobs and their statuses
- GET /jobs/{job_id}/stream → Stream progress of a specific job (Server-Sent Events)

## 📜 License
MIT License – feel free to use and adapt.