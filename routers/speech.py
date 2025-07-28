from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from moviepy.editor import VideoFileClip
import os
import uuid
from pyannote.audio import Pipeline
from fastapi import APIRouter, UploadFile, File

router = APIRouter()

# Use environment variable or fallback (You can load from dotenv in production)
HF_TOKEN = os.getenv("HF_TOKEN", None)

# Load the diarization pipeline once
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)

@router.post("/diarize/")
async def diarize_video(file: UploadFile = File(...)):
    filename = f"{uuid.uuid4()}.mp4"
    wav_path = filename.replace(".mp4", ".wav")

    try:
        # Save uploaded video
        with open(filename, "wb") as f:
            f.write(await file.read())

        # Convert video to WAV
        clip = VideoFileClip(filename)
        audio = clip.audio
        if not audio:
            raise ValueError("No audio stream found in video.")
        audio.write_audiofile(wav_path, codec='pcm_s16le', verbose=False, logger=None)

        # Run diarization
        diarization = pipeline(wav_path)

        # Collect speakers and timestamps
        speakers = set()
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.add(speaker)
            segments.append({
                "speaker": speaker,
                "start": round(turn.start, 2),
                "end": round(turn.end, 2)
            })

        return JSONResponse(content={
            "unique_speakers": len(speakers),
            "segments": segments
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        # Cleanup temporary files
        for path in [filename, wav_path]:
            if os.path.exists(path):
                os.remove(path)
