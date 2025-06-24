from PIL import UnidentifiedImageError
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
import io
import logging
import os
import uuid
from pathlib import Path
import shutil
import time
import numpy as np
import cv2
from PIL import Image
import base64
import face_recognition

from tempfile import NamedTemporaryFile
from PIL import Image
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Configuration constants
# UPLOAD_FOLDER = 'uploads/'
# KNOWN_FACES_FOLDER = 'known_faces'
# BASE_PATH = '/home/ubuntu/webapp/static'
# UPLOADS_PATH = os.path.join(BASE_PATH, 'uploads')
# KNOWN_FACES_PATH = os.path.join(BASE_PATH, 'known_faces')


from inference.bank_statement_analyzer import pdf_to_images, analyze_bank_statement_from_images
@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # try:
    suffix = os.path.splitext(file.filename)[1].lower()
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        shutil.copyfileobj(file.file, temp)
        temp_path = temp.name

    if suffix == ".pdf":
        images = pdf_to_images(temp_path)
    elif suffix in [".png", ".jpg", ".jpeg"]:
        images = [Image.open(temp_path)]
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    result = analyze_bank_statement_from_images(images)
    return {"status": "success", "data": result}

    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))
    # finally:
    #     os.remove(temp_path)