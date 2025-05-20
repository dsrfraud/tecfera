from fastapi import APIRouter, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
import io
import logging
import os
import uuid
from pathlib import Path
import shutil


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

from inference import facematching, facematching_old

# @router.post("/compare-faces")
# async def compare_faces_endpoint(
#     id_image: UploadFile = File(...),
#     selfie_image: UploadFile = File(...)
# ):
#     if id_image.content_type not in ["image/jpeg", "image/png"]:
#         raise HTTPException(status_code=400, detail="Invalid ID image format")
#     if selfie_image.content_type not in ["image/jpeg", "image/png"]:
#         raise HTTPException(status_code=400, detail="Invalid selfie image format")

#     id_image_bytes = await id_image.read()
#     selfie_image_bytes = await selfie_image.read()

#     result = facematching_old.compare_faces(id_image_bytes, selfie_image_bytes)
#     return result


@router.post("/compare-faces")
async def compare_faces_endpoint(id_image: UploadFile = File(...), selfie_image: UploadFile = File(...)):
    try:
        id_image_bytes = await id_image.read()
        selfie_image_bytes = await selfie_image.read()

        id_image_stream = io.BytesIO(id_image_bytes)
        selfie_image_stream = io.BytesIO(selfie_image_bytes)

        result = facematching_old.compare_faces(id_image_stream, selfie_image_stream)
        print(result)
        return result
        # return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def cleanup_temp_files(file_paths):
    """Remove temporary files"""
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Removed temporary file: {path}")
        except Exception as e:
            logger.error(f"Failed to remove {path}: {str(e)}")