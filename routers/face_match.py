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

@router.post("/compare-faces")
async def compare_faces_endpoint(
    id_image: UploadFile = File(...),
    selfie_image: UploadFile = File(...)
):
    if id_image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid ID image format")
    if selfie_image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid selfie image format")

    id_image_bytes = await id_image.read()
    selfie_image_bytes = await selfie_image.read()

    result = facematching_old.compare_faces(id_image_bytes, selfie_image_bytes)
    return result


# @router.post("/compare-faces")
# async def compare_faces_endpoint(
#     id_image: UploadFile = File(...), 
#     selfie_image: UploadFile = File(...),
#     background_tasks: BackgroundTasks = None
# ):
#     """
#     Endpoint to compare faces between ID image and selfie using DeepFace
#     """
#     # Create temporary unique filenames to avoid conflicts in concurrent requests
#     request_id = str(uuid.uuid4())
#     temp_id_path = f"temp_id_{request_id}.jpg"
#     temp_selfie_path = f"temp_selfie_{request_id}.jpg"
    
#     try:
#         logger.info(f"Processing images: {id_image.filename} and {selfie_image.filename}")
        
#         # Read image files completely
#         id_image_bytes = await id_image.read()
#         selfie_image_bytes = await selfie_image.read()
        
#         # Create BytesIO objects
#         id_image_stream = io.BytesIO(id_image_bytes)
#         selfie_image_stream = io.BytesIO(selfie_image_bytes)
        
#         # Log image sizes
#         logger.info(f"ID image size: {len(id_image_bytes)} bytes")
#         logger.info(f"Selfie image size: {len(selfie_image_bytes)} bytes")
        
#         # Process images and compare faces
#         result = facematching_old.compare_faces(id_image_stream, selfie_image_stream)
        
#         # Log result (excluding any binary data)
#         logger.info(f"Comparison result: {result}")
        
#         # Check if there was an error
#         if result.get("Error", ""):
#             logger.error(f"Comparison error: {result['Error']}")
#             return JSONResponse(
#                 status_code=400,
#                 content={"detail": result["Error"]}
#             )
            
#         # Add cleanup of temporary files
#         if background_tasks:
#             background_tasks.add_task(cleanup_temp_files, [temp_id_path, temp_selfie_path])
            
#         return JSONResponse(content=result)
        
#     except Exception as e:
#         logger.error(f"Exception in compare_faces_endpoint: {str(e)}", exc_info=True)
#         return JSONResponse(
#             status_code=400,
#             content={"detail": f"Error processing request: {str(e)}"}
#         )
#     finally:
#         # Ensure cleanup happens even if there's no background_tasks
#         try:
#             cleanup_temp_files([temp_id_path, temp_selfie_path])
#         except Exception as e:
#             logger.error(f"Error during cleanup: {str(e)}")

# def cleanup_temp_files(file_paths):
    """Remove temporary files"""
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Removed temporary file: {path}")
        except Exception as e:
            logger.error(f"Failed to remove {path}: {str(e)}")