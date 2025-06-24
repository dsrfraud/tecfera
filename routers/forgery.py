import io
from typing import List

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session
from db.session import get_db

from inference.qr_utils import (
    detect_file_type,
    process_image_file,
    process_pdf_file,
    process_zip_file
)
from schemas.response_schemas import BaseResponse

# Initialize router
router = APIRouter()


@router.post("/uploadfile/", response_model=BaseResponse)
async def upload_files(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """
    Endpoint to handle single or multiple file uploads.
    Supports PDF, images (including TIFF), and extracts and analyzes QR codes.
    """
    results = []

    for file in files:
        try:
            file_bytes = await file.read()
            mime_type = detect_file_type(file_bytes)

            # Process based on file type
            if mime_type == "application/pdf":
                file_result = process_pdf_file(file_bytes, file.filename)
                results.append(file_result)

            elif mime_type.startswith("image/"):
                file_result = process_image_file(file_bytes, file.filename)
                results.append(file_result)

            else:
                return BaseResponse(
                    status="error",
                    message=f"Unsupported file type: {mime_type}",
                    data=None
                )
        except Exception as e:
            return BaseResponse(
                status="error",
                message=f"Error processing file {file.filename}: {str(e)}",
                data=None
            )

    return BaseResponse(
        status="success",
        message="Files processed successfully",
        data={"uploaded_files": results}
    )


@router.post("/uploadfiles/", response_model=BaseResponse)
async def upload_multiple_files(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """
    Endpoint to handle archive file uploads.
    Supports ZIP archives containing PDFs and images, as well as direct PDF and image uploads.
    """
    results = []

    for file in files:
        try:
            file_bytes = await file.read()
            mime_type = detect_file_type(file_bytes)

            # Process based on file type
            if mime_type == "application/zip":
                zip_results = process_zip_file(file_bytes)
                results.extend(zip_results)

            elif mime_type == "application/pdf":
                file_result = process_pdf_file(file_bytes, file.filename)
                results.append(file_result)

            elif mime_type.startswith("image/"):
                file_result = process_image_file(file_bytes, file.filename)
                results.append(file_result)

            else:
                return BaseResponse(
                    status="error",
                    message=f"Unsupported file type: {mime_type}",
                    data=None
                )
        except Exception as e:
            return BaseResponse(
                status="error",
                message=f"Error processing file {file.filename}: {str(e)}",
                data=None
            )

    return BaseResponse(
        status="success",
        message="Files processed successfully",
        data={"uploaded_files": results}
    )
