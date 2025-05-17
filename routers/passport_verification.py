import io
import json
import time
import dateutil.parser
import numpy as np
import cv2

from typing import Dict
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from typing_extensions import Annotated

from schemas.response_schemas import BaseResponse
from inference.passport_utils import (
    extract_passport_data, 
    make_api_request,
    get_parsed_output,
)

router = APIRouter()

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png"}


@router.post("/upload-passport-front/", response_model=BaseResponse)
async def upload_passport_front(passport_front: UploadFile = File(...)) -> BaseResponse:
    if passport_front.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are allowed")

    file_bytes = np.frombuffer(await passport_front.read(), np.uint8)
    front_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if front_image is None:
        raise HTTPException(status_code=400, detail="Could not process image file")

    try:
        passport_data = extract_passport_data(
            image=front_image, 
            is_front=True,
            include_validation=True
        )

        return BaseResponse(
            status="success",
            message="Passport front processed successfully",
            data=passport_data
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing passport front: {str(e)}")


@router.post("/upload-passport-back/", response_model=BaseResponse)
async def upload_passport_back(passport_back: UploadFile = File(...)) -> BaseResponse:
    if passport_back.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are allowed")

    file_bytes = np.frombuffer(await passport_back.read(), np.uint8)
    back_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if back_image is None:
        raise HTTPException(status_code=400, detail="Could not process image file")

    try:
        passport_data = extract_passport_data(
            image=back_image,
            is_front=False,
            include_validation=False
        )

        return BaseResponse(
            status="success",
            message="Passport back processed successfully",
            data=passport_data
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing passport back: {str(e)}")


@router.post("/verify-filenumber", response_model=BaseResponse)
async def upload_passport(
    file_number: Annotated[str, Form(...)],
    date: Annotated[str, Form(...)]
) -> BaseResponse:
    start_time = time.time()

    try:
        parsed_date = dateutil.parser.parse(date).strftime("%d/%m/%Y")

        api_url = "https://www.passportindia.gov.in/AppOnlineProject/statusTracker/trackStatusForFileNoNew/"
        api_params = {
            "apptUrl": "",
            "apptRedirectFlag": "false",
            "optStatus": "Application_Status",
            "fileNo": file_number.upper(),
            "applDob": parsed_date,
            "rtiRefNo": "",
            "diplArnNo": "",
            "appealNo": "",
            "appealDob": "",
            "action:trackStatusForFileNoNew": "Track Status"
        }

        response = make_api_request(api_url, params=api_params, delay=2)

        if not response:
            raise HTTPException(status_code=502, detail="Failed to get response from passport service")

        response_data = get_parsed_output(response)
        return BaseResponse(
            status="success",
            message="Passport status validated successfully",
            data=response_data
        )

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating passport: {str(e)}")
