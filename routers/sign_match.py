import base64
import numpy as np
import time
from io import BytesIO
from PIL import Image
import cv2
from fastapi import FastAPI, File, UploadFile, APIRouter
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes
from inference.signature_detection import SignatureDetection
from inference.signature_matching import get_similarity_score

# Initialize FastAPI app
router = APIRouter()




# Initialize signature detection model
sign_detection = SignatureDetection("model_weights/pan_latest.pt", "")

def encode_image_to_base64(img):
    """Convert an image to base64 encoding"""
    try:
        # Convert image to numpy array if it's not already
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        # Ensure image is in BGR format for compatibility with OpenCV
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Encode image to JPEG format
        _, buffer = cv2.imencode('.jpg', img)

        # Encode to base64
        base64_encoded = base64.b64encode(buffer).decode('utf-8')

        return base64_encoded

    except Exception as e:
        print(f"Error encoding image to base64: {e}")
        return None

@router.post("/signature-detection")
async def signatureDetection(file: UploadFile = File(...)):
    """Detect signatures in an image or PDF file"""
    if file.content_type.startswith("image"):
        # Handle image file
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        result = sign_detection.process_single_image(np.array(image))
        return result
    elif file.content_type == "application/pdf":
        # Handle PDF file
        contents = await file.read()
        images = convert_from_bytes(contents)
        results = []

        for image in images:
            result = sign_detection.process_single_image(np.array(image))
            results.extend(result)

        return {"signatures": results}

@router.post("/signature-card-matching")
async def signature_card_matching(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """Match a signature from one file with a signature card in another file"""
    start_time = time.time()
    image1_bytes = await file1.read()
    image2_bytes = await file2.read()

    # Process second image
    second_image = Image.open(BytesIO(image2_bytes))
    second_result = sign_detection.process_single_image(np.array(second_image))

    # Check if second result is None (no signatures detected)
    if second_result is None or len(second_result) == 0:
        second_encoded_image = image2_bytes
    else:
        second_encoded_image = base64.b64decode(second_result[0])

    best_similarity_score, best_first_image, best_second_image = get_similarity_score(image1_bytes, second_encoded_image)

    total_time = time.time() - start_time
    
    response = {
        "similarity_score": str(best_similarity_score),
        "best_first_image": best_first_image,
        "best_second_image": best_second_image,
        "total_time": str(total_time)
    }

    return JSONResponse(content=response)

@router.post("/compare-signatures-api")
async def compare_signatures(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """Compare signatures from two files"""
    image1_bytes = await file1.read()
    image2_bytes = await file2.read()

    start_time = time.time()

    # Process first image
    first_image = Image.open(BytesIO(image1_bytes))
    first_result = sign_detection.process_single_image(np.array(first_image))
    
    # Check if first result is None (no signatures detected)
    if first_result is None or len(first_result) == 0:
        first_encoded_image = image1_bytes
    else:
        first_encoded_image = base64.b64decode(first_result[0])
    
    # Process second image
    second_image = Image.open(BytesIO(image2_bytes))
    second_result = sign_detection.process_single_image(np.array(second_image))

    # Check if second result is None (no signatures detected)
    if second_result is None or len(second_result) == 0:
        second_encoded_image = image2_bytes
    else:
        second_encoded_image = base64.b64decode(second_result[0])

    # Get similarity score between the two images
    best_similarity_score, best_first_image, best_second_image = get_similarity_score(first_encoded_image, second_encoded_image)

    total_time = time.time() - start_time

    response = {
        "similarity_score": str(best_similarity_score),
        "best_first_image": best_first_image,
        "best_second_image": best_second_image,
        "total_time": str(total_time)
    }

    return JSONResponse(content=response)
