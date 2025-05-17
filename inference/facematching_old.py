import numpy as np
from PIL import Image
import face_recognition
import io
import cv2

print(f'face_recognition {face_recognition.__version__}')
print(f'Numpy {np.__version__}')



# Function to calculate brightness of an image
def calculate_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

# Function to calculate blurriness of an image
def calculate_blurriness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# Function to load and process an image
def load_and_process_image(image_path):
    image = face_recognition.load_image_file(image_path)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    brightness = calculate_brightness(image_bgr)
    blurriness = calculate_blurriness(image_bgr)
    return image, brightness, blurriness


import io
import numpy as np
from PIL import Image
import face_recognition

def prepare_image(image_bytes: bytes) -> np.ndarray:
    # Ensure we start from raw bytes
    if isinstance(image_bytes, io.BytesIO):
        image_bytes = image_bytes.read()

    # Load and convert image to RGB
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Convert to np.uint8 array
    image_array = np.asarray(pil_image)

    # Confirm shape and type
    print(f"[DEBUG] Image dtype: {image_array.dtype}, shape: {image_array.shape}")
    return image_array

def compare_faces(id_image_bytes: bytes, selfie_image_bytes: bytes) -> dict:
    try:
        id_image = prepare_image(id_image_bytes)
        selfie_image = prepare_image(selfie_image_bytes)

        # Must be uint8 and RGB
        if id_image.dtype != np.uint8 or id_image.shape[2] != 3:
            raise ValueError("ID image is not valid RGB")
        if selfie_image.dtype != np.uint8 or selfie_image.shape[2] != 3:
            raise ValueError("Selfie image is not valid RGB")

        # Face encoding
        id_encodings = face_recognition.face_encodings(id_image)
        selfie_encodings = face_recognition.face_encodings(selfie_image)

        if not id_encodings or not selfie_encodings:
            return {"success": False, "error": "Face not detected in one or both images"}

        match = face_recognition.compare_faces([id_encodings[0]], selfie_encodings[0])[0]
        distance = face_recognition.face_distance([id_encodings[0]], selfie_encodings[0])[0]

        return {
            "success": True,
            "match": match,
            "distance": float(distance)
        }
    except Exception as e:
        print(f"[ERROR] {e}")
        return {"success": False, "error": str(e)}