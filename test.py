import face_recognition
import numpy as np
from PIL import Image
import io

def load_image(image_path: str) -> np.ndarray:
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_array = np.asarray(pil_image)
    print(f"[DEBUG] Loaded {image_path} - dtype: {image_array.dtype}, shape: {image_array.shape}")
    return image_array

def compare_faces(id_image_path: str, selfie_image_path: str) -> dict:
    try:
        id_image = load_image(id_image_path)
        selfie_image = load_image(selfie_image_path)

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

# Replace with actual paths to your images
if __name__ == "__main__":
    id_path = r"C:\Users\ACER\Downloads\raja_docs\aadhar_front.jpeg"
    selfie_path = r"C:\Users\ACER\Downloads\raja_docs\myphoto.jpeg"
    result = compare_faces(id_path, selfie_path)
    print(result)