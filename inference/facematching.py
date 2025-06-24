import cv2
import numpy as np
from deepface import DeepFace
import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_image_safely(image_stream):
    """
    Safely load an image from a stream and ensure it's in the right format
    
    Args:
        image_stream: BytesIO stream containing image data
        
    Returns:
        numpy array: Image as numpy array in BGR format (what OpenCV expects)
    """
    # Reset file pointer
    image_stream.seek(0)
    
    # Read image as binary data
    image_bytes = image_stream.read()
    
    # Convert to numpy array of bytes
    np_array = np.frombuffer(image_bytes, np.uint8)
    
    # Decode the image using OpenCV (this handles various formats)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Failed to decode image. The file may be corrupted or in an unsupported format.")
    
    return img

def calculate_brightness(image):
    """Calculate the brightness of an image"""
    # Convert to grayscale if it's not already
    if len(image.shape) == 3:  # Color image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:  # Already grayscale
        gray = image
    return float(np.mean(gray))

def calculate_blurriness(image):
    """Calculate the blurriness of an image"""
    # Convert to grayscale if it's not already
    if len(image.shape) == 3:  # Color image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:  # Already grayscale
        gray = image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def compare_faces(id_image_stream, selfie_image_stream):
    """
    Compare faces between ID and selfie images using DeepFace
    
    Args:
        id_image_stream: BytesIO stream containing ID image
        selfie_image_stream: BytesIO stream containing selfie image
        
    Returns:
        dict: Result of face comparison and image quality metrics
    """
    try:
        # Load images safely - DeepFace works with BGR images (OpenCV format)
        id_image = load_image_safely(id_image_stream)
        selfie_image = load_image_safely(selfie_image_stream)
        
        # Log image shapes for debugging
        logger.info(f"ID image shape: {id_image.shape}, dtype: {id_image.dtype}")
        logger.info(f"Selfie image shape: {selfie_image.shape}, dtype: {selfie_image.dtype}")
        
        # Calculate image quality metrics
        id_brightness = calculate_brightness(id_image)
        id_blurriness = calculate_blurriness(id_image)
        selfie_brightness = calculate_brightness(selfie_image)
        selfie_blurriness = calculate_blurriness(selfie_image)
        
        # Save input images to temp files (DeepFace sometimes works better with files)
        id_temp_path = "temp_id_image.jpg"
        selfie_temp_path = "temp_selfie_image.jpg"
        
        cv2.imwrite(id_temp_path, id_image)
        cv2.imwrite(selfie_temp_path, selfie_image)
        
        # Verify faces using DeepFace
        try:
            # Use DeepFace.verify for face comparison
            # Parameters:
            # - model_name: 'VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace'
            # - distance_metric: 'cosine', 'euclidean', 'euclidean_l2'
            result = DeepFace.verify(
                img1_path=id_temp_path,
                img2_path=selfie_temp_path,
                model_name="VGG-Face",  # Most reliable model
                distance_metric="cosine",
                detector_backend="retinaface"  # More accurate than default 'opencv'
            )
            
            # Check if verification was successful
            verified = result.get("verified", False)
            distance = result.get("distance", 1.0)  # Default to max distance (1.0) if not found
            
            logger.info(f"DeepFace verification result: {result}")
            
            return {
                "Selfie Brightness": selfie_brightness,
                "Selfie Blurriness": selfie_blurriness,
                "ID Card Brightness": id_brightness,
                "ID Card Blurriness": id_blurriness,
                "Face Distance": float(distance),
                "Match": bool(verified),
                "Error": ""
            }
            
        except Exception as e:
            logger.error(f"DeepFace verification error: {str(e)}")
            
            # Try to detect faces to provide better error messages
            try:
                # Use DeepFace.extract_faces to check if faces can be detected
                id_face_info = DeepFace.extract_faces(
                    img_path=id_temp_path, 
                    detector_backend="retinaface",
                    enforce_detection=False
                )
                
                selfie_face_info = DeepFace.extract_faces(
                    img_path=selfie_temp_path,
                    detector_backend="retinaface",
                    enforce_detection=False
                )
                
                id_faces_detected = len(id_face_info) > 0
                selfie_faces_detected = len(selfie_face_info) > 0
                
                if not id_faces_detected:
                    error_msg = "No face detected in ID image."
                elif not selfie_faces_detected:
                    error_msg = "No face detected in selfie image."
                else:
                    error_msg = f"Face verification failed: {str(e)}"
                    
            except Exception as extract_error:
                error_msg = f"Error detecting faces: {str(extract_error)}"
            
            return {
                "Selfie Brightness": selfie_brightness,
                "Selfie Blurriness": selfie_blurriness,
                "ID Card Brightness": id_brightness,
                "ID Card Blurriness": id_blurriness,
                "Face Distance": None,
                "Match": False,
                "Error": error_msg
            }
        
    except Exception as e:
        logger.error(f"Exception in compare_faces: {str(e)}", exc_info=True)
        return {
            "Match": False,
            "Error": f"Error processing images: {str(e)}"
        }