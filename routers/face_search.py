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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Configuration constants
UPLOAD_FOLDER = 'uploads/'
KNOWN_FACES_FOLDER = 'known_faces'
BASE_PATH = '/home/ubuntu/webapp/static'
UPLOADS_PATH = os.path.join(BASE_PATH, 'uploads')
KNOWN_FACES_PATH = os.path.join(BASE_PATH, 'known_faces')

# Ensure directories exist
os.makedirs(UPLOADS_PATH, exist_ok=True)
os.makedirs(KNOWN_FACES_PATH, exist_ok=True)

def save_to_csv(user_data):
    """Save user data to CSV file"""
    csv_file_path = 'user_data.csv'
    with open(csv_file_path, 'a') as file:
        file.write(','.join(str(value) for value in user_data.values()) + '\n')

def resize_image(img, size):
    """Resize image to specified size"""
    img_resized = img.resize(size)
    return img_resized

def recognize_faces_with_images(unknown_image_path):
    """Recognize faces and return matched face images with improved detection"""
    known_faces = []
    known_names = []
    known_face_paths = []

    # Load known face encodings
    for file_name in os.listdir(KNOWN_FACES_PATH):
        if file_name.endswith('.npy'):
            face_encoding_path = os.path.join(KNOWN_FACES_PATH, file_name)
            face_encoding = np.load(face_encoding_path)
            known_faces.append(face_encoding)
            known_names.append(file_name.split('.')[0])
            known_face_paths.append(face_encoding_path.replace('.npy', '.jpg'))

    # Check if no known faces are found
    if not known_faces:
        logger.warning("No known faces found in the database")
        return [], []

    # Load and process unknown image with multiple detection methods
    unknown_image = face_recognition.load_image_file(unknown_image_path)
    unknown_face_locations = []
    unknown_face_encodings = []
    
    # Method 1: Default HOG model
    unknown_face_locations = face_recognition.face_locations(unknown_image, model='hog')
    if unknown_face_locations:
        unknown_face_encodings = face_recognition.face_encodings(unknown_image, unknown_face_locations)
    
    # Method 2: Try CNN model if HOG fails
    if not unknown_face_encodings:
        logger.info("Trying CNN model for face detection...")
        unknown_face_locations = face_recognition.face_locations(unknown_image, model='cnn')
        if unknown_face_locations:
            unknown_face_encodings = face_recognition.face_encodings(unknown_image, unknown_face_locations)
    
    # Method 3: Try with upsampling
    if not unknown_face_encodings:
        logger.info("Trying with upsampling...")
        unknown_face_locations = face_recognition.face_locations(unknown_image, number_of_times_to_upsample=2)
        if unknown_face_locations:
            unknown_face_encodings = face_recognition.face_encodings(unknown_image, unknown_face_locations)
    
    # Method 4: Preprocess image and try again
    if not unknown_face_encodings:
        logger.info("Trying with image preprocessing...")
        # Convert to OpenCV format
        cv_image = cv2.imread(unknown_image_path)
        
        # Enhance contrast
        lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced_image = cv2.merge((cl,a,b))
        enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2BGR)
        
        # Save enhanced image temporarily
        enhanced_path = unknown_image_path.replace('.jpg', '_enhanced_temp.jpg')
        cv2.imwrite(enhanced_path, enhanced_image)
        
        # Try detection on enhanced image
        enhanced_unknown_image = face_recognition.load_image_file(enhanced_path)
        unknown_face_locations = face_recognition.face_locations(enhanced_unknown_image)
        if unknown_face_locations:
            unknown_face_encodings = face_recognition.face_encodings(enhanced_unknown_image, unknown_face_locations)
        
        # Clean up temporary file
        if os.path.exists(enhanced_path):
            os.remove(enhanced_path)

    # Check if no faces found in unknown image
    if not unknown_face_encodings:
        logger.warning("No faces found in the uploaded image after trying multiple methods")
        return [], []

    results = []
    matched_face_urls = []

    # Compare faces with adjustable tolerance
    tolerance = 0.6  # Default tolerance, lower = stricter matching
    
    for unknown_face_encoding, location in zip(unknown_face_encodings, unknown_face_locations):
        matches = face_recognition.compare_faces(known_faces, unknown_face_encoding, tolerance=tolerance)
        name = "Unknown"
        best_match_index = -1
        
        # Calculate face distances to find best match
        face_distances = face_recognition.face_distance(known_faces, unknown_face_encoding)
        
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]
            
            # Use the best match if it's within tolerance
            if best_distance <= tolerance:
                name = known_names[best_match_index]
                matched_face_path = os.path.join(UPLOADS_PATH, name, '1.jpg')
                
                # Check if matched face file exists
                if os.path.exists(matched_face_path):
                    # Convert matched face to base64
                    with open(matched_face_path, 'rb') as img_file:
                        encoded_img = base64.b64encode(img_file.read()).decode('utf-8')
                        matched_face_urls.append(f'data:image/jpg;base64,{encoded_img}')
                else:
                    logger.warning(f"Matched face image not found: {matched_face_path}")

        # Calculate matching score (higher is better)
        matching_score = max(0, (1 - min(face_distances)) * 100) if face_distances.size > 0 else 0

        results.append({
            "name": name,
            "matching_score": round(float(matching_score), 1),
            "confidence": "High" if matching_score > 80 else "Medium" if matching_score > 60 else "Low",
            "location": location
        })

    return results, matched_face_urls

@router.post("/train")
async def train_user(name: str, phone_no: str, file: UploadFile = File(...)):
    """Train the system with a new user's face"""
    start_time = time.time()
    try:
        # Validate input
        if not name or not phone_no:
            raise HTTPException(status_code=400, detail="Name and phone number are required")
        
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Create user directory
        user_directory = os.path.join(UPLOADS_PATH, name)
        os.makedirs(user_directory, exist_ok=True)
        filename = '1.jpg'
        image_path = os.path.join(user_directory, filename)

        # Save uploaded image
        contents = await file.read()
        with open(image_path, 'wb') as image_file:
            image_file.write(contents)

        # Process image with Pillow
        try:
            img = Image.open(image_path)
            resized_image = resize_image(img, (620, 480))
            resized_image.save(image_path, 'JPEG')
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Set file permissions
        os.chmod(image_path, 0o755)
    
        # Save user data to CSV
        user_data_dict = {'name': name, 'phone_no': phone_no, 'image_path': image_path}
        save_to_csv(user_data_dict)

        # Generate and save face encoding with multiple detection methods
        face_image = face_recognition.load_image_file(image_path)
        
        # Try different face detection models for better accuracy
        face_encodings = []
        
        # Method 1: Default HOG model (faster)
        face_encodings = face_recognition.face_encodings(face_image, model='small')
        
        # Method 2: If no faces found, try CNN model (more accurate but slower)
        if not face_encodings:
            logger.info("No faces found with HOG model, trying CNN model...")
            face_locations = face_recognition.face_locations(face_image, model='cnn')
            if face_locations:
                face_encodings = face_recognition.face_encodings(face_image, face_locations, model='small')
        
        # Method 3: If still no faces, try with different parameters
        if not face_encodings:
            logger.info("Trying face detection with adjusted parameters...")
            face_locations = face_recognition.face_locations(face_image, number_of_times_to_upsample=2)
            if face_locations:
                face_encodings = face_recognition.face_encodings(face_image, face_locations)
        
        # Method 4: Try preprocessing the image
        if not face_encodings:
            logger.info("Trying with image preprocessing...")
            # Convert to OpenCV format for preprocessing
            cv_image = cv2.imread(image_path)
            
            # Enhance contrast
            lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced_image = cv2.merge((cl,a,b))
            enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2BGR)
            
            # Save enhanced image temporarily
            enhanced_path = image_path.replace('.jpg', '_enhanced.jpg')
            cv2.imwrite(enhanced_path, enhanced_image)
            
            # Try face detection on enhanced image
            enhanced_face_image = face_recognition.load_image_file(enhanced_path)
            face_encodings = face_recognition.face_encodings(enhanced_face_image)
            
            # Clean up temporary file
            if os.path.exists(enhanced_path):
                os.remove(enhanced_path)
        
        if not face_encodings:
            # Provide more helpful error message
            raise HTTPException(
                status_code=400, 
                detail="No face detected in the image. Please ensure: 1) Face is clearly visible, 2) Good lighting, 3) Face is not too small or too large, 4) Image is not blurry"
            )
        
        face_encoding = face_encodings[0]
        face_encoding_path = os.path.join(KNOWN_FACES_PATH, f'{name}.npy')
        np.save(face_encoding_path, face_encoding)

        logger.info(f"Successfully trained user: {name}")
        response = {'Status': 'Success', 'Message': 'Successfully trained user'}
        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        response = {'Status': 'Error', 'Message': str(e)}
        return JSONResponse(content=response, status_code=500)

    finally:
        logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")

@router.post("/recognize")
async def recognize_user(file: UploadFile = File(...)):
    """Recognize user from uploaded image"""
    start_time = time.time()
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Save uploaded image
        contents = await file.read()
        unknown_image_path = os.path.join(UPLOADS_PATH, 'unknown.jpg')
        with open(unknown_image_path, 'wb') as f:
            f.write(contents)
        
        # Set file permissions
        os.chmod(unknown_image_path, 0o755)

        # Recognize faces
        recognized_names, matched_face_urls = recognize_faces_with_images(unknown_image_path)

        if len(recognized_names) == 0:
            response = {
                'Status': 'Error', 
                'Message': "No faces recognized. Please train the system with images or provide better quality images."
            }
        else:
            response = {
                'Status': 'Success', 
                'Names': recognized_names, 
                'image': matched_face_urls[0] if matched_face_urls else None
            }
        
        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recognition error: {str(e)}")
        response = {'Status': 'Error', 'Message': 'Face recognition failed. Please try again with a clearer image.'}
        return JSONResponse(content=response, status_code=500)

    finally:
        logger.info(f"Recognition completed in {time.time() - start_time:.2f} seconds")

@router.get("/users")
async def get_trained_users():
    """Get list of all trained users"""
    try:
        trained_users = []
        if os.path.exists(KNOWN_FACES_PATH):
            for file_name in os.listdir(KNOWN_FACES_PATH):
                if file_name.endswith('.npy'):
                    user_name = file_name.split('.')[0]
                    trained_users.append(user_name)
        
        return JSONResponse(content={
            'Status': 'Success',
            'Users': trained_users,
            'Count': len(trained_users)
        })
    
    except Exception as e:
        logger.error(f"Error getting users: {str(e)}")
        return JSONResponse(content={
            'Status': 'Error',
            'Message': 'Failed to retrieve user list'
        }, status_code=500)

@router.delete("/user/{name}")
async def delete_user(name: str):
    """Delete a trained user"""
    try:
        # Remove face encoding file
        face_encoding_path = os.path.join(KNOWN_FACES_PATH, f'{name}.npy')
        if os.path.exists(face_encoding_path):
            os.remove(face_encoding_path)
        
        # Remove user directory
        user_directory = os.path.join(UPLOADS_PATH, name)
        if os.path.exists(user_directory):
            shutil.rmtree(user_directory)
        
        logger.info(f"Successfully deleted user: {name}")
        return JSONResponse(content={
            'Status': 'Success',
            'Message': f'User {name} deleted successfully'
        })
    
    except Exception as e:
        logger.error(f"Error deleting user {name}: {str(e)}")
        return JSONResponse(content={
            'Status': 'Error',
            'Message': f'Failed to delete user {name}'
        }, status_code=500)