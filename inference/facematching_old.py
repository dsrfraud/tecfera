import face_recognition
import cv2
import numpy as np

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
    print(f'**************************{image_path}')
    image = face_recognition.load_image_file(image_path)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    brightness = calculate_brightness(image_bgr)
    blurriness = calculate_blurriness(image_bgr)
    return image, brightness, blurriness

# Function to compare faces in ID and selfie images
def compare_faces(id_image_path, selfie_image_path):
    # try:
    id_image, id_brightness, id_blurriness = load_and_process_image(id_image_path)
    selfie_image, selfie_brightness, selfie_blurriness = load_and_process_image(selfie_image_path)

    id_image_encodings = face_recognition.face_encodings(id_image, model='cnn')
    selfie_image_encodings = face_recognition.face_encodings(selfie_image, model= 'cnn')

    if len(id_image_encodings) == 0:
        raise ValueError("No face found in ID image.")
    if len(selfie_image_encodings) == 0:
        raise ValueError("No face found in selfie image.")

    id_image_encoding = id_image_encodings[0]
    selfie_image_encoding = selfie_image_encodings[0]

    face_distances = face_recognition.face_distance([id_image_encoding], selfie_image_encoding)
    similarity_score = 1 - face_distances[0] # similarity score based on face distance

    match = face_distances[0] <= 0.55 # you can adjust the threshold as needed

    match = bool(face_distances[0] <= 0.6) # you can adjust the threshold as needed

    result = {
        "Selfie Brightness": selfie_brightness,
        "Selfie Blurriness": selfie_blurriness,
        "ID Card Brightness": id_brightness,
        "ID Card Blurriness": id_blurriness,
        "Face Distance": float(face_distances[0]),
        "Match": match,
        "Error": ""
    }

    return result

    # except Exception as e:
    #     result = {
    #         "Selfie Image": selfie_image_path,
    #         "ID Card Image": id_image_path,
    #         "Selfie Brightness": "",
    #         "Selfie Blurriness": "",
    #         "ID Card Brightness": "",
    #         "ID Card Blurriness": "",
    #         "Face Distance": "",
    #         "Match": False,
    #         "Error": str(e)
    #     }
    #     return result