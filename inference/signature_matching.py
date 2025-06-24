import numpy as np
import cv2
import io
from PIL import Image
from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from sklearn.metrics.pairwise import cosine_similarity
import time
import matplotlib.pyplot as plt
import base64

import tensorflow
import sklearn




# Load VGG16 model with pre-trained weights
vgg16 = VGG16(weights='imagenet', include_top=False, pooling='max', input_shape=(224, 224, 3))

# Freeze model layers
for model_layer in vgg16.layers:
    model_layer.trainable = False

def dynamic_threshold(image):
    """
    Apply dynamic thresholding using Otsu's method.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary

def process_and_save_image(input_image, area_threshold):
    """
    Process the image to extract signature using the specified area threshold.
    """
    binary = dynamic_threshold(input_image)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    signature_contours = [contour for contour in contours if cv2.contourArea(contour) > area_threshold]

    mask = np.zeros_like(binary)
    cv2.drawContours(mask, signature_contours, -1, (255), thickness=cv2.FILLED)

    output = np.ones_like(input_image) * 255
    output[mask == 255] = input_image[mask == 255]

    return output

def generate_preprocessed_images(image):
    """
    Generate multiple preprocessed images with varying area thresholds.
    """
    preprocessed_images = []

    for thresh in range(100, 1500, 100):
        processed_image = process_and_save_image(image, thresh)
        if not np.all(processed_image == 255):  # Check if image is not completely white
            preprocessed_images.append(processed_image)
        else:
            break

    return preprocessed_images

# def load_image(image_bytes):
#     """
#     Load and resize an image.
#     """
#     input_image = Image.open(io.BytesIO(image_bytes))
#     resized_image = input_image.resize((224, 224))
#     return resized_image

def get_image_embeddings(object_image):
    """
    Get the embeddings of an image using VGG16.
    """
    image_array = np.expand_dims(image.img_to_array(object_image), axis=0)
    image_embedding = vgg16.predict(image_array)
    return image_embedding


from io import BytesIO

# Function to encode image to Base64 string
def encode_image_to_base64(image):
    try:
        image = Image.fromarray(image)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_str
    except Exception as e:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_str


def get_similarity_score(image1_bytes, image2_bytes):
    """
    Compute the best similarity score between two sets of preprocessed images.
    """
    image1 = cv2.imdecode(np.frombuffer(image1_bytes, np.uint8), cv2.IMREAD_COLOR)
    image2 = cv2.imdecode(np.frombuffer(image2_bytes, np.uint8), cv2.IMREAD_COLOR)

    

    first_preprocessed_images = generate_preprocessed_images(image1)
    second_preprocessed_images = generate_preprocessed_images(image2)

    best_similarity_score = -1
    best_first_image = None
    best_second_image = None

    for first_image in first_preprocessed_images:
        for second_image in second_preprocessed_images:
            first_image_pil = Image.fromarray(cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)).resize((224, 224))
            second_image_pil = Image.fromarray(cv2.cvtColor(second_image, cv2.COLOR_BGR2RGB)).resize((224, 224))

            first_image_vector = get_image_embeddings(first_image_pil)
            second_image_vector = get_image_embeddings(second_image_pil)

            similarity_score = cosine_similarity(first_image_vector, second_image_vector).reshape(1,)[0]

            if similarity_score > best_similarity_score:
                best_similarity_score = similarity_score
                best_first_image = first_image
                best_second_image = second_image


    return best_similarity_score, encode_image_to_base64(best_first_image), encode_image_to_base64(best_second_image)


