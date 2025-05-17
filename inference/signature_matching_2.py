import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import backend as K
import tensorflow as tf
import torch
from PIL import Image
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.color import rgb2gray
from scipy.stats import mode
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)

class SignatureMatcher:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)
        self.siamese_model = self.create_siamese_model((224, 224, 3))
        self.siamese_model.compile(loss="binary_crossentropy", optimizer=Adam(0.0001))

    def dynamic_threshold(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    def process_and_save_image(self, input_image):
        image = input_image
        binary = self.dynamic_threshold(image)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        signature_contours = [contour for contour in contours if cv2.contourArea(contour) > 300]
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, signature_contours, -1, (255), thickness=cv2.FILLED)
        output = np.ones_like(image) * 255
        output[mask == 255] = image[mask == 255]
        return output

    def rotate_image(self, image, angle, resize=False):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rotated = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        if resize:
            height, width = rotated.shape[:2]
            angle_rad = angle * np.pi / 180
            new_width = abs(width * np.cos(angle_rad)) + abs(height * np.sin(angle_rad))
            new_height = abs(width * np.sin(angle_rad)) + abs(height * np.cos(angle_rad))
            rotated = cv2.resize(rotated, (int(new_width), int(new_height)))
        return rotated

    def skew_angle_hough_transform(self, image):
        edges = canny(image)
        tested_angles = np.deg2rad(np.arange(0.1, 180.0))
        h, theta, d = hough_line(edges, theta=tested_angles)
        accum, angles, dists = hough_line_peaks(h, theta, d)
        most_common_angle = mode(np.around(angles, decimals=2))[0]
        skew_angle = np.rad2deg(most_common_angle - np.pi / 2)
        return skew_angle

    def cropped_image(self, image, bbox):
        x_min, y_min, x_max, y_max = map(int, bbox)
        return image[y_min:y_max, x_min:x_max]

    def get_prediction(self, img):
        input_image = np.array(img)
        results = self.model(input_image)
        prediction = results.pandas().xyxy[0]
        predictions = prediction.sort_values(['class', "confidence"]).drop_duplicates("class", keep="last")
        return predictions[predictions['name'].isin(['sig'])]

    def get_signature(self, image_path):
        real_image = Image.open(image_path)
        opencvImage = cv2.cvtColor(np.array(real_image), cv2.COLOR_RGB2BGR)
        image = rgb2gray(np.array(real_image))
        skew = self.skew_angle_hough_transform(image)
        rotated_image = self.rotate_image(opencvImage, skew)
        predictions = self.get_prediction(rotated_image)
        if predictions.empty:
            return self.process_and_save_image(np.array(real_image))
        for index, row in predictions.iterrows():
            bbox = row[['xmin', 'ymin', 'xmax', 'ymax']].values
            cropped_signature = self.cropped_image(np.array(real_image), bbox)
            return self.process_and_save_image(cropped_signature)

    def euclidean_distance(self, vects):
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))

    def eucl_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    def create_base_network(self, input_shape):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        for layer in base_model.layers:
            layer.trainable = False
        flatten = layers.Flatten()(base_model.output)
        dense = layers.Dense(128, activation='relu')(flatten)
        return Model(inputs=base_model.input, outputs=dense)

    def create_siamese_model(self, input_shape):
        input_a = layers.Input(shape=input_shape)
        input_b = layers.Input(shape=input_shape)
        base_network = self.create_base_network(input_shape)
        encoded_a = base_network(input_a)
        encoded_b = base_network(input_b)
        distance = layers.Lambda(self.euclidean_distance, output_shape=self.eucl_dist_output_shape)([encoded_a, encoded_b])
        return Model(inputs=[input_a, input_b], outputs=distance)

    def extract_features(self, image_path):
        image = cv2.imread(image_path)
        img = cv2.resize(image, (224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        return img_array, img

    def compare_signatures(self, signature_path_1, signature_path_2):
        feature_1, resized_img1 = self.extract_features(signature_path_1)
        feature_2, resized_img2 = self.extract_features(signature_path_2)
        feature_1 = self.siamese_model.layers[2].predict(feature_1)
        feature_2 = self.siamese_model.layers[2].predict(feature_2)
        euclidean_dist = np.linalg.norm(feature_1 - feature_2)
        ssim_value = self.calculate_ssim(resized_img1, resized_img2)
        return euclidean_dist, ssim_value

    def calculate_ssim(self, img1, img2):
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        s, _ = ssim(img1_gray, img2_gray, full=True)
        return s

    def process_and_compare(self, signature_path_1, signature_path_2):
        sig1 = self.get_signature(signature_path_1)
        sig2 = self.get_signature(signature_path_2)
        temp_sig1_path = r"temp_sig1.jpg"
        temp_sig2_path = r"temp_sig2.jpg"
        cv2.imwrite(temp_sig1_path, sig1)
        cv2.imwrite(temp_sig2_path, sig2)
        euclidean_dist, ssim_value = self.compare_signatures(temp_sig1_path, temp_sig2_path)
        # os.remove(temp_sig1_path)
        # os.remove(temp_sig2_path)
        return euclidean_dist, ssim_value