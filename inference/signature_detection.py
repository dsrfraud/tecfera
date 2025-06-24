import torch
import cv2
import numpy as np
from PIL import Image
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
from scipy.stats import mode
import base64


class SignatureDetection:
    def __init__(self, model_path, tesseract_cmd=None):
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        # Load the YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)

    def encode_image_to_base64(self, img):
        # Encode the OpenCV image to Base64
        _, buffer = cv2.imencode('.jpg', img)
        base64_encoded = base64.b64encode(buffer).decode('utf-8')
        return base64_encoded

    def decode_base64_to_image(self, base64_string):
        decoded_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(decoded_data, np.uint8)
        decoded_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return decoded_img

    def rotate_image(self, mat, angle):
        height, width = mat.shape[:2]
        image_center = (width / 2, height / 2)
        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]
        rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
        return rotated_mat

    def rotate90(self, image):
        timage = cv2.transpose(image)
        flipped = cv2.flip(timage, 1)
        return flipped

    def get_prediction(self, img):
        input_image = np.array(img)
        img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        results = self.model(input_image)
        prediction = results.pandas().xyxy[0]
        predictions = prediction.sort_values(['class', "confidence"])
        predictions = prediction[prediction['name'].isin(['sig'])]
        print(predictions)
        # results.show()
        return predictions

    def skew_angle_hough_transform(self, image):
        edges = canny(image)
        tested_angles = np.deg2rad(np.arange(0.1, 180.0))
        h, theta, d = hough_line(edges, theta=tested_angles)
        accum, angles, dists = hough_line_peaks(h, theta, d)
        most_common_angle = mode(np.around(angles, decimals=2))[0]
        skew_angle = np.rad2deg(most_common_angle - np.pi / 2)
        return skew_angle

    def find_skew_angle(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = canny(img)
        tested_angles = np.deg2rad(np.arange(0.1, 180.0))
        h, theta, d = hough_line(edges, theta=tested_angles)
        skew = self.skew_angle_hough_transform(img)
        return skew[0]

    def get_ocr(self, real_img, predictions):
        result_dict ={}
        result_list = []
        
        for i, pred in enumerate(predictions.iterrows()):
            label, confidence  = pred[1][6], pred[1][4]

            index, data = pred
            bbox = data[["xmin", "ymin", "xmax", "ymax"]].tolist()
            bbox = [int(x) for x in bbox]
            cropped_image = real_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            img = Image.fromarray(cropped_image)

            img = np.array(img)

            img = img[:, :, ::-1].copy()
            h,w,c = img.shape
            img = cv2.resize(img, (w,h))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            

            if label  == "sig":
                # result_dict[label] = {"value" : self.encode_image_to_base64(img), "conf_score" : confidence}
                try:
                    result_list.append(self.encode_image_to_base64(img))
                except Exception as e:
                    print(e)
                    pass
                # result_dict[i] = self.encode_image_to_base64(img)

        return result_list

    def process_single_image(self, image):
        opencvImage = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gray_image = cv2.cvtColor(opencvImage, cv2.COLOR_BGR2GRAY)
        skew = self.skew_angle_hough_transform(gray_image)
        rotated_image = self.rotate_image(opencvImage, 0)
        predictions = self.get_prediction(rotated_image)
        result = self.get_ocr(rotated_image, predictions)
        count = 1

        while count <= 4:
            if len(result):
                break
            else:
                if count == 4:
                    break
                else:
                    rotated_image = self.rotate90(rotated_image)
                    predictions = self.get_prediction(rotated_image)
                    result = self.get_ocr(rotated_image, predictions)
                    count += 1
        return result

