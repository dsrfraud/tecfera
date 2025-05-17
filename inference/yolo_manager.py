# inference/yolo_manager.py
from ultralytics import YOLO

class YoloModelManager:
    def __init__(self):
        # Load once during app startup
        self.models = {
            "model1": YOLO("models/yolov5s.pt"),
            "model2": YOLO("models/yolov5m.pt"),
            "model3": YOLO("models/yolov5_custom.pt"),
            "model4": YOLO("models/yolov8s.pt"),  # optional fourth
        }

    def get_model(self, name: str):
        return self.models[name]
