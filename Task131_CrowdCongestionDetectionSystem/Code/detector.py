import os
import sys
import torch
from ultralytics import YOLO

# PyTorch 2.6+ compatibility patch to prevent weights_only security errors
try:
    _orig_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _orig_load(*args, **kwargs)
    torch.load = _patched_load
except Exception:
    pass

class PersonDetector:
    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.25):
        """
        Initializes the YOLOv8 detector.
        
        Args:
            model_path (str): Path to the YOLOv8 model weights.
            confidence_threshold (float): Minimum confidence for person detection.
        """
        self.confidence_threshold = confidence_threshold
        
        # Load YOLO model
        if not os.path.exists(model_path):
            # Fallback if model is not in the specified path, it will download
            print(f"⚠️ Model path {model_path} not found. Attempting to download/load default yolov8n.pt")
            self.model = YOLO("yolov8n.pt")
        else:
            self.model = YOLO(model_path)
            
    def detect(self, frame):
        """
        Runs object detection on the frame and filters for 'person' class (COCO class 0).
        
        Args:
            frame (numpy.ndarray): Input video frame (BGR).
            
        Returns:
            list of dict: List of detections. Each detection is:
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float,
                    "class_id": 0
                }
        """
        results = self.model.predict(frame, conf=self.confidence_threshold, verbose=False)
        result = results[0]
        
        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for idx in range(len(boxes)):
                class_id = int(classes[idx])
                # COCO Class 0 is Person
                if class_id == 0:
                    detections.append({
                        "bbox": [float(val) for val in boxes[idx]],
                        "confidence": float(confs[idx]),
                        "class_id": class_id
                    })
        return detections
