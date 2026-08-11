import cv2
import numpy as np
import torch
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class BaseVisionModel:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None

    def load(self):
        raise NotImplementedError

    def predict(self, frame: np.ndarray):
        raise NotImplementedError

class YOLOWorker(BaseVisionModel):
    def __init__(self, model_path: str, task_name: str, classes: list = None, device: str = 'cuda'):
        super().__init__(model_path, device)
        self.task_name = task_name
        self.classes = classes # Optional list of class indices to filter

    def load(self):
        if YOLO is None:
            print("ultralytics not installed. YOLOWorker disabled.")
            return False
        print(f"Loading YOLO model {self.task_name} from {self.model_path} on {self.device}...")
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            return True
        except Exception as e:
            print(f"Failed to load {self.task_name}: {e}")
            return False

    def predict(self, frame: np.ndarray):
        if self.model is None:
            return []
        
        # Run inference
        results = self.model(frame, classes=self.classes, verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                detections.append({
                    "task": self.task_name,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "center": [float((x1+x2)/2), float(y2)], # Bottom center (feet)
                    "conf": conf,
                    "class": cls
                })
        return detections

class DepthWorker(BaseVisionModel):
    def __init__(self, model_type: str = "MiDaS_small", device: str = 'cuda'):
        # using MiDaS as a placeholder for Depth. In prod, use Depth-Anything-V2
        super().__init__(model_type, device)
        self.transform = None

    def load(self):
        print(f"Loading Depth model {self.model_path} on {self.device}...")
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if self.model_path == "DPT_Large" or self.model_path == "DPT_Hybrid":
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform
            return True
        except Exception as e:
            print(f"Failed to load Depth Model: {e}")
            return False

    def predict(self, frame: np.ndarray):
        if self.model is None or self.transform is None:
            return None # Return None if not loaded
            
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        return depth_map
