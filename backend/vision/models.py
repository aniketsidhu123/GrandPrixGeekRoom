import cv2
import os
import base64
import json
import time
import numpy as np
import torch
import logging

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from .model_manager import model_manager

logger = logging.getLogger("VisionModels")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BASE CLASS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BaseVisionModel:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.model_path = model_path
        self.device = model_manager.device  # Use manager's device
        self.model = None
        self.loaded = False

    def load(self):
        raise NotImplementedError

    def predict(self, frame: np.ndarray):
        raise NotImplementedError


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# YOLO WORKER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class YOLOWorker(BaseVisionModel):
    """Runs a specific YOLOv8 model for object detection/segmentation."""

    def __init__(self, model_path: str, task_name: str, classes: list = None, device: str = 'cuda'):
        super().__init__(model_path, device)
        self.task_name = task_name
        self.classes = classes

        # Register with the model manager
        model_manager.register(task_name, "yolo", model_path)

    def load(self):
        if YOLO is None:
            logger.warning("ultralytics not installed. YOLOWorker disabled.")
            return False

        def _loader():
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            self.loaded = True
            return True

        return model_manager.load_model(self.task_name, _loader)

    def predict(self, frame: np.ndarray):
        if not self.loaded or self.model is None:
            return []

        start_t = time.time()

        results = self.model(frame, classes=self.classes, verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                cls_name = self.model.names.get(cls, str(cls)) if self.model.names else str(cls)
                detections.append({
                    "task": self.task_name,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "center": [float((x1 + x2) / 2), float(y2)],
                    "conf": round(conf, 3),
                    "class": cls,
                    "class_name": cls_name,
                })

        latency = (time.time() - start_t) * 1000
        model_manager.record_inference(self.task_name, latency)

        return detections


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEPTH WORKER (Local MiDaS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DepthWorker(BaseVisionModel):
    """Monocular depth estimation using local MiDaS weights."""

    def __init__(self, model_path: str = "weights/midas_v21_small.pt", device: str = 'cuda'):
        super().__init__(model_path, device)
        self.transform = None

        model_manager.register("depth", "depth", model_path)

    def load(self):
        def _loader():
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=False, trust_repo=True)
            if os.path.exists(self.model_path):
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
            else:
                logger.warning(f"Local weights not found at {self.model_path}, using torch.hub fallback.")
                self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)

            self.model.to(self.device)
            self.model.eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = midas_transforms.small_transform
            self.loaded = True
            return True

        return model_manager.load_model("depth", _loader)

    def predict(self, frame: np.ndarray):
        if not self.loaded or self.model is None or self.transform is None:
            return None

        start_t = time.time()

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

        latency = (time.time() - start_t) * 1000
        model_manager.record_inference("depth", latency)

        return prediction.cpu().numpy()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VLM WORKER (SmolVLM2 2.2B via llama-cpp-python)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VLM_SECURITY_PROMPT = (
    "You are a crowd safety AI analyst. Analyze this surveillance frame and respond ONLY with a JSON object. "
    "Do NOT include markdown formatting, code fences, or any text outside the JSON. "
    "Keys: "
    '"crowd_density" (one of: "low", "moderate", "high", "critical"), '
    '"estimated_people" (integer), '
    '"safety_hazards" (list of strings, empty if none), '
    '"unusual_behavior" (list of strings, empty if none), '
    '"stampede_risk" (one of: "none", "low", "moderate", "high"), '
    '"summary" (one sentence overall assessment).'
)


class VLMWorker:
    """
    Vision Language Model worker using SmolVLM2 2.2B Q4 via llama-cpp-python.
    Loaded and managed through the ModelManager for VRAM tracking.
    """

    def __init__(self, model_path: str = "weights/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
                 mmproj_path: str = "weights/mmproj-SmolVLM2-2.2B-Instruct-f16.gguf",
                 n_gpu_layers: int = -1):
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.n_gpu_layers = n_gpu_layers  # -1 = offload all layers to GPU
        self.model = None
        self.loaded = False
        self.last_analysis = None

        model_manager.register("vlm", "vlm", model_path)

    def load(self):
        if Llama is None:
            logger.warning("llama-cpp-python not installed. VLMWorker disabled.")
            return False
        if not os.path.exists(self.mmproj_path):
            logger.error(f"VLM mmproj file not found: {self.mmproj_path}")
            return False

        def _loader():
            self.model = Llama(
                model_path=self.model_path,
                chat_format="llama-3",
                n_ctx=2048,
                n_gpu_layers=self.n_gpu_layers,
                mmproj=self.mmproj_path,
                verbose=False,
            )
            self.loaded = True
            return True

        return model_manager.load_model("vlm", _loader)

    def predict(self, frame: np.ndarray):
        """Analyze a frame and return structured scene analysis dict."""
        if not self.loaded or self.model is None:
            return None

        start_t = time.time()

        try:
            # Encode frame to base64 JPEG for the VLM
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            data_uri = f"data:image/jpeg;base64,{img_b64}"

            response = self.model.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": VLM_SECURITY_PROMPT},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    }
                ],
                max_tokens=300,
                temperature=0.3,
            )

            raw_text = response["choices"][0]["message"]["content"].strip()

            # Try to parse as JSON
            try:
                analysis = json.loads(raw_text)
            except json.JSONDecodeError:
                start = raw_text.find('{')
                end = raw_text.rfind('}')
                if start != -1 and end != -1:
                    analysis = json.loads(raw_text[start:end + 1])
                else:
                    analysis = {
                        "crowd_density": "unknown",
                        "estimated_people": 0,
                        "safety_hazards": [],
                        "unusual_behavior": [],
                        "stampede_risk": "unknown",
                        "summary": raw_text[:200],
                    }

            self.last_analysis = analysis

            latency = (time.time() - start_t) * 1000
            model_manager.record_inference("vlm", latency)

            return analysis

        except Exception as e:
            logger.error(f"VLM inference error: {e}")
            return None
